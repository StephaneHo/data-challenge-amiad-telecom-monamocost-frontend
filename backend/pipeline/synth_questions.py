"""
Génération synthétique de questions d'entraînement pour le retriever.

Pour chaque chunk du corpus, on demande à un LLM (gpt-4o-mini par défaut) de
produire N questions naturelles auxquelles le chunk répond.

Sortie : un fichier JSON au format `extra_training_examples.json`, directement
utilisable par `finetune.py --extra-examples ...`.

Cache idempotent : on relit le fichier de sortie avant chaque run et on saute
les chunks déjà couverts (par `_chunk_id`). Re-lancer après ingestion de
nouveaux PDFs ne re-paye que les nouveaux chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import settings
from database.models import Chunk


_SYSTEM_PROMPT = (
    "Tu es un analyste spécialisé en défense, renseignement et sécurité. "
    "À partir d'un extrait documentaire, tu génères des questions naturelles "
    "qu'un analyste pourrait poser et auxquelles l'extrait répond précisément."
)

_USER_TEMPLATE = """Extrait documentaire :
\"\"\"
{content}
\"\"\"

Génère exactement {n} questions naturelles auxquelles cet extrait répond.

Règles strictes :
- Questions réalistes (style analyste) en français.
- Chaque question doit pouvoir être répondue ENTIÈREMENT à partir de l'extrait fourni.
- Évite les questions oui/non — privilégie "comment", "quels", "pourquoi", "dans quelle mesure".
- Évite les questions trop génériques — ancre-les dans le contenu spécifique (chiffres, acteurs, dispositifs cités).
- Ne mentionne pas l'extrait lui-même dans la question ("d'après l'extrait...").

Retourne UNIQUEMENT un JSON :
{{"questions": ["<question 1>", "<question 2>", ...]}}
"""


@dataclass
class SyntheticExample:
    """Une paire (question synthétique, chunk) avec traçabilité."""

    question: str
    paragraph: str
    doc_name: str
    page: int
    chunk_id: int
    generator_model: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "paragraph": self.paragraph,
            "doc_name": self.doc_name,
            "page": self.page,
            "_chunk_id": self.chunk_id,
            "_generator_model": self.generator_model,
            "_generated_at": self.generated_at,
        }


class SyntheticQuestionGenerator:
    """
    Pipeline de génération + cache de paires synthétiques (question, chunk).

    Stratégie cache : on charge le JSON existant en mémoire (indexé par chunk_id),
    on ne génère que pour les chunks absents, on flush périodiquement pour
    survivre à un Ctrl+C.
    """

    def __init__(
        self,
        session: Session,
        output_path: Path,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        questions_per_chunk: int = 2,
        max_chunk_chars: int = 4000,
        min_chunk_chars: int = 80,
    ) -> None:
        self.session = session
        self.output_path = Path(output_path)
        self.model = model
        self.temperature = temperature
        self.questions_per_chunk = questions_per_chunk
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self._existing_by_chunk: dict[int, list[dict[str, Any]]] = {}
        self._existing_flat: list[dict[str, Any]] = []
        self._load_existing()
        self._client: Any = None

    # ───────────────────────── Cache ───────────────────────────────────────

    def _load_existing(self) -> None:
        if not self.output_path.exists():
            logger.info(f"[Synth] Pas de cache existant ({self.output_path})")
            return
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as ex:
            logger.warning(f"[Synth] Cache invalide ({ex}), démarre vide")
            return
        if not isinstance(data, list):
            logger.warning(f"[Synth] Cache au mauvais format (pas une liste), démarre vide")
            return
        for it in data:
            self._existing_flat.append(it)
            cid = it.get("_chunk_id")
            if cid is not None:
                self._existing_by_chunk.setdefault(cid, []).append(it)
        logger.info(
            f"[Synth] Cache chargé : {len(self._existing_flat)} entries "
            f"couvrant {len(self._existing_by_chunk)} chunks"
        )

    def _flush(self, new_entries: list[dict[str, Any]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        merged = self._existing_flat + new_entries
        self.output_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ───────────────────────── LLM ─────────────────────────────────────────

    def _llm(self) -> Any:
        if self._client is None:
            from utils.llm import get_openai_client

            self._client = get_openai_client()
        return self._client

    def _generate_for_chunk(self, chunk: Chunk) -> list[str]:
        from utils.llm import chat_json

        content = chunk.content[: self.max_chunk_chars]
        prompt = _USER_TEMPLATE.format(content=content, n=self.questions_per_chunk)
        try:
            payload = chat_json(
                self._llm(),
                model=self.model,
                system=_SYSTEM_PROMPT,
                user=prompt,
                temperature=self.temperature,
            )
            questions = payload.get("questions", [])
            # Tolère un singleton mal formé
            if isinstance(questions, str):
                questions = [questions]
            cleaned: list[str] = []
            for q in questions:
                q = q.strip() if isinstance(q, str) else ""
                if q and len(q) > 10:
                    cleaned.append(q)
            return cleaned
        except Exception as ex:
            logger.warning(f"[Synth] chunk_id={chunk.id} : erreur LLM ({ex})")
            return []

    # ─────────────────────── Sélection chunks ──────────────────────────────

    def _select_chunks(
        self,
        limit: Optional[int] = None,
        doc_filter: Optional[list[str]] = None,
        force: bool = False,
    ) -> list[Chunk]:
        stmt = select(Chunk).where(
            func.length(Chunk.content) >= self.min_chunk_chars
        )
        if doc_filter:
            stmt = stmt.where(Chunk.doc_name.in_(doc_filter))
        stmt = stmt.order_by(Chunk.doc_name, Chunk.page_number, Chunk.chunk_index)
        if limit:
            stmt = stmt.limit(limit)

        rows = list(self.session.execute(stmt).scalars())
        if not force:
            rows = [c for c in rows if c.id not in self._existing_by_chunk]
        return rows

    # ─────────────────────────── Run ───────────────────────────────────────

    def estimate_cost(self, n_chunks: int) -> float:
        """
        Estimation grossière du coût en USD pour gpt-4o-mini.
        - ~500 tokens d'input par chunk (system + user template + contenu tronqué)
        - ~60 tokens d'output par chunk (~2 questions × 30 tokens)
        - gpt-4o-mini : $0.15 / 1M input, $0.60 / 1M output
        """
        tokens_in = n_chunks * 500
        tokens_out = n_chunks * 60
        return tokens_in * 0.15 / 1_000_000 + tokens_out * 0.60 / 1_000_000

    def run(
        self,
        limit: Optional[int] = None,
        doc_filter: Optional[list[str]] = None,
        flush_every: int = 20,
        dry_run: bool = False,
        force: bool = False,
    ) -> int:
        chunks = self._select_chunks(limit=limit, doc_filter=doc_filter, force=force)
        n = len(chunks)
        cost = self.estimate_cost(n)
        logger.info(
            f"[Synth] {n} chunks à traiter (estimation coût : ~${cost:.4f}) "
            f"avec {self.model} temp={self.temperature} questions/chunk={self.questions_per_chunk}"
        )

        if dry_run or n == 0:
            if dry_run and n > 0:
                logger.info("[Synth] Dry-run — aucun appel LLM effectué.")
                for c in chunks[:3]:
                    logger.info(
                        f"  exemple chunk id={c.id} {c.doc_name} p.{c.page_number} "
                        f"({len(c.content)} chars) : {c.content[:80]}..."
                    )
                if n > 3:
                    logger.info(f"  ... et {n - 3} autres chunks")
            return 0

        new_entries: list[dict[str, Any]] = []
        n_questions = 0

        for i, chunk in enumerate(chunks, start=1):
            questions = self._generate_for_chunk(chunk)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for q in questions:
                ex = SyntheticExample(
                    question=q,
                    paragraph=chunk.content,
                    doc_name=chunk.doc_name,
                    page=chunk.page_number,
                    chunk_id=chunk.id,
                    generator_model=self.model,
                    generated_at=now,
                )
                new_entries.append(ex.to_dict())
                n_questions += 1

            if i % flush_every == 0:
                logger.info(
                    f"[Synth] {i}/{n} chunks | {n_questions} questions générées | flush"
                )
                self._flush(new_entries)

        self._flush(new_entries)
        logger.info(
            f"[Synth] Terminé : {n_questions} nouvelles questions sur {n} chunks "
            f"→ {self.output_path}"
        )
        return n_questions
