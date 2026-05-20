"""
Génération de paraphrases LLM pour les questions gold (ground truth).

Objectif : éviter la mémorisation lexicale lors de l'upweight des paires gold.
En reformulant chaque question gold de N façons via gpt-4o-mini, le modèle voit
plusieurs surfaces lexicales pour le même concept → généralise mieux.

Sortie : JSON au format `extra_training_examples.json` directement utilisable
par `finetune.py --gold-extra <fichier>`. Chaque paraphrase est associée à TOUS
les paragraphes (doc_name, page) de la question gold originale.

Cache idempotent : on dédoublonne par hash de la question source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config import settings


_SYSTEM_PROMPT = (
    "Tu es un expert en reformulation de questions. Tu produis des paraphrases "
    "naturelles et variées en préservant strictement le sens initial."
)

_USER_TEMPLATE = """Question originale :
\"\"\"
{question}
\"\"\"

Génère exactement {n} paraphrases naturelles de cette question en français.

Règles strictes :
- Préserve EXACTEMENT le sens, les entités nommées et les chiffres.
- Varie le wording : synonymes, ordre des propositions, formulation interrogative
  (Quels..., Comment..., Dans quelle mesure..., Pour quelle raison...).
- Reste réaliste — c'est ainsi qu'un analyste défense/renseignement la poserait.
- Ne paraphrase PAS les noms propres / acronymes (MQ-9 Reaper, OSINT, USV, HUMINT...).

Retourne UNIQUEMENT un JSON :
{{"paraphrases": ["<paraphrase 1>", "<paraphrase 2>", ...]}}
"""


def _question_hash(q: str) -> str:
    return hashlib.sha1(q.strip().lower().encode("utf-8")).hexdigest()[:16]


@dataclass
class GoldPair:
    """Une paire (question, paragraphe) gold telle que lue depuis la source."""

    question: str
    paragraph: str
    doc_name: str
    page: int


def _group_by_question(pairs: list[GoldPair]) -> dict[str, list[GoldPair]]:
    """Regroupe les pairs par texte de question (pour générer 1 set de paraphrases par question unique)."""
    by_q: dict[str, list[GoldPair]] = {}
    for p in pairs:
        by_q.setdefault(p.question, []).append(p)
    return by_q


class GoldParaphraser:
    """
    Génère N paraphrases LLM par question gold, et émet pour chaque paraphrase
    autant de paires (paraphrase, paragraphe) qu'il y a de paragraphes associés
    à la question originale.

    Cache idempotent : un JSON existant est relu, les questions déjà traitées
    sont skippées (clé = hash sha1 tronqué).
    """

    def __init__(
        self,
        output_path: Path,
        model: str = "gpt-4o-mini",
        temperature: float = 0.9,
        paraphrases_per_question: int = 10,
    ) -> None:
        self.output_path = Path(output_path)
        self.model = model
        self.temperature = temperature
        self.n = paraphrases_per_question
        self._existing_flat: list[dict[str, Any]] = []
        self._existing_qhashes: set[str] = set()
        self._load_existing()
        self._client: Any = None

    # ───────────────────────── Cache ───────────────────────────────────────

    def _load_existing(self) -> None:
        if not self.output_path.exists():
            logger.info(f"[Paraphrase] Pas de cache existant ({self.output_path})")
            return
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as ex:
            logger.warning(f"[Paraphrase] Cache invalide ({ex}), démarre vide")
            return
        if not isinstance(data, list):
            logger.warning("[Paraphrase] Cache au mauvais format, démarre vide")
            return
        for it in data:
            self._existing_flat.append(it)
            qh = it.get("_source_question_hash")
            if qh:
                self._existing_qhashes.add(qh)
        logger.info(
            f"[Paraphrase] Cache chargé : {len(self._existing_flat)} paraphrases "
            f"sur {len(self._existing_qhashes)} questions originales"
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
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def _generate_paraphrases(self, question: str) -> list[str]:
        prompt = _USER_TEMPLATE.format(question=question, n=self.n)
        try:
            resp = self._llm().chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            payload = json.loads(resp.choices[0].message.content or "{}")
            paraphrases = payload.get("paraphrases", [])
            if isinstance(paraphrases, str):
                paraphrases = [paraphrases]
            cleaned: list[str] = []
            for p in paraphrases:
                p = p.strip() if isinstance(p, str) else ""
                # Anti-collision : on jette les paraphrases identiques (ou quasi) à l'originale
                if p and len(p) > 10 and p.lower() != question.strip().lower():
                    cleaned.append(p)
            return cleaned
        except Exception as ex:
            logger.warning(f"[Paraphrase] Erreur LLM sur '{question[:50]}...' : {ex}")
            return []

    # ─────────────────────────── Run ───────────────────────────────────────

    def estimate_cost(self, n_questions: int) -> float:
        """gpt-4o-mini : ~400 tokens in, ~300 out par appel ; prix ~$0.00024 par appel."""
        tokens_in = n_questions * 400
        tokens_out = n_questions * 300
        return tokens_in * 0.15 / 1_000_000 + tokens_out * 0.60 / 1_000_000

    def run(
        self,
        pairs: list[GoldPair],
        dry_run: bool = False,
        force: bool = False,
    ) -> int:
        by_question = _group_by_question(pairs)
        if not force:
            todo_qs = [q for q in by_question if _question_hash(q) not in self._existing_qhashes]
        else:
            todo_qs = list(by_question.keys())

        logger.info(
            f"[Paraphrase] {len(todo_qs)} questions à traiter "
            f"({len(by_question) - len(todo_qs)} déjà en cache)"
        )
        if not todo_qs:
            return 0

        cost = self.estimate_cost(len(todo_qs))
        logger.info(
            f"[Paraphrase] Estimation coût : ~${cost:.4f} "
            f"avec {self.model} temp={self.temperature} paraphrases/q={self.n}"
        )
        if dry_run:
            logger.info("[Paraphrase] Dry-run, pas d'appel LLM.")
            for q in todo_qs[:3]:
                logger.info(f"  - {q[:100]}... ({len(by_question[q])} paragraphes liés)")
            return 0

        new_entries: list[dict[str, Any]] = []
        n_total_paraphrases = 0

        for i, q in enumerate(todo_qs, start=1):
            paraphrases = self._generate_paraphrases(q)
            if not paraphrases:
                logger.warning(f"[Paraphrase] Aucune paraphrase pour '{q[:60]}...'")
                continue
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            qh = _question_hash(q)
            # Pour chaque paraphrase, on émet 1 entry par paragraphe associé
            for para_q in paraphrases:
                for gp in by_question[q]:
                    new_entries.append(
                        {
                            "question": para_q,
                            "paragraph": gp.paragraph,
                            "doc_name": gp.doc_name,
                            "page": gp.page,
                            "_source_question": q,
                            "_source_question_hash": qh,
                            "_generator_model": self.model,
                            "_generated_at": now,
                        }
                    )
                    n_total_paraphrases += 1

            if i % 5 == 0:
                logger.info(
                    f"[Paraphrase] {i}/{len(todo_qs)} questions | "
                    f"{n_total_paraphrases} entries générées | flush"
                )
                self._flush(new_entries)

        self._flush(new_entries)
        logger.info(
            f"[Paraphrase] Terminé : {n_total_paraphrases} nouvelles entries "
            f"sur {len(todo_qs)} questions originales → {self.output_path}"
        )
        return n_total_paraphrases
