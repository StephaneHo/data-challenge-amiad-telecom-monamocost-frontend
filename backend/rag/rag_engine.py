from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from pipeline.embedder import Embedder
from pipeline.query_decomposition import QueryDecomposer
from rag.attribution import (
    AttributedSentence,
    Attributor,
    RetrievedChunkForAttribution,
)
from utils.carbon import CarbonTracker


# Granularité interne (chunk) — utilisée pour bâtir le contexte LLM.
@dataclass
class RetrievedChunk:
    doc_name: str
    page_number: int
    chunk_index: int
    content: str
    score: float


# Granularité du challenge ((doc_name, page) ranké) — utilisée pour le JSON de sortie.
@dataclass
class RetrievedPage:
    rank: int
    doc_name: str
    page: int
    score: float
    n_chunks: int  # combien de chunks de cette page ont matché dans le top-K


@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved: list[RetrievedPage] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    carbon: dict[str, Any] = field(default_factory=dict)


class RAGEngine:
    """
    Moteur RAG pour le challenge EvalLLM 2026.

    Pipeline :
      1. embed la requête (préfixe `query: ` via `Embedder`)
      2. cherche les top-K chunks par cosinus dans pgvector
      3. agrège à la granularité (doc_name, page) — le challenge évalue à cette grain-là
      4. construit un contexte pour le LLM (chunks gardés au max `context_chunks`)
      5. demande au LLM une réponse avec citations `[doc_name p.N]`
    """

    _SYSTEM_PROMPT = textwrap.dedent(
        """\
        Tu es un assistant spécialisé en synthèse documentaire pour la défense et le renseignement.

        Règles strictes :
        - Réponds UNIQUEMENT à partir des extraits fournis. N'invente aucun fait.
        - Cite tes sources à chaque affirmation factuelle au format `[nom_du_document.pdf p.N]`.
        - Si plusieurs sources convergent sur une même affirmation, cite-les toutes.
        - Si tu introduis une information de **connaissance générale** ou de mise en
          contexte qui ne provient pas explicitement des extraits, ne mets PAS de
          citation derrière. Cette absence de citation signale que la phrase est
          non sourcée — c'est attendu pour les transitions, titres et mises en perspective.
        - Si l'information demandée n'est pas du tout dans les extraits, dis-le
          explicitement plutôt que d'inventer.
        - Rédige en français, dans un style factuel et synthétique. Utilise des titres
          markdown (#, ##) pour structurer si la réponse est longue.

        Exemple de réponse bien annotée :

        Contexte fourni : un seul extrait du document `guide_osint_infrastructure_v2.pdf` p.1
        sur l'analyse des certificats SSL/TLS et l'usage des CT Logs.

        Question : Comment l'utilisation des logs de certificats peut-elle aider à
        découvrir l'infrastructure d'une cible sans se faire repérer ?

        Réponse attendue :

        # Méthodologie de reconnaissance passive via SSL

        L'exploitation des journaux de transparence (CT Logs) permet de découvrir
        des sous-domaines et environnements de pré-production en examinant les
        champs CN et SAN des certificats. [guide_osint_infrastructure_v2.pdf p.1]

        Cette approche garantit la discrétion de l'analyste car elle constitue une
        technique passive, évitant ainsi le déclenchement d'alertes au niveau des
        pare-feu applicatifs (WAF) qui ciblent habituellement les scans actifs.
        [guide_osint_infrastructure_v2.pdf p.1]

        En complément, cette phase est souvent la première étape d'une chaîne
        d'attaque plus complexe appelée "Recon-ng".

        Enfin, la validation de ces informations doit impérativement passer par
        une corrélation avec l'historique des enregistrements DNS.
        [guide_osint_infrastructure_v2.pdf p.1]

        Note : la phrase sur Recon-ng n'a pas de citation car c'est de la connaissance
        générale, pas dans l'extrait fourni. Le titre `#` n'a pas de citation non plus
        car c'est de la mise en forme.
        """
    )

    def __init__(
        self,
        session: Session,
        embedder: Optional[Embedder] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.session = session
        self.embedder = embedder or Embedder()
        if self.embedder.dim != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"[RAG] Dim mismatch : embedder={self.embedder.dim}, settings={settings.EMBEDDING_DIM}"
            )
        self.llm_provider = (llm_provider or settings.LLM_PROVIDER).lower()
        self.llm_model = llm_model or settings.LLM_MODEL
        self._llm_client: Any = None  # initialisé paresseusement
        self._decomposer: Optional[QueryDecomposer] = None

    def _build_llm(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        if self.llm_provider == "openai":
            # Supporte les serveurs LLM locaux compatibles OpenAI (vLLM, Ollama, ...)
            # via LLM_BASE_URL dans config.py.
            from utils.llm import get_openai_client

            self._llm_client = get_openai_client()
        elif self.llm_provider == "anthropic":
            import anthropic

            self._llm_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Provider LLM inconnu : {self.llm_provider}")
        return self._llm_client

    # ──────────────────────── Retrieval ────────────────────────

    def retrieve_chunks_decomposed(
        self,
        query: str,
        k: int = 30,
        per_subquery_k: int = 20,
        doc_filter: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieval avec décomposition de la question multi-hop.
          1. Décompose la question en N sous-questions (1 si déjà atomique)
          2. Pour chaque sous-question, retrieve `per_subquery_k` chunks
          3. Fusionne en gardant le meilleur score par (doc, page, chunk_index)
          4. Trie par score, garde le top `k`
        """
        if self._decomposer is None:
            self._decomposer = QueryDecomposer()
        decomp = self._decomposer.decompose(query)
        if decomp.is_atomic:
            return self.retrieve_chunks(query, k=k, doc_filter=doc_filter)

        # Fusion : on indexe par (doc, page, chunk_index) et on garde le meilleur score
        best: dict[tuple[str, int, int], RetrievedChunk] = {}
        for subq in decomp.subqueries:
            for c in self.retrieve_chunks(subq, k=per_subquery_k, doc_filter=doc_filter):
                key = (c.doc_name, c.page_number, c.chunk_index)
                cur = best.get(key)
                if cur is None or c.score > cur.score:
                    best[key] = c
        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ranked[:k]

    def retrieve_chunks(
        self,
        query: str,
        k: int = 30,
        doc_filter: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Top-K chunks par similarité cosinus dans pgvector."""
        qv = self.embedder.encode_query(query)

        sql = (
            "SELECT doc_name, page_number, chunk_index, content, "
            "1 - (embedding <=> CAST(:qv AS vector)) AS score "
            "FROM chunks "
            "WHERE embedding IS NOT NULL"
        )
        params: dict[str, Any] = {"qv": qv, "k": k}
        if doc_filter:
            sql += " AND doc_name = ANY(:docs)"
            params["docs"] = doc_filter
        sql += " ORDER BY embedding <=> CAST(:qv AS vector) LIMIT :k"

        rows = self.session.execute(text(sql), params).fetchall()
        return [
            RetrievedChunk(
                doc_name=r.doc_name,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                content=r.content,
                score=float(r.score),
            )
            for r in rows
        ]

    @staticmethod
    def aggregate_to_pages(
        chunks: list[RetrievedChunk], top_n: Optional[int] = None
    ) -> list[RetrievedPage]:
        """
        Dédoublonne par (doc_name, page) en gardant le meilleur score.
        Compte aussi le nombre de chunks par page (utile pour le scoring).
        """
        best: dict[tuple[str, int], dict[str, Any]] = {}
        for c in chunks:
            key = (c.doc_name, c.page_number)
            entry = best.get(key)
            if entry is None or c.score > entry["score"]:
                best[key] = {
                    "doc_name": c.doc_name,
                    "page": c.page_number,
                    "score": c.score,
                    "n_chunks": (entry["n_chunks"] + 1) if entry else 1,
                }
            else:
                entry["n_chunks"] += 1

        ranked = sorted(best.values(), key=lambda e: e["score"], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return [
            RetrievedPage(
                rank=i + 1,
                doc_name=e["doc_name"],
                page=e["page"],
                score=round(e["score"], 6),
                n_chunks=e["n_chunks"],
            )
            for i, e in enumerate(ranked)
        ]

    # ──────────────────────── Génération ────────────────────────

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk], max_chunks: int = 12) -> str:
        """Formate les chunks pour le LLM, avec marqueur [doc_name p.N]."""
        lines: list[str] = []
        for c in chunks[:max_chunks]:
            tag = f"[{c.doc_name} p.{c.page_number}]"
            lines.append(f"{tag}\n{c.content}")
        return "\n\n---\n\n".join(lines)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return textwrap.dedent(
            f"""\
            Question :
            {query}

            Extraits des documents sources :
            ---
            {context}
            ---

            Rédige une réponse structurée et factuelle. Cite tes sources au
            format `[doc_name p.N]` à chaque affirmation factuelle.
            """
        )

    def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> tuple[str, int]:
        """Retourne (réponse_texte, tokens_utilisés)."""
        temp = temperature if temperature is not None else settings.RAG_TEMPERATURE
        client = self._build_llm()

        if self.llm_provider == "openai":
            resp = client.chat.completions.create(
                model=self.llm_model,
                temperature=temp,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or "", int(resp.usage.total_tokens)

        if self.llm_provider == "anthropic":
            resp = client.messages.create(
                model=self.llm_model,
                max_tokens=2048,
                temperature=temp,
                system=self._SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text_out = "".join(b.text for b in resp.content if getattr(b, "text", None))
            tokens = int(resp.usage.input_tokens + resp.usage.output_tokens)
            return text_out, tokens

        raise ValueError(f"Provider LLM inconnu : {self.llm_provider}")

    # ──────────────────────── API publique ────────────────────────

    def answer(
        self,
        query: str,
        top_k_chunks: Optional[int] = None,
        top_n_pages: Optional[int] = None,
        context_chunks: int = 12,
        doc_filter: Optional[list[str]] = None,
        retrieval_only: bool = False,
        decompose: bool = False,
    ) -> RAGResponse:
        """
        Pipeline complet : retrieval → agrégation pages → génération.

        - top_k_chunks : combien de chunks scorer dans pgvector (défaut: RAG_TOP_K * 3)
        - top_n_pages  : combien de paires (doc, page) renvoyer dans `retrieved`
                         (défaut : tous les uniques du top-K)
        - context_chunks : combien de chunks passer au LLM
        - retrieval_only : court-circuite l'appel LLM (utile pour évaluer le retrieval seul)
        """
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tracker = CarbonTracker(label="rag_inference", device=device)

        with tracker.measure():
            k = top_k_chunks or (settings.RAG_TOP_K * 3)
            if decompose:
                chunks = self.retrieve_chunks_decomposed(query, k=k, doc_filter=doc_filter)
            else:
                chunks = self.retrieve_chunks(query, k=k, doc_filter=doc_filter)

            if not chunks:
                tracker.log_summary()
                return RAGResponse(
                    question=query,
                    answer="Aucun extrait pertinent trouvé dans la base documentaire.",
                    parameters=self._params_dict(k=k, top_n=top_n_pages),
                    carbon=tracker.metrics.to_dict(),
                )

            retrieved_pages = self.aggregate_to_pages(chunks, top_n=top_n_pages)

            if retrieval_only:
                tracker.log_summary()
                return RAGResponse(
                    question=query,
                    answer="",
                    retrieved=retrieved_pages,
                    chunks=chunks,
                    parameters=self._params_dict(k=k, top_n=top_n_pages),
                    carbon=tracker.metrics.to_dict(),
                )

            context = self._build_context(chunks, max_chunks=context_chunks)
            prompt = self._build_prompt(query, context)
            input_tokens_estimate = len(self._SYSTEM_PROMPT) // 4 + len(prompt) // 4
            try:
                answer_txt, tokens = self._call_llm(prompt)
                # On approxime input/output 50/50 si le provider ne sépare pas
                output_tokens_estimate = max(0, tokens - input_tokens_estimate)
                if output_tokens_estimate <= 0:
                    input_tokens_estimate = tokens // 2
                    output_tokens_estimate = tokens - input_tokens_estimate
                tracker.add_llm_tokens(
                    input_tokens=input_tokens_estimate,
                    output_tokens=output_tokens_estimate,
                )
            except Exception as ex:
                logger.exception(f"[RAG] Appel LLM échoué : {ex}")
                answer_txt, tokens = "", 0

        tracker.log_summary()
        return RAGResponse(
            question=query,
            answer=answer_txt,
            retrieved=retrieved_pages,
            chunks=chunks,
            parameters=self._params_dict(k=k, top_n=top_n_pages),
            tokens_used=tokens,
            carbon=tracker.metrics.to_dict(),
        )

    def attribute(
        self,
        qid: str,
        answer: str,
        chunks: list[RetrievedChunk],
        threshold: float = 0.80,
        secondary_threshold: float = 0.85,
        topk_per_sentence: int = 3,
        entity_filter: bool = True,
        entity_min_support_ratio: float = 0.5,
    ) -> list[AttributedSentence]:
        """
        Tâche 2 du challenge : attribue chaque phrase de `answer` à un (doc, page).

        Pipeline :
          1. Markdown (titres, séparateurs) → `[]`
          2. Filtre entité (acronymes/nombres/noms propres absents des chunks) → `[]`
          3. Citations `[doc.pdf p.N]` → attribution directe, page résolue par sim si absente
          4. Embedding similarity Top-K :
             - top-1 retenu si sim ≥ threshold
             - top-2, top-3 retenus si sim ≥ secondary_threshold (plus strict)
             - sinon `[]`
        """
        attributor = Attributor(
            embedder=self.embedder,
            threshold=threshold,
            secondary_threshold=secondary_threshold,
            topk_per_sentence=topk_per_sentence,
            entity_filter=entity_filter,
            entity_min_support_ratio=entity_min_support_ratio,
        )
        attribution_chunks = [
            RetrievedChunkForAttribution(
                doc_name=c.doc_name,
                page_number=c.page_number,
                content=c.content,
            )
            for c in chunks
        ]
        return attributor.attribute(qid=qid, answer=answer, chunks=attribution_chunks)

    def _params_dict(self, k: int, top_n: Optional[int]) -> dict[str, Any]:
        return {
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "retriever_top_k_chunks": k,
            "retriever_top_n_pages": top_n,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "temperature": settings.RAG_TEMPERATURE,
        }
