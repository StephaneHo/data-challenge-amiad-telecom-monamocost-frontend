"""
Runner et schémas Pydantic pour le challenge RAG EvalLLM 2026.

Produit deux JSON conformes au format attendu (cf. `Experimental/target.txt`) :
- Tâche 1 : retrieval (doc_name, page) + answer
- Tâche 2 : attribution phrase → source (`attributed_to: []` pour les
  passages non sourcés)
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import settings
from rag.attribution import AttributedSentence
from rag.rag_engine import RAGEngine, RetrievedChunk


# ──────────────────────── Schémas — entrée ────────────────────────────────

class InputQuery(BaseModel):
    qid: str
    question: str
    # Les autres champs (retrieved/answer/metadata) sont ignorés en entrée.

    class Config:
        extra = "ignore"


class ChallengeInput(BaseModel):
    run_id: str = "monamo-cost-evallm-2026"
    parameters: dict[str, Any] = Field(default_factory=dict)
    results: list[InputQuery]


# ──────────────────────── Schémas — sortie Tâche 1 ────────────────────────

class RetrievedItem(BaseModel):
    rank: int
    doc_name: str
    page: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task1Result(BaseModel):
    qid: str
    question: str
    retrieved: list[RetrievedItem]
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task1Run(BaseModel):
    run_id: str
    parameters: dict[str, Any]
    results: list[Task1Result]


# ──────────────────────── Schémas — sortie Tâche 2 ────────────────────────

class AttributedTo(BaseModel):
    doc_name: str
    page: int


class AttributionItem(BaseModel):
    sid: str
    text: str
    attributed_to: list[AttributedTo]


class Task2Result(BaseModel):
    qid: str
    attributions: list[AttributionItem]


class Task2Run(BaseModel):
    run_id: str
    parameters: dict[str, Any]
    results: list[Task2Result]


# ──────────────────────── Schéma combiné ──────────────────────────────────

class ChallengeOutput(BaseModel):
    """Réponse de l'API : les deux tâches dans un seul payload."""

    task1: Task1Run
    task2: Optional[Task2Run] = None


# ──────────────────────── Runner ──────────────────────────────────────────

class ChallengeRunner:
    """
    Orchestrateur des deux tâches pour un input JSON challenge.
    Garde les `chunks` retrieved en mémoire entre Tâche 1 et Tâche 2 pour
    éviter de recompter les embeddings.
    """

    def __init__(
        self,
        session: Session,
        engine: Optional[RAGEngine] = None,
        top_k_chunks: int = 20,
        top_n_pages: int = 10,
        context_chunks: int = 10,
        attribution_threshold: float = 0.80,
        attribution_secondary_threshold: float = 0.85,
        attribution_topk: int = 3,
        attribution_entity_filter: bool = True,
        attribution_entity_min_support_ratio: float = 0.5,
        retrieval_only: bool = False,
    ) -> None:
        self.session = session
        self.engine = engine or RAGEngine(session=session)
        self.top_k_chunks = top_k_chunks
        self.top_n_pages = top_n_pages
        self.context_chunks = context_chunks
        self.attribution_threshold = attribution_threshold
        self.attribution_secondary_threshold = attribution_secondary_threshold
        self.attribution_topk = attribution_topk
        self.attribution_entity_filter = attribution_entity_filter
        self.attribution_entity_min_support_ratio = attribution_entity_min_support_ratio
        self.retrieval_only = retrieval_only

    def _parameters(self) -> dict[str, Any]:
        return {
            "embedding_model": self.engine.embedder.model_name,
            "embedding_dim": self.engine.embedder.dim,
            "retriever_top_k_chunks": self.top_k_chunks,
            "retriever_top_n_pages": self.top_n_pages,
            "llm_provider": self.engine.llm_provider,
            "llm_model": self.engine.llm_model,
            "temperature": settings.RAG_TEMPERATURE,
            "attribution_threshold": self.attribution_threshold,
            "attribution_secondary_threshold": self.attribution_secondary_threshold,
            "attribution_topk_per_sentence": self.attribution_topk,
            "attribution_entity_filter": self.attribution_entity_filter,
            "attribution_entity_min_support_ratio": self.attribution_entity_min_support_ratio,
        }

    def _attributions_to_items(
        self, sentences: list[AttributedSentence]
    ) -> list[AttributionItem]:
        return [
            AttributionItem(
                sid=s.sid,
                text=s.text,
                attributed_to=[
                    AttributedTo(doc_name=a.doc_name, page=a.page)
                    for a in s.attributed_to
                ],
            )
            for s in sentences
        ]

    def run(self, payload: ChallengeInput, with_task2: bool = True) -> ChallengeOutput:
        """Exécute Tâche 1 (+ Tâche 2 si demandé) sur toutes les questions."""
        params = self._parameters()
        run_id = payload.run_id

        task1_results: list[Task1Result] = []
        task2_results: list[Task2Result] = []

        for q in payload.results:
            logger.info(f"[Challenge] {q.qid} : {q.question[:80]}...")
            resp = self.engine.answer(
                query=q.question,
                top_k_chunks=self.top_k_chunks,
                top_n_pages=self.top_n_pages,
                context_chunks=self.context_chunks,
                retrieval_only=self.retrieval_only,
            )

            task1_results.append(
                Task1Result(
                    qid=q.qid,
                    question=q.question,
                    retrieved=[
                        RetrievedItem(rank=r.rank, doc_name=r.doc_name, page=r.page)
                        for r in resp.retrieved
                    ],
                    answer=resp.answer,
                    metadata={"tokens_used": resp.tokens_used},
                )
            )

            if with_task2 and resp.answer and resp.chunks:
                sentences = self.engine.attribute(
                    qid=q.qid,
                    answer=resp.answer,
                    chunks=resp.chunks,
                    threshold=self.attribution_threshold,
                    secondary_threshold=self.attribution_secondary_threshold,
                    topk_per_sentence=self.attribution_topk,
                    entity_filter=self.attribution_entity_filter,
                    entity_min_support_ratio=self.attribution_entity_min_support_ratio,
                )
                task2_results.append(
                    Task2Result(
                        qid=q.qid,
                        attributions=self._attributions_to_items(sentences),
                    )
                )
            elif with_task2:
                # Cas LLM skipped ou aucun chunk : task2 vide pour cette question
                task2_results.append(Task2Result(qid=q.qid, attributions=[]))

        task1_run = Task1Run(run_id=run_id, parameters=params, results=task1_results)
        task2_run = (
            Task2Run(run_id=run_id, parameters=params, results=task2_results)
            if with_task2
            else None
        )
        return ChallengeOutput(task1=task1_run, task2=task2_run)
