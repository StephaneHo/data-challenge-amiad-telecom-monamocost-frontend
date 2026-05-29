from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from loguru import logger


class Reranker:
    """Cross-encoder reranker for query/chunk relevance scoring."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            f"[Reranker] Chargement {self.model_name} sur {self.device}"
        )
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        chunks: Sequence[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        """Return chunks sorted by cross-encoder score, highest first."""
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        return [replace(chunk, score=float(score)) for chunk, score in ranked]
