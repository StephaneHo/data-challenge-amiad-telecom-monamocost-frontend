from __future__ import annotations

from typing import Optional

from loguru import logger

from config import settings


class Embedder:
    """
    Encode texte → vecteur normalisé pour pgvector.

    Pour les modèles e5 (`intfloat/multilingual-e5-*`), la convention exige
    de préfixer les entrées :
        - documents indexés : "passage: <texte>"
        - requêtes utilisateur : "query: <texte>"

    Cette classe centralise ce contrat pour que l'ingestion et le retrieval
    n'aient pas à diverger.
    """

    PASSAGE_PREFIX = "passage: "
    QUERY_PREFIX = "query: "

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size
        logger.info(f"[Embedder] Chargement {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self._needs_prefix = "e5" in self.model_name.lower()
        if self._needs_prefix:
            logger.info("[Embedder] Préfixes e5 activés (passage:/query:)")

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _prep(self, texts: list[str], prefix: str) -> list[str]:
        if not self._needs_prefix:
            return texts
        return [prefix + t for t in texts]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import torch

        with torch.no_grad():
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
        return [v.tolist() for v in vectors]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(self._prep(texts, self.PASSAGE_PREFIX))

    def encode_query(self, text: str) -> list[float]:
        prepped = self._prep([text], self.QUERY_PREFIX)
        return self._encode(prepped)[0]
