from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from config import settings
from database.models import Chunk, Document, DocumentPage
from pipeline.chunker import chunk_page
from pipeline.embedder import Embedder
from pipeline.pdf_extractor import PDFExtractor


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


class PDFIngestionPipeline:
    """
    Pipeline d'ingestion du corpus du challenge.

    Pour chaque PDF d'un dossier :
      1. extrait page par page (pdfplumber, OCR en fallback)
      2. découpe en paragraphes/chunks au niveau page
      3. génère un embedding par chunk (sauf si `skip_embeddings=True`)
      4. persiste documents/document_pages/chunks
    """

    def __init__(
        self,
        session: Session,
        force_ocr: bool = False,
        ocr_lang: Optional[str] = None,
        skip_embeddings: bool = False,
    ) -> None:
        self.session = session
        self.extractor = PDFExtractor(force_ocr=force_ocr, ocr_lang=ocr_lang)
        self.skip_embeddings = skip_embeddings
        self.embedder: Optional[Embedder] = None

        if not skip_embeddings:
            self.embedder = Embedder()
            if self.embedder.dim != settings.EMBEDDING_DIM:
                raise RuntimeError(
                    f"[Ingestion] Désaccord de dimension : modèle {self.embedder.model_name} "
                    f"produit {self.embedder.dim}d mais EMBEDDING_DIM={settings.EMBEDDING_DIM}. "
                    "Aligner config.py / .env / migration."
                )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is None:
            return [[] for _ in texts]
        return self.embedder.encode_passages(texts)

    def _existing_document(self, doc_name: str) -> Optional[Document]:
        return self.session.get(Document, doc_name)

    def _delete_document_if_exists(self, doc_name: str) -> None:
        """Supprime un Document et tout son contenu (cascade)."""
        doc = self.session.get(Document, doc_name)
        if doc is not None:
            self.session.delete(doc)
            self.session.flush()

    def ingest_pdf(self, pdf_path: Path, force_reingest: bool = False) -> bool:
        """
        Ingère un PDF. Retourne True si du travail a été fait, False si skip.
        Stratégie d'idempotence : si le hash du fichier matche, on skip,
        sinon on supprime l'ancien doc et on réingère.
        """
        pdf_path = Path(pdf_path).resolve()
        doc_name = pdf_path.name
        file_hash = _sha256(pdf_path)

        existing = self._existing_document(doc_name)
        if existing and existing.file_hash == file_hash and not force_reingest:
            logger.info(f"[Ingestion] Skip (déjà ingéré) : {doc_name}")
            return False

        if existing is not None:
            logger.info(f"[Ingestion] Re-ingestion : {doc_name}")
            self._delete_document_if_exists(doc_name)

        logger.info(f"[Ingestion] Extraction : {doc_name}")
        pages = self.extractor.extract_pages(pdf_path)
        if not pages:
            logger.warning(f"[Ingestion] {doc_name} : aucune page extraite")
            return False

        modes = {p.extraction_mode for p in pages}
        extraction_mode = "mixed" if len(modes) > 1 else next(iter(modes))

        doc = Document(
            doc_name=doc_name,
            file_hash=file_hash,
            n_pages=len(pages),
            source_path=str(pdf_path),
            extraction_mode=extraction_mode,
            lang=settings.OCR_LANG[:2],  # "fra" -> "fr"
        )
        self.session.add(doc)
        self.session.flush()

        # Persiste les pages brutes
        for p in pages:
            self.session.add(
                DocumentPage(
                    doc_name=doc_name,
                    page_number=p.page_number,
                    raw_text=p.text,
                )
            )

        # Découpe + embed par page
        all_chunks: list[tuple[int, int, str]] = []  # (page_number, chunk_index, content)
        for p in pages:
            for c in chunk_page(
                page_number=p.page_number,
                text=p.text,
                min_chars=settings.CHUNK_MIN_CHARS,
                max_chars=settings.CHUNK_MAX_CHARS,
            ):
                all_chunks.append((c.page_number, c.chunk_index, c.content))

        if not all_chunks:
            logger.warning(f"[Ingestion] {doc_name} : aucun chunk produit")
            self.session.commit()
            return True

        contents = [c[2] for c in all_chunks]
        embeddings = self._embed_batch(contents)

        for (page_number, chunk_index, content), emb in zip(all_chunks, embeddings):
            self.session.add(
                Chunk(
                    doc_name=doc_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    content=content,
                    n_tokens=len(content.split()),
                    embedding=emb if emb else None,
                )
            )

        self.session.commit()
        logger.info(
            f"[Ingestion] {doc_name} : {len(pages)} pages, {len(all_chunks)} chunks "
            f"(mode={extraction_mode}, embeddings={'non' if self.skip_embeddings else 'oui'})"
        )
        return True

    def run(self, corpus_dir: Path, force_reingest: bool = False) -> int:
        """Ingère tous les PDFs d'un dossier. Retourne le nombre de fichiers traités."""
        corpus_dir = Path(corpus_dir).resolve()
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus introuvable : {corpus_dir}")

        pdfs = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() == ".pdf")
        logger.info(f"[Ingestion] {len(pdfs)} PDFs trouvés dans {corpus_dir}")

        n_done = 0
        for pdf in pdfs:
            try:
                if self.ingest_pdf(pdf, force_reingest=force_reingest):
                    n_done += 1
            except Exception as ex:
                logger.exception(f"[Ingestion] Erreur sur {pdf.name} : {ex}")
                self.session.rollback()
        logger.info(f"[Ingestion] Terminé : {n_done}/{len(pdfs)} fichiers traités")
        return n_done
