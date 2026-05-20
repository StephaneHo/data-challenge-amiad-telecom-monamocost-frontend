from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from config import settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    """
    Un PDF du corpus du challenge.
    `doc_name` est l'identifiant attendu par le format JSON du challenge
    (champ `doc_name` dans `retrieved` / `attributed_to`).
    """

    __tablename__ = "documents"

    doc_name: Mapped[str] = mapped_column(String(512), primary_key=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64))
    n_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    source_path: Mapped[Optional[str]] = mapped_column(String(1024))
    extraction_mode: Mapped[Optional[str]] = mapped_column(String(32))
    lang: Mapped[Optional[str]] = mapped_column(String(8))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    pages: Mapped[List["DocumentPage"]] = relationship(
        "DocumentPage", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentPage(Base):
    """
    Page physique d'un document (numérotation 1-based, couverture = page 1).
    On stocke le texte brut pour pouvoir rejouer le chunking ou faire de
    l'attribution de phrases à la page sans réextraire le PDF.
    """

    __tablename__ = "document_pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_name: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("documents.doc_name", ondelete="CASCADE"),
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[Optional[str]] = mapped_column(Text)

    document: Mapped["Document"] = relationship("Document", back_populates="pages")

    __table_args__ = (
        UniqueConstraint("doc_name", "page_number", name="uq_document_pages_doc_page"),
    )


class Chunk(Base):
    """
    Unité indexée pour le retrieval.
    `page_number` est la page physique de provenance — c'est le pivot du challenge :
    plusieurs chunks peuvent provenir de la même page, et la métrique de retrieval
    est calculée à la granularité (doc_name, page_number).
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_name: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("documents.doc_name", ondelete="CASCADE"),
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    n_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=True
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "doc_name", "page_number", "chunk_index", name="uq_chunks_doc_page_idx"
        ),
        Index("ix_chunks_doc_name", "doc_name"),
        Index("ix_chunks_doc_page", "doc_name", "page_number"),
    )
