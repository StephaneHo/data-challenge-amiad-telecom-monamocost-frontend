# backend/database/models.py
from __future__ import annotations
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from config import settings

"""
classe de base pour tous les modèles
DeclarativeBase transforme les classes Python en tables SQL
"""


class Base(DeclarativeBase):
    pass


"""
Many to many
un paper peut avoir plusieurs auteurs
un auteur a plusieurs papers
"""
paper_author = Table(
    "paper_author",
    Base.metadata,
    Column("paper_id", String(64), ForeignKey("papers.arxiv_id"), primary_key=True),
    Column("author_id", Integer, ForeignKey("authors.id"), primary_key=True),
)

"""
Many to Many
un paper peut avoir plusieurs catégories
une catégorie peut faire référence à plusieurs papers
"""
paper_category = Table(
    "paper_category",
    Base.metadata,
    Column("paper_id", String(64), ForeignKey("papers.arxiv_id"), primary_key=True),
    Column("category_id", Integer, ForeignKey("categories.id"), primary_key=True),
)

"""
Many to many
un paper peut faire référence à plusieurs conférences
une conférence peut faire mention de plusieurs papers
"""
paper_conference = Table(
    "paper_conference",
    Base.metadata,
    Column("paper_id", String(64), ForeignKey("papers.arxiv_id"), primary_key=True),
    Column("conference_id", Integer, ForeignKey("conferences.id"), primary_key=True),
)


class Paper(Base):
    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    pdf_url: Mapped[Optional[str]] = mapped_column(String(512))
    html_url: Mapped[Optional[str]] = mapped_column(String(512))
    doi: Mapped[Optional[str]] = mapped_column(String(128))
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    semantic_scholar_id: Mapped[Optional[str]] = mapped_column(String(64))
    accepted: Mapped[Optional[bool]] = mapped_column(Boolean)
    venue: Mapped[Optional[str]] = mapped_column(String(256))
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=True
    )

    """ many to many avec les auteurs """
    authors: Mapped[List["Author"]] = relationship(
        "Author", secondary=paper_author, back_populates="papers"
    )
    """many to many avec les catégories"""
    categories: Mapped[List["Category"]] = relationship(
        "Category", secondary=paper_category, back_populates="papers"
    )
    """many to many avec les conférences"""
    conferences: Mapped[List["Conference"]] = relationship(
        "Conference", secondary=paper_conference, back_populates="papers"
    )
    """one to many: un paper peut avoir plusieurs repos mais un repo n'est en principe que dédidé à un paper"""
    github_repos: Mapped[List["GitHubRepo"]] = relationship(
        "GitHubRepo", back_populates="paper"
    )
    """
       one-to-many: un papier peut faire référence à plusieurs modèles
       on introduit une simplification: on ne modélise pas un modèle HF peut être cité dans plusieurs papers 
       théoriquement ce serait du many-to-many, mais on simplifie en supposant que un modèle HF est associé à un seul Paper
       on veut juste afficher les modèles liés au Paper et pas besoin de mutualiser les modèles
    """
    hf_models: Mapped[List["HFModel"]] = relationship("HFModel", back_populates="paper")
    """
        one-to-many: pas d'ambiguités ici
        si on supprime un paper, on supprime tous les chunks associés et en plus, on veut que un chunk ne soit pas relié à rien
    """
    chunks: Mapped[List["PaperChunk"]] = relationship(
        "PaperChunk", back_populates="paper", cascade="all, delete-orphan"
    )


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    affiliation: Mapped[Optional[str]] = mapped_column(String(512))
    country: Mapped[Optional[str]] = mapped_column(String(128))
    email: Mapped[Optional[str]] = mapped_column(String(256))
    orcid: Mapped[Optional[str]] = mapped_column(String(32))
    scholar_id: Mapped[Optional[str]] = mapped_column(String(64))
    h_index: Mapped[Optional[int]] = mapped_column(Integer)

    papers: Mapped[List["Paper"]] = relationship(
        "Paper", secondary=paper_author, back_populates="authors"
    )

    __table_args__ = (UniqueConstraint("name", "orcid", name="uq_author_orcid"),)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(128))
    parent: Mapped[Optional[str]] = mapped_column(String(32))

    papers: Mapped[List["Paper"]] = relationship(
        "Paper", secondary=paper_category, back_populates="categories"
    )


class Conference(Base):
    __tablename__ = "conferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    acronym: Mapped[Optional[str]] = mapped_column(String(32), unique=True)
    year: Mapped[Optional[int]] = mapped_column(Integer)
    rank: Mapped[Optional[str]] = mapped_column(String(8))
    url: Mapped[Optional[str]] = mapped_column(String(512))

    papers: Mapped[List["Paper"]] = relationship(
        "Paper", secondary=paper_conference, back_populates="conferences"
    )


class GitHubRepo(Base):
    __tablename__ = "github_repos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("papers.arxiv_id")
    )
    full_name: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(String(512))
    description: Mapped[Optional[str]] = mapped_column(Text)
    stars: Mapped[int] = mapped_column(Integer, default=0)
    forks: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[Optional[str]] = mapped_column(String(64))
    topics: Mapped[Optional[list]] = mapped_column(ARRAY(String))
    last_commit: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    stars_history: Mapped[Optional[dict]] = mapped_column(JSONB)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    paper: Mapped[Optional["Paper"]] = relationship(
        "Paper", back_populates="github_repos"
    )


class HFModel(Base):
    __tablename__ = "hf_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("papers.arxiv_id")
    )
    hf_id: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(16))
    downloads: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[Optional[list]] = mapped_column(ARRAY(String))
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    paper: Mapped[Optional["Paper"]] = relationship("Paper", back_populates="hf_models")


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.arxiv_id"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(128))
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=True
    )

    paper: Mapped["Paper"] = relationship("Paper", back_populates="chunks")
