"""challenge_schema: drop arxiv tables and create challenge tables

Revision ID: b1c2d3e4f5a6
Revises: f136a1278da9
Create Date: 2026-05-20

Refonte du schéma pour le challenge RAG EvalLLM 2026 :
- supprime le schéma arXiv (papers, authors, categories, conferences, chunks de papers, etc.)
- crée documents / document_pages / chunks avec granularité (doc_name, page_number)
- crée l'extension pgvector si elle n'existe pas et un index ANN sur chunks.embedding

Migration destructive : toutes les données arXiv préexistantes sont perdues.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

from config import settings


revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, Sequence[str], None] = "f136a1278da9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop arxiv schema (ordre : tables de jointure → tables liées → tables racines)
    op.drop_table("paper_conference")
    op.drop_table("paper_chunks")
    op.drop_table("paper_category")
    op.drop_table("paper_author")
    op.drop_table("hf_models")
    op.drop_table("github_repos")
    op.drop_table("papers")
    op.drop_table("conferences")
    op.drop_table("categories")
    op.drop_table("authors")

    # S'assure que pgvector est dispo (au cas où l'init-db n'aurait pas tourné)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("doc_name", sa.String(length=512), nullable=False),
        sa.Column("file_hash", sa.String(length=64), nullable=True),
        sa.Column("n_pages", sa.Integer(), nullable=False),
        sa.Column("source_path", sa.String(length=1024), nullable=True),
        sa.Column("extraction_mode", sa.String(length=32), nullable=True),
        sa.Column("lang", sa.String(length=8), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("doc_name"),
    )

    op.create_table(
        "document_pages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doc_name", sa.String(length=512), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["doc_name"], ["documents.doc_name"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_name", "page_number", name="uq_document_pages_doc_page"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doc_name", sa.String(length=512), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("n_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=settings.EMBEDDING_DIM),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["doc_name"], ["documents.doc_name"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "doc_name", "page_number", "chunk_index", name="uq_chunks_doc_page_idx"
        ),
    )
    op.create_index("ix_chunks_doc_name", "chunks", ["doc_name"])
    op.create_index("ix_chunks_doc_page", "chunks", ["doc_name", "page_number"])

    # Index ANN sur les embeddings (HNSW si dispo, sinon ivfflat).
    # `lists` pour ivfflat = sqrt(N) ~ 100 marche bien pour quelques k chunks.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_index("ix_chunks_doc_page", table_name="chunks")
    op.drop_index("ix_chunks_doc_name", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("document_pages")
    op.drop_table("documents")

    # Recrée le schéma arXiv vide (downgrade strict — pas de récup de données).
    op.create_table(
        "authors",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("affiliation", sa.String(length=512), nullable=True),
        sa.Column("country", sa.String(length=128), nullable=True),
        sa.Column("email", sa.String(length=256), nullable=True),
        sa.Column("orcid", sa.String(length=32), nullable=True),
        sa.Column("scholar_id", sa.String(length=64), nullable=True),
        sa.Column("h_index", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "orcid", name="uq_author_orcid"),
    )
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("parent", sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_table(
        "conferences",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("acronym", sa.String(length=32), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("rank", sa.String(length=8), nullable=True),
        sa.Column("url", sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("acronym"),
    )
    op.create_table(
        "papers",
        sa.Column("arxiv_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("pdf_url", sa.String(length=512), nullable=True),
        sa.Column("html_url", sa.String(length=512), nullable=True),
        sa.Column("doi", sa.String(length=128), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("citation_count", sa.Integer(), nullable=False),
        sa.Column("semantic_scholar_id", sa.String(length=64), nullable=True),
        sa.Column("accepted", sa.Boolean(), nullable=True),
        sa.Column("venue", sa.String(length=256), nullable=True),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=settings.EMBEDDING_DIM),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("arxiv_id"),
    )
    op.create_table(
        "paper_author",
        sa.Column("paper_id", sa.String(length=64), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["author_id"], ["authors.id"]),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.arxiv_id"]),
        sa.PrimaryKeyConstraint("paper_id", "author_id"),
    )
    op.create_table(
        "paper_category",
        sa.Column("paper_id", sa.String(length=64), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["category_id"], ["categories.id"]),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.arxiv_id"]),
        sa.PrimaryKeyConstraint("paper_id", "category_id"),
    )
    op.create_table(
        "paper_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("section", sa.String(length=128), nullable=True),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=settings.EMBEDDING_DIM),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.arxiv_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "paper_conference",
        sa.Column("paper_id", sa.String(length=64), nullable=False),
        sa.Column("conference_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["conference_id"], ["conferences.id"]),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.arxiv_id"]),
        sa.PrimaryKeyConstraint("paper_id", "conference_id"),
    )
    op.create_table(
        "github_repos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.String(length=64), nullable=True),
        sa.Column("full_name", sa.String(length=256), nullable=False),
        sa.Column("url", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("stars", sa.Integer(), nullable=False),
        sa.Column("forks", sa.Integer(), nullable=False),
        sa.Column("language", sa.String(length=64), nullable=True),
        sa.Column("topics", sa.dialects.postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("last_commit", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "stars_history",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "collected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.arxiv_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("full_name"),
    )
    op.create_table(
        "hf_models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.String(length=64), nullable=True),
        sa.Column("hf_id", sa.String(length=256), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("downloads", sa.Integer(), nullable=False),
        sa.Column("likes", sa.Integer(), nullable=False),
        sa.Column("tags", sa.dialects.postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column(
            "collected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.arxiv_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hf_id"),
    )
