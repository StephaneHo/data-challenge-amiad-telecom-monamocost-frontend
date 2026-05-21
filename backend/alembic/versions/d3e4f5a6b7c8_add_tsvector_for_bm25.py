"""add tsvector content_tsv + GIN index for BM25 hybrid retrieval

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-05-21

Ajoute une colonne `content_tsv` générée automatiquement par Postgres à partir de
`content` via `to_tsvector('french', ...)`, plus un index GIN pour des recherches
plein-texte rapides (`@@`, `ts_rank`).

Permet la **recherche hybride BM25 + dense** dans le retrieval :
- Le dense capture la similarité sémantique (e5 embeddings)
- Le BM25 (`ts_rank` sur le tsvector) capture les correspondances lexicales exactes
  (acronymes, noms propres rares comme `MQ-9`, `r20-7111`)
- On fusionne les deux par RRF (Reciprocal Rank Fusion)

Aucune perte de données : la colonne est `GENERATED ALWAYS AS (...) STORED`,
donc auto-remplie pour les chunks existants et futurs.
"""

from typing import Sequence, Union

from alembic import op


revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, Sequence[str], None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Colonne générée automatiquement par Postgres (12+).
    # Le coalesce protège contre les NULL hypothétiques (content est NOT NULL en
    # principe, mais on reste défensif).
    op.execute(
        "ALTER TABLE chunks "
        "ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('french', coalesce(content, ''))) STORED"
    )
    # Index GIN pour recherches @@ rapides
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_content_tsv "
        "ON chunks USING gin(content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
