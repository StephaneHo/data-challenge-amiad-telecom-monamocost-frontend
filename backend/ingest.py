"""
Point d'entrée CLI pour l'ingestion du corpus PDF du challenge.

Usage :
    python ingest.py                         # ingère CORPUS_DIR (config)
    python ingest.py --corpus path/to/dir    # corpus custom
    python ingest.py --force-ocr             # force OCR sur toutes les pages
    python ingest.py --skip-embeddings       # n'index pas les vecteurs (test rapide)
    python ingest.py --force-reingest        # réingère même si le hash matche
    python ingest.py --pdf path/to/file.pdf  # ingère un seul PDF
    python ingest.py --files a.pdf b.pdf ... # ingère un sous-ensemble explicite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from config import settings
from database.db import get_session
from pipeline.ingestion import PDFIngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestion du corpus PDF (challenge EvalLLM)")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(settings.CORPUS_DIR),
        help=f"Dossier contenant les PDFs (défaut: {settings.CORPUS_DIR})",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Ingère un seul PDF au lieu du dossier complet",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="Ingère uniquement les PDFs listés (chemins relatifs ou absolus)",
    )
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--force-reingest", action="store_true")
    args = parser.parse_args()

    with get_session() as session:
        pipeline = PDFIngestionPipeline(
            session=session,
            force_ocr=args.force_ocr,
            skip_embeddings=args.skip_embeddings,
        )

        if args.pdf:
            ok = pipeline.ingest_pdf(args.pdf, force_reingest=args.force_reingest)
            return 0 if ok else 1

        if args.files:
            n = 0
            for f in args.files:
                try:
                    if pipeline.ingest_pdf(f, force_reingest=args.force_reingest):
                        n += 1
                except Exception as ex:
                    logger.exception(f"[CLI] Erreur sur {f} : {ex}")
                    session.rollback()
            logger.info(f"[CLI] {n}/{len(args.files)} fichiers ingérés")
            return 0

        n = pipeline.run(args.corpus, force_reingest=args.force_reingest)
        logger.info(f"[CLI] {n} fichiers ingérés")
    return 0


if __name__ == "__main__":
    sys.exit(main())
