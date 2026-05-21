"""
CLI : génération synthétique de questions par chunk pour data augmentation.

Sortie au format `extra_training_examples.json` (utilisable par
`finetune.py --extra-examples ...`).

Le cache est idempotent : relancer ne re-paye que les chunks non encore couverts.

Usage :
    # Estimation coût + aperçu sans appel LLM
    python synth_questions.py --dry-run --limit 50

    # Génération complète (901 chunks → ~1$ sur gpt-4o-mini)
    python synth_questions.py

    # Génération restreinte à certains docs
    python synth_questions.py --doc-filter "open-source-intelligence.pdf" "l16b1454_rapport-information.pdf"

    # Sortie custom + 3 questions par chunk
    python synth_questions.py --output ../Experimental/DATA/runs/synthetic_v2.json --questions-per-chunk 3

    # Ré-génère tout (ignore cache existant)
    python synth_questions.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from database.db import get_session
from pipeline.synth_questions import SyntheticQuestionGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="Génération synthétique de questions (data aug.)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Experimental/DATA/training/synthetic_questions.json"),
        help="Chemin du JSON de sortie (cache idempotent)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Modèle LLM (défaut: gpt-4o-mini)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température LLM (0.7 = diversité raisonnable)",
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=2,
        help="Nombre de questions générées par chunk (défaut: 2)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max chunks à traiter (défaut: tous)"
    )
    parser.add_argument(
        "--doc-filter",
        nargs="+",
        default=None,
        help="Restreint la génération à ces doc_names",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=80,
        help="Ignore les chunks plus courts (défaut: 80)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Sauvegarde intermédiaire toutes les N chunks (résiste au Ctrl+C)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Pas d'appel LLM, juste l'estimation")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore le cache, regénère même pour les chunks déjà couverts",
    )
    args = parser.parse_args()

    with get_session() as session:
        gen = SyntheticQuestionGenerator(
            session=session,
            output_path=args.output,
            model=args.model,
            temperature=args.temperature,
            questions_per_chunk=args.questions_per_chunk,
            min_chunk_chars=args.min_chunk_chars,
        )
        n = gen.run(
            limit=args.limit,
            doc_filter=args.doc_filter,
            flush_every=args.flush_every,
            dry_run=args.dry_run,
            force=args.force,
        )
        if not args.dry_run and n > 0:
            logger.info(
                f"[CLI] {n} nouvelles paires synthétiques. Utiliser avec :"
            )
            logger.info(
                f"  python finetune.py --gold ../Experimental/DATA/training/sample_queries.json "
                f"--extra-examples {args.output}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
