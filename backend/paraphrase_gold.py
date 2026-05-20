"""
CLI : génère des paraphrases LLM des questions gold pour data augmentation.

Sortie au format `extra_training_examples.json`, utilisable par
`finetune.py --gold-extra paraphrases.json`.

Workflow combiné avec word dropout (option A) et lower upweight (option C) :
    python paraphrase_gold.py
    python finetune.py \
      --gold ../Experimental/DATA/sample_queries.json \
      --gold-extra ../Experimental/DATA/extra_training_examples.json \
                   ../Experimental/DATA/gold_paraphrases.json \
      --extra-examples ../Experimental/DATA/synthetic_questions.json \
      --gold-upweight 5 \
      --question-dropout-gold 0.10 \
      --epochs 3

Usage standalone :
    # Estimation + aperçu sans appel LLM
    python paraphrase_gold.py --dry-run

    # Génération depuis sample_queries.json (lit la DB pour les paragraphes)
    python paraphrase_gold.py --gold ../Experimental/DATA/sample_queries.json

    # Génération depuis un fichier extra autonome (sans DB)
    python paraphrase_gold.py --gold-extra ../Experimental/DATA/extra_training_examples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from database.db import get_session
from pipeline.finetune import build_training_pairs, load_extra_examples
from pipeline.paraphrase_gold import GoldPair, GoldParaphraser


def _gather_gold_pairs(args: argparse.Namespace) -> list[GoldPair]:
    pairs: list[GoldPair] = []

    if args.gold is not None:
        with get_session() as session:
            for ex in build_training_pairs(session, args.gold):
                pairs.append(
                    GoldPair(
                        question=ex.question,
                        paragraph=ex.paragraph,
                        doc_name=ex.doc_name,
                        page=ex.page_number,
                    )
                )

    if args.gold_extra:
        for p in args.gold_extra:
            for ex in load_extra_examples(p, is_gold=True):
                pairs.append(
                    GoldPair(
                        question=ex.question,
                        paragraph=ex.paragraph,
                        doc_name=ex.doc_name,
                        page=ex.page_number,
                    )
                )

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Paraphrasing LLM des questions gold")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("../Experimental/DATA/sample_queries.json"),
        help="JSON au format sample_queries (questions + retrieved doc/page). "
        "Nécessite la DB. Mettre à None (--gold '') pour ignorer.",
    )
    parser.add_argument(
        "--gold-extra",
        type=Path,
        nargs="+",
        default=[Path("../Experimental/DATA/extra_training_examples.json")],
        help="Fichier(s) JSON gold autonomes (ex: OSINT du target.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Experimental/DATA/gold_paraphrases.json"),
        help="JSON de sortie (cache idempotent)",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Température LLM (0.9 = forte diversité, recommandé pour paraphrases)",
    )
    parser.add_argument(
        "--paraphrases-per-question",
        type=int,
        default=10,
        help="Nombre de paraphrases générées par question gold (défaut: 10)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore le cache, regénère même pour les questions déjà traitées",
    )
    args = parser.parse_args()

    # Permet de désactiver --gold via une chaîne vide
    if isinstance(args.gold, Path) and str(args.gold) == ".":
        args.gold = None

    pairs = _gather_gold_pairs(args)
    if not pairs:
        logger.error("[CLI] Aucune paire gold trouvée (vérifier --gold / --gold-extra).")
        return 1

    paraphraser = GoldParaphraser(
        output_path=args.output,
        model=args.model,
        temperature=args.temperature,
        paraphrases_per_question=args.paraphrases_per_question,
    )
    n = paraphraser.run(pairs, dry_run=args.dry_run, force=args.force)
    if not args.dry_run and n > 0:
        logger.info(f"[CLI] {n} nouvelles paraphrases. À utiliser avec :")
        logger.info(
            f"  python finetune.py --gold-extra {args.output} ..."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
