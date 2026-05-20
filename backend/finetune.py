"""
CLI : fine-tune l'embedder e5 sur les paires (question, chunk) construites
à partir d'un fichier au format `sample_queries.json`.

Usage :
    python finetune.py --gold ../Experimental/DATA/sample_queries.json
    python finetune.py --gold gold.json --output-dir models/e5-base-ft-run2 --epochs 5
    python finetune.py --gold gold.json --batch-size 4 --lr 1e-5 --val-ratio 0.0  # full train
    python finetune.py --gold gold.json --dry-run  # construit les paires sans entraîner

    # Combine plusieurs sources extra (OSINT target + questions synthétiques) :
    python finetune.py --gold ../Experimental/DATA/sample_queries.json \
        --extra-examples ../Experimental/DATA/extra_training_examples.json \
                         ../Experimental/DATA/synthetic_questions.json

À la fin, le modèle est sauvegardé dans `models/<output_dir>` (et `<output_dir>.best`
si une validation a tourné). Pour l'utiliser :

    1) Mettre `EMBEDDING_MODEL=models/e5-base-monamo-ft.best` dans `.env`
    2) Lancer la migration de dim si nécessaire (e5-base reste 768d → OK)
    3) `python ingest.py --force-reingest` pour re-embedder tous les chunks
       avec le nouveau modèle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from database.db import get_session
from pipeline.finetune import (
    FineTuneConfig,
    TrainingExample,
    build_training_pairs,
    fine_tune,
    load_extra_examples,
    train_val_split,
)


def _export_pairs(examples: list[TrainingExample], path: Path) -> None:
    """Sérialise les paires d'entraînement en JSON autonome (sans dépendance DB)."""
    rows = [
        {
            "question": ex.question,
            "paragraph": ex.paragraph,
            "doc_name": ex.doc_name,
            "page": ex.page_number,
        }
        for ex in examples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[CLI] Export : {len(rows)} paires écrites dans {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning de l'embedder e5 (multi-positifs)")
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Fichier JSON au format sample_queries (questions + retrieved doc/page). "
        "Nécessite la DB Postgres pour résoudre les chunks. Optionnel si --extra-examples fourni.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="Modèle SentenceTransformer de départ (défaut: e5-base, 768d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/e5-base-monamo-ft"),
        help="Dossier de sortie du modèle entraîné",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction de questions mises en validation (0 = tout en train)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--extra-examples",
        type=Path,
        nargs="+",
        default=None,
        help="Un ou plusieurs JSON additionnels avec des paires (question, paragraph, "
        "doc_name, page). Accepte plusieurs fichiers pour combiner sources synthétiques "
        "(ex: --extra-examples extra_training_examples.json synthetic_questions.json)",
    )
    parser.add_argument(
        "--export-to",
        type=Path,
        default=None,
        help="Au lieu d'entraîner, exporte le training set fusionné (gold + extras) "
        "dans un JSON autonome (sans dépendance DB). Utile pour passer le set à un "
        "collègue avec GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construit les paires d'entraînement et affiche un résumé, sans entraîner",
    )
    args = parser.parse_args()

    # Cas autonome (sans DB) : juste --extra-examples, pas de --gold
    if args.gold is None and args.extra_examples:
        examples: list[TrainingExample] = []
        for extra_path in args.extra_examples:
            examples.extend(load_extra_examples(extra_path))
        if not examples:
            logger.error("[CLI] Aucune paire construite depuis les --extra-examples.")
            return 1
        return _run_after_load(args, examples)

    if args.gold is None:
        logger.error("[CLI] Il faut au moins --gold ou --extra-examples.")
        return 1

    with get_session() as session:
        examples = build_training_pairs(session, args.gold)
        if args.extra_examples:
            for extra_path in args.extra_examples:
                examples.extend(load_extra_examples(extra_path))
        if not examples:
            logger.error("[CLI] Aucune paire construite — vérifier que le corpus est ingéré.")
            return 1
        return _run_after_load(args, examples)


def _run_after_load(args: argparse.Namespace, examples: list[TrainingExample]) -> int:
    """Branche commune : split, dry-run / export / training."""
    if args.export_to:
        _export_pairs(examples, args.export_to)
        return 0

    train, val = train_val_split(examples, val_ratio=args.val_ratio, seed=args.seed)

    if args.dry_run:
        logger.info(f"[CLI] Dry-run : {len(train)} train / {len(val)} val. Aperçu :")
        for ex in train[:3]:
            logger.info(
                f"  q='{ex.question[:60]}...' | {ex.doc_name} p.{ex.page_number} | "
                f"para[:80]='{ex.paragraph[:80]}...'"
            )
        return 0

    cfg = FineTuneConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        val_ratio=args.val_ratio,
        seed=args.seed,
        log_every=args.log_every,
    )
    fine_tune(train_examples=train, val_examples=val if val else None, config=cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
