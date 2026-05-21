"""
CLI : fine-tune l'embedder e5 sur les paires (question, chunk) construites
à partir d'un fichier au format `sample_queries.json`.

Usage :
    python finetune.py --gold ../Experimental/DATA/training/sample_queries.json
    python finetune.py --gold gold.json --output-dir models/e5-base-ft-run2 --epochs 5
    python finetune.py --gold gold.json --batch-size 4 --lr 1e-5 --val-ratio 0.0  # full train
    python finetune.py --gold gold.json --dry-run  # construit les paires sans entraîner

    # Combine plusieurs sources extra (OSINT target + questions synthétiques) :
    python finetune.py --gold ../Experimental/DATA/training/sample_queries.json \
        --extra-examples ../Experimental/DATA/training/extra_training_examples.json \
                         ../Experimental/DATA/training/synthetic_questions.json

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
    """
    Sérialise les paires d'entraînement en JSON autonome (sans dépendance DB).
    Le flag `_is_gold` est conservé pour qu'un consommateur (collègue GPU) puisse
    appliquer `--gold-upweight` correctement.
    """
    rows = [
        {
            "question": ex.question,
            "paragraph": ex.paragraph,
            "doc_name": ex.doc_name,
            "page": ex.page_number,
            "_is_gold": ex.is_gold,
        }
        for ex in examples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    n_gold = sum(1 for ex in examples if ex.is_gold)
    logger.info(
        f"[CLI] Export : {len(rows)} paires écrites dans {path} "
        f"({n_gold} gold, {len(rows) - n_gold} synth)"
    )


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
        "--gold-extra",
        type=Path,
        nargs="+",
        default=None,
        help="Comme --extra-examples mais les paires sont marquées comme gold "
        "(éligibles à --gold-upweight). Utile pour l'exemple OSINT du target.txt.",
    )
    parser.add_argument(
        "--gold-upweight",
        type=int,
        default=1,
        help="Duplique N fois les paires gold (--gold + --gold-extra) dans le train set, "
        "APRÈS le split val. Compense le déséquilibre face aux paires synthétiques. "
        "Reco : 5-10 avec paraphrases LLM activées (cf. paraphrase_gold.py), "
        "ou 20-50 sans paraphrases (mais risque de mémorisation). "
        "Défaut: 1 = pas d'upweight.",
    )
    parser.add_argument(
        "--question-dropout-gold",
        type=float,
        default=0.0,
        help="Probabilité [0..1] de drop d'un mot dans les questions gold à chaque "
        "récupération d'item (différent par epoch). Casse la mémorisation lexicale "
        "quand --gold-upweight > 1. Reco : 0.10-0.15. Défaut: 0 = pas de dropout.",
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

    # Cas autonome (sans DB) : juste --extra-examples / --gold-extra, pas de --gold
    if args.gold is None:
        if not args.extra_examples and not args.gold_extra:
            logger.error("[CLI] Il faut au moins --gold, --extra-examples ou --gold-extra.")
            return 1
        examples: list[TrainingExample] = []
        for p in args.gold_extra or []:
            examples.extend(load_extra_examples(p, is_gold=True))
        for p in args.extra_examples or []:
            examples.extend(load_extra_examples(p, is_gold=False))
        if not examples:
            logger.error("[CLI] Aucune paire construite depuis les --extra-examples.")
            return 1
        return _run_after_load(args, examples)

    with get_session() as session:
        examples = build_training_pairs(session, args.gold)
        for p in args.gold_extra or []:
            examples.extend(load_extra_examples(p, is_gold=True))
        for p in args.extra_examples or []:
            examples.extend(load_extra_examples(p, is_gold=False))
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

    # Gold upweighting : amplifie les paires ground-truth dans le train uniquement.
    # Préserve une val "naturelle" (sans duplication) pour une mesure honnête.
    if args.gold_upweight > 1:
        gold_train = [ex for ex in train if ex.is_gold]
        other_train = [ex for ex in train if not ex.is_gold]
        train = other_train + gold_train * args.gold_upweight
        logger.info(
            f"[CLI] Gold upweight ×{args.gold_upweight} : "
            f"{len(gold_train)} paires gold → {len(gold_train) * args.gold_upweight} "
            f"dans le train (total train = {len(train)})"
        )

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
        question_dropout_gold=args.question_dropout_gold,
    )
    fine_tune(train_examples=train, val_examples=val if val else None, config=cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
