"""
CLI : évalue un Task 1 JSON (le nôtre) contre un fichier gold (sample_queries
ou format équivalent), avec les métriques officielles du challenge.

Usage :
    # Compare nos sorties contre le gold du sample :
    python eval_retrieval.py \
      --predictions ../Experimental/DATA/sample_task1.json \
      --gold ../Experimental/DATA/sample_queries.json

    # Filtrer une seule question :
    python eval_retrieval.py --predictions ... --gold ... --filter-qids Q1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from utils.metrics import aggregate_metrics, compute_query_metrics


_DOC_ALIAS: dict[str, str] = {
    "r20-7111.pdf": "2021_rapport_senat_r20-7111.pdf",
}


def _resolve_doc(name: str) -> str:
    return _DOC_ALIAS.get(name, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval retrieval (P, R, MAP, MRR, NDCG)")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSON au format Task 1 output (champ `retrieved` avec rank+doc_name+page)",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="JSON gold (sample_queries.json) avec le même format `retrieved` annoté",
    )
    parser.add_argument(
        "--filter-qids",
        type=str,
        default=None,
        help="Liste qids séparés par virgule (ex: 'Q1,Q4')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON de sortie avec métriques par question + agrégées",
    )
    args = parser.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))

    # Indexe par qid
    preds_by_qid = {r["qid"]: r for r in preds.get("results", [])}
    gold_by_qid = {r["qid"]: r for r in gold.get("results", [])}

    if args.filter_qids:
        keep = {q.strip() for q in args.filter_qids.split(",") if q.strip()}
        preds_by_qid = {q: r for q, r in preds_by_qid.items() if q in keep}
        gold_by_qid = {q: r for q, r in gold_by_qid.items() if q in keep}

    per_query = []
    for qid, gold_entry in gold_by_qid.items():
        gold_pairs = {
            (_resolve_doc(r["doc_name"]), r["page"])
            for r in gold_entry.get("retrieved", [])
            if r.get("doc_name") and r["doc_name"] != "TODO" and r.get("page", 0) > 0
        }
        if not gold_pairs:
            continue
        pred_entry = preds_by_qid.get(qid)
        if pred_entry is None:
            logger.warning(f"[Eval] {qid} : pas de prédiction, ignoré")
            continue
        retrieved = [
            (_resolve_doc(r["doc_name"]), r["page"])
            for r in pred_entry.get("retrieved", [])
        ]
        qm = compute_query_metrics(qid, retrieved, gold_pairs)
        per_query.append(qm)
        logger.info(
            f"[Eval] {qid:4s} | P@5={qm.precision_at_5:.3f} R@5={qm.recall_at_5:.3f} "
            f"NDCG@10={qm.ndcg_at_10:.3f} MRR={qm.rr:.3f} AP={qm.ap:.3f}"
        )

    if not per_query:
        logger.error("[Eval] Aucune question évaluée.")
        return 1

    agg = aggregate_metrics(per_query)
    logger.info("=" * 60)
    logger.info(f"[Eval] Agrégat ({len(per_query)} questions) :")
    for k, v in agg.items():
        logger.info(f"  {k:12s} = {v:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "per_query": [qm.to_dict() for qm in per_query],
                    "aggregate": agg,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"[Eval] Sortie : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
