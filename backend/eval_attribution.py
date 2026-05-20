"""
CLI : évalue un Task 2 JSON (le nôtre) contre un Task 2 gold.

Calcule les **deux blocs** de métriques officiels du challenge EvalLLM 2026 :
- **Précision/Rappel/F1 sur les attributions** (paires doc-page correctement attribuées)
- **Précision/Rappel/F1 sur les segments `[]`** (détection des phrases non sourcées :
  hallucinations, connaissances générales, mise en forme)

Le gold Task 2 sera fourni par les organisateurs à la phase de test. En attendant,
ce script peut être utilisé pour :
  - comparer deux runs entre eux (run A comme « gold », run B à évaluer)
  - calibrer les seuils d'attribution sur un petit gold manuel
  - benchmarker les configs d'ablation

Usage :
    python eval_attribution.py \
      --predictions ../Experimental/DATA/my_task2.json \
      --gold ../Experimental/DATA/gold_task2.json

    # Sur un sous-ensemble :
    python eval_attribution.py --predictions ... --gold ... --filter-qids Q1,Q4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from utils.metrics import compute_attribution_metrics


def _build_aligned_lists(
    pred_attrs: list[dict], gold_attrs: list[dict]
) -> tuple[list[tuple[str, set[tuple[str, int]]]], list[tuple[str, set[tuple[str, int]]]]]:
    """
    Aligne les attributions prédites et gold par `sid`.
    Si un sid est dans le gold mais pas dans la prédiction, on considère que
    le système a omis ce segment → prédiction = `[]`.
    Inversement, sid prédit mais pas gold → ignoré (le gold est référence).
    """
    gold_by_sid = {a["sid"]: a for a in gold_attrs}
    pred_by_sid = {a["sid"]: a for a in pred_attrs}

    aligned_pred: list[tuple[str, set[tuple[str, int]]]] = []
    aligned_gold: list[tuple[str, set[tuple[str, int]]]] = []

    # On itère sur le gold (référence) — c'est lui qui définit l'ensemble des segments
    for sid, gold_a in gold_by_sid.items():
        pred_a = pred_by_sid.get(sid)
        gold_pairs = {(at["doc_name"], at["page"]) for at in gold_a.get("attributed_to", [])}
        pred_pairs = (
            {(at["doc_name"], at["page"]) for at in pred_a.get("attributed_to", [])}
            if pred_a
            else set()
        )
        aligned_pred.append((sid, pred_pairs))
        aligned_gold.append((sid, gold_pairs))

    return aligned_pred, aligned_gold


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval Task 2 (attributions + détection [])")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--filter-qids", type=str, default=None, help="Ex: 'Q1,Q4'"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))

    pred_by_qid = {r["qid"]: r for r in preds.get("results", [])}
    gold_by_qid = {r["qid"]: r for r in gold.get("results", [])}

    if args.filter_qids:
        keep = {q.strip() for q in args.filter_qids.split(",") if q.strip()}
        pred_by_qid = {q: r for q, r in pred_by_qid.items() if q in keep}
        gold_by_qid = {q: r for q, r in gold_by_qid.items() if q in keep}

    per_query: list[dict] = []
    # Pour l'agrégat global, on concatène toutes les phrases de toutes les questions
    all_pred: list[tuple[str, set]] = []
    all_gold: list[tuple[str, set]] = []

    for qid, gold_entry in gold_by_qid.items():
        pred_entry = pred_by_qid.get(qid)
        if pred_entry is None:
            logger.warning(f"[Eval2] {qid} : aucune prédiction, ignoré")
            continue
        pred_attrs = pred_entry.get("attributions", [])
        gold_attrs = gold_entry.get("attributions", [])
        if not gold_attrs:
            continue

        a_pred, a_gold = _build_aligned_lists(pred_attrs, gold_attrs)
        m = compute_attribution_metrics(a_pred, a_gold)
        per_query.append({"qid": qid, **m.to_dict()})

        # On préfixe le sid par le qid pour éviter les collisions dans l'agrégat global
        all_pred.extend([(f"{qid}:{sid}", pairs) for sid, pairs in a_pred])
        all_gold.extend([(f"{qid}:{sid}", pairs) for sid, pairs in a_gold])

        logger.info(
            f"[Eval2] {qid:4s} | "
            f"attr F1={m.attribution_f1:.3f} (P={m.attribution_precision:.3f}/R={m.attribution_recall:.3f}) "
            f"| [] F1={m.empty_f1:.3f} (P={m.empty_precision:.3f}/R={m.empty_recall:.3f}) "
            f"| n_phr={m.n_sentences}"
        )

    if not per_query:
        logger.error("[Eval2] Aucune question évaluable.")
        return 1

    # Agrégat = micro-moyenne sur toutes les phrases (recommandé en classification)
    micro = compute_attribution_metrics(all_pred, all_gold)
    logger.info("=" * 70)
    logger.info(f"[Eval2] Agrégat micro ({micro.n_sentences} phrases, {len(per_query)} questions) :")
    logger.info(
        f"  Attribution : P={micro.attribution_precision:.4f} "
        f"R={micro.attribution_recall:.4f} F1={micro.attribution_f1:.4f}"
    )
    logger.info(
        f"  [] détection : P={micro.empty_precision:.4f} "
        f"R={micro.empty_recall:.4f} F1={micro.empty_f1:.4f} "
        f"(pred={micro.n_predicted_empty}, gold={micro.n_gold_empty}, both={micro.n_both_empty})"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"per_query": per_query, "micro_aggregate": micro.to_dict()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"[Eval2] Sortie : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
