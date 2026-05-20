"""
CLI : évalue la **génération** d'un Task 1 JSON contre les réponses gold du sample.

Deux métriques :
  - **BertScore** (similarité neuronale multilingue) entre answer généré et gold
  - **LLM-as-Judge** (gpt-4o-mini) : notes 1-5 sur factualité / complétude / clarté

BertScore : on utilise le package officiel `bert-score` si disponible. À défaut,
fallback sur la similarité cosinus via notre embedder e5-base (proxy moins rigoureux
mais zero-dep additionnelle).

Usage :
    python eval_generation.py \
      --predictions ../Experimental/DATA/sample_task1.json \
      --gold ../Experimental/DATA/sample_queries.json

    # Skip le LLM judge (zero coût, juste BertScore-like) :
    python eval_generation.py --predictions ... --gold ... --no-judge

    # Limiter pour économiser :
    python eval_generation.py ... --filter-qids Q1,Q4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def _try_bert_score_official(
    candidates: list[str], references: list[str], lang: str = "fr"
) -> tuple[list[float], list[float], list[float]] | None:
    """Tente le BertScore officiel. None si indisponible."""
    try:
        from bert_score import score as bs_score
    except ImportError:
        return None
    try:
        P, R, F = bs_score(candidates, references, lang=lang, verbose=False)
        return P.tolist(), R.tolist(), F.tolist()
    except Exception as ex:
        logger.warning(f"[BertScore] échec ({ex}), fallback embedder")
        return None


def _bert_score_fallback(
    candidates: list[str], references: list[str]
) -> tuple[list[float], list[float], list[float]]:
    """
    Fallback BertScore-like via notre embedder e5-base (cosine similarity globale).
    Moins fin (pas de matching token-level) mais reste cohérent comme proxy.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from pipeline.embedder import Embedder

    emb = Embedder()
    cand_vecs = emb.encode_passages(candidates)
    ref_vecs = emb.encode_passages(references)
    sims = []
    for c, r in zip(cand_vecs, ref_vecs):
        sims.append(sum(a * b for a, b in zip(c, r)))
    # Convention : on retourne P=R=F=similarité (pas de distinction token-level dans le fallback)
    return sims, sims, sims


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval génération (BertScore + LLM-Judge)")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--filter-qids", type=str, default=None)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-Judge (économique : seulement BertScore)",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BertScore (seulement LLM-Judge)",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    args = parser.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))

    preds_by_qid = {r["qid"]: r for r in preds.get("results", [])}
    gold_by_qid = {r["qid"]: r for r in gold.get("results", [])}

    if args.filter_qids:
        keep = {q.strip() for q in args.filter_qids.split(",") if q.strip()}
        preds_by_qid = {q: r for q, r in preds_by_qid.items() if q in keep}
        gold_by_qid = {q: r for q, r in gold_by_qid.items() if q in keep}

    pairs = []  # (qid, question, gold_answer, candidate_answer)
    for qid, g in gold_by_qid.items():
        ga = (g.get("answer") or "").strip()
        if not ga or ga.upper() == "TODO":
            continue
        p = preds_by_qid.get(qid)
        if p is None:
            continue
        ca = (p.get("answer") or "").strip()
        if not ca:
            continue
        pairs.append((qid, g["question"], ga, ca))

    if not pairs:
        logger.error("[EvalGen] Aucune paire évaluable.")
        return 1

    qids = [p[0] for p in pairs]
    candidates = [p[3] for p in pairs]
    references = [p[2] for p in pairs]
    questions = [p[1] for p in pairs]

    per_q: list[dict] = [{"qid": qid} for qid in qids]

    # ───── BertScore
    if not args.no_bertscore:
        logger.info("[EvalGen] BertScore en cours...")
        official = _try_bert_score_official(candidates, references, lang="fr")
        if official is not None:
            P, R, F = official
            logger.info("[EvalGen] BertScore officiel (lang=fr) utilisé")
        else:
            logger.warning("[EvalGen] BertScore officiel indisponible → fallback embedder e5")
            P, R, F = _bert_score_fallback(candidates, references)
        for i, qid in enumerate(qids):
            per_q[i]["bertscore_precision"] = round(P[i], 4)
            per_q[i]["bertscore_recall"] = round(R[i], 4)
            per_q[i]["bertscore_f1"] = round(F[i], 4)

    # ───── LLM-as-Judge
    if not args.no_judge:
        from utils.judge import LLMJudge

        logger.info(f"[EvalGen] LLM-as-Judge ({args.judge_model})...")
        judge = LLMJudge(model=args.judge_model, temperature=0.0)
        for i, (qid, question, gold_a, cand_a) in enumerate(pairs):
            score = judge.score(qid, question, gold_a, cand_a)
            per_q[i]["judge_factuality"] = score.factuality
            per_q[i]["judge_completeness"] = score.completeness
            per_q[i]["judge_clarity"] = score.clarity
            per_q[i]["judge_avg"] = round(score.average, 3)
            per_q[i]["judge_comment"] = score.comment
            logger.info(
                f"  {qid} : F={score.factuality} C={score.completeness} "
                f"Cl={score.clarity} (avg={score.average:.2f})"
            )

    # ───── Agrégats
    def _avg(key: str) -> float:
        vals = [r[key] for r in per_q if key in r]
        return round(sum(vals) / max(len(vals), 1), 4)

    agg: dict[str, float] = {"n_questions": len(per_q)}
    if not args.no_bertscore:
        agg["bertscore_f1_avg"] = _avg("bertscore_f1")
        agg["bertscore_precision_avg"] = _avg("bertscore_precision")
        agg["bertscore_recall_avg"] = _avg("bertscore_recall")
    if not args.no_judge:
        agg["judge_factuality_avg"] = _avg("judge_factuality")
        agg["judge_completeness_avg"] = _avg("judge_completeness")
        agg["judge_clarity_avg"] = _avg("judge_clarity")
        agg["judge_overall_avg"] = _avg("judge_avg")

    logger.info("=" * 60)
    logger.info("[EvalGen] Agrégat :")
    for k, v in agg.items():
        logger.info(f"  {k:30s} = {v}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"per_query": per_q, "aggregate": agg}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[EvalGen] Sortie : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
