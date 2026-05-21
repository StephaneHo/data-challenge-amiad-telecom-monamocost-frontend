"""
Étude d'ablation : mesure l'impact individuel de chaque composant du pipeline.

Compare plusieurs configurations sur les questions gold du sample_queries.json
et produit un tableau markdown + JSON exploitable dans le rapport.

Métriques :
- **hits@K** sur le retrieval (Tâche 1) : proportion de questions dont AU MOINS UN
  chunk gold se retrouve dans les top-K retournés (granularité doc/page)
- **F1 attribution** (Tâche 2) : pour les phrases attribuées, fraction qui pointe vers
  une page gold (proxy de précision) × rappel des pages gold couvertes
- **Empreinte carbone** par config (CO2 en g)

Configurations comparées (toggles cumulables) :
- `baseline` : retrieval dense pur, attribution top-1, pas de filtre entité
- `+entity_filter` : ajoute le filtre entités pour les `[]`
- `+topk_adaptive` : Top-K adaptatif (1-3 attributions par phrase)
- `+decompose` : décomposition multi-hop des questions
- `+all` : tout activé

Usage :
    python ablation_study.py
    python ablation_study.py --filter-qids Q1,Q4  # restreindre pour économiser
    python ablation_study.py --output ../Experimental/DATA/runs/ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

from api.challenge import ChallengeInput, ChallengeRunner
from database.db import get_session
from utils.metrics import aggregate_metrics, compute_query_metrics


_DOC_ALIAS: dict[str, str] = {
    "r20-7111.pdf": "2021_rapport_senat_r20-7111.pdf",
}


def _resolve_doc(name: str) -> str:
    return _DOC_ALIAS.get(name, name)


@dataclass
class AblationConfig:
    name: str
    attribution_topk: int = 1
    attribution_threshold: float = 0.80
    attribution_secondary_threshold: float = 0.85
    attribution_entity_filter: bool = False
    decompose: bool = False


@dataclass
class AblationResult:
    config: dict
    n_questions: int
    # Tâche 1 — métriques RI officielles
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    map_score: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    # Tâche 2 (proxy F1 sur attributions ; gold phrase-level pas dispo en dev)
    attribution_precision: float = 0.0
    attribution_recall: float = 0.0
    attribution_f1: float = 0.0
    n_sentences_total: int = 0
    n_sentences_empty: int = 0
    # Breakdown des [] par cause (utile sans gold pour comparer les configs)
    n_empty_markdown: int = 0
    n_empty_entity_mismatch: int = 0
    n_empty_embedding: int = 0
    empty_rate: float = 0.0  # n_sentences_empty / n_sentences_total
    # Empreinte
    co2_g_total: float = 0.0
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _evaluate_run(runner: ChallengeRunner, payload: ChallengeInput) -> AblationResult:
    """Lance un run complet (Task 1 + Task 2) et calcule les métriques agrégées."""
    out = runner.run(payload, with_task2=True)

    # Mapping qid → gold pages (résolues via alias)
    gold_by_qid: dict[str, set[tuple[str, int]]] = {}
    for q in payload.results:
        # On récupère le retrieved gold depuis le payload d'origine
        # (qui contient le gold dans le sample_queries)
        gold = getattr(q, "retrieved", None) or []
        if hasattr(q, "model_extra") and q.model_extra:
            gold = q.model_extra.get("retrieved", gold)
        # Hack : on relit le fichier original pour le gold
    # Plus simple : on relit le gold depuis le payload original.
    # Pour cette fonction, on s'attend à ce que `payload.results[i]` ait les gold
    # via un dict additionnel passé en attribut sur le runner. On va simplifier :
    # on demande aux appelants de fournir les golds séparément.
    raise NotImplementedError("Use _evaluate_run_with_gold instead")


def _evaluate_run_with_gold(
    runner: ChallengeRunner,
    payload: ChallengeInput,
    gold_by_qid: dict[str, set[tuple[str, int]]],
) -> AblationResult:
    """Variante explicite : prend les gold pages séparément."""
    out = runner.run(payload, with_task2=True)

    per_query_metrics = []
    n_correct = 0
    n_attributed_total = 0
    pages_hit: list[int] = []
    n_gold_pages: list[int] = []
    n_sentences_total = 0
    n_sentences_empty = 0
    # Breakdown des [] par cause (depuis les sources internes de l'Attributor)
    n_empty_markdown = 0
    n_empty_entity = 0
    n_empty_embedding = 0
    co2_total = 0.0

    task1_results = {r.qid: r for r in out.task1.results}
    task2_results = {r.qid: r for r in out.task2.results} if out.task2 else {}

    for qid, gold_pages in gold_by_qid.items():
        if not gold_pages:
            continue
        t1 = task1_results.get(qid)
        if t1 is None:
            continue

        # Métriques RI officielles
        retrieved = [(r.doc_name, r.page) for r in t1.retrieved]
        qm = compute_query_metrics(qid, retrieved, gold_pages)
        per_query_metrics.append(qm)

        # Tâche 2
        t2 = task2_results.get(qid)
        if t2:
            covered: set[tuple[str, int]] = set()
            for a in t2.attributions:
                n_sentences_total += 1
                if not a.attributed_to:
                    n_sentences_empty += 1
                    continue
                for at in a.attributed_to:
                    n_attributed_total += 1
                    key = (at.doc_name, at.page)
                    if key in gold_pages:
                        n_correct += 1
                        covered.add(key)
            pages_hit.append(len(covered))
            n_gold_pages.append(len(gold_pages))

            # Breakdown des [] par cause (depuis le debug interne du runner)
            for sent in runner._last_debug_attributions.get(qid, []):
                if sent.attributed_to:
                    continue
                src = sent.sources[0] if sent.sources else "unknown"
                if src == "markdown":
                    n_empty_markdown += 1
                elif src == "entity_mismatch":
                    n_empty_entity += 1
                elif src in ("embedding", "unsourced"):
                    n_empty_embedding += 1

        # Empreinte (estimation grossière depuis tokens_used)
        tokens = (t1.metadata or {}).get("tokens_used", 0)
        co2_total += tokens * 0.04 / 1000_000.0 * 1000.0  # g CO2

    agg = aggregate_metrics(per_query_metrics) if per_query_metrics else {}
    precision = n_correct / max(n_attributed_total, 1)
    recall = (sum(pages_hit) / max(sum(n_gold_pages), 1)) if n_gold_pages else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return AblationResult(
        config={},  # rempli par l'appelant
        n_questions=len(per_query_metrics),
        precision_at_5=agg.get("precision_at_5", 0.0),
        precision_at_10=agg.get("precision_at_10", 0.0),
        recall_at_5=agg.get("recall_at_5", 0.0),
        recall_at_10=agg.get("recall_at_10", 0.0),
        map_score=agg.get("MAP", 0.0),
        mrr=agg.get("MRR", 0.0),
        ndcg_at_5=agg.get("ndcg_at_5", 0.0),
        ndcg_at_10=agg.get("ndcg_at_10", 0.0),
        attribution_precision=round(precision, 4),
        attribution_recall=round(recall, 4),
        attribution_f1=round(f1, 4),
        n_sentences_total=n_sentences_total,
        n_sentences_empty=n_sentences_empty,
        n_empty_markdown=n_empty_markdown,
        n_empty_entity_mismatch=n_empty_entity,
        n_empty_embedding=n_empty_embedding,
        empty_rate=round(n_sentences_empty / max(n_sentences_total, 1), 4),
        co2_g_total=round(co2_total, 4),
        runtime_seconds=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Étude d'ablation du pipeline RAG")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("../Experimental/DATA/training/sample_queries.json"),
        help="Fichier sample_queries avec gold retrieved",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Experimental/DATA/runs/ablation_results.json"),
    )
    parser.add_argument(
        "--filter-qids", type=str, default=None, help="Ex: 'Q1,Q4' pour limiter le coût"
    )
    args = parser.parse_args()

    raw = json.loads(args.gold.read_text(encoding="utf-8"))
    questions = raw["results"]
    if args.filter_qids:
        keep = {q.strip() for q in args.filter_qids.split(",") if q.strip()}
        questions = [q for q in questions if q["qid"] in keep]
    if not questions:
        logger.error("[Ablation] Aucune question.")
        return 1

    # Gold pages par qid
    gold_by_qid: dict[str, set[tuple[str, int]]] = {}
    for q in questions:
        pages = {
            (_resolve_doc(r["doc_name"]), r["page"])
            for r in q.get("retrieved", [])
            if r.get("doc_name") and r["doc_name"] != "TODO"
        }
        if pages:
            gold_by_qid[q["qid"]] = pages

    # Payload commun (sans le gold)
    payload = ChallengeInput.model_validate(
        {
            "run_id": raw.get("run_id", "ablation"),
            "parameters": {},
            "results": [{"qid": q["qid"], "question": q["question"]} for q in questions],
        }
    )

    # Grille de configurations à comparer
    configs = [
        AblationConfig(
            name="A0_baseline",
            attribution_topk=1,
            attribution_entity_filter=False,
            decompose=False,
        ),
        AblationConfig(
            name="A1_entity_filter",
            attribution_topk=1,
            attribution_entity_filter=True,
            decompose=False,
        ),
        AblationConfig(
            name="A2_topk_adaptive",
            attribution_topk=3,
            attribution_entity_filter=False,
            decompose=False,
        ),
        AblationConfig(
            name="A3_decompose",
            attribution_topk=1,
            attribution_entity_filter=False,
            decompose=True,
        ),
        AblationConfig(
            name="A4_all",
            attribution_topk=3,
            attribution_entity_filter=True,
            decompose=True,
        ),
    ]

    results: list[dict] = []
    with get_session() as session:
        for cfg in configs:
            logger.info(f"[Ablation] === {cfg.name} ===")
            runner = ChallengeRunner(
                session=session,
                attribution_threshold=cfg.attribution_threshold,
                attribution_secondary_threshold=cfg.attribution_secondary_threshold,
                attribution_topk=cfg.attribution_topk,
                attribution_entity_filter=cfg.attribution_entity_filter,
                decompose=cfg.decompose,
            )
            res = _evaluate_run_with_gold(runner, payload, gold_by_qid)
            res.config = asdict(cfg)
            results.append(res.to_dict())
            logger.info(
                f"[Ablation] {cfg.name} : "
                f"NDCG@10={res.ndcg_at_10:.3f} MAP={res.map_score:.3f} MRR={res.mrr:.3f} "
                f"| attr_F1={res.attribution_f1:.3f} "
                f"empty_rate={res.empty_rate:.2%} "
                f"(md={res.n_empty_markdown} ent={res.n_empty_entity_mismatch} emb={res.n_empty_embedding})"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[Ablation] Résultats : {args.output}")

    # Génère aussi un markdown pour le rapport
    md_path = args.output.with_suffix(".md")
    lines = [
        "# Ablation Study — Mon Amo Cost",
        "",
        "## Métriques retrieval (Tâche 1)",
        "",
        "| Config | NDCG@10 | NDCG@5 | MAP | MRR | P@5 | R@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']['name']} | "
            f"{r['ndcg_at_10']:.3f} | "
            f"{r['ndcg_at_5']:.3f} | "
            f"{r['map_score']:.3f} | "
            f"{r['mrr']:.3f} | "
            f"{r['precision_at_5']:.3f} | "
            f"{r['recall_at_5']:.3f} |"
        )

    lines += [
        "",
        "## Métriques attribution + breakdown des `[]` (Tâche 2)",
        "",
        "| Config | Attr F1 | `[]` rate | md | entity | emb | n phrases |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']['name']} | "
            f"{r['attribution_f1']:.3f} | "
            f"{r['empty_rate']:.2%} | "
            f"{r['n_empty_markdown']} | "
            f"{r['n_empty_entity_mismatch']} | "
            f"{r['n_empty_embedding']} | "
            f"{r['n_sentences_total']} |"
        )
    lines += [
        "",
        "_Lecture_ : `md` = phrases vidées par regex markdown ; `entity` = par filtre "
        "d'entités saillantes ; `emb` = par seuil embedding. Plus la somme est élevée, "
        "plus le système est conservateur sur les `[]`. Les P/R/F1 sur les `[]` ne sont "
        "calculables qu'avec un gold phrase-level (voir `eval_attribution.py` quand le "
        "gold du challenge sera fourni).",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[Ablation] Tableau markdown : {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
