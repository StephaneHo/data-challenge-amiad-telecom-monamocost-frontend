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
    python ablation_study.py --output ../Experimental/DATA/ablation.json
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
    # Tâche 1
    hits_at_5: float = 0.0
    hits_at_10: float = 0.0
    # Tâche 2 (proxy F1)
    attribution_precision: float = 0.0
    attribution_recall: float = 0.0
    attribution_f1: float = 0.0
    n_sentences_total: int = 0
    n_sentences_empty: int = 0
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

    hits_5 = 0
    hits_10 = 0
    n_q = 0
    n_correct = 0
    n_attributed_total = 0
    pages_hit: list[int] = []
    n_gold_pages: list[int] = []
    n_sentences_total = 0
    n_sentences_empty = 0
    co2_total = 0.0
    runtime_total = 0.0

    task1_results = {r.qid: r for r in out.task1.results}
    task2_results = {r.qid: r for r in out.task2.results} if out.task2 else {}

    for qid, gold_pages in gold_by_qid.items():
        if not gold_pages:
            continue
        n_q += 1
        t1 = task1_results.get(qid)
        if t1 is None:
            continue

        retrieved_top5 = {(r.doc_name, r.page) for r in t1.retrieved[:5]}
        retrieved_top10 = {(r.doc_name, r.page) for r in t1.retrieved[:10]}
        if gold_pages & retrieved_top5:
            hits_5 += 1
        if gold_pages & retrieved_top10:
            hits_10 += 1

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

        # Empreinte (somme sur les questions)
        carbon = (t1.metadata or {}).get("tokens_used", 0)
        # Note : RAGResponse a un champ carbon mais on l'a perdu dans Task1Result.
        # On fait une estimation grossière depuis tokens_used.
        co2_total += carbon * 0.04 / 1000_000.0 * 1000.0  # g CO2

    precision = n_correct / max(n_attributed_total, 1)
    recall = (sum(pages_hit) / max(sum(n_gold_pages), 1)) if n_gold_pages else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return AblationResult(
        config={},  # rempli par l'appelant
        n_questions=n_q,
        hits_at_5=hits_5 / max(n_q, 1),
        hits_at_10=hits_10 / max(n_q, 1),
        attribution_precision=round(precision, 4),
        attribution_recall=round(recall, 4),
        attribution_f1=round(f1, 4),
        n_sentences_total=n_sentences_total,
        n_sentences_empty=n_sentences_empty,
        co2_g_total=round(co2_total, 4),
        runtime_seconds=round(runtime_total, 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Étude d'ablation du pipeline RAG")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("../Experimental/DATA/sample_queries.json"),
        help="Fichier sample_queries avec gold retrieved",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Experimental/DATA/ablation_results.json"),
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
                f"[Ablation] {cfg.name} : hits@5={res.hits_at_5:.3f} "
                f"hits@10={res.hits_at_10:.3f} "
                f"attr_F1={res.attribution_f1:.3f} "
                f"sent_empty={res.n_sentences_empty}/{res.n_sentences_total}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[Ablation] Résultats : {args.output}")

    # Génère aussi un markdown pour le rapport
    md_path = args.output.with_suffix(".md")
    lines = [
        "# Ablation Study — Mon Amo Cost",
        "",
        "| Config | hits@5 | hits@10 | Attribution F1 | Sent. empty | n_q |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']['name']} | "
            f"{r['hits_at_5']:.3f} | "
            f"{r['hits_at_10']:.3f} | "
            f"{r['attribution_f1']:.3f} | "
            f"{r['n_sentences_empty']}/{r['n_sentences_total']} | "
            f"{r['n_questions']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[Ablation] Tableau markdown : {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
