"""
CLI : grid search sur les seuils d'attribution pour maximiser un F1 proxy.

Méthode :
- Pour chaque question d'un fichier sample_queries-like, on connaît les pages gold
  (champ `retrieved`).
- On lance le pipeline RAG (retrieval + génération + attribution) pour différentes
  combinaisons (threshold, secondary_threshold).
- On calcule un F1 proxy :
    * positif = phrase attribuée à une (doc, page) qui est dans le gold de la question
    * négatif = phrase non attribuée (attributed_to=[])
    * Précision = #attributions correctes / #attributions totales
    * Rappel = #pages gold couvertes par au moins une attribution / #pages gold

C'est un proxy car le challenge réel évalue au niveau segment, pas page agrégée.
Mais ça permet de comparer des hyperparams entre eux à coût modéré.

Usage :
    python calibrate_attribution.py --gold ../Experimental/DATA/training/sample_queries.json
    python calibrate_attribution.py --thresholds 0.75 0.80 0.85 --secondary 0.85 0.90 0.95
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from database.db import get_session
from rag.rag_engine import RAGEngine


_DOC_ALIAS: dict[str, str] = {
    "r20-7111.pdf": "2021_rapport_senat_r20-7111.pdf",
}


def _resolve_doc(name: str) -> str:
    return _DOC_ALIAS.get(name, name)


@dataclass
class CalibScore:
    threshold: float
    secondary_threshold: float
    topk: int
    entity_filter: bool
    n_questions: int
    n_sentences_total: int
    n_sentences_attributed: int
    n_sentences_empty: int
    precision_proxy: float
    recall_proxy: float
    f1_proxy: float

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "secondary_threshold": self.secondary_threshold,
            "topk": self.topk,
            "entity_filter": self.entity_filter,
            "n_questions": self.n_questions,
            "n_sentences_total": self.n_sentences_total,
            "n_sentences_attributed": self.n_sentences_attributed,
            "n_sentences_empty": self.n_sentences_empty,
            "precision_proxy": round(self.precision_proxy, 4),
            "recall_proxy": round(self.recall_proxy, 4),
            "f1_proxy": round(self.f1_proxy, 4),
        }


def _score_run(
    engine: RAGEngine,
    questions: list[dict],
    threshold: float,
    secondary_threshold: float,
    topk: int,
    entity_filter: bool,
) -> CalibScore:
    n_correct = 0  # attributions vers une page gold
    n_attributed_total = 0  # attributions non-vides
    pages_covered_per_q: list[int] = []  # par question : combien de pages gold ont été touchées
    n_gold_pages_per_q: list[int] = []
    n_sentences_total = 0
    n_sentences_attributed = 0
    n_sentences_empty = 0

    for q in questions:
        qid = q["qid"]
        question = q["question"]
        gold_pages = {(_resolve_doc(r["doc_name"]), r["page"]) for r in q.get("retrieved", [])}
        if not gold_pages or "TODO" in {d for d, _ in gold_pages}:
            continue

        resp = engine.answer(
            query=question,
            top_k_chunks=20,
            top_n_pages=10,
            context_chunks=10,
        )
        if not resp.answer or not resp.chunks:
            continue
        sentences = engine.attribute(
            qid=qid,
            answer=resp.answer,
            chunks=resp.chunks,
            threshold=threshold,
            secondary_threshold=secondary_threshold,
            topk_per_sentence=topk,
            entity_filter=entity_filter,
        )
        gold_pages_hit: set[tuple[str, int]] = set()
        for s in sentences:
            n_sentences_total += 1
            if not s.attributed_to:
                n_sentences_empty += 1
                continue
            n_sentences_attributed += 1
            for a in s.attributed_to:
                n_attributed_total += 1
                key = (a.doc_name, a.page)
                if key in gold_pages:
                    n_correct += 1
                    gold_pages_hit.add(key)

        pages_covered_per_q.append(len(gold_pages_hit))
        n_gold_pages_per_q.append(len(gold_pages))

    precision = n_correct / max(n_attributed_total, 1)
    if not n_gold_pages_per_q:
        recall = 0.0
    else:
        recall = sum(pages_covered_per_q) / max(sum(n_gold_pages_per_q), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return CalibScore(
        threshold=threshold,
        secondary_threshold=secondary_threshold,
        topk=topk,
        entity_filter=entity_filter,
        n_questions=len(n_gold_pages_per_q),
        n_sentences_total=n_sentences_total,
        n_sentences_attributed=n_sentences_attributed,
        n_sentences_empty=n_sentences_empty,
        precision_proxy=precision,
        recall_proxy=recall,
        f1_proxy=f1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search seuils Attribution")
    parser.add_argument("--gold", type=Path, default=Path("../Experimental/DATA/training/sample_queries.json"))
    parser.add_argument("--output", type=Path, default=Path("../Experimental/DATA/runs/attribution_calibration.json"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.75, 0.78, 0.80, 0.82, 0.85])
    parser.add_argument("--secondary", type=float, nargs="+", default=[0.82, 0.85, 0.88])
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--entity-filter", type=int, nargs="+", default=[0, 1], help="0 = off, 1 = on")
    parser.add_argument(
        "--filter-qids",
        type=str,
        default=None,
        help="qids séparés par virgule (ex: 'Q1,Q4') pour limiter le coût LLM",
    )
    args = parser.parse_args()

    data = json.loads(args.gold.read_text(encoding="utf-8"))
    questions = data["results"]
    if args.filter_qids:
        keep = {q.strip() for q in args.filter_qids.split(",") if q.strip()}
        questions = [q for q in questions if q["qid"] in keep]
        logger.info(f"[Calib] Filtre qids={sorted(keep)} → {len(questions)} questions")

    if not questions:
        logger.error("[Calib] Aucune question à évaluer.")
        return 1

    # Grille
    combos = list(
        itertools.product(args.thresholds, args.secondary, args.topk, args.entity_filter)
    )
    # On ne garde que les combos où secondary >= threshold
    combos = [(t, s, k, e) for (t, s, k, e) in combos if s >= t]
    logger.info(f"[Calib] {len(combos)} combinaisons × {len(questions)} questions")

    scores: list[CalibScore] = []
    with get_session() as session:
        engine = RAGEngine(session=session)
        for i, (t, s, k, e) in enumerate(combos, start=1):
            logger.info(
                f"[Calib] {i}/{len(combos)} : threshold={t} secondary={s} topk={k} entity_filter={bool(e)}"
            )
            score = _score_run(engine, questions, t, s, k, bool(e))
            scores.append(score)
            logger.info(
                f"  → P={score.precision_proxy:.3f} R={score.recall_proxy:.3f} "
                f"F1={score.f1_proxy:.3f} | sent_total={score.n_sentences_total} "
                f"sent_empty={score.n_sentences_empty}"
            )

    # Tri par F1 décroissant
    scores.sort(key=lambda s: s.f1_proxy, reverse=True)
    out = [s.to_dict() for s in scores]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[Calib] Résultats triés écrits dans {args.output}")
    best = scores[0]
    logger.info(
        f"[Calib] BEST : threshold={best.threshold} secondary={best.secondary_threshold} "
        f"topk={best.topk} entity={best.entity_filter} → F1 proxy={best.f1_proxy:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
