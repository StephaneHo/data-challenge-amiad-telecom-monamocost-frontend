"""
Métriques classiques de Recherche d'Information.

Les métriques officielles annoncées par le challenge EvalLLM 2026 sont
Précision, Rappel, NDCG. On ajoute aussi MAP et MRR (standards en RI).

Les fonctions opèrent sur :
- `gold` : set des couples (doc_name, page) considérés comme pertinents
- `retrieved` : liste ordonnée de (doc_name, page) prédits, du plus pertinent au moins

Toutes les métriques sont définies pour une seule requête ; les versions
agrégées (macro-moyenne sur les requêtes) sont produites par
`aggregate_metrics()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Pair = tuple[str, int]  # (doc_name, page)


def precision_at_k(retrieved: list[Pair], gold: set[Pair], k: int) -> float:
    """Fraction des k premiers résultats qui sont dans le gold."""
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return sum(1 for p in topk if p in gold) / len(topk)


def recall_at_k(retrieved: list[Pair], gold: set[Pair], k: int) -> float:
    """Fraction des éléments gold retrouvés dans les k premiers."""
    if not gold:
        return 0.0
    topk = set(retrieved[:k])
    return len(gold & topk) / len(gold)


def average_precision(retrieved: list[Pair], gold: set[Pair]) -> float:
    """
    AP = sum over ranks où un gold apparaît, de (precision @ ce rank)
        divisé par |gold|.
    """
    if not gold:
        return 0.0
    n_hits = 0
    score = 0.0
    for i, p in enumerate(retrieved, start=1):
        if p in gold:
            n_hits += 1
            score += n_hits / i
    return score / len(gold)


def reciprocal_rank(retrieved: list[Pair], gold: set[Pair]) -> float:
    """1 / rank du premier gold trouvé ; 0 si aucun."""
    for i, p in enumerate(retrieved, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: list[Pair], gold: set[Pair], k: int) -> float:
    """DCG binaire : rel_i ∈ {0,1}, gain = 1/log2(i+1) avec i indexé à 1."""
    score = 0.0
    for i, p in enumerate(retrieved[:k], start=1):
        if p in gold:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: list[Pair], gold: set[Pair], k: int) -> float:
    """NDCG@K binaire = DCG@K / IDCG@K (où IDCG = DCG du ranking parfait)."""
    if not gold:
        return 0.0
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg_at_k(retrieved, gold, k) / idcg if idcg > 0 else 0.0


# ──────────────────────── Agrégation ───────────────────────────────────────

@dataclass
class QueryMetrics:
    qid: str
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    ap: float
    rr: float
    ndcg_at_5: float
    ndcg_at_10: float

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "P@5": round(self.precision_at_5, 4),
            "P@10": round(self.precision_at_10, 4),
            "R@5": round(self.recall_at_5, 4),
            "R@10": round(self.recall_at_10, 4),
            "AP": round(self.ap, 4),
            "RR": round(self.rr, 4),
            "NDCG@5": round(self.ndcg_at_5, 4),
            "NDCG@10": round(self.ndcg_at_10, 4),
        }


def compute_query_metrics(qid: str, retrieved: list[Pair], gold: set[Pair]) -> QueryMetrics:
    return QueryMetrics(
        qid=qid,
        precision_at_5=precision_at_k(retrieved, gold, 5),
        precision_at_10=precision_at_k(retrieved, gold, 10),
        recall_at_5=recall_at_k(retrieved, gold, 5),
        recall_at_10=recall_at_k(retrieved, gold, 10),
        ap=average_precision(retrieved, gold),
        rr=reciprocal_rank(retrieved, gold),
        ndcg_at_5=ndcg_at_k(retrieved, gold, 5),
        ndcg_at_10=ndcg_at_k(retrieved, gold, 10),
    )


def aggregate_metrics(per_query: list[QueryMetrics]) -> dict[str, float]:
    """Macro-moyenne sur les requêtes (chaque requête a un poids égal)."""
    if not per_query:
        return {}
    keys = ["precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10",
            "ap", "rr", "ndcg_at_5", "ndcg_at_10"]
    out: dict[str, float] = {}
    for k in keys:
        vals = [getattr(qm, k) for qm in per_query]
        out[k] = round(sum(vals) / len(vals), 4)
    # MAP = mean AP, MRR = mean RR (renommage standard)
    out["MAP"] = out.pop("ap")
    out["MRR"] = out.pop("rr")
    return out
