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


# ──────────────── Tâche 2 — métriques d'attribution ──────────────────────

@dataclass
class AttributionMetrics:
    """Métriques pour la Tâche 2 du challenge EvalLLM 2026.

    Deux blocs de métriques **distincts** comme demandé par le règlement :

    - **Bloc attribution** (segments sourcés) : Précision, Rappel, F1 sur les
      paires `(doc_name, page)` attribuées vs gold (par phrase).
    - **Bloc `[]`** (segments non sourcés) : Précision, Rappel, F1 sur la classe
      « non sourcé » — mesure la fiabilité de la détection d'hallucinations,
      de connaissances générales et de mise en forme.
    """

    n_sentences: int
    # Bloc attribution
    attribution_precision: float
    attribution_recall: float
    attribution_f1: float
    # Bloc [] (détection des phrases non sourcées)
    empty_precision: float
    empty_recall: float
    empty_f1: float
    n_predicted_empty: int
    n_gold_empty: int
    n_both_empty: int

    def to_dict(self) -> dict:
        return {
            "n_sentences": self.n_sentences,
            "attribution": {
                "precision": round(self.attribution_precision, 4),
                "recall": round(self.attribution_recall, 4),
                "f1": round(self.attribution_f1, 4),
            },
            "empty_detection": {
                "precision": round(self.empty_precision, 4),
                "recall": round(self.empty_recall, 4),
                "f1": round(self.empty_f1, 4),
                "n_predicted_empty": self.n_predicted_empty,
                "n_gold_empty": self.n_gold_empty,
                "n_both_empty": self.n_both_empty,
            },
        }


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_attribution_metrics(
    predicted: list[tuple[str, set[Pair]]],
    gold: list[tuple[str, set[Pair]]],
) -> AttributionMetrics:
    """
    Calcule les deux blocs de métriques de la Tâche 2.

    Args:
        predicted : liste de (sid, set des (doc, page) attribués) ; set vide = `[]`
        gold      : même structure pour les annotations de référence

    Les listes doivent être **alignées par sid** : le i-ème élément de
    `predicted` correspond au i-ème de `gold`. Les sid présents d'un seul
    côté lèvent une ValueError.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"Tailles incohérentes : predicted={len(predicted)} vs gold={len(gold)}"
        )

    # Vérifie l'alignement par sid
    for (sid_p, _), (sid_g, _) in zip(predicted, gold):
        if sid_p != sid_g:
            raise ValueError(f"Sid désynchronisé : predicted={sid_p} gold={sid_g}")

    # ── Bloc attribution : micro-moyenne sur l'ensemble des paires (doc, page)
    # prédites vs gold. Chaque sid contribue ses paires.
    tp_attr = 0  # paires prédites qui sont dans le gold
    fp_attr = 0  # paires prédites qui ne sont pas dans le gold
    fn_attr = 0  # paires gold qui ne sont pas prédites

    # ── Bloc [] : classification binaire par segment
    n_predicted_empty = 0
    n_gold_empty = 0
    n_both_empty = 0

    for (_, pred_set), (_, gold_set) in zip(predicted, gold):
        tp_attr += len(pred_set & gold_set)
        fp_attr += len(pred_set - gold_set)
        fn_attr += len(gold_set - pred_set)

        is_pred_empty = len(pred_set) == 0
        is_gold_empty = len(gold_set) == 0
        if is_pred_empty:
            n_predicted_empty += 1
        if is_gold_empty:
            n_gold_empty += 1
        if is_pred_empty and is_gold_empty:
            n_both_empty += 1

    p_attr = tp_attr / (tp_attr + fp_attr) if (tp_attr + fp_attr) else 0.0
    r_attr = tp_attr / (tp_attr + fn_attr) if (tp_attr + fn_attr) else 0.0

    p_empty = n_both_empty / n_predicted_empty if n_predicted_empty else 0.0
    r_empty = n_both_empty / n_gold_empty if n_gold_empty else 0.0

    return AttributionMetrics(
        n_sentences=len(predicted),
        attribution_precision=p_attr,
        attribution_recall=r_attr,
        attribution_f1=_f1(p_attr, r_attr),
        empty_precision=p_empty,
        empty_recall=r_empty,
        empty_f1=_f1(p_empty, r_empty),
        n_predicted_empty=n_predicted_empty,
        n_gold_empty=n_gold_empty,
        n_both_empty=n_both_empty,
    )
