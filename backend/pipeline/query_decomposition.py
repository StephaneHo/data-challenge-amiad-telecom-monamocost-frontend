"""
Décomposition de questions multi-hop en sous-questions plus simples.

Motivation : sur les requêtes synthétiques type Q3 du sample
(« Comment les mesures anti-drones en France, les capacités d'Altius et Switchblade,
et la stratégie chinoise illustrent-elles ensemble... »), un retrieval dense pur sur
la question entière a un signal trop diffus → on rate les chunks pertinents.

Approche :
  1. Demander au LLM de décomposer la question en sous-questions (ou de la laisser
     telle quelle si déjà atomique)
  2. Faire un retrieval séparé pour chaque sous-question
  3. Fusionner les chunks (dédup par chunk_id) en gardant le meilleur score par chunk
  4. Garder le top-K du pool fusionné

Conforme à la spec Tâche 2 :
  - La décomposition utilise le LLM, mais cherche UNIQUEMENT dans le corpus indexé.
  - Aucune source externe ajoutée, aucune capacité de recherche web.
  - Les chunks retournés restent des chunks du corpus → attribution Task 2 cohérente.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from config import settings


_SYSTEM_PROMPT = (
    "Tu es un expert en analyse de questions complexes. Tu décomposes une question "
    "multi-hop en sous-questions atomiques qui, prises ensemble, couvrent toutes les "
    "informations nécessaires pour répondre à la question d'origine."
)

_USER_TEMPLATE = """Question d'origine :
\"\"\"
{question}
\"\"\"

Décompose cette question en sous-questions atomiques (1 à 4 sous-questions max).

Règles strictes :
- Si la question est DÉJÀ atomique (porte sur UN seul aspect/entité/événement),
  retourne UN SEUL élément qui est la question d'origine telle quelle.
- Sinon, identifie les axes distincts (entités, événements, comparaisons) et
  produis une sous-question PAR axe.
- Chaque sous-question doit être autoporteuse (pas de référence à « cette question »
  ni à des éléments des autres sous-questions).
- Préserve les noms propres / acronymes EXACTS (MQ-9 Reaper, OSINT, Tracfin...).
- Réponds en français.

Retourne UNIQUEMENT un JSON :
{{"subqueries": ["<sous-question 1>", "<sous-question 2>", ...]}}
"""


@dataclass
class DecompositionResult:
    """Résultat de décomposition avec traçabilité."""

    original: str
    subqueries: list[str]
    model: str

    @property
    def is_atomic(self) -> bool:
        return len(self.subqueries) <= 1


class QueryDecomposer:
    """Décompose une question complexe via LLM ; cache en mémoire pour éviter les rappels."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_subqueries: int = 4,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_subqueries = max_subqueries
        self._client: Any = None
        self._cache: dict[str, DecompositionResult] = {}

    def _llm(self) -> Any:
        if self._client is None:
            from utils.llm import get_openai_client

            self._client = get_openai_client()
        return self._client

    def decompose(self, question: str) -> DecompositionResult:
        from utils.llm import chat_json

        if question in self._cache:
            return self._cache[question]

        try:
            payload = chat_json(
                self._llm(),
                model=self.model,
                system=_SYSTEM_PROMPT,
                user=_USER_TEMPLATE.format(question=question),
                temperature=self.temperature,
            )
            subs = payload.get("subqueries", [])
            if isinstance(subs, str):
                subs = [subs]
            cleaned: list[str] = []
            for s in subs[: self.max_subqueries]:
                s = s.strip() if isinstance(s, str) else ""
                if s and len(s) > 10:
                    cleaned.append(s)
            if not cleaned:
                cleaned = [question]
        except Exception as ex:
            logger.warning(f"[Decomposer] LLM échec sur '{question[:60]}...' : {ex}")
            cleaned = [question]

        result = DecompositionResult(original=question, subqueries=cleaned, model=self.model)
        self._cache[question] = result
        if not result.is_atomic:
            logger.info(
                f"[Decomposer] '{question[:80]}...' → {len(cleaned)} sous-questions"
            )
        return result
