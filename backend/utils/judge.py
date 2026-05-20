"""
LLM-as-Judge : note la qualité d'une réponse RAG vs une réponse gold.

Le challenge annonce explicitement utiliser des métriques LLM-as-Judge.
On reproduit en interne pour s'auto-évaluer pendant le développement.

Trois critères, échelle 1-5 :
  - factualité : les faits cités sont-ils corrects vs gold ?
  - complétude : la réponse couvre-t-elle les aspects clés du gold ?
  - clarté : structure, lisibilité, citation correcte des sources ?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from config import settings


_SYSTEM_PROMPT = (
    "Tu es un expert en évaluation de réponses RAG (Retrieval-Augmented Generation) "
    "pour le domaine défense/renseignement en français. Tu compares une réponse candidate "
    "à une réponse de référence et tu notes objectivement sur 3 critères."
)

_USER_TEMPLATE = """Question :
{question}

Réponse de référence (gold) :
\"\"\"
{gold_answer}
\"\"\"

Réponse candidate à évaluer :
\"\"\"
{candidate_answer}
\"\"\"

Note la réponse candidate de 1 (très mauvais) à 5 (excellent) sur ces critères :

- **factuality** : les faits, chiffres, noms cités dans la candidate sont-ils corrects
  par rapport à la gold ? (1 = beaucoup d'erreurs ou inventions ; 5 = aligné sur la gold)
- **completeness** : la candidate couvre-t-elle les aspects clés de la gold ?
  (1 = informations majeures manquantes ; 5 = couvre tout l'essentiel)
- **clarity** : la candidate est-elle bien structurée et facile à lire ? Les citations
  `[doc.pdf p.N]` sont-elles correctement insérées après les affirmations factuelles ?
  (1 = brouillon ; 5 = clair et bien sourcé)

Tu peux aussi commenter brièvement (max 30 mots) ce qui justifie tes notes.

Retourne UNIQUEMENT ce JSON :
{{"factuality": <int 1-5>, "completeness": <int 1-5>, "clarity": <int 1-5>, "comment": "<str>"}}
"""


@dataclass
class JudgeScore:
    qid: str
    factuality: int
    completeness: int
    clarity: int
    comment: str
    error: Optional[str] = None

    @property
    def average(self) -> float:
        return (self.factuality + self.completeness + self.clarity) / 3.0

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "factuality": self.factuality,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "average": round(self.average, 3),
            "comment": self.comment,
            "error": self.error,
        }


class LLMJudge:
    """Wrapper minimaliste OpenAI ChatCompletions pour le scoring."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature  # 0 = déterministe = reproductible
        self._client: Any = None

    def _llm(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def score(self, qid: str, question: str, gold_answer: str, candidate_answer: str) -> JudgeScore:
        prompt = _USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            candidate_answer=candidate_answer,
        )
        try:
            resp = self._llm().chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            payload = json.loads(resp.choices[0].message.content or "{}")

            def _clamp(v: Any) -> int:
                try:
                    return max(1, min(5, int(v)))
                except (TypeError, ValueError):
                    return 1

            return JudgeScore(
                qid=qid,
                factuality=_clamp(payload.get("factuality")),
                completeness=_clamp(payload.get("completeness")),
                clarity=_clamp(payload.get("clarity")),
                comment=str(payload.get("comment", ""))[:300],
            )
        except Exception as ex:
            logger.warning(f"[Judge] {qid} : erreur LLM ({ex})")
            return JudgeScore(qid=qid, factuality=1, completeness=1, clarity=1, comment="", error=str(ex))
