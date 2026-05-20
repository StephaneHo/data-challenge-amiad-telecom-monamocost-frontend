"""
Helper centralisé pour instancier un client LLM compatible OpenAI.

Bascule transparente entre :
- l'API OpenAI standard (`LLM_BASE_URL` vide, `OPENAI_API_KEY` fourni),
- un serveur local OpenAI-compatible (`LLM_BASE_URL` défini) :
  - **vLLM**       : `python -m vllm.entrypoints.openai.api_server --model <hf_id>`
  - **llama-cpp**  : `python -m llama_cpp.server --model <gguf_path>`
  - **Ollama**     : `ollama serve` (endpoint `http://localhost:11434/v1`)

Ce bypass permet de remplacer `gpt-4o-mini` par Mistral-7B-Instruct ou Qwen2.5-7B
pour le bonus frugalité / open-source du challenge EvalLLM 2026, sans toucher
au code métier (les modules conservent leur usage `client.chat.completions.create`).
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import settings


def get_openai_client() -> Any:
    """
    Retourne un client OpenAI Python SDK configuré selon `.env`.

    - `LLM_BASE_URL=""` → `api.openai.com` officiel.
    - `LLM_BASE_URL="http://localhost:8000/v1"` → serveur local.

    Si `LLM_API_KEY` est vide, on utilise `OPENAI_API_KEY` ou la valeur de courtoisie
    `"sk-no-key"` (acceptée par la plupart des serveurs locaux).
    """
    from openai import OpenAI

    api_key = settings.LLM_API_KEY or settings.OPENAI_API_KEY or "sk-no-key"
    kwargs: dict[str, Any] = {"api_key": api_key}
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    return OpenAI(**kwargs)


def chat_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Appel chat complétion attendant un JSON en retour.

    Stratégie :
    - Si `settings.LLM_JSON_VIA_PROMPT` est False (défaut OpenAI), on utilise
      `response_format={"type": "json_object"}`.
    - Sinon (serveur local qui ne supporte pas ce paramètre), on demande le JSON
      via le prompt et on extrait le premier objet `{...}` du texte renvoyé.

    Renvoie un dict. Sur erreur de parsing, retourne `{}` (les callers font
    déjà des `.get(key, default)` tolérants).
    """
    if settings.LLM_JSON_VIA_PROMPT:
        # Prompt strict : on ajoute une instruction explicite
        user_aug = (
            f"{user}\n\n"
            "IMPORTANT : ta réponse doit être EXCLUSIVEMENT un objet JSON valide, "
            "sans texte avant ni après, sans markdown ni triple backtick."
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_aug},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return _extract_first_json_object(text)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return _extract_first_json_object(resp.choices[0].message.content or "")


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Extrait le premier objet `{...}` valide d'un texte (fallback pour modèles
    qui n'honorent pas `response_format`).
    """
    # 1) Tentative directe (si tout le texte est du JSON)
    text = text.strip()
    if text.startswith("```"):
        # Retire fences markdown
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Recherche du premier { ... } équilibré (parser manuel, simple)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                fragment = text[start : i + 1]
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    start = -1
    return {}
