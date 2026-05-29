"""
Couche de décodage contraint (structured generation) pour la Tâche 1 du challenge.

Motivation : forcer le LLM à produire une sortie dont la forme est garantie —
- soit via les Structured Outputs natifs d'OpenAI (json_schema strict),
- soit via le JSON mode de Mistral,
- soit via Outlines (https://huggingface.co/learn/cookbook/fr/structured_generation)
  pour un LLM open-source local servi par vLLM / llama-cpp-python / Ollama.

Pattern « best-effort, transparent caller » :
    answer = generate_structured_answer(
        client, model, system, user, provider="openai", backend="auto"
    )
    # answer.answer  : texte de la réponse (markdown)
    # answer.citations : list[ {doc_name, page} ] structurées
    # answer.tokens_used / answer.backend_used : traçabilité

Si l'on ne peut pas activer de décodage contraint (provider inconnu, Outlines
non installé pour un LLM local, etc.), on retombe sur un appel libre + extraction
JSON tolérante. Le caller voit toujours un `StructuredAnswer` valide.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from config import settings


# ───────────────────────── Schéma Pydantic ──────────────────────────────────

class StructuredCitation(BaseModel):
    """
    Une citation structurée : (nom de document PDF, numéro de page physique).
    Les deux champs sont obligatoires — c'est ce qui rend la sortie utilisable
    sans regex.
    """

    doc_name: str = Field(
        ...,
        min_length=1,
        description="Nom exact du document source, ex: 'mon_document.pdf'",
    )
    page: int = Field(
        ...,
        ge=1,
        description="Numéro de page physique (1-based, la couverture est p.1)",
    )


class StructuredAnswer(BaseModel):
    """
    Réponse structurée du LLM pour la Tâche 1.

    - `answer` : le texte de la réponse (français, markdown autorisé).
    - `citations` : la liste DÉDOUBLONNÉE des (doc, page) utilisés dans la réponse.

    Convention : `answer` peut contenir des marqueurs inline `[doc.pdf p.N]`
    comme avant ; on les régénère depuis `citations` côté caller si nécessaire
    pour rester compatible avec l'attribution Tâche 2.
    """

    answer: str = Field(
        ...,
        min_length=1,
        description="Texte de la réponse en français, factuel et synthétique.",
    )
    citations: list[StructuredCitation] = Field(
        default_factory=list,
        description="Citations utilisées dans la réponse (doc_name + page).",
    )


# ───────────────────────── Résultat enrichi ─────────────────────────────────

BackendName = Literal[
    "openai_schema",     # Structured Outputs OpenAI (json_schema strict)
    "mistral_json",      # Mistral JSON mode (json_object)
    "anthropic_tool",    # Anthropic tool use (input_schema)
    "outlines_local",    # Outlines côté client (vLLM/HF local)
    "vllm_guided",       # vLLM serveur avec extra_body.guided_json
    "fallback_repair",   # extraction JSON post-hoc
]


@dataclass
class StructuredResult:
    """Sortie unifiée du décodage contraint, indépendante du backend choisi."""

    answer: StructuredAnswer
    tokens_used: int
    backend_used: BackendName
    raw_text: Optional[str] = None  # texte brut renvoyé par le LLM (debug)


# ───────────────────────── Schéma JSON dérivé ───────────────────────────────

def get_json_schema() -> dict[str, Any]:
    """JSON Schema dérivé du modèle Pydantic, prêt pour les Structured Outputs."""
    schema = StructuredAnswer.model_json_schema()
    # OpenAI Structured Outputs exige `additionalProperties: false` partout
    # et tous les champs en `required`. On normalise le schéma.
    return _strictify_schema(schema)


def _strictify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Rend récursivement un JSON Schema compatible avec OpenAI Structured Outputs :
    - `additionalProperties: false` sur tous les `object`,
    - `required` listant TOUS les champs `properties`.
    Voir https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
    """
    if not isinstance(schema, dict):
        return schema

    out = dict(schema)
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
        out["properties"] = {
            k: _strictify_schema(v) for k, v in out["properties"].items()
        }
    if out.get("type") == "array" and "items" in out:
        out["items"] = _strictify_schema(out["items"])
    # Résout les $defs Pydantic en inlinant (OpenAI accepte $ref/$defs mais
    # plus simple de tout inliner ici, le schéma reste petit).
    if "$defs" in out:
        defs = {k: _strictify_schema(v) for k, v in out["$defs"].items()}
        out = _inline_refs(out, defs)
        out.pop("$defs", None)
    return out


def _inline_refs(node: Any, defs: dict[str, dict[str, Any]]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and node["$ref"].startswith("#/$defs/"):
            key = node["$ref"].split("/")[-1]
            return _inline_refs(defs.get(key, {}), defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


# ───────────────────────── Backends individuels ─────────────────────────────

_USER_SUFFIX_JSON = (
    "\n\nIMPORTANT : ta réponse doit être un OBJET JSON valide conforme au "
    'schéma suivant : {"answer": "<texte>", "citations": [{"doc_name": "...", "page": N}, ...]}\n'
    "- Le champ `answer` contient la réponse en français (markdown autorisé).\n"
    "- Tu DOIS aussi inclure dans `answer` les citations inline `[doc.pdf p.N]` "
    "comme avant — `citations` est une liste structurée parallèle.\n"
    "- Ne mets rien avant ni après l'objet JSON (pas de markdown fence)."
)


def _call_openai_schema(
    client: Any, model: str, system: str, user: str, temperature: float, max_tokens: int
) -> StructuredResult:
    """Décodage contraint via Structured Outputs OpenAI (json_schema strict)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_answer",
                "schema": get_json_schema(),
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + _USER_SUFFIX_JSON},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    tokens = int(resp.usage.total_tokens) if resp.usage else 0
    parsed = StructuredAnswer.model_validate_json(raw)
    return StructuredResult(
        answer=parsed, tokens_used=tokens, backend_used="openai_schema", raw_text=raw
    )


def _call_mistral_json(
    client: Any, model: str, system: str, user: str, temperature: float, max_tokens: int
) -> StructuredResult:
    """Mistral JSON mode (json_object, schéma rappelé dans le prompt)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + _USER_SUFFIX_JSON},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    tokens = int(resp.usage.total_tokens) if resp.usage else 0
    parsed = _parse_with_repair(raw)
    return StructuredResult(
        answer=parsed, tokens_used=tokens, backend_used="mistral_json", raw_text=raw
    )


def _call_anthropic_tool(
    client: Any, model: str, system: str, user: str, temperature: float, max_tokens: int
) -> StructuredResult:
    """Anthropic : tool use avec `input_schema` (équivalent fonctionnel à json_schema)."""
    schema = get_json_schema()
    tool_name = "submit_structured_answer"
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        tools=[{
            "name": tool_name,
            "description": "Soumettre la réponse structurée à la question.",
            "input_schema": schema,
        }],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    tokens = int(resp.usage.input_tokens + resp.usage.output_tokens)
    # On cherche le bloc tool_use
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            parsed = StructuredAnswer.model_validate(block.input)
            raw = json.dumps(block.input, ensure_ascii=False)
            return StructuredResult(
                answer=parsed,
                tokens_used=tokens,
                backend_used="anthropic_tool",
                raw_text=raw,
            )
    raise RuntimeError("[StructuredDecoder] Anthropic n'a pas renvoyé de tool_use.")


def _call_vllm_guided(
    client: Any, model: str, system: str, user: str, temperature: float, max_tokens: int
) -> StructuredResult:
    """
    vLLM serveur OpenAI-compat avec `extra_body.guided_json`. Le LLM local
    contraint sa génération côté serveur via Outlines/xgrammar — on récupère
    directement un JSON valide.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + _USER_SUFFIX_JSON},
        ],
        extra_body={"guided_json": get_json_schema()},
    )
    raw = resp.choices[0].message.content or "{}"
    tokens = int(resp.usage.total_tokens) if resp.usage else 0
    parsed = _parse_with_repair(raw)
    return StructuredResult(
        answer=parsed, tokens_used=tokens, backend_used="vllm_guided", raw_text=raw
    )


def _call_outlines_local(
    model_name: str, system: str, user: str, max_tokens: int
) -> StructuredResult:
    """
    Décodage contraint via Outlines côté client.

    Utilise `outlines.from_transformers(...)` pour un modèle HuggingFace local,
    avec génération bornée par le schéma Pydantic. Plus lent à charger
    (le modèle est instancié in-process) mais 100% reproductible et offline.

    Cf. https://huggingface.co/learn/cookbook/fr/structured_generation
    """
    try:
        import outlines  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as ex:
        raise RuntimeError(
            f"[StructuredDecoder] Backend outlines_local indisponible : {ex}. "
            "Installer : `uv sync --group constrained`."
        )

    # Import paresseux pour ne pas plomber le démarrage si outlines absent
    from outlines import models, generate  # type: ignore

    logger.info(f"[StructuredDecoder] Chargement Outlines local : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = models.Transformers(base_model, tokenizer)
    generator = generate.json(model, StructuredAnswer)

    # Concatène system + user en un seul prompt (chat template à la main)
    prompt = f"<|system|>\n{system}\n<|user|>\n{user}{_USER_SUFFIX_JSON}\n<|assistant|>\n"
    result: StructuredAnswer = generator(prompt, max_tokens=max_tokens)
    # tokens_used inconnu côté Outlines local : 0 (le LLM est local, pas facturé)
    return StructuredResult(
        answer=result,
        tokens_used=0,
        backend_used="outlines_local",
        raw_text=result.model_dump_json(),
    )


def _call_fallback_repair(
    client: Any, model: str, system: str, user: str, temperature: float, max_tokens: int
) -> StructuredResult:
    """Dernier recours : appel libre + extraction JSON tolérante côté Python."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + _USER_SUFFIX_JSON},
        ],
    )
    raw = resp.choices[0].message.content or ""
    tokens = int(resp.usage.total_tokens) if resp.usage else 0
    parsed = _parse_with_repair(raw)
    return StructuredResult(
        answer=parsed, tokens_used=tokens, backend_used="fallback_repair", raw_text=raw
    )


# ───────────────────────── Parsing tolérant ────────────────────────────────

_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _parse_with_repair(raw: str) -> StructuredAnswer:
    """
    Essaie 3 stratégies dans l'ordre :
      1. parse JSON direct (le LLM a respecté le format),
      2. retire les fences markdown et retente,
      3. extraction du premier objet `{...}` équilibré dans le texte.
    Si tout échoue, fabrique un StructuredAnswer minimal avec le texte brut
    comme `answer` (on perd les citations structurées mais on ne casse pas
    le pipeline).
    """
    text = raw.strip()

    for attempt in (text, _CODE_FENCE.sub("", text).strip()):
        try:
            return StructuredAnswer.model_validate_json(attempt)
        except (ValidationError, ValueError):
            pass
        try:
            obj = json.loads(attempt)
            return StructuredAnswer.model_validate(obj)
        except (json.JSONDecodeError, ValidationError):
            pass

    # 3) Premier {...} équilibré
    extracted = _extract_first_json_object(text)
    if extracted:
        try:
            return StructuredAnswer.model_validate(extracted)
        except ValidationError as ex:
            logger.warning(f"[StructuredDecoder] JSON extrait invalide : {ex}")

    logger.warning("[StructuredDecoder] Impossible de parser le JSON, fallback texte brut.")
    return StructuredAnswer(answer=text or "(réponse vide)", citations=[])


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Trouve le premier objet `{...}` équilibré, ignore les autres."""
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
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


# ───────────────────────── Dispatcher principal ─────────────────────────────

def generate_structured_answer(
    client: Any,
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    backend: str = "auto",
) -> StructuredResult:
    """
    Point d'entrée unique du décodage contraint.

    Args:
        client : client LLM déjà instancié (OpenAI / Mistral SDK / Anthropic).
        provider : "openai" | "mistral" | "anthropic".
        backend  : "auto" (choisit selon provider+config) | "openai_schema"
                   | "mistral_json" | "anthropic_tool" | "vllm_guided"
                   | "outlines_local" | "fallback_repair".

    Returns:
        StructuredResult avec `answer` (Pydantic validé), `tokens_used`,
        `backend_used` (le backend effectivement utilisé après auto-résolution).
    """
    provider = (provider or "").lower()
    backend = (backend or "auto").lower()

    if backend == "auto":
        backend = _resolve_auto_backend(provider)

    logger.info(f"[StructuredDecoder] backend={backend} provider={provider} model={model}")

    try:
        if backend == "openai_schema":
            return _call_openai_schema(client, model, system, user, temperature, max_tokens)
        if backend == "mistral_json":
            return _call_mistral_json(client, model, system, user, temperature, max_tokens)
        if backend == "anthropic_tool":
            return _call_anthropic_tool(client, model, system, user, temperature, max_tokens)
        if backend == "vllm_guided":
            return _call_vllm_guided(client, model, system, user, temperature, max_tokens)
        if backend == "outlines_local":
            return _call_outlines_local(model, system, user, max_tokens)
        if backend == "fallback_repair":
            return _call_fallback_repair(client, model, system, user, temperature, max_tokens)
        raise ValueError(f"Backend inconnu : {backend}")
    except Exception as ex:
        # Le décodage contraint a échoué (ex: modèle ne supporte pas json_schema,
        # serveur local sans guided_json, schéma rejeté) → on retombe sur le
        # fallback texte libre + repair plutôt que de casser le pipeline.
        logger.warning(
            f"[StructuredDecoder] backend={backend} a échoué ({ex}). "
            "Fallback → texte libre + parsing tolérant."
        )
        return _call_fallback_repair(client, model, system, user, temperature, max_tokens)


def _resolve_auto_backend(provider: str) -> BackendName:
    """
    Heuristique de choix du backend selon provider + config :
    - openai (cloud)  → openai_schema (Structured Outputs)
    - openai + LLM_BASE_URL → vllm_guided (serveur local OpenAI-compat)
    - mistral         → mistral_json
    - anthropic       → anthropic_tool
    """
    if provider == "anthropic":
        return "anthropic_tool"
    if provider == "mistral":
        return "mistral_json"
    if provider == "openai":
        if settings.LLM_BASE_URL:
            # Cloud-compatible local server (vLLM, llama-cpp.python avec --json-mode...)
            # On tente d'abord guided_json (vLLM), fallback automatique si ça échoue.
            return "vllm_guided"
        return "openai_schema"
    logger.warning(f"[StructuredDecoder] Provider inconnu '{provider}', fallback_repair.")
    return "fallback_repair"


# ───────────────────────── Helpers post-traitement ──────────────────────────

_CITATION_INLINE = re.compile(
    r"\[\s*(?P<doc>[^\]\[]+?\.pdf)\s*[,;\s]\s*p\.?\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)


def ensure_inline_citations(structured: StructuredAnswer) -> str:
    """
    Garantit que le texte `answer` contient les citations inline `[doc.pdf p.N]`
    pour rester compatible avec l'attribution Tâche 2. Si le LLM a déjà inséré
    les citations dans le texte (cas le plus fréquent), on ne touche à rien.
    Sinon, on annexe en fin de texte un récap `Sources : [a.pdf p.1] [b.pdf p.5]`.
    """
    inline_found = _CITATION_INLINE.findall(structured.answer or "")
    if inline_found:
        # Au moins une citation inline : on suppose que le LLM les a bien posées
        return structured.answer

    if not structured.citations:
        return structured.answer

    suffix_parts = [f"[{c.doc_name} p.{c.page}]" for c in structured.citations]
    return f"{structured.answer.rstrip()}\n\nSources : {' '.join(suffix_parts)}"


def validate_citations_against_pool(
    structured: StructuredAnswer,
    declared_pool: set[tuple[str, int]],
) -> tuple[list[StructuredCitation], list[StructuredCitation]]:
    """
    Sépare les citations structurées en (valides, hors-pool).

    `declared_pool` est l'ensemble des (doc_name, page) effectivement retournés
    par le retrieval — toute citation hors de ce pool est suspecte
    (souvent : hallucination ou erreur de typo du LLM).
    """
    valid: list[StructuredCitation] = []
    invalid: list[StructuredCitation] = []
    for c in structured.citations:
        if (c.doc_name, c.page) in declared_pool:
            valid.append(c)
        else:
            invalid.append(c)
    return valid, invalid
