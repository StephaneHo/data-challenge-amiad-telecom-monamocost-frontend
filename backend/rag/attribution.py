from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from pipeline.embedder import Embedder


# ───────────────────────────── Modèles ────────────────────────────────────

@dataclass(frozen=True)
class Attribution:
    """Référence d'attribution : un (doc_name, page) du corpus."""

    doc_name: str
    page: int

    def to_dict(self) -> dict:
        return {"doc_name": self.doc_name, "page": self.page}


@dataclass
class AttributedSentence:
    """Une phrase segmentée et attribuée à ses sources (potentiellement vide)."""

    sid: str
    text: str
    attributed_to: list[Attribution] = field(default_factory=list)
    # Champs internes (pas exportés au JSON challenge) :
    sources: tuple[str, ...] = ()  # "citation" | "embedding" | "markdown" | "unsourced"
    similarity: Optional[float] = None

    def to_dict(self) -> dict:
        """Sérialise au format challenge `attributed_to`."""
        return {
            "sid": self.sid,
            "text": self.text,
            "attributed_to": [a.to_dict() for a in self.attributed_to],
        }


# ────────────────────────── Segmentation phrase ───────────────────────────

# Abréviations FR à NE PAS découper (point interne masqué pendant la segmentation)
_FR_ABBREV = (
    "M.", "Mme.", "Mlle.", "Dr.", "Pr.", "cf.", "p.", "pp.", "fig.", "ex.",
    "etc.", "art.", "al.", "n°", "vol.", "chap.", "tome.", "ed.", "av.",
    "J.-C.", "c.-à-d.", "i.e.", "e.g.",
)
# Sépare phrases : ponctuation forte suivie d'un espace + majuscule (FR accentuées)
# Pas de lookbehind variable car Python `re` ne le supporte pas — on masque
# les points internes aux abréviations en amont (cf. `split_sentences`).
_SENT_END = re.compile(
    r"(?<=[\.!?])\s+(?=[A-ZÀÁÂÄÆÇÉÈÊËÎÏÔÖŒÙÛÜŸ«•\-#*0-9])",
)
# Placeholder pour les points d'abréviation (caractère qui n'apparaît pas en FR)
_DOT_PLACEHOLDER = "\x00"
# Détecte une ligne purement formatage Markdown (titre, séparateur, gras isolé)
_MARKDOWN_LINE = re.compile(
    r"""
    ^\s*(?:
        \#{1,6}\s.*       |   # # Titre
        ---+              |   # ---
        \*{3,}            |   # ***
        \[\d*\]           |   # tag style [0]
        \*\*[^\*]+\*\*$       # **gras seul sur sa ligne**
    )\s*$
    """,
    re.VERBOSE,
)


def _mask_abbrev(text: str) -> str:
    """Remplace les points d'abréviation par un placeholder pour éviter le split."""
    for abbr in _FR_ABBREV:
        text = text.replace(abbr, abbr.replace(".", _DOT_PLACEHOLDER))
    return text


def _unmask_abbrev(text: str) -> str:
    return text.replace(_DOT_PLACEHOLDER, ".")


# ───────────────── Extraction d'entités saillantes (sans spaCy) ────────────
# Détecte les hallucinations en vérifiant qu'au moins une entité saillante de
# la phrase apparaît dans les chunks retrieved.

# Acronymes (USV, MQ-9, OSINT, NDCG) : 2+ chars MAJ + chiffres + tirets
_RE_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*\b")
# Nombres avec espace ou unité (10 millions, 30 %, 2024)
_RE_NUMBER = re.compile(r"\b\d[\d\s.,]*\b")
# Noms propres : capitalisés ≥ 4 chars (Beehive, Reaper, Tracfin)
_RE_CAPS_WORD = re.compile(r"\b[A-ZÀÁÂÄÆÇÉÈÊËÎÏÔÖŒÙÛÜŸ][\wàáâäæçéèêëîïôöœùûüÿ\-]{3,}\b")

# Mots/lieux trop fréquents pour être discriminants
_NER_STOPWORDS = {
    "Royal", "Navy", "Cette", "Dans", "Ainsi", "Donc", "Mais", "Puis", "Alors",
    "Cela", "Ceci", "Cet", "Notamment", "Toutefois", "Cependant", "Aussi",
    "Ensuite", "Enfin", "Lorsque", "Bien", "Plus", "Tout", "Tous", "Toutes",
    "Pour", "Avec", "Sans", "Selon", "Depuis", "Pendant", "Avant", "Après",
    "Comment", "Quel", "Quels", "Quelle", "Quelles", "Quand", "Qui", "Quoi",
    "France", "Europe", "Etat", "Etats", "Union", "Nations", "Tracfin",
}


def _extract_salient_entities(text: str) -> set[str]:
    """Entités saillantes de la phrase (acronymes + nombres + noms propres), lowercase."""
    out: set[str] = set()
    for m in _RE_ACRONYM.finditer(text):
        tok = m.group()
        if len(tok) >= 2 and tok not in _NER_STOPWORDS:
            out.add(tok.lower())
    for m in _RE_NUMBER.finditer(text):
        tok = m.group().strip().replace(" ", "")
        if len(tok.replace(",", "").replace(".", "")) >= 2:
            out.add(tok.lower())
    for m in _RE_CAPS_WORD.finditer(text):
        tok = m.group()
        if tok in _NER_STOPWORDS:
            continue
        out.add(tok.lower())
    return out


def _entities_supported(
    sentence_entities: set[str], chunks_text: str, min_support_ratio: float = 0.5
) -> bool:
    """
    Vérifie qu'au moins `min_support_ratio` des entités de la phrase apparaissent
    dans le texte concaténé des chunks. Si pas d'entités à vérifier → True.
    """
    if not sentence_entities:
        return True
    haystack = chunks_text.lower()
    n_supported = sum(1 for e in sentence_entities if e in haystack)
    return (n_supported / len(sentence_entities)) >= min_support_ratio


def split_sentences(text: str) -> list[str]:
    """
    Segmente un texte (potentiellement multi-paragraphes) en phrases.

    Traite indépendamment chaque ligne pour préserver les éléments markdown
    (titres, séparateurs) comme « phrases » à part — ils seront attribués à [].
    """
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _MARKDOWN_LINE.match(line):
            out.append(line)
            continue
        # Masque les points d'abréviation, split, puis restaure
        masked = _mask_abbrev(line)
        parts = _SENT_END.split(masked)
        for part in parts:
            part = _unmask_abbrev(part).strip()
            if part:
                out.append(part)
    return out


# ───────────────────────── Parsing des citations ──────────────────────────

# Variantes acceptées :
#   [doc.pdf p.17]   [doc.pdf, p.17]   [doc.pdf p17]   [doc.pdf]
_CITATION_WITH_PAGE = re.compile(
    r"""
    \[                              # ouverture
    \s*
    (?P<doc>[^\]\[]+?\.pdf)         # nom de doc (.pdf obligatoire)
    \s*[,;\s]\s*
    p\.?\s*(?P<page>\d+)            # p.N ou pN
    \s*
    \]
    """,
    re.VERBOSE | re.IGNORECASE,
)
_CITATION_DOC_ONLY = re.compile(
    r"""
    \[                              # ouverture
    \s*
    (?P<doc>[^\]\[\s,;]+?\.pdf)     # juste un nom de doc, sans page
    \s*
    \]
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class RawCitation:
    """Citation parsée brute — la page peut être absente (à résoudre par similarité)."""

    doc_name: str
    page: Optional[int] = None


def parse_citations(text: str) -> list[RawCitation]:
    """
    Extrait toutes les citations d'un texte (dédoublonnées).
    Accepte deux formes :
      - `[doc.pdf p.N]` → `RawCitation(doc, page=N)`
      - `[doc.pdf]`     → `RawCitation(doc, page=None)` (page à résoudre)
    """
    seen: set[tuple[str, Optional[int]]] = set()
    out: list[RawCitation] = []

    # On enlève d'abord les matches avec page pour ne pas re-matcher en doc-only
    matched_spans: list[tuple[int, int]] = []
    for m in _CITATION_WITH_PAGE.finditer(text):
        doc = m.group("doc").strip()
        page = int(m.group("page"))
        key = (doc, page)
        matched_spans.append(m.span())
        if key not in seen:
            seen.add(key)
            out.append(RawCitation(doc_name=doc, page=page))

    # Puis les citations doc-only, en évitant les spans déjà capturés
    def _overlaps(start: int, end: int) -> bool:
        return any(s <= start < e or s < end <= e for s, e in matched_spans)

    for m in _CITATION_DOC_ONLY.finditer(text):
        if _overlaps(*m.span()):
            continue
        doc = m.group("doc").strip()
        key = (doc, None)
        if key not in seen:
            seen.add(key)
            out.append(RawCitation(doc_name=doc, page=None))
    return out


# ───────────────────────── Moteur d'attribution ───────────────────────────

@dataclass
class RetrievedChunkForAttribution:
    """Chunk minimaliste utilisé par l'Attributor — découplé de RAGEngine."""

    doc_name: str
    page_number: int
    content: str


class Attributor:
    """
    Attribue chaque phrase d'une réponse à un ensemble de (doc_name, page).

    Pipeline par phrase :
      1. Si la phrase est un élément markdown pur (titre, séparateur) → []
      2. Sinon, parser les citations `[doc.pdf p.N]` :
         - si présentes → attribution directe
         - sinon → calcul de similarité cosinus avec les chunks retrieved :
            - max_sim ≥ threshold → attribution à la (doc, page) du chunk
            - sinon → [] (non sourcé)
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        threshold: float = 0.80,
        secondary_threshold: float = 0.85,
        topk_per_sentence: int = 3,
        entity_filter: bool = True,
        entity_min_support_ratio: float = 0.5,
    ) -> None:
        """
        Paramètres :
        - threshold : sim minimum pour attribuer une phrase au top-1 chunk.
        - secondary_threshold : sim minimum pour ajouter un 2e/3e attribution.
          Doit être >= threshold (plus strict = moins de faux positifs).
        - topk_per_sentence : max d'attributions par phrase (dédup par (doc, page)).
        - entity_filter : si True, vérifie que les entités saillantes de la phrase
          apparaissent dans les chunks retrieved ; sinon force `[]`.
          Boost le rappel sur les `[]` (détection d'hallucinations type "Recon-ng").
        - entity_min_support_ratio : fraction minimale d'entités phrase qui doivent
          être présentes dans les chunks pour valider (1.0 = strict, 0.0 = off).
        """
        self.embedder = embedder or Embedder()
        self.threshold = threshold
        self.secondary_threshold = max(secondary_threshold, threshold)
        self.topk_per_sentence = topk_per_sentence
        self.entity_filter = entity_filter
        self.entity_min_support_ratio = entity_min_support_ratio

    def _is_markdown_line(self, text: str) -> bool:
        return bool(_MARKDOWN_LINE.match(text))

    def _resolve_citation_page(
        self,
        sentence: str,
        raw: RawCitation,
        chunks: list[RetrievedChunkForAttribution],
    ) -> Optional[Attribution]:
        """
        Pour une citation `[doc.pdf]` sans page, choisit la page du même doc
        la plus similaire à la phrase. Retourne None si aucun chunk du doc.
        """
        candidates = [c for c in chunks if c.doc_name == raw.doc_name]
        if not candidates:
            return None  # citation orpheline : doc absent du contexte retrieved
        if raw.page is not None:
            return Attribution(doc_name=raw.doc_name, page=raw.page)
        q_vec = self.embedder.encode_query(sentence)
        c_vecs = self.embedder.encode_passages([c.content for c in candidates])
        sims = [sum(a * b for a, b in zip(q_vec, cv)) for cv in c_vecs]
        best = max(zip(candidates, sims), key=lambda x: x[1])[0]
        return Attribution(doc_name=best.doc_name, page=best.page_number)

    def _attribute_by_embedding(
        self, sentence: str, chunks: list[RetrievedChunkForAttribution]
    ) -> tuple[list[Attribution], Optional[float]]:
        if not chunks:
            return [], None
        # Embed phrase (query:) vs chunks (passage:)
        q_vec = self.embedder.encode_query(sentence)
        c_vecs = self.embedder.encode_passages([c.content for c in chunks])
        sims = [sum(a * b for a, b in zip(q_vec, cv)) for cv in c_vecs]
        ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
        if not ranked:
            return [], None
        best_chunk, best_sim = ranked[0]

        # 1) Top-1 doit dépasser le seuil principal — sinon `[]`
        if best_sim < self.threshold:
            return [], best_sim

        # 2) Top-1 toujours retenu si > threshold
        seen: set[tuple[str, int]] = set()
        out: list[Attribution] = []
        key = (best_chunk.doc_name, best_chunk.page_number)
        seen.add(key)
        out.append(Attribution(doc_name=best_chunk.doc_name, page=best_chunk.page_number))

        # 3) Top-2, top-3... uniquement si sim ≥ secondary_threshold (plus strict)
        for chunk, sim in ranked[1 : self.topk_per_sentence]:
            if sim < self.secondary_threshold:
                break
            key = (chunk.doc_name, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)
            out.append(Attribution(doc_name=chunk.doc_name, page=chunk.page_number))

        return out, best_sim

    def attribute(
        self,
        qid: str,
        answer: str,
        chunks: list[RetrievedChunkForAttribution],
    ) -> list[AttributedSentence]:
        sentences = split_sentences(answer)
        # Texte concaténé des chunks pour le filtre entité (calculé une seule fois)
        chunks_text = " ".join(c.content for c in chunks) if self.entity_filter else ""
        out: list[AttributedSentence] = []
        for i, sent in enumerate(sentences):
            sid = f"{qid}_s{i}"
            # 1. Markdown pur → []
            if self._is_markdown_line(sent):
                out.append(
                    AttributedSentence(sid=sid, text=sent, sources=("markdown",))
                )
                continue
            # 1bis. Filtre entité : si les entités saillantes de la phrase
            # n'apparaissent pas dans les chunks → hallucination probable → []
            if self.entity_filter:
                sent_entities = _extract_salient_entities(sent)
                if sent_entities and not _entities_supported(
                    sent_entities, chunks_text, self.entity_min_support_ratio
                ):
                    out.append(
                        AttributedSentence(
                            sid=sid, text=sent, sources=("entity_mismatch",)
                        )
                    )
                    continue
            # 2. Citation explicite — résout les pages absentes via similarité
            raw_cits = parse_citations(sent)
            if raw_cits:
                resolved: list[Attribution] = []
                seen_keys: set[tuple[str, int]] = set()
                for raw in raw_cits:
                    attr = self._resolve_citation_page(sent, raw, chunks)
                    if attr is None:
                        continue
                    key = (attr.doc_name, attr.page)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    resolved.append(attr)
                if resolved:
                    out.append(
                        AttributedSentence(
                            sid=sid,
                            text=sent,
                            attributed_to=resolved,
                            sources=("citation",),
                        )
                    )
                    continue
                # Toutes les citations étaient orphelines (doc absent) → fallback embedding
            # 3. Fallback embedding
            attrs, sim = self._attribute_by_embedding(sent, chunks)
            out.append(
                AttributedSentence(
                    sid=sid,
                    text=sent,
                    attributed_to=attrs,
                    sources=("embedding",) if attrs else ("unsourced",),
                    similarity=sim,
                )
            )
        logger.info(
            f"[Attribution] {qid} : {len(out)} phrases "
            f"({sum(1 for s in out if s.attributed_to)} sourcées, "
            f"{sum(1 for s in out if not s.attributed_to)} sans source)"
        )
        return out
