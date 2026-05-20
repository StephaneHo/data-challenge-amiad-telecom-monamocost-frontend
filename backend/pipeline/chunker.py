from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """Un chunk indexable extrait d'une page donnée."""

    page_number: int
    chunk_index: int  # index dans la page
    content: str


_PARA_SPLIT = re.compile(r"\n\s*\n")
_LONELY_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_MULTISPACE = re.compile(r" {2,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZÉÈÀÂÎÔÙÇ])")


def split_into_paragraphs(text: str, min_chars: int = 50) -> list[str]:
    """
    Découpe un texte en paragraphes propres.

    Logique reprise du notebook `Experimental/Exploration_données.ipynb` :
    - sépare sur les doubles sauts de ligne (paragraphes)
    - recolle les sauts de ligne uniques (texte fluide)
    - normalise les espaces multiples
    - filtre les paragraphes trop courts (probables artefacts de chunking PDF)
    """
    paragraphs: list[str] = []
    for raw in _PARA_SPLIT.split(text):
        cleaned = _LONELY_NEWLINE.sub(" ", raw).strip()
        cleaned = _MULTISPACE.sub(" ", cleaned)
        if len(cleaned) >= min_chars:
            paragraphs.append(cleaned)
    return paragraphs


def split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """Découpe un paragraphe trop long sur frontières de phrase."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for s in sentences:
        if buf_len + len(s) > max_chars and buf:
            out.append(" ".join(buf).strip())
            buf, buf_len = [s], len(s)
        else:
            buf.append(s)
            buf_len += len(s) + 1
    if buf:
        out.append(" ".join(buf).strip())
    # Sécurité : si une « phrase » est elle-même monstrueuse (PDF mal extrait),
    # on coupe à la louche tous les max_chars caractères.
    final: list[str] = []
    for piece in out:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            final.extend(
                piece[i : i + max_chars] for i in range(0, len(piece), max_chars)
            )
    return final


def chunk_page(
    page_number: int,
    text: str,
    min_chars: int = 50,
    max_chars: int = 1500,
) -> list[Chunk]:
    """
    Produit la liste des chunks indexables d'une page.
    Garantit `chunk_index` séquentiel à partir de 0.
    """
    paragraphs = split_into_paragraphs(text, min_chars=min_chars)
    out: list[Chunk] = []
    idx = 0
    for para in paragraphs:
        for piece in split_long_paragraph(para, max_chars=max_chars):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            out.append(Chunk(page_number=page_number, chunk_index=idx, content=piece))
            idx += 1
    return out
