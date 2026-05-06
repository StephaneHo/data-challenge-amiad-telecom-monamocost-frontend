from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import textwrap
from loguru import logger
from sqlalchemy.orm import Session
from config import settings
from sentence_transformers import SentenceTransformer
import torch
from sqlalchemy import text


@dataclass
class RetrievedChunk:
    """un chunk retrouvé par la recherche vectorielle"""

    paper_id: str
    title: str
    published_at: Optional[str]
    venue: Optional[str]
    content: str
    section: Optional[str]
    score: float


@dataclass
class RAGResponse:
    """La réponse complère du moteur RAG"""

    query: str
    answer: str
    references: list[dict] = field(default_factory=list)
    tokens_used: int = 0
    retrieval_scores: list[float] = field(default_factory=list)


class RAGEngine:
    """
    Moteur RAG pour la veille scientifique

    Il vectorise la question
    Il recherche les chunks les plus proches dans pgvecotr
    Il envoie ensuite les chunks et la question au LLM
    Il retourne la question avec les références
    """

    _SYSTEM_PROMPT = textwrap.dedent("""\
        Tu es un assistant de veille scientifique.
        Tu t'appuies uniquement sur les extraits de publications fournis.
        Tu cites systématiquement tes sources au format [ArXiv:ID].
        Tu rédiges en français sauf si l'utilisateur écrit en anglais.
        Tu es rigoureux, factuel et neutre.
    """)

    def _build_llm(self):
        """On va construire le client LLM selon la config de .env"""
        provider = settings.LLM_PROVIDER.lower()
        if provider == "openai":
            from openai import OpenAI

            return OpenAI(api_key=settings.OPENAI_API_KEY)
        elif provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        raise ValueError(f"Provider inconnu: {provider}")

    def __init__(self, session: Session) -> None:
        self.session = session
        logger.info(f"[RAG]: Chargement embedder: {settings.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.llm = self._build_llm()
        logger.info(f"[RAG] LLM : {settings.LLM_PROVIDER} / {settings.LLM_MODEL}")

    def _retrieve(
        self,
        query: str,
        k: int = 5,
        year_from: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        C'est la première étape du RAG (Retrieval)
        """

        """Là, on vectorise la question"""
        with torch.no_grad():
            query_vector = self.embedder.encode(
                query, normalize_embeddings=True
            ).tolist()

        """ On construit nos params pour la requete SQL"""
        params = {"query_vector": query_vector, "k": k}

        if year_from:
            """ TODO: A COMPLETER PLUS TARD """
            pass
        """ 
        Recherche cosinus dans pgvector
        la distance cosinus est : <=>
        et 1 - distance, c'est le score de similarité
        """

        sql = text("""
            SELECT
                c.paper_id,
                p.title,
                p.published_at,
                p.venue,
                c.content,
                c.section,  
                1 - (c.embedding <=> CAST(:query_vector AS vector)) AS score
            FROM paper_chunks c
            JOIN papers p ON c.paper_id = p.arxiv_id
            ORDER BY c.embedding <=> CAST(:query_vector AS vector)
            LIMIT :k    
        """)

        rows = self.session.execute(sql, params).fetchall()

        return [
            RetrievedChunk(
                paper_id=r.paper_id,
                title=r.title,
                published_at=str(r.published_at.date()) if r.published_at else None,
                venue=r.venue,
                content=r.content,
                section=r.section,
                score=float(r.score),
            )
            for r in rows
        ]

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk], max_chunks: int = 10) -> str:
        """Formate les chunks en contexte lisible pour le LLM"""
        lines: list[str] = []

        for c in chunks:
            year = c.published_at[:4] if c.published_at else "?"
            venue = f" - {c.venue}" if c.venue else ""
            lines.append(f"\[ArXiv:{c.paper_id}] {c.title} ({year}){venue} ---")
            lines.append(f"[{c.section or 'Body'}] {c.content}")

        return "\n".join(lines)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return textwrap.dedent(f"""\
            Question : {query}
            
            Extraits de publications scientifiques:
            {context}
            
            Réponds de façon structurée et précise.
            
            Cite tes sources [ArXiv:ID]
        """)

    def _call_llm(
        self, prompt: str, temperature: float | None = None
    ) -> tuple[str, int]:
        """Appelle le LLM et retourne (texte_réponse, tokens utilisés)"""
        temp = temperature if temperature is not None else settings.RAG_TEMPERATURE
        provider = settings.LLM_PROVIDER.lower()

        if provider == "openai":
            response = self.llm.chat.completions.create(
                model=settings.LLM_MODEL,
                temperature=temp,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text_out = response.choices[0].message.content
            tokens = response.usage.total_tokens

        else:
            raise ValueError(f"Provider inconnu: {provider}")

        logger.debug(f"[RAG] {tokens} tokens utilisés")
        return text_out, tokens

    @staticmethod
    def _build_references(chunks: list[RetrievedChunk]) -> list[dict]:
        """Formate les références pour la réponse"""
        seen: dict[str, dict] = {}
        for c in chunks:
            if c.paper_id not in seen:
                seen[c.paper_id] = {
                    "arxiv_id": c.paper_id,
                    "title": c.title,
                    "published_at": c.published_at,
                    "venue": c.venue,
                    "url": f"https://arxiv.org/abs/{c.paper_id}",
                    "score": round(c.score, 4),
                }
        return list(seen.values())

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        year_from: int | None = None,
    ) -> RAGResponse:
        """On répond à la question en utilisant les chunk indexés"""
        k = top_k or settings.RAG_TOP_K
        chunks = self._retrieve(query, k=k, year_from=year_from)

        if not chunks:
            return RAGResponse(
                query=query,
                answer="Aucun article pertinent trouvé dans la base documentaire",
            )

        context = self._build_context(chunks)
        prompt = self._build_prompt(query, context)
        answer_txt, tokens = self._call_llm(prompt)

        return RAGResponse(
            query=query,
            answer=answer_txt,
            references=self._build_references(chunks),
            tokens_used=tokens,
            retrieval_scores=[c.score for c in chunks],
        )
