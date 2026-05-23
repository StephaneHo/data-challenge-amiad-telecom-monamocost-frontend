from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from pipeline.embedder import Embedder
from pipeline.query_decomposition import QueryDecomposer
from rag.attribution import (
    AttributedSentence,
    Attributor,
    RetrievedChunkForAttribution,
)
from utils.carbon import CarbonTracker


# Granularité interne (chunk) — utilisée pour bâtir le contexte LLM.
@dataclass
class RetrievedChunk:
    doc_name: str
    page_number: int
    chunk_index: int
    content: str
    score: float


# Granularité du challenge ((doc_name, page) ranké) — utilisée pour le JSON de sortie.
@dataclass
class RetrievedPage:
    rank: int
    doc_name: str
    page: int
    score: float
    n_chunks: int  # combien de chunks de cette page ont matché dans le top-K


@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved: list[RetrievedPage] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    carbon: dict[str, Any] = field(default_factory=dict)


class RAGEngine:
    """
    Moteur RAG pour le challenge EvalLLM 2026.

    Pipeline :
      1. embed la requête (préfixe `query: ` via `Embedder`)
      2. cherche les top-K chunks par cosinus dans pgvector
      3. agrège à la granularité (doc_name, page) — le challenge évalue à cette grain-là
      4. construit un contexte pour le LLM (chunks gardés au max `context_chunks`)
      5. demande au LLM une réponse avec citations `[doc_name p.N]`
    """

    _SYSTEM_PROMPT = textwrap.dedent(
        """\
        Tu es un assistant spécialisé en synthèse documentaire pour la défense et le renseignement.

        Règles strictes :
        - Réponds UNIQUEMENT à partir des extraits fournis. N'invente aucun fait.
        - Cite tes sources à chaque affirmation factuelle au format `[nom_du_document.pdf p.N]`.
        - Si plusieurs sources convergent sur une même affirmation, cite-les toutes.
        - Si tu introduis une information de **connaissance générale** ou de mise en
          contexte qui ne provient pas explicitement des extraits, ne mets PAS de
          citation derrière. Cette absence de citation signale que la phrase est
          non sourcée — c'est attendu pour les transitions, titres et mises en perspective.
        - Si l'information demandée n'est pas du tout dans les extraits, dis-le
          explicitement plutôt que d'inventer.
        - Rédige en français, dans un style factuel et synthétique. Utilise des titres
          markdown (#, ##) pour structurer si la réponse est longue.

        Exemple de réponse bien annotée :

        Contexte fourni : un seul extrait du document `guide_osint_infrastructure_v2.pdf` p.1
        sur l'analyse des certificats SSL/TLS et l'usage des CT Logs.

        Question : Comment l'utilisation des logs de certificats peut-elle aider à
        découvrir l'infrastructure d'une cible sans se faire repérer ?

        Réponse attendue :

        # Méthodologie de reconnaissance passive via SSL

        L'exploitation des journaux de transparence (CT Logs) permet de découvrir
        des sous-domaines et environnements de pré-production en examinant les
        champs CN et SAN des certificats. [guide_osint_infrastructure_v2.pdf p.1]

        Cette approche garantit la discrétion de l'analyste car elle constitue une
        technique passive, évitant ainsi le déclenchement d'alertes au niveau des
        pare-feu applicatifs (WAF) qui ciblent habituellement les scans actifs.
        [guide_osint_infrastructure_v2.pdf p.1]

        En complément, cette phase est souvent la première étape d'une chaîne
        d'attaque plus complexe appelée "Recon-ng".

        Enfin, la validation de ces informations doit impérativement passer par
        une corrélation avec l'historique des enregistrements DNS.
        [guide_osint_infrastructure_v2.pdf p.1]

        Note : la phrase sur Recon-ng n'a pas de citation car c'est de la connaissance
        générale, pas dans l'extrait fourni. Le titre `#` n'a pas de citation non plus
        car c'est de la mise en forme.
        """
    )

    def __init__(
        self,
        session: Session,
        embedder: Optional[Embedder] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.session = session
        self.embedder = embedder or Embedder()
        if self.embedder.dim != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"[RAG] Dim mismatch : embedder={self.embedder.dim}, settings={settings.EMBEDDING_DIM}"
            )
        self.llm_provider = (llm_provider or settings.LLM_PROVIDER).lower()
        self.llm_model = llm_model or settings.LLM_MODEL
        self._llm_client: Any = None  # initialisé paresseusement
        self._decomposer: Optional[QueryDecomposer] = None

    def _build_llm(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        if self.llm_provider == "openai":
            # Supporte les serveurs LLM locaux compatibles OpenAI (vLLM, Ollama, ...)
            # via LLM_BASE_URL dans config.py.
            from utils.llm import get_openai_client

            self._llm_client = get_openai_client()
        elif self.llm_provider == "mistral":
            from utils.llm import get_mistral_client

            self._llm_client = get_mistral_client()
        elif self.llm_provider == "anthropic":
            import anthropic

            self._llm_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Provider LLM inconnu : {self.llm_provider}")
        return self._llm_client

    # ──────────────────────── Retrieval ────────────────────────

    def retrieve_chunks_decomposed(
        self,
        query: str,
        k: int = 30,
        per_subquery_k: int = 20,
        doc_filter: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieval avec décomposition de la question multi-hop.
          1. Décompose la question en N sous-questions (1 si déjà atomique)
          2. Pour chaque sous-question, retrieve `per_subquery_k` chunks
          3. Fusionne en gardant le meilleur score par (doc, page, chunk_index)
          4. Trie par score, garde le top `k`
        """
        if self._decomposer is None:
            self._decomposer = QueryDecomposer()
        decomp = self._decomposer.decompose(query)
        mode = settings.RETRIEVAL_MODE
        if decomp.is_atomic:
            return self.retrieve_chunks(query, k=k, doc_filter=doc_filter, mode=mode)

        # Fusion : on indexe par (doc, page, chunk_index) et on garde le meilleur score
        best: dict[tuple[str, int, int], RetrievedChunk] = {}
        for subq in decomp.subqueries:
            for c in self.retrieve_chunks(subq, k=per_subquery_k, doc_filter=doc_filter, mode=mode):
                key = (c.doc_name, c.page_number, c.chunk_index)
                cur = best.get(key)
                if cur is None or c.score > cur.score:
                    best[key] = c
        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ranked[:k]

    def retrieve_chunks(
        self,
        query: str,
        k: int = 30,
        doc_filter: Optional[list[str]] = None,
        mode: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Top-K chunks par mode de retrieval.
        - `mode="dense"` (défaut config) : cosinus pgvector via e5
        - `mode="bm25"` : BM25-like via Postgres ts_rank sur `content_tsv` français
        - `mode="hybrid"` : fusion RRF des deux (Reciprocal Rank Fusion, k=60)
        """
        mode = (mode or settings.RETRIEVAL_MODE).lower()
        if mode == "dense":
            return self._retrieve_dense(query, k=k, doc_filter=doc_filter)
        if mode == "bm25":
            return self._retrieve_bm25(query, k=k, doc_filter=doc_filter)
        if mode == "hybrid":
            return self._retrieve_hybrid(query, k=k, doc_filter=doc_filter)
        raise ValueError(f"Mode de retrieval inconnu : {mode}")

    def _retrieve_dense(
        self, query: str, k: int, doc_filter: Optional[list[str]] = None
    ) -> list[RetrievedChunk]:
        """Cosinus pgvector via embeddings e5."""
        qv = self.embedder.encode_query(query)
        sql = (
            "SELECT id, doc_name, page_number, chunk_index, content, "
            "1 - (embedding <=> CAST(:qv AS vector)) AS score "
            "FROM chunks WHERE embedding IS NOT NULL"
        )
        params: dict[str, Any] = {"qv": qv, "k": k}
        if doc_filter:
            sql += " AND doc_name = ANY(:docs)"
            params["docs"] = doc_filter
        sql += " ORDER BY embedding <=> CAST(:qv AS vector) LIMIT :k"
        rows = self.session.execute(text(sql), params).fetchall()
        return [
            RetrievedChunk(
                doc_name=r.doc_name,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                content=r.content,
                score=float(r.score),
            )
            for r in rows
        ]

    @staticmethod
    def _build_or_tsquery(query: str) -> str:
        """
        Convertit une requête naturelle en `to_tsquery` Postgres avec OR (`|`).
        - Tokenize en gardant lettres FR/EN + chiffres + tirets (préserve `MQ-9`, `r20-7111`)
        - Filtre les mots < 3 chars et un set de stopwords FR
        - Joint avec ` | ` (OR logique). Si vide → chaîne nulle (retrieval renverra 0).

        `to_tsquery` exige une syntaxe stricte ; les caractères spéciaux doivent
        être échappés. On ne garde que des tokens propres pour éviter les erreurs.
        """
        import re

        tokens = re.findall(r"[A-Za-zÀ-ÿ0-9\-]{2,}", query)
        stop = {
            "les", "des", "pour", "avec", "dans", "sur", "par", "est", "sont",
            "comment", "quel", "quels", "quelle", "quelles", "aux", "que", "qui",
            "une", "ces", "leur", "leurs", "cette", "tout", "tous", "toutes",
            "etre", "etre", "avoir", "ete", "ont", "ait", "ainsi", "donc", "mais",
            "plus", "moins", "tres", "bien", "encore", "deja", "non", "oui",
        }
        # Garde tokens longs OU contenant chiffres/tirets (typiquement acronymes/codes)
        keep: list[str] = []
        for t in tokens:
            tl = t.lower()
            if tl in stop:
                continue
            if len(t) >= 3 or any(c.isdigit() or c == "-" for c in t):
                # tsquery interdit certains chars : on ne garde que [a-zA-Z0-9-_]
                clean = re.sub(r"[^A-Za-zÀ-ÿ0-9_\-]", "", t)
                if clean and len(clean) >= 2:
                    keep.append(clean.lower())
        # Dédupe en préservant l'ordre
        seen: set[str] = set()
        unique: list[str] = []
        for t in keep:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return " | ".join(unique)

    def _retrieve_bm25(
        self, query: str, k: int, doc_filter: Optional[list[str]] = None
    ) -> list[RetrievedChunk]:
        """
        BM25-like via Postgres : `ts_rank` sur `content_tsv`.

        Stratégie : tokenize la question côté code et construit une `to_tsquery`
        OR (`mot1 | mot2 | ...`) — chaque chunk contenant AU MOINS UN des termes
        clés remontera, classé par densité lexicale (`ts_rank`).

        Pourquoi pas `websearch_to_tsquery` : celui-ci met un AND implicite
        entre tous les termes, ce qui renvoie 0 résultat dès que la question
        a plus de 3-4 mots-clés (rare qu'un chunk contienne tout).
        """
        tsq_str = self._build_or_tsquery(query)
        if not tsq_str:
            return []
        sql = (
            "SELECT id, doc_name, page_number, chunk_index, content, "
            "ts_rank(content_tsv, to_tsquery('french', :tsq)) AS score "
            "FROM chunks "
            "WHERE content_tsv @@ to_tsquery('french', :tsq)"
        )
        params: dict[str, Any] = {"tsq": tsq_str, "k": k}
        if doc_filter:
            sql += " AND doc_name = ANY(:docs)"
            params["docs"] = doc_filter
        sql += " ORDER BY score DESC LIMIT :k"
        try:
            rows = self.session.execute(text(sql), params).fetchall()
        except Exception as ex:
            logger.warning(f"[BM25] tsquery invalide ({ex}) → 0 résultats")
            return []
        return [
            RetrievedChunk(
                doc_name=r.doc_name,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                content=r.content,
                score=float(r.score),
            )
            for r in rows
        ]

    def _retrieve_hybrid(
        self,
        query: str,
        k: int,
        doc_filter: Optional[list[str]] = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion : combine les rangs dense et BM25.
        RRF_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))
        avec k=60 (valeur standard, Cormack et al. 2009).
        On retrieve `k_pool = k * 2` de chaque côté pour la fusion (~k * 2 candidats
        au pool), puis on garde le top-k après RRF.
        """
        k_pool = max(k * 2, 30)
        dense = self._retrieve_dense(query, k=k_pool, doc_filter=doc_filter)
        bm25 = self._retrieve_bm25(query, k=k_pool, doc_filter=doc_filter)

        # Indexe par (doc, page, chunk_index) pour fusionner — l'`id` SQL n'est
        # pas dans RetrievedChunk, mais la clé (doc, page, idx) est unique.
        ranks_dense: dict[tuple[str, int, int], int] = {
            (c.doc_name, c.page_number, c.chunk_index): i + 1
            for i, c in enumerate(dense)
        }
        ranks_bm25: dict[tuple[str, int, int], int] = {
            (c.doc_name, c.page_number, c.chunk_index): i + 1
            for i, c in enumerate(bm25)
        }
        by_key: dict[tuple[str, int, int], RetrievedChunk] = {
            (c.doc_name, c.page_number, c.chunk_index): c for c in dense
        }
        for c in bm25:
            by_key.setdefault((c.doc_name, c.page_number, c.chunk_index), c)

        rrf_scores: dict[tuple[str, int, int], float] = {}
        for key in by_key:
            score = 0.0
            if key in ranks_dense:
                score += 1.0 / (rrf_k + ranks_dense[key])
            if key in ranks_bm25:
                score += 1.0 / (rrf_k + ranks_bm25[key])
            rrf_scores[key] = score

        ranked = sorted(by_key.values(), key=lambda c: rrf_scores[(c.doc_name, c.page_number, c.chunk_index)], reverse=True)

        # On remplace `score` par le RRF pour pouvoir le tracer en aval
        out: list[RetrievedChunk] = []
        for c in ranked[:k]:
            key = (c.doc_name, c.page_number, c.chunk_index)
            out.append(
                RetrievedChunk(
                    doc_name=c.doc_name,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    score=rrf_scores[key],
                )
            )
        return out

    @staticmethod
    def aggregate_to_pages(
        chunks: list[RetrievedChunk], top_n: Optional[int] = None
    ) -> list[RetrievedPage]:
        """
        Dédoublonne par (doc_name, page) en gardant le meilleur score.
        Compte aussi le nombre de chunks par page (utile pour le scoring).
        """
        best: dict[tuple[str, int], dict[str, Any]] = {}
        for c in chunks:
            key = (c.doc_name, c.page_number)
            entry = best.get(key)
            if entry is None or c.score > entry["score"]:
                best[key] = {
                    "doc_name": c.doc_name,
                    "page": c.page_number,
                    "score": c.score,
                    "n_chunks": (entry["n_chunks"] + 1) if entry else 1,
                }
            else:
                entry["n_chunks"] += 1

        ranked = sorted(best.values(), key=lambda e: e["score"], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return [
            RetrievedPage(
                rank=i + 1,
                doc_name=e["doc_name"],
                page=e["page"],
                score=round(e["score"], 6),
                n_chunks=e["n_chunks"],
            )
            for i, e in enumerate(ranked)
        ]

    # ──────────────────────── Génération ────────────────────────

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk], max_chunks: int = 12) -> str:
        """Formate les chunks pour le LLM, avec marqueur [doc_name p.N]."""
        lines: list[str] = []
        for c in chunks[:max_chunks]:
            tag = f"[{c.doc_name} p.{c.page_number}]"
            lines.append(f"{tag}\n{c.content}")
        return "\n\n---\n\n".join(lines)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return textwrap.dedent(
            f"""\
            Question :
            {query}

            Extraits des documents sources :
            ---
            {context}
            ---

            Rédige une réponse structurée et factuelle. Cite tes sources au
            format `[doc_name p.N]` à chaque affirmation factuelle.
            """
        )

    def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> tuple[str, int]:
        """Retourne (réponse_texte, tokens_utilisés)."""
        temp = temperature if temperature is not None else settings.RAG_TEMPERATURE
        client = self._build_llm()

        if self.llm_provider in ("openai", "mistral"):
            # max_tokens=800 borne la durée de génération sur LLM local lent (Qwen/Mistral
            # sur CPU peuvent générer indéfiniment). Sur gpt-4o-mini c'est large.
            resp = client.chat.completions.create(
                model=self.llm_model,
                temperature=temp,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or "", int(resp.usage.total_tokens)

        if self.llm_provider == "anthropic":
            resp = client.messages.create(
                model=self.llm_model,
                max_tokens=2048,
                temperature=temp,
                system=self._SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text_out = "".join(b.text for b in resp.content if getattr(b, "text", None))
            tokens = int(resp.usage.input_tokens + resp.usage.output_tokens)
            return text_out, tokens

        raise ValueError(f"Provider LLM inconnu : {self.llm_provider}")

    # ──────────────────────── API publique ────────────────────────

    def answer(
        self,
        query: str,
        top_k_chunks: Optional[int] = None,
        top_n_pages: Optional[int] = None,
        context_chunks: int = 12,
        doc_filter: Optional[list[str]] = None,
        retrieval_only: bool = False,
        decompose: bool = False,
        retrieval_mode: Optional[str] = None,
    ) -> RAGResponse:
        """
        Pipeline complet : retrieval → agrégation pages → génération.

        - top_k_chunks : combien de chunks scorer dans pgvector (défaut: RAG_TOP_K * 3)
        - top_n_pages  : combien de paires (doc, page) renvoyer dans `retrieved`
                         (défaut : tous les uniques du top-K)
        - context_chunks : combien de chunks passer au LLM
        - retrieval_only : court-circuite l'appel LLM (utile pour évaluer le retrieval seul)
        """
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tracker = CarbonTracker(label="rag_inference", device=device)

        with tracker.measure():
            k = top_k_chunks or (settings.RAG_TOP_K * 3)
            mode = retrieval_mode or settings.RETRIEVAL_MODE
            if decompose:
                chunks = self.retrieve_chunks_decomposed(query, k=k, doc_filter=doc_filter)
            else:
                chunks = self.retrieve_chunks(query, k=k, doc_filter=doc_filter, mode=mode)

            if not chunks:
                tracker.log_summary()
                return RAGResponse(
                    question=query,
                    answer="Aucun extrait pertinent trouvé dans la base documentaire.",
                    parameters=self._params_dict(k=k, top_n=top_n_pages),
                    carbon=tracker.metrics.to_dict(),
                )

            retrieved_pages = self.aggregate_to_pages(chunks, top_n=top_n_pages)

            if retrieval_only:
                tracker.log_summary()
                return RAGResponse(
                    question=query,
                    answer="",
                    retrieved=retrieved_pages,
                    chunks=chunks,
                    parameters=self._params_dict(k=k, top_n=top_n_pages),
                    carbon=tracker.metrics.to_dict(),
                )

            context = self._build_context(chunks, max_chunks=context_chunks)
            prompt = self._build_prompt(query, context)
            input_tokens_estimate = len(self._SYSTEM_PROMPT) // 4 + len(prompt) // 4
            try:
                answer_txt, tokens = self._call_llm(prompt)
                # On approxime input/output 50/50 si le provider ne sépare pas
                output_tokens_estimate = max(0, tokens - input_tokens_estimate)
                if output_tokens_estimate <= 0:
                    input_tokens_estimate = tokens // 2
                    output_tokens_estimate = tokens - input_tokens_estimate
                tracker.add_llm_tokens(
                    input_tokens=input_tokens_estimate,
                    output_tokens=output_tokens_estimate,
                )
            except Exception as ex:
                logger.exception(f"[RAG] Appel LLM échoué : {ex}")
                answer_txt, tokens = "", 0

        tracker.log_summary()
        return RAGResponse(
            question=query,
            answer=answer_txt,
            retrieved=retrieved_pages,
            chunks=chunks,
            parameters=self._params_dict(k=k, top_n=top_n_pages),
            tokens_used=tokens,
            carbon=tracker.metrics.to_dict(),
        )

    def attribute(
        self,
        qid: str,
        answer: str,
        chunks: list[RetrievedChunk],
        threshold: float = 0.80,
        secondary_threshold: float = 0.85,
        topk_per_sentence: int = 3,
        entity_filter: bool = True,
        entity_min_support_ratio: float = 0.5,
    ) -> list[AttributedSentence]:
        """
        Tâche 2 du challenge : attribue chaque phrase de `answer` à un (doc, page).

        Pipeline :
          1. Markdown (titres, séparateurs) → `[]`
          2. Filtre entité (acronymes/nombres/noms propres absents des chunks) → `[]`
          3. Citations `[doc.pdf p.N]` → attribution directe, page résolue par sim si absente
          4. Embedding similarity Top-K :
             - top-1 retenu si sim ≥ threshold
             - top-2, top-3 retenus si sim ≥ secondary_threshold (plus strict)
             - sinon `[]`
        """
        attributor = Attributor(
            embedder=self.embedder,
            threshold=threshold,
            secondary_threshold=secondary_threshold,
            topk_per_sentence=topk_per_sentence,
            entity_filter=entity_filter,
            entity_min_support_ratio=entity_min_support_ratio,
        )
        attribution_chunks = [
            RetrievedChunkForAttribution(
                doc_name=c.doc_name,
                page_number=c.page_number,
                content=c.content,
            )
            for c in chunks
        ]
        return attributor.attribute(qid=qid, answer=answer, chunks=attribution_chunks)

    def _params_dict(self, k: int, top_n: Optional[int]) -> dict[str, Any]:
        return {
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "retriever_top_k_chunks": k,
            "retriever_top_n_pages": top_n,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "temperature": settings.RAG_TEMPERATURE,
        }
