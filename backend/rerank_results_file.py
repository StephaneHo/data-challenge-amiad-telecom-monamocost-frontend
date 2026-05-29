"""
Rerank a local retrieval results file without running the full RAG pipeline.

Expected input format for --input:
{
  "Q1": [
    ["chunk text", 0.9006, "document.pdf"],
    ...
  ],
  ...
}

The questions file can be a challenge JSON with results[].{qid, question}, or a
simple mapping {"Q1": "question text"}.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from rag.reranker import Reranker


@dataclass
class CandidateChunk:
    content: str
    score: float
    embedding_score: float
    doc_name: str


def load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return {
            item["qid"]: item["question"]
            for item in payload["results"]
            if item.get("qid") and item.get("question")
        }
    if isinstance(payload, dict):
        return {
            str(qid): str(question)
            for qid, question in payload.items()
            if isinstance(question, str)
        }
    raise ValueError(
        "Le fichier de questions doit être au format challenge ou dict qid->question."
    )


def parse_candidate(raw: list[Any]) -> CandidateChunk:
    if len(raw) < 3:
        raise ValueError(f"Candidat invalide, attendu [texte, score, doc] : {raw!r}")
    return CandidateChunk(
        content=str(raw[0]),
        score=float(raw[1]),
        embedding_score=float(raw[1]),
        doc_name=str(raw[2]),
    )


def candidate_to_json(chunk: CandidateChunk) -> dict[str, Any]:
    return {
        "content": chunk.content,
        "embedding_score": round(float(chunk.embedding_score), 6),
        "reranker_score": round(float(chunk.score), 6),
        "doc_name": chunk.doc_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerank local top-k retrieval candidates with a CrossEncoder."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("resultats_finetuned_2000_opti.json"),
        help="JSON qid -> candidats [chunk, score_embedder, doc_name].",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="JSON contenant les questions: format challenge ou dict qid->question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultats_finetuned_2000_opti_reranked.json"),
        help="JSON de sortie avec candidats rerankés.",
    )
    parser.add_argument(
        "--top-k-chunks",
        type=int,
        default=100,
        help="Nombre de candidats initiaux à reranker par question.",
    )
    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=30,
        help="Nombre de candidats à garder après reranking.",
    )
    parser.add_argument(
        "--reranker-model",
        default=settings.RERANKER_MODEL,
        help="Modèle CrossEncoder à utiliser.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.RERANKER_BATCH_SIZE,
        help="Batch size d'inférence du reranker.",
    )
    parser.add_argument(
        "--filter-qids",
        default=None,
        help="Liste optionnelle de qids séparés par virgule, ex: Q1,Q2.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    logger.info(f"[RerankLocal] Questions chargées: {len(questions)}")

    logger.info(f"[RerankLocal] Chargement candidats: {args.input}")
    candidates_by_qid = json.loads(
        args.input.read_text(encoding="utf-8", errors="replace")
    )
    if not isinstance(candidates_by_qid, dict):
        raise ValueError("--input doit contenir un objet JSON qid -> candidats.")

    keep_qids = None
    if args.filter_qids:
        keep_qids = {qid.strip() for qid in args.filter_qids.split(",") if qid.strip()}

    reranker = Reranker(
        model_name=args.reranker_model,
        batch_size=args.batch_size,
    )

    results: list[dict[str, Any]] = []
    missing_questions: list[str] = []
    for qid, raw_candidates in candidates_by_qid.items():
        if keep_qids is not None and qid not in keep_qids:
            continue
        question = questions.get(qid)
        if not question:
            missing_questions.append(qid)
            continue
        if not isinstance(raw_candidates, list):
            logger.warning(f"[RerankLocal] {qid}: candidats invalides, ignoré")
            continue

        initial = raw_candidates[: args.top_k_chunks]
        chunks = [parse_candidate(item) for item in initial]
        reranked = reranker.rerank(
            query=question,
            chunks=chunks,
            top_k=args.reranker_top_k,
        )
        results.append(
            {
                "qid": qid,
                "question": question,
                "input_candidates": len(chunks),
                "reranked": [
                    {
                        "rank": rank,
                        **candidate_to_json(chunk),
                    }
                    for rank, chunk in enumerate(reranked, start=1)
                ],
            }
        )
        logger.info(
            f"[RerankLocal] {qid}: {len(chunks)} candidats -> "
            f"{len(reranked)} rerankés"
        )

    if missing_questions:
        logger.warning(
            "[RerankLocal] Questions manquantes pour "
            f"{len(missing_questions)} qids: {missing_questions[:10]}"
        )

    output = {
        "parameters": {
            "input": str(args.input),
            "questions": str(args.questions),
            "top_k_chunks": args.top_k_chunks,
            "reranker_top_k": args.reranker_top_k,
            "reranker_model": args.reranker_model,
            "batch_size": args.batch_size,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[RerankLocal] Sortie écrite: {args.output}")

    if not results:
        logger.error("[RerankLocal] Aucun résultat produit.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
