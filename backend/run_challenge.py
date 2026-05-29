"""
CLI : exécute Tâche 1 + Tâche 2 du challenge sur un fichier JSON d'entrée
et écrit les sorties sur disque.

Usage :
    python run_challenge.py --input ../Experimental/DATA/training/sample_queries.json
    python run_challenge.py --input questions.json --out-task1 t1.json --out-task2 t2.json
    python run_challenge.py --input questions.json --no-task2 --retrieval-only
    python run_challenge.py --input questions.json --filter-qids Q1,Q4

    # Tâche 2 standalone : prend un JSON Task 1 (le nôtre OU celui des organisateurs)
    # et produit uniquement les attributions Task 2, sans regénérer la réponse.
    python run_challenge.py --task2-from external_task1.json --out-task2 my_task2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from api.challenge import ChallengeInput, ChallengeRunner
from database.db import get_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner challenge EvalLLM 2026")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSON input (questions). Requis sauf si --task2-from est utilisé.",
    )
    parser.add_argument(
        "--task2-from",
        type=Path,
        default=None,
        help="JSON au format Task 1 output (question + retrieved + answer) à attribuer "
        "phrase par phrase. Mode standalone : ne regénère pas la réponse.",
    )
    parser.add_argument(
        "--out-task1",
        type=Path,
        default=None,
        help="Chemin du JSON Tâche 1 (défaut: <input>_task1.json à côté de l'input)",
    )
    parser.add_argument(
        "--out-task2",
        type=Path,
        default=None,
        help="Chemin du JSON Tâche 2 (défaut: <input>_task2.json à côté de l'input)",
    )
    parser.add_argument("--no-task2", action="store_true", help="Skip l'attribution (Tâche 2)")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip l'appel LLM")
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Active la décomposition de question multi-hop via LLM (chantier 1). "
        "Pour chaque question, génère 1-4 sous-questions, retrieve chacune, fusionne. "
        "Améliore le rappel sur les questions complexes type Q3 du sample.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "bm25", "hybrid"],
        default=None,
        help="Mode de retrieval (défaut: valeur de RETRIEVAL_MODE dans .env). "
        "`hybrid` fusionne dense (e5) et BM25 par RRF — meilleur sur les acronymes/noms propres rares.",
    )
    parser.add_argument("--top-k-chunks", type=int, default=20)
    parser.add_argument("--top-n-pages", type=int, default=10)
    parser.add_argument("--context-chunks", type=int, default=10)
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active le reranking CrossEncoder après le retrieval initial. "
        "Sans ce flag, utilise RERANK_ENABLED depuis .env.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help=(
            "Modèle CrossEncoder de reranking "
            "(défaut: RERANKER_MODEL dans .env)."
        ),
    )
    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=None,
        help=(
            "Nombre de chunks conservés après reranking "
            "(défaut: RERANKER_TOP_K)."
        ),
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size du reranker CrossEncoder "
            "(défaut: RERANKER_BATCH_SIZE)."
        ),
    )
    parser.add_argument("--attribution-threshold", type=float, default=0.80)
    parser.add_argument(
        "--filter-qids",
        type=str,
        default=None,
        help=(
            "Liste de qids séparés par des virgules (ex: 'Q1,Q4') "
            "— restreint à ces questions"
        ),
    )
    parser.add_argument(
        "--constrained-decoding",
        action="store_true",
        help="Active la couche de décodage contraint (grammar-guided generation). "
        "Le LLM est forcé à produire un JSON conforme à un schéma Pydantic "
        "(answer + citations structurées) — supprime les erreurs de format et "
        "rend les citations directement parsables. Compatible OpenAI Structured "
        "Outputs, Mistral JSON mode, vLLM guided_json et Outlines local.",
    )
    parser.add_argument(
        "--constrained-backend",
        choices=["auto", "openai_schema", "mistral_json", "anthropic_tool",
                 "vllm_guided", "outlines_local", "fallback_repair"],
        default="auto",
        help="Choix du backend de décodage contraint (défaut: auto = selon LLM_PROVIDER). "
        "`outlines_local` charge un modèle HF in-process avec Outlines "
        "(installer `uv sync --group constrained`).",
    )
    args = parser.parse_args()

    # Mode standalone Task 2 : ne nécessite pas --input
    if args.task2_from is not None:
        external = json.loads(args.task2_from.read_text(encoding="utf-8"))
        with get_session() as session:
            runner = ChallengeRunner(
                session=session,
                top_k_chunks=args.top_k_chunks,
                top_n_pages=args.top_n_pages,
                context_chunks=args.context_chunks,
                attribution_threshold=args.attribution_threshold,
                rerank=args.rerank,
                reranker_model=args.reranker_model,
                reranker_top_k=args.reranker_top_k,
                reranker_batch_size=args.reranker_batch_size,
            )
            task2 = runner.run_task2_standalone(external)
        out_task2 = args.out_task2 or args.task2_from.parent / f"{args.task2_from.stem}_task2.json"
        out_task2.write_text(
            json.dumps(task2.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[CLI] Tâche 2 (standalone) écrite : {out_task2}")
        return 0

    if args.input is None:
        logger.error("[CLI] --input requis sauf si --task2-from est fourni.")
        return 1

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    payload = ChallengeInput.model_validate(raw)

    if args.filter_qids:
        keep = {q.strip() for q in args.filter_qids.split(",") if q.strip()}
        payload.results = [q for q in payload.results if q.qid in keep]
        logger.info(f"[CLI] Filtre qids={sorted(keep)} → {len(payload.results)} questions")

    if not payload.results:
        logger.error("[CLI] Aucune question à traiter (input vide ou filtre trop strict)")
        return 1

    with get_session() as session:
        runner = ChallengeRunner(
            session=session,
            top_k_chunks=args.top_k_chunks,
            top_n_pages=args.top_n_pages,
            context_chunks=args.context_chunks,
            attribution_threshold=args.attribution_threshold,
            retrieval_only=args.retrieval_only,
            decompose=args.decompose,
            retrieval_mode=args.retrieval_mode,
            rerank=args.rerank,
            reranker_model=args.reranker_model,
            reranker_top_k=args.reranker_top_k,
            reranker_batch_size=args.reranker_batch_size,
            constrained=args.constrained_decoding,
            constrained_backend=args.constrained_backend,
        )
        out = runner.run(payload, with_task2=not args.no_task2)

    base = args.input.with_suffix("")
    out_task1 = args.out_task1 or base.parent / f"{base.name}_task1.json"
    out_task1.write_text(
        json.dumps(out.task1.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[CLI] Tâche 1 écrite : {out_task1}")

    if out.task2 is not None:
        out_task2 = args.out_task2 or base.parent / f"{base.name}_task2.json"
        out_task2.write_text(
            json.dumps(out.task2.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[CLI] Tâche 2 écrite : {out_task2}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
