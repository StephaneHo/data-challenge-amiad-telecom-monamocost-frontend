from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.challenge import ChallengeInput, ChallengeOutput, ChallengeRunner
from database.db import get_db


app = FastAPI(title="Challenge RAG EvalLLM 2026", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Système"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "monamo-cost-rag"}


def get_runner(db: Session = Depends(get_db)) -> ChallengeRunner:
    return ChallengeRunner(session=db)


@app.post(
    "/challenge/run",
    tags=["Challenge"],
    response_model=ChallengeOutput,
    summary="Exécute Tâche 1 (+ Tâche 2) sur un payload challenge",
)
async def challenge_run(
    payload: ChallengeInput,
    with_task2: bool = True,
    runner: ChallengeRunner = Depends(get_runner),
) -> ChallengeOutput:
    """
    Prend un JSON au format challenge EvalLLM 2026 (avec questions seules),
    exécute le retrieval + génération (Tâche 1) puis l'attribution (Tâche 2
    si `with_task2=True`), et retourne les deux runs dans un seul payload.
    """
    return runner.run(payload, with_task2=with_task2)
