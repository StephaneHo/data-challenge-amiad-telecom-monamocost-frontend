from __future__ import annotations

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, Query, HTTPException
from database.db import get_db
from database.models import Paper, Category
from sqlalchemy import extract
from sqlalchemy.orm import Session
from rag.rag_engine import RAGEngine
from pydantic import BaseModel, Field


app = FastAPI(title="Veille scientifique", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)
    year_from: Optional[int] = None
    categories: Optional[list[str]] = None


"""
Pour l'instant, tout le monde peut appler l'API
On restreindra plus tard
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
On définit une fonction qui nous aide à convertir le résultat de la requête (objet SQLAlchemy) en object JSON
pour l'instant, on n'interroge que arxiv, venue est toujours à None
"""


def _paper_convert_to_json(p: Paper):
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "abstract": p.abstract,
        "published_at": p.published_at,
        "citation_count": p.citation_count,
        "venue": p.venue,
        "pdf_url": p.pdf_url,
    }


@app.get("/health", tags=["Système"])
async def health():
    return {"status": "ok", "service": "am-backend"}


"""
On utilise le mécanisme d'injection de dépendances
db: le nom de la variable dans la fonction
Session: L'annotation de type dit que db est une session SQLAlchemy
Depends(get_db): Dit à FastAPI appelle get_db() pour obtenir la valeur de db
offset sert à la pagination
"""


@app.get("/papers", tags=["Papers"])
async def list_papers(
    category: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Paper)
    if category:
        q = q.join(Paper.categories).filter(Category.code == category)
    if year:
        q = q.filter(extract("year", Paper.published_at) == year)
    papers = q.order_by(Paper.published_at.desc()).offset(offset).limit(limit).all()
    return [_paper_convert_to_json(p) for p in papers]


@app.get("/papers/{arxiv_id}", tags=["Papers"])
async def get_paper(arxiv_id: str, db: Session = Depends(get_db)):
    paper = db.get(Paper, arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    return _paper_convert_to_json(paper)


def get_rag(db: Session = Depends(get_db)) -> RAGEngine:
    return RAGEngine(session=db)


@app.post("/rag/search", tags=["RAG"])
async def rag_search(
    req: SearchRequest,
    engine: RAGEngine = Depends(get_rag),
):
    """Recherche sémantique et génération de réponse."""
    r = engine.answer(
        query=req.query,
        top_k=req.top_k,
        year_from=req.year_from,
    )
    return {
        "answer": r.answer,
        "references": r.references,
        "tokens_used": r.tokens_used,
    }
