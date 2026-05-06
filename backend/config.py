from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Valeurs par défaut
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag"
    REDIS_URL: Optional[str] = None

    # Embeddings
    # si aucune valeur n'est fournie pas défaut dans .env, c'est cette valeur qui sera utilisée
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Collecte ArXiv
    ARXIV_CATEGORIES: list[str] = [
        "cs.AI",
        "cs.LG",
        "cs.CV",
        "cs.CL",
        "cs.RO",
        "cs.CR",
        "stat.ML",
    ]
    ARXIV_MAX_RESULTS: int = 500

    # RAG
    RAG_TOP_K: int = 15  # récupère les 15 meilleurs résultats
    RAG_CHUNK_SIZE: int = 512  # taille des morceaux de texte (chunk): 512 tokens
    RAG_CHUNK_OVERLAP: int = 64  # recouvrement des chunks, éviter  que les infromations importantes soit coupées en deux
    RAG_TEMPERATURE: float = 0.2  # faible température => strict

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
