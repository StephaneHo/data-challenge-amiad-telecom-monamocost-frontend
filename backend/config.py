from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base de données
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/am-rag"
    REDIS_URL: Optional[str] = None

    # Embeddings
    # Multilingue FR avec convention de préfixes :
    #   - documents : "passage: <texte>"
    #   - requêtes  : "query: <texte>"
    # Dimensions par modèle : e5-small=384, e5-base=768, e5-large=1024.
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    EMBEDDING_DIM: int = 768

    # Corpus / Ingestion PDF
    CORPUS_DIR: str = "../Experimental/DATA/Corpus_raw"
    # Si défini, force le chemin de l'exécutable Tesseract (Windows).
    # Sur Linux/Docker, laisser vide — `tesseract` est dans le PATH.
    TESSERACT_CMD: str = ""
    # Si défini, force le dossier contenant les fichiers `*.traineddata`
    # (utile sous Windows quand on n'a pas les droits sur Program Files).
    TESSDATA_DIR: str = ""
    # Si défini, force le chemin du dossier `bin` de Poppler (Windows).
    POPPLER_PATH: str = ""
    OCR_LANG: str = "fra"
    OCR_DPI: int = 300
    # Si une page extraite via pdfplumber a moins de N caractères, on bascule en OCR.
    OCR_FALLBACK_MIN_CHARS: int = 50

    # Chunking
    CHUNK_MIN_CHARS: int = 50   # paragraphes plus courts ignorés (bruit OCR/headers)
    CHUNK_MAX_CHARS: int = 1500 # paragraphes plus longs découpés par phrase

    # RAG
    RAG_TOP_K: int = 15
    RAG_TEMPERATURE: float = 0.2
    # Mode de retrieval : "dense" (e5 cosinus), "bm25" (Postgres ts_rank), "hybrid" (RRF fusion).
    # Hybride améliore les questions contenant des entités exactes rares
    # (acronymes, codes, noms propres) que le dense pur peut manquer.
    RETRIEVAL_MODE: str = "hybrid"

    # LLM
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"

    # LLM local open-source via API OpenAI-compatible (vLLM / llama-cpp-python / Ollama).
    # Si LLM_BASE_URL est défini, le client OpenAI pointe sur ce serveur local au lieu
    # d'api.openai.com. LLM_API_KEY peut rester vide (la plupart des serveurs locaux
    # acceptent n'importe quelle valeur). Permet d'exécuter Mistral-7B-Instruct, Qwen2.5-7B,
    # etc. en local pour le bonus frugalité du challenge.
    # Exemples :
    #   LLM_BASE_URL=http://localhost:8000/v1   LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
    #   LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:7b-instruct   (Ollama)
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    # Beaucoup de serveurs locaux (Ollama notamment) ne supportent pas le mode
    # `response_format={"type": "json_object"}` d'OpenAI. Quand True, on bascule sur un
    # prompt strict + parsing tolérant (extraction du premier {...} du texte).
    LLM_JSON_VIA_PROMPT: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
