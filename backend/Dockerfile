# Image backend Mon Amo Cost — pipeline RAG EvalLLM 2026
# Inclut tout ce qu'il faut pour reproduire l'expérience sans installation locale :
# Python 3.11, deps uv, Tesseract OCR (fra+eng), Poppler, et le code applicatif.

FROM python:3.11-slim AS base

# ─────────────────── Outils système ───────────────────
# - tesseract-ocr-fra : pour l'OCR en français des PDFs scannés
# - poppler-utils : pour la conversion PDF→image (pdf2image)
# - libpq-dev : pour psycopg2 / pgvector
# - curl, git : utilitaires divers
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    bash \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    poppler-utils \
    && curl -sSL https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
       -o /usr/local/bin/wait-for-it \
    && chmod +x /usr/local/bin/wait-for-it \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ─────────────────── uv (gestionnaire deps) ───────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ─────────────────── Deps Python (cache layer) ───────────────────
# On copie d'abord uniquement les fichiers de deps pour bénéficier du cache Docker
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync

# ─────────────────── Code applicatif ───────────────────
COPY . .

# Le code ingère les PDFs depuis ../Experimental/DATA/Corpus_raw — monté en volume
# en runtime via docker-compose (non baked dans l'image, conformément au règlement
# qui interdit la redistribution du corpus).

RUN chmod +x entrypoint.sh

# ─────────────────── Variables d'environnement par défaut ───────────────────
# Sur Linux/Docker, Tesseract et Poppler sont dans le PATH par défaut.
# On surcharge donc les paths Windows héritées de .env.
ENV TESSERACT_CMD="" \
    TESSDATA_DIR="" \
    POPPLER_PATH="" \
    OCR_LANG="fra" \
    HF_HUB_OFFLINE="0"

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
