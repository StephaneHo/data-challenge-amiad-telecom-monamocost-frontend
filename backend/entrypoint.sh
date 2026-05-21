#!/usr/bin/env bash
# backend/entrypoint.sh — démarrage du conteneur backend
set -e

echo "[entrypoint] Attente que PostgreSQL soit prêt..."
wait-for-it db:5432 --timeout=60 --strict -- echo "[entrypoint] PostgreSQL disponible"

echo "[entrypoint] Application des migrations Alembic..."
uv run alembic upgrade head

echo "[entrypoint] Démarrage de l'API FastAPI sur :8000"
exec uv run uvicorn api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info
