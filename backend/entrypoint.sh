#!/usr/bin/env bash
# backend/entrypoint.sh

# si une commande échoue, stoppe immédiatement
set -e

# On attend POSTGRES
echo "[entrypoint] On attend que PostgreSQL soit prete"
wait-for-it db:5432 --timeout=60 --strict -- echo "[entrypoint] PostgreSQL disponible"

echo "[entrypoint] On lance les migrations Alambic"
alembic upgread head

echo "[entrypoint] On démarre l'API FASTAPI"
exec uvicorn api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info

