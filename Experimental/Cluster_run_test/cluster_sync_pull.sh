#!/usr/bin/env bash
# Récupère les résultats du cluster vers la machine locale.
#
# Usage :
#   ./cluster_sync_pull.sh [--only-runs | --only-artifacts | --all]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_common.sh
source "$SCRIPT_DIR/cluster_common.sh"

MODE="artifacts"

usage() {
  cat <<'EOF'
Usage: cluster_sync_pull.sh [mode]

Modes (un seul) :
  --only-runs        Récupère runs/ et logs/
  --only-artifacts   Récupère models/ (défaut)
  --all              Récupère models/, runs/, logs/ et slurm-*.out/err
  -h, --help         Afficher cette aide
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-runs)      MODE="runs"; shift ;;
    --only-artifacts) MODE="artifacts"; shift ;;
    --all)            MODE="all"; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Option inconnue : $1" >&2; usage; exit 1 ;;
  esac
done

require_commands rsync ssh
load_cluster_env
resolve_remote_dir

pull_if_exists() {
  local remote_sub="$1"
  local local_sub="$2"
  if cluster_ssh "test -d '${CLUSTER_REMOTE_DIR_ABS}/${remote_sub}'"; then
    echo "→ Pull ${remote_sub}/"
    rsync_pull_dir "${CLUSTER_REMOTE_DIR_ABS}/${remote_sub}" "${CLUSTER_LOCAL_DIR}/${local_sub}"
  else
    echo "  (absent sur le cluster : ${remote_sub}/)"
  fi
}

case "$MODE" in
  runs)
    pull_if_exists "runs" "runs"
    pull_if_exists "logs" "logs"
    ;;
  artifacts)
    pull_if_exists "models" "models"
    ;;
  all)
    pull_if_exists "models" "models"
    pull_if_exists "runs" "runs"
    pull_if_exists "logs" "logs"
    mkdir -p "$CLUSTER_LOCAL_DIR"
    rsync -avz \
      --include='slurm-*.out' --include='slurm-*.err' \
      --exclude='*' \
      "${CLUSTER_SSH}:${CLUSTER_REMOTE_DIR_ABS}/" "$CLUSTER_LOCAL_DIR/"
    ;;
esac

echo "✔ Récupération terminée."
