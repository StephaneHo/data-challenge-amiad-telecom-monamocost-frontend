#!/usr/bin/env bash
# Envoie Cluster_run_test vers le cluster (rsync/SSH).
#
# Usage :
#   ./cluster_sync_push.sh [--with-runs] [--with-data] [--dry-run]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_common.sh
source "$SCRIPT_DIR/cluster_common.sh"

WITH_RUNS=0
WITH_DATA=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: cluster_sync_push.sh [options]

Options:
  --with-runs   Inclure runs/ et logs/ locaux
  --with-data   Inclure CLUSTER_EXTRA_DATA_DIR (voir cluster.env)
  --dry-run     Afficher la commande rsync sans l'exécuter
  -h, --help    Afficher cette aide
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-runs) WITH_RUNS=1; shift ;;
    --with-data) WITH_DATA=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "Option inconnue : $1" >&2; usage; exit 1 ;;
  esac
done

require_commands rsync ssh
load_cluster_env

mapfile -t EXCLUDES < <(default_rsync_excludes)

# Exclusions par défaut : gros dossiers / artefacts
EXCLUDES+=(
  --exclude=models/
  --exclude=runs/
)

if [[ "$WITH_RUNS" -eq 0 ]]; then
  EXCLUDES+=(--exclude=logs/)
fi

# DATA locale : incluse par défaut (df_paragraphe_final.csv nécessaire au train)
if [[ "$WITH_DATA" -eq 0 ]]; then
  EXCLUDES+=(--exclude=DATA/Corpus_raw/)
fi

echo "→ Push ${CLUSTER_LOCAL_DIR} → ${CLUSTER_SSH}:${CLUSTER_REMOTE_DIR}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'DRY-RUN rsync'
  printf ' %q' "${EXCLUDES[@]}"
  printf ' %q/' "$CLUSTER_LOCAL_DIR"
  printf ' %q\n' "${CLUSTER_SSH}:${CLUSTER_REMOTE_DIR}/"
  exit 0
fi

rsync_push_dir "$CLUSTER_LOCAL_DIR" "$CLUSTER_REMOTE_DIR" "${EXCLUDES[@]}"

if [[ "$WITH_DATA" -eq 1 ]]; then
  EXTRA="${CLUSTER_EXTRA_DATA_DIR:-$SCRIPT_DIR/../../Experimental/DATA}"
  if [[ -d "$EXTRA" ]]; then
    echo "→ Push données extra : $EXTRA"
    remote_mkdir
    rsync -avz \
      --exclude='Corpus_raw/' \
      --exclude='*.pdf' \
      "$EXTRA/" "${CLUSTER_SSH}:${CLUSTER_REMOTE_DIR}/DATA_EXTRA/"
  else
    echo "⚠ CLUSTER_EXTRA_DATA_DIR introuvable : $EXTRA" >&2
  fi
fi

echo "✔ Synchronisation terminée."
