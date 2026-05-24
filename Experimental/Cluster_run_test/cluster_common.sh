#!/usr/bin/env bash
# Fonctions partagées pour la synchro et l'exécution distante.
set -euo pipefail

CLUSTER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_LOCAL_DIR="${CLUSTER_LOCAL_DIR:-$CLUSTER_SCRIPT_DIR}"

# Options SSH (surchargeables via cluster.env : CLUSTER_SSH_EXTRA_OPTS="-o ProxyJump=...")
CLUSTER_SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

load_cluster_env() {
  local env_file="${CLUSTER_ENV_FILE:-$CLUSTER_SCRIPT_DIR/cluster.env}"
  if [[ ! -f "$env_file" ]]; then
    echo "Fichier de config introuvable : $env_file" >&2
    echo "Copiez cluster.env.example → cluster.env et renseignez CLUSTER_SSH_TARGET." >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  set -a
  # Tolère cluster.env édité sous Windows (CRLF)
  source <(tr -d '\r' <"$env_file")
  set +a

  : "${CLUSTER_REMOTE_DIR:?CLUSTER_REMOTE_DIR requis dans cluster.env}"

  # Cible SSH : alias ~/.ssh/config (recommandé) OU user@host explicite
  if [[ -n "${CLUSTER_SSH_TARGET:-}" ]]; then
    CLUSTER_SSH="$CLUSTER_SSH_TARGET"
  elif [[ -n "${CLUSTER_USER:-}" && -n "${CLUSTER_HOST:-}" ]]; then
    CLUSTER_SSH="${CLUSTER_USER}@${CLUSTER_HOST}"
  else
    echo "Définir CLUSTER_SSH_TARGET=gpu (alias SSH) ou CLUSTER_USER + CLUSTER_HOST." >&2
    exit 1
  fi

  if [[ -n "${CLUSTER_SSH_EXTRA_OPTS:-}" ]]; then
    # shellcheck disable=SC2206
    CLUSTER_SSH_OPTS+=($CLUSTER_SSH_EXTRA_OPTS)
  fi

  export RSYNC_RSH="ssh ${CLUSTER_SSH_OPTS[*]}"
}

require_commands() {
  local missing=()
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing+=("$cmd")
    fi
  done
  if ((${#missing[@]} > 0)); then
    echo "Commandes manquantes : ${missing[*]}" >&2
    echo "Installez OpenSSH + rsync (Git Bash / WSL / Linux)." >&2
    exit 1
  fi
}

cluster_ssh() {
  ssh "${CLUSTER_SSH_OPTS[@]}" "$CLUSTER_SSH" "$@"
}

# Résout ~/... en chemin absolu sur le cluster (requis pour sbatch --chdir, logs, etc.)
resolve_remote_dir() {
  if [[ -n "${CLUSTER_REMOTE_DIR_ABS:-}" ]]; then
    return 0
  fi
  CLUSTER_REMOTE_DIR_ABS="$(
    cluster_ssh "bash -lc $(printf '%q' "cd -- ${CLUSTER_REMOTE_DIR} && pwd")"
  )"
  if [[ -z "$CLUSTER_REMOTE_DIR_ABS" || "$CLUSTER_REMOTE_DIR_ABS" != /* ]]; then
    echo "Impossible de résoudre CLUSTER_REMOTE_DIR='${CLUSTER_REMOTE_DIR}' sur le cluster." >&2
    exit 1
  fi
}

remote_mkdir() {
  cluster_ssh "bash -lc $(printf '%q' "mkdir -p -- ${CLUSTER_REMOTE_DIR}")"
}

default_rsync_excludes() {
  cat <<'EOF'
--exclude=.venv/
--exclude=venv/
--exclude=__pycache__/
--exclude=*.pyc
--exclude=.git/
--exclude=.cache/
--exclude=.ipynb_checkpoints/
--exclude=node_modules/
--exclude=slurm-*.out
--exclude=slurm-*.err
--exclude=cluster.env
EOF
}

rsync_push_dir() {
  local src="$1"
  local dest="$2"
  shift 2
  local -a extra=( "$@" )

  remote_mkdir
  rsync -avz --delete \
    "${extra[@]}" \
    "$src/" "${CLUSTER_SSH}:${dest}/"
}

rsync_pull_dir() {
  local src="$1"
  local dest="$2"
  shift 2
  local -a extra=( "$@" )

  mkdir -p "$dest"
  rsync -avz \
    "${extra[@]}" \
    "${CLUSTER_SSH}:${src}/" "$dest/"
}

push_runtime_env() {
  local remote_env="${CLUSTER_REMOTE_DIR}/.env.cluster.runtime"
  local tmp
  tmp="$(mktemp)"

  {
    echo "# Généré localement par cluster_exec.sh submit — $(date -Iseconds)"
    if [[ -f "$CLUSTER_SCRIPT_DIR/cluster.env" ]]; then
      grep -E '^(HF_TOKEN|FINETUNE_ARGS|SLURM_)=' "$CLUSTER_SCRIPT_DIR/cluster.env" || true
    fi
    if [[ -f "$CLUSTER_SCRIPT_DIR/../../backend/.env" ]]; then
      grep -E '^HF_TOKEN=' "$CLUSTER_SCRIPT_DIR/../../backend/.env" || true
    fi
  } >"$tmp"

  rsync -avz "$tmp" "${CLUSTER_SSH}:${remote_env}"
  rm -f "$tmp"
}
