#!/usr/bin/env bash
# Point d'entrée local pour piloter le cluster GPU Télécom.
#
# Usage :
#   ./cluster_exec.sh submit [--with-runs] [--with-data] [--] [args sbatch...]
#   ./cluster_exec.sh status
#   ./cluster_exec.sh logs [JOB_ID|-f]
#   ./cluster_exec.sh cancel JOB_ID
#   ./cluster_exec.sh shell
#   ./cluster_exec.sh cmd -- COMMANDE...
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_common.sh
source "$SCRIPT_DIR/cluster_common.sh"

usage() {
  cat <<'EOF'
Usage: cluster_exec.sh COMMANDE [options]

Commandes :
  submit   Sync push + .env runtime + sbatch train.sh
  status   File d'attente SLURM (squeue)
  logs     Affiche logs/slurm-JOB_ID.out (dernier job si ID omis)
  cancel   Annule un job (scancel)
  shell    Ouvre un shell SSH interactif sur le cluster
  cmd      Exécute une commande arbitraire via SSH

Exemples :
  ./cluster_exec.sh submit
  ./cluster_exec.sh submit --with-data
  ./cluster_exec.sh submit -- --export=ALL,FINETUNE_ARGS="--epochs 1 --batch_size 64"
  ./cluster_exec.sh status
  ./cluster_exec.sh logs 12345
  ./cluster_exec.sh logs -f
  ./cluster_exec.sh cancel 12345
  ./cluster_exec.sh shell
  ./cluster_exec.sh cmd -- sinfo
EOF
}

latest_job_id() {
  resolve_remote_dir
  cluster_ssh \
    "ls -t '${CLUSTER_REMOTE_DIR_ABS}/logs/slurm-'*.out 2>/dev/null | head -1 | sed -E 's/.*slurm-([0-9]+)\\.out/\\1/'" \
    || true
}

cmd_submit() {
  local push_args=()
  local sbatch_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-runs|--with-data|--dry-run)
        push_args+=("$1")
        shift
        ;;
      --)
        shift
        sbatch_args+=("$@")
        break
        ;;
      *)
        sbatch_args+=("$1")
        shift
        ;;
    esac
  done

  load_cluster_env
  require_commands rsync ssh

  echo "=== 1/3 Push code ==="
  "$SCRIPT_DIR/cluster_sync_push.sh" "${push_args[@]}"

  echo "=== 2/3 Sync variables d'environnement ==="
  push_runtime_env

  echo "=== 3/3 Soumission SLURM ==="
  resolve_remote_dir

  local -a sbatch_cmd=(
    sbatch
    --chdir="$CLUSTER_REMOTE_DIR_ABS"
  )

  # Surcharges SLURM depuis cluster.env (si définies)
  [[ -n "${SLURM_PARTITION:-}" ]] && sbatch_cmd+=(--partition="$SLURM_PARTITION")
  [[ -n "${SLURM_GPUS:-}" ]]       && sbatch_cmd+=(--gres="gpu:${SLURM_GPUS}")
  [[ -n "${SLURM_MEM:-}" ]]        && sbatch_cmd+=(--mem="$SLURM_MEM")
  [[ -n "${SLURM_TIME:-}" ]]       && sbatch_cmd+=(--time="$SLURM_TIME")
  [[ -n "${SLURM_CPUS:-}" ]]       && sbatch_cmd+=(--cpus-per-task="$SLURM_CPUS")
  [[ -n "${SLURM_JOB_NAME:-}" ]]   && sbatch_cmd+=(--job-name="$SLURM_JOB_NAME")

  sbatch_cmd+=("${sbatch_args[@]}")
  sbatch_cmd+=("train.sh")

  printf '→ '
  printf '%q ' "${sbatch_cmd[@]}"
  echo

  cluster_ssh "$(printf '%q ' "${sbatch_cmd[@]}")"
}

cmd_status() {
  load_cluster_env
  require_commands ssh
  cluster_ssh "squeue -u '${CLUSTER_USER:-$USER}' -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"
}

cmd_logs() {
  load_cluster_env
  require_commands ssh
  resolve_remote_dir

  local job_id="${1:-}"
  local follow=0

  if [[ "$job_id" == "-f" ]]; then
    follow=1
    job_id=""
  elif [[ "${2:-}" == "-f" ]]; then
    follow=1
  fi

  if [[ -z "$job_id" ]]; then
    job_id="$(latest_job_id)"
    if [[ -z "$job_id" ]]; then
      echo "Aucun log slurm trouvé dans ${CLUSTER_REMOTE_DIR_ABS}/logs/" >&2
      exit 1
    fi
    echo "→ Dernier job : $job_id"
  fi

  local log_path="${CLUSTER_REMOTE_DIR_ABS}/logs/slurm-${job_id}.out"
  if [[ "$follow" -eq 1 ]]; then
    cluster_ssh "tail -f '$log_path'"
  else
    cluster_ssh "cat '$log_path'"
  fi
}

cmd_cancel() {
  local job_id="${1:-}"
  if [[ -z "$job_id" ]]; then
    echo "Usage: cluster_exec.sh cancel JOB_ID" >&2
    exit 1
  fi
  load_cluster_env
  require_commands ssh
  cluster_ssh "scancel '$job_id'"
  echo "✔ Job $job_id annulé."
}

cmd_shell() {
  load_cluster_env
  require_commands ssh
  ssh "${CLUSTER_SSH_OPTS[@]}" -t "$CLUSTER_SSH" \
    "bash -lc $(printf '%q' "cd -- ${CLUSTER_REMOTE_DIR} && exec bash -l")"
}

cmd_cmd() {
  if [[ "${1:-}" != "--" ]]; then
    echo "Usage: cluster_exec.sh cmd -- COMMANDE..." >&2
    exit 1
  fi
  shift
  if [[ $# -eq 0 ]]; then
    echo "Commande vide." >&2
    exit 1
  fi
  load_cluster_env
  require_commands ssh
  cluster_ssh "bash -lc $(printf '%q' "cd -- ${CLUSTER_REMOTE_DIR} && $*")"
}

main() {
  local cmd="${1:-}"
  if [[ -z "$cmd" ]]; then
    usage
    exit 1
  fi
  shift

  case "$cmd" in
    submit) cmd_submit "$@" ;;
    status) cmd_status ;;
    logs)   cmd_logs "$@" ;;
    cancel) cmd_cancel "$@" ;;
    shell)  cmd_shell ;;
    cmd)    cmd_cmd "$@" ;;
    -h|--help|help) usage ;;
    *)
      echo "Commande inconnue : $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
