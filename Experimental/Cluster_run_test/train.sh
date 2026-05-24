#!/usr/bin/env bash
#SBATCH --job-name=e5-finetune
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Script SLURM exécuté sur le cluster GPU Télécom.
# Soumission locale : ./cluster_exec.sh submit
#
set -euo pipefail

echo "=== Job SLURM ${SLURM_JOB_ID:-local} | $(date -Iseconds) ==="
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-n/a}"
echo "Workdir: $(pwd)"

mkdir -p logs models

# Variables injectées à la soumission (cluster_exec.sh) ou via .env.cluster.runtime
if [[ -f .env.cluster.runtime ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env.cluster.runtime
  set +a
fi

# Modules optionnels (adapter selon le cluster)
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda >/dev/null 2>&1 || module load CUDA/12.1 >/dev/null 2>&1 || true
fi

# Environnement Python
if [[ ! -d .venv ]]; then
  echo "→ Création du venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# PyTorch GPU + dépendances (cu128 — RTX 3090 / Ampere+)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "→ Installation PyTorch CUDA..."
  pip install torch --index-url https://download.pytorch.org/whl/cu128
fi
pip install -r requirements.txt

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

FINETUNE_ARGS="${FINETUNE_ARGS:---epochs 3 --batch_size 32 --max_paragraphs_per_doc 15 --k_neg 4 --log_every 10}"
echo "→ Lancement : python run_finetune.py ${FINETUNE_ARGS}"

# shellcheck disable=SC2086
python run_finetune.py ${FINETUNE_ARGS}

echo "=== Terminé | $(date -Iseconds) ==="
