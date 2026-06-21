#!/bin/bash
#SBATCH --job-name=vflash-pregen
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pregen_%A_%a.out
# Usage:  sbatch scripts/pregen_remote.sh remote_pregen/chunk_00.jsonl
# One array task per GPU; the 8 tasks split the chunk manifest into 8 strided shards.
# Already-cached uids and samples whose video is missing are skipped automatically,
# so this is safe to re-run and safe to run on a partially-transferred chunk.
set -euo pipefail

MANIFEST="${1:?usage: sbatch scripts/pregen_remote.sh <chunk_manifest.jsonl>}"

# --- cluster env ----------------------------------------------------------
REPO_ROOT="${VFLASH_ROOT:-/scratch300/itzikwaizman/VFlash}"
export HF_HOME="${HF_HOME:-/scratch300/itzikwaizman/cache}"
CONDA_ENV_PATH="${VFLASH_CONDA_ENV:-/scratch300/itzikwaizman/conda_envs/unlearning}"
# find and source conda init script
CONDA_BASE=$(conda info --base 2>/dev/null) || CONDA_BASE=$(dirname $(dirname $(which conda)))
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV_PATH"
# --------------------------------------------------------------------------

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
mkdir -p logs

echo "[pregen_remote] host=$(hostname) task=$SLURM_ARRAY_TASK_ID manifest=$MANIFEST"
python -u -m vflash.data.pregen \
    --config experiments/baseline.json \
    --manifest "$MANIFEST" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards 8 \
    --batch-size 1 \
    --num-workers 8
