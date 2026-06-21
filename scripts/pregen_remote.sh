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

# --- cluster env: EDIT THESE for your cluster -----------------------------
REPO_ROOT="${VFLASH_ROOT:-$HOME/VFlash}"      # repo root on the cluster
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"   # model weights cache (pulled from HF on first run)
# Activate your environment (pick whichever applies on the cluster):
#   module load cuda/12.1
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate vflash
source "${VFLASH_ENV:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${VFLASH_CONDA_ENV:-vflash}"
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
