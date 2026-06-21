#!/bin/bash
#SBATCH --job-name=vflash-fetch
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch300/itzikwaizman/VFlash/logs/fetch_%j.out
set -euo pipefail

# --- env ------------------------------------------------------------------
export HOME=/scratch300/itzikwaizman
export HF_HOME=/scratch300/itzikwaizman/VFlash/hf_cache
export PYTHONPATH=/scratch300/itzikwaizman/VFlash

source /scratch300/itzikwaizman/conda_envs/unlearning/../../miniconda3/etc/profile.d/conda.sh \
    2>/dev/null || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate /scratch300/itzikwaizman/conda_envs/unlearning
# --------------------------------------------------------------------------

cd /scratch300/itzikwaizman/VFlash
mkdir -p logs

echo "[fetch] starting on $(hostname) at $(date)"
python scripts/fetch_subset_videos_hf.py
echo "[fetch] done at $(date)"
