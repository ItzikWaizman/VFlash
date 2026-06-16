#!/bin/bash
# Mission: train the VFlash drafter (8 GPUs, DDP).
# Usage: bash scripts/train.sh [experiments/<name>.json] [extra --overrides ...]
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch300/$USER/hf_cache}"

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CONFIG="$1"; shift; else CONFIG="${CONFIG:-experiments/baseline.json}"; fi

# Use NPROC if set, else the GPUs SLURM gave us, else CUDA_VISIBLE_DEVICES, else nvidia-smi, else 1.
if [ -z "${NPROC:-}" ]; then
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NPROC="$SLURM_GPUS_ON_NODE"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L | grep -c .)"
  else
    NPROC=1
  fi
fi
echo "[train] using NPROC=$NPROC GPUs"

torchrun --standalone --nproc_per_node "$NPROC" -m vflash.train --config "$CONFIG" "$@"
