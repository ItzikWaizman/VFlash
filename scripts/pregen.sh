#!/bin/bash
# Mission: pre-generate the target's greedy responses into gen_cache (one GPU per task).
# Local:  bash scripts/pregen.sh [experiments/<name>.json]
# SLURM:  sbatch --array=0-7 --gres=gpu:1 scripts/pregen.sh experiments/baseline.json
# Each array task owns a strided shard and writes its own <uid>.pt files.
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch300/$USER/hf_cache}"

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CONFIG="$1"; shift; else CONFIG="${CONFIG:-experiments/baseline.json}"; fi

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[pregen] shard $SHARD of $NUM_SHARDS"
python -m vflash.data.pregen --config "$CONFIG" --shard "$SHARD" --num-shards "$NUM_SHARDS" "$@"
