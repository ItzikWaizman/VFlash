#!/bin/bash
# Local pre-generation across N GPUs (one process per GPU), no SLURM.
# Activate your conda/venv FIRST.
# Usage: bash scripts/local/pregen.sh [experiments/<name>.json] [NUM_GPUS]
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CONFIG="$1"; shift; else CONFIG="${CONFIG:-experiments/baseline.json}"; fi
N="${NPROC:-$(nvidia-smi -L | grep -c .)}"
echo "[pregen] launching $N shards over $N GPUs"

pids=()
for ((i = 0; i < N; i++)); do
  CUDA_VISIBLE_DEVICES="$i" python -m vflash.data.pregen --config "$CONFIG" --shard "$i" --num-shards "$N" "$@" &
  pids+=($!)
done
wait "${pids[@]}"
echo "[pregen] all shards done"
