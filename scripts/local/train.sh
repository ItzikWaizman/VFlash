#!/bin/bash
# Local training via torchrun on N GPUs, no SLURM. Activate your conda/venv FIRST.
# Usage: bash scripts/local/train.sh [experiments/<name>.json] [extra --overrides ...]
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CONFIG="$1"; shift; else CONFIG="${CONFIG:-experiments/baseline.json}"; fi
N="${NPROC:-$(nvidia-smi -L | grep -c .)}"
echo "[train] using $N GPUs"

torchrun --standalone --nproc_per_node "$N" -m vflash.train --config "$CONFIG" "$@"
