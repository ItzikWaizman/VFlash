#!/bin/bash
# Mission: evaluate speculative decoding (chain or DDtree) vs AR baseline.
# Usage: bash scripts/infer.sh [experiments/<name>.json] [extra --overrides ...]
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch300/$USER/hf_cache}"

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CONFIG="$1"; shift; else CONFIG="${CONFIG:-experiments/baseline.json}"; fi

python -m vflash.evaluate --config "$CONFIG" "$@"
