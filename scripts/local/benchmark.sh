#!/bin/bash
# Local VideoDetailCaption benchmark (speedup + M, lossless check). Activate your conda/venv FIRST.
# Usage: bash scripts/local/benchmark.sh [experiments/<name>.json] [extra --overrides ...]
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CONFIG="$1"; shift; else CONFIG="${CONFIG:-experiments/baseline.json}"; fi
python -m vflash.benchmark --config "$CONFIG" "$@"
