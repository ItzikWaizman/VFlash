#!/bin/bash
# Local speculative-decoding eval (chain or DDtree) vs AR. Activate your conda/venv FIRST.
# Usage: bash scripts/local/infer.sh [experiments/<name>.json] [extra --overrides ...]
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

if [ $# -ge 1 ] && [[ "$1" != --* ]]; then CONFIG="$1"; shift; else CONFIG="${CONFIG:-experiments/baseline.json}"; fi
python -m vflash.evaluate --config "$CONFIG" "$@"
