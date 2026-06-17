#!/bin/bash
# Local (non-SLURM) setup: download models + LLaVA-Video-178K subset, build manifest.
# Activate your conda/venv FIRST. Usage: bash scripts/local/setup.sh [experiments/<name>.json]
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

CONFIG="${1:-experiments/baseline.json}"
python -m vflash.data.setup --config "$CONFIG"
