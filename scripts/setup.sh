#!/bin/bash
# Mission: download models + LLaVA-Video-178K subset and build the manifest.
# Usage: bash scripts/setup.sh [experiments/<name>.json]
set -euo pipefail

source /scratch300/$USER/env.sh
module load anaconda
conda activate /scratch300/$USER/conda_envs/unlearning

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch300/$USER/hf_cache}"

CONFIG="${1:-experiments/baseline.json}"
python -m vflash.data.setup --config "$CONFIG"
