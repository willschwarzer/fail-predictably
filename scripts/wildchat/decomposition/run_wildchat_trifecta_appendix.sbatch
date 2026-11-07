#!/bin/bash
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=wildchat_trifecta
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
FIGURES_ROOT="${FIGURES_ROOT:-$REPO_ROOT/artifacts/figures}"
export RESULTS_ROOT

# The panel config names its score CSVs relative to $RESULTS_ROOT, under the paper's
# run tags: run the scorers with TAG_DATE=20260503 for detoxify and harmful (flagged),
# and TAG_DATE=20260504 for assistant length and NLL.
OUT_PDF="$FIGURES_ROOT/wildchat_trifecta_appendix.pdf"
OUT_PNG="$FIGURES_ROOT/wildchat_trifecta_appendix.png"

"$PYTHON" -u scripts/wildchat/decomposition/wildchat_length_hazard.py \
  --multi-panel-config scripts/wildchat/decomposition/wildchat_trifecta_panels.json \
  --out "$OUT_PDF"

"$PYTHON" -u scripts/wildchat/decomposition/wildchat_length_hazard.py \
  --multi-panel-config scripts/wildchat/decomposition/wildchat_trifecta_panels.json \
  --out "$OUT_PNG"
