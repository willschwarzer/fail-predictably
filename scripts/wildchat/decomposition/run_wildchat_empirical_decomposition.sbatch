#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=wc_empdecomp
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
OUT_PDF="$FIGURES_ROOT/wildchat_empirical_decomposition.pdf"
OUT_JSON="$RESULTS_ROOT/wildchat_empirical_decomposition.json"

"$PYTHON" -u scripts/wildchat/decomposition/wildchat_empirical_decomposition.py \
  --config scripts/wildchat/decomposition/wildchat_decomp_panels.json \
  --m 5000 \
  --N 50000 \
  --k 10 \
  --n-trials 2000 \
  --chunk 100 \
  --delta 0.5 \
  --seed 42 \
  --out-pdf "$OUT_PDF" \
  --out-json "$OUT_JSON"
