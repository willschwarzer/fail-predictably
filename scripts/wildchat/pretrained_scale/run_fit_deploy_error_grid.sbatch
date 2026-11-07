#!/bin/bash
#SBATCH --job-name=fit_deploy_error_grid
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# Baseline (fit size m, deploy size N) forecast-error grid over the full-pool
# WildChat caches of the eleven pretrained models (paper appendix
# "Forecastability on natural user prompts", subsection "Forecastability of
# pretrained models across scale"). CPU only, no training. Run it after
# scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh has
# written <model>/harmful/per_token_log_p.npz under LOGIT_CACHE_ROOT for every
# model; every model directory found there is used.
#
# The arguments reproduce the paper's run. Its summary.json records the default
# m and N grids, ols_k 10, 200 reps, seed 0 and all three estimators, its grid
# npz files record ceiling_dec 0.5 and a pool of 85,601 prompts per model, and
# the starred cell is the paper's 44/891 training cell.
#
# Usage (from the repo root): sbatch scripts/wildchat/pretrained_scale/run_fit_deploy_error_grid.sbatch

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
LOGIT_CACHE_ROOT="${LOGIT_CACHE_ROOT:-$RESULTS_ROOT/wildchat_fullpool_logits}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/fit_deploy_error_grid_fullpool}"

echo "[host] $(hostname)  [date] $(date)"
command -v "$PYTHON" && "$PYTHON" --version
echo "[run] root=$LOGIT_CACHE_ROOT out=$OUT_DIR"

"$PYTHON" -u scripts/wildchat/pretrained_scale/fit_deploy_error_grid.py \
  --root "$LOGIT_CACHE_ROOT" \
  --out-dir "$OUT_DIR" \
  --ols-k 10 \
  --reps 200 \
  --seed 0 \
  --mark-cell 44,891 \
  --ceiling-dec 0.5
