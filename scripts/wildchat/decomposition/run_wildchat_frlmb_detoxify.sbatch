#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=wildchat_detoxify
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"

TAG=${TAG:-wildchat_frlmb_tail_fit_quality_${TAG_DATE:-$(date +%Y%m%d)}}
OUTDIR=${OUTDIR:-"$RESULTS_ROOT/$TAG/detoxify_all"}
mkdir -p "$OUTDIR"

# Reads the WildChat arrow shards under $WILDCHAT_DATASET (default data/wildchat).
exec "$PYTHON" -u scripts/wildchat/decomposition/wildchat_score_detoxify.py \
  --out-dir "$OUTDIR" \
  --language-source top_level \
  --language English \
  --logit-eps 1e-6
