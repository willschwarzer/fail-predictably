#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=wildchat_assistant_length
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"

TAG=${TAG:-wildchat_frlmb_tail_fit_quality_${TAG_DATE:-$(date +%Y%m%d)}}
OUTDIR=${OUTDIR:-"$RESULTS_ROOT/$TAG/assistant_length_all"}
mkdir -p "$OUTDIR"

# Reads the WildChat arrow shards under $WILDCHAT_DATASET (default data/wildchat).
"$PYTHON" -u scripts/wildchat/decomposition/wildchat_score_assistant_length.py \
  --out-dir "$OUTDIR" \
  --language-source top_level \
  --language English

# Run the analyzer twice -- once with identity (raw length, OLS extrapolates
# in length space; under-pred would mean tail is heavier than Gumbel in
# length-space) and once with score_mode='log' (log-length, OLS in log-space;
# under-pred would mean log-length tail is heavier than Gumbel, i.e., length
# is heavier than log-Pareto).
for SCORE_COL in asst_char_len asst_word_count; do
  for SCORE_MODE in identity log; do
    SUBDIR="${OUTDIR}/analysis_${SCORE_COL}_${SCORE_MODE}"
    mkdir -p "${SUBDIR}"
    echo "[run] ${SCORE_COL} score_mode=${SCORE_MODE}"
    "$PYTHON" -u scripts/wildchat/decomposition/wildchat_frlmb_analyze.py \
      --scores-csv "$OUTDIR/wildchat_assistant_length_scores.csv" \
      --score-column "${SCORE_COL}" \
      --score-mode "${SCORE_MODE}" \
      --out-dir "${SUBDIR}" \
      --cell-name "assistant_length_all_${SCORE_COL}_${SCORE_MODE}" \
      --n-permutations 100 \
      --max-extrapolated-ranks-per-perm 200
  done
done
