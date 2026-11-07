#!/bin/bash
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH --time=0-01:30:00
#SBATCH --array=0-13
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --job-name=wc_ksweep
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"

# 14 jobs = 7 k values * 2 scores
K_VALUES=(5 10 20 50 100 200 500)
SCORES=(length mean_nll)

NK=${#K_VALUES[@]}
NS=${#SCORES[@]}

idx=$SLURM_ARRAY_TASK_ID
score_idx=$(( idx / NK ))
k_idx=$(( idx % NK ))
K=${K_VALUES[$k_idx]}
SCORE=${SCORES[$score_idx]}

# Score CSVs from the scorers run with TAG_DATE=20260504 (assistant length and NLL).
case "$SCORE" in
  length)
    CSV="$RESULTS_ROOT/wildchat_frlmb_tail_fit_quality_20260504/assistant_length_all/wildchat_assistant_length_scores.csv"
    COLUMN=asst_char_len
    SCORE_MODE=identity
    ;;
  mean_nll)
    CSV="$RESULTS_ROOT/wildchat_frlmb_tail_fit_quality_20260504/assistant_nll_all_subsample0.33/wildchat_nll_scores.csv"
    COLUMN=mean_nll
    SCORE_MODE=identity
    ;;
  *)
    echo "Unknown score: $SCORE" >&2
    exit 1
    ;;
esac

OUT_DIR="$RESULTS_ROOT/wildchat_k_sweep/${SCORE}_k${K}"
mkdir -p "$OUT_DIR"

echo "[run] score=$SCORE k=$K out=$OUT_DIR"

"$PYTHON" -u scripts/wildchat/decomposition/wildchat_frlmb_analyze.py \
  --scores-csv "$CSV" \
  --score-column "$COLUMN" \
  --score-mode "$SCORE_MODE" \
  --out-dir "$OUT_DIR" \
  --cell-name "${SCORE}_k${K}" \
  --n-grid 5000 \
  --m-multipliers 10 \
  --top-k "$K" \
  --plotting weibull \
  --n-permutations 100 \
  --seed 11

# Once all 14 tasks have finished, draw the figure (CPU, a few minutes; other flags at their defaults):
#   python scripts/wildchat/decomposition/wildchat_k_sweep_figure.py \
#     --sweep-root "${RESULTS_ROOT:-artifacts/results}/wildchat_k_sweep" \
#     --score-meta scripts/wildchat/decomposition/wildchat_k_sweep_meta.json \
#     --out-pdf "${FIGURES_ROOT:-artifacts/figures}/wildchat_k_sweep.pdf" \
#     --out-json "${RESULTS_ROOT:-artifacts/results}/wildchat_k_sweep_figure.json"
