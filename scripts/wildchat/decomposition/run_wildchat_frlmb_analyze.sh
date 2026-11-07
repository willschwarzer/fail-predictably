#!/bin/bash
# Run the tail-fit analyzer on the Detoxify and harmful-token score cells (the
# without-replacement totals of the WildChat decomposition appendix).
# Cheap (CPU-only); intended to be run on a node, not the login host.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TAG=${TAG:-wildchat_frlmb_tail_fit_quality_${TAG_DATE:-$(date +%Y%m%d)}}
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
# The tag's directory, holding one subdirectory per scored cell.
TAG_DIR=${TAG_DIR:-"$RESULTS_ROOT/$TAG"}

DETOX_DIR="${TAG_DIR}/detoxify_all"
HARM_FLAGGED_DIR="${TAG_DIR}/harmful_flagged"
HARM_ALL_DIR="${TAG_DIR}/harmful_all"

if [[ -f "${DETOX_DIR}/wildchat_detoxify_scores.csv" ]]; then
  echo "[run] Cell A: detoxify_all (FRLMB-canonical gumbel_prob on raw toxicity)"
  "$PYTHON" -u scripts/wildchat/decomposition/wildchat_frlmb_analyze.py \
    --scores-csv "${DETOX_DIR}/wildchat_detoxify_scores.csv" \
    --score-column toxicity \
    --score-mode gumbel_prob \
    --out-dir "${DETOX_DIR}" \
    --cell-name detoxify_all \
    --n-permutations 100 \
    --max-extrapolated-ranks-per-perm 200
else
  echo "[skip] ${DETOX_DIR}/wildchat_detoxify_scores.csv not found"
fi

for HD_DIR in "${TAG_DIR}"/harmful_*; do
  [[ -d "${HD_DIR}" ]] || continue
  HD_NAME="$(basename "${HD_DIR}")"
  if [[ -f "${HD_DIR}/wildchat_harmful_scores.csv" && -f "${HD_DIR}/manifest.json" ]]; then
    echo "[run] Cell B (${HD_NAME}): gumbel_log_prob on log_p_harmful"
    "$PYTHON" -u scripts/wildchat/decomposition/wildchat_frlmb_analyze.py \
      --scores-csv "${HD_DIR}/wildchat_harmful_scores.csv" \
      --score-column log_p_harmful \
      --score-mode gumbel_log_prob \
      --out-dir "${HD_DIR}" \
      --cell-name "${HD_NAME}" \
      --n-permutations 100 \
      --max-extrapolated-ranks-per-perm 200
  else
    echo "[skip] Cell B (${HD_NAME}) -- CSV missing or scoring still in progress"
  fi
done

echo "[done] analysis run complete; outputs under ${TAG_DIR}"
