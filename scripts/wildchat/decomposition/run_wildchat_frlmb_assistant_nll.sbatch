#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=wildchat_assistant_nll
# Hardware: one GPU with at least 32 GB and compute capability 8.0 or newer (the micro-batch scales with the detected VRAM).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
export TOKENIZERS_PARALLELISM=false

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
# This scorer needs full-sequence logits over the whole assistant span, not
# just last-position. Memory cost is much higher than the harmful scorer.
if [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=8
elif (( VRAM_MB < 30000 )); then MICRO_BATCH=8
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=32
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=64
else MICRO_BATCH=96; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"

TAG=${TAG:-wildchat_frlmb_tail_fit_quality_${TAG_DATE:-$(date +%Y%m%d)}}
SLICE=${SLICE:-all}
SUBSAMPLE_FRACTION=${SUBSAMPLE_FRACTION:-0.33}
SUBSAMPLE_SEED=${SUBSAMPLE_SEED:-11}
MAX_LEN=${MAX_LEN:-1024}
if [[ "${SLICE}" == "flagged" ]]; then
  SUBDIR=assistant_nll_flagged
  FLAG_ARG="--require-flagged"
else
  SUBDIR=assistant_nll_all_subsample${SUBSAMPLE_FRACTION}
  FLAG_ARG=""
fi
OUTDIR=${OUTDIR:-"$RESULTS_ROOT/$TAG/$SUBDIR"}
mkdir -p "$OUTDIR"

# Reads the WildChat arrow shards under $WILDCHAT_DATASET (default data/wildchat).
exec "$PYTHON" -u scripts/wildchat/decomposition/wildchat_score_assistant_nll.py \
  --out-dir "$OUTDIR" \
  --base-model "Qwen/Qwen3-0.6B" \
  --device "cuda:0" \
  --micro-batch "${MICRO_BATCH}" \
  --max-len "${MAX_LEN}" \
  --dtype bfloat16 \
  --language-source top_level \
  --language English \
  --subsample-fraction "${SUBSAMPLE_FRACTION}" \
  --subsample-seed "${SUBSAMPLE_SEED}" \
  ${FLAG_ARG}
