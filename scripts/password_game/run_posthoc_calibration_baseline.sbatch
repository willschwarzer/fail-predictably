#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=posthoc_cal_baseline
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# Post-hoc calibration baseline (Cal. in the paper). Takes a meta-train run dir, runs
# pretrained-model FRLMB forecast on TRAIN-target (fit, deploy) pairs to fit
# 1-param + 2-param calibrations, applies them to the heldout results from
# baseline_test_results.json, writes posthoc_calibration_baseline.json next
# to the source baseline_test_results.json.
#
# Required env vars:
#   RUN_DIR  -- absolute path to a meta-train run's seed_<N> dir
#   OUT_JSON -- absolute path for the output JSON

RUN_DIR=${RUN_DIR:?RUN_DIR env var required}
OUT_JSON=${OUT_JSON:?OUT_JSON env var required}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ -n "${MICRO_BATCH:-}" ]]; then
  : # honor explicit override
elif (( VRAM_MB < 30000 )); then MICRO_BATCH=16
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=48
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=96
else MICRO_BATCH=128; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"

EXTRA_ARGS=()
if [[ -n "${LORA_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--lora-checkpoint "${LORA_CHECKPOINT}")
fi
if [[ -n "${HELDOUT_RESULTS_JSON:-}" ]]; then
  EXTRA_ARGS+=(--heldout-results-json "${HELDOUT_RESULTS_JSON}")
fi
if [[ -n "${N_PARTITIONS:-}" ]]; then
  EXTRA_ARGS+=(--n-partitions "${N_PARTITIONS}")
fi

exec "$PYTHON" -u scripts/password_game/posthoc_calibration_baseline.py \
  --run-dir "$RUN_DIR" \
  --out-json "$OUT_JSON" \
  --micro-batch-size-override "$MICRO_BATCH" \
  "${EXTRA_ARGS[@]}"
