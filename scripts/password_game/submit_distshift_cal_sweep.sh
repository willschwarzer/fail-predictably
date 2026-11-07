#!/bin/bash
# Post-hoc calibration sweep for the 0.6B distribution-shift cells (K=1
# canonical, shift+scale). Three arms per seed:
#   1. cal standalone  : pretrained forecasts calibrated, applied to the
#      baseline heldout results (RUN_DIR = ours; no adapter).
#   2. ours + cal      : trained-adapter forecasts calibrated, applied to the
#      trained heldout results.
#   3. SFT + cal       : same as (2) on the SFT run dir.
# Inputs: the run dirs of submit_lquant_distshift_0p6b_fleet.sh (ours) and of
# run_lquant_sft_nopwseed_distshift_0p6b.sbatch (SFT), both submitted with the
# same TAG_DATE (default 20260726, the tag date of the paper's runs).
# Usage: bash scripts/password_game/submit_distshift_cal_sweep.sh once both fleets
# have finished. To queue it earlier, pass the training job ids as MAIN_JOBS and
# SFT_JOBS (space-separated, in SEEDS order); each arm then waits (afterok) on
# its training job.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"   # jobs write logs/ relative to the submit directory
SB="$REPO_ROOT/scripts/password_game/run_posthoc_calibration_baseline.sbatch"
RES="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TAG_DATE=${TAG_DATE:-20260726}
# Hardware: one GPU with at least 40 GB and compute capability 8.0 or newer.
COMMON=(--mem=40G)

SEEDS=(11 17 23 29 31 37 41 43 47 53)
read -r -a MAIN_JOBS <<< "${MAIN_JOBS:-}"
read -r -a SFT_JOBS <<< "${SFT_JOBS:-}"

for i in "${!SEEDS[@]}"; do
  SEED=${SEEDS[$i]}
  MAIN_RUN="$RES/lquant_40slots_nopwseed_distshift891_strictv2_iog_full_seed${SEED}_${TAG_DATE}/seed_${SEED}"
  SFT_RUN="$RES/lquant_40slots_nopwseed_distshift891_sft_union_qwen3_0_6b_seed${SEED}_${TAG_DATE}/seed_${SEED}"
  MAIN_DEP=()
  SFT_DEP=()
  if [[ ${#MAIN_JOBS[@]} -gt 0 ]]; then MAIN_DEP=(--dependency=afterok:${MAIN_JOBS[$i]}); fi
  if [[ ${#SFT_JOBS[@]} -gt 0 ]]; then SFT_DEP=(--dependency=afterok:${SFT_JOBS[$i]}); fi

  sbatch "${COMMON[@]}" "${MAIN_DEP[@]}" \
    --job-name=cal_base_s${SEED} \
    --export=ALL,RUN_DIR="$MAIN_RUN",OUT_JSON="$MAIN_RUN/posthoc_calibration_baseline.json" \
    "$SB"

  sbatch "${COMMON[@]}" "${MAIN_DEP[@]}" \
    --job-name=cal_ours_s${SEED} \
    --export=ALL,RUN_DIR="$MAIN_RUN",OUT_JSON="$MAIN_RUN/posthoc_calibration_trained.json",LORA_CHECKPOINT="$MAIN_RUN/final",HELDOUT_RESULTS_JSON="test_results.json" \
    "$SB"

  sbatch "${COMMON[@]}" "${SFT_DEP[@]}" \
    --job-name=cal_sft_s${SEED} \
    --export=ALL,RUN_DIR="$SFT_RUN",OUT_JSON="$SFT_RUN/posthoc_calibration_trained.json",LORA_CHECKPOINT="$SFT_RUN/final",HELDOUT_RESULTS_JSON="test_results.json" \
    "$SB"
done
