#!/bin/bash
# Submit the ten-seed fit/deployment family-shift experiment on the original
# Qwen3-0.6B password bank (appendix "Fit/deployment distribution shift", its
# table and its per-seed figure). For each seed this submits four jobs:
#   1. training, run_lquant_40slots_may_v1_distshift_0p6b.sbatch;
#   2. post-hoc calibration of the pretrained forecast (Cal.), after training;
#   3. held-out evaluation of the checkpoint selected by training-side forecast
#      error (best/), run_lquant_40slots_may_v1_distshift_0p6b_besteval.sbatch,
#      after training;
#   4. post-hoc calibration of that checkpoint's forecast (Ours + cal.), after 3.
# When all jobs have finished, collect_distshift_results.py writes the table.
# The hardened-bank shift experiment has its own submitters,
# submit_lquant_distshift_0p6b_fleet.sh and submit_distshift_cal_sweep.sh.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"   # jobs write logs/ relative to the submit directory
PG="$REPO_ROOT/scripts/password_game"
RES="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TAG_DATE=${TAG_DATE:-20260726}

for SEED in 11 17 23 29 31 37 41 43 47 53; do
  RUN="$RES/lquant_40slots_mayv1_distshift891_iog_full_seed${SEED}_${TAG_DATE}/seed_${SEED}"
  BEST="$RES/lquant_40slots_mayv1_distshift891_besteval_seed${SEED}_${TAG_DATE}/seed_${SEED}"

  train=$(sbatch --parsable --export=ALL,SEED=${SEED},TAG_DATE=${TAG_DATE} \
    "$PG/run_lquant_40slots_may_v1_distshift_0p6b.sbatch")
  train=${train%%;*}

  sbatch --dependency=afterok:${train} --job-name=cal_base_s${SEED} \
    --export=ALL,RUN_DIR="$RUN",OUT_JSON="$RUN/posthoc_calibration_baseline.json" \
    "$PG/run_posthoc_calibration_baseline.sbatch"

  best=$(sbatch --parsable --dependency=afterok:${train} \
    --export=ALL,SEED=${SEED},TAG_DATE=${TAG_DATE},ADAPTER="$RUN/best" \
    "$PG/run_lquant_40slots_may_v1_distshift_0p6b_besteval.sbatch")
  best=${best%%;*}

  sbatch --dependency=afterok:${best} --job-name=cal_ours_s${SEED} \
    --export=ALL,RUN_DIR="$BEST",OUT_JSON="$BEST/posthoc_calibration_trained.json",LORA_CHECKPOINT="$RUN/best",HELDOUT_RESULTS_JSON=test_results.json \
    "$PG/run_posthoc_calibration_baseline.sbatch"
done
