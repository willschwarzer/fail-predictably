#!/bin/bash
# Training ablation of the band's margin (paper appendix "Forecastability training with
# minimal change to the model", "Training at other margins"): the main band arm of each
# dataset (anchpen + two-sided band) retrained at seed 43 with SIGMA_FLOOR_MARGIN 0.06
# and 0.25 nats instead of 0.13, everything else exactly as the seed-43 main runs were
# submitted (WildChat lqdyn_v2e2ap50sp_seed43, from submit_lqdyn_fleet.sh; password
# lqpw_pw2apsp200_seed43, 200 steps from scratch, from submit_pw_extra_runs.sh). The
# seed-43 main runs are the 0.13 point.
#
# WildChat runs with its main arm's two readers (steps 10-30, then 40-100).
# Password readers: steps 10-50, 60-100, 110-150, 160-200.
# Set SBATCH_ACCOUNT / SBATCH_PARTITION (and SBATCH_QOS) for your cluster; sbatch reads them.
# Hardware: one GPU of 48 GB or more per trainer (the paper used A100-80GB, L40S and H100
# cards); the readers need only a small inference GPU (snapshot_heldout_eval.sbatch).
#
#   DRY=1 bash scripts/minimal_change/submit_margin_ablation.sh    -> print the sbatch lines only
set -euo pipefail
DRY=${DRY:-0}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export REPO_ROOT RESULTS_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
D=$REPO_ROOT/scripts/minimal_change
RUNS=$RESULTS_ROOT/lquant_dynamics/runs
SEED=43
WC_BANK=$TASK_BANK_ROOT/wildchat_ldnoobw_v2e2_qwen_qwen3-0_6b_T32_p2300
WC_KNOBS="BOOT=16,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_SYMMETRIC=1,ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0,BANK_ROOT=$WC_BANK,FIT_FRACTION=0.1087,UNION_REFRESH=10,TARGET_FULL_POOLS=1,HELDOUT_FROM_END=1,HELDOUT_TARGETS=8,STEPS=100,LR=2e-5,DUMP_EVERY=2,SNAPSHOT_EVERY=10,SAVE_EVERY=10,TRAIN_TARGETS=24,TARGETS_PER_STEP=12,OLS_TOP_K=50,SIGMA_REF_M=789,GAP_FLOOR_M=789,UNION_CACHE_C=789,CERT_DISTILL_SHARED_PASS=1"
PW_KNOBS="STEPS=200,ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_SYMMETRIC=1,CERT_DISTILL_SHARED_PASS=1,PWBANK=v2"

run() { if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }

for m in 0.06 0.25; do
  tag=m${m#0.}  # m06, m25
  # WildChat
  arm=v2e2ap50sp${tag}
  dir=$RUNS/lqdyn_${arm}_seed${SEED}
  jid=$(run -N 1 -c 8 --gres=gpu:1 --mem=40G --job-name="lqps_${arm}_s${SEED}" \
        --export="ALL,ARM=${arm},SEED=${SEED},SIGMA_FLOOR_MARGIN=${m},${WC_KNOBS}" "$D/run_lqdyn_arm.sbatch")
  r1=$(run --time=08:00:00 --dependency=after:"$jid"+25 --job-name="lqsnap_${arm}_seed${SEED}" \
       "$D/snapshot_heldout_eval.sbatch" "$dir" 10 20 30)
  r2=$(run --time=16:00:00 --dependency=afterany:"$r1" --job-name="lqsnap_${arm}_seed${SEED}" \
       "$D/snapshot_heldout_eval.sbatch" "$dir" 40 50 60 70 80 90 100)
  echo "[$arm s$SEED] train $jid, readers $r1 -> $r2"
  # Password
  arm=pw2apsp200${tag}
  dir=$RUNS/lqpw_${arm}_seed${SEED}
  jid=$(run -N 1 -c 4 --mem=48G --gres=gpu:1 \
        --job-name="lqpw_${arm}_s${SEED}" \
        --export="ALL,ARM=${arm},SEED=${SEED},SIGMA_FLOOR_MARGIN=${m},${PW_KNOBS}" "$D/run_lqpw_arm.sbatch")
  prev=$jid
  rs=()
  for steps in "10 20 30 40 50" "60 70 80 90 100" "110 120 130 140 150" "160 170 180 190 200"; do
    dep=$([[ $prev == "$jid" ]] && echo "after:$jid" || echo "afterany:$prev")
    # shellcheck disable=SC2086
    prev=$(run --time=08:00:00 --dependency="$dep" --job-name="lqsnap_${arm}_seed${SEED}" \
           "$D/snapshot_heldout_eval.sbatch" "$dir" $steps)
    rs+=("$prev")
  done
  echo "[$arm s$SEED] train $jid, readers ${rs[*]}"
done
