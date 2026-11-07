#!/bin/bash
# Continue the password-v2 ablations without the band (IOG alone = pw2iog, anchpen
# without the band = pw2nbap) from 100 to 200 steps at all five seeds, so every arm of
# the password 2x2 ablation has a 200-step read (IOG + band runs 200 steps as
# pw2iogsb200, the band arm as pw2apsp200; both in submit_pw_extra_runs.sh).
# The sources must have finished: extend_lqpw_arm.sbatch refuses a run with no final eval.
#
# Each trainer gets two readers (steps 110-150, then 160-200), as the band arm's
# extensions do.
# Set SBATCH_ACCOUNT / SBATCH_PARTITION (and SBATCH_QOS) for your cluster; sbatch reads them.
# Hardware: one GPU with 48 GB or more per trainer (these extensions ran on A100-80GB cards).
#
#   DRY=1 bash scripts/minimal_change/submit_pw_ablation_ext200.sh    -> print the sbatch lines only
set -euo pipefail
DRY=${DRY:-0}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
export REPO_ROOT RESULTS_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
D=$REPO_ROOT/scripts/minimal_change
RUNS=$RESULTS_ROOT/lquant_dynamics/runs

knobs() {  # the source arm's launch knobs, exactly as submit_lqpw_fleet.sh set them
  case "$1" in
    pw2iog)  echo "ANCHOR_MARGIN=0,PWBANK=v2" ;;
    pw2nbap) echo "ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0,PWBANK=v2" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

for arm in pw2iog pw2nbap; do
  for seed in 11 17 23 29 43; do
    new="${arm}200"
    run_dir="$RUNS/lqpw_${new}_seed${seed}"
    args=(-N 1 -c 4 --mem=48G
          --gres=gpu:1 --job-name="lqpw_${new}_s${seed}"
          --export="ALL,SRC_ARM=${arm},ARM=${new},SEED=${seed},STEPS=200,$(knobs "$arm")")
    if (( DRY )); then
      echo "sbatch ${args[*]} extend_lqpw_arm.sbatch"
      echo "   readers: $run_dir 110-150, then 160-200"
      continue
    fi
    jid=$(sbatch --parsable "${args[@]}" "$D/extend_lqpw_arm.sbatch")
    r1=$(sbatch --parsable --time=08:00:00 --job-name="lqsnap_${new}_seed${seed}" --dependency=after:"$jid" \
         "$D/snapshot_heldout_eval.sbatch" "$run_dir" 110 120 130 140 150)
    r2=$(sbatch --parsable --time=08:00:00 --job-name="lqsnap_${new}_seed${seed}" --dependency=afterany:"$r1" \
         "$D/snapshot_heldout_eval.sbatch" "$run_dir" 160 170 180 190 200)
    echo "[$new s$seed] train $jid, readers $r1 -> $r2"
  done
done
