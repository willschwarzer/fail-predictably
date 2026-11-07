#!/bin/bash
# Password-v2 anchpen fleet: three arms that differ only in the anchpen terms,
# on the cell of run_lqpw_arm.sbatch (paper appendix "Forecastability training with
# minimal change to the model"). The fourth arm of the paper's 2x2, IOG + band, was
# trained for 200 steps from the start (submit_pw_extra_runs.sh).
#
#   pw2apsp  anchpen + band: IOG, anchor gate 0.5 nats, anchor penalty 1.0,
#            two-sided spread band (lambda 50, margin 0.13), shared pass
#   pw2nbap  anchpen without the band (no constraint channel, so no shared pass)
#   pw2iog   the paper's IOG + KL alone, at the same run settings
#
# Usage: bash scripts/minimal_change/submit_lqpw_fleet.sh SEED [ARM ...]   (default: all three arms)
#   DRY=1 prints the sbatch lines, and each job gets a snapshot reader chained
#   behind it. Set SBATCH_ACCOUNT / SBATCH_PARTITION (and SBATCH_QOS) for your
#   cluster; sbatch reads them.
# Hardware: one GPU of 48 GB or more per arm (the paper used A100-80GB, L40S, H100 and A40 cards).
set -euo pipefail
SEED=${1:?usage: submit_lqpw_fleet.sh SEED [ARM ...]}
shift
ARMS=("$@")
(( ${#ARMS[@]} )) || ARMS=(pw2apsp pw2nbap pw2iog)
DRY=${DRY:-0}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
export REPO_ROOT RESULTS_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
GEN="$REPO_ROOT/scripts/minimal_change/run_lqpw_arm.sbatch"
READER="$REPO_ROOT/scripts/minimal_change/snapshot_heldout_eval.sbatch"

BAND="SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,SIGMA_FLOOR_SYMMETRIC=1"
ANCH="ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0"
for arm in "${ARMS[@]}"; do
  # pw2* arms run on password v2, pw1* on the paper's earlier v1 bank (PWBANK).
  case "$arm" in
    pw[12]apsp) rec="$ANCH,$BAND,CERT_DISTILL_SHARED_PASS=1" ;;
    pw[12]nbap) rec="$ANCH" ;;
    pw[12]iog)  rec="ANCHOR_MARGIN=0" ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  rec="$rec,PWBANK=v${arm:2:1}"
  args=(-N 1 -c 4
        --mem=48G --gres=gpu:1 --job-name="lqpw_${arm}_s${SEED}"
        --export="ALL,ARM=${arm},SEED=${SEED},${rec}")
  run_dir="$RESULTS_ROOT/lquant_dynamics/runs/lqpw_${arm}_seed${SEED}"
  if (( DRY )); then
    echo "sbatch ${args[*]} $(basename "$GEN")"
    continue
  fi
  jid=$(sbatch --parsable "${args[@]}" "$GEN")
  # Readers: steps 10-50 then 60-100, each job waiting up to 4 h per snapshot.
  r1=$(sbatch --parsable --time=08:00:00 --dependency=after:"$jid" "$READER" "$run_dir" 10 20 30 40 50)
  r2=$(sbatch --parsable --time=08:00:00 --dependency=afterany:"$r1" "$READER" "$run_dir" 60 70 80 90 100)
  echo "[$arm s$SEED] train $jid, readers $r1 -> $r2"
done
