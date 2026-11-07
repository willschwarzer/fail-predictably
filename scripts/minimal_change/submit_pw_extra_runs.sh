#!/bin/bash
# The 200-step password runs of the paper's 2x2 (appendix "Forecastability training with
# minimal change to the model") that the other submitters do not launch:
#   pw2iogsb200  IOG + band, no anchor terms: seeds 11 17 23 29 43, 200 steps from the start
#   pw2apsp200   anchpen + band: seeds 29 and 43, 200 steps from the start; seeds 11 17 23
#                continue their 100-step pw2apsp runs (submit_lqpw_fleet.sh) to 200 steps
#                with extend_lqpw_arm.sbatch
# Both arms use the pw2apsp knobs of submit_lqpw_fleet.sh; pw2iogsb200 sets the anchor terms
# to 0. Runs from the start get four readers (steps 10-50, 60-100, 110-150, 160-200), the
# extensions two (110-150, then 160-200). An extension whose source has not finished waits
# (afterok) on the source's queued trainer, lqpw_pw2apsp_s<seed>, or is skipped if none is
# queued; a run whose directory exists is skipped.
# Set SBATCH_ACCOUNT / SBATCH_PARTITION (and SBATCH_QOS) for your cluster; sbatch reads them.
# Hardware: one GPU of 48 GB or more per trainer (the paper used A100-80GB, L40S, H100 and A40
# cards); the readers need only a small inference GPU (snapshot_heldout_eval.sbatch).
#
#   DRY=1 bash scripts/minimal_change/submit_pw_extra_runs.sh [ARM ...]   (default: both arms)
set -euo pipefail
DRY=${DRY:-0}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
export REPO_ROOT RESULTS_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
D=$REPO_ROOT/scripts/minimal_change
RUNS=$RESULTS_ROOT/lquant_dynamics/runs

BAND="SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,SIGMA_FLOOR_SYMMETRIC=1,CERT_DISTILL_SHARED_PASS=1"
knobs() {
  case "$1" in
    pw2apsp200)  echo "ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0,$BAND,PWBANK=v2" ;;
    pw2iogsb200) echo "ANCHOR_MARGIN=0,ANCHOR_PENALTY=0,$BAND,PWBANK=v2" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}
ARMS=("$@")
(( ${#ARMS[@]} )) || ARMS=(pw2iogsb200 pw2apsp200)

run() { if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }

# readers RUN_DIR TRAINER_JOB JOB_NAME "STEPS" ...: one reader per step group, the first
# gated on the trainer starting, each later one on the reader before it.
readers() {
  local dir=$1 jid=$2 name=$3; shift 3
  local prev=$jid dep steps rs=()
  for steps in "$@"; do
    dep=$([[ $prev == "$jid" ]] && echo "after:$jid" || echo "afterany:$prev")
    # shellcheck disable=SC2086
    prev=$(run --time=08:00:00 --dependency="$dep" --job-name="$name" \
           "$D/snapshot_heldout_eval.sbatch" "$dir" $steps)
    rs+=("$prev")
  done
  echo "${rs[*]}"
}

for arm in "${ARMS[@]}"; do
  k=$(knobs "$arm")
  for s in 11 17 23 29 43; do
    dir=$RUNS/lqpw_${arm}_seed${s}
    [[ -e $dir ]] && { echo "[skip] $dir exists" >&2; continue; }
    if [[ $arm == pw2apsp200 && " 11 17 23 " == *" $s "* ]]; then
      deps=()
      if [[ ! -f $RUNS/lqpw_pw2apsp_seed${s}/test_results.json ]]; then
        tj=$(squeue -h -u "$USER" -n "lqpw_pw2apsp_s${s}" -o %i | head -n 1)
        if [[ -z $tj ]]; then echo "[skip] $arm s$s: pw2apsp s$s unfinished and no trainer queued" >&2; continue; fi
        deps=(--dependency="afterok:$tj")
      fi
      jid=$(run -N 1 -c 4 --mem=48G --gres=gpu:1 --job-name="lqpw_${arm}_s${s}" "${deps[@]}" \
            --export="ALL,SRC_ARM=pw2apsp,ARM=${arm},SEED=${s},STEPS=200,${k}" "$D/extend_lqpw_arm.sbatch")
      rs=$(readers "$dir" "$jid" "lqsnap_${arm}_seed${s}" "110 120 130 140 150" "160 170 180 190 200")
    else
      jid=$(run -N 1 -c 4 --mem=48G --gres=gpu:1 --job-name="lqpw_${arm}_s${s}" \
            --export="ALL,ARM=${arm},SEED=${s},STEPS=200,${k}" "$D/run_lqpw_arm.sbatch")
      rs=$(readers "$dir" "$jid" "lqsnap_${arm}_seed${s}" \
           "10 20 30 40 50" "60 70 80 90 100" "110 120 130 140 150" "160 170 180 190 200")
    fi
    echo "[$arm s$s] train $jid, readers $rs"
  done
done
