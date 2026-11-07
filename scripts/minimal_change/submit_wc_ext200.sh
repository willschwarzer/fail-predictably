#!/bin/bash
# Continue the WildChat 2x2 (anchpen + band, IOG alone, anchpen without the band, IOG + band)
# from 100 to 200 steps at seeds 11 17 23 29 43, into copies named lqdyn_<arm>x200_seed<seed>,
# with scripts/minimal_change/extend_lqdyn_arm.sbatch. The knobs repeat the source runs' own
# (submit_lqdyn_fleet.sh, poolscale wave), with STEPS=200.
#
# Submission order is priority order (equal nice, so age decides): every finished source of
# the listed arms first, in arm order (the band arm first), then the sources still training,
# each with afterok on its own trainer job so the copy is made only after its final eval.
# Each trainer gets two readers (steps 110-150, then 160-200).
# Set SBATCH_ACCOUNT / SBATCH_PARTITION (and SBATCH_QOS) for your cluster; sbatch reads them.
# Hardware: one 80 GB GPU per trainer (A100-80GB or H100; some extensions ran out of memory
# on 48 GB L40S cards); the readers need only a small inference GPU (snapshot_heldout_eval.sbatch).
#
#   DRY=1 bash scripts/minimal_change/submit_wc_ext200.sh [ARM ...]      (default: all four arms)
set -euo pipefail
DRY=${DRY:-0}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export REPO_ROOT RESULTS_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
D=$REPO_ROOT/scripts/minimal_change
RUNS=$RESULTS_ROOT/lquant_dynamics/runs
BANK=$TASK_BANK_ROOT/wildchat_ldnoobw_v2e2_qwen_qwen3-0_6b_T32_p2300
COMMON="BOOT=16,BANK_ROOT=$BANK,FIT_FRACTION=0.1087,UNION_REFRESH=10,TARGET_FULL_POOLS=1,HELDOUT_FROM_END=1,HELDOUT_TARGETS=8,LR=2e-5,DUMP_EVERY=2,SNAPSHOT_EVERY=10,SAVE_EVERY=10,TRAIN_TARGETS=24,TARGETS_PER_STEP=12,OLS_TOP_K=50,SIGMA_REF_M=789,GAP_FLOOR_M=789,UNION_CACHE_C=789"
BAND="SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,SIGMA_FLOOR_SYMMETRIC=1,CERT_DISTILL_SHARED_PASS=1"
knobs() {
  case "$1" in
    v2e2ap50sp)      echo "$BAND,ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0" ;;
    v2e2iogap50)     echo "" ;;
    v2e2nbap50)      echo "ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0" ;;
    v2e2iogsbap50sp) echo "$BAND" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}
ARMS=("$@")
(( ${#ARMS[@]} )) || ARMS=(v2e2ap50sp v2e2iogap50 v2e2nbap50 v2e2iogsbap50sp)
SEEDS=(11 17 23 29 43)

run() { if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }

for pass in finished training; do
  for arm in "${ARMS[@]}"; do
    k=$(knobs "$arm")
    for s in "${SEEDS[@]}"; do
      src=$RUNS/lqdyn_${arm}_seed${s}
      if [[ -f $src/test_results.json ]]; then fin=1; else fin=0; fi
      [[ $pass == finished && $fin == 0 ]] && continue
      [[ $pass == training && $fin == 1 ]] && continue
      deps=()
      if (( ! fin )); then
        tj=$(squeue -h -u "$USER" -n "lqps_${arm}_s${s}" -o %i | head -n 1)
        if [[ -z $tj ]]; then echo "[skip] $arm s$s: unfinished and no trainer queued" >&2; continue; fi
        deps+=("afterok:$tj")
      fi
      new=${arm}x200
      dir=$RUNS/lqdyn_${new}_seed${s}
      [[ -e $dir ]] && { echo "[skip] $dir exists" >&2; continue; }
      dep_arg=()
      (( ${#deps[@]} )) && dep_arg=(--dependency="$(IFS=,; echo "${deps[*]}")")
      jid=$(run -N 1 -c 8 --gres=gpu:1 --mem=40G --job-name="lqps_${new}_s${s}" "${dep_arg[@]}" \
            --export="ALL,SRC_ARM=${arm},ARM=${new},SEED=${s},STEPS=200,${COMMON}${k:+,$k}" "$D/extend_lqdyn_arm.sbatch")
      r1=$(run --time=12:00:00 --dependency=after:"$jid" --job-name="lqsnap_${new}_seed${s}" \
           "$D/snapshot_heldout_eval.sbatch" "$dir" 110 120 130 140 150)
      r2=$(run --time=12:00:00 --dependency=afterany:"$r1" --job-name="lqsnap_${new}_seed${s}" \
           "$D/snapshot_heldout_eval.sbatch" "$dir" 160 170 180 190 200)
      echo "[$new s$s] train $jid${deps:+ (${deps[*]})}, readers $r1 -> $r2"
    done
  done
done
