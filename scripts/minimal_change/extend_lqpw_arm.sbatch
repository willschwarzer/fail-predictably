#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lqpw
# Hardware: one GPU with 48 GB or more (the paper used A100-80GB, H100, L40S and A40 cards).
set -euo pipefail

# Continue a finished lqpw run past its step count, in a copy of its run dir, so
# the original keeps its own final eval.
#
# The trainer keeps a "complete" resume checkpoint at the end of every run, and
# relaunching with a larger --steps resumes training from it (train_lquant.py:
# the resume phase flips complete -> train when start_step < steps; `steps` and
# `outdir` are left out of the resume config check). With AdamW at a constant lr
# the continued run is the same recipe trained longer. The copy is made once, on
# the first start; a requeue finds it and resumes from its latest checkpoint.
# The copy carries snapshot_evals/ (the base pass, scored on the readers' GPU
# model), so readers of the new steps reuse the same base.
#
# Required env:
#   SRC_ARM  the finished arm (e.g. pw2apsp); its run dir must hold test_results.json
#   ARM      the new arm name (e.g. pw2apspx200); the copy is lqpw_${ARM}_seed${SEED}
#   SEED     the seed of both
#   STEPS    the new total step count (larger than the source's)
# plus the source's arm knobs exactly as it was launched (run_lqpw_arm.sbatch),
# e.g. ANCHOR_MARGIN, ANCHOR_PENALTY, SIGMA_FLOOR_*, CERT_DISTILL_SHARED_PASS.

SRC_ARM=${SRC_ARM:?SRC_ARM env var required}
ARM=${ARM:?ARM env var required}
SEED=${SEED:?SEED env var required}
STEPS=${STEPS:?STEPS env var required}
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
export REPO_ROOT RESULTS_ROOT   # the arm script below reads the same roots
RUNS=$RESULTS_ROOT/lquant_dynamics/runs
SRC=$RUNS/lqpw_${SRC_ARM}_seed${SEED}
DST=$RUNS/lqpw_${ARM}_seed${SEED}

if [[ ! -d "$DST" ]]; then
  if [[ ! -f "$SRC/test_results.json" || ! -f "$SRC/resume_latest.pt" ]]; then
    echo "[extend] $SRC has no final eval or no resume checkpoint; not extending" >&2
    exit 1
  fi
  SRC_STEPS=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['steps'])" "$SRC/config.json")
  if (( STEPS <= SRC_STEPS )); then
    echo "[extend] STEPS=$STEPS is not past the source's $SRC_STEPS" >&2
    exit 1
  fi
  cp -a "$SRC" "$DST.tmp.$SLURM_JOB_ID"
  mv "$DST.tmp.$SLURM_JOB_ID" "$DST"
  echo "[extend] copied $SRC -> $DST (source steps $SRC_STEPS, new steps $STEPS)"
else
  echo "[extend] $DST exists; resuming it"
fi

exec bash "$REPO_ROOT/scripts/minimal_change/run_lqpw_arm.sbatch"
