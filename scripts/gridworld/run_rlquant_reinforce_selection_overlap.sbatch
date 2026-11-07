#!/bin/bash
#SBATCH --job-name=sel_overlap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Hardware: CPU only (16 cores; the paper's run took about 2 minutes).
set -euo pipefail

# The selection diagnostic of the appendix "Forecastability training with sampled
# returns" (recall of the exact path's selected tasks as the rollout count rises from
# 1 to 32), at the final checkpoint of seed 11 of the full-IOG fleet that
# run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_10seeds.sh writes
# (TAG_DATE=20260505). Submit from the repo root:
#   sbatch scripts/gridworld/run_rlquant_reinforce_selection_overlap.sbatch

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
export REPO_ROOT
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

exec "$PYTHON" -u "$REPO_ROOT/scripts/gridworld/rlquant_reinforce_selection_overlap.py" \
  --seed-dir "$RESULTS_ROOT/rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_10seeds_20260505/seed_11" \
  --checkpoint final --n-list 1,2,4,8,16,32 --draws 200 \
  --out-json "$RESULTS_ROOT/rlquant_reinforce_selection_overlap_20260725/selection_overlap_seed11_final_iog1.json"
