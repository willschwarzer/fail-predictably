#!/bin/bash
#SBATCH --job-name=rlq_cal_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Hardware: CPU only (16 cores; each of the paper's sweeps took under 10 minutes).
set -euo pipefail

# Writes the calibration files that the gridworld figure of "Multi-task RL: gridworld
# navigation with traps" and the appendix "Forecastability training with sampled
# returns" read. Submit from the repo root:
#   sbatch scripts/gridworld/run_rlquant_posthoc_calibration_sweep.sbatch

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
export REPO_ROOT
# One thread per calibration process; the sweep runs SLURM_CPUS_PER_TASK of them at once.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec bash "$REPO_ROOT/scripts/gridworld/run_rlquant_posthoc_calibration_sweep.sh"
