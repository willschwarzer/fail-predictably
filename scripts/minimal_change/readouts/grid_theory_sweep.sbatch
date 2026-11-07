#!/bin/bash
#SBATCH --job-name=lq_grid_sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
# Hardware: CPU only.
set -euo pipefail
# G = F / S^2 of a straight tail at a range of depths lambda, per geometry of the certificate grid (grid_theory_sweep.py).
# Usage: sbatch scripts/minimal_change/readouts/grid_theory_sweep.sbatch [--draws 20000]
hostname; date
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "$PYTHON"; "$PYTHON" --version
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
mkdir -p "$RESULTS_ROOT/lquant_dynamics/readouts"
RESULTS_ROOT=$(cd "$RESULTS_ROOT" && pwd)
export REPO_ROOT RESULTS_ROOT
HERE="$REPO_ROOT/scripts/minimal_change/readouts"
cd "$RESULTS_ROOT/lquant_dynamics/readouts"
export HF_HUB_OFFLINE=1
"$PYTHON" -u "$HERE/grid_theory_sweep.py" "$@"
date
