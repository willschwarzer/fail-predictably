#!/bin/bash
#SBATCH --job-name=lq_pysmall
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Hardware: CPU only.
set -euo pipefail
# Any small CPU-only script (figures, short readouts).
# Usage: sbatch scripts/minimal_change/readouts/py_small.sbatch SCRIPT.py [ARGS ...]
# SCRIPT.py is a file name in scripts/minimal_change/readouts/, a path from the repo root, or an
# absolute path; it runs in $RESULTS_ROOT/lquant_dynamics/readouts/.
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
cd "$RESULTS_ROOT/lquant_dynamics/readouts"   # relative paths in the arguments resolve here
export HF_HUB_OFFLINE=1
script=${1:?usage: py_small.sbatch SCRIPT.py [ARGS ...]}; shift
case $script in
  /*) ;;
  */*) script="$REPO_ROOT/$script" ;;
  *) script="$HERE/$script" ;;
esac
"$PYTHON" -u "$script" "$@"
date
