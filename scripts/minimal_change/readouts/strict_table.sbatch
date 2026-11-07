#!/bin/bash
#SBATCH --job-name=lq_strict_table
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Hardware: CPU only.
set -euo pipefail
# The strict-standard table of the minimal-change runs (strict_table.py), written to --out
# under $RESULTS_ROOT/lquant_dynamics/readouts/.
# Usage: sbatch scripts/minimal_change/readouts/strict_table.sbatch [--step S | --steps S ...] [--brief] [--out NAME.csv]
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
"$PYTHON" -u "$HERE/strict_table.py" "$@"
date
