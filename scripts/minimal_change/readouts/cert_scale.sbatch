#!/bin/bash
#SBATCH --job-name=lq_cert_scale
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Hardware: CPU only.
set -euo pipefail
# Certified floors at larger M, R = N/M and k on the v2e2 population (cert_scale.py).
# Usage: sbatch scripts/minimal_change/readouts/cert_scale.sbatch OUT_CSV [--smoke]
# OUT_CSV is relative to $RESULTS_ROOT/lquant_dynamics/readouts/; the theory readouts read cert_scale.csv there.
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
"$PYTHON" -u "$HERE/cert_scale.py" "$@" --workers "${SLURM_CPUS_PER_TASK:-1}"
date
