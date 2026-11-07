#!/bin/bash
#SBATCH --job-name=lq_tail_opt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Hardware: CPU only.
set -euo pipefail

# Prepare once, then reuse that manifest across shards and retries. Submit from the repo root;
# relative paths in the arguments resolve under $RESULTS_ROOT/lquant_dynamics/readouts/. Examples:
# sbatch scripts/minimal_change/readouts/tail_opt.sbatch --fleet --bank --label fleet --prepare-only
# sbatch --array=0-124 scripts/minimal_change/readouts/tail_opt.sbatch --manifest tail_opt_results/fleet.manifest.npz
# sbatch scripts/minimal_change/readouts/tail_opt.sbatch --manifest tail_opt_results/fleet.manifest.npz --aggregate-only
# Array size must match the manifest's expected pool count. For a subset use --shard I/N.
# Repeat preparation with --band fitmax and a different label for the second band.
# Grid (168 cell/k/pool tasks per band at --grid-pools 8):
# sbatch --mem=32G scripts/minimal_change/readouts/tail_opt.sbatch --grid --prepare-only --label grid
# sbatch --mem=32G --array=0-167 scripts/minimal_change/readouts/tail_opt.sbatch --manifest tail_opt_results/grid.manifest.npz
# sbatch --mem=32G scripts/minimal_change/readouts/tail_opt.sbatch --manifest tail_opt_results/grid.manifest.npz --aggregate-only
# Repeat with --band fitmax --label grid_fitmax, then use grid_fitmax.manifest.npz.
# Smoke selections, each in its own allocation:
# sbatch --mem=32G scripts/minimal_change/readouts/tail_opt.sbatch --grid --grid-pools 1 --grid-cell M250_R8.2 --grid-k 10 --label grid_smoke_small
# sbatch --mem=32G scripts/minimal_change/readouts/tail_opt.sbatch --grid --grid-pools 1 --grid-cell M4000_R8.2 --grid-k 1000 --label grid_smoke_large
# Grid records include the exact CSV checks, partition counts, wall time and process peak RSS.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
mkdir -p "$RESULTS_ROOT/lquant_dynamics/readouts"
RESULTS_ROOT=$(cd "$RESULTS_ROOT" && pwd)
export REPO_ROOT RESULTS_ROOT
cd "$RESULTS_ROOT/lquant_dynamics/readouts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# One numerical thread avoids oversubscription for the small gathered matrices.
# srun binds the process to the CPU granted to the allocation.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
shard_args=()
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    if [[ ${SLURM_ARRAY_TASK_MIN:-0} != 0 || ${SLURM_ARRAY_TASK_STEP:-1} != 1 ]]; then
        echo "Use a contiguous, zero-based array, or pass --shard manually."
        exit 2
    fi
    shard_args=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}")
fi
srun --cpu-bind=cores "$PYTHON" -u "$REPO_ROOT/scripts/minimal_change/readouts/tail_opt.py" "$@" "${shard_args[@]}"
