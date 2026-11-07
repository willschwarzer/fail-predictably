#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lqsnap
# Hardware: one small inference GPU (the paper used 16 GB A16 cards); keep every reader of a run on one GPU model.
set -euo pipefail

# Mid-training held-out read of an lqdyn or lqpw arm on a small inference GPU, so the
# training GPUs stay free (snapshot_heldout_eval.py).
# Run every reader of a run on the same GPU model: the base pass is cached per GPU
# model, and a reader on another model re-scores and overwrites the base, which would
# leave the earlier steps' traces read against a base from a different GPU. The
# paper's readers all ran on A16 cards.
#   sbatch scripts/minimal_change/snapshot_heldout_eval.sbatch RUN_DIR STEP [STEP ...]

hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module list 2>&1 || true
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # the model must already be in the Hugging Face cache
echo "$PYTHON"; "$PYTHON" --version
"$PYTHON" -u scripts/minimal_change/snapshot_heldout_eval.py "$@"
date
