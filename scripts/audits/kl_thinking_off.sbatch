#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_klthinkoff
#SBATCH --requeue
# Hardware: one 16 GB GPU (the paper used NVIDIA A16 cards).
set -euo pipefail

# Held-out single-token KL with thinking on (the trainer's prompts) and off (the scoring template), per
# snapshot of the minimal-change runs (kl_thinking_off.py). Read-only on the run dirs; appends rows to the CSV
# given, and a requeued job skips rows already there.
#   sbatch scripts/audits/kl_thinking_off.sbatch --out CSV [--steps 100 200] [--dataset password|wildchat] \
#     [--truncation-side left|right]
#   sbatch scripts/audits/kl_thinking_off.sbatch --out CSV RUN_DIR:STEP [...]

hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module list 2>&1 || true
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # the model must already be in the Hugging Face cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "$PYTHON"; "$PYTHON" --version
"$PYTHON" -u scripts/audits/kl_thinking_off.py "$@"
date
