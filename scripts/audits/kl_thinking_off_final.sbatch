#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=kl_thinkoff_fig4
#SBATCH --requeue
# Hardware: one 16 GB GPU for Qwen3 up to 4B (the paper used NVIDIA A16 cards); Qwen3-8B needs a larger card
# (the paper's 8B read ran on an A100-80GB).
set -euo pipefail

# Held-out single-token KL with thinking on (the trainer's prompts) and off (the scoring template) for runs
# that kept only seed_*/final/ adapters, by default the two trained arms of Figure 4's KL panel
# (kl_thinking_off_final.py). Read-only on the run dirs; appends rows to the CSV, and a requeued job skips
# rows already there.
#   sbatch scripts/audits/kl_thinking_off_final.sbatch --out CSV [--adapter-subdir best] [--generate N] [LABEL=GLOB ...]

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
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
which "$PYTHON"; "$PYTHON" --version
"$PYTHON" -u scripts/audits/kl_thinking_off_final.py "$@"
date
