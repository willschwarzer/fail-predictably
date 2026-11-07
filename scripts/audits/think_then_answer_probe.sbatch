#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=think_answer
#SBATCH --requeue
# Hardware: one 16 GB GPU (the paper used NVIDIA A16 cards). On a card with 20 GB or more, the 4B size samples in
# one batch of 16 instead of two of 8, so its sampled generations differ from ours.
set -euo pipefail

# Do the trained checkpoints answer normally when they think first (think_then_answer_probe.py)?
# Read-only on the run dirs; writes one JSONL per model and summary.csv to --out-dir.
#   sbatch --job-name=think_answer_06b scripts/audits/think_then_answer_probe.sbatch --size 0.6b --out-dir DIR
#   sbatch scripts/audits/think_then_answer_probe.sbatch --group NAME --model LABEL=CONFIG_JSON:ADAPTER [...] --out-dir DIR
# scripts/audits/submit_think_then_answer.sh submits the paper's groups.

hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module list 2>&1 || true
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # the models must already be in the Hugging Face cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
which "$PYTHON"; "$PYTHON" --version
# SIZES="0.6b 1.7b 4b" (exported) runs the sizes in turn in one job; a failed size does not stop the next.
if [[ -n "${SIZES:-}" ]]; then
  for s in $SIZES; do
    "$PYTHON" -u scripts/audits/think_then_answer_probe.py --size "$s" "$@" || echo "size $s failed"
  done
else
  "$PYTHON" -u scripts/audits/think_then_answer_probe.py "$@"
fi
date
