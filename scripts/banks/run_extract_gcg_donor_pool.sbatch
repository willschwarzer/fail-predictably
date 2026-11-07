#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=gcg_donor_extract
# Hardware: one GPU with at least 16 GB and compute capability 8.0 or newer.
set -euo pipefail

# Inputs: BANK, a from-scratch strictv2 password bank for the same model (default:
# the paper's 0.6B one, lquant_prompt_bank_40slots_modbulk_gcg6_strictv2_qwen3_0_6b_20260723,
# which scripts/banks/submit_strictv2_password_bank.sh builds).
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
BANK=${BANK:-$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_strictv2_qwen3_0_6b_20260723/bank_seed_11}
OUT=${OUT:-artifacts/gcg_donor_pools/qwen3_0_6b_strictv2_seed11.json}
POOL_SIZE=${POOL_SIZE:-12}
exec "$PYTHON" -u scripts/banks/extract_gcg_donor_pool.py \
  --model "$MODEL" --bank "$BANK" --out "$OUT" --pool-size "$POOL_SIZE"
