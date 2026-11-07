#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=score_password_bank
# Hardware: one GPU with at least 24 GB (the scan takes about two minutes on an A100).
set -euo pipefail

# Leak scan by prompt family for the appendix "Fit/deployment distribution shift":
# scores every prompt of the original Qwen3-0.6B password bank (all 40 passwords,
# all five families) under the pretrained model. collect_distshift_results.py
# reads the output.
#
# Optional env vars: BANK (bank directory name), OUT_TAG (output directory name),
# PASSWORD_POLICY (default non_disclosure).

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export TOKENIZERS_PARALLELISM=false

BANK=${BANK:-lquant_prompt_bank_40slots_modbulk_gcg6_20260502}
OUT_TAG=${OUT_TAG:-password_bank_perfam_scores_may_v1_0p6b_${TAG_DATE:-$(date +%Y%m%d)}}

hostname; date -u
which "$PYTHON"; "$PYTHON" --version

exec "$PYTHON" -u scripts/password_game/score_password_bank.py \
  --bank-root "$TASK_BANK_ROOT/$BANK" \
  --bank-seed 11 \
  --base-model Qwen/Qwen3-0.6B \
  --password-policy "${PASSWORD_POLICY:-non_disclosure}" \
  --n-slots 40 \
  --out-dir "$RESULTS_ROOT/$OUT_TAG"
