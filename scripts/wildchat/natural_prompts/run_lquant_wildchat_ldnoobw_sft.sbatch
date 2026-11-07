#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_sft_ldnoobw
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# Direct-risk-minimization SFT comparator on the WildChat LDNOOBW token-union
# banks (paper appendix "Forecastability on natural user prompts"), on the same
# KL track as the mb8+klnf meta-training ladder
# (non-flagged disjoint prompts, coef 0.5). Forward count is matched to the
# paper's SFT convention (2000 x 128 = 256k prompt forwards), rebalanced
# per size for activation memory; best checkpoint is selected by interval
# SFT loss (the comparator's own objective), never by forecast error.
# The appendix's SFT runs are these defaults at seeds 11, 17 and 23 and all four
# sizes; scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh submits
# them, and select_natural_checkpoint.py applies the KL bound of 0.7.
#
# Env: SEED (required), SIZE (required, 0p6b | 1p7b | 4b | 8b), KL_COEF
# (default 0.5), KL_TRAIN (default nonflagged), RESULTS_ROOT (default
# $REPO_ROOT/artifacts/results), TASK_BANK_ROOT (default
# $REPO_ROOT/artifacts/task_banks; banks built by
# scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch), PYTHON (default python).
#
# Inputs: KL_TRAIN=nonflagged (the default) reads KL_TRAIN_JSONL (default
# $BANK_ROOT/bank_seed_11/kl_train_nonflagged_en_disjoint.jsonl): the 1792 prompts
# of the bank's kl_eval_nonflagged_en.jsonl (2048 non-flagged English prompts)
# that the trainer's seeded 256-prompt KL eval does not draw. Write it once per
# bank with scripts/wildchat/natural_prompts/make_kl_train_split.py <bank dir>.

SEED=${SEED:?SEED required}
SIZE=${SIZE:?SIZE required (0p6b|1p7b|4b|8b)}
KL_COEF=${KL_COEF:-0.5}
KL_TRAIN=${KL_TRAIN:-nonflagged}

case "$SIZE" in
  0p6b)
    MODEL="Qwen/Qwen3-0.6B"; BANK_TAG="qwen_qwen3-0_6b"
    MICRO_BATCH=96 SFT_BATCH=128 SFT_STEPS=2000
    ;;
  1p7b)
    MODEL="Qwen/Qwen3-1.7B"; BANK_TAG="qwen_qwen3-1_7b"
    MICRO_BATCH=96 SFT_BATCH=128 SFT_STEPS=2000
    ;;
  4b)
    MODEL="Qwen/Qwen3-4B"; BANK_TAG="qwen_qwen3-4b"
    MICRO_BATCH=64 SFT_BATCH=64 SFT_STEPS=4000
    ;;
  8b)
    MODEL="Qwen/Qwen3-8B"; BANK_TAG="qwen_qwen3-8b"
    MICRO_BATCH=32 SFT_BATCH=32 SFT_STEPS=8000
    ;;
  *) echo "unknown SIZE=$SIZE"; exit 1;;
esac

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
BANK_ROOT="$TASK_BANK_ROOT/wildchat_ldnoobw_${BANK_TAG}_T40_p1870"
KL_TRAIN_JSONL=${KL_TRAIN_JSONL:-$BANK_ROOT/bank_seed_11/kl_train_nonflagged_en_disjoint.jsonl}

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

KL_TRAIN_ARG="password"
if [[ "$KL_TRAIN" == "nonflagged" ]]; then
  KL_TRAIN_ARG="$KL_TRAIN_JSONL"
fi

TAG=lquant_wildchat_ldnoobw_sft_union_${BANK_TAG}_klnf$(echo "$KL_COEF" | tr -d '.')_seed${SEED}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"
echo "[run] OUTDIR=$OUTDIR"
echo "[run] SIZE=$SIZE SEED=$SEED sft-batch=$SFT_BATCH sft-steps=$SFT_STEPS micro-batch=$MICRO_BATCH kl=$KL_COEF/$KL_TRAIN"
cd "$REPO_ROOT"

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant_sft \
  --model "${MODEL}" \
  --task-kind token_union \
  --prompt-bank-root "${BANK_ROOT}" \
  --prompt-bank-seed 11 \
  --train-families wildchat_flagged_en \
  --test-families wildchat_flagged_en \
  --train-target-count 33 \
  --heldout-target-count 7 \
  --shared-family-test-fraction 0.5 \
  --split-mode disjoint \
  --fit-fraction 0.0476 \
  --score-max-len 2048 \
  --score-token-budget 8192 \
  --ols-top-k 10 \
  --plotting-position weibull \
  --steps "${SFT_STEPS}" \
  --lr 1e-5 \
  --grad-clip 1.0 \
  --micro-batch-size "${MICRO_BATCH}" \
  --seed "${SEED}" \
  --kl-coef "${KL_COEF}" \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets "${KL_TRAIN_ARG}" \
  --kl-eval-datasets "$BANK_ROOT/bank_seed_11/kl_eval_nonflagged_en.jsonl" \
  --kl-tokenize-max-len 1024 \
  --kl-truncation-side left \
  --baseline-eval-cache-dir "$REPO_ROOT/artifacts/cache/baseline_eval" \
  --log-every 50 \
  --save-every 200 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --sft-subset union \
  --sft-batch-per-step "${SFT_BATCH}" \
  --gradient-checkpointing \
  --lora-r 8 \
  --lora-alpha 16
