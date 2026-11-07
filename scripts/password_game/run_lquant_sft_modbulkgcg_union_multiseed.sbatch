#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq40_modbulk_sft_union
# Hardware: one GPU of compute capability 8.0 or newer (Ampere or later).
set -euo pipefail

# Direct-risk-minimization SFT baseline on the mod-bulk + GCG bank.
# Subset: union(fit, deploy). Tests "is the meta-training forecast improvement
# just a side-effect of pushing leak prob down on the same prompts?"

SEED=${SEED:?SEED env var required}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
# Local directory of WildChat arrow shards (searched recursively); the KL eval reads the first shard.
WILDCHAT_DATASET="${WILDCHAT_DATASET:-$REPO_ROOT/data/wildchat}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-online}

# Our runs sampled 32 records per step for 8,000 steps and put 16 records through the
# model at a time. The micro-batch size changes the update only through rounding.
SFT_BATCH="${SFT_BATCH:-32}"
SFT_STEPS="${SFT_STEPS:-8000}"
MICRO_BATCH="${MICRO_BATCH:-16}"
echo "[sft] micro-batch ${MICRO_BATCH}, sft-batch ${SFT_BATCH}, sft-steps ${SFT_STEPS}"

# Our five runs evaluated their forecasts with the trainer's default plotting positions (Gringorten)
# by mistake. The paper's Figure 4 uses the Weibull positions for them, as for every other method.

TAG=lquant_40slots_modbulkgcg_sft_union_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant_sft \
  --model "Qwen/Qwen3-0.6B" \
  --password-slots 40 \
  --train-target-count 33 \
  --heldout-target-count 7 \
  --cheap-prompts-per-family-per-target 64 \
  --cheap-prompts-per-family-map \
      compositional=466 mutate=466 indirect_extraction=466 password_seeded=466 gcg_steps80=6 \
  --train-families compositional mutate indirect_extraction password_seeded gcg_steps80 \
  --test-families compositional mutate indirect_extraction password_seeded gcg_steps80 \
  --shared-family-test-fraction 0.5 \
  --split-mode disjoint \
  --fit-fraction 0.0476 \
  --steps "${SFT_STEPS}" \
  --lr 1e-5 \
  --grad-clip 1.0 \
  --micro-batch-size "${MICRO_BATCH}" \
  --seed "${SEED}" \
  --kl-coef 0.05 \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets password \
  --kl-eval-datasets "$WILDCHAT_DATASET" \
  --log-every 50 \
  --save-every 50 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --plotting-position weibull \
  --sft-subset union \
  --sft-batch-per-step "${SFT_BATCH}" \
  --prompt-bank-root "$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_20260502" \
  --prompt-bank-seed 11
