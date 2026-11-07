#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_sft_nopwseed
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# Direct-risk-minimization SFT baseline on the rebuilt (nopwseed strictv2)
# password banks, at 4B/8B. The paper's main SFT baseline is at 0.6B
# (2000 steps x batch 128 = 256k prompt forwards, matched to meta-training's
# forward count); these runs give the 4B and 8B SFT results in the appendix
# "Password-game scaling and the role of the evaluation bank". Total forward count is held at 256k per run;
# batch/steps are rebalanced per size for activation memory, with gradient
# checkpointing (additive flag, identical numerics) doing the heavy lifting.
#
# Env: SEED (required), SIZE (required, 4b | 8b)

SEED=${SEED:?SEED required}
SIZE=${SIZE:?SIZE required (4b|8b)}

case "$SIZE" in
  4b)
    MODEL="Qwen/Qwen3-4B"
    MICRO_BATCH=64 SFT_BATCH=64 SFT_STEPS=4000
    ;;
  8b)
    MODEL="Qwen/Qwen3-8B"
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
# Local directory of WildChat arrow shards (searched recursively); the KL eval reads the first shard.
WILDCHAT_DATASET="${WILDCHAT_DATASET:-$REPO_ROOT/data/wildchat}"
BANK_ROOT="$TASK_BANK_ROOT/lquant_prompt_bank_40slots_nopwseed_gcg6warm_strictv2_qwen3_${SIZE}_20260725"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

TAG=lquant_40slots_nopwseed_strictv2_sft_union_qwen3_${SIZE}_seed${SEED}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"
echo "[run] OUTDIR=$OUTDIR"
echo "[run] SIZE=$SIZE SEED=$SEED sft-batch=$SFT_BATCH sft-steps=$SFT_STEPS micro-batch=$MICRO_BATCH"

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant_sft \
  --model "${MODEL}" \
  --password-slots 40 \
  --password-policy non_disclosure_strict_v2 \
  --train-target-count 33 \
  --heldout-target-count 7 \
  --cheap-prompts-per-family-per-target 64 \
  --cheap-prompts-per-family-map \
      compositional=932 mutate=466 indirect_extraction=466 gcg_steps80=6 \
  --train-families compositional mutate indirect_extraction gcg_steps80 \
  --test-families compositional mutate indirect_extraction gcg_steps80 \
  --shared-family-test-fraction 0.5 \
  --split-mode disjoint \
  --fit-fraction 0.0476 \
  --steps "${SFT_STEPS}" \
  --ols-top-k 10 \
  --plotting-position weibull \
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
  --save-every 200 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --sft-subset union \
  --sft-batch-per-step "${SFT_BATCH}" \
  --gradient-checkpointing \
  --prompt-bank-root "${BANK_ROOT}" \
  --prompt-bank-seed 11
