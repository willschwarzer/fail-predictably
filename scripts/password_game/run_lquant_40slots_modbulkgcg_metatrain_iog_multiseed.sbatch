#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq40_modbulk_iog_ms
# Hardware: one GPU of compute capability 8.0 or newer (Ampere or later).
set -euo pipefail

# Multi-seed IOG mod-bulk + GCG meta-train: the same runs as
# run_lquant_40slots_modbulkgcg_metatrain_multiseed.sbatch with the improving-only mask.
# These runs (seeds 11 17 23 29 31; the paper's tags end _20260502 for seed 11 and
# _20260504 for the rest, so set TAG_DATE to match) give the masked arm of the earlier
# recipe in the table of the appendix "The improving-only mask and hidden rare modes".
# Both arms keep each password's fit/deployment split fixed for the whole run. Our runs
# recorded --bootstrap-permutations 1, but the trainer of May 2-4 drew new splits only
# for values above 1. In this version 1 draws a new split at every step, so the flag
# below is 0, which keeps the split fixed as in our runs.

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

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=16
elif (( VRAM_MB < 30000 )); then MICRO_BATCH=16
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=48
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=96
else MICRO_BATCH=128; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"

TAG=lquant_40slots_modbulkgcg_metatrain_iog_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant \
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
  --ols-top-k 10 \
  --plotting-position weibull \
  --extrapolated-weighting deploy_log_uniform \
  --extrapolated-loss-space log_prob \
  --loss-mode ols_primal_extrapolated \
  --grad-mode both \
  --bootstrap-permutations 0 \
  --steps 50 \
  --lr 1e-5 \
  --grad-clip 1.0 \
  --micro-batch-size "${MICRO_BATCH}" \
  --eval-cache-top-b 200 \
  --eval-cache-refresh-interval 10 \
  --seed "${SEED}" \
  --kl-coef 0.05 \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets password \
  --kl-eval-datasets "$WILDCHAT_DATASET" \
  --log-every 5 \
  --save-every 5 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --improving-only-grad \
  --prompt-bank-root "$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_20260502" \
  --prompt-bank-seed 11
