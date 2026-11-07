#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_pw_besteval
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# Held-out eval of a saved best/ adapter from a rebuilt-bank (nopwseed strictv2)
# password run. Same purpose as
# scripts/wildchat/natural_prompts/run_lquant_ldnoobw_eval_adapter.sbatch: the
# trainer's best/ checkpoint is selected on TRAIN-side forecast error only, and
# this job settles what that checkpoint does on the held-out targets, mid-run,
# without waiting ~1 day for the fleet endpoint. Config mirrors the size
# launchers exactly where it affects the eval (model, bank, seed, split, KL
# datasets); LoRA stays at trainer defaults (r8) because the fleet trains at
# defaults. Baselines are cache hits from the fleet's shared CACHE_DIR.
#
# Env: ADAPTER (required, frozen dir with adapter_model.safetensors)
#      ARM     (required, outdir tag)
#      SEED    (required)
#      SIZE    (required, 4b | 8b)

ADAPTER=${ADAPTER:?ADAPTER required}
ARM=${ARM:?ARM required}
SEED=${SEED:?SEED required}
SIZE=${SIZE:?SIZE required (4b|8b)}

case "$SIZE" in
  4b)
    MODEL="Qwen/Qwen3-4B"
    MODEL_HIDDEN=2560 MODEL_LAYERS=36 MODEL_PARAMS_B=4.0
    ;;
  8b)
    MODEL="Qwen/Qwen3-8B"
    MODEL_HIDDEN=4096 MODEL_LAYERS=36 MODEL_PARAMS_B=8.2
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
CACHE_DIR="$REPO_ROOT/artifacts/cache/baseline_eval_nopwseed_strictv2_${SIZE}"
source "$REPO_ROOT/scripts/slurm/_micro_batch_rule.sh"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

TAG=lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_qwen3_${SIZE}_seed${SEED}_besteval_${ARM}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"
echo "[run] OUTDIR=$OUTDIR"
echo "[run] ADAPTER=$ADAPTER SIZE=$SIZE SEED=$SEED micro-batch=$MICRO_BATCH"

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant \
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
  --ols-top-k 10 \
  --plotting-position weibull \
  --extrapolated-weighting deploy_log_uniform \
  --extrapolated-loss-space log_prob \
  --loss-mode ols_primal_extrapolated \
  --grad-mode both \
  --steps 200 \
  --lr 1e-5 \
  --grad-clip 1.0 \
  --micro-batch-size "${MICRO_BATCH}" \
  --eval-cache-top-b 0 \
  --eval-cache-refresh-interval 0 \
  --bootstrap-permutations 1 \
  --union-cache-top-c 296 \
  --union-cache-refresh-interval 5 \
  --baseline-eval-cache-dir "${CACHE_DIR}" \
  --seed "${SEED}" \
  --kl-coef 0.05 \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets password \
  --kl-eval-datasets "$WILDCHAT_DATASET" \
  --log-every 10 \
  --save-every 20 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --improving-only-grad \
  --iog-scope both \
  --prompt-bank-root "${BANK_ROOT}" \
  --prompt-bank-seed 11 \
  --gradient-checkpointing \
  --eval-adapter-from "$ADAPTER"
