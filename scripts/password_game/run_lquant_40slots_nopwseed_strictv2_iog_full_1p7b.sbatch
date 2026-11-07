#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq40_nopwseed_1p7b
# Hardware: one GPU with at least 40 GB and compute capability 8.0 or newer.
set -euo pipefail

# Size-parameterized warm-start v2 password training. IDENTICAL recipe to the
# 0.6B launcher (steps 200, lr 1e-5, split, OLS/Weibull, KL, IOG, bootstrap+cache)
# -- only --model, the prompt bank, the baseline cache dir, and the micro-batch
# change per size. Micro-batch comes from the model-aware rule in
# _micro_batch_rule.sh: the 0.6B VRAM-only ladder is model-blind and OOMs larger
# models, and the hardcoded MICRO_BATCH=16 that replaced it here was VRAM-blind
# and OOM'd too (see the 4B/8B launchers).
# Set MICRO_BATCH in the environment to override the rule.
# Required env: SEED. Optional: MICRO_BATCH, STEPS, TAG_DATE.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
# Local directory of WildChat arrow shards (searched recursively); the KL eval reads the first shard.
WILDCHAT_DATASET="${WILDCHAT_DATASET:-$REPO_ROOT/data/wildchat}"

SEED=${SEED:?SEED required}
MODEL="Qwen/Qwen3-1.7B"
BANK_ROOT="$TASK_BANK_ROOT/lquant_prompt_bank_40slots_nopwseed_gcg6warm_strictv2_qwen3_1_7b_20260725"
CACHE_DIR="$REPO_ROOT/artifacts/cache/baseline_eval_nopwseed_strictv2_1_7b"
SIZE_TAG="1_7b"
STEPS=${STEPS:-200}
# BASELINE_ONLY=1 runs the untouched-model heldout eval suite and exits. Used to
# ask whether 1.7B has a FORECASTABLE tail at all: the v2 GCG attack fails there
# (a probe floored it 11-13 orders below the 0.1 success threshold), but
# that threshold measures attack success, not forecast error. A deeply-negative
# but mispredicted tail is still a valid training target.
MODEL_HIDDEN=2048 MODEL_LAYERS=28 MODEL_PARAMS_B=1.7
source "$REPO_ROOT/scripts/slurm/_micro_batch_rule.sh"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
echo "[sized] model=${MODEL} size=${SIZE_TAG} micro-batch=${MICRO_BATCH} seed=${SEED}"

TAG=lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_qwen3_${SIZE_TAG}_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"

EXTRA_ARGS=()
# Gradient checkpointing is REQUIRED at this size, not an optimization. The
# scoring call retains the autograd graph for every micro-batch until the
# caller's backward (see the comment in compute_password_logprobs_batched), so
# micro-batch does NOT bound peak memory on the grad path: 4B OOM'd at 43.7 GiB
# with micro-batch 16 and again at 43.6 GiB with micro-batch 7
# -- the same footprint. Checkpointing is what cuts the retained
# graph. Excluded from _critical_cfg_signature; results are unchanged.
if [[ "${GRAD_CKPT:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi
if [[ "${BASELINE_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--baseline-only)
fi
if [[ "${BASELINE_ONLY:-0}" != "1" && -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt"
  EXTRA_ARGS+=(--resume-checkpoint "$OUTDIR/resume_latest.pt")
fi

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
  --steps "${STEPS}" \
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
  "${EXTRA_ARGS[@]}"
