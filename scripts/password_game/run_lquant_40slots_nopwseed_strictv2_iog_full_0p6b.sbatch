#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq40_nopwseed_0p6b
# Hardware: one GPU with at least 40 GB and compute capability 8.0 or newer.
set -euo pipefail

# Warm-start v2 password bank version of run_lquant_modbulkgcg_iog_full_bootstrap
# _cached_multiseed.sbatch. IDENTICAL recipe (LoRA/steps/lr/split/OLS/Weibull/KL/
# IOG/bootstrap+cache) -- ONLY the prompt bank (warm-start GCG under v2),
# --password-policy (non_disclosure_strict_v2), the baseline cache dir (fresh, so
# the baseline is recomputed against the new bank), and the output tag change.

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
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

STEPS=${STEPS:-200}

# VRAM-only micro-batch ladder, calibrated for a 0.6B policy. It is model-blind
# and must NOT be copied to larger sizes -- see _micro_batch_rule.sh, which the
# 4B/8B launchers use instead. Because cards on this cluster are heterogeneous,
# this ladder gives DIFFERENT micro-batch to different seeds of the same fleet
# (observed: {48: 7, 96: 2, 16: 1} across the ten 0.6B v2 seeds). That is
# gradient-accumulation chunking only -- same total gradient per optimizer step,
# excluded from _critical_cfg_signature -- but it does change float summation
# order, so seeds are not bitwise comparable across cards. Set MICRO_BATCH in the
# environment to pin it.
if [[ -n "${MICRO_BATCH:-}" ]]; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH} (EXPLICIT override, ladder skipped)"
else
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=8
elif (( VRAM_MB < 20000 )); then MICRO_BATCH=8
elif (( VRAM_MB < 30000 )); then MICRO_BATCH=16
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=48
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=96
else MICRO_BATCH=128; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"
fi

TAG=lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"

EXTRA_ARGS=()
if [[ -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt"
  EXTRA_ARGS+=(--resume-checkpoint "$OUTDIR/resume_latest.pt")
fi

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant \
  --model "Qwen/Qwen3-0.6B" \
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
  --baseline-eval-cache-dir "$REPO_ROOT/artifacts/cache/baseline_eval_nopwseed_strictv2" \
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
  --prompt-bank-root "$TASK_BANK_ROOT/lquant_prompt_bank_40slots_nopwseed_gcg6warm_strictv2_qwen3_0_6b_20260725" \
  --prompt-bank-seed 11 \
  "${EXTRA_ARGS[@]}"
