#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_qwen3_8b_resume
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# Resume an in-flight 8B PoC seed onto a different GPU type. Reads the
# same hyperparameter stack as run_lquant_modbulkgcg_qwen3_8b_iog_full_1gpu.sbatch;
# the only differences are: a different output directory (TAG_SUFFIX) and an
# explicit checkpoint to resume from (RESUME_CKPT). Pick the GPU type at
# submission time (e.g. with SBATCH_CONSTRAINT or SBATCH_PARTITION).
# Inputs: RESUME_CKPT, a resume_latest.pt from a run of the 1-GPU launcher.
#
# Seed 23 of the first Qwen3-8B run (first row of the training-budget table in
# the appendix "Password-game scaling and the role of the evaluation bank") is
# this launcher with SEED=23 STEPS=200 TAG_SUFFIX=superpod_1gpu TAG_DATE=20260506,
# resumed from a step-65 checkpoint of an earlier seed-23 run of the same recipe.
#
# Submit example:
#   SEED=17 STEPS=200 BANK_TAG=lquant_prompt_bank_40slots_modbulk_gcg6_qwen3_8b_20260506 \
#     RESUME_CKPT=artifacts/results/.../seed_17/resume_latest.pt \
#     TAG_SUFFIX=l40s_resumetest \
#     sbatch scripts/password_game/run_lquant_modbulkgcg_qwen3_8b_iog_full_resume.sbatch

SEED=${SEED:?SEED env var required}
STEPS=${STEPS:-200}
RESUME_CKPT=${RESUME_CKPT:?RESUME_CKPT env var required}
TAG_SUFFIX=${TAG_SUFFIX:?TAG_SUFFIX env var required}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-online}

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d ' ')
if [[ -n "${MICRO_BATCH:-}" ]]; then
  : # honor explicit override
elif [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=8
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=8
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=16
else MICRO_BATCH=24; fi
echo "[gpu] ${GPU_NAME} vram=${VRAM_MB}MB micro_batch=${MICRO_BATCH}"

TAG=lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_qwen3_8b_1gpu_steps${STEPS}_seed${SEED}_${TAG_SUFFIX}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"

# Auto-resume on Slurm requeue: if THIS run's OUTDIR already has a checkpoint
# (from a previous attempt that got preempted), prefer that over the initial
# RESUME_CKPT so we keep training forward, not restart from the seed-source.
EFFECTIVE_RESUME="$RESUME_CKPT"
if [[ -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt (overrides initial RESUME_CKPT)"
  EFFECTIVE_RESUME="$OUTDIR/resume_latest.pt"
fi

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant \
  --model "Qwen/Qwen3-8B" \
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
  --steps "${STEPS}" \
  --lr 1e-5 \
  --grad-clip 1.0 \
  --micro-batch-size "${MICRO_BATCH}" \
  --eval-cache-top-b 0 \
  --eval-cache-refresh-interval 0 \
  --bootstrap-permutations 1 \
  --union-cache-top-c 296 \
  --union-cache-refresh-interval 5 \
  --baseline-eval-cache-dir "$REPO_ROOT/artifacts/cache/baseline_eval" \
  --seed "${SEED}" \
  --kl-coef 0.05 \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets password \
  --log-every 5 \
  --save-every 1 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --improving-only-grad \
  --iog-scope both \
  --prompt-bank-root "$TASK_BANK_ROOT/${BANK_TAG:?BANK_TAG env var required}" \
  --prompt-bank-seed 11 \
  --resume-checkpoint "${EFFECTIVE_RESUME}"
