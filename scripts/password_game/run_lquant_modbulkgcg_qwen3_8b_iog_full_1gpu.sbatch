#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_qwen3_8b_iogfull_1gpu
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# Single-GPU launcher for the Qwen3-8B PoC. Requires an 80GB GPU (A100-80G or
# H100). The default of 75 fine-tuning steps fits comfortably in a 24h walltime
# if per-step is ~17 min as estimated.
#
# The first Qwen3-8B run (first row of the training-budget table in the appendix
# "Password-game scaling and the role of the evaluation bank") used this recipe
# with STEPS=200, BANK_TAG=lquant_prompt_bank_40slots_modbulk_gcg6_qwen3_8b_20260506
# and seeds 11 17 23 (tags ..._1gpu_steps200_seed<seed>_20260506). Each seed was
# then continued from a mid-run checkpoint: seeds 11 and 17 on three GPUs with
# run_lquant_modbulkgcg_qwen3_8b_iog_full_resume_3gpu.sbatch, seed 23 on one GPU
# with run_lquant_modbulkgcg_qwen3_8b_iog_full_resume.sbatch. The reported numbers
# come from those continuations.

SEED=${SEED:?SEED env var required}
STEPS=${STEPS:-75}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-online}

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=8
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=16
else MICRO_BATCH=24; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"

TAG=lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_qwen3_8b_1gpu_steps${STEPS}_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"

# Auto-resume on Slurm requeue: if a previous attempt left a checkpoint in
# OUTDIR, pick up from it instead of starting from step 0. The cleanup logic
# in train_lquant.py deletes resume_latest.pt on successful completion, so a
# fresh sbatch into a clean OUTDIR starts from scratch as expected.
EXTRA_ARGS=()
if [[ -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt"
  EXTRA_ARGS+=(--resume-checkpoint "$OUTDIR/resume_latest.pt")
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
  "${EXTRA_ARGS[@]}"
