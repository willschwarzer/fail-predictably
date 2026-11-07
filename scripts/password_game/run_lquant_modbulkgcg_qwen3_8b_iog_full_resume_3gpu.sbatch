#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gres=gpu:3
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_qwen3_8b_resume_3gpu
# Hardware: three 80 GB GPUs on one node (the paper used A100-80GB).
set -euo pipefail

# 3-GPU DDP resume (used for the paper's first 8B run). 33 train targets / 3 ranks = 11 each,
# symmetric — sidesteps the rank-divergence issue that broke 2-GPU DDP.
# Inputs: RESUME_CKPT, a resume_latest.pt from run_lquant_modbulkgcg_qwen3_8b_iog_full_1gpu.sbatch.
# The paper's seeds 11 and 17 (SEED=11/17 STEPS=200 TAG_SUFFIX=superpod_3gpu
# TAG_DATE=20260506) resumed the step-44 and step-40 checkpoints of
# lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_qwen3_8b_1gpu_steps200_seed<seed>_20260506.

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
export OMP_NUM_THREADS=4

MICRO_BATCH=${MICRO_BATCH:-8}
echo "[gpu] $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) per_rank_micro_batch=${MICRO_BATCH}"

TAG=lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_qwen3_8b_3gpu_steps${STEPS}_seed${SEED}_${TAG_SUFFIX}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"

EFFECTIVE_RESUME="$RESUME_CKPT"
if [[ -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt (overrides initial RESUME_CKPT)"
  EFFECTIVE_RESUME="$OUTDIR/resume_latest.pt"
fi

exec "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m forecast_experiment.run_train_lquant \
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
