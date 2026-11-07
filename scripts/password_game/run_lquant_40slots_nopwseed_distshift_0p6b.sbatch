#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq40_distshift_0p6b
# Hardware: one GPU with at least 40 GB and compute capability 8.0 or newer.
set -euo pipefail

# Fit/deploy DISTRIBUTION-SHIFT arm of run_lquant_40slots_nopwseed_strictv2_iog
# _full_0p6b.sbatch (appendix "Fit/deployment distribution shift"). IDENTICAL recipe,
# bank, and cell (fit 44 / deploy 891 per slot, train and held-out alike) except:
#   * --fit-families compositional: the 44-prompt fit sample is drawn only from
#     the family that is clean at 0.6B.
#   * --deploy-families indirect_extraction mutate gcg_steps80: the 891-prompt
#     deploy set holds the disjoint families (443 indirect + 442 mutate + all 6
#     warm-start GCG per slot; largest-remainder thinning from the full pools).
#   * --bootstrap-permutations 0 (union caches dropped with it): the bootstrap
#     re-partitions the fit/deploy union each step, which assumes the sides are
#     exchangeable -- the assumption this experiment removes. The loss path is
#     the pre-bootstrap full-scoring path (eval cache off).
#   * fresh baseline cache dir (the key also differs via fit/deploy extras).
#   * MICRO_BATCH pinned to 48, so it does not vary with the card.

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
MICRO_BATCH=${MICRO_BATCH:-48}
echo "[cfg] seed=${SEED} steps=${STEPS} micro-batch ${MICRO_BATCH} (pinned)"

TAG=lquant_40slots_nopwseed_distshift891_strictv2_iog_full_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
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
  --fit-families compositional \
  --deploy-families indirect_extraction mutate gcg_steps80 \
  --shift-fit-size 44 \
  --shift-deploy-size 891 \
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
  --bootstrap-permutations 0 \
  --baseline-eval-cache-dir "$REPO_ROOT/artifacts/cache/baseline_eval_nopwseed_strictv2_distshift" \
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
