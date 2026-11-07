#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_sft_distshift_0p6b
# Hardware: one GPU with at least 40 GB and compute capability 8.0 or newer.
set -euo pipefail

# SFT comparator for the 0.6B fit/deploy DISTRIBUTION-SHIFT cell. Same paper
# 0.6B SFT recipe (direct risk minimization on union(fit, deploy) of the train
# slots, 2000 steps x batch 128 = 256k prompt forwards, lr 1e-5, KL 0.05) --
# the ONLY change from a canonical-bank SFT run is the fit/deploy family split,
# matching run_lquant_40slots_nopwseed_distshift_0p6b.sbatch: fit 44 from
# compositional, deploy 891 = 443 indirect + 442 mutate + 6 warm-start GCG.
# The union it trains on is therefore the same 935 prompts per train slot as
# the canonical union, re-partitioned. Held-out evaluation is identical to the
# meta-training runs (44/891 family split per held-out slot).
# --gradient-checkpointing is required even at 0.6B: the SFT loop retains the
# whole 128-prompt batch's activation graphs before its single backward, which
# OOM'd an 80GB card without it (first step). Additive flag,
# identical numerics (same as the 4B/8B SFT lane).
#
# Env: SEED (required)

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
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

MICRO_BATCH=${MICRO_BATCH:-48}
SFT_BATCH=${SFT_BATCH:-128}
SFT_STEPS=${SFT_STEPS:-2000}
echo "[cfg] seed=${SEED} sft-batch=${SFT_BATCH} sft-steps=${SFT_STEPS} micro-batch=${MICRO_BATCH}"

TAG=lquant_40slots_nopwseed_distshift891_sft_union_qwen3_0_6b_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"
echo "[run] OUTDIR=$OUTDIR"

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant_sft \
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
  --save-every 200 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --sft-subset union \
  --sft-batch-per-step "${SFT_BATCH}" \
  --gradient-checkpointing \
  --prompt-bank-root "$TASK_BANK_ROOT/lquant_prompt_bank_40slots_nopwseed_gcg6warm_strictv2_qwen3_0_6b_20260725" \
  --prompt-bank-seed 11
