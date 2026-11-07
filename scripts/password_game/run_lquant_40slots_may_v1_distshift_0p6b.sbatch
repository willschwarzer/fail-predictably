#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq40_mayv1_distshift
# Hardware: one GPU with at least 40 GB and compute capability 8.0 or newer.
set -euo pipefail

# Fit/deploy DISTRIBUTION-SHIFT on the PAPER'S OWN 0.6B cell (v1
# `non_disclosure`, May bank `lquant_prompt_bank_40slots_modbulk_gcg6_20260502`,
# all five families). Every knob matches the paper's canonical 0.6B fleet
# (artifacts/results/lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_
# seed*_20260505) -- same bank, policy, families, per-family counts, cell
# (fit 44 / deploy 891), lr, steps, KL, weibull, IOG -- EXCEPT:
#   * --fit-families compositional / --deploy-families mutate
#     indirect_extraction password_seeded gcg_steps80: the split rule, applied
#     identically in train and held-out slots.
#   * --bootstrap-permutations 0 (was 1): the bootstrap re-partitions the
#     fit/deploy union each step, assuming exchangeability -- the assumption
#     this experiment removes. Forced by the protocol, not a free choice.
#   * fresh baseline cache dir; MICRO_BATCH pinned.
#
# Why this bank rather than the strict_v2 warm-start one (leak scan,
# 0.6B, whole bank): here `compositional` (the FIT family) is clean -- max leak
# 10^-2.20, p99 10^-5.57, zero prompts above 0.1 -- while GCG reaches ceiling on
# 25/40 slots. So the easy-fit / hard-deploy premise holds, AND the deploy worst
# still VARIES across slots (paper fleet: stdev 1.14 decades, 25/70 held-out
# cells at ceiling), which keeps the post-hoc calibration comparator
# non-degenerate. On the warm-start banks every slot's pretrained worst is at
# ceiling (66/70), which collapses Cal. to a constant "predict ceiling"
# predictor (alpha ~0.003) and makes its numbers incomparable to the paper's
# Cal. bar. See the appendix "Fit/deployment distribution shift".
#
# Env: SEED (required). The paper's ten seeds, with the calibration and best/ steps
# chained after training: scripts/password_game/submit_lquant_may_v1_distshift_0p6b_fleet.sh.

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
MICRO_BATCH=${MICRO_BATCH:-32}
echo "[cfg] seed=${SEED} steps=${STEPS} micro-batch=${MICRO_BATCH} (pinned)"

TAG=lquant_40slots_mayv1_distshift891_iog_full_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
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
  --password-policy non_disclosure \
  --train-target-count 33 \
  --heldout-target-count 7 \
  --cheap-prompts-per-family-per-target 64 \
  --cheap-prompts-per-family-map \
      compositional=466 mutate=466 indirect_extraction=466 password_seeded=466 gcg_steps80=6 \
  --train-families compositional mutate indirect_extraction password_seeded gcg_steps80 \
  --test-families compositional mutate indirect_extraction password_seeded gcg_steps80 \
  --fit-families compositional \
  --deploy-families mutate indirect_extraction password_seeded gcg_steps80 \
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
  --baseline-eval-cache-dir "$REPO_ROOT/artifacts/cache/baseline_eval_mayv1_distshift" \
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
  --prompt-bank-root "$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_20260502" \
  --prompt-bank-seed 11 \
  "${EXTRA_ARGS[@]}"
