#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lqpw
# Hardware: one GPU with 48 GB or more (the paper used A100-80GB, H100, L40S and A40 cards).
set -euo pipefail

# One arm of the password-v2 anchpen fleet (paper appendix "Forecastability training
# with minimal change to the model").
#
# Cell: password v2, i.e. the 0.6B strict_v2 warm-start bank (built by
# scripts/banks/submit_strictv2_warmstart_bank.sh) as the paper's 10-seed password
# IOG fleet used it (40 slots, 33 train / 7 held-out per seed, 935-prompt pools,
# fit 44 / deploy 891, k10). Every task flag below is copied from that fleet's
# launcher, which is not in this release, so a seed here holds out the same 7
# passwords as the same seed of that fleet.
#
# Run settings follow the WildChat anchpen fleet (submit_lqdyn_fleet.sh
# poolscale wave): lr 2e-5 x 100 steps, 12 of the 33 train targets sampled per
# step, 16 bootstrap partitions per step, gradient checkpointing, adapter
# snapshots every 10 steps for snapshot_heldout_eval.py. The arm knobs below
# default to the paper's IOG + KL (no anchor, no band); the submit line sets
# the anchpen terms.
#
# Band geometry for M44 / N891 / k10: the deepest rank any split reads is
# L = 433 (g_transfer_k.reach, 19 extrapolated deploy ranks), so the sigma
# reference straightens the base pool's top 433 ranks, the constraint channel
# scores the top 533, and the union cache holds 433 (the fit top-10 is scored
# fresh every step).
#
# Required env:
#   ARM    short arm name, goes in the run dir (e.g. pw2apsp, pw2nbap, pw2iog)
#   SEED   training seed (also picks the held-out passwords)
# Optional env (see run_lqdyn_arm.sbatch for the meaning of each):
#   BOOT (16), IOG (1), IOG_SCOPE (both), ANCHOR_MARGIN (0), ANCHOR_PENALTY (0),
#   SIGMA_FLOOR_LAMBDA (0), SIGMA_FLOOR_MARGIN (0), SIGMA_FLOOR_SYMMETRIC (0),
#   SIGMA_REF_M (433), GAP_FLOOR_M (433, sizes the constraint channel),
#   UNION_CACHE_C (433), UNION_REFRESH (5), CERT_DISTILL_SHARED_PASS (0),
#   TARGETS_PER_STEP (12), STEPS (100), LR (2e-5), KL_COEF (0.05),
#   SNAPSHOT_EVERY (10), SAVE_EVERY (10), DUMP_EVERY (2), MICRO_BATCH,
#   TRAIN_TARGETS / HELDOUT_TARGETS (33 / 7; smaller only for smoke tests)
#   PWBANK v2 (default) | v1: v1 is the paper's earlier password bank
#          (scripts/password_game/run_lquant_modbulkgcg_iog_full_bootstrap_cached_multiseed.sbatch:
#          non_disclosure policy, the ..._gcg6_20260502 bank from
#          scripts/banks/launch_gcg_bank_for_mutate_passwords.sh at TAG_DATE=20260502,
#          its own baseline cache),
#          identical otherwise. Name v1 arms pw1* so run dirs do not collide.
#   WILDCHAT_DATASET (default $REPO_ROOT/data/wildchat): the KL evaluation prompts, as in
#          the password launchers under scripts/password_game/. The trainer's
#          load_kl_dataset reads the first .arrow shard (sorted by name) under this
#          directory and samples 256 first-turn prompts, so pass a local Hugging Face
#          copy of allenai/WildChat-1M, e.g. its datasets cache directory
#          .../allenai___wild_chat-1_m/default/0.0.0/<revision>; the paper used
#          revision 8016827cee8c947143c498df467af52753e5181b. A hub id does not work here.

ARM=${ARM:?ARM env var required}
SEED=${SEED:?SEED env var required}
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
KL_EVAL_DIR="${WILDCHAT_DATASET:-$REPO_ROOT/data/wildchat}"
[[ -d "$KL_EVAL_DIR" ]] || { echo "WILDCHAT_DATASET=$KL_EVAL_DIR is not a directory: point it at a local arrow copy of allenai/WildChat-1M (see the header)" >&2; exit 1; }
PWBANK=${PWBANK:-v2}
case "$PWBANK" in
  v2) POLICY=non_disclosure_strict_v2
      BANK_ROOT=$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6warm_strictv2_qwen3_0_6b_20260724
      BASE_CACHE=$REPO_ROOT/artifacts/cache/baseline_eval_gcg6warm_strictv2 ;;
  v1) POLICY=non_disclosure
      BANK_ROOT=$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_20260502
      BASE_CACHE=$REPO_ROOT/artifacts/cache/baseline_eval ;;
  *) echo "PWBANK must be v1 or v2" >&2; exit 1 ;;
esac
STEPS=${STEPS:-100}
LR=${LR:-2e-5}
KL_COEF=${KL_COEF:-0.05}
BOOT=${BOOT:-16}
IOG=${IOG:-1}
IOG_SCOPE=${IOG_SCOPE:-both}
ANCHOR_MARGIN=${ANCHOR_MARGIN:-0}
ANCHOR_PENALTY=${ANCHOR_PENALTY:-0}
SIGMA_FLOOR_LAMBDA=${SIGMA_FLOOR_LAMBDA:-0}
SIGMA_FLOOR_MARGIN=${SIGMA_FLOOR_MARGIN:-0}
SIGMA_FLOOR_SYMMETRIC=${SIGMA_FLOOR_SYMMETRIC:-0}
SIGMA_SYM_FLAG=""
if [[ "$SIGMA_FLOOR_SYMMETRIC" == "1" ]]; then SIGMA_SYM_FLAG=--sigma-floor-symmetric; fi
SIGMA_REF_M=${SIGMA_REF_M:-433}
GAP_FLOOR_M=${GAP_FLOOR_M:-433}
UNION_CACHE_C=${UNION_CACHE_C:-433}
UNION_REFRESH=${UNION_REFRESH:-5}
TARGETS_PER_STEP=${TARGETS_PER_STEP:-12}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-10}
DUMP_EVERY=${DUMP_EVERY:-2}

hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module list 2>&1 || true
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # the model must already be in the Hugging Face cache
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "$PYTHON"; "$PYTHON" --version

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ -n "${MICRO_BATCH:-}" ]]; then
  echo "[mb] using override MICRO_BATCH=${MICRO_BATCH}"
elif [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=16
elif (( VRAM_MB < 30000 )); then MICRO_BATCH=16
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=48
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=96
else MICRO_BATCH=128; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"

TAG=lqpw_${ARM}_seed${SEED}
OUTDIR="$RESULTS_ROOT/lquant_dynamics/runs/$TAG"
mkdir -p "$OUTDIR"
echo "[run] OUTDIR=$OUTDIR"
echo "[run] ARM=$ARM BOOT=$BOOT IOG=$IOG ANCHOR=$ANCHOR_MARGIN/$ANCHOR_PENALTY" \
     "SIGMA=$SIGMA_FLOOR_LAMBDA/$SIGMA_FLOOR_MARGIN/sym$SIGMA_FLOOR_SYMMETRIC ref_m=$SIGMA_REF_M" \
     "SP=${CERT_DISTILL_SHARED_PASS:-0} TPS=$TARGETS_PER_STEP STEPS=$STEPS LR=$LR"

EXTRA_ARGS=()
if [[ -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt"
  EXTRA_ARGS+=(--resume-checkpoint "$OUTDIR/resume_latest.pt")
fi
EXTRA_ARGS+=(--gradient-checkpointing)
if [[ "$IOG" == "1" ]]; then
  EXTRA_ARGS+=(--improving-only-grad --iog-scope "$IOG_SCOPE")
fi
if [[ "${CERT_DISTILL_SHARED_PASS:-0}" == "1" ]]; then EXTRA_ARGS+=(--cert-distill-shared-pass); fi

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant \
  --model "Qwen/Qwen3-0.6B" \
  --password-slots 40 \
  --password-policy "$POLICY" \
  --train-target-count "${TRAIN_TARGETS:-33}" \
  --heldout-target-count "${HELDOUT_TARGETS:-7}" \
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
  --steps "$STEPS" \
  --lr "$LR" \
  --grad-clip 1.0 \
  --micro-batch-size "$MICRO_BATCH" \
  --eval-cache-top-b 0 \
  --eval-cache-refresh-interval 0 \
  --bootstrap-permutations "$BOOT" \
  --union-cache-top-c "$UNION_CACHE_C" \
  --union-cache-refresh-interval "$UNION_REFRESH" \
  --targets-per-step "$TARGETS_PER_STEP" \
  --anchor-gate-margin "$ANCHOR_MARGIN" \
  --anchor-penalty-coef "$ANCHOR_PENALTY" \
  --gap-floor-m "$GAP_FLOOR_M" \
  --sigma-floor-lambda "$SIGMA_FLOOR_LAMBDA" \
  --sigma-floor-margin "$SIGMA_FLOOR_MARGIN" \
  $SIGMA_SYM_FLAG \
  --sigma-ref-m "$SIGMA_REF_M" \
  --dynamics-dump-dir "$OUTDIR/dynamics" \
  --dynamics-dump-every "$DUMP_EVERY" \
  --baseline-eval-cache-dir "$BASE_CACHE" \
  --seed "$SEED" \
  --kl-coef "$KL_COEF" \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets password \
  --kl-eval-datasets "$KL_EVAL_DIR" \
  --log-every 5 \
  --save-every "$SAVE_EVERY" \
  --adapter-snapshot-every "$SNAPSHOT_EVERY" \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --prompt-bank-root "$BANK_ROOT" \
  --prompt-bank-seed 11 \
  "${EXTRA_ARGS[@]}"
