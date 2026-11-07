#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lqdyn
# Hardware: one 80 GB GPU (the paper used A100-80GB and H100; most poolscale-wave arms also fit a 48 GB L40S).
set -euo pipefail

# One arm of the learning-dynamics fleet.
#
# Cell: the canonical WildChat LDNOOBW recipe at 0.6B, cut down to TRAIN_TARGETS
# tasks so that every task is visited every step. Dense per-task visits are the
# point: the study reads the per-step trajectory of each task's score
# distribution, which a meta-batched run only samples every 33/b steps.
#
# Everything not listed as an env knob below is byte-identical to
# scripts/wildchat/natural_prompts/run_lquant_wildchat_ldnoobw_multiseed.sbatch, so an arm differs from the
# reference arm in exactly the knob its name advertises.
#
# Required env:
#   ARM    short arm name, goes in the run dir (e.g. k1, k16, k1anchor8)
#   SEED   training seed
# Optional env (defaults reproduce the reference arm):
#   BOOT               --bootstrap-permutations           (default 1)
#   IOG                1 = --improving-only-grad, 0 = off (default 1)
#   IOG_SCOPE          both | deploy_only | fit_only      (default both)
#   ANCHOR_MARGIN      anchor-gated worsening mask, nats  (default 0 = hard mask)
#   IOG_FIT_MASK       closed_form | exact                 (default closed_form)
#   ANCHOR_PENALTY     squared-hinge penalty above the anchor (default 0)
#   SCALE_NORM_M       divide the loss by the top-m score variance (default 0 = off)
#   GAP_FLOOR_M        one-sided per-gap spread floors on the top-m (default 0 = off)
#   GAP_FLOOR_LAMBDA   hinge weight for the gap floors (exact penalty; default 0)
#   GAP_FLOOR_BLOCK    window size B: floors on B-rank spans, 1 = per-gap (default 1)
#   GAP_FLOOR_SLACK    nats subtracted from each floor (default 1e-4)
#   SIGMA_FLOOR_LAMBDA weight of the line-referenced involved-set sigma floor (default 0)
#   SIGMA_FLOOR_MARGIN nats subtracted from the sigma reference (default 0)
#   SIGMA_FLOOR_SYMMETRIC 1 = two-sided hinge on |sigma - ref| (default 0 = floor only)
#   SIGMA_REF_M        anchor ranks the sigma reference straightens (default 250)
#   TARGET_FULL_POOLS  1 = --target-holdout-full-pools: no within-target halving (default 0)
#   HELDOUT_FROM_END   1 = held-out targets from the end of the seeded shuffle (default 0)
#   SNAPSHOT_EVERY     keep the adapter every N steps under snapshots/ (default 0 = off)
#   SAVE_EVERY         resume-checkpoint cadence (default 20)
#   TARGETS_PER_STEP   sample this many train targets per step (default 0 = all)
#   CERT_DISTILL_COEF  weight of the pull onto the certified tail (default 0 = off)
#   CERT_DISTILL_RANKS anchor ranks the certified targets cover (default 300)
#   CERT_DISTILL_SPACE log_prob | logit (event log-odds; default log_prob)
#   FORECAST_LOSS_COEF weight on the forecast loss (default 1; 0 = distillation only)
#   CERT_DISTILL_SHARED_PASS 1 = --cert-distill-shared-pass: penalty from the channel pass, no
#                      double scoring of channel prompts (needs FORECAST_LOSS_COEF 0; default 0)
#   NPROC              DDP ranks via torchrun on one node; pass the matching
#                      --gres=gpu:N on the sbatch line (default 1 = plain python)
#   OPTIMIZER          adamw | sgd  (sgd needs a much larger LR; see train_lquant)
#   WEIGHTING          deploy_log_uniform | worst_rank | rank_uniform
#   FIT_FRACTION       fit share of the union pool        (default 0.0476 -> 44/891)
#   TRAIN_TARGETS / HELDOUT_TARGETS                       (default 12 / 4)
#   STEPS              (default 200)
#   UNION_REFRESH      union-cache refresh interval       (default 5)
#   DUMP_EVERY         dynamics dump cadence              (default 1)
#   MODEL, BANK_ROOT, LR, KL_COEF, LORA_R, MICRO_BATCH

ARM=${ARM:?ARM env var required}
SEED=${SEED:?SEED env var required}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
TRAIN_TARGETS=${TRAIN_TARGETS:-12}
HELDOUT_TARGETS=${HELDOUT_TARGETS:-4}
SCORE_MAX_LEN=${SCORE_MAX_LEN:-2048}
SCORE_TOKEN_BUDGET=${SCORE_TOKEN_BUDGET:-8192}
STEPS=${STEPS:-200}
LR=${LR:-1e-5}
OPTIMIZER=${OPTIMIZER:-adamw}
KL_COEF=${KL_COEF:-0.05}
OLS_TOP_K=${OLS_TOP_K:-10}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-$((LORA_R * 2))}
BOOT=${BOOT:-1}
IOG=${IOG:-1}
IOG_SCOPE=${IOG_SCOPE:-both}
IOG_WORSENING_COEF=${IOG_WORSENING_COEF:-0}
IOG_FIT_MASK=${IOG_FIT_MASK:-closed_form}
ANCHOR_MARGIN=${ANCHOR_MARGIN:-0}
ANCHOR_PENALTY=${ANCHOR_PENALTY:-0}
WEIGHTING=${WEIGHTING:-deploy_log_uniform}
FIT_FRACTION=${FIT_FRACTION:-0.0476}
UNION_CACHE_C=${UNION_CACHE_C:-296}
UNION_REFRESH=${UNION_REFRESH:-5}
DUMP_EVERY=${DUMP_EVERY:-1}
SCALE_NORM_M=${SCALE_NORM_M:-0}
GAP_FLOOR_M=${GAP_FLOOR_M:-0}
GAP_FLOOR_LAMBDA=${GAP_FLOOR_LAMBDA:-0}
GAP_FLOOR_BLOCK=${GAP_FLOOR_BLOCK:-1}
GAP_FLOOR_SLACK=${GAP_FLOOR_SLACK:-0.0001}
GAP_FLOOR_LEVEL_GATED=${GAP_FLOOR_LEVEL_GATED:-0}
LEVEL_GATED_FLAG=""
if [[ "$GAP_FLOOR_LEVEL_GATED" == "1" ]]; then LEVEL_GATED_FLAG=--gap-floor-level-gated; fi
SIGMA_FLOOR_LAMBDA=${SIGMA_FLOOR_LAMBDA:-0}
SIGMA_FLOOR_MARGIN=${SIGMA_FLOOR_MARGIN:-0}
SIGMA_FLOOR_SYMMETRIC=${SIGMA_FLOOR_SYMMETRIC:-0}
SIGMA_SYM_FLAG=""
if [[ "$SIGMA_FLOOR_SYMMETRIC" == "1" ]]; then SIGMA_SYM_FLAG=--sigma-floor-symmetric; fi
SIGMA_REF_M=${SIGMA_REF_M:-250}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-0}
SAVE_EVERY=${SAVE_EVERY:-20}
NPROC=${NPROC:-1}
TARGETS_PER_STEP=${TARGETS_PER_STEP:-0}
CERT_DISTILL_COEF=${CERT_DISTILL_COEF:-0}
CERT_DISTILL_RANKS=${CERT_DISTILL_RANKS:-300}
CERT_DISTILL_SPACE=${CERT_DISTILL_SPACE:-log_prob}
FORECAST_LOSS_COEF=${FORECAST_LOSS_COEF:-1}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_TAG=$(echo "$MODEL" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
BANK_ROOT=${BANK_ROOT:-$TASK_BANK_ROOT/wildchat_ldnoobw_${MODEL_TAG}_T40_p1870}

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ -n "${MICRO_BATCH:-}" ]]; then
  echo "[mb] using override MICRO_BATCH=${MICRO_BATCH}"
elif [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=16
elif (( VRAM_MB < 30000 )); then MICRO_BATCH=16
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=48
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=96
else MICRO_BATCH=128; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"

TAG=lqdyn_${ARM}_seed${SEED}
OUTDIR="$RESULTS_ROOT/lquant_dynamics/runs/$TAG"
mkdir -p "$OUTDIR"
echo "[run] OUTDIR=$OUTDIR"
echo "[run] ARM=$ARM BOOT=$BOOT IOG=$IOG SCOPE=$IOG_SCOPE ANCHOR=$ANCHOR_MARGIN" \
     "WEIGHTING=$WEIGHTING FIT_FRACTION=$FIT_FRACTION TARGETS=$TRAIN_TARGETS/$HELDOUT_TARGETS" \
     "STEPS=$STEPS REFRESH=$UNION_REFRESH"

EXTRA_ARGS=()
if [[ -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt"
  EXTRA_ARGS+=(--resume-checkpoint "$OUTDIR/resume_latest.pt")
fi
EXTRA_ARGS+=(--gradient-checkpointing)
if [[ "$IOG" == "1" ]]; then
  EXTRA_ARGS+=(--improving-only-grad --iog-scope "$IOG_SCOPE")
fi
if [[ "${TARGET_FULL_POOLS:-0}" == "1" ]]; then EXTRA_ARGS+=(--target-holdout-full-pools); fi
if [[ "${HELDOUT_FROM_END:-0}" == "1" ]]; then EXTRA_ARGS+=(--heldout-from-end); fi
if [[ "${CERT_DISTILL_SHARED_PASS:-0}" == "1" ]]; then EXTRA_ARGS+=(--cert-distill-shared-pass); fi
if [[ -n "${LORA_TARGETS:-}" ]]; then EXTRA_ARGS+=(--lora-target-modules "$LORA_TARGETS"); fi

LAUNCH=("$PYTHON" -u)
if (( NPROC > 1 )); then
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
  LAUNCH=("$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC")
  echo "[ddp] torchrun --nproc_per_node=$NPROC"
fi

exec "${LAUNCH[@]}" -m forecast_experiment.run_train_lquant \
  --model "$MODEL" \
  --task-kind token_union \
  --prompt-bank-root "$BANK_ROOT" \
  --prompt-bank-seed 11 \
  --train-families wildchat_flagged_en \
  --test-families wildchat_flagged_en \
  --train-target-count "$TRAIN_TARGETS" \
  --heldout-target-count "$HELDOUT_TARGETS" \
  --targets-per-step "$TARGETS_PER_STEP" \
  --shared-family-test-fraction 0.5 \
  --split-mode disjoint \
  --fit-fraction "$FIT_FRACTION" \
  --score-max-len "$SCORE_MAX_LEN" \
  --score-token-budget "$SCORE_TOKEN_BUDGET" \
  --ols-top-k "$OLS_TOP_K" \
  --plotting-position weibull \
  --extrapolated-weighting "$WEIGHTING" \
  --extrapolated-loss-space log_prob \
  --loss-mode ols_primal_extrapolated \
  --grad-mode both \
  --steps "$STEPS" \
  --lr "$LR" \
  --optimizer "$OPTIMIZER" \
  --grad-clip 1.0 \
  --micro-batch-size "${MICRO_BATCH}" \
  --eval-cache-top-b 0 \
  --eval-cache-refresh-interval 0 \
  --bootstrap-permutations "$BOOT" \
  --union-cache-top-c "$UNION_CACHE_C" \
  --union-cache-refresh-interval "$UNION_REFRESH" \
  --anchor-gate-margin "$ANCHOR_MARGIN" \
  --anchor-penalty-coef "$ANCHOR_PENALTY" \
  --loss-scale-norm-m "$SCALE_NORM_M" \
  --gap-floor-m "$GAP_FLOOR_M" \
  --gap-floor-lambda "$GAP_FLOOR_LAMBDA" \
  --gap-floor-block "$GAP_FLOOR_BLOCK" \
  --gap-floor-slack "$GAP_FLOOR_SLACK" \
  $LEVEL_GATED_FLAG \
  --sigma-floor-lambda "$SIGMA_FLOOR_LAMBDA" \
  --sigma-floor-margin "$SIGMA_FLOOR_MARGIN" \
  $SIGMA_SYM_FLAG \
  --sigma-ref-m "$SIGMA_REF_M" \
  --cert-distill-coef "$CERT_DISTILL_COEF" \
  --cert-distill-ranks "$CERT_DISTILL_RANKS" \
  --cert-distill-space "$CERT_DISTILL_SPACE" \
  --forecast-loss-coef "$FORECAST_LOSS_COEF" \
  --iog-fit-mask "$IOG_FIT_MASK" \
  --dynamics-dump-dir "$OUTDIR/dynamics" \
  --dynamics-dump-every "$DUMP_EVERY" \
  --baseline-eval-cache-dir "$REPO_ROOT/artifacts/cache/baseline_eval" \
  --seed "${SEED}" \
  --kl-coef "$KL_COEF" \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets password \
  --kl-eval-datasets "$BANK_ROOT/bank_seed_11/kl_eval_nonflagged_en.jsonl" \
  --kl-tokenize-max-len 1024 \
  --kl-truncation-side left \
  --log-every 5 \
  --save-every "$SAVE_EVERY" \
  --adapter-snapshot-every "$SNAPSHOT_EVERY" \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --iog-worsening-coef "$IOG_WORSENING_COEF" \
  "${EXTRA_ARGS[@]}"
