#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_wc_ldnoobw
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# WildChat LDNOOBW token-union forecastability meta-training (paper appendix
# "Forecastability on natural user prompts"). The defaults (Weibull, k=10,
# deploy-log-uniform log-prob loss, grad both, IOG both, bootstrap N=1 + union
# top-C cache 296/5, every training task in each update, lr 1e-5, KL 0.05 to
# frozen base on task prompts) are those of
# scripts/password_game/run_lquant_modbulkgcg_iog_full_bootstrap_cached_multiseed.sbatch
# except: token_union task kind, WildChat bank, 2048-token left-truncated scoring
# (all leaky harmful prompts are <1508 tokens; harmful log-prob anti-correlates
# with length, so 2048 loses no signal at ~4x lower memory/time), and the
# non-flagged-English KL eval sidecar.
# The appendix's runs override five of them: LR=3e-5, TARGETS_PER_STEP=8,
# UNION_REFRESH=20, KL_COEF=0.5 (2.0 and 5.0 in its KL-sensitivity runs) and
# KL_TRAIN=nonflagged. Its blocklist-prompt control kept LR=1e-5 and set
# BLOCKLIST=1. scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh
# submits every appendix run, with TAG_DATE set to the run-name suffix
# (mb8klnf05 for the main runs).
#
# Env knobs:
#   SEED           (required) training seed
#   MODEL          default Qwen/Qwen3-0.6B
#   BANK_ROOT      default $TASK_BANK_ROOT/wildchat_ldnoobw_<model tag>_T40_p1870,
#                  built by scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch
#   TRAIN_TARGETS / HELDOUT_TARGETS  default 33 / 7 (paper cell on a 40-task bank)
#   SCORE_MAX_LEN  default 2048 ; SCORE_TOKEN_BUDGET default 8192
#   STEPS          default 200
#   RESULTS_ROOT   default $REPO_ROOT/artifacts/results ; TASK_BANK_ROOT default
#                  $REPO_ROOT/artifacts/task_banks ; PYTHON default python
# GPU: needs a full 80 GB card (see Hardware above) because 8k-context grad passes
# need a real card even with gradient checkpointing; an sm_80 requirement alone
# would also match 22GB MIG slices.
#
# Inputs: KL_TRAIN=nonflagged reads KL_TRAIN_JSONL (default
# $BANK_ROOT/bank_seed_11/kl_train_nonflagged_en_disjoint.jsonl): the 1792 prompts
# of the bank's kl_eval_nonflagged_en.jsonl (2048 non-flagged English prompts)
# that the trainer's seeded 256-prompt KL eval does not draw. Write it once per
# bank with scripts/wildchat/natural_prompts/make_kl_train_split.py <bank dir>;
# the default KL_TRAIN=password does not need it.

SEED=${SEED:?SEED env var required}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
TRAIN_TARGETS=${TRAIN_TARGETS:-33}
HELDOUT_TARGETS=${HELDOUT_TARGETS:-7}
SCORE_MAX_LEN=${SCORE_MAX_LEN:-2048}
SCORE_TOKEN_BUDGET=${SCORE_TOKEN_BUDGET:-8192}
STEPS=${STEPS:-200}
# Learning rate. The default 1e-5 is the password-game value; the appendix's
# forecastability runs used 3e-5 (its blocklist-prompt control used 1e-5).
LR=${LR:-1e-5}
# Partial IOG: scale worsening gradients by this instead of zeroing them.
# 0 = canonical hard mask (bit-identical to runs from before this knob existed).
IOG_WORSENING_COEF=${IOG_WORSENING_COEF:-0}
# IOG_SCOPE=deploy_only leaves fit-side log-probs free to move, which is the
# targeted intervention for the 4B/8B failure: the model suppresses the tail at a
# size-independent rate but the OLS line does not follow it down. LORA_R raises adapter capacity,
# which is fixed at r8 while the model grows 13x. Both are ablations off the
# canonical recipe (both / r8) and must be tagged as such.
IOG_SCOPE=${IOG_SCOPE:-both}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-$((LORA_R * 2))}
# KL coefficient to the frozen base, on the prompts KL_TRAIN names. The default
# 0.05 (on task prompts) is the password-game value; the appendix used 0.5 with
# KL_TRAIN=nonflagged, and 2.0 and 5.0 in its KL-sensitivity runs. The union
# tokens carry ~1e-9 mass, so the suppression itself is KL-cheap; what a big
# coefficient penalizes is off-target movement of the distribution.
KL_COEF=${KL_COEF:-0.05}
# KL_TRAIN=nonflagged trains the KL penalty on the non-flagged-EN sidecar
# instead of the task prompts. On task (flagged) prompts the kl1 floor
# (-log(1-q_u), ~5-14 nats under successful suppression) dominates the
# penalty, so the coefficient mostly taxes an unavoidable term; on
# non-flagged prompts the union mass is negligible, the KL is floorless, and
# it is exactly the collateral the reporting guard measures. The appendix used
# nonflagged.
KL_TRAIN=${KL_TRAIN:-password}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_TAG=$(echo "$MODEL" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
BANK_ROOT=${BANK_ROOT:-$TASK_BANK_ROOT/wildchat_ldnoobw_${MODEL_TAG}_T40_p1870}
KL_TRAIN_JSONL=${KL_TRAIN_JSONL:-$BANK_ROOT/bank_seed_11/kl_train_nonflagged_en_disjoint.jsonl}

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')
# MICRO_BATCH auto-detect is VRAM-based, but token_union memory is dominated by
# forward activations that scale with MODEL hidden size, so bigger models need a
# smaller count cap on a fixed card. Allow an explicit override (set per size at
# submit time); otherwise fall back to the VRAM-based default.
if [[ -n "${MICRO_BATCH:-}" ]]; then
  echo "[mb] using override MICRO_BATCH=${MICRO_BATCH}"
elif [[ -z "${VRAM_MB}" ]]; then MICRO_BATCH=16
elif (( VRAM_MB < 30000 )); then MICRO_BATCH=16
elif (( VRAM_MB < 50000 )); then MICRO_BATCH=48
elif (( VRAM_MB < 90000 )); then MICRO_BATCH=96
else MICRO_BATCH=128; fi
echo "[vram] detected ${VRAM_MB} MB, micro-batch ${MICRO_BATCH}"

TAG=lquant_wildchat_ldnoobw_${MODEL_TAG}_iog_full_boot1_cache296_seed${SEED}_${TAG_DATE:-$(date +%Y%m%d)}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"
# Announce the outdir so the slurm log is self-identifying: in-flight analysis
# needs to map a log back to its run, and nothing else in the log carries it.
echo "[run] OUTDIR=$OUTDIR"
echo "[run] LR=$LR STEPS=$STEPS IOG_SCOPE=$IOG_SCOPE IOG_WORSENING_COEF=$IOG_WORSENING_COEF LORA_R=$LORA_R KL_COEF=$KL_COEF"
cd "$REPO_ROOT"

EXTRA_ARGS=()
if [[ -f "$OUTDIR/resume_latest.pt" ]]; then
  echo "[auto-resume] picking up from $OUTDIR/resume_latest.pt"
  EXTRA_ARGS+=(--resume-checkpoint "$OUTDIR/resume_latest.pt")
fi
if [[ "${GRAD_CKPT:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
  echo "[gc] gradient checkpointing ON"
else
  echo "[gc] gradient checkpointing OFF"
fi
# BLOCKLIST=1: put the union word list in a system prompt (context-computable
# rule canary; distinct baseline-cache key, tagged run only).
if [[ "${BLOCKLIST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--token-union-blocklist-prompt)
  echo "[blocklist] union words in system prompt"
fi
# TARGETS_PER_STEP>0: meta-batch sampling (b tasks per optimizer step). Pair
# with UNION_REFRESH scaled by ~33/b so cache-refresh cadence per VISIT matches
# the default (the interval counts wall steps, not visits). The appendix used
# TARGETS_PER_STEP=8 with UNION_REFRESH=20.
TARGETS_PER_STEP=${TARGETS_PER_STEP:-0}
UNION_REFRESH=${UNION_REFRESH:-5}
if [[ "$TARGETS_PER_STEP" != "0" ]]; then
  EXTRA_ARGS+=(--targets-per-step "$TARGETS_PER_STEP")
  echo "[meta-batch] $TARGETS_PER_STEP targets/step, union refresh $UNION_REFRESH"
fi

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant \
  --model "$MODEL" \
  --task-kind token_union \
  --prompt-bank-root "$BANK_ROOT" \
  --prompt-bank-seed 11 \
  --train-families wildchat_flagged_en \
  --test-families wildchat_flagged_en \
  --train-target-count "$TRAIN_TARGETS" \
  --heldout-target-count "$HELDOUT_TARGETS" \
  --shared-family-test-fraction 0.5 \
  --split-mode disjoint \
  --fit-fraction 0.0476 \
  --score-max-len "$SCORE_MAX_LEN" \
  --score-token-budget "$SCORE_TOKEN_BUDGET" \
  --ols-top-k 10 \
  --plotting-position weibull \
  --extrapolated-weighting deploy_log_uniform \
  --extrapolated-loss-space log_prob \
  --loss-mode ols_primal_extrapolated \
  --grad-mode both \
  --steps "$STEPS" \
  --lr "$LR" \
  --grad-clip 1.0 \
  --micro-batch-size "${MICRO_BATCH}" \
  --eval-cache-top-b 0 \
  --eval-cache-refresh-interval 0 \
  --bootstrap-permutations 1 \
  --union-cache-top-c 296 \
  --union-cache-refresh-interval "$UNION_REFRESH" \
  --baseline-eval-cache-dir "$REPO_ROOT/artifacts/cache/baseline_eval" \
  --seed "${SEED}" \
  --kl-coef "$KL_COEF" \
  --kl-batch-size 8 \
  --kl-trunc-len 0 \
  --kl-trunc-every 5 \
  --kl-train-datasets $([ "$KL_TRAIN" == "nonflagged" ] && echo "$KL_TRAIN_JSONL" || echo "password") \
  --kl-eval-datasets "$BANK_ROOT/bank_seed_11/kl_eval_nonflagged_en.jsonl" \
  --kl-tokenize-max-len 1024 \
  --kl-truncation-side left \
  --log-every 10 \
  --save-every 20 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --improving-only-grad \
  --iog-scope "$IOG_SCOPE" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --iog-worsening-coef "$IOG_WORSENING_COEF" \
  "${EXTRA_ARGS[@]}"
