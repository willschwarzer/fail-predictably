#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=lq_ldn_besteval
# Hardware: one 80 GB GPU (the paper used an A100-80GB).
set -euo pipefail

# Held-out eval of a saved best/ adapter from an LDNOOBW run (paper appendix
# "Forecastability on natural user prompts"). The appendix evaluated every
# selected checkpoint with this job, forecastability and SFT runs at all four
# sizes; scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh (stage
# eval) submits those evaluations with MODEL and SEED set, and
# select_natural_checkpoint.py applies the selection rule (KL at most 0.7).
#
# WHY THIS EXISTS. Every variant arm records its lowest train-side worst-rank
# forecast error to best/ (train targets, running batch mean). Whether that
# minimum means anything can only be settled on the HELD-OUT targets, which the
# checkpoint selection never saw. This job loads the frozen adapter snapshot and
# runs exactly the trainer's final eval suite, producing a test_results.json
# comparable 1:1 against the arm's baseline_test_results.json (6.01 at 4B,
# seed 17) and a kl_eval_results.json for the KL bound.
#
# The config below must match the ARM's run exactly where it affects the eval:
# model, bank, seed (fit/deploy split + target partition), score lengths, and
# LORA_R/LORA_ALPHA (the adapter must load into a same-shape LoRA; alpha sets
# inference-time scaling). IOG/lr knobs only shape gradients and are irrelevant
# in an eval-only run, so they are left at canonical values.
#
# Env knobs: ADAPTER (required, dir with adapter_model.safetensors)
#            ARM     (required, short tag for the outdir)
#            MODEL   default Qwen/Qwen3-4B ; SEED default 17 (set both to the run's)
#            LORA_R  default 8 ; LORA_ALPHA default 2*LORA_R
#            RESULTS_ROOT default $REPO_ROOT/artifacts/results ; TASK_BANK_ROOT
#            default $REPO_ROOT/artifacts/task_banks ; PYTHON default python

ADAPTER=${ADAPTER:?ADAPTER env var required}
ARM=${ARM:?ARM env var required}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-$((LORA_R * 2))}
# BLOCKLIST=1: eval with the union words in the system prompt -- REQUIRED when
# the adapter was trained under that prompt; scoring without it would evaluate
# the model out of its training distribution. Baseline cache key diverges
# automatically (conditional extras), so it hits the blocklist baselines.
BLOCKLIST=${BLOCKLIST:-0}
EXTRA_EVAL_ARGS=()
if [[ "$BLOCKLIST" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--token-union-blocklist-prompt)
  echo "[blocklist] eval with union words in system prompt"
fi
SEED=${SEED:-17}
MODEL=${MODEL:-Qwen/Qwen3-4B}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_TAG=$(echo "$MODEL" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
BANK_ROOT=$TASK_BANK_ROOT/wildchat_ldnoobw_${MODEL_TAG}_T40_p1870

TAG=lquant_wildchat_ldnoobw_${MODEL_TAG}_iog_full_boot1_cache296_seed${SEED}_besteval_${ARM}
OUTDIR="$RESULTS_ROOT/$TAG/seed_${SEED}"
mkdir -p "$OUTDIR"
echo "[run] OUTDIR=$OUTDIR"
echo "[run] ADAPTER=$ADAPTER LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA"
cd "$REPO_ROOT"

exec "$PYTHON" -u -m forecast_experiment.run_train_lquant \
  --model "$MODEL" \
  --task-kind token_union \
  --prompt-bank-root "$BANK_ROOT" \
  --prompt-bank-seed 11 \
  --train-families wildchat_flagged_en \
  --test-families wildchat_flagged_en \
  --train-target-count 33 \
  --heldout-target-count 7 \
  --shared-family-test-fraction 0.5 \
  --split-mode disjoint \
  --fit-fraction 0.0476 \
  --score-max-len 2048 \
  --score-token-budget 8192 \
  --ols-top-k 10 \
  --plotting-position weibull \
  --extrapolated-weighting deploy_log_uniform \
  --extrapolated-loss-space log_prob \
  --loss-mode ols_primal_extrapolated \
  --grad-mode both \
  --steps 200 \
  --lr 1e-5 \
  --grad-clip 1.0 \
  --micro-batch-size 96 \
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
  --kl-eval-datasets "$BANK_ROOT/bank_seed_11/kl_eval_nonflagged_en.jsonl" \
  --kl-tokenize-max-len 1024 \
  --kl-truncation-side left \
  --log-every 10 \
  --save-every 20 \
  --outdir "$OUTDIR" \
  --disable-thinking \
  --improving-only-grad \
  --iog-scope both \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --gradient-checkpointing \
  --eval-adapter-from "$ADAPTER" \
  "${EXTRA_EVAL_ARGS[@]}"
