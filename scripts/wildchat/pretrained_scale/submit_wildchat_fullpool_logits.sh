#!/usr/bin/env bash
# Submit full-pool WildChat harmful scoring + full-vocab logit dumps, one Slurm
# job per model size.
#
# Each job scores the ENTIRE deduped flagged-English WildChat pool (~121k
# prompts before dedup, ~85.6k after; max_len 8192, left truncation) with
# scripts/wildchat/decomposition/wildchat_score_harmful.py and,
# via --dump-full-logprobs-dir, also writes each prompt's full final-position
# log-prob vector (fp16). The dump is the reusable artifact: any next-token
# risk score over any token set becomes CPU postprocessing, and every
# (fit-size, deploy-size) forecast cell is a free subsample of the cached
# scores. Dumps are large (~37 GB per Qwen3 size, ~64 GB per Gemma-3 size), so
# point LOGIT_CACHE_ROOT (or DUMP_ROOT, for the dumps alone) at a large scratch
# disk. Per model, the small scores CSV and per-token npz land in
# $LOGIT_CACHE_ROOT/<model>/harmful/ and the dump in $DUMP_ROOT/<model>/.
#
# All jobs pass --resume, so a preempted / timed-out / resubmitted job
# continues from the rows already written instead of starting over.
#
# Usage:
#   bash scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh
# Smoke (0.6B only, 600 prompts):
#   MODELS_QWEN=0.6B MODELS_GEMMA= MAX_KEPT=600 TIME_OVERRIDE=0-00:40:00 \
#   LOGIT_CACHE_ROOT=.../wildchat_fullpool_logits_smoke \
#   bash scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh
#
# Env: RESULTS_ROOT (default $REPO_ROOT/artifacts/results); LOGIT_CACHE_ROOT
# (default $RESULTS_ROOT/wildchat_fullpool_logits); DUMP_ROOT (default
# $LOGIT_CACHE_ROOT); WILDCHAT_DATASET (directory holding the WildChat-1M-Full
# arrow shards, default $REPO_ROOT/data/wildchat); HF_HUB_DIR (a Hugging Face
# hub cache holding the model snapshots; a model not cached there is passed by
# its hub id); PYTHON (default python). Set SBATCH_ACCOUNT / SBATCH_PARTITION /
# SBATCH_QOS in the environment for your cluster; sbatch reads them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

HF_HUB_DIR=${HF_HUB_DIR:-${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}}
QWEN_HUB=${QWEN_HUB:-$HF_HUB_DIR}
GEMMA_HUB=${GEMMA_HUB:-$HF_HUB_DIR}
MODELS_QWEN=${MODELS_QWEN-"0.6B 1.7B 4B 8B 14B 32B"}
MODELS_GEMMA=${MODELS_GEMMA-"270m-it 1b-it 4b-it 12b-it 27b-it"}

RESULTS_ROOT=${RESULTS_ROOT:-"${REPO_ROOT}/artifacts/results"}
LOGIT_CACHE_ROOT=${LOGIT_CACHE_ROOT:-"${RESULTS_ROOT}/wildchat_fullpool_logits"}
DUMP_ROOT=${DUMP_ROOT:-"${LOGIT_CACHE_ROOT}"}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/logs"}
WORDLIST=${WORDLIST:-"${REPO_ROOT}/data/wordlists/ldnoobw_en.txt"}
WILDCHAT_DATASET=${WILDCHAT_DATASET:-"${REPO_ROOT}/data/wildchat"}

MAX_KEPT=${MAX_KEPT:-0}
MAX_LEN=${MAX_LEN:-8192}
MEM=${MEM:-64G}
CPUS=${CPUS:-8}
TIME_OVERRIDE=${TIME_OVERRIDE:-}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${LOGIT_CACHE_ROOT}" "${DUMP_ROOT}" "${LOG_DIR}"

# Per-model micro-batch / time limit. Times are ~2x the
# projected full-pool wall clock (25 min at 0.6B scaling roughly linearly in
# params, plus streaming/tokenization overhead); --resume makes a timeout
# recoverable anyway.
# Hardware: one GPU per job. Qwen3 0.6B-4B and Gemma-3 270m-4b fit a 16 GB card
# (the paper used Ampere, sm_80, cards); Qwen3 8B-32B and Gemma-3 12b/27b need
# one 80 GB GPU (the paper used an A100-80GB). To route sizes to different GPU
# types, submit them separately (MODELS_QWEN / MODELS_GEMMA) with
# SBATCH_PARTITION or SBATCH_CONSTRAINT set for your cluster.
model_spec() {
  case "$1" in
    Qwen3-0.6B)      echo "128 0-06:00:00" ;;
    Qwen3-1.7B)      echo "96 0-08:00:00" ;;
    Qwen3-4B)        echo "64 0-12:00:00" ;;
    Qwen3-8B)        echo "48 0-18:00:00" ;;
    Qwen3-14B)       echo "32 1-06:00:00" ;;
    Qwen3-32B)       echo "16 2-00:00:00" ;;
    gemma-3-270m-it) echo "128 0-04:00:00" ;;
    gemma-3-1b-it)   echo "128 0-06:00:00" ;;
    gemma-3-4b-it)   echo "64 0-14:00:00" ;;
    gemma-3-12b-it)  echo "32 1-06:00:00" ;;
    gemma-3-27b-it)  echo "16 2-00:00:00" ;;
    *) echo "ERROR: no spec for model $1" >&2; return 1 ;;
  esac
}

# A local snapshot (<hub>/models--<org>--<name>/snapshots/<rev>) if the hub
# cache has one, else the hub id <org>/<name> (downloaded on first use).
resolve_snapshot() {
  local hub="$1" org="$2" name="$3" d
  d=$(ls -d "${hub}/models--${org}--${name}/snapshots/"*/ 2>/dev/null | head -1 || true)
  if [ -z "${d}" ]; then
    echo "[hub] no snapshot for ${name} under ${hub}; using hub id ${org}/${name}" >&2
    echo "${org}/${name}"
    return
  fi
  echo "${d%/}"
}

submit_one() {
  local hub="$1" org="$2" model_name="$3"
  local spec mb time snap cmd
  spec=$(model_spec "${model_name}")
  read -r mb time <<< "${spec}"
  [ -n "${TIME_OVERRIDE}" ] && time="${TIME_OVERRIDE}"
  snap=$(resolve_snapshot "${hub}" "${org}" "${model_name}")

  cmd="${PYTHON} ${REPO_ROOT}/scripts/wildchat/decomposition/wildchat_score_harmful.py \
--base-model ${snap} \
--wildchat-dir ${WILDCHAT_DATASET} \
--out-dir ${LOGIT_CACHE_ROOT}/${model_name}/harmful \
--word-list ${WORDLIST} \
--require-flagged --dedup \
--max-kept ${MAX_KEPT} --max-len ${MAX_LEN} --truncation-side left \
--logits-from-forward --micro-batch ${mb} --device cuda:0 \
--dump-full-logprobs-dir ${DUMP_ROOT}/${model_name} --resume"

  local sbatch_args=(
    -J "fpl_${model_name}"
    -n 1 -c "${CPUS}" --mem="${MEM}"
    --gres=gpu:1
    --time="${time}" --requeue --open-mode=append
    --chdir="${REPO_ROOT}"
    --output="${LOG_DIR}/%x-%j.out" --error="${LOG_DIR}/%x-%j.err"
  )

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] sbatch ${sbatch_args[*]} --wrap '${cmd}'"
    return
  fi
  local out
  out=$(sbatch "${sbatch_args[@]}" --wrap "${cmd}")
  echo "${out}  (${model_name}: mb=${mb} time=${time})"
}

for size in ${MODELS_QWEN}; do
  submit_one "${QWEN_HUB}" "Qwen" "Qwen3-${size}"
done
for size in ${MODELS_GEMMA}; do
  submit_one "${GEMMA_HUB}" "google" "gemma-3-${size}"
done

echo "[logits]  ${LOGIT_CACHE_ROOT}"
echo "[dumps]   ${DUMP_ROOT}"
