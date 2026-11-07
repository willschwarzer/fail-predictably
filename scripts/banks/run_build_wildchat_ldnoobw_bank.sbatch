#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --job-name=wc_ldnoobw_bank
# Hardware: CPU only (4 cores, 48 GB).
set -euo pipefail

# Build a WildChat LDNOOBW token-union bank (CPU; streams ~2.5M WildChat rows).
# The defaults (40 targets x 1870 prompts) give the bank of the appendix
# "Forecastability on natural user prompts". The 2,300-prompt task sets of the
# appendix "Forecastability training with minimal change to the model" take
# three steps:
#   1. TARGETS=16 PER_TARGET=4600 sbatch scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch
#   2. python scripts/banks/rechunk_token_union_bank.py \
#        "$TASK_BANK_ROOT"/wildchat_ldnoobw_qwen_qwen3-0_6b_T16_p4600/bank_seed_11 \
#        "$TASK_BANK_ROOT"/wildchat_ldnoobw_qwen_qwen3-0_6b_T32_p2300/bank_seed_11 --parts 2
#   3. python scripts/banks/derive_wildchat_v2_bank.py \
#        "$TASK_BANK_ROOT"/wildchat_ldnoobw_qwen_qwen3-0_6b_T32_p2300/bank_seed_11 \
#        "$TASK_BANK_ROOT"/wildchat_ldnoobw_v2e2_qwen_qwen3-0_6b_T32_p2300/bank_seed_11 \
#        --rule none --pool-size 2300 \
#        --drop ASS,Anal,Ass,Cum,Dick,Escort,SEX,Sex,Tit,XX,XXX,anal,ass,butt,cock,coon,cum,escort,mong,poon,sex,sexo,sexual,tit,xx,xxx
# Inputs: a local copy of WildChat-1M (WILDCHAT_DATASET below) and the LDNOOBW
# English word list (WORD_LIST, shipped in data/wordlists/).
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
TARGETS=${TARGETS:-40}
PER_TARGET=${PER_TARGET:-1870}
SUBSETS_JSON=${SUBSETS_JSON:-}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
# Local directory of WildChat arrow shards (searched recursively), read by the bank builder.
export WILDCHAT_DATASET="${WILDCHAT_DATASET:-$REPO_ROOT/data/wildchat}"
WORD_LIST=${WORD_LIST:-$REPO_ROOT/data/wordlists/ldnoobw_en.txt}
export TOKENIZERS_PARALLELISM=false

MODEL_TAG=$(echo "$MODEL" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
OUT=$TASK_BANK_ROOT/wildchat_ldnoobw_${MODEL_TAG}_T${TARGETS}_p${PER_TARGET}

ARGS=(--out "$OUT" --model "$MODEL" --bank-seed 11 --targets "$TARGETS" --prompts-per-target "$PER_TARGET" --word-list "$WORD_LIST")
if [[ -n "$SUBSETS_JSON" ]]; then ARGS+=(--union-subsets-json "$SUBSETS_JSON"); fi

exec "$PYTHON" -u -m forecast_experiment.generate_wildchat_token_union_bank "${ARGS[@]}"
