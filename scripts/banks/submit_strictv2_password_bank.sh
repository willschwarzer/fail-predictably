#!/usr/bin/env bash
# Build one per-size password bank with GCG re-optimized against the
# non_disclosure_strict_v2 system prompt. Everything except password_policy,
# model, output tag, and per-size runtime matches the modbulk_gcg6 bank
# (40 slots, 466 cheap prompts/family, gcg_steps [80], 6 GCG/slot, prompt_len
# 24, top_k 8, restarts 2, disable_thinking). Non-GCG families are policy- and
# model-independent (deterministic string templates seeded by bank_seed), so
# only the GCG suffixes and the metadata (model_id, password_policy) change
# relative to the modbulk bank.
#
# These from-scratch banks (DATE 20260723 in the paper) are the donor-pool source
# for submit_strictv2_warmstart_bank.sh and the source of the 1.7B nopwseed bank;
# see scripts/banks/build_nopwseed_strictv2_banks.sh for the whole chain.
#
# Usage:  SIZE=0_6b MODEL=Qwen/Qwen3-0.6B  bash scripts/banks/submit_strictv2_password_bank.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT

SIZE=${SIZE:?SIZE env var required (e.g. 0_6b, 1_7b, 4b, 8b)}
MODEL=${MODEL:?MODEL env var required (e.g. Qwen/Qwen3-0.6B)}
DATE=${DATE:-20260723}

# Per-size GPU runtime: single-forward GCG so VRAM is model-weight bound.
# Hardware: one GPU per shard; the paper used compute capability 8.0+ with at
# least 16 GB (0_6b, 1_7b) or 32 GB (4b), and any GPU with at least 48 GB (8b).
case "${SIZE}" in
  0_6b) MEMORY=${MEMORY:-16G}; TIME_LIMIT=${TIME_LIMIT:-0-00:40:00} ;;
  1_7b) MEMORY=${MEMORY:-24G}; TIME_LIMIT=${TIME_LIMIT:-0-01:30:00} ;;
  4b)   MEMORY=${MEMORY:-40G}; TIME_LIMIT=${TIME_LIMIT:-0-02:30:00} ;;
  8b)   MEMORY=${MEMORY:-64G}; TIME_LIMIT=${TIME_LIMIT:-0-04:00:00} ;;
  *)    echo "Unknown SIZE ${SIZE}; set MEMORY/TIME_LIMIT explicitly." >&2; exit 1 ;;
esac

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-lquant_prompt_bank_40slots_modbulk_gcg6_strictv2_qwen3_${SIZE}_${DATE}}"
export MODEL
export BANK_SEED=11
export PASSWORD_SLOTS=40
export PASSWORD_POLICY=non_disclosure_strict_v2
export CHEAP_PROMPTS_PER_FAMILY=466
export TRAIN_FAMILIES_STR="compositional mutate indirect_extraction password_seeded"
export TEST_FAMILIES_STR="compositional mutate indirect_extraction password_seeded"
export DISABLE_THINKING=1
export GCG_ENABLED=1
export GCG_STEPS_SPECTRUM_STR="80"
export GCG_PROMPTS_PER_LEVEL_PER_TARGET=6
export GCG_PROMPT_LEN=24
export GCG_TOP_K=8
export GCG_RANDOM_RESTARTS=2
export GCG_TRAIN_MAX_STEPS=20
export GCG_DECAY_ALPHA=2.0
export ARRAY_CONCURRENCY=20
export CPUS_PER_TASK=2
export MEMORY
export TIME_LIMIT

echo "[strictv2-bank] ${EXPERIMENT_TAG}  model=${MODEL}  time=${TIME_LIMIT}"
exec bash "${SCRIPT_DIR}/submit_lquant_prompt_bank_target_parallel.sh"
