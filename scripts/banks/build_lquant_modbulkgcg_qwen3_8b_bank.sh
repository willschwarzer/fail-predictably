#!/usr/bin/env bash
# Rebuild the modbulk + GCG6 LM prompt bank with Qwen3-8B as the GCG target,
# preserving every other setting from the original 0.6B-targeted bank
# (lquant_prompt_bank_40slots_modbulk_gcg6_20260502, bank_seed=11). The
# resulting bank is needed for the Qwen3-8B PoC scaling runs since the
# train_lquant validator hard-rejects loading a bank whose generation
# model_id != the run's model_id.
#
# Cheap-family prompts (compositional / mutate / indirect_extraction /
# password_seeded) are model-independent in content; only GCG suffixes
# require the new target model. We rebuild everything via the canonical
# parallel builder for simplicity (cheap-family regen is trivial).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

# The 8B password-game launchers read this bank as
# lquant_prompt_bank_40slots_modbulk_gcg6_qwen3_8b_20260506; build it with
# TAG_DATE=20260506.
export EXPERIMENT_TAG="lquant_prompt_bank_40slots_modbulk_gcg6_qwen3_8b_${TAG_DATE:-$(date +%Y%m%d)}"
export OUTROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/${EXPERIMENT_TAG}"

export MODEL="Qwen/Qwen3-8B"
export BANK_SEED=11
export PASSWORD_SLOTS=40
export PASSWORD_POLICY=non_disclosure
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

# 8B model needs more VRAM than the 0.6B build; restrict to 48 GB+ GPUs.
# Per-target wall is dominated by GCG (~6 prompts x 80 steps x ~candidates
# of forward passes through 8B). On A100/L40s estimate ~1-2h per target;
# allow 4h for headroom and preempted retries.
export ARRAY_CONCURRENCY=20
export CPUS_PER_TASK=4
export MEMORY=64G
export TIME_LIMIT="0-04:00:00"
# Hardware: one GPU with at least 48 GB per shard (written for A100 and L40S).

export MERGE_CPUS=2
export MERGE_MEMORY=8G
export MERGE_TIME_LIMIT="0-00:30:00"

cd "${REPO_ROOT}"
exec bash "${REPO_ROOT}/scripts/banks/submit_lquant_prompt_bank_target_parallel.sh"
