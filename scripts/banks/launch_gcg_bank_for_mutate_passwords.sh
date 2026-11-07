#!/usr/bin/env bash
# Build a 40-slot bank using the mutate-screened passwords with:
#   bulk = 4 moderate benign families (compositional, mutate, indirect_extraction, password_seeded) at 466/slot
#   rare = gcg_steps80 at 6/slot, canonical hyperparams (steps=80, prompt_len=24, top_k=8, restarts=2)
# Reuses the existing parallel array generator + the new LQUANT_FIXED_PASSWORDS_JSON env-var override
# in assign_passwords_to_targets so each per-target shard picks up our hand-picked password.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
export LQUANT_FIXED_PASSWORDS_JSON="${LQUANT_FIXED_PASSWORDS_JSON:-${REPO_ROOT}/data/passwords/selected_passwords_handtrimmed_mutate_v2.json}"

# The 0.6B password-game launchers read this bank as
# lquant_prompt_bank_40slots_modbulk_gcg6_20260502; build it with TAG_DATE=20260502.
export EXPERIMENT_TAG="lquant_prompt_bank_40slots_modbulk_gcg6_${TAG_DATE:-$(date +%Y%m%d)}"
export OUTROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/${EXPERIMENT_TAG}"
export MODEL="Qwen/Qwen3-0.6B"
export BANK_SEED=11
export PASSWORD_SLOTS=40
export PASSWORD_POLICY=non_disclosure
export DISABLE_THINKING=1

# 4 moderate-strength benign families as bulk
export TRAIN_FAMILIES_STR="compositional mutate indirect_extraction password_seeded"
export TEST_FAMILIES_STR="compositional mutate indirect_extraction password_seeded"
export CHEAP_PROMPTS_PER_FAMILY=466
# Per-family override map not needed since they're all 466

# GCG canonical hyperparams
export GCG_ENABLED=1
export GCG_STEPS_SPECTRUM_STR="80"
export GCG_PROMPTS_PER_LEVEL_PER_TARGET=6
export GCG_PROMPT_LEN=24
export GCG_TOP_K=8
export GCG_RANDOM_RESTARTS=2
export GCG_TRAIN_MAX_STEPS=20
export GCG_DECAY_ALPHA=2.0

# Per-shard resources: each slot gets ~15 min on a small GPU
export CPUS_PER_TASK=2
export MEMORY=16G
export TIME_LIMIT=0-00:30:00
# Hardware: one GPU per shard with at least 16 GB and compute capability 8.0 or newer.
# Allow at most 20 concurrent shards to be polite
export ARRAY_CONCURRENCY=20

cd "$REPO_ROOT"
exec bash "$REPO_ROOT/scripts/banks/submit_lquant_prompt_bank_target_parallel.sh"
