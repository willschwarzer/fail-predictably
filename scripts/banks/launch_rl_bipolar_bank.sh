#!/usr/bin/env bash
# RL "bipolar-severity" bank: same component counts as canonical but with
# severities collapsed to two delta functions at 0 and 1.0 (no continuum).
# Canonical had low component spread over severity [0, 0.5], which gave the
# benign-tail enough regret variance that the OLS line could fit aggressively.
# Bipolar removes that variance: low = uniformly easy, tail = uniformly hard.
#
# Counts kept at canonical: 51920 low, 80 tail. P(rare in deploy_size=1920) ~ 95%.
# P(rare in fit_size=96) ~ 14% -- accepting some rare-in-fit pollution in
# exchange for the cleaner severity geometry.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
# The gridworld launchers read this bank as ..._bipolar_sev01_20260503, so build
# it with TAG_DATE=20260503 to run them unchanged.
export EXPERIMENT_TAG="rlquant_trap_open_room_layout_bank_bipolar_sev01_${TAG_DATE:-$(date +%Y%m%d)}"
export OUTROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/${EXPERIMENT_TAG}"

# Match canonical seeds. The 10-seed gridworld launchers also need bank seeds 47
# and 53 (each run seed reads bank_seed_<seed>), and the 20 extra RL seeds of
# scripts/gridworld/run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_20moreseeds.sh
# need 59 61 67 71 73 79 83 89 97 101 103 107 109 113 127 131 137 139 149
# 151. A bank seed's layouts depend only on that seed, so one submission with
# all of them in BANK_SEEDS gives the same banks as separate submissions.
export BANK_SEEDS="${BANK_SEEDS:-11 17 23 29 31 37 41 43}"

# Match canonical counts
export BANK_TARGET_TOTAL=52000
export LOW_TARGET_TOTAL=51920
export TAIL_TARGET_TOTAL=80

# THE FIX: bipolar severities, no continuum
export LOW_SEVERITY_MIN=0.0
export LOW_SEVERITY_MAX=0.0   # was 0.5 in canonical
export TAIL_SEVERITY=1.0

# Resources match canonical
export LOW_SHARD_TARGET_COUNT=1000
export TIME_LIMIT_PER_SHARD=0-00:10:00
export MEMORY_PER_SHARD=1G

cd "$REPO_ROOT"
exec bash "$REPO_ROOT/scripts/banks/submit_rlquant_trap_layout_bank_generation.sh"
