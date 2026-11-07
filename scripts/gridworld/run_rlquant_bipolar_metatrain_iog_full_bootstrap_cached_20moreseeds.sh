#!/usr/bin/env bash
# The other 20 of the 30 "Ours" seeds in the figure of "Multi-task RL: gridworld
# navigation with traps" (the paper's tag ends _20260506, so set TAG_DATE=20260506 to
# match). Same recipe as run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_10seeds.sh.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

export EXPERIMENT_TAG="rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_20moreseeds_${TAG_DATE:-$(date +%Y%m%d)}"
# The RL layout bank (scripts/banks/launch_rl_bipolar_bank.sh) under the bank root.
export TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/rlquant_trap_open_room_layout_bank_bipolar_sev01_20260503"
export TASK_BANK_SEED=-1

export SEEDS="59 61 67 71 73 79 83 89 97 101 103 107 109 113 127 131 137 139 149 151"

export PRETRAIN_STEPS=500
export META_STEPS=300
export LR=1e-4
export META_AUX_LOSS_WEIGHT=1.5
export META_MANIFESTS_PER_STEP=10
export META_DEPLOY_CACHE_REFRESH_INTERVAL=5
export IMPROVING_ONLY_GRAD=1
export IOG_SCOPE=both
export BOOTSTRAP_PERMUTATIONS=1
export GRAD_MODE=both

export UNION_CACHE_TOP_C=296
export UNION_CACHE_REFRESH_INTERVAL=5

export THREADS_PER_TASK=16
export MEMORY=16G
export TIME_LIMIT=0-04:00:00
export ARRAY_CONCURRENCY=0

unset REUSE_PRETRAIN_ROOT INIT_PRETRAIN_ROOT INIT_CHECKPOINT RESUME_ROOT   # these runs pretrain their own policies (see the submitter)

cd "$REPO_ROOT"
exec bash "$REPO_ROOT/scripts/gridworld/submit_rlquant_trap_report_array.sh"
