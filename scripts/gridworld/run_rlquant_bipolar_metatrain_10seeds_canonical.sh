#!/usr/bin/env bash
# Cal. in the figure of "Multi-task RL: gridworld navigation with traps": the
# after-the-fact calibration reads this fleet's pretrained policies (the paper's tag
# ends _20260503, so set TAG_DATE=20260503 to match). Recipe: 10 seeds, 500
# pretraining updates per seed on the bipolar bank, then 300 updates of
# forecastability training with the improving-only mask.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

export EXPERIMENT_TAG="rlquant_trap_report_bipolar_sev01_metatrain_10seeds_canonicalhparams_${TAG_DATE:-$(date +%Y%m%d)}"
# The RL layout bank (scripts/banks/launch_rl_bipolar_bank.sh) under the bank root.
export TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/rlquant_trap_open_room_layout_bank_bipolar_sev01_20260503"
export TASK_BANK_SEED=-1   # auto-match seed (so seed=N uses bank_seed_N)

export SEEDS="11 17 23 29 31 37 41 43 47 53"

export PRETRAIN_STEPS=500
export META_STEPS=300
export LR=1e-4
export META_AUX_LOSS_WEIGHT=1.5
export META_MANIFESTS_PER_STEP=10
export META_DEPLOY_CACHE_REFRESH_INTERVAL=5
export IMPROVING_ONLY_GRAD=1
export GRAD_MODE=both
# Our runs recorded --bootstrap-permutations 1, but the trainer of May 3 drew new
# fit/deployment splits only for values above 1, so each manifest kept its split.
# In this version 1 draws a new split at every step. With 0 the submitter leaves the
# flag at the trainer's default, which keeps the split fixed as in our runs.
export BOOTSTRAP_PERMUTATIONS=0

export THREADS_PER_TASK=16
export MEMORY=16G
export TIME_LIMIT=0-09:00:00
export ARRAY_CONCURRENCY=0   # uncapped

unset REUSE_PRETRAIN_ROOT INIT_PRETRAIN_ROOT INIT_CHECKPOINT RESUME_ROOT   # these runs pretrain their own policies (see the submitter)

cd "$REPO_ROOT"
exec bash "$REPO_ROOT/scripts/gridworld/submit_rlquant_trap_report_array.sh"
