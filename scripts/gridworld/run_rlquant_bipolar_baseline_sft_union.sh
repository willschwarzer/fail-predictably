#!/usr/bin/env bash
# RL baseline (SFT-on-union): direct REINFORCE on the union of meta-fit and
# meta-deploy tasks for the canonical bipolar bank. Pair with --meta-steps 0 so
# the run produces pretrained_results.json (with the FRLMB report on heldout)
# and stops -- no meta phase. The "pretrained" checkpoint here IS the SFT model.
#
# Hyperparams mirror run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_10seeds.sh
# (whose union cache acts only in the meta phase) except:
#   - PRETRAIN_TASK_SOURCE=meta_union (the actual swap)
#   - META_STEPS=0 (skip meta phase)
#   - PRETRAIN_STEPS bumped to 800 to roughly match total wall (canonical was 500+300)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT

export EXPERIMENT_TAG="rlquant_trap_report_bipolar_sev01_baseline_sft_union_10seeds_${TAG_DATE:-$(date +%Y%m%d)}"
# The RL layout bank (scripts/banks/launch_rl_bipolar_bank.sh) under the bank root.
export TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/rlquant_trap_open_room_layout_bank_bipolar_sev01_20260503"
export TASK_BANK_SEED=-1

export SEEDS="11 17 23 29 31 37 41 43 47 53"

export PRETRAIN_TASK_SOURCE=meta_union
export PRETRAIN_STEPS=800
export META_STEPS=0
export LR=1e-4
# Aux loss / IOG are meta-phase concerns; harmless to leave at canonical values
# since meta runs zero steps, but keep the config snapshot consistent.
export META_AUX_LOSS_WEIGHT=1.5
export META_MANIFESTS_PER_STEP=10
export META_DEPLOY_CACHE_REFRESH_INTERVAL=5
export IMPROVING_ONLY_GRAD=1
export GRAD_MODE=both

export THREADS_PER_TASK=16
export MEMORY=16G
export TIME_LIMIT=0-04:00:00
export ARRAY_CONCURRENCY=0

unset REUSE_PRETRAIN_ROOT INIT_PRETRAIN_ROOT INIT_CHECKPOINT RESUME_ROOT   # these runs pretrain their own policies (see the submitter)

cd "$REPO_ROOT"
exec bash "$REPO_ROOT/scripts/gridworld/submit_rlquant_trap_report_array.sh"
