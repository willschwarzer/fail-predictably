#!/usr/bin/env bash
# Sampled-regret fleets and the RL baseline of the appendix "Forecastability training
# with sampled returns", 10 seeds each, fresh exact pretraining, ema baseline, lr 1e-4.
# The paper reads the 600-update fleets: the weighted (plugin) estimator at n = 8 and 16
# (and its mean returns at every n), the scalar-loss (batch) estimator at every n, and the
# RL baseline at n = 1, 8 and 16. scripts/gridworld/collect_sampled_returns.py reads them.
#
#   A/B x n{1,2,4,8,16,32} @ META_STEPS=600  (tags put m600 BEFORE the n segment so
#     the ..._n{1,8}_lr*_10seeds_* globs of plot_rlquant_three_metric_bars.py
#     cannot match them)
#   A/B x n{2,4,16,32}     @ META_STEPS=300  (no paper number reads these)
#   SFT x n{1,2,4,8,16,32} @ 600 union steps (the RL baseline)
#   SFT x n{2,4,16,32}     @ 300 union steps (no paper number reads these)
#
# Inputs: the SFT (RL baseline) fleets start from the per-seed pretrained_checkpoint.pt
# of run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_10seeds.sh launched with
# TAG_DATE=20260505; set INIT_PRETRAIN_ROOT for any other date.
#
# 30 fleets / 300 runs. Rerun with AUTO_RESUME=1 to continue any unfinished seeds in
# place (completed seeds no-op); add EXCLUSIVE=1 for seeds stuck on oversubscribed nodes.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
STAMP=20260725
cd "$REPO_ROOT"

submit_method() { # est n meta_steps tag
  local est=$1 n=$2 meta_steps=$3 tag=$4
  (
    # The RL layout bank (scripts/banks/launch_rl_bipolar_bank.sh) under the bank root.
    export TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/rlquant_trap_open_room_layout_bank_bipolar_sev01_20260503"
    export TASK_BANK_SEED=-1
    export SEEDS="11 17 23 29 31 37 41 43 47 53"
    export PRETRAIN_STEPS=500
    export META_STEPS="${meta_steps}"
    export LR=1e-4
    export META_AUX_LOSS_WEIGHT=1.5
    export META_MANIFESTS_PER_STEP=10
    export META_DEPLOY_CACHE_REFRESH_INTERVAL=5
    export IMPROVING_ONLY_GRAD=1
    export IOG_SCOPE=both
    export BOOTSTRAP_PERMUTATIONS=1
    export GRAD_MODE=both
    export UNION_CACHE_TOP_C=0
    export GRAD_ESTIMATOR=reinforce
    export PRETRAIN_GRAD_ESTIMATOR=exact
    export REINFORCE_META_ESTIMATOR="${est}"
    export REINFORCE_ROLLOUTS_PER_TASK="${n}"
    export REINFORCE_BASELINE=ema
    export THREADS_PER_TASK=16
    export MEMORY=16G
    export TIME_LIMIT=0-08:00:00
    export ARRAY_CONCURRENCY=0
    unset REUSE_PRETRAIN_ROOT INIT_PRETRAIN_ROOT INIT_CHECKPOINT RESUME_ROOT   # these runs pretrain their own policies (see the submitter)
    export EXPERIMENT_TAG="${tag}"
    bash "$REPO_ROOT/scripts/gridworld/submit_rlquant_trap_report_array.sh"
  )
}

submit_sft() { # n steps tag
  local n=$1 steps=$2 tag=$3
  (
    # The RL layout bank (scripts/banks/launch_rl_bipolar_bank.sh) under the bank root.
    export TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/rlquant_trap_open_room_layout_bank_bipolar_sev01_20260503"
    export TASK_BANK_SEED=-1
    export SEEDS="11 17 23 29 31 37 41 43 47 53"
    export INIT_PRETRAIN_ROOT="${INIT_PRETRAIN_ROOT:-${RESULTS_ROOT:-${REPO_ROOT}/artifacts/results}/rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_10seeds_20260505}"
    export PRETRAIN_STEPS="${steps}"
    export META_STEPS=0
    export LR=1e-4
    export PRETRAIN_TASK_SOURCE=meta_union
    export GRAD_ESTIMATOR=reinforce
    export REINFORCE_ROLLOUTS_PER_TASK="${n}"
    export REINFORCE_BASELINE=ema
    export THREADS_PER_TASK=16
    export MEMORY=16G
    export TIME_LIMIT=0-04:00:00
    export ARRAY_CONCURRENCY=0
    unset REUSE_PRETRAIN_ROOT RESUME_ROOT   # init only (INIT_PRETRAIN_ROOT): reuse would skip the SFT phase
    export EXPERIMENT_TAG="${tag}"
    bash "$REPO_ROOT/scripts/gridworld/submit_rlquant_trap_report_array.sh"
  )
}

for est in plugin batch; do
  for n in 1 2 4 8 16 32; do
    submit_method "$est" "$n" 600 \
      "rlquant_trap_report_bipolar_sev01_reinforce_${est}_m600_n${n}_lr1e4_prex_10seeds_${STAMP}"
  done
  for n in 2 4 16 32; do
    submit_method "$est" "$n" 300 \
      "rlquant_trap_report_bipolar_sev01_reinforce_${est}_n${n}_lr1e4_prex_10seeds_${STAMP}"
  done
done

for n in 1 2 4 8 16 32; do
  submit_sft "$n" 600 \
    "rlquant_trap_report_bipolar_sev01_baseline_sft_union_reinforce_m600_n${n}_10seeds_${STAMP}"
done
for n in 2 4 16 32; do
  submit_sft "$n" 300 \
    "rlquant_trap_report_bipolar_sev01_baseline_sft_union_reinforce_n${n}_10seeds_${STAMP}"
done

echo "All overnight fleets submitted."
