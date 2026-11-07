#!/usr/bin/env bash
# The runs of the appendix "Forecastability training with sampled returns" that were
# launched by hand, as environment overrides of submit_rlquant_trap_report_array.sh.
# Pick one with the first argument:
#   exact600         exact-regret training (the recipe of
#                    run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_10seeds.sh)
#                    run fresh for 600 updates; one of the two exact-regret run sets
#                    extended to 600 updates in the companion experiment.
#   exact600ext      the other one: that launcher's 300-update fleet (TAG_DATE=20260505)
#                    continued to 600 updates from each seed's saved checkpoint.
#   n32_last_stable  the n = 32 seeds of run_rlquant_bipolar_reinforce_negret_m600_20260725.sh
#                    whose policy went non-finite late in training (17, 23 and 31 in the
#                    paper), evaluated at the last checkpoint each saved before that
#                    (update 550 for seeds 17 and 23, 500 for seed 31). The paper's n = 32
#                    result pools these with the other seven seeds of that fleet.
# Usage: bash scripts/gridworld/run_rlquant_bipolar_sampled_returns_extras.sh exact600
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/artifacts/results}"
cd "$REPO_ROOT"

common() {
  # The RL layout bank (scripts/banks/launch_rl_bipolar_bank.sh) under the bank root.
  export TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}/rlquant_trap_open_room_layout_bank_bipolar_sev01_20260503"
  export TASK_BANK_SEED=-1
  export SEEDS="11 17 23 29 31 37 41 43 47 53"
  export PRETRAIN_STEPS=500
  export META_STEPS=600
  export LR=1e-4
  export META_AUX_LOSS_WEIGHT=1.5
  export META_MANIFESTS_PER_STEP=10
  export META_DEPLOY_CACHE_REFRESH_INTERVAL=5
  export IMPROVING_ONLY_GRAD=1
  export IOG_SCOPE=both
  export BOOTSTRAP_PERMUTATIONS=1
  export GRAD_MODE=both
  export THREADS_PER_TASK=16
  export MEMORY=16G
  export ARRAY_CONCURRENCY=0
  unset REUSE_PRETRAIN_ROOT INIT_PRETRAIN_ROOT INIT_CHECKPOINT RESUME_ROOT   # these runs pretrain or resume their own policies
}

exact_recipe() {
  export UNION_CACHE_TOP_C=296
  export UNION_CACHE_REFRESH_INTERVAL=5
}

negret_n32_recipe() {
  export UNION_CACHE_TOP_C=0
  export GRAD_ESTIMATOR=reinforce
  export PRETRAIN_GRAD_ESTIMATOR=exact
  export REINFORCE_META_ESTIMATOR=plugin
  export REINFORCE_ROLLOUTS_PER_TASK=32
  export REINFORCE_BASELINE=ema
  export REINFORCE_SCORE=neg_return
}

SUBMIT="$REPO_ROOT/scripts/gridworld/submit_rlquant_trap_report_array.sh"

case "${1:?usage: $0 exact600|exact600ext|n32_last_stable}" in
  exact600)
    (
      common; exact_recipe
      export TIME_LIMIT=0-06:00:00
      export EXPERIMENT_TAG="rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_m600_10seeds_20260725"
      bash "$SUBMIT"
    )
    ;;
  exact600ext)
    (
      common; exact_recipe
      export TIME_LIMIT=0-02:00:00
      export RESUME_ROOT="${RESULTS_ROOT}/rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_10seeds_20260505"
      export EXPERIMENT_TAG="rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_m600ext_10seeds_20260725"
      bash "$SUBMIT"
    )
    ;;
  n32_last_stable)
    # Each seed resumes from the resume_latest.pt its 600-update run left behind and
    # stops at that checkpoint's update count, so the final evaluation reads it.
    for spec in "17 23|550|2seeds" "31|500|1seed"; do
      IFS='|' read -r seeds steps count <<< "$spec"
      (
        common; negret_n32_recipe
        export SEEDS="$seeds"
        export META_STEPS="$steps"
        export TIME_LIMIT=0-02:00:00
        export RESUME_ROOT="${RESULTS_ROOT}/rlquant_trap_report_bipolar_sev01_reinforce_plugin_negret_m600_n32_lr1e4_prex_10seeds_20260725"
        export EXPERIMENT_TAG="rlquant_trap_report_bipolar_sev01_reinforce_plugin_negret_prenan_m${steps}_n32_lr1e4_prex_${count}_20260726"
        bash "$SUBMIT"
      )
    done
    ;;
  *)
    echo "unknown run set: $1 (expected exact600, exact600ext or n32_last_stable)" >&2
    exit 1
    ;;
esac
