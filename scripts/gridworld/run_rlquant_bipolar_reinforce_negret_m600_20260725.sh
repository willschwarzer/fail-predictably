#!/usr/bin/env bash
# Oracle-free score n-grid for the paper's appendix "Forecastability training
# with sampled returns": REINFORCE meta phase of 600 steps with
# REINFORCE_SCORE=neg_return (scores are -sampled return, so no V* enters the
# forecastability phase; the ema baseline keeps the PG term V*-free too).
# 12 fleets / 120 runs. Tags put "negret" between the estimator and m600
# segments so globs for the regret-score fleets cannot match them.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
STAMP=20260725
cd "$REPO_ROOT"

for est in plugin batch; do
  for n in 1 2 4 8 16 32; do
    (
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
      export UNION_CACHE_TOP_C=0
      export GRAD_ESTIMATOR=reinforce
      export PRETRAIN_GRAD_ESTIMATOR=exact
      export REINFORCE_META_ESTIMATOR="${est}"
      export REINFORCE_ROLLOUTS_PER_TASK="${n}"
      export REINFORCE_BASELINE=ema
      export REINFORCE_SCORE=neg_return
      export THREADS_PER_TASK=16
      export MEMORY=16G
      export TIME_LIMIT=0-08:00:00
      export ARRAY_CONCURRENCY=0
      unset REUSE_PRETRAIN_ROOT INIT_PRETRAIN_ROOT INIT_CHECKPOINT RESUME_ROOT   # these runs pretrain their own policies (see the submitter)
      export EXPERIMENT_TAG="rlquant_trap_report_bipolar_sev01_reinforce_${est}_negret_m600_n${n}_lr1e4_prex_10seeds_${STAMP}"
      bash "$REPO_ROOT/scripts/gridworld/submit_rlquant_trap_report_array.sh"
    )
  done
done

echo "All negret m600 fleets submitted."
