#!/usr/bin/env bash
# Post-hoc calibration of every gridworld fleet the paper reads: the figure of
# "Multi-task RL: gridworld navigation with traps" (Cal., RL + cal., Ours + cal.) and
# the appendix "Forecastability training with sampled returns" (its forecast folds and
# its calibration paragraph). Runs posthoc_calibration_baseline_rlquant.py (fit
# affine calibration on the meta-split forecast pairs, apply to heldout) for BOTH
# models (pretrained and meta_learned) and both scales (regret and neg_return) on
# every seed of every fleet matching the glob list below. Pure post-processing of
# forecast_traces.npz; no model reruns. Writes
# posthoc_calibration_{model}[_negret].json into each seed dir.
# Idempotent: re-running overwrites. Intended to run inside an sbatch allocation
# (run_rlquant_posthoc_calibration_sweep.sbatch); parallelism = SLURM_CPUS_PER_TASK
# (1 outside Slurm; the paper's sweep ran 16 at a time).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PYTHON="${PYTHON:-python}"
RESULTS="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
NPROC=${SLURM_CPUS_PER_TASK:-1}
cd "$REPO_ROOT"

GLOBS=(
  # The gridworld figure in "Multi-task RL: gridworld navigation with traps".
  "rlquant_trap_report_bipolar_sev01_metatrain_10seeds_canonicalhparams_*"
  "rlquant_trap_report_bipolar_sev01_baseline_sft_union_10seeds_*"
  "rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_[0-9]*seeds_*"
  # The appendix "Forecastability training with sampled returns".
  "rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_m600_10seeds_*"
  "rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_m600ext_10seeds_*"
  "rlquant_trap_report_bipolar_sev01_reinforce_*_10seeds_*"
  "rlquant_trap_report_bipolar_sev01_reinforce_plugin_negret_prenan_*"
  "rlquant_trap_report_bipolar_sev01_baseline_sft_union_reinforce_*_10seeds_*"
)

jobs_file=$(mktemp)
for g in "${GLOBS[@]}"; do
  for run_root in "$RESULTS"/$g; do
    [[ -d "$run_root" ]] || continue
    for seed_dir in "$run_root"/seed_*; do
      [[ -f "$seed_dir/forecast_traces.npz" ]] || continue
      for model in pretrained meta_learned; do
        for scale in regret neg_return; do
          echo "$seed_dir|$model|$scale" >> "$jobs_file"
        done
      done
    done
  done
done
sort -u "$jobs_file" -o "$jobs_file"
echo "calibration jobs: $(wc -l < "$jobs_file")"

run_one() {
  local seed_dir model scale suffix
  IFS='|' read -r seed_dir model scale <<< "$1"
  suffix=""
  [[ "$scale" == "neg_return" ]] && suffix="_negret"
  "$PYTHON" "$REPO_ROOT/scripts/gridworld/posthoc_calibration_baseline_rlquant.py" \
    --run-dir "$seed_dir" \
    --out-json "$seed_dir/posthoc_calibration_${model}${suffix}.json" \
    --model "$model" --scale "$scale" > /dev/null \
    || echo "FAILED: $seed_dir ($model, $scale)" >&2
}
export -f run_one
export PYTHON REPO_ROOT

xargs -a "$jobs_file" -P "$NPROC" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs_file"
echo "calibration sweep done."
