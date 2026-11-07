#!/bin/bash
# Held-out reads of the 100-step WildChat runs of the four minimal-change arms (Section 6.3
# and the appendix "Forecastability training with minimal change to the model"). The
# poolscale wave of submit_lqdyn_fleet.sh trains these runs but queues no readers, so run
# this once the trainers have finished. Each run gets two readers, for steps 10-30 and then
# 40-100, which should run on the same GPU model (see snapshot_heldout_eval.sbatch).
# Usage: bash scripts/minimal_change/submit_wc_heldout_reads.sh [SEED ...]   (default: 11 17 23 29 43)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"   # jobs write logs/ relative to the submit directory
RUNS="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}/lquant_dynamics/runs"
SB="$REPO_ROOT/scripts/minimal_change/snapshot_heldout_eval.sbatch"
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then SEEDS=(11 17 23 29 43); fi

for arm in v2e2iogap50 v2e2nbap50 v2e2iogsbap50sp v2e2ap50sp; do
  for s in "${SEEDS[@]}"; do
    r="$RUNS/lqdyn_${arm}_seed$s"
    if [[ ! -d "$r" ]]; then echo "skip $r (no such run)"; continue; fi
    j=$(sbatch --parsable --time=06:00:00 --job-name=lqsnap_${arm}_seed$s "$SB" "$r" 10 20 30)
    j=${j%%;*}
    sbatch --time=16:00:00 --dependency=afterany:$j --job-name=lqsnap_${arm}_seed$s \
      "$SB" "$r" 40 50 60 70 80 90 100
  done
done
