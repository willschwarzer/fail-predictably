#!/bin/bash
# Submit the 10-seed 0.6B fit/deploy distribution-shift fleet (canonical seeds).
# See run_lquant_40slots_nopwseed_distshift_0p6b.sbatch for the recipe.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"   # jobs write logs/ relative to the submit directory
TAG_DATE=${TAG_DATE:-20260726}

for SEED in 11 17 23 29 31 37 41 43 47 53; do
  sbatch --export=ALL,SEED=${SEED},TAG_DATE=${TAG_DATE} \
    "$REPO_ROOT/scripts/password_game/run_lquant_40slots_nopwseed_distshift_0p6b.sbatch"
done
