#!/usr/bin/env bash
# The four password banks without the password_seeded family that the password-game
# scaling runs read (lquant_prompt_bank_40slots_nopwseed_gcg6warm_strictv2_qwen3_<size>_20260725,
# sizes 0_6b 1_7b 4b 8b), as the paper built them. CPU only, seconds per bank: the GCG
# rows are copied and only the template families are regenerated.
#
# The whole chain, per size (Qwen/Qwen3-0.6B, -1.7B, -4B, -8B):
#   1. From-scratch strictv2 bank, all four sizes (DATE=20260723):
#        SIZE=<size> MODEL=<model> bash scripts/banks/submit_strictv2_password_bank.sh
#   2. Donor pool from seed 11 of that bank, 0_6b 4b 8b:
#        sbatch --export=ALL,MODEL=<model>,BANK=$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_strictv2_qwen3_<size>_20260723/bank_seed_11,OUT=artifacts/gcg_donor_pools/qwen3_<size>_strictv2_seed11.json \
#          scripts/banks/run_extract_gcg_donor_pool.sbatch
#   3. Warm-start bank from that pool, 0_6b 4b 8b (DATE=20260724):
#        SIZE=<size> MODEL=<model> DONOR_POOL_JSON=artifacts/gcg_donor_pools/qwen3_<size>_strictv2_seed11.json \
#          bash scripts/banks/submit_strictv2_warmstart_bank.sh
#   4. This script.
# 1.7B has no warm-start bank: from-scratch GCG broke no slot at that size (its best
# suffix leaked about 1e-12), so its donor pool holds nothing to warm-start from. Its
# nopwseed bank is derived from the from-scratch bank of step 1 and keeps the gcg6warm
# name, so the 1.7B launchers find it under the same pattern as the other sizes.
#
# Usage:  bash scripts/banks/build_nopwseed_strictv2_banks.sh            (all four)
#         SIZES="0_6b 4b" bash scripts/banks/build_nopwseed_strictv2_banks.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"

SIZES=${SIZES:-"0_6b 1_7b 4b 8b"}

for size in $SIZES; do
  if [[ "$size" == "1_7b" ]]; then
    src="lquant_prompt_bank_40slots_modbulk_gcg6_strictv2_qwen3_1_7b_20260723"
  else
    src="lquant_prompt_bank_40slots_modbulk_gcg6warm_strictv2_qwen3_${size}_20260724"
  fi
  dst="lquant_prompt_bank_40slots_nopwseed_gcg6warm_strictv2_qwen3_${size}_20260725"
  echo "[nopwseed-bank] ${size}: ${src} -> ${dst}"
  "$PYTHON" scripts/banks/rebuild_bank_drop_family.py \
    --src-bank "$TASK_BANK_ROOT/$src" \
    --dst-bank "$TASK_BANK_ROOT/$dst" \
    --drop-family password_seeded \
    --allocate compositional=932 indirect_extraction=466 mutate=466
done
