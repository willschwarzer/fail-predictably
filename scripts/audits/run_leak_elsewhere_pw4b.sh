#!/bin/bash
#SBATCH --job-name=leak_elsewhere_pw4b
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Hardware: one GPU with at least 16 GB (the paper used an Ampere, sm_80, card).
# Leak-elsewhere eval (paper appendix "Checks beyond the training forecast and
# score", subsection "Password leakage beyond the first token") on the
# rebuilt-bank password 4B fleet (base/ours/SFT x 3 seeds).
# Usage (from the repo root): sbatch scripts/audits/run_leak_elsewhere_pw4b.sh
# Inputs: the ours runs from scripts/password_game/run_lquant_40slots_nopwseed_strictv2_iog_full_4b.sbatch
# (run with TAG_DATE=20260725, the date in the run tags below) and the SFT runs from
# scripts/password_game/run_lquant_sft_nopwseed_strictv2.sbatch, under $RESULTS_ROOT.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"

for S in 11 17 23; do
  OURS="$RESULTS_ROOT/lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_qwen3_4b_seed${S}_20260725/seed_${S}"
  SFT="$RESULTS_ROOT/lquant_40slots_nopwseed_strictv2_sft_union_qwen3_4b_seed${S}/seed_${S}"
  "$PYTHON" scripts/audits/leak_elsewhere_eval.py \
    --run-dir "$OURS" \
    --model base=none \
    --model ours="$OURS/final" \
    --model sft="$SFT/final" \
    --tail 32 --rand 64 --num-samples 4 --max-new-tokens 64 \
    --gen-batch 8 \
    --out "artifacts/misc/leak_elsewhere_pw4b_seed${S}.json" \
    || echo "seed ${S} FAILED (continuing)"
done
echo "all seeds done"
