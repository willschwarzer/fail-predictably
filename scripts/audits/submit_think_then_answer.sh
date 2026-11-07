#!/bin/bash
# Submit the think-then-answer probe (think_then_answer_probe.sbatch) for the checkpoints whose generations with
# thinking on the paper reports, in the appendices "Experimental details" (paragraph "KL divergence from the base
# model"), "Forecastability on natural user prompts" and "Forecastability training with minimal change to the
# model". GROUP is one of
#   natural  the seed-11 Ours and SFT checkpoints at Qwen3 0.6B, 1.7B and 4B, one job per size
#   fig4     the runs of Figure 4 (Section 6.1): Ours at 10 seeds and SFT at 5, their final adapters
#   mcpw     the minimal-change runs on the password bank at step 200: the anchor and spread constraints
#            together ("both") at 5 seeds, and the other three arms at seed 11
#   mcwc     the minimal-change runs on WildChat at step 200: the anchor and spread constraints together at
#            5 seeds, the mask alone and the anchor constraint alone at seed 11, and the mask with the spread
#            constraint at seed 43
# The default is all four. The last three groups run their trained checkpoints four at a time. The first job
# also runs the base model, and the others wait for it and reuse its output (the probe's --only).
# Outputs, one JSONL per model and summary.csv in each directory: $RESULTS_ROOT/think_then_answer/ (natural) and
# $RESULTS_ROOT/think_then_answer_fig4_mc/ (the other groups). Classify the generations afterwards with
#   python scripts/audits/think_then_answer_classify.py $RESULTS_ROOT/think_then_answer $RESULTS_ROOT/think_then_answer_fig4_mc
# Usage (from the repo root, on a login node): [DRY=1] bash scripts/audits/submit_think_then_answer.sh [GROUP ...]
# DRY=1 prints the sbatch lines without submitting them.
set -euo pipefail

DRY=${DRY:-0}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
export REPO_ROOT RESULTS_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
RUNS=$RESULTS_ROOT/lquant_dynamics/runs
SB=scripts/audits/think_then_answer_probe.sbatch
OUT_NATURAL=$RESULTS_ROOT/think_then_answer
OUT_GROUPS=$RESULTS_ROOT/think_then_answer_fig4_mc
CHUNK=4   # trained checkpoints per job; Figure 4's 15 in one job would take about two hours on an A16

# Prints one think_then_answer_probe.py argument per line for group $1.
group_args() {
  local g=$1 s d x
  echo --group; echo "$g"
  case $g in
    fig4)   # Figure 4 (password game): Ours 10 seeds, SFT 5 seeds, final adapters
      for s in 11 17 23 29 31 37 41 43 47 53; do
        d=$RESULTS_ROOT/lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_seed${s}_20260505/seed_$s
        echo --model; echo "Ours_s$s=$d/config.json:$d/final/adapter_model.safetensors"; done
      for s in 11 17 23 29 31; do
        d=$RESULTS_ROOT/lquant_40slots_modbulkgcg_sft_union_seed${s}_20260504/seed_$s
        echo --model; echo "SFT_s$s=$d/config.json:$d/final/adapter_model.safetensors"; done ;;
    mcpw)   # minimal change, password, step 200: both constraints 5 seeds; the other three arms seed 11
      for s in 11 17 23 29 43; do d=$RUNS/lqpw_pw2apsp200_seed$s
        echo --model; echo "both_s$s=$d/config.json:$d/snapshots/step_0200/adapter_model.safetensors"; done
      for x in mask:lqpw_pw2iog200_seed11 anchor:lqpw_pw2nbap200_seed11 maskspread:lqpw_pw2iogsb200_seed11; do
        d=$RUNS/${x#*:}; echo --model; echo "${x%%:*}_s11=$d/config.json:$d/snapshots/step_0200/adapter_model.safetensors"; done ;;
    mcwc)   # minimal change, WildChat, step 200: both constraints 5 seeds; the other arms one seed (mask + spread: s43)
      for s in 11 17 23 29 43; do d=$RUNS/lqdyn_v2e2ap50spx200_seed$s
        echo --model; echo "both_s$s=$d/config.json:$d/snapshots/step_0200/adapter_model.safetensors"; done
      for x in mask_s11:lqdyn_v2e2iogap50x200_seed11 anchor_s11:lqdyn_v2e2nbap50x200_seed11 maskspread_s43:lqdyn_v2e2iogsbap50spx200_seed43; do
        d=$RUNS/${x#*:}; echo --model; echo "${x%%:*}=$d/config.json:$d/snapshots/step_0200/adapter_model.safetensors"; done ;;
    *) echo "unknown group $g" >&2; return 1 ;;
  esac
}

run() { if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }

groups=("$@")
if [[ ${#groups[@]} -eq 0 ]]; then groups=(natural fig4 mcpw mcwc); fi
for g in "${groups[@]}"; do
  if [[ $g == natural ]]; then
    for s in 0.6b 1.7b 4b; do
      run --job-name="think_answer_$s" "$SB" --size "$s" --out-dir "$OUT_NATURAL" >/dev/null
    done
    continue
  fi
  args=()
  while IFS= read -r line; do args+=("$line"); done < <(group_args "$g")
  labels=()
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ ${args[i]} == --model ]]; then labels+=("${args[i + 1]%%=*}"); fi
  done
  first=""
  for ((i = 0; i < ${#labels[@]}; i += CHUNK)); do
    only=$(IFS=,; echo "${labels[*]:i:CHUNK}")
    if [[ -z $first ]]; then
      first=$(run --job-name="think_answer_$g" "$SB" "${args[@]}" --out-dir "$OUT_GROUPS" --only "$only")
      first=${first%%;*}
    else
      run --job-name="think_answer_$g" --dependency="afterok:$first" "$SB" "${args[@]}" --out-dir "$OUT_GROUPS" \
        --only "$only" >/dev/null
    fi
  done
done
