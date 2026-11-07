#!/bin/bash
# Submits the runs of the paper appendix "Forecastability on natural user prompts" and the
# evaluations of their selected checkpoints, as environment overrides of the launchers in this
# directory. Run it with bash on a login node; it submits from the repo root.
#
#   [DRY=1] bash scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh STAGE [GROUP ...]
#
# STAGE
#   train     Submit the training runs. A run whose final/ exists, or whose job is queued or
#             running, is skipped. A forecastability run that stopped early resumes from its
#             resume_latest.pt when resubmitted; an SFT run starts over.
#   snapshot  Copy each run's current best/ and best_step.json to
#             $RESULTS_ROOT/natural_best_snapshots/<run tag>_at<step>/. The trainers keep only
#             the latest best/, so an earlier one can be selected only if a copy exists: run
#             this now and then while the runs train (for example every ten minutes with
#             `watch`). The paper's copies were taken by hand at irregular times.
#   eval      For each finished run, evaluate its best/ with run_lquant_ldnoobw_eval_adapter.sbatch
#             (ARM=<label>at<step>). If that evaluation's single-token KL exceeds 0.7, the next
#             call evaluates the newest earlier snapshot instead, and so on; with no earlier
#             snapshot left, selection falls back to the run's own end-of-run evaluation.
#             Then select_natural_checkpoint.py applies the selection rule, and
#             collect_natural_results.py writes $RESULTS_ROOT/summaries/natural_results_verified.json.
#             The choice depends on which checkpoints were kept. In the paper, 8B seed 17 fell
#             back from step 184 to a copy of step 155; 8B seed 23 at KL weight 2.0 fell back
#             to its endpoint because no earlier best/ had been copied (its steps 110 and 128
#             had lower training error than the endpoint).
#   cal       For each finished forecastability run of the main group, fit the post-hoc
#             calibration of the pretrained forecast with
#             scripts/password_game/run_posthoc_calibration_baseline.sbatch. It writes
#             posthoc_calibration_baseline.json into the run directory, where
#             collect_natural_results.py reads it.
# GROUP
#   main   (default) the 12 forecastability runs (Qwen3 0.6B, 1.7B, 4B, 8B; seeds 11, 17, 23) and
#          the 12 SFT runs of Table natural-main and the natural_paired_errors figure
#   kl     8B seed 17 at KL weights 2.0 and 5.0 (Table natural-kl; weight 0.5 is in main)
#   extra  8B seeds 11 and 23 at KL weight 2.0, and the 0.6B seed-17 blocklist-prompt control
#   all    all three
#
# Forecastability runs: run_lquant_wildchat_ldnoobw_multiseed.sbatch with LR=3e-5,
# TARGETS_PER_STEP=8, UNION_REFRESH=20, KL_TRAIN=nonflagged and KL_COEF=0.5 (2.0 or 5.0 in the KL
# controls); the blocklist-prompt control used LR=1e-5 and BLOCKLIST=1. TAG_DATE carries the
# run-name suffix (mb8klnf05, mb8klnf20, mb8klnf50, blkklnf05). MICRO_BATCH and
# SCORE_TOKEN_BUDGET only set how prompts are batched (memory and speed); the values below are
# those of the paper's runs. SFT runs: run_lquant_wildchat_ldnoobw_sft.sbatch at its defaults.
# The evaluation labels repeat the paper's run names, including the 8B seed-11 and seed-23
# labels without the mb8 prefix; those runs used the same eight tasks per update.
#
# Prerequisites: the four banks (scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch) and, in
# each, the KL training split written by make_kl_train_split.py.
# Set SBATCH_ACCOUNT / SBATCH_PARTITION (and SBATCH_QOS) for your cluster; sbatch reads them.
# Hardware: one 80 GB GPU per job (the paper used A100-80GB cards).
set -euo pipefail
DRY=${DRY:-0}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/results}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
PYTHON="${PYTHON:-python}"
export REPO_ROOT RESULTS_ROOT TASK_BANK_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
D=$REPO_ROOT/scripts/wildchat/natural_prompts
SNAP=$RESULTS_ROOT/natural_best_snapshots
KL_MAX=0.7

STAGE=${1:?usage: [DRY=1] bash $0 train|snapshot|eval|cal [main|kl|extra|all ...]}
shift
SEL=("$@")
(( ${#SEL[@]} )) || SEL=(main)
[[ " ${SEL[*]} " == *" all "* ]] && SEL=(main kl extra)
case $STAGE in train|snapshot|eval|cal) ;; *) echo "unknown stage $STAGE (train, snapshot, eval or cal)" >&2; exit 1 ;; esac
for g in "${SEL[@]}"; do
  case $g in main|kl|extra) ;; *) echo "unknown group $g (main, kl, extra or all)" >&2; exit 1 ;; esac
done

# kind group size seed run-suffix eval-label LR KL_COEF MICRO_BATCH SCORE_TOKEN_BUDGET BLOCKLIST
RUNS=(
  "ours main  0.6b 11 mb8klnf05 mb8klnf05 3e-5 0.5 256 32768 0"
  "ours main  0.6b 17 mb8klnf05 mb8klnf05 3e-5 0.5 48  8192  0"
  "ours main  0.6b 23 mb8klnf05 mb8klnf05 3e-5 0.5 256 32768 0"
  "ours main  1.7b 11 mb8klnf05 mb8klnf05 3e-5 0.5 48  8192  0"
  "ours main  1.7b 17 mb8klnf05 mb8klnf05 3e-5 0.5 128 16384 0"
  "ours main  1.7b 23 mb8klnf05 mb8klnf05 3e-5 0.5 128 16384 0"
  "ours main  4b   11 mb8klnf05 mb8klnf05 3e-5 0.5 96  8192  0"
  "ours main  4b   17 mb8klnf05 mb8klnf05 3e-5 0.5 96  8192  0"
  "ours main  4b   23 mb8klnf05 mb8klnf05 3e-5 0.5 96  8192  0"
  "ours main  8b   11 mb8klnf05 klnf05    3e-5 0.5 96  8192  0"
  "ours main  8b   17 mb8klnf05 mb8klnf05 3e-5 0.5 96  8192  0"
  "ours main  8b   23 mb8klnf05 klnf05    3e-5 0.5 96  8192  0"
  "sft  main  0.6b 11 - sftklnf05"
  "sft  main  0.6b 17 - sftklnf05"
  "sft  main  0.6b 23 - sftklnf05"
  "sft  main  1.7b 11 - sftklnf05"
  "sft  main  1.7b 17 - sftklnf05"
  "sft  main  1.7b 23 - sftklnf05"
  "sft  main  4b   11 - sft4bklnf05"
  "sft  main  4b   17 - sft4bklnf05"
  "sft  main  4b   23 - sft4bklnf05"
  "sft  main  8b   11 - sft8bklnf05"
  "sft  main  8b   17 - sft8bklnf05"
  "sft  main  8b   23 - sft8bklnf05"
  "ours kl    8b   17 mb8klnf20 mb8klnf20 3e-5 2.0 96  8192  0"
  "ours kl    8b   17 mb8klnf50 klnf50    3e-5 5.0 96  8192  0"
  "ours extra 8b   11 mb8klnf20 klnf20    3e-5 2.0 96  8192  0"
  "ours extra 8b   23 mb8klnf20 klnf20    3e-5 2.0 96  8192  0"
  "ours extra 0.6b 17 blkklnf05 blkklnf05 1e-5 0.5 256 32768 1"
)

run() { if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }
queued() { [[ -n $(squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null | head -n 1) ]]; }
json_get() { "$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
kl_ok() { "$PYTHON" -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["single_token"] > float(sys.argv[2]))' "$1" "$KL_MAX"; }
model_of() { case $1 in 0.6b) echo Qwen/Qwen3-0.6B ;; 1.7b) echo Qwen/Qwen3-1.7B ;; 4b) echo Qwen/Qwen3-4B ;; 8b) echo Qwen/Qwen3-8B ;; esac; }
size_tag() { case $1 in 0.6b) echo 0p6b ;; 1.7b) echo 1p7b ;; *) echo "$1" ;; esac; }
model_tag() { echo "$1" | tr '/.' '__' | tr '[:upper:]' '[:lower:]'; }
run_dir() {  # kind size seed suffix
  local mt; mt=$(model_tag "$(model_of "$2")")
  if [[ $1 == sft ]]; then echo "$RESULTS_ROOT/lquant_wildchat_ldnoobw_sft_union_${mt}_klnf05_seed${3}/seed_${3}"
  else echo "$RESULTS_ROOT/lquant_wildchat_ldnoobw_${mt}_iog_full_boot1_cache296_seed${3}_${4}/seed_${3}"; fi
}
eval_dir() {  # size seed arm
  echo "$RESULTS_ROOT/lquant_wildchat_ldnoobw_$(model_tag "$(model_of "$1")")_iog_full_boot1_cache296_seed${2}_besteval_${3}/seed_${2}"
}

train() {  # kind size seed suffix label lr kl mb stb blk
  local kind=$1 size=$2 seed=$3 suffix=$4 lr=${6:-} kl=${7:-} mb=${8:-} stb=${9:-} blk=${10:-}
  local rd model split name jid
  rd=$(run_dir "$kind" "$size" "$seed" "$suffix")
  model=$(model_of "$size")
  split=$TASK_BANK_ROOT/wildchat_ldnoobw_$(model_tag "$model")_T40_p1870/bank_seed_11/kl_train_nonflagged_en_disjoint.jsonl
  name=nat_$([[ $kind == sft ]] && echo sft || echo "$suffix")_$(size_tag "$size")_s${seed}
  [[ -d $rd/final ]] && { echo "[skip] $name: finished ($rd)" >&2; return; }
  queued "$name" && { echo "[skip] $name: queued or running" >&2; return; }
  [[ -f $split ]] || { echo "[missing] $split: build the bank, then run make_kl_train_split.py on it" >&2; return; }
  if [[ $kind == ours ]]; then
    jid=$(run --job-name="$name" \
          --export="ALL,MODEL=$model,SEED=$seed,TAG_DATE=$suffix,LR=$lr,KL_COEF=$kl,KL_TRAIN=nonflagged,TARGETS_PER_STEP=8,UNION_REFRESH=20,MICRO_BATCH=$mb,SCORE_TOKEN_BUDGET=$stb,BLOCKLIST=$blk" \
          "$D/run_lquant_wildchat_ldnoobw_multiseed.sbatch")
  else
    jid=$(run --job-name="$name" --export="ALL,SIZE=$(size_tag "$size"),SEED=$seed" \
          "$D/run_lquant_wildchat_ldnoobw_sft.sbatch")
  fi
  echo "[train] $name -> job $jid ($rd)"
}

snapshot() {  # kind size seed suffix
  local rd tag step dest tmp
  rd=$(run_dir "$1" "$2" "$3" "$4")
  [[ -f $rd/best_step.json && -f $rd/best/adapter_model.safetensors ]] || return 0
  # The trainer writes best/ first and best_step.json second; an adapter newer than the JSON
  # means a rewrite is in progress.
  [[ $rd/best/adapter_model.safetensors -nt $rd/best_step.json ]] && { echo "[retry] $rd: best/ is being rewritten" >&2; return 0; }
  tag=$(basename "$(dirname "$rd")")
  step=$(json_get "$rd/best_step.json" step)
  dest=$SNAP/${tag}_at${step}
  [[ -e $dest ]] && return 0
  if (( DRY )); then echo "[dry] copy $rd/best and best_step.json to $dest" >&2; return 0; fi
  mkdir -p "$SNAP"
  tmp=$(mktemp -d "$SNAP/.${tag}_at${step}.XXXXXX")
  cp -r "$rd/best" "$tmp/best"
  cp "$rd/best_step.json" "$tmp/best_step.json"
  if [[ $(json_get "$tmp/best_step.json" step) == "$step" ]] && ! [[ $rd/best/adapter_model.safetensors -nt $rd/best_step.json ]]; then
    mv "$tmp" "$dest"
    echo "[snapshot] $dest" >&2
  else
    rm -rf -- "$tmp"
    echo "[retry] $rd: best/ changed during the copy" >&2
  fi
}

evaluate() {  # kind size seed suffix label lr kl mb stb blk
  local kind=$1 size=$2 seed=$3 suffix=$4 label=$5 blk=${10:-0}
  local rd tag step adapter arm ev name jid s d prev
  rd=$(run_dir "$kind" "$size" "$seed" "$suffix")
  [[ -d $rd/final && -f $rd/best_step.json ]] || { echo "[wait] $rd: not finished" >&2; return; }
  tag=$(basename "$(dirname "$rd")")
  step=$(json_get "$rd/best_step.json" step)
  adapter=$rd/best
  while :; do
    arm=${label}at${step}
    ev=$(eval_dir "$size" "$seed" "$arm")
    name=nateval_${arm}_$(size_tag "$size")_s${seed}
    if [[ -f $ev/kl_eval_results.json && -f $ev/test_results.json ]]; then
      if kl_ok "$ev/kl_eval_results.json"; then echo "[done] $arm ($size s$seed): KL within $KL_MAX" >&2; return; fi
      # Walk back: the newest snapshot of this run before $step.
      prev=
      for d in "$SNAP/${tag}"_at*/; do
        [[ -f $d/best_step.json ]] || continue
        s=${d%/}; s=${s##*_at}
        (( s < step )) && [[ -z $prev || $s -gt ${prev##*_at} ]] && prev=${d%/}
      done
      if [[ -z $prev ]]; then
        echo "[endpoint] $arm ($size s$seed): KL above $KL_MAX and no earlier snapshot; selection uses the run's end-of-run evaluation" >&2
        return
      fi
      step=${prev##*_at}
      adapter=$prev/best
      continue
    fi
    queued "$name" && { echo "[skip] $name: queued or running" >&2; return; }
    [[ -d $ev ]] && { echo "[check] $ev exists without results: a failed evaluation? Move it aside to resubmit." >&2; return; }
    jid=$(run --job-name="$name" \
          --export="ALL,ADAPTER=$adapter,ARM=$arm,SEED=$seed,MODEL=$(model_of "$size"),BLOCKLIST=$blk" \
          "$D/run_lquant_ldnoobw_eval_adapter.sbatch")
    echo "[eval] $name -> job $jid (adapter $adapter)"
    return
  done
}

calibrate() {  # kind size seed suffix
  local rd name jid
  rd=$(run_dir "$1" "$2" "$3" "$4")
  name=natcal_$(size_tag "$2")_s${3}
  [[ -d $rd/final ]] || { echo "[wait] $rd: not finished" >&2; return; }
  [[ -f $rd/posthoc_calibration_baseline.json ]] && { echo "[done] $name" >&2; return; }
  queued "$name" && { echo "[skip] $name: queued or running" >&2; return; }
  jid=$(run --job-name="$name" --export="ALL,RUN_DIR=$rd,OUT_JSON=$rd/posthoc_calibration_baseline.json" \
        "$REPO_ROOT/scripts/password_game/run_posthoc_calibration_baseline.sbatch")
  echo "[cal] $name -> job $jid ($rd)"
}

mkdir -p logs
for spec in "${RUNS[@]}"; do
  read -r kind group size seed suffix label lr kl mb stb blk <<< "$spec"
  [[ " ${SEL[*]} " == *" $group "* ]] || continue
  case $STAGE in
    train)    train "$kind" "$size" "$seed" "$suffix" "$label" "${lr:-}" "${kl:-}" "${mb:-}" "${stb:-}" "${blk:-}" ;;
    snapshot) snapshot "$kind" "$size" "$seed" "$suffix" ;;
    eval)     evaluate "$kind" "$size" "$seed" "$suffix" "$label" "${lr:-}" "${kl:-}" "${mb:-}" "${stb:-}" "${blk:-0}" ;;
    cal)      if [[ $kind == ours && $group == main ]]; then calibrate "$kind" "$size" "$seed" "$suffix"; fi ;;
  esac
done
