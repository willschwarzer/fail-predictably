#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT
SUBMIT_SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}"
# Hardware: CPU only.

EXPERIMENT_TAG=${EXPERIMENT_TAG:-rlquant_trap_open_room_bank_v2_iid_binary_tail_20260403}
OUTROOT=${OUTROOT:-"${TASK_BANK_ROOT}/${EXPERIMENT_TAG}"}
SHARD_TASKS_FILE="${OUTROOT}/shard_tasks.txt"
MERGE_TASKS_FILE="${OUTROOT}/merge_tasks.txt"
GENERATION_PLAN_FILE="${OUTROOT}/generation_plan.json"
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/artifacts/logs/task_banks"}

BANK_SEEDS_STR=${BANK_SEEDS:-"11 17 23 29 31 37 41 43"}
BANK_TARGET_TOTAL=${BANK_TARGET_TOTAL:-52000}
LOW_SHARD_TARGET_COUNT=${LOW_SHARD_TARGET_COUNT:-2000}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-0}
MEMORY_PER_SHARD=${MEMORY_PER_SHARD:-2G}
TIME_LIMIT_PER_SHARD=${TIME_LIMIT_PER_SHARD:-0-00:15:00}
MERGE_MEMORY=${MERGE_MEMORY:-2G}
MERGE_TIME_LIMIT=${MERGE_TIME_LIMIT:-0-00:15:00}

FAMILY_ID=${FAMILY_ID:-trap_open_room}
LOW_TARGET_TOTAL=${LOW_TARGET_TOTAL:-51920}
TAIL_TARGET_TOTAL=${TAIL_TARGET_TOTAL:-80}
LOW_SEVERITY_MIN=${LOW_SEVERITY_MIN:-0.0}
LOW_SEVERITY_MAX=${LOW_SEVERITY_MAX:-0.5}
TAIL_SEVERITY=${TAIL_SEVERITY:-1.0}
ROWS=${ROWS:-8}
COLS=${COLS:-8}
HORIZON=${HORIZON:-10}
GAMMA=${GAMMA:-1.0}
STEP_COST=${STEP_COST:--0.01}
GOAL_REWARD=${GOAL_REWARD:-1.0}
CATASTROPHE_PENALTY=${CATASTROPHE_PENALTY:--1.0}
MIN_START_GOAL_DISTANCE=${MIN_START_GOAL_DISTANCE:-5}
MAX_GENERATION_ATTEMPTS_PER_TASK=${MAX_GENERATION_ATTEMPTS_PER_TASK:-200}

mkdir -p "${OUTROOT}" "${LOG_DIR}"
read -r -a BANK_SEED_ARRAY <<< "${BANK_SEEDS_STR}"
if (( LOW_TARGET_TOTAL + TAIL_TARGET_TOTAL != BANK_TARGET_TOTAL )); then
  echo "LOW_TARGET_TOTAL + TAIL_TARGET_TOTAL must equal BANK_TARGET_TOTAL" >&2
  exit 1
fi
if (( LOW_TARGET_TOTAL < 0 || TAIL_TARGET_TOTAL < 0 )); then
  echo "Component totals must be non-negative." >&2
  exit 1
fi

LOW_FULL_SHARDS=$(( LOW_TARGET_TOTAL / LOW_SHARD_TARGET_COUNT ))
LOW_REMAINDER=$(( LOW_TARGET_TOTAL % LOW_SHARD_TARGET_COUNT ))
TAIL_SHARDS_PER_BANK=$(( TAIL_TARGET_TOTAL > 0 ? 1 : 0 ))
SHARDS_PER_BANK=$(( LOW_FULL_SHARDS + (LOW_REMAINDER > 0 ? 1 : 0) + TAIL_SHARDS_PER_BANK ))

cat > "${GENERATION_PLAN_FILE}" <<EOF
{
  "family_id": "${FAMILY_ID}",
  "generator_config": {
    "rows": ${ROWS},
    "cols": ${COLS},
    "horizon": ${HORIZON},
    "gamma": ${GAMMA},
    "step_cost": ${STEP_COST},
    "goal_reward": ${GOAL_REWARD},
    "catastrophe_penalty": ${CATASTROPHE_PENALTY},
    "min_start_goal_distance": ${MIN_START_GOAL_DISTANCE},
    "max_generation_attempts_per_task": ${MAX_GENERATION_ATTEMPTS_PER_TASK}
  },
  "components": [
    {
      "name": "low",
      "target_total": ${LOW_TARGET_TOTAL},
      "severity_min": ${LOW_SEVERITY_MIN},
      "severity_max": ${LOW_SEVERITY_MAX}
    },
    {
      "name": "tail",
      "target_total": ${TAIL_TARGET_TOTAL},
      "severity_min": ${TAIL_SEVERITY},
      "severity_max": ${TAIL_SEVERITY}
    }
  ]
}
EOF

: > "${SHARD_TASKS_FILE}"
: > "${MERGE_TASKS_FILE}"

global_task_index=0
for bank_seed in "${BANK_SEED_ARRAY[@]}"; do
  bank_dir="${OUTROOT}/bank_seed_${bank_seed}"
  shard_dir="${bank_dir}/shards"
  mkdir -p "${shard_dir}"
  shard_idx=0

  for (( local_idx=0; local_idx<LOW_FULL_SHARDS; local_idx++ )); do
    shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
    shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
    cmd=(
      "${PYTHON_BIN}" -m forecast_experiment.generate_rlquant_bank_shard
      --out "${shard_out}"
      --family-id "${FAMILY_ID}"
      --count "${LOW_SHARD_TARGET_COUNT}"
      --seed "${shard_seed}"
      --split-name "bank_${bank_seed}_shard_${shard_idx}"
      --severity-min "${LOW_SEVERITY_MIN}"
      --severity-max "${LOW_SEVERITY_MAX}"
      --rows "${ROWS}"
      --cols "${COLS}"
      --horizon "${HORIZON}"
      --gamma "${GAMMA}"
      --step-cost "${STEP_COST}"
      --goal-reward "${GOAL_REWARD}"
      --catastrophe-penalty "${CATASTROPHE_PENALTY}"
      --min-start-goal-distance "${MIN_START_GOAL_DISTANCE}"
      --max-generation-attempts-per-task "${MAX_GENERATION_ATTEMPTS_PER_TASK}"
    )
    printf '%q ' "${cmd[@]}" >> "${SHARD_TASKS_FILE}"
    printf '\n' >> "${SHARD_TASKS_FILE}"
    global_task_index=$((global_task_index + 1))
    shard_idx=$((shard_idx + 1))
  done

  if (( LOW_REMAINDER > 0 )); then
    shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
    shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
    cmd=(
      "${PYTHON_BIN}" -m forecast_experiment.generate_rlquant_bank_shard
      --out "${shard_out}"
      --family-id "${FAMILY_ID}"
      --count "${LOW_REMAINDER}"
      --seed "${shard_seed}"
      --split-name "bank_${bank_seed}_shard_${shard_idx}"
      --severity-min "${LOW_SEVERITY_MIN}"
      --severity-max "${LOW_SEVERITY_MAX}"
      --rows "${ROWS}"
      --cols "${COLS}"
      --horizon "${HORIZON}"
      --gamma "${GAMMA}"
      --step-cost "${STEP_COST}"
      --goal-reward "${GOAL_REWARD}"
      --catastrophe-penalty "${CATASTROPHE_PENALTY}"
      --min-start-goal-distance "${MIN_START_GOAL_DISTANCE}"
      --max-generation-attempts-per-task "${MAX_GENERATION_ATTEMPTS_PER_TASK}"
    )
    printf '%q ' "${cmd[@]}" >> "${SHARD_TASKS_FILE}"
    printf '\n' >> "${SHARD_TASKS_FILE}"
    global_task_index=$((global_task_index + 1))
    shard_idx=$((shard_idx + 1))
  fi

  if (( TAIL_TARGET_TOTAL > 0 )); then
    shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
    shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
    cmd=(
      "${PYTHON_BIN}" -m forecast_experiment.generate_rlquant_bank_shard
      --out "${shard_out}"
      --family-id "${FAMILY_ID}"
      --count "${TAIL_TARGET_TOTAL}"
      --seed "${shard_seed}"
      --split-name "bank_${bank_seed}_shard_${shard_idx}"
      --severity-min "${TAIL_SEVERITY}"
      --severity-max "${TAIL_SEVERITY}"
      --rows "${ROWS}"
      --cols "${COLS}"
      --horizon "${HORIZON}"
      --gamma "${GAMMA}"
      --step-cost "${STEP_COST}"
      --goal-reward "${GOAL_REWARD}"
      --catastrophe-penalty "${CATASTROPHE_PENALTY}"
      --min-start-goal-distance "${MIN_START_GOAL_DISTANCE}"
      --max-generation-attempts-per-task "${MAX_GENERATION_ATTEMPTS_PER_TASK}"
    )
    printf '%q ' "${cmd[@]}" >> "${SHARD_TASKS_FILE}"
    printf '\n' >> "${SHARD_TASKS_FILE}"
    global_task_index=$((global_task_index + 1))
    shard_idx=$((shard_idx + 1))
  fi

  merged_out="${bank_dir}/bank_specs.jsonl"
  merge_cmd=(
    "${PYTHON_BIN}" -m forecast_experiment.merge_rlquant_bank_shards
    --shard-glob "${shard_dir}/shard_*.jsonl"
    --out "${merged_out}"
    --target-total "${BANK_TARGET_TOTAL}"
    --generation-plan "${GENERATION_PLAN_FILE}"
  )
  printf '%q ' "${merge_cmd[@]}" >> "${MERGE_TASKS_FILE}"
  printf '\n' >> "${MERGE_TASKS_FILE}"
done

array_max=$(( global_task_index - 1 ))
if (( array_max < 0 )); then
  echo "No shard tasks configured." >&2
  exit 1
fi

merge_array_max=$(( ${#BANK_SEED_ARRAY[@]} - 1 ))

shard_sbatch_args=(
  -c 1
  --mem="${MEMORY_PER_SHARD}"
  --time="${TIME_LIMIT_PER_SHARD}"
  --output="${LOG_DIR}/%A_%a.out"
  --error="${LOG_DIR}/%A_%a.err"
  --export="ALL,TASKS_FILE=${SHARD_TASKS_FILE},PYTHONPATH=${PYTHONPATH}"
)
if (( ARRAY_CONCURRENCY > 0 )); then
  shard_sbatch_args+=(--array="0-${array_max}%${ARRAY_CONCURRENCY}")
else
  shard_sbatch_args+=(--array="0-${array_max}")
fi

submit_out=$(sbatch "${shard_sbatch_args[@]}" "${REPO_ROOT}/scripts/slurm/array_runner.sh")
echo "${submit_out}"
shard_jobid=$(awk '{print $4}' <<< "${submit_out}")
if [[ -z "${shard_jobid}" ]]; then
  echo "Failed to parse shard array job id." >&2
  exit 1
fi

merge_sbatch_args=(
  --dependency="afterok:${shard_jobid}"
  -c 1
  --mem="${MERGE_MEMORY}"
  --time="${MERGE_TIME_LIMIT}"
  --array="0-${merge_array_max}"
  --output="${LOG_DIR}/%A_%a.out"
  --error="${LOG_DIR}/%A_%a.err"
  --export="ALL,TASKS_FILE=${MERGE_TASKS_FILE},PYTHONPATH=${PYTHONPATH}"
)

merge_out=$(sbatch "${merge_sbatch_args[@]}" "${REPO_ROOT}/scripts/slurm/array_runner.sh")
echo "${merge_out}"
merge_jobid=$(awk '{print $4}' <<< "${merge_out}")

cat > "${OUTROOT}/submission.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "outroot": "${OUTROOT}",
  "submit_script": "${SUBMIT_SCRIPT_PATH}",
  "shard_tasks_file": "${SHARD_TASKS_FILE}",
  "merge_tasks_file": "${MERGE_TASKS_FILE}",
  "generation_plan_file": "${GENERATION_PLAN_FILE}",
  "shard_array_job_id": "${shard_jobid}",
  "merge_array_job_id": "${merge_jobid}",
  "bank_seeds": [$(printf '%s' "${BANK_SEED_ARRAY[*]}" | sed 's/ /, /g')],
  "bank_target_total": ${BANK_TARGET_TOTAL},
  "low_target_total": ${LOW_TARGET_TOTAL},
  "tail_target_total": ${TAIL_TARGET_TOTAL},
  "low_shard_target_count": ${LOW_SHARD_TARGET_COUNT},
  "shards_per_bank": ${SHARDS_PER_BANK},
  "array_concurrency": ${ARRAY_CONCURRENCY},
  "account": "${SBATCH_ACCOUNT:-}",
  "partition": "${SBATCH_PARTITION:-}",
  "merge_partition": "${SBATCH_PARTITION:-}",
  "family_id": "${FAMILY_ID}",
  "low_severity_min": ${LOW_SEVERITY_MIN},
  "low_severity_max": ${LOW_SEVERITY_MAX},
  "tail_severity": ${TAIL_SEVERITY},
  "compatibility_note": "Experiments using this bank should not set horizon greater than the bank generation horizon, because protected-path generation and task validity were computed at generation time.",
  "qos": "${SBATCH_QOS:-}"
}
EOF

cat > "${OUTROOT}/EXPERIMENT.md" <<EOF
# RLQuant Task Bank Generation

## Submission
- Shard array job id: \`${shard_jobid}\`
- Merge array job id: \`${merge_jobid}\`
- Output root: \`${OUTROOT}\`
- Submission script: \`${SUBMIT_SCRIPT_PATH}\`
- Bank seeds: \`${BANK_SEEDS_STR}\`
- Account: \`${SBATCH_ACCOUNT:-<default>}\`
- Partition: \`${SBATCH_PARTITION:-<default>}\`

## Bank Geometry
- Target unique tasks per bank: \`${BANK_TARGET_TOTAL}\`
- Low-mode target tasks per bank: \`${LOW_TARGET_TOTAL}\`
- Tail target tasks per bank: \`${TAIL_TARGET_TOTAL}\`
- Low-mode target tasks per full shard: \`${LOW_SHARD_TARGET_COUNT}\`
- Shards per bank: \`${SHARDS_PER_BANK}\`
- Family: \`${FAMILY_ID}\`

## IID Severity Mixture
- Low mode: \`${LOW_TARGET_TOTAL}\` tasks with severity sampled from \`${LOW_SEVERITY_MIN} .. ${LOW_SEVERITY_MAX}\`
- Tail mode: \`${TAIL_TARGET_TOTAL}\` tasks with severity fixed at \`${TAIL_SEVERITY}\`

## Generator Config
- Grid: \`${ROWS}x${COLS}\`
- Horizon: \`${HORIZON}\`
- Gamma: \`${GAMMA}\`
- Step cost: \`${STEP_COST}\`
- Goal reward: \`${GOAL_REWARD}\`
- Catastrophe penalty: \`${CATASTROPHE_PENALTY}\`
- Min start-goal distance: \`${MIN_START_GOAL_DISTANCE}\`

## Compatibility Note
- Experiments using this bank should not set horizon greater than the bank-generation horizon.
- For trap banks in particular, protected-path generation and task validity depend on the generation-time horizon, so a larger experiment horizon changes the task distribution.

## Files
- Shard tasks file: \`${SHARD_TASKS_FILE}\`
- Merge tasks file: \`${MERGE_TASKS_FILE}\`
- Generation plan file: \`${GENERATION_PLAN_FILE}\`
EOF

echo "Shard Array JobID: ${shard_jobid}"
echo "Merge Array JobID: ${merge_jobid}"
echo "Experiment note: ${OUTROOT}/EXPERIMENT.md"
