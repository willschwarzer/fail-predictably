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

EXPERIMENT_TAG=${EXPERIMENT_TAG:-rlquant_trap_open_room_layout_bank_v1_iid_binary_tail_20260404}
OUTROOT=${OUTROOT:-"${TASK_BANK_ROOT}/${EXPERIMENT_TAG}"}
SHARD_TASKS_FILE="${OUTROOT}/shard_tasks.txt"
MERGE_TASKS_FILE="${OUTROOT}/merge_tasks.txt"
GENERATION_PLAN_FILE="${OUTROOT}/generation_plan.json"
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/artifacts/logs/task_banks"}

BANK_SEEDS_STR=${BANK_SEEDS:-"11 17 23 29 31 37 41 43"}
BANK_TARGET_TOTAL=${BANK_TARGET_TOTAL:-52000}
LOW_SHARD_TARGET_COUNT=${LOW_SHARD_TARGET_COUNT:-1000}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-0}
MEMORY_PER_SHARD=${MEMORY_PER_SHARD:-1G}
TIME_LIMIT_PER_SHARD=${TIME_LIMIT_PER_SHARD:-0-00:10:00}
MERGE_MEMORY=${MERGE_MEMORY:-2G}
MERGE_TIME_LIMIT=${MERGE_TIME_LIMIT:-0-00:15:00}

FAMILY_ID=${FAMILY_ID:-trap_open_room}
LOW_TARGET_TOTAL=${LOW_TARGET_TOTAL:-51920}
TAIL_TARGET_TOTAL=${TAIL_TARGET_TOTAL:-80}
LOW_SEVERITY_MIN=${LOW_SEVERITY_MIN:-0.0}
LOW_SEVERITY_MAX=${LOW_SEVERITY_MAX:-0.5}
TAIL_SEVERITY=${TAIL_SEVERITY:-1.0}
# Slip-level mode: when SLIP_LEVELS (space-separated values) is non-empty, the
# bank is generated as one (low, tail) component pair per slip level, each
# record carrying slip_value, and the per-level counts below apply instead of
# LOW_TARGET_TOTAL/TAIL_TARGET_TOTAL. Empty SLIP_LEVELS reproduces the legacy
# single-component-pair bank exactly.
# Dynamics mode: DYNAMICS_SPECS (space-separated "slip:wind_dir:wind_strength"
# tuples, e.g. "0.10:2:0.30") supersedes SLIP_LEVELS; each tuple becomes one
# dynamics group (dynamics_id = its index) with its own (low, tail) pair.
SLIP_LEVELS_STR=${SLIP_LEVELS:-""}
DYNAMICS_SPECS_STR=${DYNAMICS_SPECS:-""}
if [[ -n "${DYNAMICS_SPECS_STR}" ]]; then
  SLIP_LEVELS_STR="${DYNAMICS_SPECS_STR}"
fi
LOW_TARGET_TOTAL_PER_LEVEL=${LOW_TARGET_TOTAL_PER_LEVEL:-2077}
TAIL_TARGET_TOTAL_PER_LEVEL=${TAIL_TARGET_TOTAL_PER_LEVEL:-3}
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
SLIP_LEVEL_ARRAY=()
if [[ -n "${SLIP_LEVELS_STR}" ]]; then
  read -r -a SLIP_LEVEL_ARRAY <<< "${SLIP_LEVELS_STR}"
fi
SLIP_LEVEL_COUNT=${#SLIP_LEVEL_ARRAY[@]}

if (( SLIP_LEVEL_COUNT == 0 )); then
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
else
  if (( LOW_TARGET_TOTAL_PER_LEVEL < 0 || TAIL_TARGET_TOTAL_PER_LEVEL < 0 )); then
    echo "Per-level component totals must be non-negative." >&2
    exit 1
  fi
  if (( SLIP_LEVEL_COUNT * (LOW_TARGET_TOTAL_PER_LEVEL + TAIL_TARGET_TOTAL_PER_LEVEL) != BANK_TARGET_TOTAL )); then
    echo "SLIP_LEVEL_COUNT * (LOW_TARGET_TOTAL_PER_LEVEL + TAIL_TARGET_TOTAL_PER_LEVEL) must equal BANK_TARGET_TOTAL" >&2
    exit 1
  fi
  LOW_TARGET_TOTAL=$(( SLIP_LEVEL_COUNT * LOW_TARGET_TOTAL_PER_LEVEL ))
  TAIL_TARGET_TOTAL=$(( SLIP_LEVEL_COUNT * TAIL_TARGET_TOTAL_PER_LEVEL ))
  LOW_FULL_SHARDS_PER_LEVEL=$(( LOW_TARGET_TOTAL_PER_LEVEL / LOW_SHARD_TARGET_COUNT ))
  LOW_REMAINDER_PER_LEVEL=$(( LOW_TARGET_TOTAL_PER_LEVEL % LOW_SHARD_TARGET_COUNT ))
  TAIL_SHARDS_PER_LEVEL=$(( TAIL_TARGET_TOTAL_PER_LEVEL > 0 ? 1 : 0 ))
  SHARDS_PER_BANK=$(( SLIP_LEVEL_COUNT * (LOW_FULL_SHARDS_PER_LEVEL + (LOW_REMAINDER_PER_LEVEL > 0 ? 1 : 0) + TAIL_SHARDS_PER_LEVEL) ))
fi

if (( SLIP_LEVEL_COUNT == 0 )); then
cat > "${GENERATION_PLAN_FILE}" <<EOF
{
  "record_format": "trap_layout",
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
else
  SLIP_LEVELS_FOR_PLAN="${SLIP_LEVEL_ARRAY[*]}" \
  PLAN_OUT="${GENERATION_PLAN_FILE}" \
  FAMILY_ID="${FAMILY_ID}" \
  ROWS="${ROWS}" COLS="${COLS}" HORIZON="${HORIZON}" GAMMA="${GAMMA}" \
  STEP_COST="${STEP_COST}" GOAL_REWARD="${GOAL_REWARD}" \
  CATASTROPHE_PENALTY="${CATASTROPHE_PENALTY}" \
  MIN_START_GOAL_DISTANCE="${MIN_START_GOAL_DISTANCE}" \
  MAX_GENERATION_ATTEMPTS_PER_TASK="${MAX_GENERATION_ATTEMPTS_PER_TASK}" \
  LOW_TARGET_TOTAL_PER_LEVEL="${LOW_TARGET_TOTAL_PER_LEVEL}" \
  TAIL_TARGET_TOTAL_PER_LEVEL="${TAIL_TARGET_TOTAL_PER_LEVEL}" \
  LOW_SEVERITY_MIN="${LOW_SEVERITY_MIN}" LOW_SEVERITY_MAX="${LOW_SEVERITY_MAX}" \
  TAIL_SEVERITY="${TAIL_SEVERITY}" \
  "${PYTHON_BIN}" - <<'PYEOF'
import json
import os

env = os.environ
tokens = env["SLIP_LEVELS_FOR_PLAN"].split()
specs = []
for idx, token in enumerate(tokens):
    parts = token.split(":")
    if len(parts) == 1:
        specs.append({"slip_value": float(parts[0])})
    elif len(parts) == 3:
        specs.append(
            {
                "slip_value": float(parts[0]),
                "wind_dir": int(parts[1]),
                "wind_strength": float(parts[2]),
                "dynamics_id": idx,
            }
        )
    else:
        raise ValueError(f"Bad dynamics spec token: {token!r}")

components = []
for idx, spec in enumerate(specs):
    if "dynamics_id" in spec:
        low_name, tail_name = f"low_dyn{idx:02d}", f"tail_dyn{idx:02d}"
    else:
        label = f"{spec['slip_value']:.3f}"
        low_name, tail_name = f"low_slip{label}", f"tail_slip{label}"
    components.append(
        {
            "name": low_name,
            "target_total": int(env["LOW_TARGET_TOTAL_PER_LEVEL"]),
            "severity_min": float(env["LOW_SEVERITY_MIN"]),
            "severity_max": float(env["LOW_SEVERITY_MAX"]),
            **spec,
        }
    )
    components.append(
        {
            "name": tail_name,
            "target_total": int(env["TAIL_TARGET_TOTAL_PER_LEVEL"]),
            "severity_min": float(env["TAIL_SEVERITY"]),
            "severity_max": float(env["TAIL_SEVERITY"]),
            **spec,
        }
    )
plan = {
    "record_format": "trap_layout",
    "family_id": env["FAMILY_ID"],
    "generator_config": {
        "rows": int(env["ROWS"]),
        "cols": int(env["COLS"]),
        "horizon": int(env["HORIZON"]),
        "gamma": float(env["GAMMA"]),
        "step_cost": float(env["STEP_COST"]),
        "goal_reward": float(env["GOAL_REWARD"]),
        "catastrophe_penalty": float(env["CATASTROPHE_PENALTY"]),
        "min_start_goal_distance": int(env["MIN_START_GOAL_DISTANCE"]),
        "max_generation_attempts_per_task": int(env["MAX_GENERATION_ATTEMPTS_PER_TASK"]),
    },
    "slip_levels": [spec["slip_value"] for spec in specs],
    "dynamics_specs": specs,
    "components": components,
}
with open(env["PLAN_OUT"], "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)
    f.write("\n")
PYEOF
fi

: > "${SHARD_TASKS_FILE}"
: > "${MERGE_TASKS_FILE}"

# Appends one shard-generation command to SHARD_TASKS_FILE. The spec argument
# may be empty (legacy mode: --slip-value omitted so task lines stay
# byte-identical to the pre-slip script), a plain slip value, or a
# "slip:wind_dir:wind_strength" tuple (dynamics mode; needs dynamics_id).
emit_shard_cmd() {
  local shard_out=$1 count=$2 shard_seed=$3 split_name=$4 sev_min=$5 sev_max=$6 spec=${7:-} dynamics_id=${8:-}
  local cmd=(
    "${PYTHON_BIN}" -m forecast_experiment.generate_rlquant_trap_layout_bank_shard
    --out "${shard_out}"
    --family-id "${FAMILY_ID}"
    --count "${count}"
    --seed "${shard_seed}"
    --split-name "${split_name}"
    --severity-min "${sev_min}"
    --severity-max "${sev_max}"
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
  if [[ -n "${spec}" ]]; then
    if [[ "${spec}" == *:* ]]; then
      local spec_slip spec_wdir spec_wstr
      IFS=':' read -r spec_slip spec_wdir spec_wstr <<< "${spec}"
      cmd+=(--slip-value "${spec_slip}")
      cmd+=(--wind-dir "${spec_wdir}")
      cmd+=(--wind-strength "${spec_wstr}")
      cmd+=(--dynamics-id "${dynamics_id}")
    else
      cmd+=(--slip-value "${spec}")
    fi
  fi
  printf '%q ' "${cmd[@]}" >> "${SHARD_TASKS_FILE}"
  printf '\n' >> "${SHARD_TASKS_FILE}"
}

global_task_index=0
for bank_seed in "${BANK_SEED_ARRAY[@]}"; do
  bank_dir="${OUTROOT}/bank_seed_${bank_seed}"
  shard_dir="${bank_dir}/shards"
  mkdir -p "${shard_dir}"
  shard_idx=0

  if (( SLIP_LEVEL_COUNT == 0 )); then
    for (( local_idx=0; local_idx<LOW_FULL_SHARDS; local_idx++ )); do
      shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
      shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
      emit_shard_cmd "${shard_out}" "${LOW_SHARD_TARGET_COUNT}" "${shard_seed}" \
        "layout_bank_${bank_seed}_shard_${shard_idx}" "${LOW_SEVERITY_MIN}" "${LOW_SEVERITY_MAX}"
      global_task_index=$((global_task_index + 1))
      shard_idx=$((shard_idx + 1))
    done

    if (( LOW_REMAINDER > 0 )); then
      shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
      shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
      emit_shard_cmd "${shard_out}" "${LOW_REMAINDER}" "${shard_seed}" \
        "layout_bank_${bank_seed}_shard_${shard_idx}" "${LOW_SEVERITY_MIN}" "${LOW_SEVERITY_MAX}"
      global_task_index=$((global_task_index + 1))
      shard_idx=$((shard_idx + 1))
    fi

    if (( TAIL_TARGET_TOTAL > 0 )); then
      shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
      shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
      emit_shard_cmd "${shard_out}" "${TAIL_TARGET_TOTAL}" "${shard_seed}" \
        "layout_bank_${bank_seed}_shard_${shard_idx}" "${TAIL_SEVERITY}" "${TAIL_SEVERITY}"
      global_task_index=$((global_task_index + 1))
      shard_idx=$((shard_idx + 1))
    fi
  else
    level_idx=0
    for slip_level in "${SLIP_LEVEL_ARRAY[@]}"; do
      remaining_low=${LOW_TARGET_TOTAL_PER_LEVEL}
      while (( remaining_low > 0 )); do
        shard_count=$(( remaining_low < LOW_SHARD_TARGET_COUNT ? remaining_low : LOW_SHARD_TARGET_COUNT ))
        shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
        shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
        emit_shard_cmd "${shard_out}" "${shard_count}" "${shard_seed}" \
          "layout_bank_${bank_seed}_shard_${shard_idx}" "${LOW_SEVERITY_MIN}" "${LOW_SEVERITY_MAX}" \
          "${slip_level}" "${level_idx}"
        global_task_index=$((global_task_index + 1))
        shard_idx=$((shard_idx + 1))
        remaining_low=$(( remaining_low - shard_count ))
      done

      if (( TAIL_TARGET_TOTAL_PER_LEVEL > 0 )); then
        shard_seed=$(( bank_seed * 100000 + shard_idx * 1009 + 17 ))
        shard_out="${shard_dir}/shard_$(printf '%03d' "${shard_idx}").jsonl"
        emit_shard_cmd "${shard_out}" "${TAIL_TARGET_TOTAL_PER_LEVEL}" "${shard_seed}" \
          "layout_bank_${bank_seed}_shard_${shard_idx}" "${TAIL_SEVERITY}" "${TAIL_SEVERITY}" \
          "${slip_level}" "${level_idx}"
        global_task_index=$((global_task_index + 1))
        shard_idx=$((shard_idx + 1))
      fi
      level_idx=$((level_idx + 1))
    done
  fi

  merged_out="${bank_dir}/bank_layouts.jsonl"
  merge_cmd=(
    "${PYTHON_BIN}" -m forecast_experiment.merge_rlquant_trap_layout_bank_shards
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

SLIP_LEVELS_JSON="[]"
if (( SLIP_LEVEL_COUNT > 0 )); then
  if [[ "${SLIP_LEVELS_STR}" == *:* ]]; then
    SLIP_LEVELS_JSON="[\"$(printf '%s' "${SLIP_LEVEL_ARRAY[*]}" | sed 's/ /", "/g')\"]"
  else
    SLIP_LEVELS_JSON="[$(printf '%s' "${SLIP_LEVEL_ARRAY[*]}" | sed 's/ /, /g')]"
  fi
fi

cat > "${OUTROOT}/submission.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "outroot": "${OUTROOT}",
  "submit_script": "${SUBMIT_SCRIPT_PATH}",
  "shard_tasks_file": "${SHARD_TASKS_FILE}",
  "merge_tasks_file": "${MERGE_TASKS_FILE}",
  "generation_plan_file": "${GENERATION_PLAN_FILE}",
  "record_format": "trap_layout",
  "shard_array_job_id": "${shard_jobid}",
  "merge_array_job_id": "${merge_jobid}",
  "bank_seeds": [$(printf '%s' "${BANK_SEED_ARRAY[*]}" | sed 's/ /, /g')],
  "bank_target_total": ${BANK_TARGET_TOTAL},
  "low_target_total": ${LOW_TARGET_TOTAL},
  "tail_target_total": ${TAIL_TARGET_TOTAL},
  "slip_levels": ${SLIP_LEVELS_JSON},
  "low_target_total_per_level": ${LOW_TARGET_TOTAL_PER_LEVEL},
  "tail_target_total_per_level": ${TAIL_TARGET_TOTAL_PER_LEVEL},
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
  "compatibility_note": "Experiments using this bank should not set horizon greater than the bank generation horizon, because realized layouts were generated and validated at generation time.",
  "qos": "${SBATCH_QOS:-}"
}
EOF

cat > "${OUTROOT}/EXPERIMENT.md" <<EOF
# RLQuant Trap Layout Bank Generation

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
- Slip levels: \`${SLIP_LEVELS_STR:-none (legacy single-dynamics bank)}\`
- Low-mode target tasks per slip level: \`${LOW_TARGET_TOTAL_PER_LEVEL}\` (slip mode only)
- Tail target tasks per slip level: \`${TAIL_TARGET_TOTAL_PER_LEVEL}\` (slip mode only)
- Low-mode target tasks per full shard: \`${LOW_SHARD_TARGET_COUNT}\`
- Shards per bank: \`${SHARDS_PER_BANK}\`
- Family: \`${FAMILY_ID}\`
- Record format: \`trap_layout\`

## IID Severity Mixture
- Low mode: severity sampled from \`${LOW_SEVERITY_MIN} .. ${LOW_SEVERITY_MAX}\`
- Tail mode: severity fixed at \`${TAIL_SEVERITY}\`
- In slip mode, each slip level gets its own independent low+tail component pair
  with uniform per-cell slip probability equal to the level.

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
- Realized layouts were generated and validated at generation time, so a larger experiment horizon changes the task distribution.

## Files
- Shard tasks file: \`${SHARD_TASKS_FILE}\`
- Merge tasks file: \`${MERGE_TASKS_FILE}\`
- Generation plan: \`${GENERATION_PLAN_FILE}\`
EOF

echo "Shard Array JobID:   ${shard_jobid}"
echo "Merge Array JobID:   ${merge_jobid}"
echo "Experiment note: ${OUTROOT}/EXPERIMENT.md"
