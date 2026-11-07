#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT
SUBMIT_SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
TASK_BANK_ROOT="${TASK_BANK_ROOT:-${REPO_ROOT}/artifacts/task_banks}"
# Hardware: CPU only, unless USE_GPU=1 (the default when GCG_ENABLED=1) adds one
# GPU per bank seed; the paper's GPU builds used compute capability 8.0+ and at
# least 16 GB (set SBATCH_CONSTRAINT or SBATCH_PARTITION to get such GPUs).

EXPERIMENT_TAG=${EXPERIMENT_TAG:-lquant_prompt_bank_v1_$(date +%Y%m%d)}
OUTROOT=${OUTROOT:-"${TASK_BANK_ROOT}/${EXPERIMENT_TAG}"}
TASKS_FILE="${OUTROOT}/tasks.txt"
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/artifacts/logs/task_banks"}

MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
BANK_SEEDS_STR=${BANK_SEEDS:-"11 17 23 29 31 37 41 43"}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-0}
PASSWORD_SLOTS=${PASSWORD_SLOTS:-5}
PASSWORD_POLICY=${PASSWORD_POLICY:-non_disclosure}
CHEAP_PROMPTS_PER_FAMILY=${CHEAP_PROMPTS_PER_FAMILY:-64}
CHEAP_PROMPTS_PER_FAMILY_MAP_STR=${CHEAP_PROMPTS_PER_FAMILY_MAP_STR:-}
TRAIN_FAMILIES_STR=${TRAIN_FAMILIES_STR:-}
TEST_FAMILIES_STR=${TEST_FAMILIES_STR:-}
DISABLE_THINKING=${DISABLE_THINKING:-1}

GCG_ENABLED=${GCG_ENABLED:-0}
GCG_STEPS_SPECTRUM_STR=${GCG_STEPS_SPECTRUM_STR:-"5 10 20 40 80"}
GCG_PROMPTS_PER_LEVEL_PER_TARGET=${GCG_PROMPTS_PER_LEVEL_PER_TARGET:-16}
GCG_PROMPT_LEN=${GCG_PROMPT_LEN:-24}
GCG_TOP_K=${GCG_TOP_K:-8}
GCG_RANDOM_RESTARTS=${GCG_RANDOM_RESTARTS:-2}
GCG_TRAIN_MAX_STEPS=${GCG_TRAIN_MAX_STEPS:-20}
GCG_DECAY_ALPHA=${GCG_DECAY_ALPHA:-2.0}

USE_GPU=${USE_GPU:-${GCG_ENABLED}}
DEVICE=${DEVICE:-$([[ "${USE_GPU}" == "1" ]] && echo cuda || echo cpu)}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEMORY=${MEMORY:-$([[ "${USE_GPU}" == "1" ]] && echo 32G || echo 8G)}
TIME_LIMIT=${TIME_LIMIT:-$([[ "${USE_GPU}" == "1" ]] && echo 0-02:00:00 || echo 0-00:20:00)}

mkdir -p "${OUTROOT}" "${LOG_DIR}"
read -r -a BANK_SEED_ARRAY <<< "${BANK_SEEDS_STR}"
read -r -a TRAIN_FAMILIES_ARRAY <<< "${TRAIN_FAMILIES_STR}"
read -r -a TEST_FAMILIES_ARRAY <<< "${TEST_FAMILIES_STR}"
read -r -a CHEAP_PROMPTS_PER_FAMILY_MAP_ARRAY <<< "${CHEAP_PROMPTS_PER_FAMILY_MAP_STR}"
read -r -a GCG_STEPS_SPECTRUM_ARRAY <<< "${GCG_STEPS_SPECTRUM_STR}"

: > "${TASKS_FILE}"
for bank_seed in "${BANK_SEED_ARRAY[@]}"; do
  outdir="${OUTROOT}/bank_seed_${bank_seed}"
  cmd=(
    "${PYTHON_BIN}" -m forecast_experiment.generate_lquant_prompt_bank
    --model "${MODEL}"
    --device "${DEVICE}"
    --outdir "${outdir}"
    --bank-seed "${bank_seed}"
    --password-slots "${PASSWORD_SLOTS}"
    --password-policy "${PASSWORD_POLICY}"
    --cheap-prompts-per-family-per-target "${CHEAP_PROMPTS_PER_FAMILY}"
  )
  if [[ ${#CHEAP_PROMPTS_PER_FAMILY_MAP_ARRAY[@]} -gt 0 ]]; then
    cmd+=(--cheap-prompts-per-family-map "${CHEAP_PROMPTS_PER_FAMILY_MAP_ARRAY[@]}")
  fi
  if [[ ${#TRAIN_FAMILIES_ARRAY[@]} -gt 0 ]]; then
    cmd+=(--train-families "${TRAIN_FAMILIES_ARRAY[@]}")
  fi
  if [[ ${#TEST_FAMILIES_ARRAY[@]} -gt 0 ]]; then
    cmd+=(--test-families "${TEST_FAMILIES_ARRAY[@]}")
  fi
  if [[ "${DISABLE_THINKING}" == "1" ]]; then
    cmd+=(--disable-thinking)
  fi
  if [[ "${GCG_ENABLED}" == "1" ]]; then
    cmd+=(
      --gcg
      --gcg-steps-spectrum "${GCG_STEPS_SPECTRUM_ARRAY[@]}"
      --gcg-prompts-per-level-per-target "${GCG_PROMPTS_PER_LEVEL_PER_TARGET}"
      --gcg-prompt-len "${GCG_PROMPT_LEN}"
      --gcg-top-k "${GCG_TOP_K}"
      --gcg-random-restarts "${GCG_RANDOM_RESTARTS}"
      --gcg-train-max-steps "${GCG_TRAIN_MAX_STEPS}"
      --gcg-decay-alpha "${GCG_DECAY_ALPHA}"
    )
  fi
  printf '%q ' "${cmd[@]}" >> "${TASKS_FILE}"
  printf '\n' >> "${TASKS_FILE}"
done

array_max=$(( ${#BANK_SEED_ARRAY[@]} - 1 ))
if (( array_max < 0 )); then
  echo "No bank seeds configured." >&2
  exit 1
fi

sbatch_args=(
  -c "${CPUS_PER_TASK}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
  --output="${LOG_DIR}/%A_%a.out"
  --error="${LOG_DIR}/%A_%a.err"
  --export="ALL,TASKS_FILE=${TASKS_FILE},PYTHONPATH=${PYTHONPATH},TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
)
if [[ "${USE_GPU}" == "1" ]]; then
  sbatch_args+=(--gres=gpu:1)
fi
if (( ARRAY_CONCURRENCY > 0 )); then
  sbatch_args+=(--array="0-${array_max}%${ARRAY_CONCURRENCY}")
else
  sbatch_args+=(--array="0-${array_max}")
fi

submit_out=$(sbatch "${sbatch_args[@]}" "${REPO_ROOT}/scripts/slurm/array_runner.sh")
echo "${submit_out}"
array_jobid=$(awk '{print $4}' <<< "${submit_out}")
if [[ -z "${array_jobid}" ]]; then
  echo "Failed to parse array job id." >&2
  exit 1
fi

cat > "${OUTROOT}/submission.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "outroot": "${OUTROOT}",
  "submit_script": "${SUBMIT_SCRIPT_PATH}",
  "tasks_file": "${TASKS_FILE}",
  "array_job_id": "${array_jobid}",
  "bank_seeds": [$(printf '%s' "${BANK_SEED_ARRAY[*]}" | sed 's/ /, /g')],
  "model": "${MODEL}",
  "device": "${DEVICE}",
  "password_slots": ${PASSWORD_SLOTS},
  "password_policy": "${PASSWORD_POLICY}",
  "cheap_prompts_per_family_per_target": ${CHEAP_PROMPTS_PER_FAMILY},
  "cheap_prompts_per_family_map": "${CHEAP_PROMPTS_PER_FAMILY_MAP_STR}",
  "train_families": "${TRAIN_FAMILIES_STR}",
  "test_families": "${TEST_FAMILIES_STR}",
  "disable_thinking": ${DISABLE_THINKING},
  "gcg_enabled": ${GCG_ENABLED},
  "gcg_steps_spectrum": "${GCG_STEPS_SPECTRUM_STR}",
  "gcg_prompts_per_level_per_target": ${GCG_PROMPTS_PER_LEVEL_PER_TARGET},
  "gcg_prompt_len": ${GCG_PROMPT_LEN},
  "gcg_top_k": ${GCG_TOP_K},
  "gcg_random_restarts": ${GCG_RANDOM_RESTARTS},
  "gcg_train_max_steps": ${GCG_TRAIN_MAX_STEPS},
  "gcg_decay_alpha": ${GCG_DECAY_ALPHA},
  "array_concurrency": ${ARRAY_CONCURRENCY},
  "cpus_per_task": ${CPUS_PER_TASK},
  "memory": "${MEMORY}",
  "time_limit": "${TIME_LIMIT}",
  "account": "${SBATCH_ACCOUNT:-}",
  "partition": "${SBATCH_PARTITION:-}",
  "constraint": "${SBATCH_CONSTRAINT:-}",
  "qos": "${SBATCH_QOS:-}"
}
EOF

cat > "${OUTROOT}/EXPERIMENT.md" <<EOF
# LQuant Prompt Bank Generation

## Submission
- Array job id: \`${array_jobid}\`
- Output root: \`${OUTROOT}\`
- Submission script: \`${SUBMIT_SCRIPT_PATH}\`
- Submission metadata: \`${OUTROOT}/submission.json\`
- Tasks file: \`${TASKS_FILE}\`
- Bank seeds: \`${BANK_SEEDS_STR}\`

## Prompt Pool
- Model: \`${MODEL}\`
- Device: \`${DEVICE}\`
- Password slots: \`${PASSWORD_SLOTS}\`
- Password policy: \`${PASSWORD_POLICY}\`
- Cheap prompts per family per target: \`${CHEAP_PROMPTS_PER_FAMILY}\`
- Cheap prompt overrides: \`${CHEAP_PROMPTS_PER_FAMILY_MAP_STR}\`
- Train families override: \`${TRAIN_FAMILIES_STR}\`
- Test families override: \`${TEST_FAMILIES_STR}\`
- Disable thinking: \`${DISABLE_THINKING}\`

## GCG
- Enabled: \`${GCG_ENABLED}\`
- Steps spectrum: \`${GCG_STEPS_SPECTRUM_STR}\`
- Prompts per level per target: \`${GCG_PROMPTS_PER_LEVEL_PER_TARGET}\`
- Prompt length: \`${GCG_PROMPT_LEN}\`
- Top-k: \`${GCG_TOP_K}\`
- Random restarts: \`${GCG_RANDOM_RESTARTS}\`
- Train max steps: \`${GCG_TRAIN_MAX_STEPS}\`
- Decay alpha: \`${GCG_DECAY_ALPHA}\`

## Runtime
- Array concurrency: \`${ARRAY_CONCURRENCY}\`
- CPUs per task: \`${CPUS_PER_TASK}\`
- Memory: \`${MEMORY}\`
- Time limit: \`${TIME_LIMIT}\`
- Account: \`${SBATCH_ACCOUNT:-<default>}\`
- Partition: \`${SBATCH_PARTITION:-<default>}\`
- Constraint: \`${SBATCH_CONSTRAINT:-<none>}\`
- QoS: \`${SBATCH_QOS:-<default>}\`
EOF

