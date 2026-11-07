#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # jobs are submitted from the repo root
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Generic array runner: executes the command at line (SLURM_ARRAY_TASK_ID+1)
# from the tasks file specified by $TASKS_FILE.

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-online}

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set" >&2
  exit 1
fi

if [ -z "${TASKS_FILE:-}" ]; then
  echo "TASKS_FILE env var is not set" >&2
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID}
LINE_NUM=$((IDX + 1))
CMD=$(sed -n "${LINE_NUM}p" "${TASKS_FILE}")
if [ -z "${CMD}" ]; then
  echo "No command found for index ${IDX} in ${TASKS_FILE}" >&2
  exit 2
fi
echo "[array ${SLURM_ARRAY_JOB_ID:-unknown}-${SLURM_ARRAY_TASK_ID}] Executing: ${CMD}"
eval "${CMD}"

