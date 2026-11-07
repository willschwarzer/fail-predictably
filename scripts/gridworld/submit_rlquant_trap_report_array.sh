#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export REPO_ROOT
SUBMIT_SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/artifacts/results}"
# Hardware: CPU only; each seed is one array task with THREADS_PER_TASK cores and MEMORY.

EXPERIMENT_TAG=${EXPERIMENT_TAG:-rlquant_trap_report_v6_multitask_cached_96_1920_pretrain500_20260403}
OUTROOT=${OUTROOT:-"${RESULTS_ROOT}/${EXPERIMENT_TAG}"}
TASKS_FILE="${OUTROOT}/tasks.txt"
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/artifacts/logs/rlquant"}

SEEDS_STR=${SEEDS:-"11 17 23 29 31 37 41 43"}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-4}
THREADS_PER_TASK=${THREADS_PER_TASK:-8}
MEMORY=${MEMORY:-24G}
TIME_LIMIT=${TIME_LIMIT:-0-02:30:00}

HORIZON=${HORIZON:-10}
MIN_START_GOAL_DISTANCE=${MIN_START_GOAL_DISTANCE:-5}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-500}
META_STEPS=${META_STEPS:-100}
SAVE_EVERY=${SAVE_EVERY:-50}
META_MANIFEST_COUNT=${META_MANIFEST_COUNT:-20}
HELDOUT_MANIFEST_COUNT=${HELDOUT_MANIFEST_COUNT:-5}
META_MANIFESTS_PER_STEP=${META_MANIFESTS_PER_STEP:-4}
META_DEPLOY_CACHE_TOP_B=${META_DEPLOY_CACHE_TOP_B:-200}
META_DEPLOY_CACHE_REFRESH_INTERVAL=${META_DEPLOY_CACHE_REFRESH_INTERVAL:-10}
PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE:-16}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
LOG_EVERY=${LOG_EVERY:-50}
TASK_HIDDEN_DIM=${TASK_HIDDEN_DIM:-64}
OLS_TOP_K=${OLS_TOP_K:-10}

TRAP_PRETRAIN_TASKS=${TRAP_PRETRAIN_TASKS:-192}
TRAP_FIT_TASKS=${TRAP_FIT_TASKS:-96}
TRAP_DEPLOY_EXTRA_TASKS=${TRAP_DEPLOY_EXTRA_TASKS:-1920}
TRAP_HELDOUT_FIT_TASKS=${TRAP_HELDOUT_FIT_TASKS:-96}
TRAP_HELDOUT_DEPLOY_EXTRA_TASKS=${TRAP_HELDOUT_DEPLOY_EXTRA_TASKS:-1920}
META_FIT_SUBSAMPLE_MIN=${META_FIT_SUBSAMPLE_MIN:-${TRAP_FIT_TASKS}}
META_FIT_SUBSAMPLE_MAX=${META_FIT_SUBSAMPLE_MAX:-${TRAP_FIT_TASKS}}

SUPPORT_AUDIT_NORMAL_SAMPLES=${SUPPORT_AUDIT_NORMAL_SAMPLES:-2048}
SUPPORT_AUDIT_CHALLENGE_SAMPLES=${SUPPORT_AUDIT_CHALLENGE_SAMPLES:-256}
SUPPORT_AUDIT_TOTAL_MIN_UNIQUE=${SUPPORT_AUDIT_TOTAL_MIN_UNIQUE:-1024}
SUPPORT_AUDIT_NORMAL_TOTAL_MIN_UNIQUE=${SUPPORT_AUDIT_NORMAL_TOTAL_MIN_UNIQUE:-1024}
SUPPORT_AUDIT_NORMAL_MIN_UNIQUE_PER_FAMILY=${SUPPORT_AUDIT_NORMAL_MIN_UNIQUE_PER_FAMILY:-1024}
SUPPORT_AUDIT_NORMAL_MAX_COLLISION_RATE=${SUPPORT_AUDIT_NORMAL_MAX_COLLISION_RATE:-0.25}
SUPPORT_AUDIT_NORMAL_MIN_ACCEPTANCE=${SUPPORT_AUDIT_NORMAL_MIN_ACCEPTANCE:-0.5}

LR=${LR:-3e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
META_LOSS_MODE=${META_LOSS_MODE:-ols_primal_extrapolated}
META_AUX_LOSS_MODE=${META_AUX_LOSS_MODE:-pretrain_return}
META_AUX_LOSS_WEIGHT=${META_AUX_LOSS_WEIGHT:-1.0}
META_AUX_TASK_SOURCE=${META_AUX_TASK_SOURCE:-pretrain_train}
PRETRAIN_TASK_SOURCE=${PRETRAIN_TASK_SOURCE:-pretrain_train}
GRAD_MODE=${GRAD_MODE:-both}
IMPROVING_ONLY_GRAD=${IMPROVING_ONLY_GRAD:-0}
IOG_SCOPE=${IOG_SCOPE:-both}
CALIBRATION_AWARE=${CALIBRATION_AWARE:-0}
CALIBRATION_MODE=${CALIBRATION_MODE:-shift_scale}
CALIBRATION_MIN_POINTS=${CALIBRATION_MIN_POINTS:-5}
CALIBRATION_RIDGE_ALPHA=${CALIBRATION_RIDGE_ALPHA:-0.05}
CALIBRATION_EMA=${CALIBRATION_EMA:-0.0}
GRAD_ESTIMATOR=${GRAD_ESTIMATOR:-exact}
PRETRAIN_GRAD_ESTIMATOR=${PRETRAIN_GRAD_ESTIMATOR:-inherit}
REINFORCE_ROLLOUTS_PER_TASK=${REINFORCE_ROLLOUTS_PER_TASK:-8}
REINFORCE_BASELINE=${REINFORCE_BASELINE:-rollout_mean}
REINFORCE_EMA_DECAY=${REINFORCE_EMA_DECAY:-0.9}
REINFORCE_META_ESTIMATOR=${REINFORCE_META_ESTIMATOR:-plugin}
REINFORCE_SPLIT_SAMPLES=${REINFORCE_SPLIT_SAMPLES:-0}
REINFORCE_SCORE=${REINFORCE_SCORE:-regret}
BOOTSTRAP_PERMUTATIONS=${BOOTSTRAP_PERMUTATIONS:-1}
UNION_CACHE_TOP_C=${UNION_CACHE_TOP_C:-0}
UNION_CACHE_REFRESH_INTERVAL=${UNION_CACHE_REFRESH_INTERVAL:-5}
SPLIT_MODE=${SPLIT_MODE:-disjoint}
MANIFEST_GROUPING=${MANIFEST_GROUPING:-iid}
TASK_VECTOR_SLIP=${TASK_VECTOR_SLIP:-0}
BLIND_DYNAMICS=${BLIND_DYNAMICS:-0}
PLOTTING_POSITION=${PLOTTING_POSITION:-weibull}
EXTRAPOLATED_WEIGHTING=${EXTRAPOLATED_WEIGHTING:-deploy_log_uniform}
DETERMINISTIC_DEBUG=${DETERMINISTIC_DEBUG:-0}
# TASK_BANK_ROOT here is ONE bank's directory (the one holding bank_seed_<seed>/),
# passed as --task-bank-root; the run_*.sh launchers set it. Empty means tasks
# are sampled online.
TASK_BANK_ROOT=${TASK_BANK_ROOT:-}
TASK_BANK_SEED=${TASK_BANK_SEED:--1}
BANK_MATERIALIZATION=${BANK_MATERIALIZATION:-lazy}
PRETRAINED_EXACT_EVAL_PHASE=${PRETRAINED_EXACT_EVAL_PHASE:-post}
PRETRAINED_QUICKCHECK_TASK_COUNT=${PRETRAINED_QUICKCHECK_TASK_COUNT:-4000}
# REUSE_PRETRAIN_ROOT: the run root of an earlier fleet whose seed_<seed>/ dirs
# hold pretrained_checkpoint.pt and pretrained_results.json. Each seed then
# starts from that pretrained policy and skips pretraining (--skip-pretrain),
# so arms that share a pretrained policy differ only in the later training. The
# probe below turns reuse off if the earlier fleet used another bank or bank seed.
REUSE_PRETRAIN_ROOT=${REUSE_PRETRAIN_ROOT:-}
# Like REUSE_PRETRAIN_ROOT (per-seed pretrained_checkpoint.pt) but WITHOUT
# --skip-pretrain: the pretrain phase still runs, initialized from the prior
# checkpoint. Used by SFT-REINFORCE baselines (canonical pretrained init +
# sampled SFT steps on the swapped pool).
INIT_PRETRAIN_ROOT=${INIT_PRETRAIN_ROOT:-}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
# AUTO_RESUME=1: seeds whose outdir already holds resume_latest.pt continue
# from it (crash/timeout resubmits of the same tag; completed seeds no-op).
# RESUME_ROOT: per-seed resume_latest.pt from ANOTHER fleet's run dirs -- for
# extension runs, pair with a fresh EXPERIMENT_TAG and a higher META_STEPS so
# the original fleet's artifacts are never overwritten.
AUTO_RESUME=${AUTO_RESUME:-0}
RESUME_ROOT=${RESUME_ROOT:-}

mkdir -p "${OUTROOT}" "${LOG_DIR}"
read -r -a SEED_ARRAY <<< "${SEEDS_STR}"

REUSE_PRETRAIN_ROOT_EFFECTIVE="${REUSE_PRETRAIN_ROOT}"
REUSE_PRETRAIN_DISABLED_REASON=""
if [[ -n "${REUSE_PRETRAIN_ROOT}" && ${#SEED_ARRAY[@]} -gt 0 ]]; then
  probe_seed="${SEED_ARRAY[0]}"
  probe_cfg="${REUSE_PRETRAIN_ROOT}/seed_${probe_seed}/config.json"
  if [[ ! -f "${probe_cfg}" ]]; then
    REUSE_PRETRAIN_ROOT_EFFECTIVE=""
    REUSE_PRETRAIN_DISABLED_REASON="missing prior config for reuse probe at ${probe_cfg}"
  else
    reuse_check="$("${PYTHON_BIN}" - "${probe_cfg}" "${TASK_BANK_ROOT}" "${TASK_BANK_SEED}" "${probe_seed}" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
current_task_bank_root = sys.argv[2]
current_task_bank_seed = int(sys.argv[3])
probe_seed = int(sys.argv[4])
prior_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

prior_task_bank_root = str(prior_cfg.get("task_bank_root", ""))
prior_task_bank_seed = int(prior_cfg.get("task_bank_seed", -1))
prior_effective_task_bank_seed = int(prior_cfg.get("seed", 0)) if prior_task_bank_seed < 0 else prior_task_bank_seed
current_effective_task_bank_seed = probe_seed if current_task_bank_seed < 0 else current_task_bank_seed

issues = []
if prior_task_bank_root != current_task_bank_root:
    issues.append(
        "task_bank_root mismatch: "
        f"prior={prior_task_bank_root or '<online sampling>'} "
        f"current={current_task_bank_root or '<online sampling>'}"
    )
if prior_effective_task_bank_seed != current_effective_task_bank_seed:
    issues.append(
        "effective_task_bank_seed mismatch: "
        f"prior={prior_effective_task_bank_seed} current={current_effective_task_bank_seed}"
    )

print(" | ".join(issues))
PY
)"
    if [[ -n "${reuse_check}" ]]; then
      REUSE_PRETRAIN_ROOT_EFFECTIVE=""
      REUSE_PRETRAIN_DISABLED_REASON="${reuse_check}"
    fi
  fi
fi
if [[ -n "${REUSE_PRETRAIN_DISABLED_REASON}" ]]; then
  echo "Reuse pretraining disabled: ${REUSE_PRETRAIN_DISABLED_REASON}" >&2
fi

: > "${TASKS_FILE}"
for seed in "${SEED_ARRAY[@]}"; do
  outdir="${OUTROOT}/seed_${seed}"
  cmd=(
    "${PYTHON_BIN}" -m forecast_experiment.train_rlquant
    --device cpu
    --family-suite trap_open_room_only
    --seed "${seed}"
    --horizon "${HORIZON}"
    --min-start-goal-distance "${MIN_START_GOAL_DISTANCE}"
    --pretrain-steps "${PRETRAIN_STEPS}"
    --meta-steps "${META_STEPS}"
    --save-every "${SAVE_EVERY}"
    --meta-manifest-count "${META_MANIFEST_COUNT}"
    --heldout-manifest-count "${HELDOUT_MANIFEST_COUNT}"
    --meta-manifests-per-step "${META_MANIFESTS_PER_STEP}"
    --meta-deploy-cache-top-b "${META_DEPLOY_CACHE_TOP_B}"
    --meta-deploy-cache-refresh-interval "${META_DEPLOY_CACHE_REFRESH_INTERVAL}"
    --pretrain-batch-size "${PRETRAIN_BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --grad-estimator "${GRAD_ESTIMATOR}"
    --task-hidden-dim "${TASK_HIDDEN_DIM}"
    --ols-top-k "${OLS_TOP_K}"
    --plotting-position "${PLOTTING_POSITION}"
    --extrapolated-weighting "${EXTRAPOLATED_WEIGHTING}"
    --meta-fit-subsample-min "${META_FIT_SUBSAMPLE_MIN}"
    --meta-fit-subsample-max "${META_FIT_SUBSAMPLE_MAX}"
    --meta-loss-mode "${META_LOSS_MODE}"
    --meta-aux-loss-mode "${META_AUX_LOSS_MODE}"
    --meta-aux-loss-weight "${META_AUX_LOSS_WEIGHT}"
    --meta-aux-task-source "${META_AUX_TASK_SOURCE}"
    --pretrain-task-source "${PRETRAIN_TASK_SOURCE}"
    --grad-mode "${GRAD_MODE}"
    --split-mode "${SPLIT_MODE}"
    --lr "${LR}"
    --grad-clip "${GRAD_CLIP}"
    --trap-pretrain-per-family "${TRAP_PRETRAIN_TASKS}"
    --trap-meta-fit-per-family "${TRAP_FIT_TASKS}"
    --trap-meta-deploy-extra-per-family "${TRAP_DEPLOY_EXTRA_TASKS}"
    --trap-heldout-fit-per-family "${TRAP_HELDOUT_FIT_TASKS}"
    --trap-heldout-deploy-extra-per-family "${TRAP_HELDOUT_DEPLOY_EXTRA_TASKS}"
    --support-audit-normal-samples "${SUPPORT_AUDIT_NORMAL_SAMPLES}"
    --support-audit-challenge-samples "${SUPPORT_AUDIT_CHALLENGE_SAMPLES}"
    --support-audit-total-min-unique "${SUPPORT_AUDIT_TOTAL_MIN_UNIQUE}"
    --support-audit-normal-total-min-unique "${SUPPORT_AUDIT_NORMAL_TOTAL_MIN_UNIQUE}"
    --support-audit-normal-min-unique-per-family "${SUPPORT_AUDIT_NORMAL_MIN_UNIQUE_PER_FAMILY}"
    --support-audit-rare-min-unique 1
    --support-audit-normal-max-collision-rate "${SUPPORT_AUDIT_NORMAL_MAX_COLLISION_RATE}"
    --support-audit-rare-max-collision-rate 1.0
    --support-audit-normal-min-acceptance "${SUPPORT_AUDIT_NORMAL_MIN_ACCEPTANCE}"
    --support-audit-rare-min-acceptance 0.0
    --log-every "${LOG_EVERY}"
    --pretrained-exact-eval-phase "${PRETRAINED_EXACT_EVAL_PHASE}"
    --pretrained-quickcheck-task-count "${PRETRAINED_QUICKCHECK_TASK_COUNT}"
    --outdir "${outdir}"
  )
  if [[ "${DETERMINISTIC_DEBUG}" == "1" ]]; then
    cmd+=(--deterministic-debug)
  fi
  if [[ "${GRAD_ESTIMATOR}" == "reinforce" ]]; then
    cmd+=(--pretrain-grad-estimator "${PRETRAIN_GRAD_ESTIMATOR}")
    cmd+=(--reinforce-rollouts-per-task "${REINFORCE_ROLLOUTS_PER_TASK}")
    cmd+=(--reinforce-baseline "${REINFORCE_BASELINE}")
    cmd+=(--reinforce-ema-decay "${REINFORCE_EMA_DECAY}")
    cmd+=(--reinforce-meta-estimator "${REINFORCE_META_ESTIMATOR}")
    cmd+=(--reinforce-score "${REINFORCE_SCORE}")
    if [[ "${REINFORCE_SPLIT_SAMPLES}" == "1" ]]; then
      cmd+=(--reinforce-split-samples)
    fi
  fi
  if [[ "${IMPROVING_ONLY_GRAD}" == "1" ]]; then
    cmd+=(--improving-only-grad)
    cmd+=(--iog-scope "${IOG_SCOPE}")
  fi
  if (( BOOTSTRAP_PERMUTATIONS >= 1 )); then
    cmd+=(--bootstrap-permutations "${BOOTSTRAP_PERMUTATIONS}")
  fi
  if (( UNION_CACHE_TOP_C > 0 )); then
    cmd+=(--union-cache-top-c "${UNION_CACHE_TOP_C}")
    cmd+=(--union-cache-refresh-interval "${UNION_CACHE_REFRESH_INTERVAL}")
  fi
  if [[ -n "${IOG_PENALTY_COEF:-}" && "${IOG_PENALTY_COEF}" != "0" && "${IOG_PENALTY_COEF}" != "0.0" ]]; then
    cmd+=(--iog-penalty-coef "${IOG_PENALTY_COEF}")
  fi
  if [[ "${CALIBRATION_AWARE}" == "1" ]]; then
    cmd+=(--calibration-aware)
    cmd+=(--calibration-mode "${CALIBRATION_MODE}")
    cmd+=(--calibration-min-points "${CALIBRATION_MIN_POINTS}")
    cmd+=(--calibration-ridge-alpha "${CALIBRATION_RIDGE_ALPHA}")
    cmd+=(--calibration-ema "${CALIBRATION_EMA}")
  fi
  # A resume checkpoint carries model+optimizer+EMA+RNG state and is mutually
  # exclusive with the init/skip-pretrain flags, so when one is selected the
  # reuse/init chain is skipped entirely.
  resume_ckpt=""
  if [[ "${AUTO_RESUME}" == "1" && -f "${outdir}/resume_latest.pt" ]]; then
    resume_ckpt="${outdir}/resume_latest.pt"
  elif [[ -n "${RESUME_ROOT}" ]]; then
    resume_ckpt="${RESUME_ROOT}/seed_${seed}/resume_latest.pt"
    if [[ ! -f "${resume_ckpt}" ]]; then
      echo "RESUME_ROOT set but missing ${resume_ckpt}" >&2
      exit 1
    fi
    # Keep the extension outdir self-contained: the pretrained artifacts are
    # not inside the resume checkpoint, and the archival trace recompute
    # silently skips when pretrained_checkpoint.pt is absent.
    mkdir -p "${outdir}"
    for f in pretrained_checkpoint.pt pretrained_results.json; do
      if [[ ! -f "${outdir}/${f}" && -f "${RESUME_ROOT}/seed_${seed}/${f}" ]]; then
        cp "${RESUME_ROOT}/seed_${seed}/${f}" "${outdir}/${f}"
      fi
    done
  elif [[ -n "${RESUME_CHECKPOINT}" ]]; then
    resume_ckpt="${RESUME_CHECKPOINT}"
  fi
  if [[ -n "${resume_ckpt}" ]]; then
    cmd+=(--resume-checkpoint "${resume_ckpt}")
  elif [[ -n "${REUSE_PRETRAIN_ROOT_EFFECTIVE}" ]]; then
    reuse_seed_dir="${REUSE_PRETRAIN_ROOT_EFFECTIVE}/seed_${seed}"
    cmd+=(--init-checkpoint "${reuse_seed_dir}/pretrained_checkpoint.pt")
    cmd+=(--reuse-pretrained-results "${reuse_seed_dir}/pretrained_results.json")
    cmd+=(--skip-pretrain)
  elif [[ -n "${INIT_PRETRAIN_ROOT}" ]]; then
    cmd+=(--init-checkpoint "${INIT_PRETRAIN_ROOT}/seed_${seed}/pretrained_checkpoint.pt")
  elif [[ -n "${INIT_CHECKPOINT}" ]]; then
    cmd+=(--init-checkpoint "${INIT_CHECKPOINT}")
  fi
  if [[ -n "${TASK_BANK_ROOT}" ]]; then
    cmd+=(--task-bank-root "${TASK_BANK_ROOT}")
    cmd+=(--bank-materialization "${BANK_MATERIALIZATION}")
    if [[ "${TASK_BANK_SEED}" != "-1" ]]; then
      cmd+=(--task-bank-seed "${TASK_BANK_SEED}")
    fi
  fi
  if [[ "${MANIFEST_GROUPING}" != "iid" ]]; then
    cmd+=(--manifest-grouping "${MANIFEST_GROUPING}")
  fi
  if [[ "${TASK_VECTOR_SLIP}" == "1" ]]; then
    cmd+=(--task-vector-slip)
  fi
  if [[ "${BLIND_DYNAMICS}" == "1" ]]; then
    cmd+=(--blind-dynamics)
  fi
  printf 'OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q VECLIB_MAXIMUM_THREADS=%q ' \
    "${THREADS_PER_TASK}" "${THREADS_PER_TASK}" "${THREADS_PER_TASK}" "${THREADS_PER_TASK}" "${THREADS_PER_TASK}" >> "${TASKS_FILE}"
  printf '%q ' "${cmd[@]}" >> "${TASKS_FILE}"
  printf '\n' >> "${TASKS_FILE}"
done

array_max=$(( ${#SEED_ARRAY[@]} - 1 ))
if (( array_max < 0 )); then
  echo "No seeds configured." >&2
  exit 1
fi

sbatch_args=(
  -c "${THREADS_PER_TASK}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
  --output="${LOG_DIR}/%A_%a.out"
  --error="${LOG_DIR}/%A_%a.err"
  --export="ALL,TASKS_FILE=${TASKS_FILE},PYTHONPATH=${PYTHONPATH}"
)
# Whole-node allocation: shields against CPU oversubscription by other
# users' jobs on shared nodes (we saw node load at 2-7.5x the core count
# slow identical runs 40-90x).
if [[ "${EXCLUSIVE:-0}" == "1" ]]; then
  sbatch_args+=(--exclusive)
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

collate_out=$(sbatch \
  --dependency="afterok:${array_jobid}" \
  -c 1 \
  --mem=2G \
  --time=0-00:10:00 \
  --output="${LOG_DIR}/${array_jobid}_collate.out" \
  --error="${LOG_DIR}/${array_jobid}_collate.err" \
  --export="ALL,PYTHONPATH=${PYTHONPATH}" \
  --wrap="${PYTHON_BIN} -m forecast_experiment.collate_rlquant_trials --run-root ${OUTROOT}")
echo "${collate_out}"
collate_jobid=$(awk '{print $4}' <<< "${collate_out}")

cat > "${OUTROOT}/submission.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "outroot": "${OUTROOT}",
  "submit_script": "${SUBMIT_SCRIPT_PATH}",
  "tasks_file": "${TASKS_FILE}",
  "array_job_id": "${array_jobid}",
  "collate_job_id": "${collate_jobid}",
  "seeds": [$(printf '%s' "${SEED_ARRAY[*]}" | sed 's/ /, /g')],
  "threads_per_task": ${THREADS_PER_TASK},
  "memory": "${MEMORY}",
  "time_limit": "${TIME_LIMIT}",
  "account": "${SBATCH_ACCOUNT:-}",
  "partition": "${SBATCH_PARTITION:-}",
  "task_bank_root": "${TASK_BANK_ROOT}",
  "task_bank_seed": ${TASK_BANK_SEED},
  "bank_materialization": "${BANK_MATERIALIZATION}",
  "pretrained_exact_eval_phase": "${PRETRAINED_EXACT_EVAL_PHASE}",
  "pretrained_quickcheck_task_count": ${PRETRAINED_QUICKCHECK_TASK_COUNT},
  "requested_reuse_pretrain_root": "${REUSE_PRETRAIN_ROOT}",
  "reuse_pretrain_root": "${REUSE_PRETRAIN_ROOT_EFFECTIVE}",
  "reuse_pretrain_disabled_reason": "${REUSE_PRETRAIN_DISABLED_REASON}",
  "split_mode": "${SPLIT_MODE}",
  "manifest_grouping": "${MANIFEST_GROUPING}",
  "task_vector_slip": ${TASK_VECTOR_SLIP},
  "blind_dynamics": ${BLIND_DYNAMICS},
  "plotting_position": "${PLOTTING_POSITION}",
  "extrapolated_weighting": "${EXTRAPOLATED_WEIGHTING}",
  "meta_manifest_count": ${META_MANIFEST_COUNT},
  "heldout_manifest_count": ${HELDOUT_MANIFEST_COUNT},
  "meta_manifests_per_step": ${META_MANIFESTS_PER_STEP},
  "save_every": ${SAVE_EVERY},
  "meta_deploy_cache_top_b": ${META_DEPLOY_CACHE_TOP_B},
  "meta_deploy_cache_refresh_interval": ${META_DEPLOY_CACHE_REFRESH_INTERVAL},
  "meta_fit_subsample_min": ${META_FIT_SUBSAMPLE_MIN},
  "meta_fit_subsample_max": ${META_FIT_SUBSAMPLE_MAX},
  "qos": "${SBATCH_QOS:-}"
}
EOF

cat > "${OUTROOT}/EXPERIMENT.md" <<EOF
# RLQuant Trap Benchmark Report Run

## Submission
- Array job id: \`${array_jobid}\`
- Dependent collate job id: \`${collate_jobid}\`
- Output root: \`${OUTROOT}\`
- Submission script: \`${SUBMIT_SCRIPT_PATH}\`
- Tasks file: \`${TASKS_FILE}\`
- Seeds: \`${SEEDS_STR}\`
- Account: \`${SBATCH_ACCOUNT:-<default>}\`
- Partition: \`${SBATCH_PARTITION:-<default>}\`
- Task bank root: \`${TASK_BANK_ROOT:-<online sampling>}\`
- Task bank seed override: \`${TASK_BANK_SEED}\`
- Bank materialization: \`${BANK_MATERIALIZATION}\`
- Pretrained exact eval phase: \`${PRETRAINED_EXACT_EVAL_PHASE}\`
- Pretrained quickcheck task count: \`${PRETRAINED_QUICKCHECK_TASK_COUNT}\`
- Requested reuse pretrained root: \`${REUSE_PRETRAIN_ROOT:-<none>}\`
- Effective reuse pretrained root: \`${REUSE_PRETRAIN_ROOT_EFFECTIVE:-<none>}\`
- Reuse disabled reason: \`${REUSE_PRETRAIN_DISABLED_REASON:-<none>}\`

## Benchmark
- Family suite: \`trap_open_room_only\`
- Environment: \`8x8\` grid, horizon \`${HORIZON}\`, start-goal min distance \`${MIN_START_GOAL_DISTANCE}\`
- Task mechanism: visible nonterminal traps with known goal

## Training Objective
- Gradient estimator: \`exact\`
- Meta loss: \`${META_LOSS_MODE}\`
- Meta auxiliary loss: \`${META_AUX_LOSS_MODE}\`
- Meta auxiliary loss weight: \`${META_AUX_LOSS_WEIGHT}\`
- Meta auxiliary task source: \`${META_AUX_TASK_SOURCE}\`
- Gradient flow mode: \`${GRAD_MODE}\`
- Improving-only grad: \`${IMPROVING_ONLY_GRAD}\`
- Split mode: \`${SPLIT_MODE}\`
- Manifest grouping: \`${MANIFEST_GROUPING}\`
- Task vector slip: \`${TASK_VECTOR_SLIP}\`
- Blind dynamics: \`${BLIND_DYNAMICS}\`
- Meta manifest count: \`${META_MANIFEST_COUNT}\`
- Heldout manifest count: \`${HELDOUT_MANIFEST_COUNT}\`
- Meta manifests per step: \`${META_MANIFESTS_PER_STEP}\`
- Meta deploy cache top-B: \`${META_DEPLOY_CACHE_TOP_B}\`
- Meta deploy cache refresh interval: \`${META_DEPLOY_CACHE_REFRESH_INTERVAL}\`
- OLS top-k: \`${OLS_TOP_K}\`
- OLS score mode: \`raw_regret\`
- Plotting position: \`${PLOTTING_POSITION}\`
- Extrapolated weighting: \`${EXTRAPOLATED_WEIGHTING}\`

## Split Sizes
- Pretrain train per seed: \`${TRAP_PRETRAIN_TASKS}\`
- Meta manifest count per seed: \`${META_MANIFEST_COUNT}\`
- Heldout manifest count per seed: \`${HELDOUT_MANIFEST_COUNT}\`
- Meta fit: \`${TRAP_FIT_TASKS}\`
- Meta deploy extra: \`${TRAP_DEPLOY_EXTRA_TASKS}\`
- Heldout fit: \`${TRAP_HELDOUT_FIT_TASKS}\`
- Heldout deploy extra: \`${TRAP_HELDOUT_DEPLOY_EXTRA_TASKS}\`

## Optimization
- Pretrain steps: \`${PRETRAIN_STEPS}\`
- Meta steps: \`${META_STEPS}\`
- Save-every checkpoint interval: \`${SAVE_EVERY}\`
- Learning rate: \`${LR}\`
- Grad clip: \`${GRAD_CLIP}\`
- Pretrain batch size: \`${PRETRAIN_BATCH_SIZE}\`
- Eval batch size: \`${EVAL_BATCH_SIZE}\`
- Meta fit subsample min: \`${META_FIT_SUBSAMPLE_MIN}\`
- Meta fit subsample max: \`${META_FIT_SUBSAMPLE_MAX}\`
- Deterministic debug: \`${DETERMINISTIC_DEBUG}\`

## Runtime / Resources
- CPU threads per task: \`${THREADS_PER_TASK}\`
- Memory per task: \`${MEMORY}\`
- Wall-clock limit per task: \`${TIME_LIMIT}\`
- Array concurrency cap: \`${ARRAY_CONCURRENCY}\`
- QoS: \`${SBATCH_QOS:-<default>}\`

## Support / Validation Settings
- Support audit normal samples: \`${SUPPORT_AUDIT_NORMAL_SAMPLES}\`
- Support audit challenge samples: \`${SUPPORT_AUDIT_CHALLENGE_SAMPLES}\`
- Support audit normal min unique per family: \`${SUPPORT_AUDIT_NORMAL_MIN_UNIQUE_PER_FAMILY}\`
- Difficulty resample attempts: \`1\`

## Notes
- This run keeps pretraining inside each seed rather than warm-starting from a saved checkpoint unless an effective reuse root is recorded above.
- This run uses the strict extrapolated primal objective together with the pretrain-return auxiliary term.
- A dependent collate job will write \`summary.json\` and \`trial_metrics.csv\` under the output root after all seeds succeed.
EOF

echo "Array JobID:   ${array_jobid}"
echo "Collate JobID: ${collate_jobid}"
echo "Experiment note: ${OUTROOT}/EXPERIMENT.md"
