from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence

import numpy as np
import torch

from .eval_cache import (
    DeploySubsetCache,
    UnionTopCache,
    deserialize_deploy_caches,
    deserialize_union_caches,
    serialize_deploy_caches,
    serialize_union_caches,
)
from .ols_utils import (
    AffineCalibrationBuffer,
    count_extrapolated_eval_points,
    differentiable_ols_coefficients,
    differentiable_ols_extrapolated_primal_loss,
    differentiable_ols_extrapolated_primal_loss_bootstrap,
    differentiable_ols_primal_loss,
    fit_ols_tail_from_scores,
    plan_bootstrap_permutations,
    select_top_k_indices,
)
from .rlquant_gridworld import (
    CHALLENGE_FAMILIES,
    NORMAL_FAMILIES,
    DensePolicyNet,
    FamilySlice,
    GridworldGeneratorConfig,
    GridworldTask,
    GridworldTaskSpec,
    TrapLayoutRecord,
    evaluate_dense_policy_exact,
    materialize_task_from_spec,
    materialize_task_from_trap_layout_record,
    optimal_values_for_batch,
    prepare_task_batch,
    run_family_support_audit,
    sample_policy_rollouts,
    sample_policy_rollouts_batched,
    sample_unique_task_specs,
)


@dataclass
class RLQuantConfig:
    device: str = ""
    seed: int = 42
    family_suite: str = "naturalistic"  # naturalistic | pressure_plate | pressure_plate_open_room_only | trap_open_room_only | hybrid
    rows: int = 8
    cols: int = 8
    horizon: int = 20
    gamma: float = 1.0
    step_cost: float = -0.01
    goal_reward: float = 1.0
    catastrophe_penalty: float = -1.0
    min_start_goal_distance: int = 7
    hidden_channels: int = 32
    task_hidden_dim: int = 64

    pretrain_steps: int = 100
    meta_steps: int = 100
    meta_manifest_count: int = 20
    heldout_manifest_count: int = 5
    meta_manifests_per_step: int = 4
    meta_deploy_cache_top_b: int = 0
    meta_deploy_cache_refresh_interval: int = 0
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    init_checkpoint: str = ""
    resume_checkpoint: str = ""
    reuse_pretrained_results: str = ""
    skip_pretrain: bool = False
    deterministic_debug: bool = False
    pretrain_batch_size: int = 32
    eval_batch_size: int = 32

    grad_estimator: str = "exact"  # exact | reinforce
    # Pretraining-phase override. 'inherit' (default): use grad_estimator for
    # both phases. Setting 'exact' with grad_estimator='reinforce' keeps
    # pretraining exact while only the forecastability (meta) phase is
    # sampling-based -- the configuration of the paper's sampled-returns
    # experiment, which asks whether forecastability training still works when
    # it only sees sampled returns.
    pretrain_grad_estimator: str = "inherit"  # inherit | exact | reinforce
    reinforce_rollouts_per_task: int = 8
    # REINFORCE advantage baseline. 'rollout_mean' (default, legacy): per-task
    # mean of this step's rollouts -- identically-zero advantage at n=1, so
    # single-rollout training requires one of the other modes. 'optimal_value':
    # the bank's stored V* (advantage = return - V* = -sampled regret;
    # action-independent, hence unbiased, and no oracle beyond what sampled
    # regret itself requires). 'ema': per-task running mean of returns across
    # steps (oracle-free); updated after the advantage is computed, so the
    # baseline never depends on the current rollouts.
    reinforce_baseline: str = "rollout_mean"  # rollout_mean | optimal_value | ema
    reinforce_ema_decay: float = 0.9
    # Meta-phase gradient estimator when grad_estimator='reinforce'.
    # 'plugin' (estimator A): evaluate dLoss/dregret at the sampled mean
    # regrets and multiply by per-task REINFORCE gradients of expected regret;
    # low variance, O(1/n) bias. 'batch' (estimator B): treat the scalar loss
    # at the sampled regrets as a black-box reward for every trajectory with
    # per-task leave-one-out baselines; unbiased for the smoothed objective
    # E[loss(n-sample regrets)].
    reinforce_meta_estimator: str = "plugin"  # plugin | batch
    # Plugin-only: use the no-grad scoring rollouts for regret values, rank
    # selection, and loss weights, and a second independent rollout set for
    # the log-prob policy-gradient term (removes the weight/advantage
    # correlation bias term). Default off = coupled (values and log-probs
    # from the same rollouts, matching the original implementation).
    reinforce_split_samples: bool = False
    # Training-time score for the reinforce meta loss. 'regret' (default,
    # paper score): V* - sampled return, requires the bank's stored V*.
    # 'neg_return': -sampled return; combined with reinforce_baseline='ema'
    # this makes the whole training loop oracle-free (no value iteration
    # anywhere). Evaluation is unaffected either way (paper regret metrics).
    reinforce_score: str = "regret"  # regret | neg_return

    ols_top_k: int = 10
    plotting_position: str = "weibull"
    extrapolated_weighting: str = "rank_uniform"  # rank_uniform | deploy_log_uniform | deploy_uniform
    ols_score_mode: str = "raw_regret"  # raw_regret | log_regret
    meta_loss_mode: str = "ols_coeff"  # ols_coeff | ols_primal | ols_primal_weighted | ols_primal_extrapolated
    meta_aux_loss_mode: str = "none"  # none | pretrain_return | anchored_pretrain_return
    meta_aux_loss_weight: float = 0.0
    meta_aux_task_source: str = "pretrain_train"  # pretrain_train | meta_fit
    # SFT-style baselines (M1, M3): swap the pretrain task pool with tasks drawn
    # from the meta manifests instead of the canonical pretrain_train pool. Pair
    # with --meta-steps 0 to skip meta entirely; pretrained eval still runs and
    # produces forecast_traces.npz/pretrained_results.json for downstream calibration.
    pretrain_task_source: str = "pretrain_train"  # pretrain_train | meta_union | meta_deploy
    grad_mode: str = "fit"  # fit | eval | both
    improving_only_grad: bool = False
    # Scope of the improving-only mask. 'both' (default, current behavior):
    # mask both deploy-side and fit-side gradients that would worsen per-task
    # regret. 'deploy_only': only mask deploy-side. 'fit_only': symmetric
    # ablation. Ignored when improving_only_grad=False.
    iog_scope: str = "both"  # both | deploy_only | fit_only
    # Soft replacement for improving_only_grad: a squared-hinge penalty on
    # per-task regret growing past a pretrained baseline. 0 = disabled.
    # The penalty is added to the meta-loss per (fit, deploy) manifest and
    # is a well-defined gradient on a real objective (unlike the masked IOG
    # gradient, which is a biased subgradient and can let the loss climb).
    # Scoped to the same deploy candidate tasks that feed the OLS loss each
    # step. RL value scale (regret in [0,1]) differs from LM (log_prob in
    # [-25,0]); RL needs a much larger coef than LM for comparable effect.
    iog_penalty_coef: float = 0.0
    # Calibration-aware forecastability training. At deployment the forecast is
    # affinely corrected (alpha*Q + beta) by a calibrator fit on the train
    # manifests; with this enabled, training optimizes the CALIBRATED forecast
    # so the model targets the residual the affine map cannot fix. The
    # calibrator is fit leave-one-group-out over a persistent buffer of
    # per-manifest worst-rank (forecast, actual) pairs (no extra rollouts, no
    # gradient through the fit); the buffer spans many steps, which matters
    # because only meta_manifests_per_step manifests are seen per step. Only
    # takes effect when meta_loss_mode == "ols_primal_extrapolated".
    calibration_aware: bool = False
    calibration_mode: str = "shift_scale"  # shift | shift_scale
    calibration_min_points: int = 5        # warmup donor manifests before calibrating
    calibration_ridge_alpha: float = 0.05  # slope shrink toward 1 (dimensionless)
    calibration_ema: float = 0.0           # per-manifest EMA of contributions (0 = exact)
    # Bootstrap-permutation meta-loss. >=1 enables permuting fit_subset ∪
    # deploy_tasks every meta-step; the per-permutation losses are averaged.
    # Only takes effect when meta_loss_mode == "ols_primal_extrapolated".
    # 0 = disabled (use the standard fixed-partition path). Re-partitioning
    # per step breaks per-manifest asymmetric overfit. N=1 is sufficient for
    # asymmetry-breaking; N>1 adds variance reduction.
    bootstrap_permutations: int = 0
    # Partition-invariant union top-C cache for the bootstrap path.
    # Caches the top-C indices of fit_subset+deploy_tasks; reuses for
    # refresh_interval steps. Set to >= deploy_extrap_k + n_fit (typically
    # ~296 at canonical scale) for deterministic deploy coverage. 0 =
    # disabled (full union scored every step).
    union_cache_top_c: int = 0
    union_cache_refresh_interval: int = 5
    split_mode: str = "disjoint"  # disjoint | nested
    meta_fit_subsample_min: int = 16
    meta_fit_subsample_max: int = 0
    # How manifests are carved out of a prebuilt task bank. 'iid' (default,
    # legacy): one global shuffle then disjoint slices, so every manifest is an
    # IID sample of the same pool. 'slip_level': group bank records by their
    # slip_value and build one manifest per level; requires a bank whose number
    # of distinct slip values equals meta_manifest_count + heldout_manifest_count.
    # 'dynamics_id': same but grouped by the records' dynamics_id (slip + wind
    # banks). Which groups are heldout is drawn from cfg.seed, so heldout
    # dynamics are never trained on for that seed.
    manifest_grouping: str = "iid"  # iid | slip_level | dynamics_id
    # Append the task's uniform slip probability as an extra task-vector entry
    # (FiLM conditioning at every policy block). Changes the network parameter
    # shapes; default off preserves the legacy 29-dim vector bit-exactly.
    task_vector_slip: bool = False
    # Withhold the dynamics from the policy's observation: the slip feature
    # channel is zeroed (wind is never observed) while the transition spec
    # keeps the true dynamics. With dynamics varying across manifests, this
    # gives per-manifest regret an irreducible hedging floor that training
    # cannot flatten on unseen layouts.
    blind_dynamics: bool = False
    task_bank_root: str = ""
    task_bank_seed: int = -1
    bank_materialization: str = "lazy"  # eager | lazy
    pretrained_exact_eval_phase: str = "post"  # pre | post
    pretrained_quickcheck_task_count: int = 0

    pretrain_normal_per_family: int = 96
    pretrain_visible_challenge_per_family: int = 48
    meta_fit_normal_per_family: int = 64
    meta_fit_visible_challenge_per_family: int = 48
    meta_deploy_extra_per_family: int = 96
    meta_deploy_hard_visible_per_family: int = 32
    heldout_fit_normal_per_family: int = 48
    heldout_fit_decoy_per_family: int = 48
    heldout_deploy_extra_per_family: int = 96
    heldout_deploy_bridge_extra_per_family: int = 24
    pressure_pretrain_per_family: int = 96
    pressure_meta_fit_per_family: int = 64
    pressure_meta_deploy_extra_per_family: int = 96
    pressure_heldout_fit_per_family: int = 48
    pressure_heldout_deploy_extra_per_family: int = 96
    trap_pretrain_per_family: int = 96
    trap_meta_fit_per_family: int = 64
    trap_meta_deploy_extra_per_family: int = 96
    trap_heldout_fit_per_family: int = 48
    trap_heldout_deploy_extra_per_family: int = 96

    support_audit_normal_samples: int = 100000
    support_audit_challenge_samples: int = 20000
    support_audit_total_min_unique: int = 100000
    support_audit_normal_total_min_unique: int = 100000
    support_audit_normal_min_unique_per_family: int = 50000
    support_audit_rare_min_unique: int = 5000
    support_audit_normal_max_collision_rate: float = 0.10
    support_audit_rare_max_collision_rate: float = 0.25
    support_audit_normal_min_acceptance: float = 0.30
    support_audit_rare_min_acceptance: float = 0.10

    log_every: int = 10
    save_every: int = 0
    outdir: str = "./artifacts/results/rlquant"


TaskRecord = GridworldTask | GridworldTaskSpec | TrapLayoutRecord
TaskMaterializationCache = dict[str, GridworldTask]


@dataclass
class TaskBankInfo:
    root: Path
    bank_dir: Path
    bank_path: Path
    bank_seed: int
    record_format: str
    submission_meta: dict[str, object]
    bank_meta: dict[str, object]
    shard_meta: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "bank_dir": str(self.bank_dir),
            "bank_path": str(self.bank_path),
            "bank_seed": int(self.bank_seed),
            "record_format": str(self.record_format),
            "unique_count": int(self.bank_meta.get("unique_count", 0)),
            "target_total": int(self.bank_meta.get("target_total", 0)),
            "family_counts": dict(self.bank_meta.get("family_counts", {})),
        }


def _task_key(task: TaskRecord) -> str:
    return str(task.canonical_hash)


def _is_tail_task(task: TaskRecord) -> bool:
    return abs(float(task.severity) - 1.0) <= 1e-9


def _task_family_counts(tasks: Sequence[TaskRecord]) -> dict[str, int]:
    family_counts: dict[str, int] = {}
    for task in tasks:
        family_counts[task.family_id] = family_counts.get(task.family_id, 0) + 1
    return family_counts


def _task_collection_summary(tasks: Sequence[TaskRecord]) -> dict[str, object]:
    if tasks:
        severities = [float(task.severity) for task in tasks]
        tail_count = sum(1 for task in tasks if _is_tail_task(task))
        severity_summary = {
            "min": float(min(severities)),
            "mean": float(sum(severities) / len(severities)),
            "max": float(max(severities)),
            "tail_count": int(tail_count),
            "tail_fraction": float(tail_count / len(tasks)),
        }
    else:
        severity_summary = {
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "tail_count": 0,
            "tail_fraction": 0.0,
        }
    out: dict[str, object] = {
        "count": len(tasks),
        "family_counts": _task_family_counts(tasks),
        "severity": severity_summary,
    }
    slip_values = [
        float(task.slip_value) for task in tasks if hasattr(task, "slip_value")
    ]
    if slip_values:
        out["slip"] = {
            "min": float(min(slip_values)),
            "mean": float(sum(slip_values) / len(slip_values)),
            "max": float(max(slip_values)),
        }
    wind_values = [
        float(task.wind_strength) for task in tasks if hasattr(task, "wind_strength")
    ]
    if wind_values and any(value != 0.0 for value in wind_values):
        out["wind_strength"] = {
            "min": float(min(wind_values)),
            "mean": float(sum(wind_values) / len(wind_values)),
            "max": float(max(wind_values)),
        }
    return out


def _flatten_manifest_suite_tasks(manifest_suite: "TaskManifestSuite") -> list[TaskRecord]:
    flat = list(manifest_suite.pretrain_train)
    for manifest in manifest_suite.meta_manifests:
        flat.extend(manifest.fit_tasks)
        flat.extend(manifest.deploy_tasks)
    for manifest in manifest_suite.heldout_manifests:
        flat.extend(manifest.fit_tasks)
        flat.extend(manifest.deploy_tasks)
    return flat


def _effective_task_bank_seed(cfg: RLQuantConfig) -> int:
    return int(cfg.seed if cfg.task_bank_seed < 0 else cfg.task_bank_seed)


# Signature keys added after the first runs shipped; configs/checkpoints from
# before those keys existed are treated as having these values.
_LEGACY_SIGNATURE_DEFAULTS: dict[str, object] = {
    "manifest_grouping": "iid",
    "task_vector_slip": False,
    "blind_dynamics": False,
}


def _pretrained_reuse_signature(cfg: RLQuantConfig) -> dict[str, object]:
    signature = {
        "seed": int(cfg.seed),
        "family_suite": cfg.family_suite,
        "rows": int(cfg.rows),
        "cols": int(cfg.cols),
        "horizon": int(cfg.horizon),
        "gamma": float(cfg.gamma),
        "step_cost": float(cfg.step_cost),
        "goal_reward": float(cfg.goal_reward),
        "catastrophe_penalty": float(cfg.catastrophe_penalty),
        "min_start_goal_distance": int(cfg.min_start_goal_distance),
        "split_mode": cfg.split_mode,
        "manifest_grouping": cfg.manifest_grouping,
        "task_vector_slip": bool(cfg.task_vector_slip),
        "blind_dynamics": bool(cfg.blind_dynamics),
        "meta_manifest_count": int(cfg.meta_manifest_count),
        "heldout_manifest_count": int(cfg.heldout_manifest_count),
        "task_bank_root": str(cfg.task_bank_root),
        "effective_task_bank_seed": _effective_task_bank_seed(cfg),
    }
    if cfg.family_suite in {"naturalistic", "hybrid"}:
        signature.update(
            {
                "pretrain_normal_per_family": int(cfg.pretrain_normal_per_family),
                "pretrain_visible_challenge_per_family": int(cfg.pretrain_visible_challenge_per_family),
                "meta_fit_normal_per_family": int(cfg.meta_fit_normal_per_family),
                "meta_fit_visible_challenge_per_family": int(cfg.meta_fit_visible_challenge_per_family),
                "meta_deploy_extra_per_family": int(cfg.meta_deploy_extra_per_family),
                "meta_deploy_hard_visible_per_family": int(cfg.meta_deploy_hard_visible_per_family),
                "heldout_fit_normal_per_family": int(cfg.heldout_fit_normal_per_family),
                "heldout_fit_decoy_per_family": int(cfg.heldout_fit_decoy_per_family),
                "heldout_deploy_extra_per_family": int(cfg.heldout_deploy_extra_per_family),
                "heldout_deploy_bridge_extra_per_family": int(cfg.heldout_deploy_bridge_extra_per_family),
            }
        )
    if cfg.family_suite in {"pressure_plate", "pressure_plate_open_room_only", "hybrid"}:
        signature.update(
            {
                "pressure_pretrain_per_family": int(cfg.pressure_pretrain_per_family),
                "pressure_meta_fit_per_family": int(cfg.pressure_meta_fit_per_family),
                "pressure_meta_deploy_extra_per_family": int(cfg.pressure_meta_deploy_extra_per_family),
                "pressure_heldout_fit_per_family": int(cfg.pressure_heldout_fit_per_family),
                "pressure_heldout_deploy_extra_per_family": int(cfg.pressure_heldout_deploy_extra_per_family),
            }
        )
    if cfg.family_suite in {"trap_open_room_only", "hybrid"}:
        signature.update(
            {
                "trap_pretrain_per_family": int(cfg.trap_pretrain_per_family),
                "trap_meta_fit_per_family": int(cfg.trap_meta_fit_per_family),
                "trap_meta_deploy_extra_per_family": int(cfg.trap_meta_deploy_extra_per_family),
                "trap_heldout_fit_per_family": int(cfg.trap_heldout_fit_per_family),
                "trap_heldout_deploy_extra_per_family": int(cfg.trap_heldout_deploy_extra_per_family),
            }
        )
    return signature


def _format_elapsed_short(elapsed_sec: float) -> str:
    total_seconds = max(0, int(round(elapsed_sec)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _resume_latest_checkpoint_path(outdir: Path) -> Path:
    return outdir / "resume_latest.pt"


def _resume_prev_checkpoint_path(outdir: Path) -> Path:
    return outdir / "resume_prev.pt"


def _save_resume_checkpoint(outdir: Path, state: dict[str, object]) -> None:
    latest = _resume_latest_checkpoint_path(outdir)
    prev = _resume_prev_checkpoint_path(outdir)
    tmp = outdir / "resume_latest.pt.tmp"
    if latest.exists():
        if prev.exists():
            prev.unlink()
        latest.replace(prev)
    torch.save(state, tmp)
    tmp.replace(latest)


def _load_resume_checkpoint(path: str) -> dict[str, object]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported resume checkpoint format at {path}")
    return state


def _critical_cfg_signature(cfg: RLQuantConfig) -> dict[str, object]:
    return {
        "family_suite": cfg.family_suite,
        "rows": int(cfg.rows),
        "cols": int(cfg.cols),
        "horizon": int(cfg.horizon),
        "gamma": float(cfg.gamma),
        "step_cost": float(cfg.step_cost),
        "goal_reward": float(cfg.goal_reward),
        "catastrophe_penalty": float(cfg.catastrophe_penalty),
        "min_start_goal_distance": int(cfg.min_start_goal_distance),
        "split_mode": cfg.split_mode,
        "manifest_grouping": cfg.manifest_grouping,
        "task_vector_slip": bool(cfg.task_vector_slip),
        "blind_dynamics": bool(cfg.blind_dynamics),
        "meta_manifest_count": int(cfg.meta_manifest_count),
        "heldout_manifest_count": int(cfg.heldout_manifest_count),
        "task_bank_root": str(cfg.task_bank_root),
        "task_bank_seed": int(cfg.task_bank_seed),
        "bank_materialization": cfg.bank_materialization,
        "pretrained_exact_eval_phase": cfg.pretrained_exact_eval_phase,
        "pretrained_quickcheck_task_count": int(cfg.pretrained_quickcheck_task_count),
    }


def _validate_resume_checkpoint_config(
    checkpoint_state: dict[str, object],
    cfg: RLQuantConfig,
) -> None:
    saved_signature = checkpoint_state.get("config_signature")
    if not isinstance(saved_signature, dict):
        return
    failures: list[str] = []
    current = _critical_cfg_signature(cfg)
    for key, expected in current.items():
        actual = saved_signature.get(key)
        if actual is None and key in _LEGACY_SIGNATURE_DEFAULTS:
            actual = _LEGACY_SIGNATURE_DEFAULTS[key]
        if actual != expected:
            failures.append(f"{key}: checkpoint={actual} != cfg={expected}")
    if failures:
        raise RuntimeError("Resume checkpoint config mismatch: " + "; ".join(failures))


@dataclass
class FitDeployManifest:
    name: str
    fit_tasks: list[TaskRecord]
    deploy_tasks: list[TaskRecord]
    slip_level: float | None = None

    def summary(self) -> dict[str, object]:
        fit_ids = {_task_key(task) for task in self.fit_tasks}
        deploy_ids = {_task_key(task) for task in self.deploy_tasks}
        out: dict[str, object] = {
            "name": self.name,
            "fit": _task_collection_summary(self.fit_tasks),
            "deploy": _task_collection_summary(self.deploy_tasks),
            "overlap_count": len(fit_ids & deploy_ids),
        }
        if self.slip_level is not None:
            out["slip_level"] = float(self.slip_level)
        return out


@dataclass
class TaskManifestSuite:
    pretrain_train: list[TaskRecord]
    meta_manifests: list[FitDeployManifest]
    heldout_manifests: list[FitDeployManifest]

    def summary(self) -> dict[str, object]:
        return {
            "pretrain_train": _task_collection_summary(self.pretrain_train),
            "meta_manifest_count": len(self.meta_manifests),
            "heldout_manifest_count": len(self.heldout_manifests),
            "meta_manifests": [manifest.summary() for manifest in self.meta_manifests],
            "heldout_manifests": [manifest.summary() for manifest in self.heldout_manifests],
            "aggregate_counts": {
                "meta_fit_total": sum(len(manifest.fit_tasks) for manifest in self.meta_manifests),
                "meta_deploy_total": sum(len(manifest.deploy_tasks) for manifest in self.meta_manifests),
                "heldout_fit_total": sum(len(manifest.fit_tasks) for manifest in self.heldout_manifests),
                "heldout_deploy_total": sum(len(manifest.deploy_tasks) for manifest in self.heldout_manifests),
            },
        }


def _batched(items: Sequence[TaskRecord], batch_size: int) -> Iterable[Sequence[TaskRecord]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _seed_everything(seed: int, *, deterministic_debug: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_debug:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _eval_batch_size(tasks: Sequence[TaskRecord], cfg: RLQuantConfig) -> int:
    if cfg.deterministic_debug:
        return max(1, len(tasks))
    return max(1, cfg.eval_batch_size)


def _pretrain_batch(tasks: Sequence[TaskRecord], cfg: RLQuantConfig, rng: random.Random) -> list[TaskRecord]:
    if cfg.deterministic_debug or cfg.pretrain_batch_size <= 0 or cfg.pretrain_batch_size >= len(tasks):
        return list(tasks)
    return rng.sample(list(tasks), min(cfg.pretrain_batch_size, len(tasks)))


def _make_generator_cfg(cfg: RLQuantConfig) -> GridworldGeneratorConfig:
    return GridworldGeneratorConfig(
        rows=cfg.rows,
        cols=cfg.cols,
        horizon=cfg.horizon,
        gamma=cfg.gamma,
        step_cost=cfg.step_cost,
        goal_reward=cfg.goal_reward,
        catastrophe_penalty=cfg.catastrophe_penalty,
        min_start_goal_distance=cfg.min_start_goal_distance,
    )


def _default_manifest_slices(cfg: RLQuantConfig) -> dict[str, list[FamilySlice]]:
    naturalistic = {
        "pretrain_train": [
            FamilySlice("open_room", cfg.pretrain_normal_per_family, 0.05, 0.35),
            FamilySlice("wall_detour", cfg.pretrain_normal_per_family, 0.05, 0.40),
            FamilySlice("bridge_shortcut", cfg.pretrain_visible_challenge_per_family, 0.20, 0.55),
            FamilySlice("key_door", cfg.pretrain_visible_challenge_per_family, 0.20, 0.55),
        ],
        "meta_fit": [
            FamilySlice("open_room", cfg.meta_fit_normal_per_family, 0.05, 0.35),
            FamilySlice("wall_detour", cfg.meta_fit_normal_per_family, 0.10, 0.45),
            FamilySlice("bridge_shortcut", cfg.meta_fit_visible_challenge_per_family, 0.25, 0.55),
            FamilySlice("key_door", cfg.meta_fit_visible_challenge_per_family, 0.25, 0.55),
        ],
        "meta_deploy_extra": [
            FamilySlice("bridge_shortcut", cfg.meta_deploy_hard_visible_per_family, 0.55, 0.95),
            FamilySlice("key_door", cfg.meta_deploy_hard_visible_per_family, 0.55, 0.95),
            FamilySlice("one_way_fork", cfg.meta_deploy_extra_per_family, 0.55, 0.95),
            FamilySlice("slip_zone", cfg.meta_deploy_extra_per_family, 0.55, 0.95),
        ],
        "heldout_fit": [
            FamilySlice("open_room", cfg.heldout_fit_normal_per_family, 0.05, 0.35),
            FamilySlice("wall_detour", cfg.heldout_fit_normal_per_family, 0.10, 0.45),
            FamilySlice("decoy_reward", cfg.heldout_fit_decoy_per_family, 0.30, 0.60),
        ],
        "heldout_deploy_extra": [
            FamilySlice("decoy_reward", cfg.heldout_deploy_extra_per_family, 0.60, 0.98),
            FamilySlice("one_way_fork", cfg.heldout_deploy_extra_per_family, 0.60, 0.98),
            FamilySlice("slip_zone", cfg.heldout_deploy_extra_per_family, 0.60, 0.98),
            FamilySlice("bridge_shortcut", cfg.heldout_deploy_bridge_extra_per_family, 0.60, 0.98),
        ],
    }
    pressure_plate = {
        "pretrain_train": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_pretrain_per_family, 0.08, 0.35),
        ],
        "meta_fit": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_meta_fit_per_family, 0.12, 0.45),
        ],
        "meta_deploy_extra": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_meta_deploy_extra_per_family, 0.55, 0.95),
        ],
        "heldout_fit": [
            FamilySlice("pressure_plate_wall_detour", cfg.pressure_heldout_fit_per_family, 0.12, 0.45),
        ],
        "heldout_deploy_extra": [
            FamilySlice("pressure_plate_wall_detour", cfg.pressure_heldout_deploy_extra_per_family, 0.55, 0.98),
        ],
    }
    pressure_plate_open_room_only = {
        "pretrain_train": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_pretrain_per_family, 0.08, 0.45),
        ],
        "meta_fit": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_meta_fit_per_family, 0.10, 0.45),
        ],
        "meta_deploy_extra": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_meta_deploy_extra_per_family, 0.55, 0.95),
        ],
        "heldout_fit": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_heldout_fit_per_family, 0.10, 0.45),
        ],
        "heldout_deploy_extra": [
            FamilySlice("pressure_plate_open_room", cfg.pressure_heldout_deploy_extra_per_family, 0.55, 0.98),
        ],
    }
    trap_open_room_only = {
        "pretrain_train": [
            FamilySlice("trap_open_room", cfg.trap_pretrain_per_family, 0.08, 0.40),
        ],
        "meta_fit": [
            FamilySlice("trap_open_room", cfg.trap_meta_fit_per_family, 0.12, 0.45),
        ],
        "meta_deploy_extra": [
            FamilySlice("trap_open_room", cfg.trap_meta_deploy_extra_per_family, 0.55, 0.98),
        ],
        "heldout_fit": [
            FamilySlice("trap_open_room", cfg.trap_heldout_fit_per_family, 0.12, 0.45),
        ],
        "heldout_deploy_extra": [
            FamilySlice("trap_open_room", cfg.trap_heldout_deploy_extra_per_family, 0.55, 0.98),
        ],
    }
    if cfg.family_suite == "naturalistic":
        return naturalistic
    if cfg.family_suite == "pressure_plate":
        return pressure_plate
    if cfg.family_suite == "pressure_plate_open_room_only":
        return pressure_plate_open_room_only
    if cfg.family_suite == "trap_open_room_only":
        return trap_open_room_only
    if cfg.family_suite == "hybrid":
        return {
            key: list(naturalistic[key]) + list(pressure_plate[key])
            for key in naturalistic.keys()
        }
    raise ValueError(f"Unsupported family_suite: {cfg.family_suite}")


def _planned_usage_by_family(slice_map: dict[str, list[FamilySlice]], cfg: RLQuantConfig) -> dict[str, int]:
    usage = {family: 0 for family in NORMAL_FAMILIES + CHALLENGE_FAMILIES}
    split_multipliers = {
        "pretrain_train": 1,
        "meta_fit": cfg.meta_manifest_count,
        "meta_deploy_extra": cfg.meta_manifest_count,
        "heldout_fit": cfg.heldout_manifest_count,
        "heldout_deploy_extra": cfg.heldout_manifest_count,
    }
    for split_name, slices in slice_map.items():
        multiplier = int(split_multipliers.get(split_name, 1))
        for family_slice in slices:
            usage[family_slice.family_id] = usage.get(family_slice.family_id, 0) + multiplier * int(family_slice.count)
    return usage


def _validate_support_audit(audit: dict, cfg: RLQuantConfig) -> dict:
    failures: list[str] = []
    family_results = audit["family_results"]
    if int(audit["overall_unique_total"]) < cfg.support_audit_total_min_unique:
        failures.append(
            f"overall_unique_total={int(audit['overall_unique_total'])} < {cfg.support_audit_total_min_unique}"
        )
    if int(audit["normal_unique_total"]) < cfg.support_audit_normal_total_min_unique:
        failures.append(
            f"normal_unique_total={int(audit['normal_unique_total'])} < {cfg.support_audit_normal_total_min_unique}"
        )
    for family_id, metrics in family_results.items():
        unique_count = int(metrics["unique_count"])
        acceptance_rate = float(metrics["acceptance_rate"])
        collision_rate = float(metrics["collision_rate"])
        required_unique = int(metrics["min_required_unique"])
        if family_id in NORMAL_FAMILIES:
            required_unique = max(required_unique, cfg.support_audit_normal_min_unique_per_family)
            if acceptance_rate < cfg.support_audit_normal_min_acceptance:
                failures.append(f"{family_id}: acceptance_rate={acceptance_rate:.3f}")
            if collision_rate > cfg.support_audit_normal_max_collision_rate:
                failures.append(f"{family_id}: collision_rate={collision_rate:.3f}")
        else:
            required_unique = max(required_unique, cfg.support_audit_rare_min_unique)
            if acceptance_rate < cfg.support_audit_rare_min_acceptance:
                failures.append(f"{family_id}: acceptance_rate={acceptance_rate:.3f}")
            if collision_rate > cfg.support_audit_rare_max_collision_rate:
                failures.append(f"{family_id}: collision_rate={collision_rate:.3f}")
        if unique_count < required_unique:
            failures.append(f"{family_id}: unique_count={unique_count} < {required_unique}")
    return {
        "passed": len(failures) == 0,
        "failures": failures,
    }


def _bank_mode_enabled(cfg: RLQuantConfig) -> bool:
    return bool(str(cfg.task_bank_root).strip())


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_task_bank_info(cfg: RLQuantConfig) -> TaskBankInfo:
    root = Path(cfg.task_bank_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"task_bank_root does not exist: {root}")
    requested_seed = int(cfg.task_bank_seed) if cfg.task_bank_seed >= 0 else int(cfg.seed)
    if (root / "bank_specs.jsonl").exists():
        bank_dir = root
        bank_path = root / "bank_specs.jsonl"
        bank_seed = requested_seed
        submission_path = root.parent / "submission.json"
        record_format = "spec"
    elif (root / "bank_layouts.jsonl").exists():
        bank_dir = root
        bank_path = root / "bank_layouts.jsonl"
        bank_seed = requested_seed
        submission_path = root.parent / "submission.json"
        record_format = "trap_layout"
    else:
        bank_dir = root / f"bank_seed_{requested_seed}"
        if (bank_dir / "bank_specs.jsonl").exists():
            bank_path = bank_dir / "bank_specs.jsonl"
            record_format = "spec"
        elif (bank_dir / "bank_layouts.jsonl").exists():
            bank_path = bank_dir / "bank_layouts.jsonl"
            record_format = "trap_layout"
        else:
            bank_path = bank_dir / "bank_specs.jsonl"
            record_format = "spec"
        bank_seed = requested_seed
        submission_path = root / "submission.json"
    if not bank_path.exists():
        raise FileNotFoundError(f"Could not find task bank records at {bank_path}")
    bank_meta_path = bank_path.with_suffix(bank_path.suffix + ".meta.json")
    if not bank_meta_path.exists():
        raise FileNotFoundError(f"Missing bank metadata at {bank_meta_path}")
    shard_meta_paths = sorted((bank_dir / "shards").glob("*.jsonl.meta.json"))
    if not shard_meta_paths:
        raise FileNotFoundError(f"Missing shard metadata under {bank_dir / 'shards'}")
    submission_meta: dict[str, object] = {}
    if submission_path.exists():
        submission_meta = _read_json(submission_path)
    return TaskBankInfo(
        root=root if (root / "bank_specs.jsonl").exists() else root,
        bank_dir=bank_dir,
        bank_path=bank_path,
        bank_seed=bank_seed,
        record_format=record_format,
        submission_meta=submission_meta,
        bank_meta=_read_json(bank_meta_path),
        shard_meta=_read_json(shard_meta_paths[0]),
    )


def _validate_task_bank_compatibility(cfg: RLQuantConfig, bank_info: TaskBankInfo) -> dict[str, object]:
    failures: list[str] = []
    family_counts = bank_info.bank_meta.get("family_counts", {})
    family_ids = sorted(str(family_id) for family_id in family_counts.keys())
    expected_family_ids = sorted(
        family_slice.family_id
        for split_slices in _default_manifest_slices(cfg).values()
        for family_slice in split_slices
        if family_slice.count > 0
    )
    expected_family_ids = sorted(set(expected_family_ids))
    if family_ids != expected_family_ids:
        failures.append(f"bank families {family_ids} != expected {expected_family_ids}")
    shard_meta = bank_info.shard_meta
    comparisons = {
        "rows": cfg.rows,
        "cols": cfg.cols,
        "horizon": cfg.horizon,
        "gamma": cfg.gamma,
        "step_cost": cfg.step_cost,
        "goal_reward": cfg.goal_reward,
        "catastrophe_penalty": cfg.catastrophe_penalty,
        "min_start_goal_distance": cfg.min_start_goal_distance,
    }
    for key, expected in comparisons.items():
        actual = shard_meta.get(key)
        if actual is None:
            failures.append(f"bank shard metadata missing {key}")
            continue
        if float(actual) != float(expected):
            failures.append(f"{key}: bank={actual} != cfg={expected}")
    if int(bank_info.bank_meta.get("unique_count", 0)) <= 0:
        failures.append("bank unique_count must be positive")
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "bank": bank_info.summary(),
    }


def _required_manifest_counts(cfg: RLQuantConfig) -> dict[str, int]:
    slices = _default_manifest_slices(cfg)
    return {
        "pretrain_train": int(sum(family_slice.count for family_slice in slices["pretrain_train"])),
        "meta_fit": int(sum(family_slice.count for family_slice in slices["meta_fit"])),
        "meta_deploy_extra": int(sum(family_slice.count for family_slice in slices["meta_deploy_extra"])),
        "heldout_fit": int(sum(family_slice.count for family_slice in slices["heldout_fit"])),
        "heldout_deploy_extra": int(sum(family_slice.count for family_slice in slices["heldout_deploy_extra"])),
    }


def _bank_required_total(cfg: RLQuantConfig) -> int:
    counts = _required_manifest_counts(cfg)
    return (
        counts["pretrain_train"]
        + cfg.meta_manifest_count * (counts["meta_fit"] + counts["meta_deploy_extra"])
        + cfg.heldout_manifest_count * (counts["heldout_fit"] + counts["heldout_deploy_extra"])
    )


def _load_task_bank_records(bank_info: TaskBankInfo) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    bank_path = bank_info.bank_path
    with bank_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if bank_info.record_format == "spec":
                records.append(GridworldTaskSpec.from_json(payload))
            elif bank_info.record_format == "trap_layout":
                records.append(TrapLayoutRecord.from_json(payload))
            else:
                raise ValueError(f"Unsupported task bank record_format: {bank_info.record_format}")
    return records


def _take_partition_slice(
    specs: Sequence[TaskRecord],
    *,
    cursor: int,
    count: int,
    label: str,
) -> tuple[list[TaskRecord], int]:
    next_cursor = cursor + int(count)
    if next_cursor > len(specs):
        raise RuntimeError(
            f"Bank partition ran out of tasks while taking {label}: "
            f"needed {count} from cursor {cursor}, only {len(specs) - cursor} remain"
        )
    return list(specs[cursor:next_cursor]), next_cursor


def _build_manifest_suite_from_bank(
    cfg: RLQuantConfig,
    *,
    seed: int,
    bank_info: TaskBankInfo,
) -> TaskManifestSuite:
    counts = _required_manifest_counts(cfg)
    specs = _load_task_bank_records(bank_info)
    required_total = _bank_required_total(cfg)
    if len(specs) < required_total:
        raise RuntimeError(
            f"Task bank too small: {len(specs)} specs < required_total={required_total}"
        )
    shuffle_rng = random.Random(seed)
    shuffle_rng.shuffle(specs)
    cursor = 0
    pretrain_train, cursor = _take_partition_slice(
        specs,
        cursor=cursor,
        count=counts["pretrain_train"],
        label="pretrain_train",
    )
    meta_manifests: list[FitDeployManifest] = []
    for idx in range(cfg.meta_manifest_count):
        fit_tasks, cursor = _take_partition_slice(
            specs,
            cursor=cursor,
            count=counts["meta_fit"],
            label=f"meta_{idx:02d}_fit",
        )
        deploy_extra_tasks, cursor = _take_partition_slice(
            specs,
            cursor=cursor,
            count=counts["meta_deploy_extra"],
            label=f"meta_{idx:02d}_deploy_extra",
        )
        deploy_tasks = list(fit_tasks) + list(deploy_extra_tasks) if cfg.split_mode == "nested" else list(deploy_extra_tasks)
        meta_manifests.append(
            FitDeployManifest(
                name=f"meta_{idx:02d}",
                fit_tasks=list(fit_tasks),
                deploy_tasks=deploy_tasks,
            )
        )
    heldout_manifests: list[FitDeployManifest] = []
    for idx in range(cfg.heldout_manifest_count):
        fit_tasks, cursor = _take_partition_slice(
            specs,
            cursor=cursor,
            count=counts["heldout_fit"],
            label=f"heldout_{idx:02d}_fit",
        )
        deploy_extra_tasks, cursor = _take_partition_slice(
            specs,
            cursor=cursor,
            count=counts["heldout_deploy_extra"],
            label=f"heldout_{idx:02d}_deploy_extra",
        )
        deploy_tasks = list(fit_tasks) + list(deploy_extra_tasks) if cfg.split_mode == "nested" else list(deploy_extra_tasks)
        heldout_manifests.append(
            FitDeployManifest(
                name=f"heldout_{idx:02d}",
                fit_tasks=list(fit_tasks),
                deploy_tasks=deploy_tasks,
            )
        )
    print(
        f"  task bank partitioned | consumed={cursor} | remaining={len(specs) - cursor}",
        flush=True,
    )
    return TaskManifestSuite(
        pretrain_train=pretrain_train,
        meta_manifests=meta_manifests,
        heldout_manifests=heldout_manifests,
    )


def _build_manifest_suite_from_bank_slip_grouped(
    cfg: RLQuantConfig,
    *,
    seed: int,
    bank_info: TaskBankInfo,
) -> TaskManifestSuite:
    """One manifest per slip level instead of IID slices of one pool.

    Records are grouped by their slip_value; each group becomes exactly one
    meta or heldout manifest, so manifests differ in transition dynamics and
    the cross-manifest forecast carries real signal. Heldout levels are drawn
    from `seed`, so each training seed holds out different dynamics. Pretrain
    tasks come only from meta-level leftovers: heldout dynamics stay fully
    unseen during training.
    """
    counts = _required_manifest_counts(cfg)
    if (
        counts["meta_fit"] != counts["heldout_fit"]
        or counts["meta_deploy_extra"] != counts["heldout_deploy_extra"]
    ):
        raise RuntimeError(
            "slip_level manifest grouping requires meta and heldout manifests to share "
            f"fit/deploy sizes, got meta=({counts['meta_fit']}, {counts['meta_deploy_extra']}) "
            f"heldout=({counts['heldout_fit']}, {counts['heldout_deploy_extra']})"
        )
    records = _load_task_bank_records(bank_info)
    groups: dict[float, list[TaskRecord]] = {}
    if cfg.manifest_grouping == "dynamics_id":
        for record in records:
            key = float(int(getattr(record, "dynamics_id", -1)))
            groups.setdefault(key, []).append(record)
        if -1.0 in groups:
            raise RuntimeError(
                f"dynamics_id grouping found {len(groups[-1.0])} records without a "
                "dynamics_id; the bank was not built in dynamics mode"
            )
    else:
        for record in records:
            # slip_value round-trips through a float32 grid; 6 decimals separates
            # levels >= 1e-5 apart while absorbing float32 rounding.
            slip = round(float(getattr(record, "slip_value", 0.0)), 6)
            groups.setdefault(slip, []).append(record)
    levels = sorted(groups)
    expected_levels = cfg.meta_manifest_count + cfg.heldout_manifest_count
    if len(levels) != expected_levels:
        raise RuntimeError(
            f"{cfg.manifest_grouping} grouping expects {expected_levels} distinct groups "
            f"(meta {cfg.meta_manifest_count} + heldout {cfg.heldout_manifest_count}); "
            f"bank has {len(levels)}: {levels}"
        )
    sizes_by_level = {level: len(groups[level]) for level in levels}
    if len(set(sizes_by_level.values())) != 1:
        raise RuntimeError(f"slip_level groups are unevenly sized: {sizes_by_level}")
    group_size = sizes_by_level[levels[0]]
    per_manifest = counts["meta_fit"] + counts["meta_deploy_extra"]
    if group_size < per_manifest:
        raise RuntimeError(
            f"slip_level group size {group_size} < fit+deploy requirement {per_manifest}"
        )
    leftover_per_level = group_size - per_manifest
    if cfg.meta_manifest_count * leftover_per_level < counts["pretrain_train"]:
        raise RuntimeError(
            "Not enough meta-level leftovers for pretrain_train: "
            f"{cfg.meta_manifest_count} levels x {leftover_per_level} leftovers < "
            f"{counts['pretrain_train']}"
        )

    rng = random.Random(seed)
    heldout_level_set = (
        set(rng.sample(levels, cfg.heldout_manifest_count))
        if cfg.heldout_manifest_count > 0
        else set()
    )
    meta_levels = [level for level in levels if level not in heldout_level_set]
    heldout_levels = [level for level in levels if level in heldout_level_set]

    def _group_label(level: float) -> str:
        if cfg.manifest_grouping == "dynamics_id":
            sample = groups[level][0]
            return (
                f"dyn{int(level):02d}"
                f"_slip{float(getattr(sample, 'slip_value', 0.0)):.3f}"
                f"_wind{int(getattr(sample, 'wind_dir', 0))}x{float(getattr(sample, 'wind_strength', 0.0)):.3f}"
            )
        return f"slip{level:.3f}"

    def _group_slip(level: float) -> float:
        if cfg.manifest_grouping == "dynamics_id":
            return round(float(getattr(groups[level][0], "slip_value", 0.0)), 6)
        return level

    def _slice_level(level: float) -> tuple[list[TaskRecord], list[TaskRecord], list[TaskRecord]]:
        group = list(groups[level])
        random.Random(seed * 1_000_003 + levels.index(level)).shuffle(group)
        cursor = 0
        fit_tasks, cursor = _take_partition_slice(
            group, cursor=cursor, count=counts["meta_fit"], label=f"{_group_label(level)}_fit"
        )
        deploy_extra, cursor = _take_partition_slice(
            group, cursor=cursor, count=counts["meta_deploy_extra"], label=f"{_group_label(level)}_deploy_extra"
        )
        leftovers = list(group[cursor:])
        deploy_tasks = (
            list(fit_tasks) + list(deploy_extra) if cfg.split_mode == "nested" else list(deploy_extra)
        )
        return list(fit_tasks), deploy_tasks, leftovers

    meta_manifests: list[FitDeployManifest] = []
    leftovers_by_level: dict[float, list[TaskRecord]] = {}
    for idx, level in enumerate(meta_levels):
        fit_tasks, deploy_tasks, leftovers = _slice_level(level)
        leftovers_by_level[level] = leftovers
        meta_manifests.append(
            FitDeployManifest(
                name=f"meta_{idx:02d}_{_group_label(level)}",
                fit_tasks=fit_tasks,
                deploy_tasks=deploy_tasks,
                slip_level=_group_slip(level),
            )
        )
    heldout_manifests: list[FitDeployManifest] = []
    for idx, level in enumerate(heldout_levels):
        fit_tasks, deploy_tasks, _ = _slice_level(level)
        heldout_manifests.append(
            FitDeployManifest(
                name=f"heldout_{idx:02d}_{_group_label(level)}",
                fit_tasks=fit_tasks,
                deploy_tasks=deploy_tasks,
                slip_level=_group_slip(level),
            )
        )

    pretrain_order = list(meta_levels)
    rng.shuffle(pretrain_order)
    pretrain_train: list[TaskRecord] = []
    depth = 0
    while len(pretrain_train) < counts["pretrain_train"]:
        progressed = False
        for level in pretrain_order:
            leftovers = leftovers_by_level[level]
            if depth < len(leftovers):
                pretrain_train.append(leftovers[depth])
                progressed = True
                if len(pretrain_train) >= counts["pretrain_train"]:
                    break
        if not progressed:
            raise RuntimeError("Ran out of meta-level leftovers while filling pretrain_train")
        depth += 1

    print(
        f"  {cfg.manifest_grouping} manifest grouping "
        f"| levels={len(levels)} | group_size={group_size} "
        f"| heldout={[ _group_label(level) for level in heldout_levels]} "
        f"| pretrain={len(pretrain_train)} from meta leftovers",
        flush=True,
    )
    return TaskManifestSuite(
        pretrain_train=pretrain_train,
        meta_manifests=meta_manifests,
        heldout_manifests=heldout_manifests,
    )


def _skipped_support_audit_for_bank(bank_info: TaskBankInfo) -> dict[str, object]:
    return {
        "skipped": True,
        "reason": "prebuilt_task_bank",
        "bank": bank_info.summary(),
    }


def _build_fit_deploy_manifest(
    cfg: RLQuantConfig,
    *,
    name: str,
    fit_slices: Sequence[FamilySlice],
    deploy_slices: Sequence[FamilySlice],
    fit_seed: int,
    deploy_seed: int,
    gen_cfg: GridworldGeneratorConfig,
    global_seen: set[str],
) -> FitDeployManifest:
    print(f"  sampling {name} fit tasks ...", flush=True)
    fit_tasks = sample_unique_task_specs(
        split_name=f"{name}_fit",
        slices=fit_slices,
        cfg=gen_cfg,
        seed=fit_seed,
        globally_seen_hashes=global_seen,
    )
    print(f"  sampling {name} deploy tasks ...", flush=True)
    deploy_extra_tasks = sample_unique_task_specs(
        split_name=f"{name}_deploy_extra",
        slices=deploy_slices,
        cfg=gen_cfg,
        seed=deploy_seed,
        globally_seen_hashes=global_seen,
    )
    if cfg.split_mode == "nested":
        deploy_tasks = list(fit_tasks) + list(deploy_extra_tasks)
    elif cfg.split_mode == "disjoint":
        deploy_tasks = list(deploy_extra_tasks)
    else:
        raise ValueError(f"Unsupported split_mode: {cfg.split_mode}")
    return FitDeployManifest(
        name=name,
        fit_tasks=list(fit_tasks),
        deploy_tasks=deploy_tasks,
    )


def _build_manifest_suite(cfg: RLQuantConfig, *, seed: int) -> TaskManifestSuite:
    slices = _default_manifest_slices(cfg)
    gen_cfg = _make_generator_cfg(cfg)
    global_seen: set[str] = set()
    print("  sampling pretrain_train tasks ...", flush=True)
    pretrain = sample_unique_task_specs(
        split_name="pretrain_train",
        slices=slices["pretrain_train"],
        cfg=gen_cfg,
        seed=seed + 11,
        globally_seen_hashes=global_seen,
    )

    meta_manifests: list[FitDeployManifest] = []
    for idx in range(cfg.meta_manifest_count):
        meta_manifests.append(
            _build_fit_deploy_manifest(
                cfg,
                name=f"meta_{idx:02d}",
                fit_slices=slices["meta_fit"],
                deploy_slices=slices["meta_deploy_extra"],
                fit_seed=seed + 1000 + idx * 37,
                deploy_seed=seed + 2000 + idx * 37,
                gen_cfg=gen_cfg,
                global_seen=global_seen,
            )
        )

    heldout_manifests: list[FitDeployManifest] = []
    for idx in range(cfg.heldout_manifest_count):
        heldout_manifests.append(
            _build_fit_deploy_manifest(
                cfg,
                name=f"heldout_{idx:02d}",
                fit_slices=slices["heldout_fit"],
                deploy_slices=slices["heldout_deploy_extra"],
                fit_seed=seed + 5000 + idx * 37,
                deploy_seed=seed + 6000 + idx * 37,
                gen_cfg=gen_cfg,
                global_seen=global_seen,
            )
        )
    print("  manifest suite ready", flush=True)
    return TaskManifestSuite(
        pretrain_train=list(pretrain),
        meta_manifests=meta_manifests,
        heldout_manifests=heldout_manifests,
    )


def _manifest_reference_task(manifest_suite: TaskManifestSuite, *, cfg: RLQuantConfig) -> GridworldTask:
    gen_cfg = _make_generator_cfg(cfg)

    def _materialize_reference(task: TaskRecord) -> GridworldTask:
        if isinstance(task, GridworldTask):
            return task
        if isinstance(task, GridworldTaskSpec):
            return materialize_task_from_spec(
                task,
                cfg=gen_cfg,
                split_name="reference",
                task_id=task.canonical_hash,
                include_slip_in_task_vector=cfg.task_vector_slip,
                blind_dynamics=cfg.blind_dynamics,
            )
        return materialize_task_from_trap_layout_record(
            task,
            cfg=gen_cfg,
            split_name="reference",
            task_id=task.canonical_hash,
            include_slip_in_task_vector=cfg.task_vector_slip,
            blind_dynamics=cfg.blind_dynamics,
        )

    if manifest_suite.pretrain_train:
        return _materialize_reference(manifest_suite.pretrain_train[0])
    for manifest in list(manifest_suite.meta_manifests) + list(manifest_suite.heldout_manifests):
        if manifest.fit_tasks:
            return _materialize_reference(manifest.fit_tasks[0])
        if manifest.deploy_tasks:
            return _materialize_reference(manifest.deploy_tasks[0])
    raise ValueError("Manifest suite contains no tasks")


def _meta_aux_tasks(manifest_suite: TaskManifestSuite, cfg: RLQuantConfig) -> list[TaskRecord]:
    if cfg.meta_aux_task_source == "pretrain_train":
        return list(manifest_suite.pretrain_train)
    if cfg.meta_aux_task_source == "meta_fit":
        return [
            task
            for manifest in manifest_suite.meta_manifests
            for task in manifest.fit_tasks
        ]
    raise ValueError(f"Unsupported meta_aux_task_source: {cfg.meta_aux_task_source}")


def _pretrain_tasks(manifest_suite: TaskManifestSuite, cfg: RLQuantConfig) -> list[TaskRecord]:
    """Resolve the task pool used in the pretraining loop.

    Default ('pretrain_train') is the canonical path. The other modes power the
    SFT-style baselines (M1, M3): they swap in tasks drawn from the meta
    manifests so REINFORCE descends regret directly on the same pool the meta
    method extrapolates over. Tasks are de-duplicated by key to avoid double-
    weighting tasks that appear in both fit and deploy lists.
    """
    if cfg.pretrain_task_source == "pretrain_train":
        return list(manifest_suite.pretrain_train)
    seen: set[str] = set()
    out: list[TaskRecord] = []
    for manifest in manifest_suite.meta_manifests:
        if cfg.pretrain_task_source == "meta_union":
            iterables = (manifest.fit_tasks, manifest.deploy_tasks)
        elif cfg.pretrain_task_source == "meta_deploy":
            iterables = (manifest.deploy_tasks,)
        else:
            raise ValueError(f"Unsupported pretrain_task_source: {cfg.pretrain_task_source}")
        for task_iter in iterables:
            for task in task_iter:
                key = _task_key(task)
                if key in seen:
                    continue
                seen.add(key)
                out.append(task)
    if not out:
        raise ValueError(
            f"pretrain_task_source={cfg.pretrain_task_source} resolved to an empty pool"
        )
    return out


def _resolve_optimal_values(
    tasks: Sequence[GridworldTask],
    batch: dict[str, torch.Tensor],
    *,
    cfg: RLQuantConfig,
) -> dict[str, torch.Tensor]:
    if torch.isnan(batch["optimal_values"]).any():
        computed = optimal_values_for_batch(
            batch,
            horizon=cfg.horizon,
            gamma=cfg.gamma,
        )
        batch["optimal_values"] = computed
        for task, value in zip(tasks, computed.detach().cpu().tolist()):
            task.optimal_value = float(value)
    return batch


def _materialize_records(records: Sequence[TaskRecord], *, cfg: RLQuantConfig, split_name: str) -> list[GridworldTask]:
    return _materialize_records_cached(
        records,
        cfg=cfg,
        split_name=split_name,
        cache=None,
    )


def _materialize_records_cached(
    records: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
    split_name: str,
    cache: dict[str, GridworldTask] | None,
) -> list[GridworldTask]:
    gen_cfg = _make_generator_cfg(cfg)
    out: list[GridworldTask] = []
    for idx, record in enumerate(records):
        if isinstance(record, GridworldTask):
            out.append(record)
        else:
            cached = None if cache is None else cache.get(record.canonical_hash)
            if cached is None:
                if isinstance(record, GridworldTaskSpec):
                    materialized = materialize_task_from_spec(
                        record,
                        cfg=gen_cfg,
                        split_name=split_name,
                        task_id=f"{split_name}:{idx}:{record.canonical_hash}",
                        include_slip_in_task_vector=cfg.task_vector_slip,
                        blind_dynamics=cfg.blind_dynamics,
                    )
                else:
                    materialized = materialize_task_from_trap_layout_record(
                        record,
                        cfg=gen_cfg,
                        split_name=split_name,
                        task_id=f"{split_name}:{idx}:{record.canonical_hash}",
                        include_slip_in_task_vector=cfg.task_vector_slip,
                        blind_dynamics=cfg.blind_dynamics,
                    )
                out.append(materialized)
                if cache is not None:
                    cache[record.canonical_hash] = materialized
            else:
                out.append(
                    replace(
                        cached,
                        split_name=split_name,
                        task_id=f"{split_name}:{idx}:{record.canonical_hash}",
                    )
                )
    return out


def _materialize_manifest_suite(
    manifest_suite: TaskManifestSuite,
    *,
    cfg: RLQuantConfig,
) -> tuple[TaskManifestSuite, dict[str, int]]:
    cache: dict[str, GridworldTask] = {}
    pretrain_train = _materialize_records_cached(
        manifest_suite.pretrain_train,
        cfg=cfg,
        split_name="pretrain_train",
        cache=cache,
    )
    meta_manifests: list[FitDeployManifest] = []
    for manifest in manifest_suite.meta_manifests:
        meta_manifests.append(
            FitDeployManifest(
                name=manifest.name,
                fit_tasks=_materialize_records_cached(
                    manifest.fit_tasks,
                    cfg=cfg,
                    split_name=f"{manifest.name}_fit",
                    cache=cache,
                ),
                deploy_tasks=_materialize_records_cached(
                    manifest.deploy_tasks,
                    cfg=cfg,
                    split_name=f"{manifest.name}_deploy",
                    cache=cache,
                ),
            )
        )
    heldout_manifests: list[FitDeployManifest] = []
    for manifest in manifest_suite.heldout_manifests:
        heldout_manifests.append(
            FitDeployManifest(
                name=manifest.name,
                fit_tasks=_materialize_records_cached(
                    manifest.fit_tasks,
                    cfg=cfg,
                    split_name=f"{manifest.name}_fit",
                    cache=cache,
                ),
                deploy_tasks=_materialize_records_cached(
                    manifest.deploy_tasks,
                    cfg=cfg,
                    split_name=f"{manifest.name}_deploy",
                    cache=cache,
                ),
            )
        )
    materialized_suite = TaskManifestSuite(
        pretrain_train=pretrain_train,
        meta_manifests=meta_manifests,
        heldout_manifests=heldout_manifests,
    )
    counts = {
        "unique_materialized_tasks": len(cache),
        "pretrain_train": len(pretrain_train),
        "meta_fit_total": sum(len(manifest.fit_tasks) for manifest in meta_manifests),
        "meta_deploy_total": sum(len(manifest.deploy_tasks) for manifest in meta_manifests),
        "heldout_fit_total": sum(len(manifest.fit_tasks) for manifest in heldout_manifests),
        "heldout_deploy_total": sum(len(manifest.deploy_tasks) for manifest in heldout_manifests),
    }
    return materialized_suite, counts


def _exact_returns_and_regrets(
    model: DensePolicyNet,
    tasks: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
    require_grad: bool,
    materialization_cache: TaskMaterializationCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not tasks:
        device = next(model.parameters()).device
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return empty, empty
    returns_parts: list[torch.Tensor] = []
    regrets_parts: list[torch.Tensor] = []
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        for batch_records in _batched(tasks, _eval_batch_size(tasks, cfg)):
            batch_tasks = _materialize_records_cached(
                batch_records,
                cfg=cfg,
                split_name="bank",
                cache=materialization_cache,
            )
            batch = prepare_task_batch(batch_tasks, str(next(model.parameters()).device))
            batch = _resolve_optimal_values(batch_tasks, batch, cfg=cfg)
            dense_logits = model(batch["features"], batch["task_vectors"])
            returns = evaluate_dense_policy_exact(
                dense_logits,
                batch,
                horizon=cfg.horizon,
                gamma=cfg.gamma,
            )
            regrets = batch["optimal_values"] - returns
            returns_parts.append(returns)
            regrets_parts.append(regrets)
    return torch.cat(returns_parts, dim=0), torch.cat(regrets_parts, dim=0)


def _meta_coeff_loss(
    fit_regrets: torch.Tensor,
    deploy_regrets: torch.Tensor,
    *,
    cfg: RLQuantConfig,
) -> tuple[torch.Tensor, dict]:
    fit_topk = select_top_k_indices(fit_regrets, cfg.ols_top_k)
    deploy_topk = select_top_k_indices(deploy_regrets, cfg.ols_top_k)
    a_fit, b_fit = differentiable_ols_coefficients(
        fit_regrets,
        fit_topk,
        len(fit_regrets),
        plotting=cfg.plotting_position,
        score_mode=cfg.ols_score_mode,
    )
    a_deploy, b_deploy = differentiable_ols_coefficients(
        deploy_regrets,
        deploy_topk,
        len(deploy_regrets),
        plotting=cfg.plotting_position,
        score_mode=cfg.ols_score_mode,
    )
    if cfg.grad_mode == "fit":
        a_deploy, b_deploy = a_deploy.detach(), b_deploy.detach()
    elif cfg.grad_mode == "eval":
        a_fit, b_fit = a_fit.detach(), b_fit.detach()
    loss = (a_fit - a_deploy) ** 2 + (b_fit - b_deploy) ** 2
    return loss, {
        "fit_a": float(a_fit.detach().item()),
        "fit_b": float(b_fit.detach().item()),
        "deploy_a": float(a_deploy.detach().item()),
        "deploy_b": float(b_deploy.detach().item()),
    }


def _meta_primal_loss(
    fit_regrets: torch.Tensor,
    deploy_regrets: torch.Tensor,
    *,
    cfg: RLQuantConfig,
) -> tuple[torch.Tensor, dict]:
    fit_topk = select_top_k_indices(fit_regrets, cfg.ols_top_k)
    if cfg.meta_loss_mode == "ols_primal_extrapolated":
        deploy_extrap_k = count_extrapolated_eval_points(
            len(fit_regrets),
            len(deploy_regrets),
            plotting=cfg.plotting_position,
        )
        if deploy_extrap_k <= 0:
            device = fit_regrets.device
            nan = float("nan")
            return torch.full((), nan, dtype=torch.float32, device=device), {
                "fit_a": nan,
                "fit_b": nan,
                "deploy_a": nan,
                "deploy_b": nan,
                "weighted_coeff_mse": nan,
                "deploy_residual_mse": nan,
                "primal_mse": nan,
                "deploy_points_used": 0.0,
            }
        deploy_extrap_idx = select_top_k_indices(deploy_regrets, deploy_extrap_k)
        loss, stats = differentiable_ols_extrapolated_primal_loss(
            fit_regrets,
            fit_topk,
            len(fit_regrets),
            deploy_regrets,
            deploy_extrap_idx,
            len(deploy_regrets),
            plotting=cfg.plotting_position,
            score_mode=cfg.ols_score_mode,
            grad_mode=cfg.grad_mode,
            extrapolated_weighting=cfg.extrapolated_weighting,
            improving_only_grad=cfg.improving_only_grad,
            iog_scope=cfg.iog_scope,
        )
        nan = float("nan")
        result = {
            "fit_a": float(stats["fit_a"].detach().item()),
            "fit_b": float(stats["fit_b"].detach().item()),
            "deploy_a": nan,
            "deploy_b": nan,
            "weighted_coeff_mse": nan,
            "deploy_residual_mse": nan,
            "primal_mse": float(stats["primal_mse"].detach().item()),
            "deploy_points_used": float(stats["eval_points_used"].detach().item()),
        }
        if cfg.improving_only_grad:
            result["improving_only_deploy_frac"] = float(stats["improving_only_deploy_frac"].item())
            result["improving_only_fit_frac"] = float(stats["improving_only_fit_frac"].item())
        return loss, result
    deploy_all = torch.argsort(deploy_regrets.detach(), descending=True)
    use_weighted = cfg.meta_loss_mode == "ols_primal_weighted"
    loss, stats = differentiable_ols_primal_loss(
        fit_regrets,
        fit_topk,
        len(fit_regrets),
        deploy_regrets,
        deploy_all,
        len(deploy_regrets),
        plotting=cfg.plotting_position,
        score_mode=cfg.ols_score_mode,
        grad_mode=cfg.grad_mode,
        use_weighted_decomposition=use_weighted,
    )
    return loss, {
        "fit_a": float(stats["fit_a"].detach().item()),
        "fit_b": float(stats["fit_b"].detach().item()),
        "deploy_a": float(stats["eval_a"].detach().item()),
        "deploy_b": float(stats["eval_b"].detach().item()),
        "weighted_coeff_mse": float(stats["weighted_coeff_mse"].detach().item()),
        "deploy_residual_mse": float(stats["eval_residual_mse"].detach().item()),
        "primal_mse": float(stats["primal_mse"].detach().item()),
    }


def _pretrain_reference_returns(
    model: DensePolicyNet,
    tasks: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
    materialization_cache: TaskMaterializationCache | None = None,
) -> dict[str, float]:
    returns, _ = _exact_returns_and_regrets(
        model,
        tasks,
        cfg=cfg,
        require_grad=False,
        materialization_cache=materialization_cache,
    )
    return {
        _task_key(task): float(ret)
        for task, ret in zip(tasks, returns.detach().cpu().tolist())
    }


def _meta_aux_loss_exact(
    model: DensePolicyNet,
    pretrain_tasks: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
    rng: random.Random,
    reference_returns: dict[str, float] | None,
    materialization_cache: TaskMaterializationCache | None = None,
) -> tuple[torch.Tensor, dict]:
    device = next(model.parameters()).device
    zero = torch.zeros((), dtype=torch.float32, device=device)
    if cfg.meta_aux_loss_mode == "none" or cfg.meta_aux_loss_weight <= 0 or not pretrain_tasks:
        return zero, {}
    batch_tasks = _pretrain_batch(pretrain_tasks, cfg, rng)
    returns, regrets = _exact_returns_and_regrets(
        model,
        batch_tasks,
        cfg=cfg,
        require_grad=True,
        materialization_cache=materialization_cache,
    )
    stats = {
        "aux_batch_mean_return": float(returns.detach().mean().item()),
        "aux_batch_mean_regret": float(regrets.detach().mean().item()),
    }
    if cfg.meta_aux_loss_mode == "pretrain_return":
        return -returns.mean(), stats
    if cfg.meta_aux_loss_mode == "anchored_pretrain_return":
        if reference_returns is None:
            raise ValueError("anchored_pretrain_return requires reference_returns")
        ref = torch.tensor(
            [reference_returns[_task_key(task)] for task in batch_tasks],
            dtype=returns.dtype,
            device=returns.device,
        )
        gaps = torch.relu(ref - returns)
        stats["aux_reference_mean_return"] = float(ref.mean().item())
        stats["aux_return_gap_mean"] = float(gaps.detach().mean().item())
        return (gaps**2).mean(), stats
    raise ValueError(f"Unsupported meta_aux_loss_mode: {cfg.meta_aux_loss_mode}")


def _meta_aux_loss_reinforce(
    pretrain_tasks: Sequence[TaskRecord],
    *,
    model: DensePolicyNet,
    cfg: RLQuantConfig,
    rng: random.Random,
    reference_returns: dict[str, float] | None,
    materialization_cache: TaskMaterializationCache | None = None,
    ema_state: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    device = str(next(model.parameters()).device)
    zero = torch.zeros((), dtype=torch.float32, device=device)
    empty = torch.zeros(0, dtype=torch.float32, device=device)
    if cfg.meta_aux_loss_mode == "none" or cfg.meta_aux_loss_weight <= 0 or not pretrain_tasks:
        return zero, empty, empty, {}
    batch_tasks = _materialize_records_cached(
        _pretrain_batch(pretrain_tasks, cfg, rng),
        cfg=cfg,
        split_name="pretrain_aux",
        cache=materialization_cache,
    )
    batch = prepare_task_batch(batch_tasks, device)
    batch = _resolve_optimal_values(batch_tasks, batch, cfg=cfg)
    dense_logits = model(batch["features"], batch["task_vectors"])
    returns, logprobs = sample_policy_rollouts_batched(
        batch_tasks,
        dense_logits,
        batch,
        n_rollouts=cfg.reinforce_rollouts_per_task,
        device=device,
    )
    mean_returns = returns.detach().mean(dim=1)
    mean_regrets = batch["optimal_values"] - mean_returns
    stats = {
        "aux_batch_mean_return": float(mean_returns.mean().item()),
        "aux_batch_mean_regret": float(mean_regrets.mean().item()),
    }
    if cfg.meta_aux_loss_mode == "pretrain_return":
        weights = -torch.ones_like(mean_returns) / max(1, mean_returns.numel())
        aux_value = -mean_returns.mean()
    elif cfg.meta_aux_loss_mode == "anchored_pretrain_return":
        if reference_returns is None:
            raise ValueError("anchored_pretrain_return requires reference_returns")
        ref = torch.tensor(
            [reference_returns[_task_key(task)] for task in batch_tasks],
            dtype=mean_returns.dtype,
            device=mean_returns.device,
        )
        gaps = torch.relu(ref - mean_returns)
        weights = -2.0 * gaps / max(1, mean_returns.numel())
        aux_value = (gaps**2).mean()
        stats["aux_reference_mean_return"] = float(ref.mean().item())
        stats["aux_return_gap_mean"] = float(gaps.mean().item())
    else:
        raise ValueError(f"Unsupported meta_aux_loss_mode: {cfg.meta_aux_loss_mode}")
    advantages = _reinforce_advantages(
        returns,
        cfg=cfg,
        optimal_values=batch["optimal_values"],
        task_keys=[_task_key(task) for task in batch_tasks],
        ema_state=ema_state if ema_state is not None else {},
    )
    aux_pg_loss = (weights.unsqueeze(1).detach() * advantages * logprobs).mean()
    return aux_value, aux_pg_loss, mean_returns, stats


def _sample_fit_subset(tasks: Sequence[TaskRecord], cfg: RLQuantConfig, rng: random.Random) -> list[TaskRecord]:
    if not tasks:
        return []
    if cfg.deterministic_debug:
        return list(tasks)
    min_size = min(max(cfg.meta_fit_subsample_min, cfg.ols_top_k), len(tasks))
    max_size = len(tasks) if cfg.meta_fit_subsample_max <= 0 else min(cfg.meta_fit_subsample_max, len(tasks))
    if max_size < min_size:
        max_size = min_size
    m = rng.randint(min_size, max_size)
    if m >= len(tasks):
        return list(tasks)
    return rng.sample(list(tasks), m)


def _sample_meta_manifest_batch(
    manifests: Sequence[FitDeployManifest],
    cfg: RLQuantConfig,
    rng: random.Random,
) -> list[FitDeployManifest]:
    if not manifests:
        return []
    batch_size = min(max(1, cfg.meta_manifests_per_step), len(manifests))
    if cfg.deterministic_debug or batch_size >= len(manifests):
        return list(manifests[:batch_size])
    return rng.sample(list(manifests), batch_size)


def _task_sublist(tasks: Sequence[TaskRecord], indices: Sequence[int]) -> list[TaskRecord]:
    return [tasks[int(idx)] for idx in indices]


def _mean_scalar_dict(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        out[key] = float(sum(float(row[key]) for row in rows) / len(rows))
    return out


def _selected_loss_indices(length: int, *, device: torch.device) -> torch.LongTensor:
    return torch.arange(length, dtype=torch.long, device=device)


def _meta_coeff_loss_selected(
    fit_regrets_topk: torch.Tensor,
    *,
    fit_n_total: int,
    deploy_regrets_candidates: torch.Tensor,
    deploy_n_total: int,
    cfg: RLQuantConfig,
) -> tuple[torch.Tensor, dict]:
    fit_topk = _selected_loss_indices(len(fit_regrets_topk), device=fit_regrets_topk.device)
    deploy_topk = select_top_k_indices(deploy_regrets_candidates, cfg.ols_top_k)
    a_fit, b_fit = differentiable_ols_coefficients(
        fit_regrets_topk,
        fit_topk,
        fit_n_total,
        plotting=cfg.plotting_position,
        score_mode=cfg.ols_score_mode,
    )
    a_deploy, b_deploy = differentiable_ols_coefficients(
        deploy_regrets_candidates,
        deploy_topk,
        deploy_n_total,
        plotting=cfg.plotting_position,
        score_mode=cfg.ols_score_mode,
    )
    if cfg.grad_mode == "fit":
        a_deploy, b_deploy = a_deploy.detach(), b_deploy.detach()
    elif cfg.grad_mode == "eval":
        a_fit, b_fit = a_fit.detach(), b_fit.detach()
    loss = (a_fit - a_deploy) ** 2 + (b_fit - b_deploy) ** 2
    return loss, {
        "fit_a": float(a_fit.detach().item()),
        "fit_b": float(b_fit.detach().item()),
        "deploy_a": float(a_deploy.detach().item()),
        "deploy_b": float(b_deploy.detach().item()),
    }


def _meta_primal_loss_bootstrap(
    union_regrets: torch.Tensor,
    n_fit: int,
    n_deploy: int,
    *,
    cfg: RLQuantConfig,
    rng_seed: int,
    calibration_alpha: float | None = None,
    calibration_beta: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Bootstrap-permutation primal loss in score-mode for RL regrets.

    ``union_regrets`` must have shape ``[n_fit + n_deploy]`` with grad
    attached on entries the caller wants to influence (caller decides
    whether that's the full union or a stop-gradient subset). Returns the
    permutation-averaged loss and a coefficient/metric dict matching
    ``_meta_primal_loss``'s return shape.
    """
    n_union = int(union_regrets.shape[0])
    assert n_union == n_fit + n_deploy, (n_union, n_fit, n_deploy)
    eval_extrap_k = count_extrapolated_eval_points(
        n_fit, n_deploy, plotting=cfg.plotting_position,
    )
    nan = float("nan")
    if eval_extrap_k <= 0:
        device = union_regrets.device
        return torch.full((), nan, dtype=torch.float32, device=device), {
            "fit_a": nan,
            "fit_b": nan,
            "deploy_a": nan,
            "deploy_b": nan,
            "weighted_coeff_mse": nan,
            "deploy_residual_mse": nan,
            "primal_mse": nan,
            "deploy_points_used": 0.0,
        }

    fit_partitions, eval_partitions, _needed = plan_bootstrap_permutations(
        n_union=n_union,
        n_fit=n_fit,
        n_eval=n_deploy,
        n_permutations=cfg.bootstrap_permutations,
        rng_seed=rng_seed,
        fit_top_k=cfg.ols_top_k,
        eval_extrap_k=eval_extrap_k,
        union_values_detached=union_regrets.detach(),
        grad_mode=cfg.grad_mode,
    )
    loss, stats = differentiable_ols_extrapolated_primal_loss_bootstrap(
        union_regrets,
        fit_partitions,
        eval_partitions,
        n_fit=n_fit,
        n_eval=n_deploy,
        fit_top_k=cfg.ols_top_k,
        plotting=cfg.plotting_position,
        score_mode=cfg.ols_score_mode,
        grad_mode=cfg.grad_mode,
        extrapolated_weighting=cfg.extrapolated_weighting,
        improving_only_grad=cfg.improving_only_grad,
        iog_scope=cfg.iog_scope,
        calibration_alpha=calibration_alpha,
        calibration_beta=calibration_beta,
    )
    result = {
        "fit_a": float(stats["fit_a"].detach().item()),
        "fit_b": float(stats["fit_b"].detach().item()),
        "deploy_a": nan,
        "deploy_b": nan,
        "weighted_coeff_mse": nan,
        "deploy_residual_mse": nan,
        "primal_mse": float(stats["primal_mse"].detach().item()),
        "deploy_points_used": float(eval_extrap_k),
        "cal_pred_worst": float(stats["cal_pred_worst"].detach().item()),
        "cal_actual_worst": float(stats["cal_actual_worst"].detach().item()),
    }
    if cfg.improving_only_grad and "improving_only_deploy_frac" in stats:
        result["improving_only_deploy_frac"] = float(stats["improving_only_deploy_frac"].item())
        result["improving_only_fit_frac"] = float(stats["improving_only_fit_frac"].item())
    return loss, result


def _meta_primal_loss_selected(
    fit_regrets_topk: torch.Tensor,
    *,
    fit_n_total: int,
    deploy_regrets_candidates: torch.Tensor,
    deploy_n_total: int,
    cfg: RLQuantConfig,
    calibration_alpha: float | None = None,
    calibration_beta: float | None = None,
) -> tuple[torch.Tensor, dict]:
    fit_topk = _selected_loss_indices(len(fit_regrets_topk), device=fit_regrets_topk.device)
    if cfg.meta_loss_mode == "ols_primal_extrapolated":
        deploy_extrap_k = count_extrapolated_eval_points(
            fit_n_total,
            deploy_n_total,
            plotting=cfg.plotting_position,
        )
        if deploy_extrap_k <= 0:
            device = fit_regrets_topk.device
            nan = float("nan")
            return torch.full((), nan, dtype=torch.float32, device=device), {
                "fit_a": nan,
                "fit_b": nan,
                "deploy_a": nan,
                "deploy_b": nan,
                "weighted_coeff_mse": nan,
                "deploy_residual_mse": nan,
                "primal_mse": nan,
                "deploy_points_used": 0.0,
            }
        deploy_k = min(int(deploy_extrap_k), int(deploy_regrets_candidates.shape[0]))
        deploy_extrap_idx = select_top_k_indices(deploy_regrets_candidates, deploy_k)
        loss, stats = differentiable_ols_extrapolated_primal_loss(
            fit_regrets_topk,
            fit_topk,
            fit_n_total,
            deploy_regrets_candidates,
            deploy_extrap_idx,
            deploy_n_total,
            plotting=cfg.plotting_position,
            score_mode=cfg.ols_score_mode,
            grad_mode=cfg.grad_mode,
            extrapolated_weighting=cfg.extrapolated_weighting,
            improving_only_grad=cfg.improving_only_grad,
            iog_scope=cfg.iog_scope,
            calibration_alpha=calibration_alpha,
            calibration_beta=calibration_beta,
        )
        nan = float("nan")
        result = {
            "fit_a": float(stats["fit_a"].detach().item()),
            "fit_b": float(stats["fit_b"].detach().item()),
            "deploy_a": nan,
            "deploy_b": nan,
            "weighted_coeff_mse": nan,
            "deploy_residual_mse": nan,
            "primal_mse": float(stats["primal_mse"].detach().item()),
            "deploy_points_used": float(stats["eval_points_used"].detach().item()),
            "cal_pred_worst": float(stats["cal_pred_worst"].detach().item()),
            "cal_actual_worst": float(stats["cal_actual_worst"].detach().item()),
        }
        if cfg.improving_only_grad:
            result["improving_only_deploy_frac"] = float(stats["improving_only_deploy_frac"].item())
            result["improving_only_fit_frac"] = float(stats["improving_only_fit_frac"].item())
        return loss, result
    deploy_all = torch.argsort(deploy_regrets_candidates.detach(), descending=True)
    use_weighted = cfg.meta_loss_mode == "ols_primal_weighted"
    loss, stats = differentiable_ols_primal_loss(
        fit_regrets_topk,
        fit_topk,
        fit_n_total,
        deploy_regrets_candidates,
        deploy_all,
        deploy_n_total,
        plotting=cfg.plotting_position,
        score_mode=cfg.ols_score_mode,
        grad_mode=cfg.grad_mode,
        use_weighted_decomposition=use_weighted,
    )
    return loss, {
        "fit_a": float(stats["fit_a"].detach().item()),
        "fit_b": float(stats["fit_b"].detach().item()),
        "deploy_a": float(stats["eval_a"].detach().item()),
        "deploy_b": float(stats["eval_b"].detach().item()),
        "weighted_coeff_mse": float(stats["weighted_coeff_mse"].detach().item()),
        "deploy_residual_mse": float(stats["eval_residual_mse"].detach().item()),
        "primal_mse": float(stats["primal_mse"].detach().item()),
    }


def _deploy_cache_enabled(cfg: RLQuantConfig, deploy_task_count: int) -> bool:
    return (
        cfg.meta_deploy_cache_top_b > 0
        and cfg.meta_deploy_cache_refresh_interval > 0
        and cfg.meta_deploy_cache_top_b < int(deploy_task_count)
    )


def _pretrain_exact(
    model: DensePolicyNet,
    optimizer: torch.optim.Optimizer,
    tasks: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
    rng: random.Random,
    materialization_cache: TaskMaterializationCache | None = None,
    start_step: int = 0,
    history: list[dict] | None = None,
    checkpoint_callback=None,
) -> list[dict]:
    history = list(history) if history is not None else []
    elapsed_offset = float(history[-1]["elapsed_sec"]) if history else 0.0
    phase_start = time.perf_counter()
    for step in range(start_step, cfg.pretrain_steps):
        batch_tasks = _pretrain_batch(tasks, cfg, rng)
        optimizer.zero_grad(set_to_none=True)
        returns, regrets = _exact_returns_and_regrets(
            model,
            batch_tasks,
            cfg=cfg,
            require_grad=True,
            materialization_cache=materialization_cache,
        )
        loss = -returns.mean()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        entry = {
            "phase": "pretrain",
            "step": step,
            "elapsed_sec": float(elapsed_offset + (time.perf_counter() - phase_start)),
            "loss": float(loss.item()),
            "mean_return": float(returns.detach().mean().item()),
            "mean_regret": float(regrets.detach().mean().item()),
        }
        history.append(entry)
        if checkpoint_callback is not None and cfg.save_every > 0:
            if ((step + 1) % cfg.save_every == 0) or (step + 1 == cfg.pretrain_steps):
                checkpoint_callback(step + 1, history)
        if step % cfg.log_every == 0:
            print(
                f"  pretrain step {step:4d} | loss={entry['loss']:.6f} "
                f"| return={entry['mean_return']:.4f} | regret={entry['mean_regret']:.4f} "
                f"| elapsed={_format_elapsed_short(entry['elapsed_sec'])}",
                flush=True,
            )
    return history


def _reinforce_advantages(
    returns: torch.Tensor,
    *,
    cfg: RLQuantConfig,
    optimal_values: torch.Tensor | None = None,
    task_keys: Sequence[str] | None = None,
    ema_state: dict[str, float] | None = None,
) -> torch.Tensor:
    """Detached REINFORCE advantages [T, R] under cfg.reinforce_baseline.

    'rollout_mean' is bit-identical to the historical inline expression
    (per-task mean over this step's rollouts); it is identically zero at
    n_rollouts=1, which startup validation rejects. 'optimal_value' subtracts
    the task's stored V* (a constant w.r.t. the sampled actions, hence a valid
    baseline; the advantage is minus the sampled regret). 'ema' subtracts a
    per-task running mean of returns maintained in `ema_state`; the state is
    updated AFTER the advantage is computed so the baseline never depends on
    the current rollouts (first visit falls back to this step's rollout mean,
    keeping the mode oracle-free at the cost of a one-step warmup).
    """
    returns_det = returns.detach()
    if cfg.reinforce_baseline == "rollout_mean":
        return returns_det - returns_det.mean(dim=1, keepdim=True)
    if cfg.reinforce_baseline == "optimal_value":
        if optimal_values is None:
            raise ValueError("reinforce_baseline='optimal_value' requires optimal_values")
        return returns_det - optimal_values.detach().unsqueeze(1)
    if cfg.reinforce_baseline == "ema":
        if task_keys is None or ema_state is None:
            raise ValueError("reinforce_baseline='ema' requires task_keys and ema_state")
        mean_returns = returns_det.mean(dim=1)
        baselines = torch.tensor(
            [
                ema_state.get(key, float(mean_returns[i]))
                for i, key in enumerate(task_keys)
            ],
            dtype=returns_det.dtype,
            device=returns_det.device,
        )
        decay = float(cfg.reinforce_ema_decay)
        for i, key in enumerate(task_keys):
            fresh = float(mean_returns[i])
            ema_state[key] = decay * ema_state[key] + (1.0 - decay) * fresh if key in ema_state else fresh
        return returns_det - baselines.unsqueeze(1)
    raise ValueError(f"Unsupported reinforce_baseline: {cfg.reinforce_baseline}")


def _pretrain_reinforce(
    model: DensePolicyNet,
    optimizer: torch.optim.Optimizer,
    tasks: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
    rng: random.Random,
    materialization_cache: TaskMaterializationCache | None = None,
    start_step: int = 0,
    history: list[dict] | None = None,
    checkpoint_callback=None,
    ema_state: dict[str, float] | None = None,
) -> list[dict]:
    history = list(history) if history is not None else []
    device = str(next(model.parameters()).device)
    elapsed_offset = float(history[-1]["elapsed_sec"]) if history else 0.0
    phase_start = time.perf_counter()
    # Persistent when the caller passes a dict (resume checkpoints carry it).
    ema_state = {} if ema_state is None else ema_state
    for step in range(start_step, cfg.pretrain_steps):
        batch_tasks = _materialize_records_cached(
            _pretrain_batch(tasks, cfg, rng),
            cfg=cfg,
            split_name="pretrain",
            cache=materialization_cache,
        )
        optimizer.zero_grad(set_to_none=True)
        batch = prepare_task_batch(batch_tasks, device)
        batch = _resolve_optimal_values(batch_tasks, batch, cfg=cfg)
        dense_logits = model(batch["features"], batch["task_vectors"])
        returns, logprobs = sample_policy_rollouts_batched(
            batch_tasks,
            dense_logits,
            batch,
            n_rollouts=cfg.reinforce_rollouts_per_task,
            device=device,
        )
        advantages = _reinforce_advantages(
            returns,
            cfg=cfg,
            optimal_values=batch["optimal_values"],
            task_keys=[_task_key(task) for task in batch_tasks],
            ema_state=ema_state,
        )
        loss = -(advantages * logprobs).mean()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        mean_returns = returns.detach().mean(dim=1)
        mean_regrets = batch["optimal_values"] - mean_returns
        entry = {
            "phase": "pretrain",
            "step": step,
            "elapsed_sec": float(elapsed_offset + (time.perf_counter() - phase_start)),
            "loss": float(loss.item()),
            "mean_return": float(mean_returns.mean().item()),
            "mean_regret": float(mean_regrets.mean().item()),
        }
        history.append(entry)
        if checkpoint_callback is not None and cfg.save_every > 0:
            if ((step + 1) % cfg.save_every == 0) or (step + 1 == cfg.pretrain_steps):
                checkpoint_callback(step + 1, history)
        if step % cfg.log_every == 0:
            print(
                f"  pretrain step {step:4d} | loss={entry['loss']:.6f} "
                f"| return={entry['mean_return']:.4f} | regret={entry['mean_regret']:.4f} "
                f"| elapsed={_format_elapsed_short(entry['elapsed_sec'])}",
                flush=True,
            )
    return history


def _stage_calibration_update(
    cal_buffer: AffineCalibrationBuffer | None,
    cal_updates_local: list[tuple[str, float, float]],
    group_id: str,
    coeffs: dict,
) -> None:
    """Queue a manifest's raw worst-rank (forecast, actual) pair for the buffer.

    Uses the UNCALIBRATED forecast and observed worst actual so the calibrator
    learns the same raw_forecast -> actual map applied at deployment. Applied to
    the buffer after the full step, so each manifest's leave-one-group-out fit
    uses the previous step's state.
    """
    if cal_buffer is None:
        return
    q = coeffs.get("cal_pred_worst")
    y = coeffs.get("cal_actual_worst")
    if q is None or y is None:
        return
    q = float(q.item()) if torch.is_tensor(q) else float(q)
    y = float(y.item()) if torch.is_tensor(y) else float(y)
    if math.isfinite(q) and math.isfinite(y):
        cal_updates_local.append((group_id, q, y))


def _meta_train_exact(
    model: DensePolicyNet,
    optimizer: torch.optim.Optimizer,
    pretrain_tasks: Sequence[TaskRecord],
    meta_manifests: Sequence[FitDeployManifest],
    *,
    cfg: RLQuantConfig,
    rng: random.Random,
    reference_returns: dict[str, float] | None,
    materialization_cache: TaskMaterializationCache | None = None,
    start_step: int = 0,
    history: list[dict] | None = None,
    deploy_caches: dict[str, DeploySubsetCache] | None = None,
    union_caches: dict[str, UnionTopCache] | None = None,
    checkpoint_callback=None,
    iog_penalty_baseline_regrets: dict[str, float] | None = None,
) -> tuple[list[dict], dict]:
    history = list(history) if history is not None else []
    deploy_caches = dict(deploy_caches) if deploy_caches is not None else {}
    union_caches = dict(union_caches) if union_caches is not None else {}
    elapsed_offset = float(history[-1]["elapsed_sec"]) if history else 0.0
    phase_start = time.perf_counter()

    # Calibration-aware training: leave-one-group-out affine calibrator over a
    # persistent per-manifest buffer of worst-rank (forecast, actual) pairs.
    # Only meta_manifests_per_step manifests are seen per step, so the buffer
    # (spanning many steps) is what gives the calibrator a full-train-sized
    # sample rather than a high-variance few-manifest fit.
    cal_buffer: AffineCalibrationBuffer | None = None
    if cfg.calibration_aware:
        if cfg.meta_loss_mode != "ols_primal_extrapolated":
            raise ValueError(
                "calibration_aware requires meta_loss_mode='ols_primal_extrapolated' "
                f"(got {cfg.meta_loss_mode!r})."
            )
        cal_buffer = AffineCalibrationBuffer(
            mode=cfg.calibration_mode,
            min_points=cfg.calibration_min_points,
            ridge_alpha=cfg.calibration_ridge_alpha,
            ema=cfg.calibration_ema,
        )

    for step in range(start_step, cfg.meta_steps):
        optimizer.zero_grad(set_to_none=True)
        selected_manifests = _sample_meta_manifest_batch(meta_manifests, cfg, rng)
        if not selected_manifests:
            raise ValueError("meta_manifests must be non-empty for meta training")
        meta_losses: list[torch.Tensor] = []
        coeff_rows: list[dict[str, float]] = []
        fit_mean_returns: list[float] = []
        fit_mean_regrets: list[float] = []
        fit_top1_regrets: list[float] = []
        deploy_mean_returns: list[float] = []
        deploy_mean_regrets: list[float] = []
        deploy_top1_regrets: list[float] = []
        selected_fit_counts: list[int] = []
        selected_deploy_candidate_counts: list[int] = []
        selected_deploy_refreshes = 0
        selected_cache_ages: list[int] = []
        # Calibration-aware: this step's fresh (manifest, raw_forecast, actual)
        # worst-rank contributions, applied to the buffer at step end.
        cal_updates_local: list[tuple[str, float, float]] = []
        for manifest in selected_manifests:
            fit_subset = _sample_fit_subset(manifest.fit_tasks, cfg, rng)
            fit_n_total = len(fit_subset)
            deploy_n_total_pre = len(manifest.deploy_tasks)

            # Leave-one-group-out calibrator for THIS manifest from the buffer's
            # other-manifest statistics (stop-gradient floats; None = no cal).
            cal_alpha: float | None = None
            cal_beta: float | None = None
            if cal_buffer is not None:
                cal_alpha, cal_beta = cal_buffer.loo_coeffs(manifest.name)

            # ----------------------------------------------------------
            # Bootstrap-permutation branch (EXACT path).
            # Treat fit_subset ∪ deploy_tasks as a single union pool;
            # average the meta-loss over many fit/deploy partitions per
            # step. Optional partition-invariant union top-C cache:
            # cfg.union_cache_top_c > 0 caches top-C indices across the
            # union and only re-scores them between refreshes (lazy fit
            # fallback if any permutation has |F_p ∩ cached| < ols_top_k).
            # ----------------------------------------------------------
            if (
                cfg.bootstrap_permutations >= 1
                and cfg.meta_loss_mode == "ols_primal_extrapolated"
                and fit_n_total > 0
                and deploy_n_total_pre > 0
            ):
                eval_extrap_k_b = count_extrapolated_eval_points(
                    fit_n_total, deploy_n_total_pre, plotting=cfg.plotting_position,
                )
                if eval_extrap_k_b <= 0:
                    continue
                union_tasks_b = list(fit_subset) + list(manifest.deploy_tasks)
                n_union_b = len(union_tasks_b)
                rng_seed_b = rng.randrange(2**31)

                # Decide whether to use the cache
                _ucache_b = union_caches.get(manifest.name)
                _ucache_on_b = (
                    cfg.union_cache_top_c > 0
                    and cfg.union_cache_refresh_interval > 0
                    and cfg.union_cache_top_c < n_union_b
                )
                _ucache_hit_b = (
                    _ucache_on_b and _ucache_b is not None
                    and _ucache_b.is_valid(
                        step=step,
                        n_fit=fit_n_total,
                        n_union=n_union_b,
                        refresh_interval=cfg.union_cache_refresh_interval,
                    )
                )

                if _ucache_hit_b:
                    # Cache hit: score only cached indices
                    assert _ucache_b is not None
                    cached_indices_b = list(_ucache_b.cached_union_indices)
                    cached_tasks_b = [union_tasks_b[i] for i in cached_indices_b]
                    cached_returns_b, cached_regrets_b = _exact_returns_and_regrets(
                        model,
                        cached_tasks_b,
                        cfg=cfg,
                        require_grad=False,
                        materialization_cache=materialization_cache,
                    )
                    union_returns_det_b = torch.full(
                        (n_union_b,), float("-inf"),
                        dtype=cached_returns_b.dtype, device=cached_returns_b.device,
                    )
                    union_regrets_det_b = torch.full(
                        (n_union_b,), float("-inf"),
                        dtype=cached_regrets_b.dtype, device=cached_regrets_b.device,
                    )
                    cached_idx_t_b = torch.tensor(
                        cached_indices_b, dtype=torch.long, device=cached_returns_b.device,
                    )
                    union_returns_det_b = union_returns_det_b.index_copy(
                        0, cached_idx_t_b, cached_returns_b,
                    )
                    union_regrets_det_b = union_regrets_det_b.index_copy(
                        0, cached_idx_t_b, cached_regrets_b,
                    )

                    # Lazy fit fallback: pre-sample partitions, check coverage
                    cached_set_b = set(cached_indices_b)
                    g_b = torch.Generator(device="cpu")
                    g_b.manual_seed(int(rng_seed_b))
                    extra_to_score_b: set[int] = set()
                    for _ in range(cfg.bootstrap_permutations):
                        perm_b = torch.randperm(n_union_b, generator=g_b)
                        fp_set_b = set(perm_b[:fit_n_total].tolist())
                        if len(fp_set_b & cached_set_b) < cfg.ols_top_k:
                            extra_to_score_b.update(fp_set_b - cached_set_b)
                    if extra_to_score_b:
                        _ucache_b.fit_fallback_count += 1
                        extra_list_b = sorted(extra_to_score_b)
                        extra_tasks_b = [union_tasks_b[i] for i in extra_list_b]
                        extra_returns_b, extra_regrets_b = _exact_returns_and_regrets(
                            model,
                            extra_tasks_b,
                            cfg=cfg,
                            require_grad=False,
                            materialization_cache=materialization_cache,
                        )
                        extra_idx_t_b = torch.tensor(
                            extra_list_b, dtype=torch.long, device=extra_regrets_b.device,
                        )
                        union_returns_det_b = union_returns_det_b.index_copy(
                            0, extra_idx_t_b, extra_returns_b,
                        )
                        union_regrets_det_b = union_regrets_det_b.index_copy(
                            0, extra_idx_t_b, extra_regrets_b,
                        )
                    _ucache_b.record_hit(step=step)
                else:
                    # Cache miss: score full union, refresh cache if enabled
                    union_returns_det_b, union_regrets_det_b = _exact_returns_and_regrets(
                        model,
                        union_tasks_b,
                        cfg=cfg,
                        require_grad=False,
                        materialization_cache=materialization_cache,
                    )
                    if _ucache_on_b:
                        c_size_b = min(cfg.union_cache_top_c, n_union_b)
                        top_c_b = select_top_k_indices(
                            union_regrets_det_b, c_size_b,
                        ).detach().cpu().tolist()
                        prev_b = _ucache_b
                        new_cache_b = UnionTopCache(
                            cached_union_indices=top_c_b,
                            last_refresh_step=step,
                            n_fit_at_refresh=fit_n_total,
                            n_union_at_refresh=n_union_b,
                            refresh_count=(prev_b.refresh_count if prev_b else 0) + 1,
                            hit_count=prev_b.hit_count if prev_b else 0,
                            miss_count=(prev_b.miss_count if prev_b else 0) + 1,
                            fit_fallback_count=prev_b.fit_fallback_count if prev_b else 0,
                            deploy_fallback_count=prev_b.deploy_fallback_count if prev_b else 0,
                            total_cache_age=prev_b.total_cache_age if prev_b else 0,
                        )
                        union_caches[manifest.name] = new_cache_b
                fit_p_b, eval_p_b, needed_idx_b = plan_bootstrap_permutations(
                    n_union=n_union_b,
                    n_fit=fit_n_total,
                    n_eval=deploy_n_total_pre,
                    n_permutations=cfg.bootstrap_permutations,
                    rng_seed=rng_seed_b,
                    fit_top_k=cfg.ols_top_k,
                    eval_extrap_k=eval_extrap_k_b,
                    union_values_detached=union_regrets_det_b,
                    grad_mode=cfg.grad_mode,
                )
                if needed_idx_b.numel() > 0:
                    grad_tasks_b = [union_tasks_b[int(i)] for i in needed_idx_b.tolist()]
                    _, regrets_grad_compact_b = _exact_returns_and_regrets(
                        model,
                        grad_tasks_b,
                        cfg=cfg,
                        require_grad=True,
                        materialization_cache=materialization_cache,
                    )
                    mask_b = torch.zeros(
                        n_union_b, dtype=torch.bool, device=union_regrets_det_b.device,
                    )
                    mask_b[needed_idx_b] = True
                    regrets_grad_full_b = torch.zeros_like(union_regrets_det_b).index_copy(
                        0, needed_idx_b, regrets_grad_compact_b,
                    )
                    union_regrets_b = torch.where(
                        mask_b, regrets_grad_full_b, union_regrets_det_b,
                    )
                else:
                    union_regrets_b = union_regrets_det_b

                meta_loss_i, coeffs_i = _meta_primal_loss_bootstrap(
                    union_regrets_b,
                    fit_n_total,
                    deploy_n_total_pre,
                    cfg=cfg,
                    rng_seed=rng_seed_b,
                    calibration_alpha=cal_alpha,
                    calibration_beta=cal_beta,
                )
                meta_losses.append(meta_loss_i)
                coeff_rows.append(coeffs_i)
                _stage_calibration_update(cal_buffer, cal_updates_local, manifest.name, coeffs_i)
                fit_returns_det_slice = union_returns_det_b[:fit_n_total]
                deploy_returns_det_slice = union_returns_det_b[fit_n_total:]
                fit_regrets_det_slice = union_regrets_det_b[:fit_n_total]
                deploy_regrets_det_slice = union_regrets_det_b[fit_n_total:]
                # Under cache hits we sentinel non-evaluated positions to
                # -inf; mask those out for the diagnostic mean (top1/max
                # ignores -inf naturally).
                def _masked_mean(t: torch.Tensor) -> float:
                    finite = torch.isfinite(t)
                    if not bool(finite.any()):
                        return float("nan")
                    return float(t[finite].mean().item())
                fit_mean_returns.append(_masked_mean(fit_returns_det_slice))
                fit_mean_regrets.append(_masked_mean(fit_regrets_det_slice))
                fit_top1_regrets.append(float(fit_regrets_det_slice.max().item()))
                deploy_mean_returns.append(_masked_mean(deploy_returns_det_slice))
                deploy_mean_regrets.append(_masked_mean(deploy_regrets_det_slice))
                deploy_top1_regrets.append(float(deploy_regrets_det_slice.max().item()))
                selected_fit_counts.append(min(int(cfg.ols_top_k), int(fit_n_total)))
                selected_deploy_candidate_counts.append(int(eval_extrap_k_b))
                continue

            fit_returns_detached, fit_regrets_detached = _exact_returns_and_regrets(
                model,
                fit_subset,
                cfg=cfg,
                require_grad=False,
                materialization_cache=materialization_cache,
            )
            fit_topk_idx = select_top_k_indices(fit_regrets_detached, cfg.ols_top_k)
            fit_topk_tasks = _task_sublist(fit_subset, fit_topk_idx.detach().cpu().tolist())
            fit_returns, fit_regrets = _exact_returns_and_regrets(
                model,
                fit_topk_tasks,
                cfg=cfg,
                require_grad=True,
                materialization_cache=materialization_cache,
            )

            deploy_n_total = len(manifest.deploy_tasks)
            cache_enabled = _deploy_cache_enabled(cfg, deploy_n_total)
            cache = deploy_caches.get(manifest.name)
            if cache_enabled and cache is not None and (step - cache.last_refresh_step) < cfg.meta_deploy_cache_refresh_interval:
                deploy_candidate_idx = list(cache.cached_indices)
            else:
                _, deploy_regrets_detached = _exact_returns_and_regrets(
                    model,
                    manifest.deploy_tasks,
                    cfg=cfg,
                    require_grad=False,
                    materialization_cache=materialization_cache,
                )
                if cache_enabled:
                    deploy_candidate_count = min(cfg.meta_deploy_cache_top_b, deploy_n_total)
                elif cfg.meta_loss_mode == "ols_primal_extrapolated":
                    deploy_candidate_count = count_extrapolated_eval_points(
                        fit_n_total,
                        deploy_n_total,
                        plotting=cfg.plotting_position,
                    )
                    deploy_candidate_count = max(1, min(deploy_candidate_count, deploy_n_total))
                else:
                    deploy_candidate_count = deploy_n_total
                deploy_candidate_idx = select_top_k_indices(
                    deploy_regrets_detached,
                    deploy_candidate_count,
                ).detach().cpu().tolist()
                if cache_enabled:
                    overlap = 0
                    if cache is not None:
                        overlap = len(set(cache.cached_indices) & set(deploy_candidate_idx))
                    cache = DeploySubsetCache(
                        cached_indices=list(deploy_candidate_idx),
                        last_refresh_step=step,
                        refresh_count=0 if cache is None else cache.refresh_count,
                        total_uses=0 if cache is None else cache.total_uses,
                        total_cache_age=0 if cache is None else cache.total_cache_age,
                        total_candidate_count=0 if cache is None else cache.total_candidate_count,
                        total_overlap_with_previous=0 if cache is None else cache.total_overlap_with_previous,
                        overlap_measurements=0 if cache is None else cache.overlap_measurements,
                    )
                    cache.refresh_count += 1
                    if deploy_caches.get(manifest.name) is not None:
                        cache.total_overlap_with_previous += overlap
                        cache.overlap_measurements += 1
                    deploy_caches[manifest.name] = cache
                    selected_deploy_refreshes += 1
            if cache_enabled:
                cache = deploy_caches[manifest.name]
                cache.record_use(step=step)
                selected_cache_ages.append(max(0, step - cache.last_refresh_step))

            deploy_candidate_tasks = _task_sublist(manifest.deploy_tasks, deploy_candidate_idx)
            deploy_returns, deploy_regrets = _exact_returns_and_regrets(
                model,
                deploy_candidate_tasks,
                cfg=cfg,
                require_grad=True,
                materialization_cache=materialization_cache,
            )
            if cfg.meta_loss_mode == "ols_coeff":
                meta_loss_i, coeffs_i = _meta_coeff_loss_selected(
                    fit_regrets,
                    fit_n_total=fit_n_total,
                    deploy_regrets_candidates=deploy_regrets,
                    deploy_n_total=deploy_n_total,
                    cfg=cfg,
                )
            else:
                meta_loss_i, coeffs_i = _meta_primal_loss_selected(
                    fit_regrets,
                    fit_n_total=fit_n_total,
                    deploy_regrets_candidates=deploy_regrets,
                    deploy_n_total=deploy_n_total,
                    cfg=cfg,
                    calibration_alpha=cal_alpha,
                    calibration_beta=cal_beta,
                )
                _stage_calibration_update(cal_buffer, cal_updates_local, manifest.name, coeffs_i)
            # --- IOG penalty (squared hinge against pretrained baseline) ---
            # Replaces the IOG mask: a real gradient on a real loss term.
            # Symmetric over deploy candidate tasks AND fit top-k tasks to
            # match the IOG mask's coverage of both.
            if cfg.iog_penalty_coef > 0 and iog_penalty_baseline_regrets is not None:
                # Deploy-side penalty
                if deploy_regrets.requires_grad and len(deploy_candidate_tasks) > 0:
                    baseline_vals: list[float] = []
                    hits: list[int] = []
                    for j, task in enumerate(deploy_candidate_tasks):
                        key = _task_key(task)
                        if key in iog_penalty_baseline_regrets:
                            baseline_vals.append(iog_penalty_baseline_regrets[key])
                            hits.append(j)
                    if hits:
                        hits_t = torch.tensor(hits, dtype=torch.long, device=deploy_regrets.device)
                        baseline_t = torch.tensor(
                            baseline_vals, dtype=deploy_regrets.dtype, device=deploy_regrets.device,
                        )
                        excess = torch.relu(deploy_regrets.index_select(0, hits_t) - baseline_t)
                        penalty_i = cfg.iog_penalty_coef * (excess ** 2).mean()
                        if torch.isfinite(penalty_i):
                            meta_loss_i = meta_loss_i + penalty_i
                            coeffs_i = dict(coeffs_i)
                            coeffs_i["iog_penalty"] = float(penalty_i.detach().item())
                            coeffs_i["iog_penalty_violation_frac"] = float(
                                (excess > 0).float().mean().detach().item()
                            )
                # Fit-side penalty (symmetric)
                if fit_regrets.requires_grad and len(fit_topk_tasks) > 0:
                    f_baseline_vals: list[float] = []
                    f_hits: list[int] = []
                    for j, task in enumerate(fit_topk_tasks):
                        key = _task_key(task)
                        if key in iog_penalty_baseline_regrets:
                            f_baseline_vals.append(iog_penalty_baseline_regrets[key])
                            f_hits.append(j)
                    if f_hits:
                        f_hits_t = torch.tensor(f_hits, dtype=torch.long, device=fit_regrets.device)
                        f_baseline_t = torch.tensor(
                            f_baseline_vals, dtype=fit_regrets.dtype, device=fit_regrets.device,
                        )
                        f_excess = torch.relu(fit_regrets.index_select(0, f_hits_t) - f_baseline_t)
                        f_penalty_i = cfg.iog_penalty_coef * (f_excess ** 2).mean()
                        if torch.isfinite(f_penalty_i):
                            meta_loss_i = meta_loss_i + f_penalty_i
                            coeffs_i = dict(coeffs_i) if not isinstance(coeffs_i, dict) else coeffs_i
                            coeffs_i["iog_penalty_fit"] = float(f_penalty_i.detach().item())
                            coeffs_i["iog_penalty_fit_violation_frac"] = float(
                                (f_excess > 0).float().mean().detach().item()
                            )
            meta_losses.append(meta_loss_i)
            coeff_rows.append(coeffs_i)
            fit_mean_returns.append(float(fit_returns.detach().mean().item()))
            fit_mean_regrets.append(float(fit_regrets.detach().mean().item()))
            fit_top1_regrets.append(float(fit_regrets.detach().max().item()))
            deploy_mean_returns.append(float(deploy_returns.detach().mean().item()))
            deploy_mean_regrets.append(float(deploy_regrets.detach().mean().item()))
            deploy_top1_regrets.append(float(deploy_regrets.detach().max().item()))
            selected_fit_counts.append(len(fit_topk_tasks))
            selected_deploy_candidate_counts.append(len(deploy_candidate_tasks))
        # Apply this step's fresh worst-rank pairs to the calibration buffer
        # (after all selected manifests, so each LOO fit above used the prior
        # step's state). RL meta-training is single-process; no cross-rank
        # gather is needed.
        if cal_buffer is not None:
            for _mname, _q, _y in cal_updates_local:
                cal_buffer.update(_mname, _q, _y)
        meta_loss = torch.stack(meta_losses).mean()
        coeffs = _mean_scalar_dict(coeff_rows)
        aux_loss, aux_stats = _meta_aux_loss_exact(
            model,
            pretrain_tasks,
            cfg=cfg,
            rng=rng,
            reference_returns=reference_returns,
            materialization_cache=materialization_cache,
        )
        total_loss = meta_loss + cfg.meta_aux_loss_weight * aux_loss
        total_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        entry = {
            "phase": "meta",
            "step": step,
            "elapsed_sec": float(elapsed_offset + (time.perf_counter() - phase_start)),
            "loss": float(meta_loss.detach().item()),
            "meta_loss": float(meta_loss.detach().item()),
            "aux_loss": float(aux_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "fit_mean_return": float(sum(fit_mean_returns) / len(fit_mean_returns)),
            "fit_mean_regret": float(sum(fit_mean_regrets) / len(fit_mean_regrets)),
            "fit_top1_regret": float(sum(fit_top1_regrets) / len(fit_top1_regrets)),
            "deploy_mean_return": float(sum(deploy_mean_returns) / len(deploy_mean_returns)),
            "deploy_mean_regret": float(sum(deploy_mean_regrets) / len(deploy_mean_regrets)),
            "deploy_top1_regret": float(sum(deploy_top1_regrets) / len(deploy_top1_regrets)),
            "selected_manifest_count": float(len(selected_manifests)),
            "selected_fit_count": float(sum(selected_fit_counts) / len(selected_fit_counts)),
            "selected_deploy_candidate_count": float(
                sum(selected_deploy_candidate_counts) / len(selected_deploy_candidate_counts)
            ),
            "selected_deploy_refresh_count": float(selected_deploy_refreshes),
            "selected_deploy_mean_cache_age": float(
                sum(selected_cache_ages) / len(selected_cache_ages)
            )
            if selected_cache_ages
            else 0.0,
            **coeffs,
            **aux_stats,
        }
        if cal_buffer is not None:
            _ca, _cb = cal_buffer.global_coeffs()
            entry["calibration_alpha"] = float(_ca)
            entry["calibration_beta"] = float(_cb)
            entry["calibration_n_groups"] = float(cal_buffer.n_groups)
        history.append(entry)
        if checkpoint_callback is not None and cfg.save_every > 0:
            if ((step + 1) % cfg.save_every == 0) or (step + 1 == cfg.meta_steps):
                checkpoint_callback(step + 1, history, deploy_caches, union_caches)
        if step % cfg.log_every == 0:
            iog_text = ""
            if "improving_only_deploy_frac" in entry and "improving_only_fit_frac" in entry:
                iog_text = (
                    f" | iog_fit={entry['improving_only_fit_frac']:.2f}"
                    f" | iog_deploy={entry['improving_only_deploy_frac']:.2f}"
                )
            if "iog_penalty" in entry:
                iog_text += (
                    f" | iog_pen_d={entry['iog_penalty']:.4f}"
                    f" | viol_d={entry['iog_penalty_violation_frac']:.2f}"
                )
            if "iog_penalty_fit" in entry:
                iog_text += (
                    f" | iog_pen_f={entry['iog_penalty_fit']:.4f}"
                    f" | viol_f={entry['iog_penalty_fit_violation_frac']:.2f}"
                )
            if "calibration_alpha" in entry:
                iog_text += (
                    f" | cal a={entry['calibration_alpha']:.3f}"
                    f" b={entry['calibration_beta']:.3f}"
                    f" n={entry['calibration_n_groups']:.0f}"
                )
            print(
                f"  meta step {step:4d} | meta={entry['meta_loss']:.6f} "
                f"| aux={entry['aux_loss']:.6f} | total={entry['total_loss']:.6f} "
                f"| fit_regret={entry['fit_mean_regret']:.4f} | deploy_regret={entry['deploy_mean_regret']:.4f} "
                f"| fit_top1={entry['fit_top1_regret']:.4f} | deploy_top1={entry['deploy_top1_regret']:.4f} "
                f"| deploy_B={entry['selected_deploy_candidate_count']:.1f} "
                f"| refreshes={entry['selected_deploy_refresh_count']:.0f} "
                f"{iog_text}"
                f"| elapsed={_format_elapsed_short(entry['elapsed_sec'])}",
                flush=True,
            )
    cache_summary = {
        "enabled": bool(any(cache.refresh_count > 0 for cache in deploy_caches.values())),
        "top_b": float(cfg.meta_deploy_cache_top_b),
        "refresh_interval": float(cfg.meta_deploy_cache_refresh_interval),
        "per_manifest": {
            name: cache.summary()
            for name, cache in deploy_caches.items()
        },
    }
    if deploy_caches:
        total_refreshes = sum(cache.refresh_count for cache in deploy_caches.values())
        total_uses = sum(cache.total_uses for cache in deploy_caches.values())
        total_age = sum(cache.total_cache_age for cache in deploy_caches.values())
        total_candidates = sum(cache.total_candidate_count for cache in deploy_caches.values())
        total_overlap = sum(cache.total_overlap_with_previous for cache in deploy_caches.values())
        overlap_n = sum(cache.overlap_measurements for cache in deploy_caches.values())
        cache_summary.update(
            {
                "total_refreshes": float(total_refreshes),
                "total_uses": float(total_uses),
                "mean_cache_age": float(total_age / total_uses) if total_uses else 0.0,
                "mean_candidate_count": float(total_candidates / total_uses) if total_uses else 0.0,
                "mean_overlap_with_previous": float(total_overlap / overlap_n) if overlap_n else 0.0,
            }
        )
    return history, cache_summary


def _batch_loo_baseline_value(
    union_values_det: torch.Tensor,
    u_idx: int,
    key: str,
    ema_regret_state: dict[str, float],
    *,
    fit_n_total: int,
    deploy_n_total: int,
    cfg: RLQuantConfig,
    rng_seed: int,
) -> float:
    """Estimator-B leave-one-out baseline: the loss with task u_idx's sampled
    regret replaced by a value independent of its current rollouts (its
    cross-step EMA, or the mean of the OTHER tasks' current values on first
    visit). By construction the result does not depend on
    union_values_det[u_idx], which is what makes it a valid baseline."""
    if key in ema_regret_state:
        substitute = float(ema_regret_state[key])
    else:
        others = torch.cat([union_values_det[:u_idx], union_values_det[u_idx + 1:]])
        substitute = float(others.mean().item())
    baseline_values = union_values_det.clone()
    baseline_values[u_idx] = substitute
    with torch.no_grad():
        baseline_obj, _ = _meta_primal_loss_bootstrap(
            baseline_values,
            fit_n_total,
            deploy_n_total,
            cfg=cfg,
            rng_seed=rng_seed,
        )
    return float(baseline_obj.item())


def _reinforce_manifest_losses(
    model: DensePolicyNet,
    fit_subset: Sequence[GridworldTask],
    deploy_tasks: Sequence[GridworldTask],
    *,
    cfg: RLQuantConfig,
    rng_seed: int,
    device: str,
    ema_return_state: dict[str, float],
    ema_regret_state: dict[str, float],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict, dict] | None:
    """One manifest's sampled forecast loss and its policy-gradient surrogate.

    Returns (meta_objective_detached, meta_pg_loss, coeffs, stats), or None
    when the manifest has no extrapolated deploy ranks. Shared by
    `_meta_train_reinforce` and the gradient-fidelity diagnostic.
    """
    fit_n_total = len(fit_subset)
    deploy_n_total = len(deploy_tasks)
    eval_extrap_k = count_extrapolated_eval_points(
        fit_n_total, deploy_n_total, plotting=cfg.plotting_position,
    )
    if eval_extrap_k <= 0:
        return None
    n_rollouts = int(cfg.reinforce_rollouts_per_task)
    ema_decay = float(cfg.reinforce_ema_decay)
    union_tasks = list(fit_subset) + list(deploy_tasks)
    n_union = len(union_tasks)

    # ---- Scoring pass: no-grad sampled regrets for the full union.
    # Rank selection and partition planning run on these estimates, so the
    # whole training loop stays sampling-based (V* enters only as the
    # per-task constant that defines regret).
    score_batch = prepare_task_batch(union_tasks, device)
    score_batch = _resolve_optimal_values(union_tasks, score_batch, cfg=cfg)
    with torch.no_grad():
        score_dense_logits = model(score_batch["features"], score_batch["task_vectors"])
        score_returns, _ = sample_policy_rollouts_batched(
            union_tasks,
            score_dense_logits,
            score_batch,
            n_rollouts=n_rollouts,
            device=device,
            generator=generator,
        )
    union_returns_det = score_returns.mean(dim=1)
    if cfg.reinforce_score == "neg_return":
        union_regrets_det = -union_returns_det
    else:
        union_regrets_det = score_batch["optimal_values"] - union_returns_det

    fit_partitions, eval_partitions, needed_idx = plan_bootstrap_permutations(
        n_union=n_union,
        n_fit=fit_n_total,
        n_eval=deploy_n_total,
        n_permutations=cfg.bootstrap_permutations,
        rng_seed=rng_seed,
        fit_top_k=cfg.ols_top_k,
        eval_extrap_k=eval_extrap_k,
        union_values_detached=union_regrets_det,
        grad_mode=cfg.grad_mode,
    )
    del fit_partitions, eval_partitions  # replanned inside the loss helper
    needed_list = [int(i) for i in needed_idx.tolist()]
    if not needed_list:
        return None

    # ---- PG pass: fresh rollouts WITH log-prob graph, only for the tasks
    # the loss actually touches (fit top-k + extrapolated deploy ranks across
    # permutations) -- the same selection-as-fixed treatment the exact path
    # applies through select_top_k_indices.
    needed_tasks = [union_tasks[i] for i in needed_list]
    pg_batch = prepare_task_batch(needed_tasks, device)
    pg_batch = _resolve_optimal_values(needed_tasks, pg_batch, cfg=cfg)
    pg_dense_logits = model(pg_batch["features"], pg_batch["task_vectors"])
    pg_returns, pg_logprobs = sample_policy_rollouts_batched(
        needed_tasks,
        pg_dense_logits,
        pg_batch,
        n_rollouts=n_rollouts,
        device=device,
        generator=generator,
    )
    if cfg.reinforce_score == "neg_return":
        pg_regrets_det = -pg_returns.detach().mean(dim=1)
    else:
        pg_regrets_det = pg_batch["optimal_values"] - pg_returns.detach().mean(dim=1)
    needed_idx_t = torch.tensor(needed_list, dtype=torch.long, device=device)

    # Leaf values: coupled (default) puts the PG-pass sampled regrets at the
    # needed positions, so the loss value/weights are functions of the same
    # trajectories whose log-probs get gradient (required for estimator
    # 'batch' to be a valid score-function estimator). Plugin with
    # --reinforce-split-samples keeps the scoring-pass values instead,
    # removing the weight/advantage correlation bias.
    union_values = union_regrets_det.clone()
    if not cfg.reinforce_split_samples:
        union_values = union_values.index_copy(0, needed_idx_t, pg_regrets_det)
    union_leaf = union_values.detach().requires_grad_(True)
    meta_objective, coeffs = _meta_primal_loss_bootstrap(
        union_leaf,
        fit_n_total,
        deploy_n_total,
        cfg=cfg,
        rng_seed=rng_seed,
    )
    (leaf_grad,) = torch.autograd.grad(
        meta_objective,
        [union_leaf],
        allow_unused=True,
    )
    weights_full = torch.zeros_like(union_leaf) if leaf_grad is None else leaf_grad.detach()
    weights_needed = weights_full[needed_idx_t]

    stats: dict[str, float] = {}
    if cfg.reinforce_meta_estimator == "plugin":
        advantages = _reinforce_advantages(
            pg_returns,
            cfg=cfg,
            optimal_values=pg_batch["optimal_values"],
            task_keys=[_task_key(task) for task in needed_tasks],
            ema_state=ema_return_state,
        )
        # d(regret)/d(return) = -1, so the surrogate whose gradient estimates
        # dLoss/dtheta at the exact objective's scale is
        # -sum_i w_i * mean_r(advantage * logprob).
        per_task_pg = (advantages * pg_logprobs).mean(dim=1)
        meta_pg_loss = -(weights_needed * per_task_pg).sum()
        stats["reinforce_pg_task_count"] = float((weights_needed != 0).sum().item())
    else:  # 'batch': score-function REINFORCE on the scalar loss value
        ell = float(meta_objective.detach().item())
        union_values_det = union_leaf.detach()
        active_positions = [
            pos for pos in range(len(needed_list))
            if float(weights_needed[pos].item()) != 0.0
        ]
        terms: list[torch.Tensor] = []
        gaps: list[float] = []
        for pos in active_positions:
            u_idx = needed_list[pos]
            b_i = _batch_loo_baseline_value(
                union_values_det,
                u_idx,
                _task_key(union_tasks[u_idx]),
                ema_regret_state,
                fit_n_total=fit_n_total,
                deploy_n_total=deploy_n_total,
                cfg=cfg,
                rng_seed=rng_seed,
            )
            gaps.append(abs(ell - b_i))
            terms.append((ell - b_i) * pg_logprobs[pos].mean())
        if terms:
            meta_pg_loss = torch.stack(terms).sum()
        else:
            meta_pg_loss = torch.zeros((), dtype=torch.float32, device=device)
        stats["reinforce_pg_task_count"] = float(len(active_positions))
        if gaps:
            stats["reinforce_batch_baseline_gap"] = float(sum(gaps) / len(gaps))

    # Update the regret EMA store AFTER the estimator-B baselines were
    # computed (baseline independence from the current rollouts).
    for i, task in enumerate(union_tasks):
        key = _task_key(task)
        fresh = float(union_regrets_det[i].item())
        if key in ema_regret_state:
            ema_regret_state[key] = ema_decay * ema_regret_state[key] + (1.0 - ema_decay) * fresh
        else:
            ema_regret_state[key] = fresh

    stats["fit_mean_return"] = float(union_returns_det[:fit_n_total].mean().item())
    stats["fit_mean_regret"] = float(union_regrets_det[:fit_n_total].mean().item())
    stats["fit_top1_regret"] = float(union_regrets_det[:fit_n_total].max().item())
    stats["deploy_mean_return"] = float(union_returns_det[fit_n_total:].mean().item())
    stats["deploy_mean_regret"] = float(union_regrets_det[fit_n_total:].mean().item())
    stats["deploy_top1_regret"] = float(union_regrets_det[fit_n_total:].max().item())
    return meta_objective.detach(), meta_pg_loss, coeffs, stats


def _meta_train_reinforce(
    model: DensePolicyNet,
    optimizer: torch.optim.Optimizer,
    pretrain_tasks: Sequence[TaskRecord],
    meta_manifests: Sequence[FitDeployManifest],
    *,
    cfg: RLQuantConfig,
    rng: random.Random,
    reference_returns: dict[str, float] | None,
    materialization_cache: TaskMaterializationCache | None = None,
    start_step: int = 0,
    history: list[dict] | None = None,
    checkpoint_callback=None,
    ema_return_state: dict[str, float] | None = None,
    ema_regret_state: dict[str, float] | None = None,
) -> list[dict]:
    if cfg.calibration_aware:
        raise NotImplementedError(
            "calibration_aware is only implemented for the exact meta-train path "
            "(grad_estimator='exact'), not the reinforce path."
        )
    history = list(history) if history is not None else []
    device = str(next(model.parameters()).device)
    elapsed_offset = float(history[-1]["elapsed_sec"]) if history else 0.0
    phase_start = time.perf_counter()
    # Per-task running mean of RETURNS (advantage baseline, mode 'ema') and of
    # REGRETS (estimator-B leave-one-out loss baselines). Both are updated only
    # after the quantities that must stay independent of the current rollouts
    # have been computed. Callers pass persistent dicts (mutated in place) so
    # resume checkpoints carry them; standalone use gets fresh state.
    ema_return_state = {} if ema_return_state is None else ema_return_state
    ema_regret_state = {} if ema_regret_state is None else ema_regret_state
    for step in range(start_step, cfg.meta_steps):
        optimizer.zero_grad(set_to_none=True)
        selected_manifests = _sample_meta_manifest_batch(meta_manifests, cfg, rng)
        if not selected_manifests:
            raise ValueError("meta_manifests must be non-empty for meta training")
        meta_objectives: list[torch.Tensor] = []
        meta_pg_losses: list[torch.Tensor] = []
        coeff_rows: list[dict[str, float]] = []
        fit_mean_returns: list[float] = []
        fit_mean_regrets: list[float] = []
        fit_top1_regrets: list[float] = []
        deploy_mean_returns: list[float] = []
        deploy_mean_regrets: list[float] = []
        deploy_top1_regrets: list[float] = []
        pg_task_counts: list[float] = []
        batch_baseline_gaps: list[float] = []
        for manifest in selected_manifests:
            fit_subset = _materialize_records_cached(
                _sample_fit_subset(manifest.fit_tasks, cfg, rng),
                cfg=cfg,
                split_name=f"{manifest.name}_fit",
                cache=materialization_cache,
            )
            use_bootstrap = (
                cfg.bootstrap_permutations >= 1
                and cfg.meta_loss_mode == "ols_primal_extrapolated"
                and len(fit_subset) > 0
                and len(manifest.deploy_tasks) > 0
            )
            if not use_bootstrap:
                raise NotImplementedError(
                    "The reinforce meta path currently requires the bootstrap "
                    "objective (bootstrap_permutations >= 1 with "
                    "meta_loss_mode='ols_primal_extrapolated'), matching the "
                    "canonical recipe."
                )
            deploy_tasks = _materialize_records_cached(
                manifest.deploy_tasks,
                cfg=cfg,
                split_name=f"{manifest.name}_deploy",
                cache=materialization_cache,
            )
            result = _reinforce_manifest_losses(
                model,
                fit_subset,
                deploy_tasks,
                cfg=cfg,
                rng_seed=rng.randrange(2**31),
                device=device,
                ema_return_state=ema_return_state,
                ema_regret_state=ema_regret_state,
            )
            if result is None:
                continue
            meta_objective_i, meta_pg_loss_i, coeffs_i, stats_i = result
            meta_objectives.append(meta_objective_i)
            meta_pg_losses.append(meta_pg_loss_i)
            coeff_rows.append(coeffs_i)
            fit_mean_returns.append(stats_i["fit_mean_return"])
            fit_mean_regrets.append(stats_i["fit_mean_regret"])
            fit_top1_regrets.append(stats_i["fit_top1_regret"])
            deploy_mean_returns.append(stats_i["deploy_mean_return"])
            deploy_mean_regrets.append(stats_i["deploy_mean_regret"])
            deploy_top1_regrets.append(stats_i["deploy_top1_regret"])
            pg_task_counts.append(stats_i["reinforce_pg_task_count"])
            if "reinforce_batch_baseline_gap" in stats_i:
                batch_baseline_gaps.append(stats_i["reinforce_batch_baseline_gap"])
        if not meta_pg_losses:
            raise ValueError(
                "No manifest produced a reinforce meta loss this step "
                "(all skipped for eval_extrap_k <= 0?)"
            )
        meta_objective = torch.stack(meta_objectives).mean()
        meta_pg_loss = torch.stack(meta_pg_losses).mean()
        coeffs = _mean_scalar_dict(coeff_rows)
        aux_loss, aux_pg_loss, aux_mean_returns, aux_stats = _meta_aux_loss_reinforce(
            pretrain_tasks,
            model=model,
            cfg=cfg,
            rng=rng,
            reference_returns=reference_returns,
            materialization_cache=materialization_cache,
            ema_state=ema_return_state,
        )
        total_pg_loss = meta_pg_loss + cfg.meta_aux_loss_weight * aux_pg_loss
        total_pg_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        entry = {
            "phase": "meta",
            "step": step,
            "elapsed_sec": float(elapsed_offset + (time.perf_counter() - phase_start)),
            "loss": float(meta_objective.detach().item()),
            "meta_loss": float(meta_objective.detach().item()),
            "aux_loss": float(aux_loss.detach().item()),
            "pg_loss": float(meta_pg_loss.detach().item()),
            "aux_pg_loss": float(aux_pg_loss.detach().item()),
            "total_pg_loss": float(total_pg_loss.detach().item()),
            "fit_mean_return": float(sum(fit_mean_returns) / len(fit_mean_returns)),
            "fit_mean_regret": float(sum(fit_mean_regrets) / len(fit_mean_regrets)),
            "fit_top1_regret": float(sum(fit_top1_regrets) / len(fit_top1_regrets)),
            "deploy_mean_return": float(sum(deploy_mean_returns) / len(deploy_mean_returns)),
            "deploy_mean_regret": float(sum(deploy_mean_regrets) / len(deploy_mean_regrets)),
            "deploy_top1_regret": float(sum(deploy_top1_regrets) / len(deploy_top1_regrets)),
            "selected_manifest_count": float(len(selected_manifests)),
            **coeffs,
            **aux_stats,
        }
        if pg_task_counts:
            entry["reinforce_pg_task_count"] = float(sum(pg_task_counts) / len(pg_task_counts))
        if batch_baseline_gaps:
            entry["reinforce_batch_baseline_gap"] = float(
                sum(batch_baseline_gaps) / len(batch_baseline_gaps)
            )
        if aux_mean_returns.numel() > 0:
            entry["aux_batch_mean_return"] = float(aux_mean_returns.mean().item())
        history.append(entry)
        if checkpoint_callback is not None and cfg.save_every > 0:
            if ((step + 1) % cfg.save_every == 0) or (step + 1 == cfg.meta_steps):
                checkpoint_callback(step + 1, history)
        if step % cfg.log_every == 0:
            iog_text = ""
            if "improving_only_deploy_frac" in entry and "improving_only_fit_frac" in entry:
                iog_text = (
                    f" | iog_fit={entry['improving_only_fit_frac']:.2f}"
                    f" | iog_deploy={entry['improving_only_deploy_frac']:.2f}"
                )
            print(
                f"  meta step {step:4d} | meta={entry['meta_loss']:.6f} | aux={entry['aux_loss']:.6f} "
                f"| pg={entry['pg_loss']:.6f} | total_pg={entry['total_pg_loss']:.6f} "
                f"| fit_regret={entry['fit_mean_regret']:.4f} | deploy_regret={entry['deploy_mean_regret']:.4f} "
                f"| fit_top1={entry['fit_top1_regret']:.4f} | deploy_top1={entry['deploy_top1_regret']:.4f} "
                f"{iog_text}"
                f"| elapsed={_format_elapsed_short(entry['elapsed_sec'])}",
                flush=True,
            )
    return history


def _group_family_metrics(
    tasks: Sequence[TaskRecord],
    returns: torch.Tensor,
    regrets: torch.Tensor,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for task, ret, reg in zip(tasks, returns.tolist(), regrets.tolist()):
        grouped.setdefault(task.family_id, []).append((float(ret), float(reg)))
    out: dict[str, dict[str, float]] = {}
    for family_id, rows in grouped.items():
        family_returns = [r for r, _ in rows]
        family_regrets = [g for _, g in rows]
        out[family_id] = {
            "count": float(len(rows)),
            "mean_return": float(sum(family_returns) / len(family_returns)),
            "mean_regret": float(sum(family_regrets) / len(family_regrets)),
            "worst_regret": float(max(family_regrets)),
        }
    return out


def _quantile_summary(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(values.min().item()),
        "p10": float(torch.quantile(values, 0.10).item()),
        "p25": float(torch.quantile(values, 0.25).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p75": float(torch.quantile(values, 0.75).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "max": float(values.max().item()),
    }


def _evaluate_split_metrics(
    model: DensePolicyNet,
    tasks: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
) -> dict:
    returns, regrets = _exact_returns_and_regrets(model, tasks, cfg=cfg, require_grad=False)
    return _split_metrics_from_outputs(tasks, returns, regrets)


def _evaluate_flat_task_sample_quickcheck(
    model: DensePolicyNet,
    task_pool: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
    label: str,
    sample_count: int,
    seed: int,
    materialization_cache: TaskMaterializationCache | None = None,
) -> dict:
    if sample_count <= 0 or not task_pool:
        return {
            "mode": "disabled",
            "task_count_requested": int(sample_count),
            "task_pool_size": len(task_pool),
            "task_count_evaluated": 0,
            "sample_summary": _task_collection_summary([]),
            "metrics": _split_metrics_from_outputs([], torch.zeros(0), torch.zeros(0)),
        }
    sample_size = min(int(sample_count), len(task_pool))
    rng = random.Random(seed)
    if sample_size == len(task_pool):
        sampled_tasks = list(task_pool)
    else:
        sampled_tasks = rng.sample(list(task_pool), sample_size)
    print(
        f"Starting {label} flat task quickcheck | sample={len(sampled_tasks)}/{len(task_pool)}",
        flush=True,
    )
    returns, regrets = _exact_returns_and_regrets(
        model,
        sampled_tasks,
        cfg=cfg,
        require_grad=False,
        materialization_cache=materialization_cache,
    )
    return {
        "mode": "flat_task_mc_exact",
        "task_count_requested": int(sample_count),
        "task_pool_size": len(task_pool),
        "task_count_evaluated": len(sampled_tasks),
        "sample_seed": int(seed),
        "sample_summary": _task_collection_summary(sampled_tasks),
        "metrics": _split_metrics_from_outputs(sampled_tasks, returns, regrets),
    }


def _split_metrics_from_outputs(
    tasks: Sequence[TaskRecord],
    returns: torch.Tensor,
    regrets: torch.Tensor,
) -> dict:
    return {
        "n_tasks": len(tasks),
        "mean_return": float(returns.mean().item()) if len(returns) else 0.0,
        "mean_regret": float(regrets.mean().item()) if len(regrets) else 0.0,
        "worst_regret": float(regrets.max().item()) if len(regrets) else 0.0,
        "return_quantiles": _quantile_summary(returns),
        "regret_quantiles": _quantile_summary(regrets),
        "family_metrics": _group_family_metrics(tasks, returns, regrets),
    }


def _aggregate_family_metrics(family_metric_rows: Sequence[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not family_metric_rows:
        return {}
    family_ids = sorted({family_id for row in family_metric_rows for family_id in row.keys()})
    out: dict[str, dict[str, float]] = {}
    for family_id in family_ids:
        present = [row[family_id] for row in family_metric_rows if family_id in row]
        total_count = float(sum(entry["count"] for entry in present))
        if total_count <= 0:
            continue
        out[family_id] = {
            "count": total_count,
            "mean_return": float(sum(entry["mean_return"] * entry["count"] for entry in present) / total_count),
            "mean_regret": float(sum(entry["mean_regret"] * entry["count"] for entry in present) / total_count),
            "worst_regret": float(sum(entry["worst_regret"] for entry in present) / len(present)),
        }
    return out


def _aggregate_split_metrics(metric_rows: Sequence[dict]) -> dict:
    if not metric_rows:
        return {
            "n_tasks": 0,
            "n_manifests": 0,
            "aggregation_mode": "mean_over_manifests",
            "mean_return": 0.0,
            "mean_regret": 0.0,
            "worst_regret": 0.0,
            "return_quantiles": {key: 0.0 for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")},
            "regret_quantiles": {key: 0.0 for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")},
            "family_metrics": {},
        }
    quantile_keys = list(metric_rows[0]["return_quantiles"].keys())
    return {
        "n_tasks": int(sum(int(row["n_tasks"]) for row in metric_rows)),
        "n_manifests": len(metric_rows),
        "aggregation_mode": "mean_over_manifests",
        "mean_return": float(sum(float(row["mean_return"]) for row in metric_rows) / len(metric_rows)),
        "mean_regret": float(sum(float(row["mean_regret"]) for row in metric_rows) / len(metric_rows)),
        "worst_regret": float(sum(float(row["worst_regret"]) for row in metric_rows) / len(metric_rows)),
        "return_quantiles": {
            key: float(sum(float(row["return_quantiles"][key]) for row in metric_rows) / len(metric_rows))
            for key in quantile_keys
        },
        "regret_quantiles": {
            key: float(sum(float(row["regret_quantiles"][key]) for row in metric_rows) / len(metric_rows))
            for key in quantile_keys
        },
        "family_metrics": _aggregate_family_metrics([row["family_metrics"] for row in metric_rows]),
    }


def _forecast_pair(
    model: DensePolicyNet,
    fit_tasks: Sequence[TaskRecord],
    deploy_tasks: Sequence[TaskRecord],
    *,
    cfg: RLQuantConfig,
) -> dict:
    fit_returns, fit_regrets = _exact_returns_and_regrets(model, fit_tasks, cfg=cfg, require_grad=False)
    deploy_returns, deploy_regrets = _exact_returns_and_regrets(model, deploy_tasks, cfg=cfg, require_grad=False)
    return _forecast_pair_from_outputs(
        fit_tasks,
        fit_returns,
        fit_regrets,
        deploy_tasks,
        deploy_returns,
        deploy_regrets,
        cfg=cfg,
    )


def _forecast_pair_from_outputs(
    fit_tasks: Sequence[TaskRecord],
    fit_returns: torch.Tensor,
    fit_regrets: torch.Tensor,
    deploy_tasks: Sequence[TaskRecord],
    deploy_returns: torch.Tensor,
    deploy_regrets: torch.Tensor,
    *,
    cfg: RLQuantConfig,
) -> dict:
    predicted, fit_a, fit_b = fit_ols_tail_from_scores(
        fit_regrets.tolist(),
        n_top=cfg.ols_top_k,
        plotting=cfg.plotting_position,
        deploy_queries=len(deploy_tasks),
        score_mode=cfg.ols_score_mode,
    )
    _, deploy_a, deploy_b = fit_ols_tail_from_scores(
        deploy_regrets.tolist(),
        n_top=cfg.ols_top_k,
        plotting=cfg.plotting_position,
        deploy_queries=len(deploy_tasks),
        score_mode=cfg.ols_score_mode,
    )
    actual = float(deploy_regrets.max().item()) if len(deploy_regrets) > 0 else 0.0
    fit_topk = select_top_k_indices(fit_regrets, cfg.ols_top_k)
    deploy_extrap_k = count_extrapolated_eval_points(
        len(fit_regrets),
        len(deploy_regrets),
        plotting=cfg.plotting_position,
    )
    if deploy_extrap_k > 0:
        deploy_extrap_idx = select_top_k_indices(deploy_regrets, deploy_extrap_k)
        _, extrap_stats = differentiable_ols_extrapolated_primal_loss(
            fit_regrets,
            fit_topk,
            len(fit_regrets),
            deploy_regrets,
            deploy_extrap_idx,
            len(deploy_regrets),
            plotting=cfg.plotting_position,
            score_mode=cfg.ols_score_mode,
            grad_mode="both",
            extrapolated_weighting=cfg.extrapolated_weighting,
            improving_only_grad=False,
        )
        heldout_extrapolated_loss = float(extrap_stats["primal_mse"].detach().item())
        extrapolated_mean_abs_error = float(extrap_stats["unweighted_original_mae"].detach().item())
        extrapolated_weighted_mean_abs_error = float(extrap_stats["weighted_original_mae"].detach().item())
        extrapolated_mean_sq_error = float(extrap_stats["unweighted_original_mse"].detach().item())
        extrapolated_weighted_mean_sq_error = float(extrap_stats["weighted_original_mse"].detach().item())
        extrapolated_eval_points = float(extrap_stats["eval_points_used"].detach().item())
    else:
        heldout_extrapolated_loss = float("nan")
        extrapolated_mean_abs_error = float("nan")
        extrapolated_weighted_mean_abs_error = float("nan")
        extrapolated_mean_sq_error = float("nan")
        extrapolated_weighted_mean_sq_error = float("nan")
        extrapolated_eval_points = 0.0

    return {
        "fit_count": len(fit_tasks),
        "deploy_count": len(deploy_tasks),
        "predicted_worst_regret": float(predicted),
        "actual_worst_regret": actual,
        "forecast_abs_error": float(abs(predicted - actual)),
        "forecast_sq_error": float((predicted - actual) ** 2),
        "fit_ols_a": float(fit_a),
        "fit_ols_b": float(fit_b),
        "deploy_ols_a": float(deploy_a),
        "deploy_ols_b": float(deploy_b),
        "fit_mean_regret": float(fit_regrets.mean().item()) if len(fit_regrets) else 0.0,
        "deploy_mean_regret": float(deploy_regrets.mean().item()) if len(deploy_regrets) else 0.0,
        "fit_mean_return": float(fit_returns.mean().item()) if len(fit_returns) else 0.0,
        "deploy_mean_return": float(deploy_returns.mean().item()) if len(deploy_returns) else 0.0,
        "heldout_extrapolated_loss": heldout_extrapolated_loss,
        "heldout_extrapolated_loss_space": "score",
        "extrapolated_mean_abs_error": extrapolated_mean_abs_error,
        "extrapolated_weighted_mean_abs_error": extrapolated_weighted_mean_abs_error,
        "extrapolated_mean_sq_error": extrapolated_mean_sq_error,
        "extrapolated_weighted_mean_sq_error": extrapolated_weighted_mean_sq_error,
        "extrapolated_eval_points": extrapolated_eval_points,
        "extrapolated_weighting": cfg.extrapolated_weighting,
    }


def _aggregate_forecast_pairs(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "manifest_count": 0,
            "fit_count": 0.0,
            "deploy_count": 0.0,
            "fit_count_total": 0,
            "deploy_count_total": 0,
            "predicted_worst_regret": 0.0,
            "actual_worst_regret": 0.0,
            "forecast_abs_error": 0.0,
            "forecast_sq_error": 0.0,
            "fit_ols_a": 0.0,
            "fit_ols_b": 0.0,
            "deploy_ols_a": 0.0,
            "deploy_ols_b": 0.0,
            "fit_mean_regret": 0.0,
            "deploy_mean_regret": 0.0,
            "fit_mean_return": 0.0,
            "deploy_mean_return": 0.0,
            "heldout_extrapolated_loss": 0.0,
            "extrapolated_mean_abs_error": 0.0,
            "extrapolated_weighted_mean_abs_error": 0.0,
            "extrapolated_mean_sq_error": 0.0,
            "extrapolated_weighted_mean_sq_error": 0.0,
            "extrapolated_eval_points": 0.0,
            "per_manifest": [],
        }
    scalar_keys = [
        "predicted_worst_regret",
        "actual_worst_regret",
        "forecast_abs_error",
        "forecast_sq_error",
        "fit_ols_a",
        "fit_ols_b",
        "deploy_ols_a",
        "deploy_ols_b",
        "fit_mean_regret",
        "deploy_mean_regret",
        "fit_mean_return",
        "deploy_mean_return",
        "heldout_extrapolated_loss",
        "extrapolated_mean_abs_error",
        "extrapolated_weighted_mean_abs_error",
        "extrapolated_mean_sq_error",
        "extrapolated_weighted_mean_sq_error",
        "extrapolated_eval_points",
    ]
    out = {
        "manifest_count": len(rows),
        "fit_count": float(sum(float(row["fit_count"]) for row in rows) / len(rows)),
        "deploy_count": float(sum(float(row["deploy_count"]) for row in rows) / len(rows)),
        "fit_count_total": int(sum(int(row["fit_count"]) for row in rows)),
        "deploy_count_total": int(sum(int(row["deploy_count"]) for row in rows)),
        "per_manifest": list(rows),
    }
    for key in scalar_keys:
        out[key] = float(sum(float(row[key]) for row in rows) / len(rows))
    return out


def _tail_gap_row_from_outputs(
    *,
    name: str,
    fit_returns: torch.Tensor,
    fit_regrets: torch.Tensor,
    deploy_returns: torch.Tensor,
    deploy_regrets: torch.Tensor,
) -> dict:
    fit_worst_regret = float(fit_regrets.max().item()) if len(fit_regrets) else 0.0
    deploy_worst_regret = float(deploy_regrets.max().item()) if len(deploy_regrets) else 0.0
    worst_gap = float(deploy_worst_regret - fit_worst_regret)
    return {
        "name": name,
        "fit_mean_return": float(fit_returns.mean().item()) if len(fit_returns) else 0.0,
        "fit_mean_regret": float(fit_regrets.mean().item()) if len(fit_regrets) else 0.0,
        "deploy_mean_return": float(deploy_returns.mean().item()) if len(deploy_returns) else 0.0,
        "deploy_mean_regret": float(deploy_regrets.mean().item()) if len(deploy_regrets) else 0.0,
        "fit_worst_regret": fit_worst_regret,
        "deploy_worst_regret": deploy_worst_regret,
        "worst_gap": worst_gap,
        "deploy_worst_exceeds_fit_worst": bool(worst_gap > 0.0),
    }


def _forecast_trace_row_from_outputs(
    *,
    name: str,
    fit_returns: torch.Tensor,
    fit_regrets: torch.Tensor,
    deploy_returns: torch.Tensor,
    deploy_regrets: torch.Tensor,
) -> dict:
    return {
        "name": name,
        "fit_returns": fit_returns.detach().cpu().to(torch.float32).numpy(),
        "fit_regrets": fit_regrets.detach().cpu().to(torch.float32).numpy(),
        "deploy_returns": deploy_returns.detach().cpu().to(torch.float32).numpy(),
        "deploy_regrets": deploy_regrets.detach().cpu().to(torch.float32).numpy(),
    }


def _save_forecast_trace_archive(
    outdir: Path,
    *,
    pretrained_traces: dict[str, list[dict]] | None,
    meta_learned_traces: dict[str, list[dict]] | None,
    cfg: RLQuantConfig,
) -> None:
    bundles = {
        "pretrained": pretrained_traces or {},
        "meta_learned": meta_learned_traces or {},
    }
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "format": "rlquant_forecast_traces_v1",
        "dtype": "float32",
        "plotting_position": cfg.plotting_position,
        "ols_score_mode": cfg.ols_score_mode,
        "ols_top_k": int(cfg.ols_top_k),
        "extrapolated_weighting": cfg.extrapolated_weighting,
        "models": {},
    }
    for model_label, split_rows in bundles.items():
        model_meta: dict[str, list[dict[str, object]]] = {}
        for split_name in ("meta", "heldout"):
            rows = list(split_rows.get(split_name, []))
            split_meta: list[dict[str, object]] = []
            for idx, row in enumerate(rows):
                prefix = f"{model_label}__{split_name}__{idx:02d}"
                fit_returns = np.asarray(row["fit_returns"], dtype=np.float32)
                fit_regrets = np.asarray(row["fit_regrets"], dtype=np.float32)
                deploy_returns = np.asarray(row["deploy_returns"], dtype=np.float32)
                deploy_regrets = np.asarray(row["deploy_regrets"], dtype=np.float32)
                arrays[f"{prefix}__fit_returns"] = fit_returns
                arrays[f"{prefix}__fit_regrets"] = fit_regrets
                arrays[f"{prefix}__deploy_returns"] = deploy_returns
                arrays[f"{prefix}__deploy_regrets"] = deploy_regrets
                split_meta.append(
                    {
                        "name": str(row["name"]),
                        "array_prefix": prefix,
                        "fit_count": int(fit_regrets.shape[0]),
                        "deploy_count": int(deploy_regrets.shape[0]),
                    }
                )
            model_meta[split_name] = split_meta
        metadata["models"][model_label] = model_meta

    np.savez_compressed(outdir / "forecast_traces.npz", **arrays)
    with open(outdir / "forecast_traces_manifests.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _aggregate_tail_gap_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "manifest_count": 0,
            "deploy_worst_exceeds_fit_worst_count": 0,
            "mean_fit_worst_regret": 0.0,
            "mean_deploy_worst_regret": 0.0,
            "mean_worst_gap": 0.0,
            "positive_gap_fraction": 0.0,
            "all_deploy_worst_exceeds_fit_worst": False,
            "per_manifest": [],
        }
    return {
        "manifest_count": len(rows),
        "deploy_worst_exceeds_fit_worst_count": int(sum(bool(row["deploy_worst_exceeds_fit_worst"]) for row in rows)),
        "mean_fit_worst_regret": float(sum(float(row["fit_worst_regret"]) for row in rows) / len(rows)),
        "mean_deploy_worst_regret": float(sum(float(row["deploy_worst_regret"]) for row in rows) / len(rows)),
        "mean_worst_gap": float(sum(float(row["worst_gap"]) for row in rows) / len(rows)),
        "positive_gap_fraction": float(
            sum(float(bool(row["deploy_worst_exceeds_fit_worst"])) for row in rows) / len(rows)
        ),
        "all_deploy_worst_exceeds_fit_worst": bool(
            all(bool(row["deploy_worst_exceeds_fit_worst"]) for row in rows)
        ),
        "per_manifest": list(rows),
    }


def _evaluate_fit_deploy_manifest_once(
    model: DensePolicyNet,
    manifest: FitDeployManifest,
    *,
    cfg: RLQuantConfig,
    materialization_cache: TaskMaterializationCache | None = None,
) -> tuple[dict, dict, dict]:
    fit_returns, fit_regrets = _exact_returns_and_regrets(
        model,
        manifest.fit_tasks,
        cfg=cfg,
        require_grad=False,
        materialization_cache=materialization_cache,
    )
    deploy_returns, deploy_regrets = _exact_returns_and_regrets(
        model,
        manifest.deploy_tasks,
        cfg=cfg,
        require_grad=False,
        materialization_cache=materialization_cache,
    )
    report_row = {
        "name": manifest.name,
        "fit_metrics": _split_metrics_from_outputs(manifest.fit_tasks, fit_returns, fit_regrets),
        "deploy_metrics": _split_metrics_from_outputs(manifest.deploy_tasks, deploy_returns, deploy_regrets),
        "forecast_pair": _forecast_pair_from_outputs(
            manifest.fit_tasks,
            fit_returns,
            fit_regrets,
            manifest.deploy_tasks,
            deploy_returns,
            deploy_regrets,
            cfg=cfg,
        ),
    }
    tail_row = _tail_gap_row_from_outputs(
        name=manifest.name,
        fit_returns=fit_returns,
        fit_regrets=fit_regrets,
        deploy_returns=deploy_returns,
        deploy_regrets=deploy_regrets,
    )
    trace_row = _forecast_trace_row_from_outputs(
        name=manifest.name,
        fit_returns=fit_returns,
        fit_regrets=fit_regrets,
        deploy_returns=deploy_returns,
        deploy_regrets=deploy_regrets,
    )
    return report_row, tail_row, trace_row


def _evaluate_policy_report_and_tail_gap(
    model: DensePolicyNet,
    manifest_suite: TaskManifestSuite,
    *,
    cfg: RLQuantConfig,
    label: str,
    materialization_cache: TaskMaterializationCache | None = None,
) -> tuple[dict, dict, dict[str, list[dict]]]:
    print(f"Starting {label} evaluation", flush=True)
    pretrain_returns, pretrain_regrets = _exact_returns_and_regrets(
        model,
        manifest_suite.pretrain_train,
        cfg=cfg,
        require_grad=False,
        materialization_cache=materialization_cache,
    )
    pretrain_metrics = _split_metrics_from_outputs(
        manifest_suite.pretrain_train,
        pretrain_returns,
        pretrain_regrets,
    )
    meta_rows: list[dict] = []
    meta_tail_rows: list[dict] = []
    meta_trace_rows: list[dict] = []
    for idx, manifest in enumerate(manifest_suite.meta_manifests):
        print(
            f"  {label} eval meta manifest {idx + 1}/{len(manifest_suite.meta_manifests)}",
            flush=True,
        )
        report_row, tail_row, trace_row = _evaluate_fit_deploy_manifest_once(
            model,
            manifest,
            cfg=cfg,
            materialization_cache=materialization_cache,
        )
        meta_rows.append(report_row)
        meta_tail_rows.append(tail_row)
        meta_trace_rows.append(trace_row)
    heldout_rows: list[dict] = []
    heldout_tail_rows: list[dict] = []
    heldout_trace_rows: list[dict] = []
    for idx, manifest in enumerate(manifest_suite.heldout_manifests):
        print(
            f"  {label} eval heldout manifest {idx + 1}/{len(manifest_suite.heldout_manifests)}",
            flush=True,
        )
        report_row, tail_row, trace_row = _evaluate_fit_deploy_manifest_once(
            model,
            manifest,
            cfg=cfg,
            materialization_cache=materialization_cache,
        )
        heldout_rows.append(report_row)
        heldout_tail_rows.append(tail_row)
        heldout_trace_rows.append(trace_row)
    report = {
        "aggregation_mode": "mean_over_manifests",
        "split_metrics": {
            "pretrain_train": pretrain_metrics,
            "meta_fit": _aggregate_split_metrics([row["fit_metrics"] for row in meta_rows]),
            "meta_deploy": _aggregate_split_metrics([row["deploy_metrics"] for row in meta_rows]),
            "heldout_fit": _aggregate_split_metrics([row["fit_metrics"] for row in heldout_rows]),
            "heldout_deploy": _aggregate_split_metrics([row["deploy_metrics"] for row in heldout_rows]),
        },
        "forecast_pairs": {
            "meta": _aggregate_forecast_pairs([row["forecast_pair"] for row in meta_rows]),
            "heldout": _aggregate_forecast_pairs([row["forecast_pair"] for row in heldout_rows]),
        },
        "manifest_evaluations": {
            "meta": meta_rows,
            "heldout": heldout_rows,
        },
    }
    tail_gap = {
        "aggregation_mode": "mean_over_manifests",
        "check_kind": "deploy_worst_minus_fit_worst",
        "meta": _aggregate_tail_gap_rows(meta_tail_rows),
        "heldout": _aggregate_tail_gap_rows(heldout_tail_rows),
    }
    return report, tail_gap, {"meta": meta_trace_rows, "heldout": heldout_trace_rows}


def _deprecated_difficulty_audit_note() -> dict[str, object]:
    return {
        "deprecated": True,
        "reason": (
            "The old difficulty audit enforced deploy>fit mean/p90 regret and could trigger resampling. "
            "That is incompatible with the intended IID regime. Use tail_gap_check diagnostics instead."
        ),
        "replacement": "tail_gap_check",
    }


def _load_reused_pretrained_results(
    path: str,
    *,
    cfg: RLQuantConfig,
) -> tuple[dict | None, dict | None, dict | None]:
    results_path = Path(path)
    if not results_path.exists():
        raise FileNotFoundError(f"Could not find reused pretrained results at {results_path}")
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported pretrained results format at {results_path}")
    prior_config_path = results_path.parent / "config.json"
    if not prior_config_path.exists():
        raise FileNotFoundError(f"Could not find prior config.json next to {results_path}")
    with prior_config_path.open("r", encoding="utf-8") as f:
        prior_config = json.load(f)
    current_sig = _pretrained_reuse_signature(cfg)
    prior_sig = dict(prior_config)
    prior_sig["effective_task_bank_seed"] = int(
        prior_config.get("seed", 0) if int(prior_config.get("task_bank_seed", -1)) < 0 else prior_config["task_bank_seed"]
    )
    failures: list[str] = []
    for key, expected in current_sig.items():
        actual = prior_sig.get(key)
        if actual is None and key in _LEGACY_SIGNATURE_DEFAULTS:
            actual = _LEGACY_SIGNATURE_DEFAULTS[key]
        if actual != expected:
            failures.append(f"{key}: prior={actual} != cfg={expected}")
    if failures:
        raise RuntimeError(
            "Reused pretrained results config mismatch: " + "; ".join(failures)
        )
    report = payload.get("report")
    tail_gap = payload.get("tail_gap_check")
    quickcheck = payload.get("quickcheck")
    if report is not None and not isinstance(report, dict):
        raise TypeError(f"Expected 'report' dict in {results_path}")
    if tail_gap is not None and not isinstance(tail_gap, dict):
        raise TypeError(f"Expected 'tail_gap_check' dict in {results_path}")
    if quickcheck is not None and not isinstance(quickcheck, dict):
        raise TypeError(f"Expected 'quickcheck' dict in {results_path}")
    return report, tail_gap, quickcheck


def _save_checkpoint(model: DensePolicyNet, outdir: Path, name: str) -> None:
    path = outdir / f"{name}.pt"
    torch.save(model.state_dict(), path)


def _load_checkpoint(model: DensePolicyNet, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")
    model.load_state_dict(state)


def train_rlquant(cfg: RLQuantConfig) -> dict:
    if not cfg.device:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(cfg.seed, deterministic_debug=cfg.deterministic_debug)
    if cfg.resume_checkpoint and (cfg.init_checkpoint or cfg.skip_pretrain):
        raise ValueError("resume_checkpoint may not be combined with init_checkpoint or skip_pretrain")
    if cfg.reuse_pretrained_results and not (cfg.init_checkpoint and cfg.skip_pretrain):
        raise ValueError("reuse_pretrained_results requires both init_checkpoint and skip_pretrain")
    if cfg.meta_fit_subsample_min < cfg.ols_top_k:
        raise ValueError("meta_fit_subsample_min must be >= ols_top_k")
    if cfg.skip_pretrain and not cfg.init_checkpoint:
        raise ValueError("skip_pretrain requires init_checkpoint to be set")
    if cfg.meta_aux_task_source not in {"pretrain_train", "meta_fit"}:
        raise ValueError("meta_aux_task_source must be one of {'pretrain_train', 'meta_fit'}")
    if cfg.pretrain_task_source not in {"pretrain_train", "meta_union", "meta_deploy"}:
        raise ValueError("pretrain_task_source must be one of {'pretrain_train', 'meta_union', 'meta_deploy'}")
    if cfg.family_suite not in {
        "naturalistic",
        "pressure_plate",
        "pressure_plate_open_room_only",
        "trap_open_room_only",
        "hybrid",
    }:
        raise ValueError(
            "family_suite must be one of {'naturalistic', 'pressure_plate', "
            "'pressure_plate_open_room_only', 'trap_open_room_only', 'hybrid'}"
        )
    if cfg.split_mode not in {"disjoint", "nested"}:
        raise ValueError("split_mode must be one of {'disjoint', 'nested'}")
    if cfg.manifest_grouping not in {"iid", "slip_level", "dynamics_id"}:
        raise ValueError("manifest_grouping must be one of {'iid', 'slip_level', 'dynamics_id'}")
    if cfg.manifest_grouping != "iid" and not _bank_mode_enabled(cfg):
        raise ValueError(
            f"manifest_grouping='{cfg.manifest_grouping}' requires a prebuilt task bank (--task-bank-root)"
        )
    if cfg.blind_dynamics and cfg.task_vector_slip:
        raise ValueError("--blind-dynamics contradicts --task-vector-slip (slip would leak into the task vector)")
    if cfg.meta_manifest_count <= 0:
        raise ValueError("meta_manifest_count must be positive")
    if cfg.heldout_manifest_count <= 0:
        raise ValueError("heldout_manifest_count must be positive")
    if cfg.meta_manifests_per_step <= 0:
        raise ValueError("meta_manifests_per_step must be positive")
    if cfg.meta_deploy_cache_top_b < 0:
        raise ValueError("meta_deploy_cache_top_b must be non-negative")
    if cfg.meta_deploy_cache_refresh_interval < 0:
        raise ValueError("meta_deploy_cache_refresh_interval must be non-negative")
    if cfg.pretrained_exact_eval_phase not in {"pre", "post"}:
        raise ValueError("pretrained_exact_eval_phase must be one of {'pre', 'post'}")
    if cfg.pretrained_quickcheck_task_count < 0:
        raise ValueError("pretrained_quickcheck_task_count must be non-negative")
    if cfg.reinforce_baseline not in {"rollout_mean", "optimal_value", "ema"}:
        raise ValueError("reinforce_baseline must be one of {'rollout_mean', 'optimal_value', 'ema'}")
    if cfg.reinforce_meta_estimator not in {"plugin", "batch"}:
        raise ValueError("reinforce_meta_estimator must be one of {'plugin', 'batch'}")
    if cfg.reinforce_score not in {"regret", "neg_return"}:
        raise ValueError("reinforce_score must be one of {'regret', 'neg_return'}")
    if cfg.pretrain_grad_estimator not in {"inherit", "exact", "reinforce"}:
        raise ValueError("pretrain_grad_estimator must be one of {'inherit', 'exact', 'reinforce'}")
    if cfg.grad_estimator == "reinforce":
        if cfg.reinforce_rollouts_per_task < 1:
            raise ValueError("reinforce_rollouts_per_task must be >= 1")
        if cfg.reinforce_rollouts_per_task == 1 and cfg.reinforce_baseline == "rollout_mean":
            raise ValueError(
                "reinforce_rollouts_per_task=1 with reinforce_baseline='rollout_mean' has "
                "identically-zero advantages (the baseline IS the single sample), so every "
                "gradient is exactly zero. Use --reinforce-baseline optimal_value or ema."
            )
        if cfg.reinforce_split_samples and cfg.reinforce_meta_estimator != "plugin":
            raise ValueError("--reinforce-split-samples only applies to --reinforce-meta-estimator plugin")
        if not (0.0 <= cfg.reinforce_ema_decay < 1.0):
            raise ValueError("reinforce_ema_decay must be in [0, 1)")
        if cfg.reinforce_meta_estimator == "batch" and cfg.meta_steps > 0:
            if cfg.bootstrap_permutations < 1 or cfg.meta_loss_mode != "ols_primal_extrapolated":
                raise ValueError(
                    "reinforce_meta_estimator='batch' is only implemented for the "
                    "bootstrap path (bootstrap_permutations >= 1 with "
                    "meta_loss_mode='ols_primal_extrapolated')."
                )
        if cfg.union_cache_top_c > 0 and cfg.meta_steps > 0:
            raise NotImplementedError(
                "union_cache_top_c > 0 is not implemented on the reinforce meta path; "
                "the canonical bipolar recipe runs with the cache disabled."
            )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    pretrain_history: list[dict] = []
    meta_history: list[dict] = []
    meta_training_cache_summary: dict | None = None
    pretrained_report: dict | None = None
    pretrained_trace_bundle: dict[str, list[dict]] | None = None
    pretrained_quickcheck: dict | None = None
    final_report: dict | None = None
    final_trace_bundle: dict[str, list[dict]] | None = None
    manifest_summary: dict | None = None
    pretrained_tail_gap_check: dict | None = None
    final_tail_gap_check: dict | None = None
    manifest_suite: TaskManifestSuite | None = None
    pretrain_reference_returns: dict[str, float] | None = None
    task_bank_info: TaskBankInfo | None = None

    if _bank_mode_enabled(cfg):
        print("Loading task bank", flush=True)
        task_bank_info = _resolve_task_bank_info(cfg)
        bank_check = _validate_task_bank_compatibility(cfg, task_bank_info)
        if not bank_check["passed"]:
            raise RuntimeError("Task bank compatibility check failed: " + "; ".join(bank_check["failures"]))
        support_audit = _skipped_support_audit_for_bank(task_bank_info)
        with open(outdir / "support_audit.json", "w", encoding="utf-8") as f:
            json.dump(support_audit, f, indent=2)
        print(f"Building manifest suite from bank seed {task_bank_info.bank_seed}", flush=True)
        if cfg.manifest_grouping in {"slip_level", "dynamics_id"}:
            manifest_suite = _build_manifest_suite_from_bank_slip_grouped(
                cfg, seed=cfg.seed, bank_info=task_bank_info
            )
        else:
            manifest_suite = _build_manifest_suite_from_bank(cfg, seed=cfg.seed, bank_info=task_bank_info)
    else:
        slice_map = _default_manifest_slices(cfg)
        planned_usage = _planned_usage_by_family(slice_map, cfg)
        audit_families = [family for family, count in planned_usage.items() if count > 0]
        gen_cfg = _make_generator_cfg(cfg)
        support_audit = run_family_support_audit(
            cfg=gen_cfg,
            seed=cfg.seed,
            normal_sample_count=cfg.support_audit_normal_samples,
            challenge_sample_count=cfg.support_audit_challenge_samples,
            planned_usage_by_family=planned_usage,
            families=audit_families,
            normal_min_unique_per_family=cfg.support_audit_normal_min_unique_per_family,
            rare_min_unique_base=cfg.support_audit_rare_min_unique,
        )
        support_check = _validate_support_audit(support_audit, cfg)
        support_audit["validation"] = support_check
        with open(outdir / "support_audit.json", "w", encoding="utf-8") as f:
            json.dump(support_audit, f, indent=2)
        if not support_check["passed"]:
            raise RuntimeError("Support audit failed: " + "; ".join(support_check["failures"]))

        print("Building manifest suite", flush=True)
        manifest_suite = _build_manifest_suite(cfg, seed=cfg.seed)

    if task_bank_info is not None:
        print(f"  using bank file {task_bank_info.bank_path}", flush=True)
        if cfg.bank_materialization == "eager":
            print("  eagerly materializing banked task suite ...", flush=True)
            materialize_start = time.perf_counter()
            manifest_suite, materialization_counts = _materialize_manifest_suite(manifest_suite, cfg=cfg)
            print(
                "  banked task suite materialized "
                f"| unique={materialization_counts['unique_materialized_tasks']} "
                f"| elapsed={_format_elapsed_short(time.perf_counter() - materialize_start)}",
                flush=True,
            )
            task_materialization_cache: TaskMaterializationCache | None = None
        else:
            materialization_counts = {
                "mode": "lazy",
                "unique_materialized_tasks": 0,
            }
            task_materialization_cache = {}
            print("  using shared lazy bank task cache", flush=True)
    else:
        materialization_counts = {
            "mode": "online",
            "unique_materialized_tasks": 0,
        }
        task_materialization_cache = None
    manifest_summary = manifest_suite.summary()
    if task_bank_info is not None:
        manifest_summary["task_bank"] = task_bank_info.summary()
        manifest_summary["task_bank_materialization"] = {
            "mode": cfg.bank_materialization,
            **materialization_counts,
        }
    with open(outdir / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)
    flat_task_pool = _flatten_manifest_suite_tasks(manifest_suite)

    reference_task = _manifest_reference_task(manifest_suite, cfg=cfg)
    feature_channels = reference_task.feature_tensor.shape[0]
    task_dim = reference_task.task_vector.shape[0]

    def _make_model_instance() -> DensePolicyNet:
        return DensePolicyNet(
            feature_channels,
            task_dim,
            rows=cfg.rows,
            cols=cfg.cols,
            horizon=cfg.horizon,
            hidden_channels=cfg.hidden_channels,
            task_hidden_dim=cfg.task_hidden_dim,
        )

    model = _make_model_instance().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pretrain_rng = random.Random(cfg.seed + 17)
    meta_rng = random.Random(cfg.seed + 9991)
    pretrain_start_step = 0
    meta_start_step = 0
    resume_phase = "pretrain"
    meta_deploy_caches: dict[str, DeploySubsetCache] = {}
    meta_union_caches: dict[str, UnionTopCache] = {}
    # REINFORCE per-task EMA baselines: owned here (like the caches above) so
    # resume checkpoints persist them, analogous to batch-norm running stats.
    pretrain_reinforce_ema_state: dict[str, float] = {}
    meta_reinforce_ema_return_state: dict[str, float] = {}
    meta_reinforce_ema_regret_state: dict[str, float] = {}

    def _write_pretrained_results() -> None:
        deprecated_difficulty_audit = _deprecated_difficulty_audit_note()
        with open(outdir / "pretrained_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "exact_eval_phase": cfg.pretrained_exact_eval_phase,
                    "quickcheck": pretrained_quickcheck,
                    "report": pretrained_report,
                    "tail_gap_check": pretrained_tail_gap_check,
                    "difficulty_audit": deprecated_difficulty_audit,
                },
                f,
                indent=2,
            )

    def _save_phase_resume_checkpoint(
        phase: str,
        next_step: int,
        *,
        deploy_caches: dict[str, DeploySubsetCache] | None = None,
        union_caches: dict[str, UnionTopCache] | None = None,
    ) -> None:
        state = {
            "config_signature": _critical_cfg_signature(cfg),
            "phase": phase,
            "next_step": int(next_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pretrain_history": pretrain_history,
            "meta_history": meta_history,
            "pretrained_quickcheck": pretrained_quickcheck,
            "pretrained_report": pretrained_report,
            "pretrained_tail_gap_check": pretrained_tail_gap_check,
            "final_report": final_report,
            "final_tail_gap_check": final_tail_gap_check,
            "pretrain_reference_returns": pretrain_reference_returns,
            "pretrain_rng_state": pretrain_rng.getstate(),
            "meta_rng_state": meta_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "meta_deploy_caches": serialize_deploy_caches(deploy_caches or {}),
            "meta_union_caches": serialize_union_caches(union_caches or {}),
            "pretrain_reinforce_ema_state": dict(pretrain_reinforce_ema_state),
            "meta_reinforce_ema_return_state": dict(meta_reinforce_ema_return_state),
            "meta_reinforce_ema_regret_state": dict(meta_reinforce_ema_regret_state),
        }
        _save_resume_checkpoint(outdir, state)

    if cfg.resume_checkpoint:
        print(f"Resuming from checkpoint {cfg.resume_checkpoint}", flush=True)
        resume_state = _load_resume_checkpoint(cfg.resume_checkpoint)
        _validate_resume_checkpoint_config(resume_state, cfg)
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        pretrain_history = list(resume_state.get("pretrain_history", []))
        meta_history = list(resume_state.get("meta_history", []))
        if isinstance(resume_state.get("pretrained_quickcheck"), dict):
            pretrained_quickcheck = resume_state["pretrained_quickcheck"]
        if isinstance(resume_state.get("pretrained_report"), dict):
            pretrained_report = resume_state["pretrained_report"]
        if isinstance(resume_state.get("pretrained_tail_gap_check"), dict):
            pretrained_tail_gap_check = resume_state["pretrained_tail_gap_check"]
        if isinstance(resume_state.get("final_report"), dict):
            final_report = resume_state["final_report"]
        if isinstance(resume_state.get("final_tail_gap_check"), dict):
            final_tail_gap_check = resume_state["final_tail_gap_check"]
        if isinstance(resume_state.get("pretrain_reference_returns"), dict):
            pretrain_reference_returns = resume_state["pretrain_reference_returns"]
        if resume_state.get("pretrain_rng_state") is not None:
            pretrain_rng.setstate(resume_state["pretrain_rng_state"])
        if resume_state.get("meta_rng_state") is not None:
            meta_rng.setstate(resume_state["meta_rng_state"])
        if resume_state.get("torch_rng_state") is not None:
            torch.set_rng_state(torch.as_tensor(resume_state["torch_rng_state"], dtype=torch.uint8).cpu())
        if isinstance(resume_state.get("pretrain_reinforce_ema_state"), dict):
            pretrain_reinforce_ema_state.update(resume_state["pretrain_reinforce_ema_state"])
        if isinstance(resume_state.get("meta_reinforce_ema_return_state"), dict):
            meta_reinforce_ema_return_state.update(resume_state["meta_reinforce_ema_return_state"])
        if isinstance(resume_state.get("meta_reinforce_ema_regret_state"), dict):
            meta_reinforce_ema_regret_state.update(resume_state["meta_reinforce_ema_regret_state"])
        resume_phase = str(resume_state.get("phase", "pretrain"))
        next_step = int(resume_state.get("next_step", 0))
        if resume_phase == "pretrain":
            pretrain_start_step = next_step
        elif resume_phase == "meta":
            meta_start_step = next_step
            meta_deploy_caches = deserialize_deploy_caches(resume_state.get("meta_deploy_caches"))
            meta_union_caches = deserialize_union_caches(resume_state.get("meta_union_caches"))
        elif resume_phase == "pretrained_eval":
            if cfg.pretrained_exact_eval_phase == "post":
                resume_phase = "meta"
            else:
                pass
        elif resume_phase in {"final_eval", "complete"}:
            # Meta history holds one entry per completed step, so a raised
            # --meta-steps re-enters the meta loop from where training ended
            # (extension); otherwise fall through and reuse the stored reports.
            completed_meta_steps = len(meta_history)
            if cfg.meta_steps > completed_meta_steps:
                print(
                    f"  extending meta training: {completed_meta_steps} -> {cfg.meta_steps} steps",
                    flush=True,
                )
                resume_phase = "meta"
                meta_start_step = completed_meta_steps
                meta_deploy_caches = deserialize_deploy_caches(resume_state.get("meta_deploy_caches"))
                meta_union_caches = deserialize_union_caches(resume_state.get("meta_union_caches"))
                final_report = None
                final_tail_gap_check = None
        else:
            raise ValueError(f"Unsupported resume phase in checkpoint: {resume_phase}")
        print(f"  resume phase={resume_phase} next_step={next_step}", flush=True)
        if resume_phase == "complete" and (outdir / "final_results.json").exists():
            with open(outdir / "final_results.json", "r", encoding="utf-8") as f:
                return json.load(f)
    elif cfg.init_checkpoint:
        print(f"Loading checkpoint from {cfg.init_checkpoint}", flush=True)
        _load_checkpoint(model, cfg.init_checkpoint)
        if cfg.reuse_pretrained_results:
            print(f"Reusing pretrained results from {cfg.reuse_pretrained_results}", flush=True)
            pretrained_report, pretrained_tail_gap_check, pretrained_quickcheck = _load_reused_pretrained_results(
                cfg.reuse_pretrained_results,
                cfg=cfg,
            )
            _write_pretrained_results()
        if cfg.skip_pretrain:
            print("Skipping pretraining and using loaded checkpoint as pretrained policy.", flush=True)
            _save_checkpoint(model, outdir, "pretrained_checkpoint")
            resume_phase = "pretrained_eval" if cfg.pretrained_exact_eval_phase == "pre" else "meta"

    def _pretrain_checkpoint_callback(next_step: int, history_rows: list[dict]) -> None:
        nonlocal pretrain_history
        pretrain_history = list(history_rows)
        _save_phase_resume_checkpoint("pretrain", next_step)

    def _meta_checkpoint_callback(
        next_step: int,
        history_rows: list[dict],
        deploy_caches: dict[str, DeploySubsetCache] | None = None,
        union_caches: dict[str, UnionTopCache] | None = None,
    ) -> None:
        nonlocal meta_history
        meta_history = list(history_rows)
        _save_phase_resume_checkpoint(
            "meta", next_step,
            deploy_caches=deploy_caches,
            union_caches=union_caches,
        )

    if resume_phase == "pretrain":
        pretrain_pool = _pretrain_tasks(manifest_suite, cfg)
        if cfg.pretrain_task_source != "pretrain_train":
            print(
                f"  pretrain pool overridden via --pretrain-task-source={cfg.pretrain_task_source} "
                f"(|tasks|={len(pretrain_pool)})",
                flush=True,
            )
        pretrain_estimator = (
            cfg.grad_estimator
            if cfg.pretrain_grad_estimator == "inherit"
            else cfg.pretrain_grad_estimator
        )
        if pretrain_estimator == "exact":
            pretrain_history = _pretrain_exact(
                model,
                optimizer,
                pretrain_pool,
                cfg=cfg,
                rng=pretrain_rng,
                materialization_cache=task_materialization_cache,
                start_step=pretrain_start_step,
                history=pretrain_history,
                checkpoint_callback=_pretrain_checkpoint_callback,
            )
        else:
            pretrain_history = _pretrain_reinforce(
                model,
                optimizer,
                pretrain_pool,
                cfg=cfg,
                rng=pretrain_rng,
                materialization_cache=task_materialization_cache,
                start_step=pretrain_start_step,
                history=pretrain_history,
                checkpoint_callback=_pretrain_checkpoint_callback,
                ema_state=pretrain_reinforce_ema_state,
            )
        _save_checkpoint(model, outdir, "pretrained_checkpoint")
        if cfg.pretrained_exact_eval_phase == "pre":
            _save_phase_resume_checkpoint("pretrained_eval", 0)
            resume_phase = "pretrained_eval"
        else:
            _save_phase_resume_checkpoint(
                "meta", 0,
                deploy_caches=meta_deploy_caches,
                union_caches=meta_union_caches,
            )
            resume_phase = "meta"
    elif resume_phase == "pretrained_eval" and not (outdir / "pretrained_checkpoint.pt").exists():
        _save_checkpoint(model, outdir, "pretrained_checkpoint")

    if (
        pretrained_quickcheck is None
        and cfg.pretrained_quickcheck_task_count > 0
        and resume_phase in {"pretrained_eval", "meta"}
        and meta_start_step == 0
    ):
        pretrained_quickcheck = _evaluate_flat_task_sample_quickcheck(
            model,
            flat_task_pool,
            cfg=cfg,
            label="pretrained",
            sample_count=cfg.pretrained_quickcheck_task_count,
            seed=cfg.seed + 41011,
            materialization_cache=task_materialization_cache,
        )
        _write_pretrained_results()

    if (
        cfg.pretrained_exact_eval_phase == "pre"
        and (pretrained_report is None or pretrained_tail_gap_check is None)
    ):
        pretrained_report, pretrained_tail_gap_check, pretrained_trace_bundle = _evaluate_policy_report_and_tail_gap(
            model,
            manifest_suite,
            cfg=cfg,
            label="pretrained",
            materialization_cache=task_materialization_cache,
        )
        _write_pretrained_results()
    deprecated_difficulty_audit = _deprecated_difficulty_audit_note()
    aux_tasks = _meta_aux_tasks(manifest_suite, cfg)
    if cfg.meta_aux_loss_mode == "anchored_pretrain_return" and cfg.meta_aux_loss_weight > 0:
        if pretrain_reference_returns is None:
            pretrain_reference_returns = _pretrain_reference_returns(
                model,
                aux_tasks,
                cfg=cfg,
                materialization_cache=task_materialization_cache,
            )
    # --- IOG penalty baseline ---
    # Snapshot pretrained per-task regret across all meta-manifest fit AND
    # deploy pools. Used to add a squared-hinge penalty on regret growing
    # past the baseline, as a soft replacement for the masked IOG gradient
    # (which covers BOTH fit and deploy gradients; see ols_utils.py:506-566).
    iog_penalty_baseline_regrets: dict[str, float] = {}
    if cfg.iog_penalty_coef > 0:
        if cfg.split_mode != "disjoint":
            raise ValueError(
                "--iog-penalty-coef currently only supports --split-mode disjoint."
            )
        if meta_start_step > 0:
            raise ValueError(
                "--iog-penalty-coef does not yet support resume from "
                "meta_start_step > 0; the baseline would use the resumed "
                "checkpoint state instead of the pretrained state."
            )
        log_msg = "Caching pretrained per-task regret as IOG-penalty baseline (fit + deploy) ..."
        try:
            print(log_msg, flush=True)
        except Exception:
            pass
        seen_keys: set[str] = set()
        baseline_tasks: list[TaskRecord] = []
        for manifest in manifest_suite.meta_manifests:
            for task in list(manifest.fit_tasks) + list(manifest.deploy_tasks):
                key = _task_key(task)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                baseline_tasks.append(task)
        if baseline_tasks:
            _, baseline_regrets = _exact_returns_and_regrets(
                model,
                baseline_tasks,
                cfg=cfg,
                require_grad=False,
                materialization_cache=task_materialization_cache,
            )
            for task, regret in zip(baseline_tasks, baseline_regrets.detach().cpu().tolist()):
                iog_penalty_baseline_regrets[_task_key(task)] = float(regret)
        try:
            print(f"  cached baseline for {len(iog_penalty_baseline_regrets)} unique tasks", flush=True)
        except Exception:
            pass

    _save_phase_resume_checkpoint(
        "meta", meta_start_step,
        deploy_caches=meta_deploy_caches,
        union_caches=meta_union_caches,
    )
    if resume_phase in {"pretrained_eval", "meta"} and meta_start_step < cfg.meta_steps:
        if cfg.grad_estimator == "exact":
            meta_history, meta_training_cache_summary = _meta_train_exact(
                model,
                optimizer,
                aux_tasks,
                manifest_suite.meta_manifests,
                cfg=cfg,
                rng=meta_rng,
                reference_returns=pretrain_reference_returns,
                materialization_cache=task_materialization_cache,
                start_step=meta_start_step,
                history=meta_history,
                deploy_caches=meta_deploy_caches,
                union_caches=meta_union_caches,
                checkpoint_callback=_meta_checkpoint_callback,
                iog_penalty_baseline_regrets=iog_penalty_baseline_regrets,
            )
        else:
            meta_history = _meta_train_reinforce(
                model,
                optimizer,
                aux_tasks,
                manifest_suite.meta_manifests,
                cfg=cfg,
                rng=meta_rng,
                reference_returns=pretrain_reference_returns,
                materialization_cache=task_materialization_cache,
                start_step=meta_start_step,
                history=meta_history,
                checkpoint_callback=_meta_checkpoint_callback,
                ema_return_state=meta_reinforce_ema_return_state,
                ema_regret_state=meta_reinforce_ema_regret_state,
            )
            meta_training_cache_summary = None
    _save_checkpoint(model, outdir, "final_checkpoint")
    _save_phase_resume_checkpoint("final_eval", 0)
    if final_report is None or final_tail_gap_check is None:
        final_report, final_tail_gap_check, final_trace_bundle = _evaluate_policy_report_and_tail_gap(
            model,
            manifest_suite,
            cfg=cfg,
            label="final",
            materialization_cache=task_materialization_cache,
        )
        _save_phase_resume_checkpoint("final_eval", 0)
    if (
        cfg.pretrained_exact_eval_phase == "post"
        and (pretrained_report is None or pretrained_tail_gap_check is None)
    ):
        print("Loading pretrained checkpoint for deferred exact evaluation", flush=True)
        pretrained_model = _make_model_instance().to(cfg.device)
        _load_checkpoint(pretrained_model, str(outdir / "pretrained_checkpoint.pt"))
        pretrained_report, pretrained_tail_gap_check, pretrained_trace_bundle = _evaluate_policy_report_and_tail_gap(
            pretrained_model,
            manifest_suite,
            cfg=cfg,
            label="pretrained",
            materialization_cache=task_materialization_cache,
        )
        _write_pretrained_results()

    history = {
        "pretrain": pretrain_history,
        "meta": meta_history,
    }
    with open(outdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if final_report is None or pretrained_report is None or manifest_summary is None:
        raise RuntimeError("RLQuant training did not produce final reports")
    if final_trace_bundle is None:
        print("Recomputing final forecast traces for archival", flush=True)
        _report, _tail_gap, final_trace_bundle = _evaluate_policy_report_and_tail_gap(
            model,
            manifest_suite,
            cfg=cfg,
            label="final_trace_archive",
            materialization_cache=task_materialization_cache,
        )
    if pretrained_trace_bundle is None and (outdir / "pretrained_checkpoint.pt").exists():
        print("Recomputing pretrained forecast traces for archival", flush=True)
        pretrained_model = _make_model_instance().to(cfg.device)
        _load_checkpoint(pretrained_model, str(outdir / "pretrained_checkpoint.pt"))
        _report, _tail_gap, pretrained_trace_bundle = _evaluate_policy_report_and_tail_gap(
            pretrained_model,
            manifest_suite,
            cfg=cfg,
            label="pretrained_trace_archive",
            materialization_cache=task_materialization_cache,
        )
    _save_forecast_trace_archive(
        outdir,
        pretrained_traces=pretrained_trace_bundle,
        meta_learned_traces=final_trace_bundle,
        cfg=cfg,
    )
    results = {
        "pretrained": pretrained_report,
        "pretrained_quickcheck": pretrained_quickcheck,
        "meta_learned": final_report,
        "tail_gap_checks": {
            "pretrained": pretrained_tail_gap_check,
            "meta_learned": final_tail_gap_check,
        },
        "difficulty_audit": _deprecated_difficulty_audit_note(),
        "manifest_summary": manifest_summary,
        "support_audit": support_audit,
        "meta_training_cache_summary": meta_training_cache_summary,
        "forecast_traces": {
            "archive": "forecast_traces.npz",
            "metadata": "forecast_traces_manifests.json",
            "dtype": "float32",
            "fields": ["fit_returns", "fit_regrets", "deploy_returns", "deploy_regrets"],
        },
    }
    if task_bank_info is not None:
        results["task_bank"] = task_bank_info.summary()
    results["task_materialization_cache"] = {
        "mode": cfg.bank_materialization if task_bank_info is not None else "online",
        "unique_materialized_tasks": len(task_materialization_cache) if task_materialization_cache is not None else 0,
    }
    with open(outdir / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _save_phase_resume_checkpoint("complete", cfg.meta_steps)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train RLQuant on procedural gridworld families")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--family-suite",
        choices=["naturalistic", "pressure_plate", "pressure_plate_open_room_only", "trap_open_room_only", "hybrid"],
        default="naturalistic",
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--step-cost", type=float, default=-0.01)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--catastrophe-penalty", type=float, default=-1.0)
    parser.add_argument("--min-start-goal-distance", type=int, default=7)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--task-hidden-dim", type=int, default=64)

    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--meta-steps", type=int, default=100)
    parser.add_argument("--meta-manifest-count", type=int, default=20)
    parser.add_argument("--heldout-manifest-count", type=int, default=5)
    parser.add_argument("--meta-manifests-per-step", type=int, default=4)
    parser.add_argument("--meta-deploy-cache-top-b", type=int, default=0)
    parser.add_argument("--meta-deploy-cache-refresh-interval", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--reuse-pretrained-results", default="")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--deterministic-debug", action="store_true")
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-estimator", choices=["exact", "reinforce"], default="exact")
    parser.add_argument(
        "--pretrain-grad-estimator",
        choices=["inherit", "exact", "reinforce"],
        default="inherit",
    )
    parser.add_argument("--reinforce-rollouts-per-task", type=int, default=8)
    parser.add_argument(
        "--reinforce-baseline",
        choices=["rollout_mean", "optimal_value", "ema"],
        default="rollout_mean",
    )
    parser.add_argument("--reinforce-ema-decay", type=float, default=0.9)
    parser.add_argument(
        "--reinforce-meta-estimator",
        choices=["plugin", "batch"],
        default="plugin",
    )
    parser.add_argument("--reinforce-split-samples", action="store_true")
    parser.add_argument(
        "--reinforce-score",
        choices=["regret", "neg_return"],
        default="regret",
    )

    parser.add_argument("--ols-top-k", type=int, default=10)
    parser.add_argument("--plotting-position", choices=["gringorten", "weibull"], default="weibull")
    parser.add_argument(
        "--extrapolated-weighting",
        choices=["rank_uniform", "deploy_log_uniform", "deploy_uniform"],
        default="rank_uniform",
    )
    parser.add_argument("--ols-score-mode", choices=["raw_regret", "log_regret"], default="raw_regret")
    parser.add_argument(
        "--meta-loss-mode",
        choices=["ols_coeff", "ols_primal", "ols_primal_weighted", "ols_primal_extrapolated"],
        default="ols_coeff",
    )
    parser.add_argument(
        "--meta-aux-loss-mode",
        choices=["none", "pretrain_return", "anchored_pretrain_return"],
        default="none",
    )
    parser.add_argument("--meta-aux-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--meta-aux-task-source",
        choices=["pretrain_train", "meta_fit"],
        default="pretrain_train",
    )
    parser.add_argument(
        "--pretrain-task-source",
        choices=["pretrain_train", "meta_union", "meta_deploy"],
        default="pretrain_train",
        help=(
            "Override the pretrain task pool. Default uses pretrain_train; "
            "set meta_union (M1 SFT-on-union baseline) or meta_deploy (M3 oracle) "
            "to swap in tasks drawn from the meta manifests. Pair with --meta-steps 0."
        ),
    )
    parser.add_argument("--grad-mode", choices=["fit", "eval", "both"], default="fit")
    parser.add_argument("--improving-only-grad", action="store_true", default=False)
    parser.add_argument(
        "--iog-scope",
        choices=["both", "deploy_only", "fit_only"],
        default="both",
        help=(
            "Scope of the improving-only mask. 'both' (default): mask both "
            "deploy- and fit-side gradients that would worsen per-task regret. "
            "'deploy_only': only mask deploy-side. 'fit_only': symmetric "
            "ablation. Ignored when --improving-only-grad is not set."
        ),
    )
    parser.add_argument(
        "--iog-penalty-coef", type=float, default=0.0,
        help=(
            "Squared-hinge penalty coefficient on per-task regret growing past "
            "pretrained baseline. 0 disables. Penalty added to meta-loss; "
            "scoped to the deploy candidate subset that feeds the OLS loss "
            "(same scope as IOG mask). Recommended starting point for RL: "
            "1.0 (regret magnitudes ~1.0). LM value scale differs (log_prob "
            "in [-25,0]) and uses a much smaller coef of 0.05 in train_lquant.py."
        ),
    )
    parser.add_argument(
        "--calibration-aware", action="store_true", default=False,
        help=(
            "Train the CALIBRATED forecast: a stop-gradient affine map (fit "
            "leave-one-manifest-out over a persistent buffer of worst-rank "
            "(forecast, actual) pairs) is applied before the squared error, "
            "mirroring the deployed post-hoc calibrator. Requires "
            "meta_loss_mode=ols_primal_extrapolated and grad_estimator=exact."
        ),
    )
    parser.add_argument(
        "--calibration-mode", choices=["shift", "shift_scale"], default="shift_scale",
        help="Affine form mirrored in training: shift only (beta) or shift+scale (alpha,beta).",
    )
    parser.add_argument(
        "--calibration-min-points", type=int, default=5,
        help="Donor manifests required before calibrating (identity forecast until then).",
    )
    parser.add_argument(
        "--calibration-ridge-alpha", type=float, default=0.05,
        help="Slope shrink toward 1 (dimensionless): alpha=(alpha_ols+r)/(1+r). Keeps alpha>0.",
    )
    parser.add_argument(
        "--calibration-ema", type=float, default=0.0,
        help="Per-manifest EMA decay for buffer contributions (0 = exact replacement).",
    )
    parser.add_argument(
        "--bootstrap-permutations", type=int, default=0,
        help=(
            "If >=1 and meta_loss_mode=ols_primal_extrapolated, re-partition "
            "fit_subset ∪ deploy_tasks every step (with this many permutations "
            "per step) instead of using the fixed per-manifest partition. "
            "Breaks per-manifest asymmetric overfit. N=1 is sufficient for "
            "asymmetry-breaking; N>1 adds variance reduction. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--union-cache-top-c", type=int, default=0,
        help=(
            "Partition-invariant union top-C cache for the bootstrap path. "
            "Caches top-C indices of fit_subset∪deploy_tasks and reuses for "
            "refresh_interval steps. Set to >= deploy_extrap_k + n_fit "
            "(typically ~296 at canonical scale) for deterministic deploy "
            "coverage. 0 = disabled (full union scored every step)."
        ),
    )
    parser.add_argument(
        "--union-cache-refresh-interval", type=int, default=5,
        help="Steps between full union rescans when union-cache-top-c > 0.",
    )
    parser.add_argument("--split-mode", choices=["disjoint", "nested"], default="disjoint")
    parser.add_argument(
        "--manifest-grouping", choices=["iid", "slip_level", "dynamics_id"], default="iid",
        help="How bank records become manifests: 'iid' shuffle-and-slice (legacy), one manifest per slip level, or one per dynamics_id group.",
    )
    parser.add_argument(
        "--task-vector-slip", action="store_true",
        help="Append the task's uniform slip probability to the policy task vector (FiLM conditioning).",
    )
    parser.add_argument(
        "--blind-dynamics", action="store_true",
        help="Withhold dynamics from the observation: zero the slip feature channel (wind is never observed).",
    )
    parser.add_argument("--meta-fit-subsample-min", type=int, default=16)
    parser.add_argument("--meta-fit-subsample-max", type=int, default=0)
    parser.add_argument("--task-bank-root", default="")
    parser.add_argument("--task-bank-seed", type=int, default=-1)
    parser.add_argument("--bank-materialization", choices=["eager", "lazy"], default="lazy")
    parser.add_argument("--pretrained-exact-eval-phase", choices=["pre", "post"], default="post")
    parser.add_argument("--pretrained-quickcheck-task-count", type=int, default=0)

    parser.add_argument("--pretrain-normal-per-family", type=int, default=96)
    parser.add_argument("--pretrain-visible-challenge-per-family", type=int, default=48)
    parser.add_argument("--meta-fit-normal-per-family", type=int, default=64)
    parser.add_argument("--meta-fit-visible-challenge-per-family", type=int, default=48)
    parser.add_argument("--meta-deploy-extra-per-family", type=int, default=96)
    parser.add_argument("--meta-deploy-hard-visible-per-family", type=int, default=32)
    parser.add_argument("--heldout-fit-normal-per-family", type=int, default=48)
    parser.add_argument("--heldout-fit-decoy-per-family", type=int, default=48)
    parser.add_argument("--heldout-deploy-extra-per-family", type=int, default=96)
    parser.add_argument("--heldout-deploy-bridge-extra-per-family", type=int, default=24)
    parser.add_argument("--pressure-pretrain-per-family", type=int, default=96)
    parser.add_argument("--pressure-meta-fit-per-family", type=int, default=64)
    parser.add_argument("--pressure-meta-deploy-extra-per-family", type=int, default=96)
    parser.add_argument("--pressure-heldout-fit-per-family", type=int, default=48)
    parser.add_argument("--pressure-heldout-deploy-extra-per-family", type=int, default=96)
    parser.add_argument("--trap-pretrain-per-family", type=int, default=96)
    parser.add_argument("--trap-meta-fit-per-family", type=int, default=64)
    parser.add_argument("--trap-meta-deploy-extra-per-family", type=int, default=96)
    parser.add_argument("--trap-heldout-fit-per-family", type=int, default=48)
    parser.add_argument("--trap-heldout-deploy-extra-per-family", type=int, default=96)

    parser.add_argument("--support-audit-normal-samples", type=int, default=100000)
    parser.add_argument("--support-audit-challenge-samples", type=int, default=20000)
    parser.add_argument("--support-audit-total-min-unique", type=int, default=100000)
    parser.add_argument("--support-audit-normal-total-min-unique", type=int, default=100000)
    parser.add_argument("--support-audit-normal-min-unique-per-family", type=int, default=50000)
    parser.add_argument("--support-audit-rare-min-unique", type=int, default=5000)
    parser.add_argument("--support-audit-normal-max-collision-rate", type=float, default=0.10)
    parser.add_argument("--support-audit-rare-max-collision-rate", type=float, default=0.25)
    parser.add_argument("--support-audit-normal-min-acceptance", type=float, default=0.30)
    parser.add_argument("--support-audit-rare-min-acceptance", type=float, default=0.10)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    cfg = RLQuantConfig(
        device=args.device,
        seed=args.seed,
        family_suite=args.family_suite,
        rows=args.rows,
        cols=args.cols,
        horizon=args.horizon,
        gamma=args.gamma,
        step_cost=args.step_cost,
        goal_reward=args.goal_reward,
        catastrophe_penalty=args.catastrophe_penalty,
        min_start_goal_distance=args.min_start_goal_distance,
        hidden_channels=args.hidden_channels,
        task_hidden_dim=args.task_hidden_dim,
        pretrain_steps=args.pretrain_steps,
        meta_steps=args.meta_steps,
        meta_manifest_count=args.meta_manifest_count,
        heldout_manifest_count=args.heldout_manifest_count,
        meta_manifests_per_step=args.meta_manifests_per_step,
        meta_deploy_cache_top_b=args.meta_deploy_cache_top_b,
        meta_deploy_cache_refresh_interval=args.meta_deploy_cache_refresh_interval,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        init_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        reuse_pretrained_results=args.reuse_pretrained_results,
        skip_pretrain=bool(args.skip_pretrain),
        deterministic_debug=bool(args.deterministic_debug),
        pretrain_batch_size=args.pretrain_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_estimator=args.grad_estimator,
        pretrain_grad_estimator=args.pretrain_grad_estimator,
        reinforce_rollouts_per_task=args.reinforce_rollouts_per_task,
        reinforce_baseline=args.reinforce_baseline,
        reinforce_ema_decay=args.reinforce_ema_decay,
        reinforce_meta_estimator=args.reinforce_meta_estimator,
        reinforce_split_samples=bool(args.reinforce_split_samples),
        reinforce_score=args.reinforce_score,
        ols_top_k=args.ols_top_k,
        plotting_position=args.plotting_position,
        extrapolated_weighting=args.extrapolated_weighting,
        ols_score_mode=args.ols_score_mode,
        meta_loss_mode=args.meta_loss_mode,
        meta_aux_loss_mode=args.meta_aux_loss_mode,
        meta_aux_loss_weight=args.meta_aux_loss_weight,
        meta_aux_task_source=args.meta_aux_task_source,
        pretrain_task_source=args.pretrain_task_source,
        grad_mode=args.grad_mode,
        improving_only_grad=args.improving_only_grad,
        iog_scope=args.iog_scope,
        iog_penalty_coef=args.iog_penalty_coef,
        calibration_aware=args.calibration_aware,
        calibration_mode=args.calibration_mode,
        calibration_min_points=args.calibration_min_points,
        calibration_ridge_alpha=args.calibration_ridge_alpha,
        calibration_ema=args.calibration_ema,
        bootstrap_permutations=args.bootstrap_permutations,
        union_cache_top_c=args.union_cache_top_c,
        union_cache_refresh_interval=args.union_cache_refresh_interval,
        split_mode=args.split_mode,
        manifest_grouping=args.manifest_grouping,
        task_vector_slip=bool(args.task_vector_slip),
        blind_dynamics=bool(args.blind_dynamics),
        meta_fit_subsample_min=args.meta_fit_subsample_min,
        meta_fit_subsample_max=args.meta_fit_subsample_max,
        task_bank_root=args.task_bank_root,
        task_bank_seed=args.task_bank_seed,
        bank_materialization=args.bank_materialization,
        pretrained_exact_eval_phase=args.pretrained_exact_eval_phase,
        pretrained_quickcheck_task_count=args.pretrained_quickcheck_task_count,
        pretrain_normal_per_family=args.pretrain_normal_per_family,
        pretrain_visible_challenge_per_family=args.pretrain_visible_challenge_per_family,
        meta_fit_normal_per_family=args.meta_fit_normal_per_family,
        meta_fit_visible_challenge_per_family=args.meta_fit_visible_challenge_per_family,
        meta_deploy_extra_per_family=args.meta_deploy_extra_per_family,
        meta_deploy_hard_visible_per_family=args.meta_deploy_hard_visible_per_family,
        heldout_fit_normal_per_family=args.heldout_fit_normal_per_family,
        heldout_fit_decoy_per_family=args.heldout_fit_decoy_per_family,
        heldout_deploy_extra_per_family=args.heldout_deploy_extra_per_family,
        heldout_deploy_bridge_extra_per_family=args.heldout_deploy_bridge_extra_per_family,
        pressure_pretrain_per_family=args.pressure_pretrain_per_family,
        pressure_meta_fit_per_family=args.pressure_meta_fit_per_family,
        pressure_meta_deploy_extra_per_family=args.pressure_meta_deploy_extra_per_family,
        pressure_heldout_fit_per_family=args.pressure_heldout_fit_per_family,
        pressure_heldout_deploy_extra_per_family=args.pressure_heldout_deploy_extra_per_family,
        trap_pretrain_per_family=args.trap_pretrain_per_family,
        trap_meta_fit_per_family=args.trap_meta_fit_per_family,
        trap_meta_deploy_extra_per_family=args.trap_meta_deploy_extra_per_family,
        trap_heldout_fit_per_family=args.trap_heldout_fit_per_family,
        trap_heldout_deploy_extra_per_family=args.trap_heldout_deploy_extra_per_family,
        support_audit_normal_samples=args.support_audit_normal_samples,
        support_audit_challenge_samples=args.support_audit_challenge_samples,
        support_audit_total_min_unique=args.support_audit_total_min_unique,
        support_audit_normal_total_min_unique=args.support_audit_normal_total_min_unique,
        support_audit_normal_min_unique_per_family=args.support_audit_normal_min_unique_per_family,
        support_audit_rare_min_unique=args.support_audit_rare_min_unique,
        support_audit_normal_max_collision_rate=args.support_audit_normal_max_collision_rate,
        support_audit_rare_max_collision_rate=args.support_audit_rare_max_collision_rate,
        support_audit_normal_min_acceptance=args.support_audit_normal_min_acceptance,
        support_audit_rare_min_acceptance=args.support_audit_rare_min_acceptance,
        log_every=args.log_every,
        save_every=args.save_every,
        outdir=args.outdir,
    )
    train_rlquant(cfg)


if __name__ == "__main__":
    main()
