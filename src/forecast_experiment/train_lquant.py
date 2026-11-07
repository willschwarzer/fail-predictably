"""
train_lquant.py
===============

Fine-tune an LLM via LoRA to minimize the downstream forecast loss (L_quant):
the squared error between the FRLMB OLS forecast on a fit subset and a
ground-truth OLS forecast from a larger eval subset.

The password game task provides a clean differentiable path:
    p_θ(x) = softmax(logits[:,-1,:])[password_token]
is already continuous — no smooth-indicator surrogate is needed.

Usage::

    python -m src.forecast_experiment.train_lquant \\
        --model Qwen/Qwen3-0.6B --steps 50 --outdir ./lquant_out \\
        --disable-thinking
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import time
import warnings
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from . import baseline_cache as baseline_cache_module
from .eval_cache import (
    DeploySubsetCache,
    UnionTopCache,
    deserialize_deploy_caches,
    deserialize_union_caches,
    serialize_deploy_caches,
    serialize_union_caches,
)
from .evaluate_model import (
    forecast_accuracy_from_log_probs,
    prepare_messages,
    safe_prob_from_log_prob,
    signed_log_error_from_log_probs,
)
from .dynamics_probe import (
    GAP_FLOOR_EPS,
    certified_tail_targets,
    involved_sigma_reference,
    population_line_gaps,
)
from .lquant_prompt_bank import PromptBankInfo, resolve_prompt_bank
from .ols_utils import (
    AffineCalibrationBuffer,
    count_extrapolated_eval_points as generic_count_extrapolated_eval_points,
    differentiable_ols_coefficients as generic_differentiable_ols_coefficients,
    differentiable_ols_extrapolated_primal_loss as generic_differentiable_ols_extrapolated_primal_loss,
    differentiable_ols_extrapolated_primal_loss_bootstrap as generic_differentiable_ols_extrapolated_primal_loss_bootstrap,
    differentiable_ols_forecast as generic_differentiable_ols_forecast,
    differentiable_ols_primal_loss as generic_differentiable_ols_primal_loss,
    fit_ols_tail_from_scores,
    plan_bootstrap_permutations as generic_plan_bootstrap_permutations,
    select_top_k_indices as generic_select_top_k_indices,
)
from .password_game import (
    assign_passwords_to_targets,
    build_password_system_prompt,
    get_password_slots,
)
from .password_gcg import GCGConfig, generate_gcg_prompts
from .prompt_generators import (
    generate_password_prompts,
    get_password_prompt_family_support,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_CHEAP_FAMILIES = [
    "compositional",
    "full_random_words",
    "full_random_tokens",
    "mutate",
    "password_seeded",
    "single_token_coercion",
    "indirect_extraction",
    "benign_control",
    "instruction_confusion",
]

DEFAULT_TRAIN_FAMILIES = [
    "compositional",
    "full_random_words",
    "full_random_tokens",
    "mutate",
    "password_seeded",
    "indirect_extraction",
]

DEFAULT_TEST_FAMILIES = [
    "single_token_coercion",
    "benign_control",
    "instruction_confusion",
]


@dataclass
class LQuantConfig:
    # Model
    model_id: str = "Qwen/Qwen3-0.6B"
    device: str = ""

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Task kind: "password" (single-token password event, generated prompts) or
    # "token_union" (WildChat prompts; event = union of first-response token ids
    # carried in the bank metadata). token_union requires a prompt bank.
    task_kind: str = "password"
    # token_union only: put the union's word list in a system prompt ("Never use
    # any of the following words..."), converting the memorized-blocklist rule
    # into a context-computable one like the password task's. Canary for the
    # theory that LDNOOBW collapses slowly BECAUSE the rule is absent from the
    # prompt (2026-07-25); changes scoring semantics, so it participates in the
    # resume signature and (conditionally) in the baseline cache key.
    token_union_blocklist_prompt: bool = False
    # Meta-task batch size: 0 = all train targets per optimizer step (canonical
    # full-batch). b > 0 samples b targets per step (fresh step-seeded draw), so
    # a fixed FLOP budget buys ~33/b times more optimizer updates. Motivated by
    # a clipped-gradient argument: under grad-clip every update has the same
    # norm, so updates per hour, not tasks per update, set the collapse speed.
    targets_per_step: int = 0

    # Password game
    password_slots: int = 19
    password_policy: str = "non_disclosure"
    train_target_count: int = 0
    heldout_target_count: int = 0

    # Score-time prompt truncation. 0 = no truncation. When > 0 the tokenized
    # prompt is LEFT-sliced to this many tokens so the generation-prompt suffix
    # and (for jailbreaks) the tail of the prompt always survive. Used for the
    # long WildChat prompts; password prompts are short so the default is 0.
    score_max_len: int = 0
    # Length-sorted micro-batching under a token budget (batch grows while
    # B * L_max <= budget and B <= micro_batch_size). 0 = legacy fixed-size
    # contiguous micro-batches (keeps the password path bit-for-bit unchanged).
    score_token_budget: int = 0
    # Activation checkpointing on the backbone. Needed for long-context
    # (token_union, 8k) training where the per-target grad set of long prompts
    # otherwise exceeds GPU memory. Result-preserving (memory/compute trade).
    gradient_checkpointing: bool = False

    # Prompt pool
    cheap_prompts_per_family_per_target: int = 64
    cheap_prompts_per_family_map: Dict[str, int] = field(default_factory=dict)
    train_families: List[str] = field(default_factory=lambda: list(DEFAULT_TRAIN_FAMILIES))
    test_families: List[str] = field(default_factory=lambda: list(DEFAULT_TEST_FAMILIES))
    # Fit/deploy distribution-shift protocol. When non-empty, every slot's fit
    # sample is drawn only from `fit_families` and its deploy set only from the
    # disjoint `deploy_families` -- in train slots (meta-training) and held-out
    # slots (evaluation) alike, with identical per-family composition in every
    # slot. Cell sizes are explicit (`shift_fit_size` / `shift_deploy_size`) so
    # the paper's fit/deploy cell can be matched exactly; the deploy set is
    # thinned to size by a deterministic largest-remainder allocation across
    # the deploy families. In this mode each slot uses its FULL family pools
    # (the shared-family 50/50 halving is skipped: a slot serves only one of
    # train/held-out, and fit/deploy are disjoint by family, so the halving
    # protects nothing and would halve the reachable deploy size).
    # Empty lists = canonical IID split. Requires split_mode 'disjoint' and
    # bootstrap_permutations == 0: the bootstrap re-partitions the fit/deploy
    # union each step, which assumes the two sides are exchangeable -- exactly
    # the assumption this protocol removes.
    fit_families: List[str] = field(default_factory=list)
    deploy_families: List[str] = field(default_factory=list)
    # Required (>0) when the shift is enabled: number of fit prompts per slot.
    shift_fit_size: int = 0
    # Deploy prompts per slot; 0 = use every deploy-family record.
    shift_deploy_size: int = 0
    prompt_bank_root: str = ""
    prompt_bank_seed: int = -1

    # GCG spectrum
    # Deprecated in the training consumer: GCG families should be controlled
    # explicitly via train_families/test_families or by loading a prompt bank.
    # This flag remains only for legacy online prompt generation.
    gcg_enabled: bool = False
    gcg_steps_spectrum: List[int] = field(default_factory=lambda: [5, 10, 20, 40, 80])
    gcg_prompts_per_level_per_target: int = 16
    gcg_prompt_len: int = 24
    gcg_top_k: int = 8
    gcg_random_restarts: int = 2
    # Deprecated no-op in the training consumer; retained for compatibility
    # with legacy configs and prompt-bank generation metadata.
    gcg_train_max_steps: int = 20
    gcg_decay_alpha: float = 2.0

    # Split
    split_mode: str = "disjoint"  # disjoint | nested | resample
    fit_fraction: float = 0.3

    # OLS
    ols_top_k: int = 10
    plotting_position: str = "gringorten"
    extrapolated_weighting: str = "rank_uniform"  # rank_uniform | deploy_log_uniform | deploy_uniform
    extrapolated_loss_space: str = "log_prob"  # log_prob | score

    # Deployment scale randomization
    log_n_min: float = 3.0  # 10^3
    log_n_max: float = 6.0  # 10^6
    n_deploy_samples: int = 1  # number of deployment scales to sample per step

    # Fit size randomization (resample mode only)
    fit_size_min: int = 20
    fit_size_max: int = 0  # 0 = auto (pool_size / 2)

    # Fit subsampling (disjoint/nested modes): each step, draw a random number
    # of prompts in [fit_subsample_min, len(fit_set)] from the fit set so the
    # top-k bump structure varies.  0 = disabled (use full fit set every step).
    fit_subsample_min: int = 0

    # Loss mode
    loss_mode: str = "ols_coeff"  # ols_coeff | ols_primal | ols_primal_weighted | ols_primal_extrapolated | extrapolate (legacy)

    # Gradient flow
    grad_mode: str = "fit"  # fit | eval | both
    improving_only_grad: bool = False
    # Scope of the improving-only mask when --improving-only-grad is set.
    # 'both' (default, current behavior): mask both deploy-side and fit-side
    # gradients that would worsen per-prompt log_prob. 'deploy_only': only
    # mask deploy-side; let fit-side log_probs move freely (intended to free
    # the OLS line to follow the lowered worst-rank actual under IOG).
    # 'fit_only': symmetric ablation. Ignored when improving_only_grad=False.
    iog_scope: str = "both"  # both | deploy_only | fit_only
    # Partial IOG: instead of zeroing a worsening gradient, scale it by this
    # coefficient. 0.0 = the canonical hard mask; 1.0 = no masking at all.
    # Motivation (2026-07-25): at 4B/8B the forecast tracks the suppressed tail
    # far more slowly than at 0.6B/1.7B, and a hard mask discards every
    # gradient component that would raise a score even when doing so would
    # improve the forecast overall. A partial mask keeps the safety bias while
    # letting the OLS line follow the tail. Ignored when improving_only_grad=False.
    iog_worsening_coef: float = 0.0
    # How the fit-side improving-only mask decides a point's gradient sign.
    # 'closed_form' (default, and what every run to date used) differentiates the
    # SCORE-space squared error analytically. The LM recipe optimizes the
    # log-prob-space error instead, and the two disagree in sign on ~6% of active
    # fit points, so the mask lets through a worsening gradient it is meant to
    # block. 'exact' differentiates whichever loss is configured, via one tiny
    # autograd pass over a k-element leaf, and blocks correctly.
    iog_fit_mask: str = "closed_form"  # closed_form | exact
    # Earliest step eligible for the `best/` checkpoint. 0 means step 0 can win,
    # which is informative: it says training never improved the train-side
    # forecast error. Excluded from _critical_cfg_signature (bookkeeping only).
    best_ckpt_min_step: int = 0
    # Soft replacement for improving_only_grad: a squared-hinge penalty on
    # per-prompt log_prob (LM) / regret (RL) growing past a pretrained
    # baseline. 0 = disabled. The penalty is added to the meta-loss and is a
    # well-defined gradient on a real objective (unlike the masked IOG
    # gradient, which can be a biased subgradient). Penalty is scoped to the
    # same top-k extrapolated prompts that feed the OLS loss.
    iog_penalty_coef: float = 0.0

    # Calibration-aware forecastability training. At deployment the forecast is
    # affinely corrected (alpha*Q + beta) by a calibrator fit on the train
    # targets; with this enabled, training optimizes the CALIBRATED forecast so
    # the model spends its capacity on the residual the affine map cannot fix.
    # The calibrator is fit leave-one-group-out over a persistent buffer of
    # worst-rank (forecast, actual) pairs across targets (no extra forward
    # passes, no gradient through the fit). Only takes effect when
    # loss_mode == "ols_primal_extrapolated".
    calibration_aware: bool = False
    calibration_mode: str = "shift_scale"  # shift | shift_scale
    calibration_min_points: int = 5        # warmup donor groups before calibrating
    calibration_ridge_alpha: float = 0.05  # slope shrink toward 1 (dimensionless)
    calibration_ema: float = 0.0           # per-group EMA of contributions (0 = exact)

    # Bootstrap-permutation meta-loss. >=1 enables permuting the union of
    # fit_subset+eval_subset every meta-step. Only takes effect when
    # loss_mode == "ols_primal_extrapolated". 0 = disabled (use the standard
    # fixed-partition path). Re-partitioning per step breaks per-manifest
    # asymmetric overfit (fit-side and deploy-side identities cycle, so the
    # LoRA can't encode "raise these specific fit prompts, leave these
    # deploy prompts alone"). N>1 adds variance reduction by averaging over
    # multiple per-step permutations; N=1 is sufficient for asymmetry-
    # breaking and is the cheapest configuration.
    bootstrap_permutations: int = 0
    # Partition-invariant union top-C cache for the bootstrap path. Caches
    # the top-C indices of fit_subset+eval_subset and intersects with each
    # fresh permutation. Set to >= ols_top_b + n_fit (typically 296 for
    # canonical scale) for deterministic deploy-top-B coverage. 0 = disabled
    # (full union scored every step, the original bootstrap behavior).
    union_cache_top_c: int = 0
    union_cache_refresh_interval: int = 5

    # Anchor-gated worsening mask (2026-07-30). Generalizes the IOG mask: each
    # prompt carries an anchor -- its log_prob under the FROZEN base model --
    # and the gradient that would raise its log_prob is scaled by how far
    # training has already pushed it below that anchor,
    # coef = clip((anchor - now) / margin, 0, 1). A prompt at its starting
    # score is hard-blocked exactly as today; a prompt suppressed by `margin`
    # nats or more is free to move back up. Motivation: the hard mask forbids
    # every upward move, so the only way it can reduce forecast error is to
    # compress the tail downward, which is the "abolished extrapolation"
    # failure. Letting over-suppressed points recover gives the model a way to
    # restore tail SHAPE while still never making any prompt worse than the
    # base model. 0 = disabled (canonical hard mask; bit-identical).
    # Requires improving_only_grad, split_mode != resample, fit_subsample_min 0.
    anchor_gate_margin: float = 0.0
    # Squared-hinge penalty on any prompt sitting above its base-model score,
    # added to the meta-loss over the prompts that step scored with gradient.
    # This is an alternative to the gate: instead of removing the
    # constraint on worsening gradients, add a force that pushes a prompt back
    # down in proportion to how far above its starting score it has drifted.
    # 0 = disabled. Needs the same anchors the gate needs, so the same guards
    # apply; the two can be combined but are meant to be compared.
    anchor_penalty_coef: float = 0.0

    # Divide the forecast loss by the variance of the union's top-m scores,
    # differentiably, normalized so the factor is 1 at step 0. The loss is
    # homogeneous of degree two in how far the tail reaches below the bulk, and
    # so is its irreducible sampling floor, so an unnormalized loss can always be
    # reduced by pulling the tail in whatever its shape. This makes the objective
    # scale-invariant, leaving only the shape to descend on. 0 = disabled.
    loss_scale_norm_m: int = 0

    # One-sided tail-spread constraint (2026-07-30). Every consecutive v-space
    # gap among the union's top-m order statistics must stay at or above the
    # corresponding gap of the population OLS line through the ANCHOR pool's
    # top-m -- the corrected tail: same level and reach as the base pool, zero
    # shape mismatch. The line is the smallest tail the constraint admits, and
    # wider tails cost more sampling floor, so minimizing the forecast loss
    # over the feasible set lands exactly on the line. Per-gap floors close the
    # loopholes coarser constraints leave open: an aggregate (variance) can be
    # preserved while the top compresses, and per-rank value floors admit a
    # plateau raised from below; a gap cannot shrink without violating its own
    # floor. Implemented as an l1 hinge with weight gap_floor_lambda -- an
    # EXACT penalty, so any lambda above the largest dual multiplier gives the
    # constrained optimum itself, not a lambda-dependent compromise. One-sided:
    # extra spread stays feasible and is discouraged only by the loss itself.
    # Needs the anchor machinery (same guards). 0 = disabled.
    gap_floor_m: int = 0
    gap_floor_lambda: float = 0.0
    # Window generalization: with block B > 1 the floors are on every B-rank
    # span of the top-m order statistics instead of every consecutive gap. The
    # per-gap arms (block 1) trained against a constraint whose features
    # (~0.02-0.05 nat gaps) sat below the scoring channel's pass-to-pass noise
    # (~0.05-0.07 nats/prompt): spans of B ranks grow the feature ~B-fold while
    # the pass noise on a span stays ~sd*sqrt(2), so B in the tens puts the
    # signal back above the channel. Slack is subtracted from every floor and
    # should be set from the measured pass noise (~2-3x sd*sqrt(2)); the
    # per-gap default GAP_FLOOR_EPS was calibrated to fp32 rounding only.
    gap_floor_block: int = 1
    gap_floor_slack: float = GAP_FLOOR_EPS
    # Open violated spans only by pushing the lower point down (upper side
    # detached). The un-gated hinge's upward half is the one strong force
    # outside the improving-only mask; in round 1 it ratcheted the pools
    # into the ceiling. Mirrors IOG's one-sidedness; the anchor gate
    # supplies the restoring force from below.
    gap_floor_level_gated: bool = False
    # The alternative one-sided spread constraint: a floor on the std of the
    # realized involved set (this step's fit-partition top-k plus the eval
    # partition's extrapolated ranks), referenced to the LINE through the
    # anchor pool and averaged across the K bootstrap permutations before the
    # hinge (so partition noise shrinks by sqrt(K)). The reference must be the
    # line's sigma, not the observed pool's: the observed reference mis-pins
    # the optimum's scale by the sigma the observed curvature carries
    # (measured +47%/-16% across the two cells), which either lets the loss
    # buy a ceiling-ward contraction of the straight tail or excludes the
    # corrected tail outright. margin is subtracted from the reference to
    # absorb residual partition noise (~2 sd of the realized sigma / sqrt(K)).
    sigma_floor_lambda: float = 0.0
    sigma_floor_margin: float = 0.0
    # Two-sided variant: penalize |sigma_bar - reference| beyond the margin
    # instead of only sigma_bar below it. The one-sided floor lets training
    # expand the involved set's spread freely; this pins it to the anchor
    # line's spread from both sides (dead zone of +/- margin).
    sigma_floor_symmetric: bool = False
    # How many top anchor ranks the sigma reference straightens (linearize's
    # m). Must cover every rank the involved set can read: at M250 k50 the fit
    # top-50 reaches rank ~460, so the default 250 makes the reference read the
    # displaced bulk and demand 2-2.7x the pool's own spread. 250 reproduces
    # the earlier runs; the minimal-change launchers pass a larger value.
    sigma_ref_m: int = 250
    # Certified-tail distillation: a squared-error pull of the anchor's top
    # cert_distill_ranks prompts, read through the constraint channel, onto the
    # certified tail -- the straight line that raises no prompt above its anchor
    # value, keeps the sigma band's spread, and minimizes the split-averaged
    # forecast error (dynamics_probe.certified_tail_targets). Asks directly for
    # the floor the forecast loss is meant to reach. Needs the sigma floor,
    # which supplies the channel and the band. 0 = off.
    cert_distill_coef: float = 0.0
    cert_distill_ranks: int = 300
    # Space of the squared pull: "log_prob" (nats) or "logit" (event log-odds,
    # u = v - log(1 - e^v)). d v / d(event-logit shift) = 1 - p, so a log-prob
    # loss hands a near-ceiling prompt ~1% of a bulk prompt's gradient at
    # p = 0.99; in log-odds the derivative is exactly 1 at every p.
    cert_distill_space: str = "log_prob"
    # Weight on the forecast loss itself. 0 with cert_distill_coef > 0 trains on
    # the certified targets alone (the other terms keep their weights).
    forecast_loss_coef: float = 1.0
    # With the forecast loss off, the bootstrap's dropout-on grad pass feeds only
    # the anchor penalty, and every prompt it scores inside the constraint
    # channel is scored a second time there. True reads the penalty from the
    # channel's (dropout-off) pass wherever the channel covers a prompt and drops
    # those prompts from the bootstrap pass, so the penalty differs only by LoRA
    # dropout noise. At k50 (an 889-prompt channel) the double scoring is what
    # overflows a 46 GB card. With the forecast loss on, the loss
    # reads the same merged scores, so it sees the channel's dropout-off values
    # for channel prompts (LoRA dropout 0.05) and the bootstrap's values for the
    # rest; with it off, its input is unchanged. Needs no scale normalization,
    # the other reader of the bootstrap pass.
    cert_distill_shared_pass: bool = False

    # Per-step score-space dumps for the learning-dynamics analysis. Purely
    # diagnostic: excluded from the resume config signature, and every write is
    # guarded so an unset dir leaves the training path untouched.
    dynamics_dump_dir: str = ""
    dynamics_dump_every: int = 1

    # Training
    optimizer: str = "adamw"  # adamw | sgd
    steps: int = 200
    lr: float = 1e-4
    grad_clip: float = 1.0
    # Deprecated compatibility knob: fit top-k is now recomputed from the
    # current forward pass every step to avoid optimizing against stale ranks.
    topk_refresh_interval: int = 10
    # Eval deploy cache: score only the cached top-B eval prompts per target
    # instead of the full eval pool.  0 = disabled.
    eval_cache_top_b: int = 0
    eval_cache_refresh_interval: int = 10
    # Disk-backed cache for the baseline test eval. When set, the BASE-model
    # eval that runs once per training start is hash-keyed on the cache-
    # relevant config fields and cached at <baseline_eval_cache_dir>. Multi-
    # seed sweeps that vary only seed-related fields incur the eval cost
    # only once. "" = disabled (existing behavior).
    baseline_eval_cache_dir: str = ""
    micro_batch_size: int = 32
    seed: int = 42

    # KL regularization
    kl_coef: float = 0.0               # weight on KL term (0 = disabled)
    kl_trunc_len: int = 8              # tokens to sample for truncated KL
    kl_trunc_every: int = 5            # compute truncated KL every N steps
    kl_batch_size: int = 16            # prompts per KL batch (password reuse or external)
    kl_train_datasets: List[str] = field(default_factory=lambda: ["password"])
    # "password" = reuse task prompts; paths = external datasets (arrow dir, or a
    # .jsonl of {"p": text} rows such as the token-union bank's KL sidecar)
    kl_eval_datasets: List[str] = field(default_factory=list)
    # external datasets for eval-time KL (not used in training loss)
    # KL prompt tokenization: token_union runs use left/1024 so 8k-token WildChat
    # prompts keep their generation marker instead of being right-truncated mid-
    # prompt. Defaults preserve the legacy password behavior (right, 512).
    kl_tokenize_max_len: int = 512
    kl_truncation_side: str = "right"

    # Logging / checkpointing
    log_every: int = 10
    save_every: int = 50
    # Keep a copy of the adapter every N steps under `snapshots/step_NNNN/`, so
    # held-out pools can be scored at intermediate steps afterwards (via
    # eval_adapter_from). 0 = off. Bookkeeping only; cannot alter a weight.
    adapter_snapshot_every: int = 0
    resume_checkpoint: str = ""
    # Path to a save_pretrained adapter dir (e.g. a run's `best/`). Loads those
    # LoRA weights and jumps straight to the final held-out eval, skipping the
    # training loop. Exists to answer "what would the paper metric be at the
    # lowest-train-error checkpoint" without re-training to that step.
    eval_adapter_from: str = ""
    outdir: str = "./lquant_out"
    disable_thinking: bool = True
    baseline_only: bool = False
    shared_family_test_fraction: float = 0.5
    # Under target-level hold-out the within-target train/test halving protects
    # nothing (held-out targets are never trained on, and train targets' test
    # halves are never evaluated), so it only halves every pool. When set, each
    # train target trains on ALL its records and each held-out target is
    # evaluated on ALL its records. Off reproduces every run to date.
    target_holdout_full_pools: bool = False
    # Draw the held-out targets from the END of the seeded target shuffle, so
    # that for a given seed the held-out set does not depend on
    # train_target_count (a train-pool-count ladder then compares arms on the
    # same held-out pools). Off reproduces every run to date.
    heldout_from_end: bool = False
    # Extended bank: partition only the first `partition_base_targets` bank
    # targets (the same seeded shuffle as a bank holding just those, so the
    # same held-out pools), and train on every later target as well. 0 =
    # partition all targets (every run to date).
    partition_base_targets: int = 0


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


def _cpu_state(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _cpu_state(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_cpu_state(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_state(value) for value in obj)
    return obj


def _critical_cfg_signature(cfg: LQuantConfig) -> dict[str, object]:
    payload = asdict(cfg)
    for key in {
        "device",
        "steps",
        "log_every",
        "save_every",
        "adapter_snapshot_every",
        "resume_checkpoint",
        "outdir",
        "baseline_only",
        # micro_batch_size only controls gradient-accumulation chunking; the
        # optimizer step boundary aggregates the same total gradient regardless,
        # so cross-mb resume is correctness-preserving.
        "micro_batch_size",
        # score_token_budget only controls length-sorted batch packing (numerics
        # noise, same class as micro_batch_size); a requeue onto a different-VRAM
        # node may re-tune it, and that must not block resume.
        "score_token_budget",
        # gradient_checkpointing is a memory/compute trade with identical results;
        # toggling it across a resume must not block.
        "gradient_checkpointing",
        # `best/` checkpoint bookkeeping; changing it cannot alter any weight.
        "best_ckpt_min_step",
        # eval-only workflow field; never present in a training run's checkpoint.
        "eval_adapter_from",
        # Learning-dynamics dumps: read-only diagnostics that cannot alter a
        # weight, so turning them on or off must not block a resume.
        "dynamics_dump_dir",
        "dynamics_dump_every",
    }:
        payload.pop(key, None)
    return payload


def _validate_resume_checkpoint_config(
    checkpoint_state: dict[str, object],
    cfg: LQuantConfig,
) -> None:
    saved_signature = checkpoint_state.get("config_signature")
    if not isinstance(saved_signature, dict):
        return
    # A key ABSENT from an old checkpoint means the field did not exist when that
    # checkpoint was written, which is semantically identical to running at the
    # field's default. Treating absent as a mismatch would make every new config
    # field retroactively un-resumable: adding `iog_worsening_coef` on 2026-07-25
    # broke resume for every checkpoint written before it (two resumed jobs died
    # with "checkpoint=None != cfg=0.0" while trying to finish LDNOOBW 8B from
    # step 160), and would have killed any preempted fleet job on requeue.
    # A non-default value still blocks resume, which is the behaviour we want: it
    # would genuinely change the gradient partway through a run.
    defaults = {}
    for f in dataclasses.fields(cfg):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            # Fields like fit_families/deploy_families use default_factory=list;
            # without this branch their [] default reads as <unknown> and every
            # pre-field checkpoint becomes un-resumable.
            defaults[f.name] = f.default_factory()
    current = _critical_cfg_signature(cfg)
    failures: list[str] = []
    for key, expected in current.items():
        if key not in saved_signature:
            if key in defaults and expected == defaults[key]:
                continue
            failures.append(
                f"{key}: absent from checkpoint and cfg={expected} is not the "
                f"default {defaults.get(key, '<unknown>')}"
            )
            continue
        actual = saved_signature.get(key)
        if actual != expected:
            failures.append(f"{key}: checkpoint={actual} != cfg={expected}")
    if failures:
        raise RuntimeError("Resume checkpoint config mismatch: " + "; ".join(failures))


def _trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name in trainable_names:
            state[name] = tensor.detach().cpu().clone()
    return state


def _load_trainable_state_dict(model: torch.nn.Module, state: dict[str, object]) -> None:
    if not isinstance(state, dict):
        raise TypeError("Expected dict for trainable_state_dict")
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, tensor in state.items():
            if not torch.is_tensor(tensor):
                continue
            if name in params:
                params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))
            elif name in buffers:
                buffers[name].copy_(tensor.to(device=buffers[name].device, dtype=buffers[name].dtype))


def _move_optimizer_state_to_param_device(optimizer: torch.optim.Optimizer) -> None:
    for param, state in optimizer.state.items():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=param.device)


def _write_history_json(outdir: Path, history: list[dict]) -> None:
    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def _save_lquant_test_trace_archive(
    outdir: Path,
    *,
    prefix: str,
    target_traces: list[dict[str, object]],
    cfg: "LQuantConfig",
) -> dict[str, str]:
    archive_name = f"{prefix}test_forecast_traces.npz"
    metadata_name = f"{prefix}test_forecast_traces_metadata.json"
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "format": "lquant_test_forecast_traces_v1",
        "dtype": "float32",
        "plotting_position": cfg.plotting_position,
        "ols_top_k": int(cfg.ols_top_k),
        "extrapolated_weighting": cfg.extrapolated_weighting,
        "extrapolated_loss_space": cfg.extrapolated_loss_space,
        "targets": [],
    }
    for idx, row in enumerate(target_traces):
        prefix_key = f"target__{idx:02d}"
        fit_log_probs = np.asarray(row["fit_log_probs"], dtype=np.float32)
        deploy_log_probs = np.asarray(row["deploy_log_probs"], dtype=np.float32)
        arrays[f"{prefix_key}__fit_log_probs"] = fit_log_probs
        arrays[f"{prefix_key}__deploy_log_probs"] = deploy_log_probs
        metadata["targets"].append(
            {
                "target": str(row["target"]),
                "array_prefix": prefix_key,
                "fit_count": int(fit_log_probs.shape[0]),
                "deploy_count": int(deploy_log_probs.shape[0]),
            }
        )
    np.savez_compressed(outdir / archive_name, **arrays)
    with open(outdir / metadata_name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return {"archive": archive_name, "metadata": metadata_name}


def _load_saved_eval_suite(outdir: Path, prefix: str) -> dict[str, dict]:
    def _load_one(name: str) -> dict:
        path = outdir / f"{prefix}{name}.json"
        if not path.exists():
            if name == "kl_eval_results":
                return {}
            raise FileNotFoundError(f"Missing saved eval artifact at {path}")
        with path.open("r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload at {path}")
        return payload

    trace_artifacts: dict[str, str] = {}
    trace_archive = outdir / f"{prefix}test_forecast_traces.npz"
    trace_metadata = outdir / f"{prefix}test_forecast_traces_metadata.json"
    if trace_archive.exists() and trace_metadata.exists():
        trace_artifacts = {
            "archive": trace_archive.name,
            "metadata": trace_metadata.name,
        }

    return {
        "test_results": _load_one("test_results"),
        "test_pool_diagnostic": _load_one("test_pool_diagnostic"),
        "kl_eval_results": _load_one("kl_eval_results"),
        "test_trace_artifacts": trace_artifacts,
    }


def _cleanup_resume_checkpoints(outdir: Path) -> None:
    for path in (_resume_latest_checkpoint_path(outdir), _resume_prev_checkpoint_path(outdir)):
        if path.exists():
            path.unlink()


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _distributed_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _distributed_enabled() -> bool:
    return _distributed_world_size() > 1


def _distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_object(obj: Any, *, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    payload = [obj if _distributed_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _gather_cache_payload(local_payload: dict[str, object]) -> dict[str, object]:
    if not (dist.is_available() and dist.is_initialized()):
        return dict(local_payload)
    gathered: list[dict[str, object] | None] = [None for _ in range(_distributed_world_size())]
    dist.all_gather_object(gathered, local_payload)
    merged: dict[str, object] = {}
    for payload in gathered:
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _shard_items(items: List[str], *, rank: int, world_size: int) -> List[str]:
    if world_size <= 1:
        return list(items)
    return list(items[rank::world_size])


def _step_seed(seed: int, step: int, stream: int = 0) -> int:
    return int(seed) + 1_000_003 * int(step + 1) + 97_451 * int(stream)


def _step_rng(seed: int, step: int, stream: int = 0) -> random.Random:
    return random.Random(_step_seed(seed, step, stream))


def _target_step_rng(seed: int, step: int, target: str, stream: int = 0) -> random.Random:
    return random.Random(_step_seed(seed, step, stream) + 10_007 * _target_seed_offset(target))


# ---------------------------------------------------------------------------
# Differentiable OLS tail forecast
# ---------------------------------------------------------------------------


def differentiable_ols_forecast(
    log_probs: torch.Tensor,
    top_k_indices: torch.LongTensor,
    n_total: int,
    deploy_queries: "int | list[int]",
    plotting: str = "gringorten",
) -> torch.Tensor:
    """Differentiable FRLMB OLS tail forecast.

    Args:
        log_probs: [N] tensor of log p(password | prompt).  May carry gradients.
        top_k_indices: [K] indices into log_probs for top-K (frozen, no grad).
        n_total: total number of prompts (for plotting positions).
        deploy_queries: deployment scale M, or a list of M values to batch over.
        plotting: "gringorten" or "weibull".

    Returns:
        If deploy_queries is a scalar: scalar tensor log(p*).
        If deploy_queries is a list: [B] tensor of log(p*) for each M.
        Gradients flow through log_probs at the selected positions.
    """
    return generic_differentiable_ols_forecast(
        log_probs,
        top_k_indices,
        n_total,
        deploy_queries,
        plotting=plotting,
        score_mode="gumbel_log_prob",
    )


def differentiable_ols_coefficients(
    log_probs: torch.Tensor,
    top_k_indices: torch.LongTensor,
    n_total: int,
    plotting: str = "gringorten",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable FRLMB OLS tail fit — returns (a, b) coefficients.

    Same OLS as differentiable_ols_forecast but returns the line coefficients
    directly instead of extrapolating to a deployment scale.

    Args:
        log_probs: [N] tensor of log p(password | prompt).  May carry gradients.
        top_k_indices: [K] indices into log_probs for top-K (frozen, no grad).
        n_total: total number of prompts (for plotting positions).
        plotting: "gringorten" or "weibull".

    Returns:
        (a, b) scalar tensors — OLS intercept and slope in the Gumbel tail fit
        y = a + b * s, where s = -log(-log_prob) and y = log(survival).
    """
    return generic_differentiable_ols_coefficients(
        log_probs,
        top_k_indices,
        n_total,
        plotting=plotting,
        score_mode="gumbel_log_prob",
    )


def differentiable_ols_primal_loss(
    fit_log_probs: torch.Tensor,
    fit_indices: torch.LongTensor,
    fit_n_total: int,
    eval_log_probs: torch.Tensor,
    eval_indices: torch.LongTensor,
    eval_n_total: int,
    *,
    plotting: str = "gringorten",
    grad_mode: str = "both",
    use_weighted_decomposition: bool = False,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return generic_differentiable_ols_primal_loss(
        fit_log_probs,
        fit_indices,
        fit_n_total,
        eval_log_probs,
        eval_indices,
        eval_n_total,
        plotting=plotting,
        score_mode="gumbel_log_prob",
        grad_mode=grad_mode,
        use_weighted_decomposition=use_weighted_decomposition,
    )


def differentiable_ols_extrapolated_primal_loss(
    fit_log_probs: torch.Tensor,
    fit_indices: torch.LongTensor,
    fit_n_total: int,
    eval_log_probs: torch.Tensor,
    eval_indices: torch.LongTensor,
    eval_n_total: int,
    *,
    plotting: str = "gringorten",
    grad_mode: str = "both",
    extrapolated_weighting: str = "rank_uniform",
    loss_space: str = "log_prob",
    improving_only_grad: bool = False,
    iog_scope: str = "both",
    iog_worsening_coef: float = 0.0,
    iog_fit_mask: str = "closed_form",
    calibration_alpha: float | None = None,
    calibration_beta: float | None = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return generic_differentiable_ols_extrapolated_primal_loss(
        fit_log_probs,
        fit_indices,
        fit_n_total,
        eval_log_probs,
        eval_indices,
        eval_n_total,
        plotting=plotting,
        score_mode="gumbel_log_prob",
        grad_mode=grad_mode,
        extrapolated_weighting=extrapolated_weighting,
        loss_space=("original" if loss_space == "log_prob" else loss_space),
        improving_only_grad=improving_only_grad,
        iog_scope=iog_scope,
        iog_worsening_coef=iog_worsening_coef,
        iog_fit_mask=iog_fit_mask,
        calibration_alpha=calibration_alpha,
        calibration_beta=calibration_beta,
    )


def select_top_k_indices(log_probs: torch.Tensor, k: int) -> torch.LongTensor:
    """Return indices of the top-k largest log_probs (no grad)."""
    return generic_select_top_k_indices(log_probs, k)


class DynamicsDumper:
    """Append-only per-step score-space dumps for the learning-dynamics study.

    One ``.npz`` per dumped step holds, for every target the step touched: the
    union log_probs the step actually saw, the exact score-space gradient
    ``dL/dv_i`` over the union, the permutations drawn, and the per-permutation
    OLS/mask statistics. That is everything needed to replay a step offline --
    which prompts the objective selected, what it asked of each of them, and
    (by differencing consecutive dumps) what the model did in response.

    Cache-refresh steps already score the whole union, so on those steps the
    dump is a free full-distribution snapshot; on cache-hit steps only the
    cached top-C entries are fresh and ``union_is_full`` is 0.
    """

    def __init__(self, root: Path, *, every: int, rank: int = 0) -> None:
        self.root = Path(root)
        self.every = max(1, int(every))
        self.rank = int(rank)
        self.root.mkdir(parents=True, exist_ok=True)
        self._pending: List[Dict[str, object]] = []

    def wants(self, step: int) -> bool:
        return (step % self.every) == 0

    def add_target(self, **payload: object) -> None:
        self._pending.append(dict(payload))

    def flush(self, step: int) -> None:
        if not self._pending:
            return
        out: Dict[str, object] = {}
        names: List[str] = []
        for idx, rec in enumerate(self._pending):
            target = str(rec.pop("target"))
            names.append(target)
            for key, val in rec.items():
                out[f"{idx}/{key}"] = np.asarray(val)
        out["targets"] = np.asarray(names)
        path = self.root / f"step_{step:04d}_rank{self.rank}.npz"
        np.savez_compressed(path, **out)
        self._pending = []


def _stage_calibration_update(
    cal_buffer: AffineCalibrationBuffer | None,
    cal_updates_local: List[Tuple[str, float, float]],
    target: str,
    primal_stats: dict,
) -> None:
    """Queue this target's raw worst-rank (forecast, actual) pair for the buffer.

    The pair is the UNCALIBRATED forecast and the observed worst actual, so the
    calibrator learns the map raw_forecast -> actual exactly as the deployed
    post-hoc calibrator does. Applied to the buffer after the full step (so the
    leave-one-group-out fit for each target uses the previous step's state).
    """
    if cal_buffer is None:
        return
    cpw = primal_stats.get("cal_pred_worst", None)
    caw = primal_stats.get("cal_actual_worst", None)
    if cpw is None or caw is None:
        return
    q = float(cpw.item()) if torch.is_tensor(cpw) else float(cpw)
    y = float(caw.item()) if torch.is_tensor(caw) else float(caw)
    if math.isfinite(q) and math.isfinite(y):
        cal_updates_local.append((target, q, y))


def count_extrapolated_eval_points(
    fit_n_total: int,
    eval_n_total: int,
    *,
    plotting: str = "gringorten",
    fit_rank_cutoff: int = 1,
) -> int:
    return generic_count_extrapolated_eval_points(
        fit_n_total,
        eval_n_total,
        plotting=plotting,
        fit_rank_cutoff=fit_rank_cutoff,
    )


# ---------------------------------------------------------------------------
# Batched password log-prob computation
# ---------------------------------------------------------------------------


def _apply_chat_template_safe(tokenizer, messages, enable_thinking: bool) -> str:
    """Apply chat template with fallback for tokenizers without enable_thinking."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _format_elapsed_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _should_log_progress(index: int, total: int, *, every: int = 5) -> bool:
    if total <= 0:
        return False
    if index <= 1 or index >= total:
        return True
    if total <= 10:
        return True
    return index % max(1, every) == 0


def compute_password_logprobs_batched(
    model,
    tokenizer,
    records: List[Dict],
    *,
    enable_thinking: bool,
    device: str,
    micro_batch_size: int = 32,
    require_grad: bool = True,
    score_max_len: int = 0,
    token_budget: int = 0,
) -> torch.Tensor:
    """Compute the per-record rare-event log-probability at the first response token.

    For a password record the event is a single token (``event_token_id``); for a
    token-union record it is a set (``event_token_ids``) and the score is the
    logsumexp over that set. A singleton union reduces to the password value
    exactly, so the password path is numerically unchanged.

    Returns a 1-D tensor of shape [N]. If require_grad is True and the model has
    trainable parameters, gradients flow through the result.

    score_max_len > 0 LEFT-truncates each tokenized prompt so the generation-prompt
    suffix (and, for jailbreaks, the tail of the prompt) always survives.
    token_budget > 0 packs length-sorted micro-batches under ``B * L_max <=
    token_budget``; 0 keeps the legacy fixed-size contiguous batches.
    """
    if not records:
        return torch.zeros(0, device=device)

    # Pre-tokenize all prompts and resolve each record's event id set.
    all_input_ids: List[torch.Tensor] = []
    all_event_ids: List[List[int]] = []
    for rec in records:
        messages = prepare_messages(
            str(rec["prompt"]),
            system_prompt=rec.get("system_prompt"),
        )
        text = _apply_chat_template_safe(tokenizer, messages, enable_thinking)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if score_max_len and ids.shape[0] > score_max_len:
            ids = ids[-score_max_len:]  # LEFT truncation
        all_input_ids.append(ids)
        if "event_token_ids" in rec:
            all_event_ids.append([int(x) for x in rec["event_token_ids"]])
        else:
            all_event_ids.append([int(rec["event_token_id"])])

    n = len(all_input_ids)
    # Plan micro-batches (each is a list of original indices).
    if token_budget and token_budget > 0:
        order = sorted(range(n), key=lambda i: int(all_input_ids[i].shape[0]))
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        for i in order:
            length = int(all_input_ids[i].shape[0])
            new_max = max(cur_max, length)
            if cur and (
                len(cur) + 1 > micro_batch_size
                or (len(cur) + 1) * new_max > token_budget
            ):
                batches.append(cur)
                cur, cur_max, new_max = [], 0, length
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
    else:
        batches = [
            list(range(start, min(start + micro_batch_size, n)))
            for start in range(0, n, micro_batch_size)
        ]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out_list: List[Optional[torch.Tensor]] = [None] * n

    # HF gradient checkpointing only engages inside layers whose .training is
    # True; scoring runs with the model in eval mode, so a require_grad call
    # retains full activations for every micro-batch graph until the caller's
    # backward (~1.5 MB/token for Qwen3-0.6B -- enough to OOM an 80 GB card
    # once the top-ranked prompts grow to score_max_len). For grad passes on a
    # checkpointing-enabled model, switch to train mode with dropout forced
    # off; RMSNorm/attention are mode-independent, so scores are unchanged.
    engage_gc = (
        require_grad
        and not model.training
        and bool(getattr(model, "is_gradient_checkpointing", False))
    )
    if engage_gc:
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
    memlog = bool(os.environ.get("LQUANT_MEMLOG"))
    if memlog and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch in batches:
        batch_ids = [all_input_ids[i] for i in batch]
        batch_evt = [all_event_ids[i] for i in batch]
        max_len = max(ids.shape[0] for ids in batch_ids)
        padded = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
        for r, ids in enumerate(batch_ids):
            offset = max_len - ids.shape[0]
            padded[r, offset:] = ids
            attention_mask[r, offset:] = 1

        padded = padded.to(device)
        attention_mask = attention_mask.to(device)

        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            # logits_to_keep=1 restricts the (possibly 150k-260k vocab) LM head to
            # the last position, so an 8k-token batch does not materialize a
            # [B, L, V] logit tensor. Older forwards lack the kwarg -> full logits.
            try:
                out = model(
                    input_ids=padded, attention_mask=attention_mask, logits_to_keep=1
                )
            except TypeError:
                out = model(input_ids=padded, attention_mask=attention_mask)
            next_lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)  # [B, V]

            u_max = max(len(e) for e in batch_evt)
            evt_idx = torch.zeros(len(batch_evt), u_max, dtype=torch.long, device=device)
            evt_mask = torch.zeros(len(batch_evt), u_max, dtype=torch.bool, device=device)
            for r, evt in enumerate(batch_evt):
                evt_idx[r, : len(evt)] = torch.tensor(evt, dtype=torch.long, device=device)
                evt_mask[r, : len(evt)] = True
            gathered = next_lp.gather(1, evt_idx).masked_fill(~evt_mask, float("-inf"))
            lp = torch.logsumexp(gathered, dim=1)  # [B]

        for k, i in enumerate(batch):
            out_list[i] = lp[k]

    if engage_gc:
        model.eval()
    if memlog:
        total_tok = int(sum(int(t.shape[0]) for t in all_input_ids))
        max_tok = int(max(int(t.shape[0]) for t in all_input_ids))
        peak = (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0.0
        )
        print(
            f"[memlog] grad={int(require_grad)} gc={int(engage_gc)} n={n} "
            f"tok_total={total_tok} tok_max={max_tok} peak_gib={peak:.2f}",
            flush=True,
        )
    tokenizer.padding_side = old_padding_side
    return torch.stack(out_list, dim=0)


# ---------------------------------------------------------------------------
# KL divergence helpers
# ---------------------------------------------------------------------------


def _tokenize_for_kl(
    tokenizer,
    prompts: List[str],
    device: str,
    max_len: int = 512,
    truncation_side: str = "right",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and left-pad a batch of prompt strings.

    Returns (input_ids, attention_mask) on *device*, both [B, L]. truncation_side
    "left" keeps the tail + generation marker of long prompts (needed for the
    8k-token WildChat KL prompts; right-truncation there would cut the marker and
    measure "first-token KL" mid-prompt).
    """
    old_side = tokenizer.padding_side
    old_trunc = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = truncation_side
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    tokenizer.padding_side = old_side
    tokenizer.truncation_side = old_trunc
    return enc.input_ids.to(device), enc.attention_mask.to(device)


@contextlib.contextmanager
def _gc_train_mode(model):
    """Enter train mode with dropout disabled so HF gradient checkpointing
    (which only engages when layers are in train mode) applies to a
    grad-enabled forward. No-op unless the model has checkpointing enabled and
    is currently in eval mode. Numerics are unchanged: dropout is forced off
    and the remaining modules are mode-independent."""
    engage = (
        not model.training
        and bool(getattr(model, "is_gradient_checkpointing", False))
    )
    if engage:
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
    try:
        yield
    finally:
        if engage:
            model.eval()


def _kl_last_logits(model, input_ids, attention_mask) -> torch.Tensor:
    """Last-position logits, using logits_to_keep=1 when the forward supports it
    so long KL prompts do not materialize a [B, L, V] logit tensor."""
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, -1, :].float()


def compute_kl_single_token(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(p_theta || p_base) at the first generated position (last prompt token).

    Returns a scalar (mean over batch).  Gradients flow through *model* only.
    """
    with torch.enable_grad(), _gc_train_mode(model):
        logits_cur = _kl_last_logits(model, input_ids, attention_mask)
    with torch.no_grad():
        logits_base = _kl_last_logits(base_model, input_ids, attention_mask)

    log_p = F.log_softmax(logits_cur, dim=-1)
    log_q = F.log_softmax(logits_base, dim=-1)
    # KL(p || q) = sum_v p(v) * (log p(v) - log q(v))
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [B]
    return kl.mean()


def evaluate_kl_single_token_batched(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    device: str,
    batch_size: int,
    max_prompts: int = 256,
    max_len: int = 512,
    truncation_side: str = "right",
) -> Tuple[float, int]:
    """Evaluate one-token KL on held-out prompts in memory-safe batches."""
    eval_prompts = prompts[:max_prompts]
    if not eval_prompts:
        return 0.0, 0

    batch_size = max(1, batch_size)
    kl_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(eval_prompts), batch_size):
            prompt_batch = eval_prompts[start:start + batch_size]
            kl_ids, kl_mask = _tokenize_for_kl(
                tokenizer, prompt_batch, device,
                max_len=max_len, truncation_side=truncation_side,
            )
            logits_cur = _kl_last_logits(model, kl_ids, kl_mask)
            logits_base = _kl_last_logits(base_model, kl_ids, kl_mask)
            log_p = F.log_softmax(logits_cur, dim=-1)
            log_q = F.log_softmax(logits_base, dim=-1)
            kl_chunks.append((log_p.exp() * (log_p - log_q)).sum(dim=-1).cpu())

            del kl_ids, kl_mask, logits_cur, logits_base, log_p, log_q

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return torch.cat(kl_chunks, dim=0).mean().item(), len(eval_prompts)


def compute_kl_truncated(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_tokens: int = 8,
) -> torch.Tensor:
    """Truncated-completion KL: sample n_tokens from model, teacher-force base.

    Returns an unbiased estimate of KL(p_theta || p_base) for the first
    n_tokens of the completion, averaged over the batch.
    """
    B = input_ids.shape[0]
    device = input_ids.device

    # Autoregressive sampling from current model
    sampled_ids = []
    log_p_cur_tokens = []
    cur_ids = input_ids
    cur_mask = attention_mask

    with torch.enable_grad(), _gc_train_mode(model):
        for _t in range(n_tokens):
            out = model(input_ids=cur_ids, attention_mask=cur_mask)
            logits = out.logits[:, -1, :].float()
            log_p = F.log_softmax(logits, dim=-1)
            probs = log_p.exp()
            # Sample from current model
            next_tok = torch.multinomial(probs.detach(), 1)  # [B, 1]
            # Gather log-prob of sampled token
            lp_tok = log_p.gather(1, next_tok).squeeze(1)  # [B]
            sampled_ids.append(next_tok)
            log_p_cur_tokens.append(lp_tok)
            # Extend sequence
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=device)],
                dim=1,
            )

    # Stack sampled log-probs: [B, n_tokens]
    log_p_cur = torch.stack(log_p_cur_tokens, dim=1)

    # Teacher-force base model on prompt + sampled continuation
    full_ids = torch.cat([input_ids] + sampled_ids, dim=1)  # [B, L + n_tokens]
    full_mask = torch.cat(
        [attention_mask, torch.ones(B, n_tokens, dtype=attention_mask.dtype, device=device)],
        dim=1,
    )
    with torch.no_grad():
        base_out = base_model(input_ids=full_ids, attention_mask=full_mask)
    # base_out.logits: [B, L + n_tokens, V]
    # We need logits at positions L-1, L, ..., L+n_tokens-2 (predicting tokens L, L+1, ..., L+n_tokens-1)
    L = input_ids.shape[1]
    base_logits = base_out.logits[:, L - 1 : L - 1 + n_tokens, :].float()  # [B, n_tokens, V]
    log_q_base = F.log_softmax(base_logits, dim=-1)
    sampled_tensor = torch.cat(sampled_ids, dim=1)  # [B, n_tokens]
    log_q_cur = log_q_base.gather(2, sampled_tensor.unsqueeze(2)).squeeze(2)  # [B, n_tokens]

    # KL estimate = E_{y~p}[log p(y) - log q(y)] per token, summed over tokens
    kl_per_prompt = (log_p_cur - log_q_cur).sum(dim=1)  # [B]
    return kl_per_prompt.mean()


# ---------------------------------------------------------------------------
# External dataset loading for KL evaluation
# ---------------------------------------------------------------------------


def load_kl_dataset(
    path: str,
    tokenizer,
    n_samples: int = 256,
    seed: int = 42,
    max_len: int = 512,
) -> List[str]:
    """Load and subsample prompts from a HuggingFace arrow dataset directory.

    Supports WildChat-style datasets (conversation list with role/content).
    Returns a list of formatted prompt strings ready for tokenization.
    """
    from pathlib import Path as P

    # JSONL sidecar of {"p": text} rows (the token-union bank's non-flagged KL set,
    # built disjoint from the training prompts). Apply the chat template so the KL
    # is measured at the same first-generated position as the task scores.
    if str(path).endswith(".jsonl"):
        rows: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("p") or obj.get("prompt") or ""
                if not text:
                    continue
                try:
                    formatted = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False, add_generation_prompt=True,
                    )
                except TypeError:
                    formatted = text
                rows.append(formatted)
        rng = random.Random(seed)
        if len(rows) > n_samples:
            rows = [rows[i] for i in rng.sample(range(len(rows)), n_samples)]
        return rows

    from datasets import Dataset as HFDataset

    # Find arrow files
    arrow_dir = P(path)
    arrow_files = sorted(arrow_dir.rglob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {path}")

    # Load first shard (enough for subsampling)
    ds = HFDataset.from_file(str(arrow_files[0]))
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))

    prompts = []
    for idx in indices:
        row = ds[idx]
        # WildChat format: conversation list of {role, content}
        if "conversation" in row:
            conv = row["conversation"]
            if conv and isinstance(conv, list) and len(conv) >= 1:
                # Take first user turn as the prompt
                user_msg = conv[0].get("content", "")
                if user_msg:
                    messages = [{"role": "user", "content": user_msg}]
                    try:
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                    except TypeError:
                        text = user_msg
                    prompts.append(text)
        elif "text" in row:
            # Plain text format (C4-style)
            text = row["text"]
            if text:
                prompts.append(text[:max_len * 4])  # rough char limit
    return prompts


# ---------------------------------------------------------------------------
# Prompt pool construction
# ---------------------------------------------------------------------------


def _effective_prompt_bank_seed(cfg: LQuantConfig) -> int:
    return int(cfg.seed if cfg.prompt_bank_seed < 0 else cfg.prompt_bank_seed)


def _prompt_bank_required_signature(cfg: LQuantConfig) -> Dict[str, Any]:
    if str(cfg.task_kind) == "token_union":
        # A token-union bank is a neutral prompt pool + a token-id union; only the
        # tokenizer (model_id) and the task kind must agree. Password-specific
        # fields (slots/policy) are meaningless here.
        return {
            "model_id": str(cfg.model_id),
            "task_kind": "token_union",
        }
    return {
        "model_id": str(cfg.model_id),
        "password_slots": int(cfg.password_slots),
        "password_policy": str(cfg.password_policy),
    }


def _prompt_bank_advisory_signature(cfg: LQuantConfig) -> Dict[str, Any]:
    return {
        "cheap_prompts_per_family_per_target": int(cfg.cheap_prompts_per_family_per_target),
        "cheap_prompts_per_family_map": {
            str(key): int(value) for key, value in sorted(cfg.cheap_prompts_per_family_map.items())
        },
        "train_families": list(cfg.train_families),
        "test_families": list(cfg.test_families),
        "gcg_enabled": bool(cfg.gcg_enabled),
        "gcg_steps_spectrum": [int(value) for value in cfg.gcg_steps_spectrum],
        "gcg_prompts_per_level_per_target": int(cfg.gcg_prompts_per_level_per_target),
        "gcg_prompt_len": int(cfg.gcg_prompt_len),
        "gcg_top_k": int(cfg.gcg_top_k),
        "gcg_random_restarts": int(cfg.gcg_random_restarts),
        "gcg_train_max_steps": int(cfg.gcg_train_max_steps),
        "gcg_decay_alpha": float(cfg.gcg_decay_alpha),
        "disable_thinking": bool(cfg.disable_thinking),
    }


def _validate_prompt_bank(bank_info: PromptBankInfo, cfg: LQuantConfig) -> None:
    metadata = bank_info.metadata
    required = _prompt_bank_required_signature(cfg)
    failures: list[str] = []
    for key, expected in required.items():
        actual = metadata.get(key)
        if actual != expected:
            failures.append(f"{key}: bank={actual} != cfg={expected}")
    if failures:
        raise RuntimeError("Prompt bank config mismatch: " + "; ".join(failures))
    if str(cfg.task_kind) == "token_union":
        # Advisory fields are password-generation metadata, absent from a
        # token-union bank; skip the check so real warnings stay visible.
        return
    advisory = _prompt_bank_advisory_signature(cfg)
    advisory_mismatches: list[str] = []
    for key, expected in advisory.items():
        actual = metadata.get(key)
        if actual != expected:
            advisory_mismatches.append(key)
    if advisory_mismatches:
        warnings.warn(
            "Prompt bank generation metadata differs from the current consumer config for "
            + ", ".join(advisory_mismatches)
            + ". This is allowed because prompt banks are treated as neutral pools; "
              "consumption is controlled by the current split/family config.",
            stacklevel=2,
        )


def _load_prompt_pool_from_bank(
    bank_info: PromptBankInfo,
    *,
    cfg: LQuantConfig,
    tokenizer,
) -> Dict[str, Any]:
    _validate_prompt_bank(bank_info, cfg)
    metadata = bank_info.metadata
    targets = list(metadata["targets"])
    families = list(metadata["families"])
    slot_to_password = {str(key): str(value) for key, value in metadata["slot_to_password"].items()}
    pool_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        target: {} for target in targets
    }
    prompt_bank_rows = int(metadata.get("row_count", 0))

    target_payload: Dict[str, tuple[str, int, str]] = {}
    for target in targets:
        password = slot_to_password[target]
        event_ids = tokenizer(password, add_special_tokens=False).input_ids
        if len(event_ids) != 1:
            raise RuntimeError(
                f"Bank password '{password}' for {target} is not a single token for {cfg.model_id}."
            )
        target_payload[target] = (
            password,
            int(event_ids[0]),
            build_password_system_prompt(password, policy=cfg.password_policy),
        )

    loaded_rows = 0
    with open(bank_info.rows_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            target = targets[int(row["t"])]
            family_id = families[int(row["f"])]
            prompt = str(row["p"])
            password, event_token_id, system_prompt = target_payload[target]
            pool_by_target.setdefault(target, {}).setdefault(family_id, []).append(
                {
                    "prompt": prompt,
                    "target": target,
                    "event_text": password,
                    "event_token_id": event_token_id,
                    "system_prompt": system_prompt,
                    "family_id": family_id,
                }
            )
            loaded_rows += 1
    if prompt_bank_rows and loaded_rows != prompt_bank_rows:
        raise RuntimeError(
            f"Prompt bank row count mismatch: metadata={prompt_bank_rows} loaded={loaded_rows}"
        )

    return {
        "targets": targets,
        "slot_to_password": slot_to_password,
        "pool_by_target": pool_by_target,
        "family_support_by_target": metadata.get("family_support_by_target", {}),
        "prompt_bank": bank_info.summary(),
    }


def _load_prompt_pool_from_bank_token_union(
    bank_info: PromptBankInfo,
    *,
    cfg: LQuantConfig,
    tokenizer,
) -> Dict[str, Any]:
    """Load a token-union task bank into the pool schema the trainer consumes.

    Records deliberately carry ``event_token_ids`` (a list) and NO
    ``event_token_id`` key, so any code path that was only updated for the
    password (single-token) event raises KeyError instead of silently scoring one
    token. ``system_prompt`` is None (WildChat prompts have no system turn), which
    ``prepare_messages`` renders as a bare user turn -- matching the scored
    artifacts exactly.
    """
    from .wildchat_token_union import (
        verify_token_union,
        resolve_target_event_token_ids,
    )

    _validate_prompt_bank(bank_info, cfg)
    metadata = bank_info.metadata
    verify_token_union(tokenizer, metadata)

    targets = list(metadata["targets"])
    families = list(metadata["families"])
    if len(families) != 1:
        raise RuntimeError(
            f"token_union bank expects one family, got {families!r}."
        )
    family_id = families[0]
    if family_id not in cfg.train_families or family_id not in cfg.test_families:
        raise RuntimeError(
            f"token_union family '{family_id}' must appear in both --train-families "
            f"and --test-families (got train={cfg.train_families}, test={cfg.test_families})."
        )

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    target_ids = resolve_target_event_token_ids(metadata, targets)
    for target, ids in target_ids.items():
        if not ids:
            raise RuntimeError(f"token_union target {target} has an empty event id set.")
        if vocab_size and max(ids) >= vocab_size:
            raise RuntimeError(
                f"token_union target {target} has an id >= tokenizer vocab_size ({vocab_size})."
            )

    blocklist_system_prompt: str | None = None
    if cfg.token_union_blocklist_prompt:
        union_meta = metadata.get("event_token_union", {}) or {}
        words = sorted({w.strip().lower() for w in (union_meta.get("string_to_id") or {}) if w.strip()})
        if not words:
            raise RuntimeError(
                "token_union_blocklist_prompt=True but the bank metadata carries no "
                "resolved union words (event_token_union.string_to_id)."
            )
        blocklist_system_prompt = (
            "Never use any of the following words in your response: "
            + ", ".join(words)
            + "."
        )
        print(
            f"token_union blocklist system prompt active ({len(words)} words): "
            f"{blocklist_system_prompt[:100]}...",
            flush=True,
        )

    prompts_per_target = int(metadata.get("prompts_per_target", 0))
    pool_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]] = {t: {} for t in targets}
    prompt_bank_rows = int(metadata.get("row_count", 0))

    loaded_rows = 0
    with open(bank_info.rows_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            target = targets[int(row["t"])]
            prompt = str(row["p"])
            pool_by_target[target].setdefault(family_id, []).append(
                {
                    "prompt": prompt,
                    "target": target,
                    "event_text": "token_union",
                    "event_token_ids": target_ids[target],
                    "system_prompt": blocklist_system_prompt,
                    "family_id": family_id,
                }
            )
            loaded_rows += 1
    if prompt_bank_rows and loaded_rows != prompt_bank_rows:
        raise RuntimeError(
            f"Prompt bank row count mismatch: metadata={prompt_bank_rows} loaded={loaded_rows}"
        )
    if prompts_per_target:
        for target in targets:
            got = len(pool_by_target[target].get(family_id, []))
            if got != prompts_per_target:
                raise RuntimeError(
                    f"token_union target {target} has {got} prompts != "
                    f"prompts_per_target={prompts_per_target} (extrapolation geometry "
                    f"depends on the fixed pool size)."
                )

    return {
        "targets": targets,
        "pool_by_target": pool_by_target,
        "family_support_by_target": {},
        "prompt_bank": bank_info.summary(),
    }


def _make_records_for_family(
    prompts: List[Any],
    *,
    target: str,
    password: str,
    event_token_id: int,
    family_id: str,
    password_policy: str,
) -> List[Dict]:
    """Wrap prompt strings or prompt records into the training record schema."""
    sys_prompt = build_password_system_prompt(password, policy=password_policy)
    out: list[Dict[str, Any]] = []
    for item in prompts:
        if isinstance(item, dict):
            prompt = str(item.get("prompt", ""))
        else:
            prompt = str(item)
        record: Dict[str, Any] = {
            "prompt": prompt,
            "target": target,
            "event_text": password,
            "event_token_id": event_token_id,
            "system_prompt": sys_prompt,
            "family_id": family_id,
        }
        if isinstance(item, dict):
            if "gcg_logprob_first_token" in item:
                record["gcg_logprob_first_token"] = float(item["gcg_logprob_first_token"])
            if "gcg_token_ids" in item:
                record["gcg_token_ids"] = [int(x) for x in item["gcg_token_ids"]]
        out.append(record)
    return out


def build_prompt_pool(
    cfg: LQuantConfig,
    model,
    tokenizer,
    *,
    generation_seed: int | None = None,
    targets_subset: Optional[Sequence[str]] = None,
) -> Dict:
    """Generate the full prompt pool with family tags.

    Returns a dict with keys:
        targets: List[str]
        slot_to_password: Dict[str, str]
        pool_by_target: Dict[str, Dict[str, List[Dict]]]
            target -> family_id -> list of records
        family_support_by_target: Dict[str, Dict[str, Dict]]
    """
    generation_seed = int(cfg.seed if generation_seed is None else generation_seed)
    all_targets = get_password_slots(cfg.password_slots)
    slot_to_password_all = assign_passwords_to_targets(all_targets, tokenizer, seed=generation_seed)
    if targets_subset is None:
        targets = list(all_targets)
    else:
        requested = [str(target) for target in targets_subset]
        unknown = [target for target in requested if target not in slot_to_password_all]
        if unknown:
            raise ValueError(f"Unknown target(s) requested for prompt pool shard: {unknown}")
        targets = [target for target in all_targets if target in set(requested)]
    slot_to_password = {target: slot_to_password_all[target] for target in targets}

    pool_by_target: Dict[str, Dict[str, List[Dict]]] = {}
    family_support_by_target: Dict[str, Dict[str, Dict]] = {}
    all_families = sorted(set(cfg.train_families + cfg.test_families))

    def stable_seed(*parts: object) -> int:
        raw = "||".join(str(p) for p in (generation_seed,) + parts)
        digest = hashlib.sha256(raw.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    for target in tqdm(targets, desc="Building pool"):
        password = slot_to_password[target]
        event_ids = tokenizer(password, add_special_tokens=False).input_ids
        if len(event_ids) != 1:
            raise RuntimeError(
                f"Password '{password}' for {target} is not a single token for {cfg.model_id}."
            )
        event_token_id = int(event_ids[0])

        target_families: Dict[str, List[Dict]] = {}
        target_family_support: Dict[str, Dict] = {}

        # Cheap families
        for fam in all_families:
            if fam.startswith("gcg_"):
                continue
            requested_n = int(
                cfg.cheap_prompts_per_family_map.get(
                    fam,
                    cfg.cheap_prompts_per_family_per_target,
                )
            )
            if requested_n <= 0:
                continue
            support = get_password_prompt_family_support(
                fam,
                password_text=password,
            )
            fam_rng = random.Random(stable_seed(target, fam, "cheap"))
            prompts = generate_password_prompts(
                requested_n,
                fam,
                fam_rng,
                password_text=password,
                unique=True,
            )
            unique_count = len(set(prompts))
            if unique_count != len(prompts):
                raise RuntimeError(
                    f"Family '{fam}' for target '{target}' produced duplicates "
                    f"despite unique sampling ({unique_count}/{len(prompts)} unique)."
                )
            records = _make_records_for_family(
                prompts,
                target=target,
                password=password,
                event_token_id=event_token_id,
                family_id=fam,
                password_policy=cfg.password_policy,
            )
            target_families[fam] = records
            target_family_support[fam] = {
                "requested_prompts": int(requested_n),
                "generated_prompts": int(len(prompts)),
                "unique_prompts": int(unique_count),
                "max_distinct_prompts": int(support.max_distinct_prompts),
                "exact_capacity": bool(support.exact_capacity),
                "unique_sampler": str(support.unique_sampler),
                "capacity_utilization": (
                    float(len(prompts)) / float(support.max_distinct_prompts)
                    if support.max_distinct_prompts > 0
                    else None
                ),
            }

        # GCG spectrum
        if cfg.gcg_enabled:
            sys_prompt = build_password_system_prompt(password, policy=cfg.password_policy)
            for steps in cfg.gcg_steps_spectrum:
                fam_id = f"gcg_steps{steps}"
                fam_seed = stable_seed(target, fam_id, "gcg") % (10**9)
                gcg_cfg = GCGConfig(
                    prompt_len=cfg.gcg_prompt_len,
                    steps=steps,
                    top_k=cfg.gcg_top_k,
                    random_restarts=cfg.gcg_random_restarts,
                    seed=fam_seed,
                )
                if model is None:
                    raise RuntimeError("GCG prompt generation requires a loaded model.")
                gcg_prompts = generate_gcg_prompts(
                    model, tokenizer,
                    system_prompt=sys_prompt,
                    event_token_id=event_token_id,
                    n_prompts=cfg.gcg_prompts_per_level_per_target,
                    cfg=gcg_cfg,
                    enable_thinking=(not cfg.disable_thinking),
                    device=cfg.device,
                )
                records = _make_records_for_family(
                    gcg_prompts,
                    target=target,
                    password=password,
                    event_token_id=event_token_id,
                    family_id=fam_id,
                    password_policy=cfg.password_policy,
                )
                target_families[fam_id] = records

        pool_by_target[target] = target_families
        family_support_by_target[target] = target_family_support

    return {
        "targets": targets,
        "all_targets": list(all_targets),
        "slot_to_password": slot_to_password,
        "pool_by_target": pool_by_target,
        "family_support_by_target": family_support_by_target,
    }


def _parse_family_prompt_count_map(items: Optional[List[str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid family prompt override '{item}'. Expected family=count."
            )
        family, value = item.split("=", 1)
        family = family.strip()
        if not family:
            raise ValueError(f"Invalid family prompt override '{item}': empty family name.")
        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid family prompt override '{item}': count must be an integer."
            ) from exc
        if count < 0:
            raise ValueError(
                f"Invalid family prompt override '{item}': count must be >= 0."
            )
        out[family] = count
    return out


def _fit_deploy_shift_enabled(cfg: LQuantConfig) -> bool:
    return bool(cfg.fit_families) or bool(cfg.deploy_families)


def _validate_fit_deploy_family_shift(cfg: LQuantConfig) -> None:
    """Reject invalid fit/deploy family-shift configs up front (see LQuantConfig)."""
    if not _fit_deploy_shift_enabled(cfg):
        return
    if not cfg.fit_families or not cfg.deploy_families:
        raise ValueError(
            "fit_families and deploy_families must both be non-empty when the "
            f"family shift is enabled (got fit={cfg.fit_families}, "
            f"deploy={cfg.deploy_families})."
        )
    overlap = sorted(set(cfg.fit_families) & set(cfg.deploy_families))
    if overlap:
        raise ValueError(
            f"fit_families and deploy_families must be disjoint; both contain {overlap}."
        )
    if str(cfg.task_kind) != "password":
        raise ValueError(
            f"fit/deploy family shift is defined for task_kind='password' only "
            f"(got {cfg.task_kind!r})."
        )
    if cfg.split_mode != "disjoint":
        raise ValueError(
            f"fit/deploy family shift requires split_mode='disjoint' (got {cfg.split_mode!r})."
        )
    if cfg.bootstrap_permutations >= 1:
        raise ValueError(
            "fit/deploy family shift is incompatible with bootstrap_permutations >= 1: "
            "the bootstrap re-partitions the fit/deploy union each step, which assumes "
            "the two sides are exchangeable. Run with --bootstrap-permutations 0."
        )
    known = set(cfg.train_families) | set(cfg.test_families)
    unknown = sorted(
        (set(cfg.fit_families) | set(cfg.deploy_families)) - known
    )
    if unknown:
        raise ValueError(
            f"fit/deploy families {unknown} are not in train_families/test_families "
            f"({sorted(known)}); the pool would contain no records for them."
        )
    if cfg.shift_fit_size < 1:
        raise ValueError(
            "fit/deploy family shift requires an explicit --shift-fit-size >= 1 "
            f"(got {cfg.shift_fit_size}); cell sizes are part of the protocol."
        )
    if cfg.shift_deploy_size < 0:
        raise ValueError(f"shift_deploy_size must be >= 0 (got {cfg.shift_deploy_size}).")


def _largest_remainder_alloc(available: Dict[str, int], total: int) -> Dict[str, int]:
    """Deterministically split `total` across keys proportionally to `available`.

    Standard largest-remainder rule; ties broken by key name. Used to thin the
    deploy pool to an exact cell size while keeping every slot's per-family
    composition identical.
    """
    n_avail = sum(available.values())
    if total > n_avail:
        raise ValueError(f"Cannot allocate {total} from {n_avail} available.")
    quotas = {key: total * count / n_avail for key, count in available.items()}
    alloc = {key: int(math.floor(q)) for key, q in quotas.items()}
    remainder = total - sum(alloc.values())
    order = sorted(available, key=lambda key: (-(quotas[key] - alloc[key]), key))
    for key in order[:remainder]:
        alloc[key] += 1
    for key, count in alloc.items():
        if count > available[key]:
            raise ValueError(
                f"Allocation for {key} ({count}) exceeds available ({available[key]})."
            )
    return alloc


def split_pool(
    pool: Dict,
    cfg: LQuantConfig,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Split pool into train, eval, test sets.

    Returns (train_records, eval_records, test_records) each keyed by target.
    Train and eval draw from train_families; test draws from test_families.
    """
    rng = random.Random(cfg.seed + 999)
    train_records: Dict[str, List[Dict]] = {}
    eval_records: Dict[str, List[Dict]] = {}
    test_records: Dict[str, List[Dict]] = {}

    for target in pool["targets"]:
        fam_map = pool["pool_by_target"][target]

        if _fit_deploy_shift_enabled(cfg):
            # Family-shift protocol: fit sample only from fit_families, deploy
            # set only from deploy_families, identical composition in every
            # slot. Full family pools are used (see LQuantConfig: the 50/50
            # halving protects nothing here and would halve the reachable
            # deploy size). The deploy pool is thinned to shift_deploy_size by
            # a deterministic largest-remainder allocation across families.
            fit_candidates: List[Dict] = []
            for fam_id in cfg.fit_families:
                fit_candidates.extend(fam_map.get(fam_id, []))
            rng.shuffle(fit_candidates)
            if len(fit_candidates) < cfg.shift_fit_size:
                raise ValueError(
                    f"target {target}: fit families {cfg.fit_families} supply "
                    f"{len(fit_candidates)} records < shift_fit_size {cfg.shift_fit_size}."
                )
            fit_recs_t = fit_candidates[:cfg.shift_fit_size]

            deploy_avail = {
                fam_id: list(fam_map.get(fam_id, [])) for fam_id in cfg.deploy_families
            }
            avail_counts = {fam_id: len(recs) for fam_id, recs in deploy_avail.items()}
            n_deploy = cfg.shift_deploy_size or sum(avail_counts.values())
            if n_deploy < 1:
                raise ValueError(
                    f"target {target}: deploy families {cfg.deploy_families} "
                    "supply no records."
                )
            alloc = _largest_remainder_alloc(avail_counts, n_deploy)
            deploy_recs_t: List[Dict] = []
            for fam_id in cfg.deploy_families:
                fam_recs = deploy_avail[fam_id]
                rng.shuffle(fam_recs)
                deploy_recs_t.extend(fam_recs[:alloc[fam_id]])

            train_records[target] = fit_recs_t
            eval_records[target] = deploy_recs_t
            # Held-out slots evaluate on exactly this fit/deploy partition;
            # _split_test_records_for_target re-separates it by family.
            test_records[target] = fit_recs_t + deploy_recs_t
            continue

        # Collect train-eligible and test-eligible records
        train_pool: List[Dict] = []
        test_pool: List[Dict] = []
        for fam_id, records in fam_map.items():
            in_train = fam_id in cfg.train_families
            in_test = fam_id in cfg.test_families
            if in_train and in_test and cfg.target_holdout_full_pools:
                shared_records = list(records)
                rng.shuffle(shared_records)
                test_pool.extend(shared_records)
                train_pool.extend(shared_records)
            elif in_train and in_test:
                shared_records = list(records)
                rng.shuffle(shared_records)
                n_test = int(len(shared_records) * cfg.shared_family_test_fraction)
                n_test = min(max(1, n_test), max(1, len(shared_records) - 1))
                test_pool.extend(shared_records[:n_test])
                train_pool.extend(shared_records[n_test:])
            elif in_train:
                train_pool.extend(records)
            elif in_test:
                test_pool.extend(records)

        # Split train_pool into fit (train) and eval
        rng.shuffle(train_pool)
        if cfg.split_mode == "nested":
            # Fit is a subset; eval is the full pool
            n_fit = max(1, int(len(train_pool) * cfg.fit_fraction))
            train_records[target] = train_pool[:n_fit]
            eval_records[target] = list(train_pool)  # full pool
        elif cfg.split_mode == "resample":
            # Both draw from the same pool; resampled each step (handled in training loop)
            train_records[target] = list(train_pool)
            eval_records[target] = list(train_pool)
        else:
            # disjoint
            n_fit = max(1, int(len(train_pool) * cfg.fit_fraction))
            train_records[target] = train_pool[:n_fit]
            eval_records[target] = train_pool[n_fit:]

        test_records[target] = test_pool

    return train_records, eval_records, test_records


def _target_partition_seed(cfg: LQuantConfig) -> int:
    raw = f"{cfg.seed}||target_partition"
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _resolve_target_partition(
    targets: List[str],
    cfg: LQuantConfig,
) -> Dict[str, object]:
    ordered_targets = list(targets)
    extra_train: List[str] = []
    n_base = int(getattr(cfg, "partition_base_targets", 0) or 0)
    if n_base > 0:
        if n_base > len(ordered_targets):
            raise ValueError(
                f"partition_base_targets={n_base} exceeds the bank's {len(ordered_targets)} targets"
            )
        extra_train = ordered_targets[n_base:]
        ordered_targets = ordered_targets[:n_base]
    total = len(ordered_targets)
    train_count = int(cfg.train_target_count)
    heldout_count = int(cfg.heldout_target_count)
    if extra_train and (train_count == 0 or heldout_count == 0):
        raise ValueError("partition_base_targets needs explicit train and held-out target counts")

    if train_count < 0 or heldout_count < 0:
        raise ValueError("train_target_count and heldout_target_count must be >= 0")
    if total == 0:
        return {
            "partition_seed": _target_partition_seed(cfg),
            "all_targets": [],
            "train_targets": [],
            "heldout_targets": [],
            "evaluation_targets": [],
            "unused_targets": [],
            "mode": "empty",
        }
    if train_count == 0 and heldout_count == 0:
        return {
            "partition_seed": _target_partition_seed(cfg),
            "all_targets": ordered_targets,
            "train_targets": ordered_targets,
            "heldout_targets": ordered_targets,
            "evaluation_targets": ordered_targets,
            "unused_targets": [],
            "mode": "legacy_all_targets",
        }

    if train_count == 0:
        train_count = total - heldout_count
    if heldout_count == 0:
        heldout_count = total - train_count

    if train_count < 0 or heldout_count < 0:
        raise ValueError(
            "Target counts imply a negative partition size. "
            f"Got total={total}, train_target_count={cfg.train_target_count}, "
            f"heldout_target_count={cfg.heldout_target_count}."
        )
    if train_count + heldout_count > total:
        raise ValueError(
            "Target counts exceed available targets. "
            f"Got total={total}, resolved_train={train_count}, resolved_heldout={heldout_count}."
        )

    shuffled_targets = list(ordered_targets)
    partition_seed = _target_partition_seed(cfg)
    random.Random(partition_seed).shuffle(shuffled_targets)
    train_targets = shuffled_targets[:train_count]
    if cfg.heldout_from_end:
        heldout_targets = shuffled_targets[total - heldout_count:]
        unused_targets = shuffled_targets[train_count:total - heldout_count]
    else:
        heldout_targets = shuffled_targets[train_count:train_count + heldout_count]
        unused_targets = shuffled_targets[train_count + heldout_count:]
    evaluation_targets = heldout_targets if heldout_targets else train_targets

    out = {
        "partition_seed": partition_seed,
        "all_targets": ordered_targets + extra_train,
        "train_targets": train_targets + extra_train,
        "heldout_targets": heldout_targets,
        "evaluation_targets": evaluation_targets,
        "unused_targets": unused_targets,
        "mode": "count_partition",
    }
    if extra_train:
        out["partition_base_targets"] = n_base
        out["extra_train_targets"] = list(extra_train)
    return out


def _filter_target_records(
    records: Dict[str, List[Dict]],
    targets: List[str],
) -> Dict[str, List[Dict]]:
    return {target: list(records.get(target, [])) for target in targets}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_lquant(cfg: LQuantConfig) -> Dict:
    """Main training loop.  Returns a summary dict."""
    from importance_sampling.policy_lm import PolicyLM, LoRASetup

    distributed = _distributed_enabled()
    world_size = _distributed_world_size()
    rank = _distributed_rank()
    local_rank = _distributed_local_rank()
    is_main_process = rank == 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU LQuant training requires CUDA.")
        if cfg.device and not str(cfg.device).startswith("cuda"):
            raise ValueError(f"Distributed LQuant only supports CUDA devices, got {cfg.device!r}.")
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device(cfg.device),
            )
    elif not cfg.device:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.resume_checkpoint and cfg.baseline_only:
        raise ValueError("resume_checkpoint may not be combined with baseline_only")
    if cfg.eval_adapter_from and (cfg.resume_checkpoint or cfg.baseline_only):
        raise ValueError(
            "eval_adapter_from may not be combined with resume_checkpoint or baseline_only"
        )

    resume_state: dict[str, object] | None = None
    if cfg.resume_checkpoint:
        resume_state = _load_resume_checkpoint(cfg.resume_checkpoint)
        _validate_resume_checkpoint_config(resume_state, cfg)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.fit_subsample_min > 0 and cfg.fit_subsample_min < cfg.ols_top_k:
        raise ValueError(
            f"fit_subsample_min ({cfg.fit_subsample_min}) must be >= ols_top_k ({cfg.ols_top_k})"
        )
    _validate_fit_deploy_family_shift(cfg)

    outdir = Path(cfg.outdir)
    if is_main_process:
        outdir.mkdir(parents=True, exist_ok=True)
    _distributed_barrier()

    def log(*args, **kwargs) -> None:
        if is_main_process:
            kwargs.setdefault("flush", True)
            print(*args, **kwargs)

    # Save config
    if is_main_process:
        with open(outdir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

    # Load model with LoRA
    log(f"Loading model {cfg.model_id} with LoRA={cfg.use_lora} ...")
    lora_setup = LoRASetup(
        use_lora=cfg.use_lora,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
    )
    policy = PolicyLM(
        model_id=cfg.model_id,
        device=cfg.device,
        lora=lora_setup,
    )
    raw_model = policy.model
    if cfg.gradient_checkpointing:
        # Long-context (token_union) grad passes retain activations for ~29
        # leak-biased (hence long) prompts per target; at 8k tokens that exceeds
        # 80GB. Checkpointing trades ~30% compute for a ~10-20x activation-memory
        # cut. enable_input_require_grads is required so the frozen base's
        # embedding output carries grad into the LoRA path under checkpointing.
        try:
            raw_model.enable_input_require_grads()
        except AttributeError:
            pass
        raw_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if raw_model.config.use_cache:
            raw_model.config.use_cache = False
        log("Gradient checkpointing enabled (use_reentrant=False, use_cache=False).")
    model = raw_model
    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    tokenizer = policy.tokenizer

    # Build or load prompt pool
    if cfg.prompt_bank_root:
        bank_seed = _effective_prompt_bank_seed(cfg)
        bank_info = resolve_prompt_bank(cfg.prompt_bank_root, bank_seed)
        bank_task_kind = str(bank_info.metadata.get("task_kind", "password"))
        if bank_task_kind != str(cfg.task_kind):
            raise RuntimeError(
                f"Prompt bank task_kind={bank_task_kind!r} != cfg.task_kind={cfg.task_kind!r} "
                f"({bank_info.bank_dir})."
            )
        log(
            f"Loading {bank_task_kind} prompt bank seed {bank_info.bank_seed} "
            f"from {bank_info.bank_dir} ..."
        )
        if bank_task_kind == "token_union":
            pool = _load_prompt_pool_from_bank_token_union(
                bank_info, cfg=cfg, tokenizer=tokenizer
            )
        else:
            pool = _load_prompt_pool_from_bank(bank_info, cfg=cfg, tokenizer=tokenizer)
    else:
        if str(cfg.task_kind) == "token_union":
            raise RuntimeError(
                "task_kind=token_union requires --prompt-bank-root (online prompt "
                "generation is password-only)."
            )
        log("Building prompt pool ...")
        pool = build_prompt_pool(cfg, raw_model, tokenizer)
    if "family_support_by_target" in pool and is_main_process:
        with open(outdir / "prompt_pool_metadata.json", "w") as f:
            json.dump(pool["family_support_by_target"], f, indent=2)
    if "prompt_bank" in pool and is_main_process:
        with open(outdir / "prompt_bank.json", "w") as f:
            json.dump(pool["prompt_bank"], f, indent=2)
    train_recs, eval_recs, test_recs = split_pool(pool, cfg)
    target_partition = _resolve_target_partition(list(pool["targets"]), cfg)
    train_targets = list(target_partition["train_targets"])
    heldout_targets = list(target_partition["heldout_targets"])
    evaluation_targets = list(target_partition["evaluation_targets"])
    unused_targets = list(target_partition["unused_targets"])
    if str(cfg.task_kind) == "token_union":
        if target_partition["mode"] == "legacy_all_targets":
            raise RuntimeError(
                "task_kind=token_union requires --train-target-count and "
                "--heldout-target-count; the legacy all-targets mode makes every "
                "task both train and heldout (heldout leakage)."
            )
        # A train-pool-count ladder (heldout_from_end) leaves tasks unused on
        # purpose: the lower rungs train on a subset while holding out the same
        # pools as the top rung.
        if unused_targets and not cfg.heldout_from_end:
            raise RuntimeError(
                f"task_kind=token_union stranded {len(unused_targets)} tasks as unused; "
                f"train ({len(train_targets)}) + heldout ({len(heldout_targets)}) must "
                f"cover all {len(pool['targets'])} tasks."
            )
    local_targets = _shard_items(train_targets, rank=rank, world_size=world_size)
    if distributed and not local_targets:
        raise ValueError(
            f"Distributed LQuant world_size={world_size} exceeds training target count {len(train_targets)}."
        )
    train_recs_eval = _filter_target_records(train_recs, evaluation_targets)
    eval_recs_eval = _filter_target_records(eval_recs, evaluation_targets)
    test_recs_eval = _filter_target_records(test_recs, evaluation_targets)
    if is_main_process:
        with open(outdir / "target_partition.json", "w") as f:
            json.dump(target_partition, f, indent=2)

    # Summary
    for target in train_targets[:3]:
        n_train = len(train_recs.get(target, []))
        n_eval = len(eval_recs.get(target, []))
        n_test = len(test_recs.get(target, []))
        log(f"  {target}: train={n_train}, eval={n_eval}, test={n_test}")
    if len(train_targets) > 3:
        log(f"  ... ({len(train_targets)} training targets)")
    log(
        "Target partition: "
        f"total={len(target_partition['all_targets'])} "
        f"train={len(train_targets)} "
        f"heldout={len(heldout_targets)} "
        f"eval={len(evaluation_targets)} "
        f"unused={len(unused_targets)}"
    )
    if heldout_targets:
        log(f"  target partition seed={target_partition['partition_seed']}")
    if distributed:
        log(f"Distributed target sharding: world_size={world_size}, local_targets={len(local_targets)}")

    # --- KL setup ---
    base_model = None
    kl_train_prompts: List[str] = []
    kl_eval_prompts: List[str] = []
    kl_active = cfg.kl_coef > 0 or cfg.kl_eval_datasets

    if kl_active:
        # Load frozen base model for KL reference
        log("Loading frozen base model for KL ...")
        kl_load_started = time.perf_counter()
        from transformers import AutoModelForCausalLM
        base_dtype = next(raw_model.parameters()).dtype
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=base_dtype,
        ).to(cfg.device).eval()
        for p in base_model.parameters():
            p.requires_grad_(False)
        log(
            "  KL base model loaded | "
            f"elapsed={_format_elapsed_seconds(time.perf_counter() - kl_load_started)}"
        )

        # Build KL training prompts
        for ds_name in cfg.kl_train_datasets:
            if ds_name == "password":
                log(f"Building KL train prompts from password data | targets={len(train_targets)}")
                kl_train_started = time.perf_counter()
                # Collect formatted prompt strings from all train+eval records
                for idx, target in enumerate(train_targets, start=1):
                    for rec in train_recs.get(target, []) + eval_recs.get(target, []):
                        msgs = prepare_messages(
                            str(rec["prompt"]),
                            system_prompt=rec.get("system_prompt"),
                        )
                        text = _apply_chat_template_safe(
                            tokenizer, msgs, not cfg.disable_thinking,
                        )
                        kl_train_prompts.append(text)
                    if _should_log_progress(idx, len(train_targets)):
                        log(
                            "  KL train prompt build "
                            f"{idx}/{len(train_targets)} targets | prompts={len(kl_train_prompts)} "
                            f"| elapsed={_format_elapsed_seconds(time.perf_counter() - kl_train_started)}"
                        )
            else:
                dataset_started = time.perf_counter()
                log(f"Loading KL train dataset {ds_name} ...")
                kl_train_prompts.extend(
                    load_kl_dataset(ds_name, tokenizer, n_samples=cfg.kl_batch_size * 20)
                )
                log(
                    "  Loaded KL train dataset "
                    f"{ds_name} | prompts={len(kl_train_prompts)} "
                    f"| elapsed={_format_elapsed_seconds(time.perf_counter() - dataset_started)}"
                )
        log(f"  KL train prompts: {len(kl_train_prompts)}")

        # Build KL eval prompts
        for ds_name in cfg.kl_eval_datasets:
            if ds_name == "password":
                log(f"Building KL eval prompts from password data | targets={len(evaluation_targets)}")
                kl_eval_started = time.perf_counter()
                for idx, target in enumerate(evaluation_targets, start=1):
                    for rec in test_recs.get(target, []):
                        msgs = prepare_messages(
                            str(rec["prompt"]),
                            system_prompt=rec.get("system_prompt"),
                        )
                        text = _apply_chat_template_safe(
                            tokenizer, msgs, not cfg.disable_thinking,
                        )
                        kl_eval_prompts.append(text)
                    if _should_log_progress(idx, len(evaluation_targets)):
                        log(
                            "  KL eval prompt build "
                            f"{idx}/{len(evaluation_targets)} targets | prompts={len(kl_eval_prompts)} "
                            f"| elapsed={_format_elapsed_seconds(time.perf_counter() - kl_eval_started)}"
                        )
            else:
                dataset_started = time.perf_counter()
                log(f"Loading KL eval dataset {ds_name} ...")
                kl_eval_prompts.extend(
                    load_kl_dataset(ds_name, tokenizer, n_samples=256)
                )
                log(
                    "  Loaded KL eval dataset "
                    f"{ds_name} | prompts={len(kl_eval_prompts)} "
                    f"| elapsed={_format_elapsed_seconds(time.perf_counter() - dataset_started)}"
                )
        if kl_eval_prompts:
            log(f"  KL eval prompts: {len(kl_eval_prompts)}")

    baseline_eval: dict[str, dict] | None
    if resume_state is not None:
        log(f"Resuming from checkpoint {cfg.resume_checkpoint}", flush=True)
        baseline_eval = resume_state.get("baseline_eval")
        if not isinstance(baseline_eval, dict):
            baseline_eval = _load_saved_eval_suite(outdir, "baseline_") if is_main_process else None
    else:
        baseline_eval = None
        if is_main_process:
            cache_dir = cfg.baseline_eval_cache_dir or None
            cache_hit = baseline_cache_module.try_load_from_cache(
                cache_dir, cfg, outdir, log=log,
            )
            if cache_hit:
                baseline_eval = _load_saved_eval_suite(outdir, "baseline_")
            else:
                baseline_eval = run_evaluation_suite(
                    raw_model,
                    tokenizer,
                    train_recs_eval,
                    eval_recs_eval,
                    test_recs_eval,
                    cfg,
                    outdir=outdir,
                    prefix="baseline_",
                    label="baseline ",
                    base_model=base_model,
                    kl_eval_prompts=kl_eval_prompts,
                    kl_model=base_model if base_model is not None else None,
                )
                baseline_cache_module.save_to_cache(
                    cache_dir, cfg, outdir, log=log,
                )
    baseline_eval = _broadcast_object(baseline_eval, src=0)
    if not isinstance(baseline_eval, dict):
        raise RuntimeError("Failed to initialize baseline evaluation results.")
    if cfg.baseline_only:
        _distributed_barrier()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return {
            "baseline_test_results": baseline_eval["test_results"],
            "baseline_test_pool_diagnostic": baseline_eval["test_pool_diagnostic"],
            "baseline_kl_eval_results": baseline_eval["kl_eval_results"],
            "baseline_test_trace_artifacts": baseline_eval.get("test_trace_artifacts", {}),
            "history": [],
            "elapsed_sec": 0.0,
        }

    # Optimizer. AdamW is the canonical choice and the default. Plain SGD is
    # available as a mechanism probe: Adam normalizes per coordinate, which
    # amplifies rarely-used directions and so makes an update far more selective
    # in score space than the raw Jacobian would be. The learning-dynamics study
    # can account for a clipped SGD step from the measured response kernel but
    # not for a real AdamW trajectory, and this flag is what tells the two apart.
    # Note the two are only comparable at matched parameter-step size: with
    # clipping, SGD moves ||lr|| per step while AdamW moves about
    # lr * sqrt(n_params), so an SGD arm needs a correspondingly larger lr.
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable, lr=cfg.lr)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)
    log(f"Optimizer: {cfg.optimizer} at lr {cfg.lr}")
    log(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    rng = random.Random(cfg.seed)
    history: List[Dict] = []
    eval_caches: dict[str, DeploySubsetCache] = {}
    union_caches: dict[str, UnionTopCache] = {}
    resume_phase = "train"
    start_step = 0
    best_sq_log10: float | None = None
    elapsed_before = 0.0

    def _save_phase_resume_checkpoint(phase: str, next_step: int, *, elapsed_sec: float) -> None:
        merged_caches = _gather_cache_payload(serialize_deploy_caches(eval_caches))
        merged_union_caches = _gather_cache_payload(serialize_union_caches(union_caches))
        if not is_main_process:
            return
        state = {
            "config_signature": _critical_cfg_signature(cfg),
            "phase": phase,
            "next_step": int(next_step),
            "trainable_state_dict": _trainable_state_dict(raw_model),
            "optimizer_state_dict": _cpu_state(optimizer.state_dict()),
            "history": history,
            "baseline_eval": baseline_eval,
            "eval_caches": merged_caches,
            "union_caches": merged_union_caches,
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "elapsed_sec": float(elapsed_sec),
        }
        _save_resume_checkpoint(outdir, state)

    if resume_state is not None:
        _load_trainable_state_dict(raw_model, resume_state.get("trainable_state_dict", {}))
        optimizer_state_dict = resume_state.get("optimizer_state_dict")
        if isinstance(optimizer_state_dict, dict):
            optimizer.load_state_dict(optimizer_state_dict)
            _move_optimizer_state_to_param_device(optimizer)
        if isinstance(resume_state.get("history"), list):
            history = list(resume_state["history"])
            _prior = [
                h["train_forecast_sq_log10_error"]
                for h in history
                if isinstance(h, dict) and h.get("train_forecast_sq_log10_error") is not None
            ]
            if _prior:
                best_sq_log10 = min(_prior)
        eval_caches = deserialize_deploy_caches(resume_state.get("eval_caches"))
        union_caches = deserialize_union_caches(resume_state.get("union_caches"))
        if resume_state.get("python_rng_state") is not None:
            rng.setstate(resume_state["python_rng_state"])
        if resume_state.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        resume_phase = str(resume_state.get("phase", "train"))
        start_step = int(resume_state.get("next_step", 0))
        elapsed_before = float(resume_state.get("elapsed_sec", 0.0))
        if resume_phase not in {"train", "final_eval", "complete"}:
            raise ValueError(f"Unsupported resume phase in checkpoint: {resume_phase}")
        if resume_phase in {"final_eval", "complete"} and start_step < cfg.steps:
            resume_phase = "train"
        log(f"  resume phase={resume_phase} next_step={start_step}", flush=True)
        if resume_phase == "complete":
            # Do NOT clean up the checkpoint here: it is what allows a finished
            # run to be extended later with a larger --steps. Re-running a
            # complete run at the same --steps is a no-op that returns the
            # saved eval.
            if is_main_process:
                final_eval = _load_saved_eval_suite(outdir, "")
            else:
                final_eval = None
            final_eval = _broadcast_object(final_eval, src=0)
            _distributed_barrier()
            if distributed and dist.is_initialized():
                dist.destroy_process_group()
            return {
                "history": history,
                "baseline_test_results": baseline_eval["test_results"],
                "baseline_test_pool_diagnostic": baseline_eval["test_pool_diagnostic"],
                "baseline_kl_eval_results": baseline_eval["kl_eval_results"],
                "baseline_test_trace_artifacts": baseline_eval.get("test_trace_artifacts", {}),
                "test_results": final_eval["test_results"],
                "test_pool_diagnostic": final_eval["test_pool_diagnostic"],
                "kl_eval_results": final_eval["kl_eval_results"],
                "test_trace_artifacts": final_eval.get("test_trace_artifacts", {}),
                "elapsed_sec": elapsed_before,
            }

    if cfg.eval_adapter_from:
        # Load a saved adapter and jump straight to the final held-out eval. The
        # loop below is skipped by pointing start_step past cfg.steps; everything
        # downstream (final save, eval suite, KL eval) runs unchanged, so the
        # resulting test_results.json is directly comparable to a trained run's.
        from safetensors.torch import load_file as _load_safetensors_file
        try:
            from peft import set_peft_model_state_dict
        except ImportError:
            from peft.utils import set_peft_model_state_dict
        adapter_file = os.path.join(cfg.eval_adapter_from, "adapter_model.safetensors")
        adapter_state = _load_safetensors_file(adapter_file)
        if not adapter_state:
            raise RuntimeError(f"Empty adapter state at {adapter_file}")
        # LoRA B matrices initialize to zero, so a trained adapter must change the
        # trainable-parameter norm; an unchanged norm means the load silently
        # missed every key and the "eval" would just re-score the base model.
        norm_before = sum(
            float(p.detach().float().norm()) for p in raw_model.parameters() if p.requires_grad
        )
        load_result = set_peft_model_state_dict(raw_model, adapter_state)
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        if unexpected:
            raise RuntimeError(
                f"Adapter load had {len(unexpected)} unexpected keys "
                f"(first: {unexpected[:3]}) from {adapter_file}"
            )
        norm_after = sum(
            float(p.detach().float().norm()) for p in raw_model.parameters() if p.requires_grad
        )
        if abs(norm_after - norm_before) < 1e-8:
            raise RuntimeError(
                f"Adapter load from {adapter_file} left trainable weights "
                f"unchanged (norm {norm_before:.6f}); refusing to eval the base "
                f"model under an adapter's name"
            )
        start_step = cfg.steps
        log(
            f"Loaded eval adapter from {cfg.eval_adapter_from} "
            f"({len(adapter_state)} tensors, trainable norm {norm_before:.3f} -> "
            f"{norm_after:.3f}); skipping training loop.",
            flush=True,
        )

    # --- Training loop ---
    if resume_phase == "train":
        log(f"\nStarting training for {cfg.steps} steps ...")
    else:
        log(f"\nSkipping training loop and resuming at phase {resume_phase} ...")
    if cfg.split_mode == "resample" and cfg.eval_cache_top_b > 0 and cfg.eval_cache_refresh_interval > 0:
        log(
            "  Eval cache disabled in resample split mode because eval subsets are redrawn each step.",
            flush=True,
        )
    t_start = time.perf_counter() - elapsed_before

    # --- IOG penalty baseline ---
    # Snapshot pretrained per-prompt log_probs for both fit and eval pools.
    # The penalty replaces the IOG mask (which covers BOTH fit and deploy
    # gradients, see ols_utils.py:506-566) with a symmetric squared hinge:
    # L += coef * mean(relu(lp_now - lp_baseline)^2) on the prompts feeding
    # the OLS loss each step (top-k extrapolated eval prompts AND top-k fit
    # prompts). The mask is a biased subgradient (loss can climb under it);
    # the penalty is a real gradient on a real augmented loss.
    pretrained_eval_lp_baseline: Dict[str, torch.Tensor] = {}
    pretrained_fit_lp_baseline: Dict[str, torch.Tensor] = {}
    if cfg.iog_penalty_coef > 0:
        if cfg.split_mode == "resample":
            raise ValueError(
                "--iog-penalty-coef requires --split-mode disjoint (or any "
                "non-resample mode). Resample mode breaks the per-prompt "
                "indexing the penalty needs."
            )
        if start_step > 0:
            raise ValueError(
                "--iog-penalty-coef does not yet support resume from a "
                "checkpoint > step 0; the penalty baseline would be wrong "
                "(would use the checkpoint state instead of the pretrained "
                "state). Either start a fresh run or extend the implementation "
                "to persist the baseline."
            )
        if cfg.fit_subsample_min > 0:
            raise ValueError(
                "--iog-penalty-coef does not currently support "
                "--fit-subsample-min > 0 (fit-side baseline indexing assumes "
                "fit_subset is the full fit pool). Set --fit-subsample-min 0 "
                "or extend the implementation to track fit_subset record IDs."
            )
        log("Caching pretrained log_probs for IOG-penalty baseline (fit + eval) ...")
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for target in train_targets:
                eval_pool = eval_recs.get(target, [])
                if eval_pool:
                    lp_eval = compute_password_logprobs_batched(
                        model, tokenizer, eval_pool,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=False,
                    )
                    pretrained_eval_lp_baseline[target] = lp_eval.detach()
                fit_pool = train_recs.get(target, [])
                if fit_pool:
                    lp_fit = compute_password_logprobs_batched(
                        model, tokenizer, fit_pool,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=False,
                    )
                    pretrained_fit_lp_baseline[target] = lp_fit.detach()
        if was_training:
            model.train()
        log(
            f"  cached eval baseline for {len(pretrained_eval_lp_baseline)} "
            f"targets, fit baseline for {len(pretrained_fit_lp_baseline)} targets"
        )

    # --- Anchor-gated worsening mask baseline ---
    # The gate needs each union record's log_prob under the FROZEN base model,
    # in the same order the bootstrap branch builds the union
    # (train_recs[target] + eval_recs[target]). That ordering is stable only
    # when the fit pool is not resampled per step, hence the guards.
    #
    # Source of truth, in order: a cached file (so resume never re-derives an
    # anchor from a trained checkpoint), then the frozen `base_model` that the
    # KL term already keeps in memory, then the LoRA model with its adapter
    # disabled. All three are the same base-model scores.
    union_anchor_lp: Dict[str, torch.Tensor] = {}
    gap_floor_on = cfg.gap_floor_m > 1 and cfg.gap_floor_lambda > 0
    sigma_floor_on = cfg.sigma_floor_lambda > 0
    # Constraint-channel caches, populated with the anchor below. The frozen
    # index-sorted scoring set gives the constraint a composition that never
    # changes (the scoring pass is deterministic given its composition), and
    # the floors are derived from anchor scores measured through that same
    # channel so composition offsets cancel inside every span.
    constraint_grad_idx: Dict[str, torch.Tensor] = {}
    constraint_anchor_lp: Dict[str, torch.Tensor] = {}
    if (
        cfg.anchor_gate_margin > 0 or cfg.anchor_penalty_coef > 0
        or gap_floor_on or sigma_floor_on
    ):
        if cfg.anchor_gate_margin > 0 and not cfg.improving_only_grad:
            raise ValueError(
                "--anchor-gate-margin > 0 requires --improving-only-grad: the "
                "gate softens the improving-only mask, and does nothing without it."
            )
        if cfg.bootstrap_permutations < 1 or cfg.loss_mode != "ols_primal_extrapolated":
            raise ValueError(
                "--anchor-gate-margin / --anchor-penalty-coef / --gap-floor-m "
                "are implemented for the bootstrap path only "
                "(--bootstrap-permutations >= 1 with "
                "--loss-mode ols_primal_extrapolated)."
            )
        if cfg.split_mode == "resample" or cfg.fit_subsample_min > 0:
            raise ValueError(
                "the anchor options require a stable union ordering: use "
                "--split-mode disjoint (or nested) and --fit-subsample-min 0."
            )
        anchor_path = outdir / "anchor_union_lp.pt"
        constraint_anchor_path = outdir / "anchor_constraint_lp.pt"
        _need_constraint_anchor = gap_floor_on or sigma_floor_on
        if anchor_path.exists() and (
            not _need_constraint_anchor or constraint_anchor_path.exists()
        ):
            _payload = torch.load(anchor_path, map_location="cpu")
            union_anchor_lp = {
                str(k): v.to(cfg.device) for k, v in _payload.items()
            }
            log(f"Loaded anchor baseline for {len(union_anchor_lp)} targets from {anchor_path}")
            if _need_constraint_anchor:
                _payload = torch.load(constraint_anchor_path, map_location="cpu")
                constraint_anchor_lp = {
                    str(k): v.to(cfg.device) for k, v in _payload.items()
                }
                log(
                    f"Loaded constraint-channel anchor for "
                    f"{len(constraint_anchor_lp)} targets"
                )
        else:
            log("Caching base-model log_probs for the anchor-gated mask ...")
            _anchor_model = base_model if base_model is not None else model
            _adapter_ctx = (
                contextlib.nullcontext() if base_model is not None
                else raw_model.disable_adapter()
            )
            _was_training = model.training
            model.eval()
            with torch.no_grad(), _adapter_ctx:
                for target in train_targets:
                    _union = list(train_recs.get(target, [])) + list(eval_recs.get(target, []))
                    if not _union:
                        continue
                    union_anchor_lp[target] = compute_password_logprobs_batched(
                        _anchor_model, tokenizer, _union,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=False,
                    ).detach().float()
                    if _need_constraint_anchor:
                        # The floors (line gaps, sigma reference) must be
                        # derived from scores measured through the SAME
                        # channel the constraint reads at train time: the
                        # frozen index-sorted top-(m+100) set in its own
                        # scoring call. Composition offsets between passes
                        # are ~0.08 nats/prompt (measured), larger than most
                        # gaps, and they cancel in spans only when both
                        # sides of the hinge share one composition.
                        _acidx = torch.topk(
                            union_anchor_lp[target],
                            min(
                                union_anchor_lp[target].shape[0],
                                int(cfg.gap_floor_m or 250) + 100,
                            ),
                        ).indices.sort().values
                        constraint_grad_idx[target] = _acidx
                        _aclp = compute_password_logprobs_batched(
                            _anchor_model, tokenizer,
                            [_union[int(i)] for i in _acidx.tolist()],
                            enable_thinking=(not cfg.disable_thinking),
                            device=cfg.device,
                            micro_batch_size=cfg.micro_batch_size,
                            score_max_len=cfg.score_max_len,
                            token_budget=cfg.score_token_budget,
                            require_grad=False,
                        ).detach().float()
                        _acfull = union_anchor_lp[target].clone()
                        _acfull[_acidx] = _aclp
                        constraint_anchor_lp[target] = _acfull
            if _was_training:
                model.train()
            if is_main_process:
                torch.save(
                    {k: v.cpu() for k, v in union_anchor_lp.items()}, anchor_path,
                )
                if _need_constraint_anchor:
                    torch.save(
                        {k: v.cpu() for k, v in constraint_anchor_lp.items()},
                        constraint_anchor_path,
                    )
            log(
                f"  anchor baseline cached for {len(union_anchor_lp)} targets "
                f"(margin {cfg.anchor_gate_margin} nats)"
            )

    # Per-target reference variance for scale normalization. Captured at the
    # first step that scores each target and then frozen, so the normalizing
    # factor starts at 1 and every later step is measured against the tail the
    # run began with. Persisted for the same reason the anchor is: a resume must
    # not re-derive the reference from a partly trained model.
    scale_norm_ref: Dict[str, float] = {}
    scale_norm_dirty = False
    scale_norm_path = outdir / "scale_norm_ref.json"
    if cfg.loss_scale_norm_m > 0:
        if cfg.loss_mode != "ols_primal_extrapolated" or cfg.bootstrap_permutations < 1:
            raise ValueError(
                "--loss-scale-norm-m is implemented for the bootstrap path only "
                "(--loss-mode ols_primal_extrapolated with "
                "--bootstrap-permutations >= 1)."
            )
        if cfg.split_mode == "resample" or cfg.fit_subsample_min > 0:
            raise ValueError(
                "--loss-scale-norm-m requires a stable union ordering: use "
                "--split-mode disjoint (or nested) and --fit-subsample-min 0."
            )
        if scale_norm_path.exists():
            scale_norm_ref = {
                str(k): float(v)
                for k, v in json.loads(scale_norm_path.read_text()).items()
            }
            log(
                f"Loaded scale-normalization reference for "
                f"{len(scale_norm_ref)} targets from {scale_norm_path}"
            )

    # Per-target gap floors for the one-sided spread constraint. Derived
    # deterministically from the frozen anchor scores, so unlike the scale-norm
    # reference they need no persistence: a resume recomputes the same floors.
    gap_floor_g0: Dict[str, torch.Tensor] = {}
    # Line-referenced sigma floors, keyed by (target, n_fit, extrap_k):
    # the realized-set geometry varies with the step's fit subsample, and
    # the reference is deterministic given the frozen anchor, so a lazy
    # cache needs no persistence.
    sigma_floor_ref: Dict[tuple, float] = {}
    # Certified-tail targets per (target, n_fit, extrap_k), deterministic given
    # the frozen anchor, so they are recomputed rather than persisted on resume.
    cert_targets: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
    if cfg.cert_distill_coef > 0 and not sigma_floor_on:
        raise ValueError("cert_distill_coef > 0 needs sigma_floor_lambda > 0 (channel and band)")
    shared_pass = bool(cfg.cert_distill_shared_pass)
    if shared_pass and (cfg.loss_scale_norm_m > 0 or not sigma_floor_on):
        raise ValueError(
            "cert_distill_shared_pass needs loss_scale_norm_m 0 "
            "and the sigma floor (which builds the constraint channel)"
        )

    dynamics_dumper: DynamicsDumper | None = None
    if cfg.dynamics_dump_dir:
        dynamics_dumper = DynamicsDumper(
            Path(cfg.dynamics_dump_dir),
            every=cfg.dynamics_dump_every,
            rank=rank,
        )
        log(
            f"Dynamics dumps -> {cfg.dynamics_dump_dir} "
            f"(every {cfg.dynamics_dump_every} steps)"
        )

    # Calibration-aware training: leave-one-group-out affine calibrator over a
    # persistent buffer of per-target worst-rank (forecast, actual) pairs. The
    # buffer holds last step's global statistics; each target is forecast with
    # the calibrator fit on the OTHER targets, then contributes its fresh pair
    # at step end (gathered across ranks so all buffers stay in sync).
    cal_buffer: AffineCalibrationBuffer | None = None
    if cfg.calibration_aware:
        if cfg.loss_mode != "ols_primal_extrapolated":
            raise ValueError(
                "calibration_aware requires loss_mode='ols_primal_extrapolated' "
                f"(got {cfg.loss_mode!r})."
            )
        cal_buffer = AffineCalibrationBuffer(
            mode=cfg.calibration_mode,
            min_points=cfg.calibration_min_points,
            ridge_alpha=cfg.calibration_ridge_alpha,
            ema=cfg.calibration_ema,
        )
        log(
            f"Calibration-aware training: mode={cfg.calibration_mode}, "
            f"min_points={cfg.calibration_min_points}, "
            f"ridge_alpha={cfg.calibration_ridge_alpha}, ema={cfg.calibration_ema}"
        )

    for step in range(start_step, cfg.steps):
        if resume_phase != "train":
            break
        t_step = time.perf_counter()

        # Sample deployment scale(s) — only needed for legacy extrapolate mode
        deploy_queries = None
        if cfg.loss_mode == "extrapolate":
            deploy_rng = _step_rng(cfg.seed, step, stream=11)
            deploy_queries = [
                int(10 ** deploy_rng.uniform(cfg.log_n_min, cfg.log_n_max))
                for _ in range(cfg.n_deploy_samples)
            ]
            if len(deploy_queries) == 1:
                deploy_queries = deploy_queries[0]

        optimizer.zero_grad(set_to_none=True)
        # Meta-batch sampling. A dedicated RNG stream keeps the main `rng` and
        # the per-target streams untouched, so targets_per_step=0 runs are
        # bit-identical to pre-flag behavior and resume stays deterministic.
        step_targets = local_targets
        n_step_targets_global = len(train_targets)
        if cfg.targets_per_step > 0 and cfg.targets_per_step < len(local_targets):
            _mb_rng = _step_rng(cfg.seed, step, stream=37)
            step_targets = _mb_rng.sample(local_targets, cfg.targets_per_step)
            n_step_targets_global = len(step_targets) * world_size
        step_loss_local = 0.0
        n_targets_ok_local = 0
        iog_deploy_sum_local = 0.0
        iog_deploy_count_local = 0
        iog_fit_sum_local = 0.0
        iog_fit_count_local = 0
        # Track per-step worst observed deploy log_prob and OLS-line predicted
        # log_prob at the deepest extrapolated rank, averaged across targets.
        # These let us read off, from the streaming log, whether IOG/etc is
        # actually driving leak rates down vs. just shifting predicted scores.
        eval_max_lp_sum_local = 0.0
        pred_max_lp_sum_local = 0.0
        lp_count_local = 0
        # Train-side analogue of the held-out `forecast_sq_log10_error`: the mean
        # over train targets of the SQUARED worst-rank error. history.json used to
        # carry only the means of predicted and actual, whose difference is a bias
        # statistic -- it can sit at zero while per-target errors are large and
        # cancelling, which is exactly how the 4B step-140 checkpoint looked
        # healthy and evaluated badly.
        sq_log10_sum_local = 0.0
        # Learning-dynamics accumulators. `loss_perm_std` is the direct
        # gradient-consistency proxy: how much this step's loss depended on
        # which partition happened to be drawn. The tail-shape means quantify
        # collapse -- both spreads and the slope going to zero is "abolished
        # extrapolation"; error falling while they stay finite is "fixed".
        perm_std_sum_local = 0.0
        perm_std_count_local = 0
        fit_slope_sum_local = 0.0
        fit_x_spread_sum_local = 0.0
        eval_x_spread_sum_local = 0.0
        fit_v_spread_sum_local = 0.0
        eval_v_spread_sum_local = 0.0
        shape_count_local = 0
        worsening_coef_sum_local = 0.0
        worsening_coef_count_local = 0
        scale_norm_sum_local = 0.0
        scale_norm_count_local = 0
        gap_hinge_sum_local = 0.0
        gap_viol_frac_sum_local = 0.0
        gap_count_local = 0
        sigma_hinge_sum_local = 0.0
        sigma_bar_sum_local = 0.0
        sigma_count_local = 0
        cert_distill_sum_local = 0.0
        cert_distill_count_local = 0
        # IOG-penalty per-step accumulators (only used when iog_penalty_coef > 0)
        penalty_sum_local = 0.0
        penalty_count_local = 0
        penalty_violation_frac_sum_local = 0.0
        # Fit-side penalty (symmetric to deploy-side)
        penalty_fit_sum_local = 0.0
        penalty_fit_violation_frac_sum_local = 0.0
        # Calibration-aware: this step's fresh (target, raw_forecast, actual)
        # worst-rank contributions, applied to the buffer at step end.
        cal_updates_local: List[Tuple[str, float, float]] = []

        for target in step_targets:
            fit_recs = train_recs[target]
            ev_recs = eval_recs[target]
            if not fit_recs or not ev_recs:
                continue

            # Leave-one-group-out calibrator for THIS target, from the buffer's
            # other-target statistics (stop-gradient floats; None = no cal).
            cal_alpha: float | None = None
            cal_beta: float | None = None
            if cal_buffer is not None:
                cal_alpha, cal_beta = cal_buffer.loo_coeffs(target)

            # Resample mode: draw fresh subsets
            if cfg.split_mode == "resample":
                pool_all = fit_recs  # same as eval in resample mode
                target_rng = _target_step_rng(cfg.seed, step, target, stream=21)
                m = target_rng.randint(
                    cfg.fit_size_min,
                    max(cfg.fit_size_min + 1, cfg.fit_size_max or len(pool_all) // 2),
                )
                m = min(m, len(pool_all))
                fit_subset = target_rng.sample(pool_all, m)
                eval_subset = target_rng.sample(pool_all, min(len(pool_all), m * 3))
            else:
                # Subsample fit set each step so top-k bump structure varies
                if cfg.fit_subsample_min > 0 and cfg.fit_subsample_min < len(fit_recs):
                    target_rng = _target_step_rng(cfg.seed, step, target, stream=22)
                    m = target_rng.randint(cfg.fit_subsample_min, len(fit_recs))
                    fit_subset = target_rng.sample(fit_recs, m)
                else:
                    fit_subset = fit_recs
                eval_subset = ev_recs

            # ============================================================
            # Bootstrap-permutation branch.
            # When enabled, fit_subset ∪ eval_subset is treated as a single
            # union pool. Each permutation samples |fit_subset| records for
            # the fit partition and |eval_subset| for eval, OLS is fit on
            # each partition, and the per-permutation losses are averaged.
            #
            # Optional partition-invariant union top-C cache: when
            # cfg.union_cache_top_c > 0, we cache the top-C indices of the
            # union and only score those each step (refreshed every
            # cfg.union_cache_refresh_interval steps). For C >= deploy_top_b
            # + n_fit, deploy-side coverage is deterministic; fit-side may
            # need a lazy fallback (score full F_p for that permutation) if
            # |F_p ∩ cached| < ols_top_k. Falls back to full union scoring
            # when the cache is disabled or invalid.
            # ============================================================
            if (
                cfg.bootstrap_permutations >= 1
                and cfg.loss_mode == "ols_primal_extrapolated"
            ):
                n_fit = len(fit_subset)
                n_eval = len(eval_subset)
                if n_fit == 0 or n_eval == 0:
                    continue
                eval_extrap_k = count_extrapolated_eval_points(
                    n_fit, n_eval, plotting=cfg.plotting_position,
                )
                if eval_extrap_k == 0:
                    continue
                union_recs = list(fit_subset) + list(eval_subset)
                n_union = len(union_recs)

                bootstrap_seed = _target_step_rng(
                    cfg.seed, step, target, stream=23,
                ).randrange(2**31)

                # Decide whether to use the cache or do a full union pass
                _ucache = union_caches.get(target)
                _ucache_on = (
                    cfg.union_cache_top_c > 0
                    and cfg.union_cache_refresh_interval > 0
                    and cfg.union_cache_top_c < n_union
                )
                _ucache_hit = (
                    _ucache_on and _ucache is not None
                    and _ucache.is_valid(
                        step=step,
                        n_fit=n_fit,
                        n_union=n_union,
                        refresh_interval=cfg.union_cache_refresh_interval,
                    )
                )

                if _ucache_hit:
                    # Cache hit: score only cached indices, fill -inf elsewhere.
                    assert _ucache is not None
                    cached_indices = list(_ucache.cached_union_indices)
                    cached_recs = [union_recs[i] for i in cached_indices]
                    cached_lp = compute_password_logprobs_batched(
                        model, tokenizer, cached_recs,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=False,
                    )
                    union_lp_detached = torch.full(
                        (n_union,), float("-inf"),
                        dtype=cached_lp.dtype, device=cached_lp.device,
                    )
                    cached_idx_t = torch.tensor(
                        cached_indices, dtype=torch.long, device=cached_lp.device,
                    )
                    union_lp_detached = union_lp_detached.index_copy(
                        0, cached_idx_t, cached_lp,
                    )

                    # Lazy fit fallback: pre-sample partitions (matching the
                    # planner's RNG) and detect any permutation where
                    # |F_p ∩ cached| < ols_top_k. For each such permutation,
                    # we need to score the rest of F_p so the planner picks
                    # the true top-K rather than -inf placeholders.
                    cached_set = set(cached_indices)
                    g = torch.Generator(device="cpu")
                    g.manual_seed(int(bootstrap_seed))
                    extra_to_score: set[int] = set()
                    for _ in range(cfg.bootstrap_permutations):
                        perm = torch.randperm(n_union, generator=g)
                        fp_set = set(perm[:n_fit].tolist())
                        if len(fp_set & cached_set) < cfg.ols_top_k:
                            extra_to_score.update(fp_set - cached_set)
                    if extra_to_score:
                        _ucache.fit_fallback_count += 1
                        extra_list = sorted(extra_to_score)
                        extra_recs = [union_recs[i] for i in extra_list]
                        extra_lp = compute_password_logprobs_batched(
                            model, tokenizer, extra_recs,
                            enable_thinking=(not cfg.disable_thinking),
                            device=cfg.device,
                            micro_batch_size=cfg.micro_batch_size,
                            score_max_len=cfg.score_max_len,
                            token_budget=cfg.score_token_budget,
                            require_grad=False,
                        )
                        extra_idx_t = torch.tensor(
                            extra_list, dtype=torch.long, device=extra_lp.device,
                        )
                        union_lp_detached = union_lp_detached.index_copy(
                            0, extra_idx_t, extra_lp,
                        )
                    _ucache.record_hit(step=step)
                else:
                    # Cache miss: score full union, refresh cache if enabled.
                    union_lp_detached = compute_password_logprobs_batched(
                        model, tokenizer, union_recs,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=False,
                    )
                    if _ucache_on:
                        c_size = min(cfg.union_cache_top_c, n_union)
                        top_c = select_top_k_indices(
                            union_lp_detached, c_size,
                        ).detach().cpu().tolist()
                        prev = _ucache
                        new_cache = UnionTopCache(
                            cached_union_indices=top_c,
                            last_refresh_step=step,
                            n_fit_at_refresh=n_fit,
                            n_union_at_refresh=n_union,
                            refresh_count=(prev.refresh_count if prev else 0) + 1,
                            hit_count=prev.hit_count if prev else 0,
                            miss_count=(prev.miss_count if prev else 0) + 1,
                            fit_fallback_count=prev.fit_fallback_count if prev else 0,
                            deploy_fallback_count=prev.deploy_fallback_count if prev else 0,
                            total_cache_age=prev.total_cache_age if prev else 0,
                        )
                        union_caches[target] = new_cache
                fit_partitions, eval_partitions, needed_grad_idx = (
                    generic_plan_bootstrap_permutations(
                        n_union=n_union,
                        n_fit=n_fit,
                        n_eval=n_eval,
                        n_permutations=cfg.bootstrap_permutations,
                        rng_seed=bootstrap_seed,
                        fit_top_k=cfg.ols_top_k,
                        eval_extrap_k=eval_extrap_k,
                        union_values_detached=union_lp_detached,
                        grad_mode=cfg.grad_mode,
                    )
                )
                if cfg.loss_scale_norm_m > 0:
                    # Scale normalization divides the loss by the tail's own
                    # variance, so that variance has to be differentiable. The
                    # planner only asks for gradient on the prompts the loss
                    # selects, which is fewer than the top-m, so widen the set.
                    _m = max(2, min(int(cfg.loss_scale_norm_m), n_union))
                    _top_m = torch.topk(union_lp_detached, _m).indices
                    needed_grad_idx = torch.unique(
                        torch.cat([needed_grad_idx, _top_m])
                    ).sort().values

                boot_grad_idx = needed_grad_idx
                if shared_pass and needed_grad_idx.numel() > 0:
                    _chan = constraint_grad_idx.get(target)
                    if _chan is not None and _chan.numel() <= n_union:
                        boot_grad_idx = needed_grad_idx[
                            ~torch.isin(needed_grad_idx, _chan.to(needed_grad_idx.device))
                        ]
                if boot_grad_idx.numel() > 0:
                    grad_recs = [union_recs[int(i)] for i in boot_grad_idx.tolist()]
                    lp_grad_compact = compute_password_logprobs_batched(
                        model, tokenizer, grad_recs,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=True,
                    )
                    mask = torch.zeros(n_union, dtype=torch.bool, device=union_lp_detached.device)
                    mask[boot_grad_idx] = True
                    lp_grad_full = torch.zeros_like(union_lp_detached).index_copy(
                        0, boot_grad_idx, lp_grad_compact,
                    )
                    union_lp = torch.where(mask, lp_grad_full, union_lp_detached)
                else:
                    union_lp = union_lp_detached

                target_anchor = union_anchor_lp.get(target) if union_anchor_lp else None
                if target_anchor is not None and target_anchor.shape[0] != n_union:
                    raise ValueError(
                        f"anchor for target {target} has {target_anchor.shape[0]} "
                        f"entries but the union has {n_union}; the union ordering "
                        "changed between the anchor pass and training."
                    )

                # Constraint channel: the spread constraints read consecutive
                # spans, the finest feature of the pool, and the scoring pass
                # is exactly deterministic given its batch composition but
                # shifts by ~0.08 nats/prompt when the composition changes
                # (we measured this by rescoring the same prompts in different batches).
                # needed_grad_idx changes every step, so hinges computed on
                # that pass chase composition noise larger than most gaps --
                # this is what flatlined the per-gap arms. Scoring the
                # constraint's prompts in a SEPARATE pass with frozen
                # membership (anchor top-(m+buffer)) and canonical order makes
                # the channel deterministic across steps; the gradient still
                # flows through the model at the current weights.
                union_lp_c = union_lp
                if (
                    (gap_floor_on or sigma_floor_on)
                    and target_anchor is not None
                ):
                    _cidx = constraint_grad_idx.get(target)
                    if _cidx is None or _cidx.numel() > n_union:
                        # Fallback for resumes of runs whose anchor predates
                        # the constraint-channel cache: same construction.
                        _cn = min(n_union, int(cfg.gap_floor_m or 250) + 100)
                        _cidx = torch.topk(target_anchor, _cn).indices
                        _cidx = _cidx.sort().values  # canonical composition
                        constraint_grad_idx[target] = _cidx
                    _crecs = [union_recs[int(i)] for i in _cidx.tolist()]
                    # LoRA dropout re-rolls scores by ~0.09 nats/prompt per
                    # pass once the adapters move off zero (measured:
                    # pass_noise.json condition E; exactly 0 in eval mode).
                    # Silence dropout for this pass only, so the constraint
                    # channel stays deterministic while the loss path keeps
                    # the same regularization as every earlier arm.
                    _drops = [
                        _mod for _mod in model.modules()
                        if isinstance(_mod, torch.nn.Dropout) and _mod.training
                    ]
                    for _mod in _drops:
                        _mod.eval()
                    try:
                        _clp = compute_password_logprobs_batched(
                            model, tokenizer, _crecs,
                            enable_thinking=(not cfg.disable_thinking),
                            device=cfg.device,
                            micro_batch_size=cfg.micro_batch_size,
                            score_max_len=cfg.score_max_len,
                            token_budget=cfg.score_token_budget,
                            require_grad=True,
                        )
                    finally:
                        for _mod in _drops:
                            _mod.train()
                    _cmask = torch.zeros(
                        n_union, dtype=torch.bool, device=union_lp.device,
                    )
                    _cmask[_cidx] = True
                    _cfull = torch.zeros_like(union_lp).index_copy(0, _cidx, _clp)
                    union_lp_c = torch.where(_cmask, _cfull, union_lp)
                union_lp_c_detached = union_lp_c.detach()

                _boot_kwargs = dict(
                    n_fit=n_fit,
                    n_eval=n_eval,
                    fit_top_k=cfg.ols_top_k,
                    plotting=cfg.plotting_position,
                    score_mode="gumbel_log_prob",
                    grad_mode=cfg.grad_mode,
                    extrapolated_weighting=cfg.extrapolated_weighting,
                    loss_space=(
                        "original" if cfg.extrapolated_loss_space == "log_prob"
                        else cfg.extrapolated_loss_space
                    ),
                    improving_only_grad=cfg.improving_only_grad,
                    iog_scope=cfg.iog_scope,
                    iog_worsening_coef=cfg.iog_worsening_coef,
                    iog_fit_mask=cfg.iog_fit_mask,
                    union_anchor=target_anchor,
                    anchor_margin=cfg.anchor_gate_margin,
                    calibration_alpha=cal_alpha,
                    calibration_beta=cal_beta,
                )
                # Under the shared pass the bootstrap pass skipped the channel's
                # prompts, so a live forecast loss reads the merged scores.
                _fl_lp = (
                    union_lp_c if (shared_pass and cfg.forecast_loss_coef != 0)
                    else union_lp
                )
                lquant, primal_stats = (
                    generic_differentiable_ols_extrapolated_primal_loss_bootstrap(
                        _fl_lp, fit_partitions, eval_partitions, **_boot_kwargs,
                    )
                )
                if cfg.forecast_loss_coef != 1.0:
                    lquant = lquant * cfg.forecast_loss_coef

                # Scale normalization. The forecast loss is homogeneous of
                # degree two in how far the tail reaches below the bulk, so it
                # can always be reduced by pulling the tail in, whatever its
                # shape, and its irreducible sampling floor shrinks the same
                # way. Dividing by the tail's variance removes that: at fixed
                # shape the ratio does not depend on the scale, so the only
                # remaining descent direction is the shape itself. The reference
                # variance is the step-0 value, captured once per target and
                # persisted, so the term equals 1 at initialization and the
                # balance against the KL penalty is unchanged.
                scale_norm_val = float("nan")
                if cfg.loss_scale_norm_m > 0:
                    _m = max(2, min(int(cfg.loss_scale_norm_m), n_union))
                    _idx = torch.topk(union_lp_detached, _m).indices
                    _var = union_lp.index_select(0, _idx).var(unbiased=False)
                    _ref = scale_norm_ref.get(target)
                    if _ref is None:
                        _ref = float(_var.detach().item())
                        scale_norm_ref[target] = _ref
                        scale_norm_dirty = True
                    _factor = float(_ref) / _var.clamp_min(1e-6)
                    lquant = lquant * _factor
                    scale_norm_val = float(_factor.detach().item())
                    scale_norm_sum_local += scale_norm_val
                    scale_norm_count_local += 1

                # One-sided spread constraint: an l1 hinge on any consecutive
                # top-m gap that has fallen below the corresponding gap of the
                # population line through the anchor pool's tail. An exact
                # penalty: with lambda above the largest dual multiplier the
                # combined objective's minimum is the constrained minimum
                # itself, the corrected tail. One-sided, so extra spread stays
                # feasible; the loss itself already discourages it. Added
                # before the dynamics dump so the dumped gradient is the one
                # the model actually receives.
                if (
                    cfg.gap_floor_m > 1
                    and cfg.gap_floor_lambda > 0
                    and target_anchor is not None
                ):
                    _gm = min(int(cfg.gap_floor_m), n_union)
                    _g0 = gap_floor_g0.get(target)
                    if _g0 is None or _g0.shape[0] != _gm - 1:
                        _floor_src = constraint_anchor_lp.get(target, target_anchor)
                        _g0 = torch.as_tensor(
                            population_line_gaps(
                                _floor_src.detach().cpu().numpy(), m=_gm,
                            ),
                            dtype=union_lp.dtype, device=union_lp.device,
                        )
                        gap_floor_g0[target] = _g0
                    _tidx = torch.topk(union_lp_c_detached, _gm).indices
                    _tvals = union_lp_c.index_select(0, _tidx)
                    _b = max(1, min(int(cfg.gap_floor_block), _gm - 1))
                    # Floors for B-rank spans are sums of B consecutive line
                    # gaps; B = 1 recovers the per-gap constraint.
                    _csum = torch.cat([
                        torch.zeros(1, dtype=_g0.dtype, device=_g0.device),
                        torch.cumsum(_g0, 0),
                    ])
                    _w0 = _csum[_b:] - _csum[:-_b]
                    _upper = (
                        _tvals[:-_b].detach() if cfg.gap_floor_level_gated
                        else _tvals[:-_b]
                    )
                    _viol = torch.relu(
                        _w0 - cfg.gap_floor_slack - (_upper - _tvals[_b:])
                    )
                    _hinge = cfg.gap_floor_lambda * _viol.sum()
                    if torch.isfinite(_hinge):
                        lquant = lquant + _hinge
                        gap_hinge_sum_local += float(_viol.sum().item())
                        gap_viol_frac_sum_local += float(
                            (_viol > 0).float().mean().item()
                        )
                        gap_count_local += 1

                # The sigma-floor alternative: one-sided hinge on the std of
                # the realized involved set, averaged over the K permutations
                # before the hinge, referenced to the LINE through the anchor
                # pool (see the config comment for why the line and not the
                # observed pool).
                if cfg.sigma_floor_lambda > 0 and target_anchor is not None:
                    _ref_key = (target, int(n_fit), int(eval_extrap_k))
                    _sref = sigma_floor_ref.get(_ref_key)
                    if _sref is None:
                        _sref = involved_sigma_reference(
                            constraint_anchor_lp.get(
                                target, target_anchor
                            ).detach().cpu().numpy(),
                            n_fit=int(n_fit),
                            fit_top_k=int(cfg.ols_top_k),
                            extrap_k=int(eval_extrap_k),
                            m=int(cfg.sigma_ref_m),
                        )
                        sigma_floor_ref[_ref_key] = _sref
                    _sigs = []
                    for _p in range(fit_partitions.shape[0]):
                        _fidx = fit_partitions[_p]
                        _eidx = eval_partitions[_p]
                        _fsel = _fidx[torch.topk(
                            union_lp_c_detached.index_select(0, _fidx),
                            min(int(cfg.ols_top_k), _fidx.numel()),
                        ).indices]
                        _esel = _eidx[torch.topk(
                            union_lp_c_detached.index_select(0, _eidx),
                            min(int(eval_extrap_k), _eidx.numel()),
                        ).indices]
                        _pts = union_lp_c.index_select(
                            0, torch.cat([_fsel, _esel]),
                        )
                        _sigs.append(_pts.std(unbiased=False))
                    _sigma_bar = torch.stack(_sigs).mean()
                    _sgap = _sigma_bar.new_tensor(_sref) - _sigma_bar
                    if cfg.sigma_floor_symmetric:
                        _sviol = torch.relu(
                            _sgap.abs() - cfg.sigma_floor_margin
                        )
                    else:
                        _sviol = torch.relu(_sgap - cfg.sigma_floor_margin)
                    _shinge = cfg.sigma_floor_lambda * _sviol
                    if torch.isfinite(_shinge):
                        lquant = lquant + _shinge
                        sigma_hinge_sum_local += float(_sviol.item())
                        sigma_bar_sum_local += float(_sigma_bar.item())
                        sigma_count_local += 1

                if cfg.cert_distill_coef > 0 and target_anchor is not None:
                    _ct_key = (target, int(n_fit), int(eval_extrap_k))
                    _ct = cert_targets.get(_ct_key)
                    if _ct is None:
                        _canchor = constraint_anchor_lp.get(
                            target, target_anchor
                        ).detach().cpu().numpy()
                        _ct_sref = involved_sigma_reference(
                            _canchor,
                            n_fit=int(n_fit),
                            fit_top_k=int(cfg.ols_top_k),
                            extrap_k=int(eval_extrap_k),
                            m=int(cfg.sigma_ref_m),
                        )
                        _ct_idx, _ct_val, _ct_info = certified_tail_targets(
                            _canchor,
                            n_fit=int(n_fit),
                            fit_top_k=int(cfg.ols_top_k),
                            extrap_k=int(eval_extrap_k),
                            sigma_ref=float(_ct_sref),
                            sigma_margin=float(cfg.sigma_floor_margin),
                            two_sided=bool(cfg.sigma_floor_symmetric),
                            max_rank=int(cfg.cert_distill_ranks),
                        )
                        # Only the constraint channel's prompts carry gradient
                        # on union_lp_c; drop any target outside it.
                        _chan = constraint_grad_idx.get(target)
                        _inchan = (
                            np.isin(_ct_idx, _chan.cpu().numpy())
                            if _chan is not None else np.ones(_ct_idx.size, bool)
                        )
                        _ct_info["in_channel"] = int(_inchan.sum())
                        _ct = (
                            torch.as_tensor(_ct_idx[_inchan], device=union_lp.device),
                            torch.as_tensor(
                                _ct_val[_inchan], dtype=union_lp.dtype,
                                device=union_lp.device,
                            ),
                        )
                        cert_targets[_ct_key] = _ct
                        _ct_dir = outdir / "cert_targets"
                        _ct_dir.mkdir(parents=True, exist_ok=True)
                        (_ct_dir / f"{target}_fit{int(n_fit)}_e{int(eval_extrap_k)}.json").write_text(
                            json.dumps({
                                "info": _ct_info,
                                "idx": _ct_idx.tolist(),
                                "target": [float(t) for t in _ct_val],
                            })
                        )
                        log(
                            f"  cert targets {target}: base {_ct_info['base_abs_log10']:.3f} -> "
                            f"cert {_ct_info['cert_abs_log10']:.3f} dec, top "
                            f"{_ct_info['top_anchor_nats']:+.2f} -> {_ct_info['top_target_nats']:+.2f} "
                            f"nats, mean drop {_ct_info['mean_drop_nats']:.2f}, "
                            f"{_ct_info['in_channel']}/{_ct_info['ranks']} in channel"
                        )
                    _cd_cur = union_lp_c.index_select(0, _ct[0])
                    _cd_tgt = _ct[1]
                    if cfg.cert_distill_space == "logit":
                        _cd_cur = _cd_cur.clamp(max=-1e-6)
                        _cd_cur = _cd_cur - torch.log(-torch.expm1(_cd_cur))
                        _cd_tgt = _cd_tgt.clamp(max=-1e-6)
                        _cd_tgt = _cd_tgt - torch.log(-torch.expm1(_cd_tgt))
                    _cd_mse = ((_cd_cur - _cd_tgt) ** 2).mean()
                    _cd_term = cfg.cert_distill_coef * _cd_mse
                    if torch.isfinite(_cd_term):
                        lquant = lquant + _cd_term
                        cert_distill_sum_local += float(_cd_mse.item())
                        cert_distill_count_local += 1

                # Exact score-space gradient dL/dv_i over the whole union. The
                # loss depends on the model only through these 935 numbers, so
                # this vector IS what the objective asks of the model this step,
                # separated from whatever the model can actually deliver. Costs
                # one autograd pass over a 935-element leaf; no model involved.
                if dynamics_dumper is not None and dynamics_dumper.wants(step):
                    _leaf = union_lp_detached.detach().float().requires_grad_(True)
                    _leaf_loss, _leaf_stats = (
                        generic_differentiable_ols_extrapolated_primal_loss_bootstrap(
                            _leaf, fit_partitions, eval_partitions, **_boot_kwargs,
                        )
                    )
                    if cfg.loss_scale_norm_m > 0:
                        # Dump the gradient the model actually receives, which
                        # under scale normalization is not the raw one.
                        _lm = max(2, min(int(cfg.loss_scale_norm_m), n_union))
                        _lidx = torch.topk(_leaf.detach(), _lm).indices
                        _lvar = _leaf.index_select(0, _lidx).var(unbiased=False)
                        _leaf_loss = _leaf_loss * (
                            float(scale_norm_ref[target]) / _lvar.clamp_min(1e-6)
                        )
                    if (
                        cfg.gap_floor_m > 1
                        and cfg.gap_floor_lambda > 0
                        and target in gap_floor_g0
                    ):
                        # Same reason: the dumped gradient must include the
                        # hinge the model descends.
                        _lgm = min(int(cfg.gap_floor_m), n_union)
                        _lgidx = torch.topk(_leaf.detach(), _lgm).indices
                        _lgvals = _leaf.index_select(0, _lgidx)
                        _lg0 = gap_floor_g0[target].to(_leaf.dtype)
                        _lb = max(1, min(int(cfg.gap_floor_block), _lgm - 1))
                        _lcsum = torch.cat([
                            torch.zeros(1, dtype=_lg0.dtype, device=_lg0.device),
                            torch.cumsum(_lg0, 0),
                        ])
                        _lgu = (
                            _lgvals[:-_lb].detach() if cfg.gap_floor_level_gated
                            else _lgvals[:-_lb]
                        )
                        _lgviol = torch.relu(
                            (_lcsum[_lb:] - _lcsum[:-_lb]) - cfg.gap_floor_slack
                            - (_lgu - _lgvals[_lb:])
                        )
                        _leaf_loss = _leaf_loss + cfg.gap_floor_lambda * _lgviol.sum()
                    if cfg.sigma_floor_lambda > 0 and target_anchor is not None:
                        _lref = sigma_floor_ref.get(
                            (target, int(n_fit), int(eval_extrap_k)),
                        )
                        if _lref is not None:
                            _lsigs = []
                            for _p in range(fit_partitions.shape[0]):
                                _lf = fit_partitions[_p]
                                _le = eval_partitions[_p]
                                _lfs = _lf[torch.topk(
                                    _leaf.detach().index_select(0, _lf),
                                    min(int(cfg.ols_top_k), _lf.numel()),
                                ).indices]
                                _les = _le[torch.topk(
                                    _leaf.detach().index_select(0, _le),
                                    min(int(eval_extrap_k), _le.numel()),
                                ).indices]
                                _lsigs.append(_leaf.index_select(
                                    0, torch.cat([_lfs, _les]),
                                ).std(unbiased=False))
                            _lsbar = torch.stack(_lsigs).mean()
                            _lsgap = _lsbar.new_tensor(_lref) - _lsbar
                            _lsviol = (
                                torch.relu(
                                    _lsgap.abs() - cfg.sigma_floor_margin
                                )
                                if cfg.sigma_floor_symmetric
                                else torch.relu(
                                    _lsgap - cfg.sigma_floor_margin
                                )
                            )
                            _leaf_loss = (
                                _leaf_loss
                                + cfg.sigma_floor_lambda * _lsviol
                            )
                    (_leaf_grad,) = torch.autograd.grad(
                        _leaf_loss, [_leaf], allow_unused=True,
                    )
                    _score_grad = (
                        torch.zeros_like(_leaf) if _leaf_grad is None else _leaf_grad
                    )
                    _perm_losses = _leaf_stats.get("perm_losses")
                    dynamics_dumper.add_target(
                        target=target,
                        union_lp=union_lp_detached.detach().float().cpu().numpy(),
                        union_is_full=np.int8(0 if _ucache_hit else 1),
                        union_cached_idx=np.asarray(
                            sorted(_ucache.cached_union_indices)
                            if (_ucache_hit and _ucache is not None) else [],
                            dtype=np.int32,
                        ),
                        score_grad=_score_grad.detach().float().cpu().numpy(),
                        fit_partitions=fit_partitions.detach().cpu().numpy().astype(np.int32),
                        eval_partitions=eval_partitions.detach().cpu().numpy().astype(np.int32),
                        needed_grad_idx=needed_grad_idx.detach().cpu().numpy().astype(np.int32),
                        perm_losses=(
                            _perm_losses.detach().float().cpu().numpy()
                            if _perm_losses is not None else np.zeros(0, dtype=np.float32)
                        ),
                        anchor_lp=(
                            target_anchor.detach().float().cpu().numpy()
                            if target_anchor is not None else np.zeros(0, dtype=np.float32)
                        ),
                        fit_a=np.float32(float(_leaf_stats["fit_a"].item())),
                        fit_b=np.float32(float(_leaf_stats["fit_b"].item())),
                        n_fit=np.int32(n_fit),
                        n_eval=np.int32(n_eval),
                        eval_extrap_k=np.int32(eval_extrap_k),
                    )

                # Anchor penalty: a real term on a real loss, unlike the mask,
                # so the combined objective stays monotonically decreasable.
                # Scoped to the prompts this step scored with gradient, which is
                # the only set whose log_probs can move it.
                anchor_penalty_val = 0.0
                if (
                    cfg.anchor_penalty_coef > 0
                    and target_anchor is not None
                    and needed_grad_idx.numel() > 0
                ):
                    _sel = needed_grad_idx
                    _pen_lp = union_lp_c if shared_pass else union_lp
                    _excess = torch.relu(
                        _pen_lp.index_select(0, _sel)
                        - target_anchor.index_select(0, _sel).to(_pen_lp.dtype)
                    )
                    _pen = cfg.anchor_penalty_coef * (_excess ** 2).mean()
                    if torch.isfinite(_pen):
                        lquant = lquant + _pen
                        anchor_penalty_val = float(_pen.item())
                        penalty_sum_local += anchor_penalty_val
                        penalty_violation_frac_sum_local += float(
                            (_excess > 0).float().mean().item()
                        )
                        penalty_count_local += 1

                _pl = primal_stats.get("perm_losses")
                if _pl is not None and _pl.numel() > 1:
                    perm_std_sum_local += float(_pl.std(unbiased=False).item())
                    perm_std_count_local += 1
                _shape = {
                    k: float(primal_stats[k].item())
                    for k in ("fit_b", "fit_x_spread", "eval_x_spread",
                              "fit_v_spread", "eval_v_spread")
                    if k in primal_stats
                }
                if len(_shape) == 5 and all(math.isfinite(v) for v in _shape.values()):
                    fit_slope_sum_local += _shape["fit_b"]
                    fit_x_spread_sum_local += _shape["fit_x_spread"]
                    eval_x_spread_sum_local += _shape["eval_x_spread"]
                    fit_v_spread_sum_local += _shape["fit_v_spread"]
                    eval_v_spread_sum_local += _shape["eval_v_spread"]
                    shape_count_local += 1
                _wc = primal_stats.get("worsening_coef_deploy_mean")
                if _wc is not None and math.isfinite(float(_wc.item())):
                    worsening_coef_sum_local += float(_wc.item())
                    worsening_coef_count_local += 1

                if cfg.improving_only_grad and "improving_only_deploy_frac" in primal_stats:
                    iog_deploy_sum_local += float(primal_stats["improving_only_deploy_frac"].item())
                    iog_deploy_count_local += 1
                    iog_fit_sum_local += float(primal_stats["improving_only_fit_frac"].item())
                    iog_fit_count_local += 1
                _stage_calibration_update(cal_buffer, cal_updates_local, target, primal_stats)
                if "eval_max_value" in primal_stats and "predicted_max_value" in primal_stats:
                    _emv = float(primal_stats["eval_max_value"].item())
                    _pmv = float(primal_stats["predicted_max_value"].item())
                    if math.isfinite(_emv) and math.isfinite(_pmv):
                        eval_max_lp_sum_local += _emv
                        pred_max_lp_sum_local += _pmv
                        sq_log10_sum_local += ((_pmv - _emv) / _LN10) ** 2
                        lp_count_local += 1

                if not torch.isfinite(lquant) or not lquant.requires_grad:
                    continue
                scaled_lquant = lquant * (float(world_size) / float(max(1, n_step_targets_global)))
                scaled_lquant.backward()
                step_loss_local += lquant.item()
                n_targets_ok_local += 1
                continue

            # Fit path
            fit_grad = cfg.grad_mode in ("fit", "both")
            fit_lp_for_selection = compute_password_logprobs_batched(
                model, tokenizer, fit_subset,
                enable_thinking=(not cfg.disable_thinking),
                device=cfg.device,
                micro_batch_size=cfg.micro_batch_size,
                score_max_len=cfg.score_max_len,
                token_budget=cfg.score_token_budget,
                require_grad=False,
            )

            # Fit top-k selection is detached, so when gradients are needed we
            # can cheaply recompute only the selected prompts with grad instead
            # of retaining a graph for the entire fit pool.
            fit_topk = select_top_k_indices(fit_lp_for_selection, cfg.ols_top_k)
            # Record indices into fit_subset (== train_recs[target] when
            # fit_subsample_min == 0, which the IOG-penalty path enforces) for
            # aligning with pretrained_fit_lp_baseline.
            fit_lp_indices_in_subset: List[int] = []
            if fit_grad:
                fit_topk_records = [fit_subset[int(i)] for i in fit_topk.tolist()]
                fit_values_for_loss = compute_password_logprobs_batched(
                    model, tokenizer, fit_topk_records,
                    enable_thinking=(not cfg.disable_thinking),
                    device=cfg.device,
                    micro_batch_size=cfg.micro_batch_size,
                    score_max_len=cfg.score_max_len,
                    token_budget=cfg.score_token_budget,
                    require_grad=True,
                )
                fit_topk_for_loss = torch.arange(
                    fit_values_for_loss.shape[0],
                    device=fit_values_for_loss.device,
                    dtype=torch.long,
                )
                fit_lp_indices_in_subset = fit_topk.tolist()
            else:
                fit_values_for_loss = fit_lp_for_selection
                fit_topk_for_loss = fit_topk
                fit_lp_indices_in_subset = list(range(len(fit_subset)))

            # Eval path
            eval_grad = cfg.grad_mode in ("eval", "both")
            if cfg.loss_mode == "ols_primal_extrapolated":
                eval_extrap_k = count_extrapolated_eval_points(
                    len(fit_subset),
                    len(eval_subset),
                    plotting=cfg.plotting_position,
                )
                if eval_extrap_k == 0:
                    continue

                # --- Eval scoring with optional deploy cache ---
                _ecache = eval_caches.get(target)
                _cache_on = (
                    cfg.split_mode != "resample"
                    and
                    cfg.eval_cache_top_b > 0
                    and cfg.eval_cache_refresh_interval > 0
                    and cfg.eval_cache_top_b < len(eval_subset)
                )
                _cache_hit = (
                    _cache_on and _ecache is not None
                    and (step - _ecache.last_refresh_step) < cfg.eval_cache_refresh_interval
                )

                # Tracks which indices into eval_subset (== eval_recs[target]
                # in non-resample mode) appear in eval_lp_for_loss; used to
                # pull aligned baseline values for the IOG penalty.
                eval_lp_indices_in_subset: List[int] = []
                if _cache_hit:
                    # Cache hit: score only cached top-B subset
                    assert _ecache is not None
                    _cached_recs = [eval_subset[i] for i in _ecache.cached_indices]
                    _lp_cached = compute_password_logprobs_batched(
                        model, tokenizer, _cached_recs,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=False,
                    )
                    _topk_in_cached = select_top_k_indices(_lp_cached, eval_extrap_k)
                    _ecache.record_use(step=step)
                    if eval_grad:
                        _orig_idx = [_ecache.cached_indices[i] for i in _topk_in_cached.tolist()]
                        eval_lp_for_loss = compute_password_logprobs_batched(
                            model, tokenizer, [eval_subset[i] for i in _orig_idx],
                            enable_thinking=(not cfg.disable_thinking),
                            device=cfg.device,
                            micro_batch_size=cfg.micro_batch_size,
                            score_max_len=cfg.score_max_len,
                            token_budget=cfg.score_token_budget,
                            require_grad=True,
                        )
                        eval_indices_for_loss = torch.arange(
                            eval_lp_for_loss.shape[0], dtype=torch.long,
                            device=eval_lp_for_loss.device,
                        )
                        eval_lp_indices_in_subset = list(_orig_idx)
                    else:
                        eval_lp_for_loss = _lp_cached
                        eval_indices_for_loss = _topk_in_cached
                        eval_lp_indices_in_subset = list(_ecache.cached_indices)
                else:
                    # Cache miss or disabled: score full eval set
                    _lp_full = compute_password_logprobs_batched(
                        model, tokenizer, eval_subset,
                        enable_thinking=(not cfg.disable_thinking),
                        device=cfg.device,
                        micro_batch_size=cfg.micro_batch_size,
                        score_max_len=cfg.score_max_len,
                        token_budget=cfg.score_token_budget,
                        require_grad=False,
                    )
                    _eval_topk = select_top_k_indices(_lp_full, eval_extrap_k)
                    if _cache_on:
                        _b = min(cfg.eval_cache_top_b, len(eval_subset))
                        _cidx = select_top_k_indices(_lp_full, _b).detach().cpu().tolist()
                        _overlap = len(set(_ecache.cached_indices) & set(_cidx)) if _ecache else 0
                        _new = DeploySubsetCache(
                            cached_indices=_cidx,
                            last_refresh_step=step,
                            refresh_count=(_ecache.refresh_count if _ecache else 0) + 1,
                            total_uses=_ecache.total_uses if _ecache else 0,
                            total_cache_age=_ecache.total_cache_age if _ecache else 0,
                            total_candidate_count=_ecache.total_candidate_count if _ecache else 0,
                            total_overlap_with_previous=(
                                (_ecache.total_overlap_with_previous + _overlap) if _ecache else 0
                            ),
                            overlap_measurements=(
                                (_ecache.overlap_measurements + 1) if _ecache else 0
                            ),
                        )
                        _new.record_use(step=step)
                        eval_caches[target] = _new
                    if eval_grad:
                        eval_lp_for_loss = compute_password_logprobs_batched(
                            model, tokenizer,
                            [eval_subset[i] for i in _eval_topk.tolist()],
                            enable_thinking=(not cfg.disable_thinking),
                            device=cfg.device,
                            micro_batch_size=cfg.micro_batch_size,
                            score_max_len=cfg.score_max_len,
                            token_budget=cfg.score_token_budget,
                            require_grad=True,
                        )
                        eval_indices_for_loss = torch.arange(
                            eval_lp_for_loss.shape[0], dtype=torch.long,
                            device=eval_lp_for_loss.device,
                        )
                        eval_lp_indices_in_subset = _eval_topk.tolist()
                    else:
                        eval_lp_for_loss = _lp_full
                        eval_indices_for_loss = _eval_topk
                        eval_lp_indices_in_subset = list(range(len(eval_subset)))
            else:
                eval_lp_for_loss = compute_password_logprobs_batched(
                    model, tokenizer, eval_subset,
                    enable_thinking=(not cfg.disable_thinking),
                    device=cfg.device,
                    micro_batch_size=cfg.micro_batch_size,
                    score_max_len=cfg.score_max_len,
                    token_budget=cfg.score_token_budget,
                    require_grad=eval_grad,
                )
                eval_topk = select_top_k_indices(eval_lp_for_loss, cfg.ols_top_k)
                eval_lp_indices_in_subset = list(range(len(eval_subset)))

            # Compute loss
            if cfg.loss_mode == "ols_coeff":
                a_fit, b_fit = differentiable_ols_coefficients(
                    fit_values_for_loss, fit_topk_for_loss, n_total=len(fit_subset),
                    plotting=cfg.plotting_position,
                )
                # Eval set is the deployment/val ground truth: fit on ALL
                # eval points so (a_eval, b_eval) is the best available
                # estimate of the true tail line.
                eval_all_idx = torch.argsort(eval_lp_for_loss.detach(), descending=True)
                a_eval, b_eval = differentiable_ols_coefficients(
                    eval_lp_for_loss, eval_all_idx, n_total=len(eval_subset),
                    plotting=cfg.plotting_position,
                )
                if cfg.grad_mode == "fit":
                    a_eval, b_eval = a_eval.detach(), b_eval.detach()
                elif cfg.grad_mode == "eval":
                    a_fit, b_fit = a_fit.detach(), b_fit.detach()
                lquant = (a_fit - a_eval) ** 2 + (b_fit - b_eval) ** 2
            elif cfg.loss_mode in {"ols_primal", "ols_primal_weighted"}:
                eval_all_idx = torch.argsort(eval_lp_for_loss.detach(), descending=True)
                lquant, primal_stats = differentiable_ols_primal_loss(
                    fit_values_for_loss,
                    fit_topk_for_loss,
                    fit_n_total=len(fit_subset),
                    eval_log_probs=eval_lp_for_loss,
                    eval_indices=eval_all_idx,
                    eval_n_total=len(eval_subset),
                    plotting=cfg.plotting_position,
                    grad_mode=cfg.grad_mode,
                    use_weighted_decomposition=(cfg.loss_mode == "ols_primal_weighted"),
                )
            elif cfg.loss_mode == "ols_primal_extrapolated":
                lquant, primal_stats = differentiable_ols_extrapolated_primal_loss(
                    fit_values_for_loss,
                    fit_topk_for_loss,
                    fit_n_total=len(fit_subset),
                    eval_log_probs=eval_lp_for_loss,
                    eval_indices=eval_indices_for_loss,
                    eval_n_total=len(eval_subset),
                    plotting=cfg.plotting_position,
                    grad_mode=cfg.grad_mode,
                    extrapolated_weighting=cfg.extrapolated_weighting,
                    loss_space=cfg.extrapolated_loss_space,
                    improving_only_grad=cfg.improving_only_grad,
                    iog_scope=cfg.iog_scope,
                    iog_worsening_coef=cfg.iog_worsening_coef,
                    iog_fit_mask=cfg.iog_fit_mask,
                    calibration_alpha=cal_alpha,
                    calibration_beta=cal_beta,
                )
                _stage_calibration_update(cal_buffer, cal_updates_local, target, primal_stats)
                if cfg.improving_only_grad and "improving_only_deploy_frac" in primal_stats:
                    iog_deploy_sum_local += float(primal_stats["improving_only_deploy_frac"].item())
                    iog_deploy_count_local += 1
                    iog_fit_sum_local += float(primal_stats["improving_only_fit_frac"].item())
                    iog_fit_count_local += 1
                if "eval_max_value" in primal_stats and "predicted_max_value" in primal_stats:
                    _emv = float(primal_stats["eval_max_value"].item())
                    _pmv = float(primal_stats["predicted_max_value"].item())
                    if math.isfinite(_emv) and math.isfinite(_pmv):
                        eval_max_lp_sum_local += _emv
                        pred_max_lp_sum_local += _pmv
                        sq_log10_sum_local += ((_pmv - _emv) / _LN10) ** 2
                        lp_count_local += 1
            else:
                # Legacy: extrapolate to sampled deployment scale(s)
                log_p_star_fit = differentiable_ols_forecast(
                    fit_values_for_loss, fit_topk_for_loss,
                    n_total=len(fit_subset),
                    deploy_queries=deploy_queries,
                    plotting=cfg.plotting_position,
                )
                log_p_star_eval = differentiable_ols_forecast(
                    eval_lp_for_loss, eval_topk,
                    n_total=len(eval_subset),
                    deploy_queries=deploy_queries,
                    plotting=cfg.plotting_position,
                )
                if cfg.grad_mode == "fit":
                    log_p_star_eval = log_p_star_eval.detach()
                elif cfg.grad_mode == "eval":
                    log_p_star_fit = log_p_star_fit.detach()
                lquant = ((log_p_star_fit - log_p_star_eval) ** 2).mean()

            # --- IOG penalty (squared hinge against pretrained baseline) ---
            # Replaces the IOG mask: a real gradient on a real loss term, so the
            # combined objective is monotonically decreasable (under SGD on a
            # fixed batch). Symmetric over fit-side AND deploy-side prompts to
            # match the IOG mask's coverage of both.
            penalty_value = 0.0
            penalty_violation_frac = 0.0
            penalty_fit_value = 0.0
            penalty_fit_violation_frac = 0.0
            if cfg.iog_penalty_coef > 0:
                # Deploy-side penalty
                if (
                    target in pretrained_eval_lp_baseline
                    and eval_lp_for_loss.requires_grad
                    and eval_lp_indices_in_subset
                ):
                    idx_t = torch.tensor(
                        eval_lp_indices_in_subset,
                        dtype=torch.long,
                        device=eval_lp_for_loss.device,
                    )
                    baseline_for_subset = pretrained_eval_lp_baseline[target].to(
                        device=eval_lp_for_loss.device,
                    ).index_select(0, idx_t)
                    if baseline_for_subset.shape == eval_lp_for_loss.shape:
                        excess = torch.relu(eval_lp_for_loss - baseline_for_subset)
                        penalty = cfg.iog_penalty_coef * (excess ** 2).mean()
                        if torch.isfinite(penalty):
                            lquant = lquant + penalty
                            penalty_value = float(penalty.item())
                            penalty_violation_frac = float(
                                (excess > 0).float().mean().item()
                            )
                # Fit-side penalty (symmetric)
                if (
                    target in pretrained_fit_lp_baseline
                    and fit_values_for_loss.requires_grad
                    and fit_lp_indices_in_subset
                ):
                    fidx_t = torch.tensor(
                        fit_lp_indices_in_subset,
                        dtype=torch.long,
                        device=fit_values_for_loss.device,
                    )
                    fbaseline = pretrained_fit_lp_baseline[target].to(
                        device=fit_values_for_loss.device,
                    ).index_select(0, fidx_t)
                    if fbaseline.shape == fit_values_for_loss.shape:
                        fexcess = torch.relu(fit_values_for_loss - fbaseline)
                        fpenalty = cfg.iog_penalty_coef * (fexcess ** 2).mean()
                        if torch.isfinite(fpenalty):
                            lquant = lquant + fpenalty
                            penalty_fit_value = float(fpenalty.item())
                            penalty_fit_violation_frac = float(
                                (fexcess > 0).float().mean().item()
                            )

            if not torch.isfinite(lquant) or not lquant.requires_grad:
                continue

            scaled_lquant = lquant * (float(world_size) / float(max(1, n_step_targets_global)))
            # Backpropagate per target so we do not retain graphs for every
            # local target until the end of the step.
            scaled_lquant.backward()
            step_loss_local += lquant.item()
            n_targets_ok_local += 1
            if cfg.iog_penalty_coef > 0:
                penalty_sum_local += penalty_value
                penalty_count_local += 1
                penalty_violation_frac_sum_local += penalty_violation_frac
                penalty_fit_sum_local += penalty_fit_value
                penalty_fit_violation_frac_sum_local += penalty_fit_violation_frac

        # --- Calibration buffer update (end of step, after all local targets) ---
        # Gather every rank's fresh contributions so all buffers stay identical,
        # then apply. Each contribution is ~3 numbers; this is the only
        # cross-rank communication calibration adds.
        if cal_buffer is not None:
            all_updates = cal_updates_local
            if distributed and dist.is_initialized():
                gathered: list[list[Tuple[str, float, float]] | None] = [
                    None for _ in range(world_size)
                ]
                dist.all_gather_object(gathered, cal_updates_local)
                all_updates = [u for part in gathered if part for u in part]
            for _tgt, _q, _y in all_updates:
                cal_buffer.update(_tgt, _q, _y)

        # --- KL regularization ---
        kl_single = 0.0
        kl_trunc = 0.0
        if cfg.kl_coef > 0 and base_model is not None and kl_train_prompts:
            kl_rng = _step_rng(cfg.seed, step, stream=31)
            batch_prompts = kl_rng.sample(
                kl_train_prompts, min(cfg.kl_batch_size, len(kl_train_prompts)),
            )
            kl_ids, kl_mask = _tokenize_for_kl(
                tokenizer, batch_prompts, cfg.device,
                max_len=cfg.kl_tokenize_max_len, truncation_side=cfg.kl_truncation_side,
            )

            # Single-token KL every step
            kl_1 = compute_kl_single_token(model, base_model, kl_ids, kl_mask)
            kl_loss = cfg.kl_coef * kl_1
            kl_single = kl_1.item()

            # Truncated KL every N steps
            if cfg.kl_trunc_len > 0 and step % cfg.kl_trunc_every == 0:
                kl_t = compute_kl_truncated(
                    model, base_model, kl_ids, kl_mask,
                    n_tokens=cfg.kl_trunc_len,
                )
                kl_loss = kl_loss + cfg.kl_coef * kl_t
                kl_trunc = kl_t.item()

            if torch.isfinite(kl_loss) and kl_loss.requires_grad:
                kl_loss.backward()

        # Gradient step. clip_grad_norm_ returns the PRE-clip total norm, which
        # is the honest measure of how big a gradient the step produced; after
        # clipping every step looks the same size by construction.
        grad_norm_local = 0.0
        if n_targets_ok_local > 0 or (cfg.kl_coef > 0 and kl_single > 0):
            if cfg.grad_clip > 0:
                grad_norm_local = float(
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip).item()
                )
            elif dynamics_dumper is not None:
                grad_norm_local = float(
                    torch.nn.utils.clip_grad_norm_(trainable, float("inf")).item()
                )
            optimizer.step()

        stats = torch.tensor(
            [
                step_loss_local,
                float(n_targets_ok_local),
                iog_deploy_sum_local,
                float(iog_deploy_count_local),
                iog_fit_sum_local,
                float(iog_fit_count_local),
                kl_single,
                kl_trunc,
                eval_max_lp_sum_local,
                pred_max_lp_sum_local,
                float(lp_count_local),
                penalty_sum_local,
                penalty_violation_frac_sum_local,
                float(penalty_count_local),
                penalty_fit_sum_local,
                penalty_fit_violation_frac_sum_local,
                sq_log10_sum_local,
                perm_std_sum_local,
                float(perm_std_count_local),
                fit_slope_sum_local,
                fit_x_spread_sum_local,
                eval_x_spread_sum_local,
                fit_v_spread_sum_local,
                eval_v_spread_sum_local,
                float(shape_count_local),
                worsening_coef_sum_local,
                float(worsening_coef_count_local),
                grad_norm_local,
                scale_norm_sum_local,
                float(scale_norm_count_local),
                gap_hinge_sum_local,
                gap_viol_frac_sum_local,
                float(gap_count_local),
                sigma_hinge_sum_local,
                sigma_bar_sum_local,
                float(sigma_count_local),
            ],
            device=cfg.device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        step_loss = float(stats[0].item())
        n_targets_ok = int(round(stats[1].item()))
        mean_iog_deploy = None
        mean_iog_fit = None
        if stats[3].item() > 0:
            mean_iog_deploy = float(stats[2].item() / stats[3].item())
            mean_iog_fit = float(stats[4].item() / stats[5].item())
        if distributed:
            kl_single = float(stats[6].item() / world_size)
            kl_trunc = float(stats[7].item() / world_size)
        else:
            kl_single = float(stats[6].item())
            kl_trunc = float(stats[7].item())
        mean_eval_max_lp = None
        mean_pred_max_lp = None
        mean_sq_log10 = None
        if stats[10].item() > 0:
            mean_eval_max_lp = float(stats[8].item() / stats[10].item())
            mean_pred_max_lp = float(stats[9].item() / stats[10].item())
            mean_sq_log10 = float(stats[16].item() / stats[10].item())
        mean_iog_penalty = None
        mean_iog_penalty_violation_frac = None
        mean_iog_penalty_fit = None
        mean_iog_penalty_fit_violation_frac = None
        if stats[13].item() > 0:
            mean_iog_penalty = float(stats[11].item() / stats[13].item())
            mean_iog_penalty_violation_frac = float(stats[12].item() / stats[13].item())
            mean_iog_penalty_fit = float(stats[14].item() / stats[13].item())
            mean_iog_penalty_fit_violation_frac = float(stats[15].item() / stats[13].item())
        mean_perm_std = (
            float(stats[17].item() / stats[18].item()) if stats[18].item() > 0 else None
        )
        shape_means: dict[str, float] | None = None
        if stats[24].item() > 0:
            _n_shape = stats[24].item()
            shape_means = {
                "fit_slope_mean": float(stats[19].item() / _n_shape),
                "fit_topk_x_spread": float(stats[20].item() / _n_shape),
                "deploy_topb_x_spread": float(stats[21].item() / _n_shape),
                "fit_topk_spread_log10": float(stats[22].item() / _n_shape / _LN10),
                "deploy_topb_spread_log10": float(stats[23].item() / _n_shape / _LN10),
            }
        mean_worsening_coef = (
            float(stats[25].item() / stats[26].item()) if stats[26].item() > 0 else None
        )
        grad_norm = (
            float(stats[27].item() / max(1, world_size)) if distributed else grad_norm_local
        )
        mean_scale_norm = (
            float(stats[28].item() / stats[29].item()) if stats[29].item() > 0 else None
        )
        mean_gap_hinge = (
            float(stats[30].item() / stats[32].item()) if stats[32].item() > 0 else None
        )
        mean_gap_viol_frac = (
            float(stats[31].item() / stats[32].item()) if stats[32].item() > 0 else None
        )
        mean_sigma_hinge = (
            float(stats[33].item() / stats[35].item()) if stats[35].item() > 0 else None
        )
        mean_sigma_bar = (
            float(stats[34].item() / stats[35].item()) if stats[35].item() > 0 else None
        )

        avg_loss = step_loss / max(1, n_targets_ok)
        step_sec = time.perf_counter() - t_step
        if distributed:
            step_sec_t = torch.tensor([step_sec], device=cfg.device, dtype=torch.float64)
            dist.all_reduce(step_sec_t, op=dist.ReduceOp.MAX)
            step_sec = float(step_sec_t.item())

        history_entry = {
            "step": step,
            "loss": avg_loss,
            "n_targets": n_targets_ok,
            "step_sec": step_sec,
        }
        if cfg.kl_coef > 0:
            history_entry["kl_single"] = kl_single
            if kl_trunc > 0:
                history_entry["kl_trunc"] = kl_trunc
        if cfg.loss_mode == "extrapolate":
            history_entry["deploy_queries"] = deploy_queries
        if mean_iog_deploy is not None and mean_iog_fit is not None:
            history_entry["improving_only_deploy_frac"] = mean_iog_deploy
            history_entry["improving_only_fit_frac"] = mean_iog_fit
        if mean_eval_max_lp is not None and mean_pred_max_lp is not None:
            history_entry["worst_actual_log_prob_mean"] = mean_eval_max_lp
            history_entry["worst_predicted_log_prob_mean"] = mean_pred_max_lp
        if mean_sq_log10 is not None:
            history_entry["train_forecast_sq_log10_error"] = mean_sq_log10
        if mean_perm_std is not None:
            history_entry["loss_perm_std"] = mean_perm_std
        if shape_means is not None:
            history_entry.update(shape_means)
        if mean_worsening_coef is not None:
            history_entry["worsening_coef_deploy_mean"] = mean_worsening_coef
        if mean_scale_norm is not None:
            # 1 at step 0 by construction; above 1 means the tail has narrowed
            # since, which is exactly the movement the normalization stops paying
            # for. Reading it against the loss says whether a run is descending
            # on shape or on scale.
            history_entry["scale_norm_factor"] = mean_scale_norm
        if mean_gap_hinge is not None:
            # Nats still missing below the gap floors, summed over gaps and
            # averaged over targets. 0 means the step is inside the feasible
            # set, which is the post-hoc certificate the exact penalty needs.
            history_entry["gap_hinge_mean"] = mean_gap_hinge
            history_entry["gap_violation_frac"] = mean_gap_viol_frac
        if mean_sigma_hinge is not None:
            # Nats of realized-set sigma still missing below the line's
            # reference, averaged over targets. 0 is the feasibility
            # certificate, same as for the gap floors.
            history_entry["sigma_hinge_mean"] = mean_sigma_hinge
            history_entry["sigma_bar_mean"] = mean_sigma_bar
        if cert_distill_count_local > 0:
            # Rank-0 targets only (not all-reduced): mean squared nats between
            # the channel scores and the certified tail.
            history_entry["cert_distill_mse_local"] = (
                cert_distill_sum_local / cert_distill_count_local
            )
        if grad_norm > 0:
            history_entry["grad_norm"] = grad_norm
        if mean_iog_penalty is not None and cfg.anchor_penalty_coef > 0:
            # Same accumulators, different term: name it for what it is so an
            # anchor-penalty run's history is not read as an IOG-penalty run's.
            history_entry["anchor_penalty_mean"] = mean_iog_penalty
            history_entry["anchor_penalty_violation_frac"] = mean_iog_penalty_violation_frac
        elif mean_iog_penalty is not None:
            history_entry["iog_penalty_mean"] = mean_iog_penalty
            history_entry["iog_penalty_violation_frac"] = mean_iog_penalty_violation_frac
            history_entry["iog_penalty_fit_mean"] = mean_iog_penalty_fit
            history_entry["iog_penalty_fit_violation_frac"] = mean_iog_penalty_fit_violation_frac
        cal_alpha_log = cal_beta_log = None
        if cal_buffer is not None:
            cal_alpha_log, cal_beta_log = cal_buffer.global_coeffs()
            history_entry["calibration_alpha"] = float(cal_alpha_log)
            history_entry["calibration_beta"] = float(cal_beta_log)
            history_entry["calibration_n_groups"] = float(cal_buffer.n_groups)
        history.append(history_entry)
        if scale_norm_dirty and is_main_process:
            # References are captured the first time each target is scored, so
            # this settles within the first few steps and then never rewrites.
            scale_norm_path.write_text(json.dumps(scale_norm_ref, indent=2))
            scale_norm_dirty = False
        if dynamics_dumper is not None and dynamics_dumper.wants(step):
            dynamics_dumper.flush(step)

        if step % cfg.log_every == 0:
            if deploy_queries is None:
                extra = ""
            elif isinstance(deploy_queries, list):
                extra = f" | N={min(deploy_queries):.0e}..{max(deploy_queries):.0e}(x{len(deploy_queries)})"
            else:
                extra = f" | N={deploy_queries:.0e}"
            if mean_iog_fit is not None and mean_iog_deploy is not None:
                extra += f" | iog_fit={mean_iog_fit:.2f} | iog_deploy={mean_iog_deploy:.2f}"
            if mean_eval_max_lp is not None and mean_pred_max_lp is not None:
                extra += f" | worst_lp={mean_eval_max_lp:+.2f} | pred_lp={mean_pred_max_lp:+.2f}"
            if mean_sq_log10 is not None:
                # Honest per-step forecast error (mean of SQUARED per-target
                # errors). Logged as well as stored because history.json is only
                # flushed every save_every steps, and in-flight comparison of
                # long 4B/8B runs needs it sooner than that.
                extra += f" | sq10={mean_sq_log10:.4f}"
            if mean_iog_penalty is not None:
                extra += (
                    f" | iog_pen_d={mean_iog_penalty:.3f}"
                    f" | viol_d={mean_iog_penalty_violation_frac:.2f}"
                    f" | iog_pen_f={mean_iog_penalty_fit:.3f}"
                    f" | viol_f={mean_iog_penalty_fit_violation_frac:.2f}"
                )
            if cfg.kl_coef > 0:
                extra += f" | kl1={kl_single:.4f}"
                if kl_trunc > 0:
                    extra += f" klt={kl_trunc:.4f}"
            if cal_alpha_log is not None:
                extra += (
                    f" | cal a={cal_alpha_log:.3f} b={cal_beta_log:.3f}"
                    f" n={cal_buffer.n_groups}"
                )
            log(
                f"  step {step:4d} | loss={avg_loss:.6f}{extra} | {step_sec:.1f}s"
            )

        # Best-checkpoint tracking. Saved on the TRAIN-side squared forecast
        # error, which never touches the held-out split, so selecting on it is
        # not leakage. Written to `best/` alongside `best_step.json`; `final/`
        # is untouched and remains the canonical reported checkpoint unless a
        # run explicitly reports the best one.
        if is_main_process and mean_sq_log10 is not None and step >= cfg.best_ckpt_min_step:
            if best_sq_log10 is None or mean_sq_log10 < best_sq_log10:
                best_sq_log10 = mean_sq_log10
                best_dir = outdir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                raw_model.save_pretrained(str(best_dir))
                (outdir / "best_step.json").write_text(json.dumps({
                    "step": step,
                    # Steps before this were not eligible: either the run started
                    # here, or it resumed from a checkpoint written before
                    # `train_forecast_sq_log10_error` existed, so earlier steps
                    # carry no comparable value. Without this field a resumed run's
                    # `best/` reads as a best-over-the-whole-run when it is not.
                    "considered_from_step": min(
                        (h["step"] for h in history
                         if h.get("train_forecast_sq_log10_error") is not None),
                        default=step,
                    ),
                    "train_forecast_sq_log10_error": mean_sq_log10,
                    "worst_actual_log_prob_mean": mean_eval_max_lp,
                    "worst_predicted_log_prob_mean": mean_pred_max_lp,
                    "loss": avg_loss,
                    "selection_metric": "train_forecast_sq_log10_error",
                    "note": (
                        "Selected on train-side error only; the held-out split "
                        "was not consulted."
                    ),
                }, indent=2))

        if (
            is_main_process
            and cfg.adapter_snapshot_every > 0
            and (step + 1) % cfg.adapter_snapshot_every == 0
        ):
            raw_model.save_pretrained(str(outdir / "snapshots" / f"step_{step + 1:04d}"))

        # Checkpoint
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            elapsed_so_far = time.perf_counter() - t_start
            if is_main_process:
                _write_history_json(outdir, history)
            _save_phase_resume_checkpoint("train", step + 1, elapsed_sec=elapsed_so_far)
            log(f"  Resume checkpoint saved at step {step + 1}")

    elapsed = time.perf_counter() - t_start
    if resume_phase == "train":
        log(f"\nTraining complete in {elapsed:.1f}s")

    if is_main_process:
        # Save final checkpoint
        final_dir = outdir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        # Save history
        _write_history_json(outdir, history)
    _save_phase_resume_checkpoint("final_eval", cfg.steps, elapsed_sec=elapsed)
    if is_main_process:
        final_eval = run_evaluation_suite(
            raw_model,
            tokenizer,
            train_recs_eval,
            eval_recs_eval,
            test_recs_eval,
            cfg,
            outdir=outdir,
            prefix="",
            label="",
            base_model=base_model,
            kl_eval_prompts=kl_eval_prompts,
        )
    else:
        final_eval = None
    final_eval = _broadcast_object(final_eval, src=0)
    # Keep a resumable "complete" checkpoint instead of deleting them: it holds
    # the full optimizer/rng/cache state at step cfg.steps, so a finished run
    # can be EXTENDED later by relaunching with a larger --steps (the resume
    # path flips complete -> train when start_step < steps). Deleting these
    # made every completed run a dead end -- the 400step arm existed only
    # because canonical s17 could not be continued (2026-07-25). Cost: ~100MB
    # per run dir. Called on all ranks: the cache gather inside is collective.
    _save_phase_resume_checkpoint("complete", cfg.steps, elapsed_sec=elapsed)
    test_results = final_eval["test_results"]
    test_pool_diagnostic = final_eval["test_pool_diagnostic"]
    kl_eval_results = final_eval["kl_eval_results"]
    _distributed_barrier()
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    return {
        "history": history,
        "baseline_test_results": baseline_eval["test_results"],
        "baseline_test_pool_diagnostic": baseline_eval["test_pool_diagnostic"],
        "baseline_kl_eval_results": baseline_eval["kl_eval_results"],
        "baseline_test_trace_artifacts": baseline_eval.get("test_trace_artifacts", {}),
        "test_results": test_results,
        "test_pool_diagnostic": test_pool_diagnostic,
        "kl_eval_results": kl_eval_results,
        "test_trace_artifacts": final_eval.get("test_trace_artifacts", {}),
        "elapsed_sec": elapsed,
    }


_LN10 = math.log(10.0)


def _target_seed_offset(target: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(target))


def _test_target_counts_for_target(
    target: str,
    train_recs: Dict[str, List[Dict]],
    eval_recs: Dict[str, List[Dict]],
    n_total: int,
    cfg: LQuantConfig,
) -> Tuple[int, int]:
    meta_fit_n = len(train_recs.get(target, []))
    meta_eval_n = len(eval_recs.get(target, []))
    if n_total <= 1:
        return n_total, 0

    min_fit = min(max(1, cfg.ols_top_k), n_total - 1)
    if meta_fit_n > 0:
        n_fit = min(meta_fit_n, n_total - 1)
    else:
        n_fit = int(n_total * min(max(cfg.fit_fraction, 1e-6), 1.0 - 1e-6))
        n_fit = max(min_fit, n_fit)
        n_fit = min(n_fit, n_total - 1)

    if meta_fit_n > 0 and meta_eval_n > 0 and n_total >= meta_fit_n + meta_eval_n:
        n_fit = meta_fit_n
        n_deploy = meta_eval_n
    else:
        n_deploy = n_total - n_fit

    n_fit = max(min_fit, min(n_fit, n_total - 1))
    n_deploy = max(1, min(n_deploy, n_total - n_fit))
    return n_fit, n_deploy


def _split_test_records_for_target(
    target: str,
    recs: List[Dict],
    train_recs: Dict[str, List[Dict]],
    eval_recs: Dict[str, List[Dict]],
    cfg: LQuantConfig,
) -> Tuple[List[Dict], List[Dict]]:
    if len(recs) <= 1:
        return list(recs), []

    if _fit_deploy_shift_enabled(cfg):
        # Family-shift protocol: split_pool already drew this slot's exact
        # fit/deploy partition (test_records = fit + deploy); re-separate it
        # by family membership. No re-shuffle, no re-sizing.
        fit_out = [r for r in recs if r.get("family_id") in cfg.fit_families]
        deploy_out = [r for r in recs if r.get("family_id") in cfg.deploy_families]
        return fit_out, deploy_out

    n_total = len(recs)
    n_fit, n_deploy = _test_target_counts_for_target(target, train_recs, eval_recs, n_total, cfg)

    shuffled = list(recs)
    rng = random.Random(cfg.seed + 20260401 + _target_seed_offset(target))
    rng.shuffle(shuffled)
    return shuffled[:n_fit], shuffled[n_fit:n_fit + n_deploy]


def evaluate_on_test(
    model,
    tokenizer,
    train_recs: Dict[str, List[Dict]],
    eval_recs: Dict[str, List[Dict]],
    test_recs: Dict[str, List[Dict]],
    cfg: LQuantConfig,
) -> tuple[Dict, list[dict[str, object]]]:
    """Run RL-style held-out fit/deploy evaluation on test records."""
    results = {}
    trace_rows: list[dict[str, object]] = []
    items = [(target, recs) for target, recs in test_recs.items() if recs]
    started_at = time.perf_counter()
    total_items = len(items)
    for idx, (target, recs) in enumerate(items, start=1):
        fit_recs, deploy_recs = _split_test_records_for_target(target, recs, train_recs, eval_recs, cfg)
        if not fit_recs or not deploy_recs:
            continue
        if _should_log_progress(idx, total_items):
            print(
                "  held-out fit/deploy target "
                f"{idx}/{total_items} | target={target} "
                f"fit={len(fit_recs)} deploy={len(deploy_recs)} "
                f"| elapsed={_format_elapsed_seconds(time.perf_counter() - started_at)}",
                flush=True,
            )
        with torch.no_grad():
            fit_lp = compute_password_logprobs_batched(
                model, tokenizer, fit_recs,
                enable_thinking=(not cfg.disable_thinking),
                device=cfg.device,
                micro_batch_size=cfg.micro_batch_size,
                score_max_len=cfg.score_max_len,
                token_budget=cfg.score_token_budget,
                require_grad=False,
            )
            deploy_lp = compute_password_logprobs_batched(
                model, tokenizer, deploy_recs,
                enable_thinking=(not cfg.disable_thinking),
                device=cfg.device,
                micro_batch_size=cfg.micro_batch_size,
                score_max_len=cfg.score_max_len,
                token_budget=cfg.score_token_budget,
                require_grad=False,
            )
        fit_lp = fit_lp.float()
        deploy_lp = deploy_lp.float()

        fit_log_probs = [float(v) for v in fit_lp.tolist()]
        deploy_log_probs = [float(v) for v in deploy_lp.tolist()]

        pred_log_prob, fit_a, fit_b = fit_ols_tail_from_scores(
            fit_log_probs,
            n_top=cfg.ols_top_k,
            plotting=cfg.plotting_position,
            deploy_queries=len(deploy_log_probs),
            score_mode="gumbel_log_prob",
        )
        _deploy_self_pred_log_prob, deploy_a, deploy_b = fit_ols_tail_from_scores(
            deploy_log_probs,
            n_top=cfg.ols_top_k,
            plotting=cfg.plotting_position,
            deploy_queries=len(deploy_log_probs),
            score_mode="gumbel_log_prob",
        )
        actual_worst_log_prob = float(deploy_lp.max().item()) if deploy_log_probs else float("-inf")
        predicted_worst_prob = safe_prob_from_log_prob(pred_log_prob)
        actual_worst_prob = safe_prob_from_log_prob(actual_worst_log_prob)
        fit_mean_log_prob = float(torch.logsumexp(fit_lp, dim=0).item() - math.log(len(fit_log_probs)))
        deploy_mean_log_prob = float(torch.logsumexp(deploy_lp, dim=0).item() - math.log(len(deploy_log_probs)))

        extrapolated_eval_points = count_extrapolated_eval_points(
            len(fit_log_probs),
            len(deploy_log_probs),
            plotting=cfg.plotting_position,
        )
        fit_topk = select_top_k_indices(fit_lp, cfg.ols_top_k)
        deploy_topk = select_top_k_indices(deploy_lp, extrapolated_eval_points)
        heldout_objective, _heldout_details = differentiable_ols_extrapolated_primal_loss(
            fit_lp,
            fit_topk,
            len(fit_log_probs),
            deploy_lp,
            deploy_topk,
            len(deploy_log_probs),
            plotting=cfg.plotting_position,
            grad_mode="both",
            extrapolated_weighting=cfg.extrapolated_weighting,
            loss_space=cfg.extrapolated_loss_space,
        )
        log10_e = 1.0 / math.log(10.0)
        worst_abs_log10_error = forecast_accuracy_from_log_probs(pred_log_prob, actual_worst_log_prob)
        weighted_mean_abs_log10_error = float(_heldout_details["weighted_original_mae"].item() * log10_e)
        mean_abs_log10_error = float(_heldout_details["unweighted_original_mae"].item() * log10_e)
        weighted_mean_sq_log10_error = float(_heldout_details["weighted_original_mse"].item() * (log10_e ** 2))
        mean_sq_log10_error = float(_heldout_details["unweighted_original_mse"].item() * (log10_e ** 2))

        results[target] = {
            "fit_count": len(fit_recs),
            "deploy_count": len(deploy_recs),
            "n_test_prompts_total": len(recs),
            "n_test_prompts_used": len(fit_recs) + len(deploy_recs),
            "n_test_prompts_unused": len(recs) - len(fit_recs) - len(deploy_recs),
            "fit_fraction": len(fit_recs) / float(len(recs)),
            "predicted_worst_log_prob": float(pred_log_prob),
            "actual_worst_log_prob": float(actual_worst_log_prob),
            "predicted_worst_prob": float(predicted_worst_prob),
            "actual_worst_prob": float(actual_worst_prob),
            "forecast_abs_error": float(abs(predicted_worst_prob - actual_worst_prob)),
            "forecast_abs_log10_error": float(worst_abs_log10_error),
            "forecast_sq_log10_error": float(worst_abs_log10_error ** 2),
            "forecast_signed_log10_error": float(
                signed_log_error_from_log_probs(pred_log_prob, actual_worst_log_prob)
            ),
            "extrapolated_mean_abs_log10_error": mean_abs_log10_error,
            "extrapolated_weighted_mean_abs_log10_error": weighted_mean_abs_log10_error,
            "extrapolated_mean_sq_log10_error": mean_sq_log10_error,
            "extrapolated_weighted_mean_sq_log10_error": weighted_mean_sq_log10_error,
            "fit_ols_a": float(fit_a),
            "fit_ols_b": float(fit_b),
            "deploy_ols_a": float(deploy_a),
            "deploy_ols_b": float(deploy_b),
            "fit_mean_log_prob": float(fit_mean_log_prob),
            "deploy_mean_log_prob": float(deploy_mean_log_prob),
            "fit_mean_prob": float(safe_prob_from_log_prob(fit_mean_log_prob)),
            "deploy_mean_prob": float(safe_prob_from_log_prob(deploy_mean_log_prob)),
            "heldout_extrapolated_loss": float(heldout_objective.item()),
            "heldout_extrapolated_loss_space": cfg.extrapolated_loss_space,
            "extrapolated_eval_points": int(extrapolated_eval_points),
            "extrapolated_weighting": cfg.extrapolated_weighting,
        }
        trace_rows.append(
            {
                "target": target,
                "fit_log_probs": np.asarray(fit_log_probs, dtype=np.float32),
                "deploy_log_probs": np.asarray(deploy_log_probs, dtype=np.float32),
            }
        )
    return results, trace_rows


def evaluate_test_pool_diagnostic(
    model,
    tokenizer,
    test_recs: Dict[str, List[Dict]],
    cfg: LQuantConfig,
) -> Dict:
    """Legacy whole-pool diagnostic kept for backwards comparison."""
    results = {}
    items = [(target, recs) for target, recs in test_recs.items() if recs]
    started_at = time.perf_counter()
    total_items = len(items)
    for idx, (target, recs) in enumerate(items, start=1):
        if not recs:
            continue
        if _should_log_progress(idx, total_items):
            print(
                "  whole-pool diagnostic target "
                f"{idx}/{total_items} | target={target} n={len(recs)} "
                f"| elapsed={_format_elapsed_seconds(time.perf_counter() - started_at)}",
                flush=True,
            )
        with torch.no_grad():
            lp = compute_password_logprobs_batched(
                model, tokenizer, recs,
                enable_thinking=(not cfg.disable_thinking),
                device=cfg.device,
                micro_batch_size=cfg.micro_batch_size,
                score_max_len=cfg.score_max_len,
                token_budget=cfg.score_token_budget,
                require_grad=False,
            )
        lp = lp.float()
        log_probs = [float(v) for v in lp.tolist()]
        pred_log_prob, _a, _b = fit_ols_tail_from_scores(
            log_probs,
            n_top=cfg.ols_top_k,
            plotting=cfg.plotting_position,
            deploy_queries=int(10 ** cfg.log_n_max),
            score_mode="gumbel_log_prob",
        )
        actual_worst_log_prob = float(lp.max().item()) if log_probs else float("-inf")
        results[target] = {
            "ols_pred_log_prob": float(pred_log_prob),
            "actual_worst_log_prob": float(actual_worst_log_prob),
            "ols_pred": float(safe_prob_from_log_prob(pred_log_prob)),
            "actual_worst": float(safe_prob_from_log_prob(actual_worst_log_prob)),
            "n_prompts": len(recs),
        }
    return results


def evaluate_heldout_kl(
    model,
    base_model,
    tokenizer,
    kl_eval_prompts: List[str],
    cfg: LQuantConfig,
) -> Dict:
    """Evaluate held-out single-token KL against the frozen reference model."""
    kl_eval_results = {}
    if base_model is None or not kl_eval_prompts:
        return kl_eval_results

    print(
        "  Starting held-out KL eval | "
        f"n_prompts={len(kl_eval_prompts)} batch_size={cfg.kl_batch_size}",
        flush=True,
    )
    started_at = time.perf_counter()
    gc.collect()
    if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()
    kl_eval_1tok, n_eval_prompts = evaluate_kl_single_token_batched(
        model,
        base_model,
        tokenizer,
        kl_eval_prompts,
        device=cfg.device,
        batch_size=cfg.kl_batch_size,
        max_prompts=256,
        max_len=cfg.kl_tokenize_max_len,
        truncation_side=cfg.kl_truncation_side,
    )
    kl_eval_results["single_token"] = kl_eval_1tok
    kl_eval_results["n_prompts"] = n_eval_prompts
    kl_eval_results["datasets"] = cfg.kl_eval_datasets
    print(
        "  Finished held-out KL eval | "
        f"elapsed={_format_elapsed_seconds(time.perf_counter() - started_at)}",
        flush=True,
    )
    return kl_eval_results


def run_evaluation_suite(
    model,
    tokenizer,
    train_recs: Dict[str, List[Dict]],
    eval_recs: Dict[str, List[Dict]],
    test_recs: Dict[str, List[Dict]],
    cfg: LQuantConfig,
    *,
    outdir: Path,
    prefix: str,
    label: str,
    base_model=None,
    kl_eval_prompts: Optional[List[str]] = None,
    kl_model=None,
) -> Dict[str, Dict]:
    """Run and persist the held-out eval suite with an optional filename prefix."""
    print(f"\nEvaluating {label}held-out fit/deploy forecast on test set ...", flush=True)
    test_results, test_trace_rows = evaluate_on_test(model, tokenizer, train_recs, eval_recs, test_recs, cfg)
    with open(outdir / f"{prefix}test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    trace_artifacts = _save_lquant_test_trace_archive(
        outdir,
        prefix=prefix,
        target_traces=test_trace_rows,
        cfg=cfg,
    )

    print(f"\nEvaluating {label}whole-pool test diagnostic ...", flush=True)
    test_pool_diagnostic = evaluate_test_pool_diagnostic(model, tokenizer, test_recs, cfg)
    with open(outdir / f"{prefix}test_pool_diagnostic.json", "w") as f:
        json.dump(test_pool_diagnostic, f, indent=2)

    kl_eval_results = {}
    if base_model is not None and kl_eval_prompts:
        print(f"\nEvaluating {label}KL on held-out datasets ...", flush=True)
        kl_eval_results = evaluate_heldout_kl(
            kl_model if kl_model is not None else model,
            base_model,
            tokenizer,
            kl_eval_prompts,
            cfg,
        )
        print(f"  KL (single-token, eval): {kl_eval_results['single_token']:.6f}", flush=True)
        with open(outdir / f"{prefix}kl_eval_results.json", "w") as f:
            json.dump(kl_eval_results, f, indent=2)

    return {
        "test_results": test_results,
        "test_pool_diagnostic": test_pool_diagnostic,
        "kl_eval_results": kl_eval_results,
        "test_trace_artifacts": trace_artifacts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train L_quant: downstream forecast loss for log-linear tails"
    )

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="")

    # LoRA
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names the LoRA adapts (add gate_proj,up_proj,down_proj for the MLPs)",
    )

    # Task kind
    parser.add_argument(
        "--task-kind", default="password", choices=["password", "token_union"],
        help="password: single-token password event, generated prompts. "
             "token_union: WildChat prompts, event = union of first-response token "
             "ids from the bank metadata (requires --prompt-bank-root).",
    )
    parser.add_argument(
        "--score-max-len", type=int, default=0,
        help="LEFT-truncate tokenized prompts to this many tokens at scoring time "
             "(0 = no truncation). Use 8192 for long WildChat prompts.",
    )
    parser.add_argument(
        "--score-token-budget", type=int, default=0,
        help="Length-sorted micro-batching under B*L_max<=budget (0 = legacy "
             "fixed-size batches). Keeps 8k-token batches from OOMing.",
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=False,
        help="Enable backbone activation checkpointing (needed for 8k-context "
             "token_union training; result-preserving memory/compute trade).",
    )

    # Password game
    parser.add_argument("--password-slots", type=int, default=19)
    parser.add_argument("--password-policy", default="non_disclosure")
    parser.add_argument(
        "--train-target-count",
        type=int,
        default=0,
        help="Number of targets to use for training. 0 keeps legacy all-target behavior unless heldout-target-count is set.",
    )
    parser.add_argument(
        "--heldout-target-count",
        type=int,
        default=0,
        help="Number of targets to reserve for held-out evaluation. 0 infers the remainder if train-target-count is set.",
    )

    # Prompt pool
    parser.add_argument("--cheap-prompts-per-family-per-target", type=int, default=64)
    parser.add_argument(
        "--cheap-prompts-per-family-map",
        nargs="*",
        default=None,
        help="Optional per-family prompt counts of the form family=count. "
             "Overrides --cheap-prompts-per-family-per-target for named families.",
    )
    parser.add_argument("--train-families", nargs="*", default=None)
    parser.add_argument("--test-families", nargs="*", default=None)
    parser.add_argument(
        "--fit-families", nargs="*", default=None,
        help="Fit/deploy family shift: draw every slot's fit sample only from "
             "these families (train and held-out slots alike). Requires "
             "--deploy-families, disjoint from this list, and "
             "--bootstrap-permutations 0.",
    )
    parser.add_argument(
        "--deploy-families", nargs="*", default=None,
        help="Fit/deploy family shift: restrict every slot's deploy set to "
             "these families. See --fit-families.",
    )
    parser.add_argument(
        "--shift-fit-size", type=int, default=0,
        help="Family shift: fit prompts per slot (required >= 1 when the "
             "shift is enabled; cell sizes are part of the protocol).",
    )
    parser.add_argument(
        "--shift-deploy-size", type=int, default=0,
        help="Family shift: deploy prompts per slot, thinned across deploy "
             "families by largest remainder. 0 = use every deploy-family record.",
    )
    parser.add_argument("--prompt-bank-root", default="")
    parser.add_argument("--prompt-bank-seed", type=int, default=-1)

    # GCG
    parser.add_argument(
        "--gcg",
        dest="gcg_enabled",
        action="store_true",
        help="Deprecated legacy online prompt-generation knob. GCG families are no longer auto-routed by strength; include gcg_steps* explicitly in train-families/test-families.",
    )
    parser.add_argument("--gcg-steps-spectrum", nargs="*", type=int, default=None)
    parser.add_argument("--gcg-prompts-per-level-per-target", type=int, default=16)
    parser.add_argument("--gcg-prompt-len", type=int, default=24)
    parser.add_argument("--gcg-top-k", type=int, default=8)
    parser.add_argument("--gcg-random-restarts", type=int, default=2)
    parser.add_argument(
        "--gcg-train-max-steps",
        type=int,
        default=20,
        help="Deprecated no-op in the training consumer; retained only for compatibility with legacy configs.",
    )
    parser.add_argument("--gcg-decay-alpha", type=float, default=2.0)

    # Split
    parser.add_argument("--split-mode", default="disjoint",
                        choices=["disjoint", "nested", "resample"])
    parser.add_argument("--fit-fraction", type=float, default=0.3)
    parser.add_argument(
        "--shared-family-test-fraction",
        type=float,
        default=0.5,
        help="When a family appears in both train-families and test-families, reserve this fraction for test.",
    )

    # OLS
    parser.add_argument("--ols-top-k", type=int, default=10)
    parser.add_argument("--plotting-position", default="gringorten",
                        choices=["gringorten", "weibull"])
    parser.add_argument(
        "--extrapolated-weighting",
        default="rank_uniform",
        choices=["rank_uniform", "worst_rank", "deploy_log_uniform", "deploy_uniform"],
        help=(
            "How to weight extrapolated deploy ranks in ols_primal_extrapolated. "
            "worst_rank puts all weight on rank 1, matching the reported "
            "held-out metric exactly."
        ),
    )
    parser.add_argument(
        "--extrapolated-loss-space",
        default="log_prob",
        choices=["log_prob", "score"],
        help="Space for ols_primal_extrapolated loss: raw log-probability (matches eval log error) or transformed score space.",
    )

    # Loss mode
    parser.add_argument("--loss-mode", default="ols_coeff",
                        choices=["ols_coeff", "ols_primal", "ols_primal_weighted", "ols_primal_extrapolated", "extrapolate"],
                        help="ols_coeff: match (a,b) directly; ols_primal*: eval-set MSE objectives; ols_primal_extrapolated: score only eval points beyond the fit-max regime; extrapolate: legacy N-sampling")

    # Deployment scale (only used in extrapolate mode)
    parser.add_argument("--log-n-min", type=float, default=3.0)
    parser.add_argument("--log-n-max", type=float, default=6.0)
    parser.add_argument("--n-deploy-samples", type=int, default=1,
                        help="Number of deployment scales to sample per step")

    # Fit size
    parser.add_argument("--fit-size-min", type=int, default=20)
    parser.add_argument("--fit-size-max", type=int, default=0)
    parser.add_argument("--fit-subsample-min", type=int, default=0,
                        help="Min fit subsample size; each step draws randint(min, pool_size) prompts (0=disabled)")

    # Gradient flow
    parser.add_argument("--grad-mode", default="fit",
                        choices=["fit", "eval", "both"])
    parser.add_argument("--improving-only-grad", action="store_true", default=False)
    parser.add_argument(
        "--iog-scope",
        choices=["both", "deploy_only", "fit_only"],
        default="both",
        help=(
            "Scope of the improving-only mask. 'both' (default): mask both "
            "deploy- and fit-side gradients that would worsen per-prompt "
            "log_prob. 'deploy_only': only mask deploy-side, leave fit free "
            "(intended to free the OLS line under IOG). 'fit_only': symmetric "
            "ablation. Ignored when --improving-only-grad is not set."
        ),
    )
    parser.add_argument(
        "--iog-worsening-coef", type=float, default=0.0,
        help=(
            "Partial IOG: scale worsening gradients by this coefficient instead "
            "of zeroing them. 0.0 (default) is the canonical hard mask; 1.0 "
            "disables masking entirely. Intended for the 4B/8B case where the "
            "forecast tracks the suppressed tail too slowly. Ignored when "
            "--improving-only-grad is not set."
        ),
    )
    parser.add_argument(
        "--best-ckpt-min-step", type=int, default=0,
        help=(
            "Earliest step eligible for the `best/` checkpoint, which is chosen "
            "by train-side squared forecast error. 0 (default) lets step 0 win, "
            "which reports that training never improved the forecast."
        ),
    )
    parser.add_argument(
        "--iog-penalty-coef", type=float, default=0.0,
        help=(
            "Squared-hinge penalty coefficient on per-prompt log_prob growing "
            "past pretrained baseline. 0 disables. Penalty added to meta-loss; "
            "scoped to top-k extrapolated prompts (same scope as IOG mask). "
            "Recommended starting point for LM: 0.05. RL value scale differs "
            "(regret in [0,1] vs log_prob in [-25,0]) and needs a much larger "
            "coef when ported to train_rlquant.py."
        ),
    )
    parser.add_argument(
        "--calibration-aware", action="store_true", default=False,
        help=(
            "Train the CALIBRATED forecast: a stop-gradient affine map (fit "
            "leave-one-target-out over a persistent buffer of worst-rank "
            "(forecast, actual) pairs) is applied before the squared error, "
            "mirroring the deployed post-hoc calibrator. Requires "
            "loss_mode=ols_primal_extrapolated."
        ),
    )
    parser.add_argument(
        "--calibration-mode", choices=["shift", "shift_scale"], default="shift_scale",
        help="Affine form mirrored in training: shift only (beta) or shift+scale (alpha,beta).",
    )
    parser.add_argument(
        "--calibration-min-points", type=int, default=5,
        help="Donor targets required before calibrating (identity forecast until then).",
    )
    parser.add_argument(
        "--calibration-ridge-alpha", type=float, default=0.05,
        help="Slope shrink toward 1 (dimensionless): alpha=(alpha_ols+r)/(1+r). Keeps alpha>0.",
    )
    parser.add_argument(
        "--calibration-ema", type=float, default=0.0,
        help="Per-target EMA decay for buffer contributions (0 = exact replacement).",
    )
    parser.add_argument(
        "--bootstrap-permutations", type=int, default=0,
        help=(
            "If >=1 and loss_mode=ols_primal_extrapolated, re-partition "
            "fit_subset ∪ eval_subset every step (with this many permutations "
            "per step) instead of using the fixed per-target partition. "
            "Breaks per-manifest asymmetric overfit. N=1 is sufficient for "
            "asymmetry-breaking; N>1 adds variance reduction. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--union-cache-top-c", type=int, default=0,
        help=(
            "Partition-invariant union top-C cache for the bootstrap path. "
            "Caches top-C indices of fit_subset∪eval_subset; reuses for "
            "refresh_interval steps. Set to >= ols_top_b + n_fit (typically "
            "296 at canonical scale) for deterministic deploy-top-B coverage. "
            "0 = disabled (full union scored every step)."
        ),
    )
    parser.add_argument(
        "--union-cache-refresh-interval", type=int, default=5,
        help="Steps between full union rescans when union-cache-top-c > 0.",
    )
    parser.add_argument(
        "--iog-fit-mask", choices=["closed_form", "exact"], default="closed_form",
        help=(
            "Fit-side improving-only mask: 'closed_form' uses the analytic "
            "score-space sign (the original implementation; leaks ~6%% of "
            "worsening fit gradients when the loss is in log-prob space), "
            "'exact' differentiates the configured loss and blocks correctly."
        ),
    )
    parser.add_argument(
        "--anchor-gate-margin", type=float, default=0.0,
        help=(
            "Anchor-gated worsening mask (nats). Each prompt's worsening "
            "gradient is scaled by clip((base_model_log_prob - current)/margin, "
            "0, 1), so a prompt at its starting score stays hard-blocked and a "
            "prompt already suppressed by this many nats is free to move back "
            "up. 0 = disabled (canonical hard IOG mask). Requires "
            "--improving-only-grad and the bootstrap path."
        ),
    )
    parser.add_argument(
        "--anchor-penalty-coef", type=float, default=0.0,
        help=(
            "Squared-hinge penalty on prompts above their base-model score, added "
            "to the meta-loss (an alternative to the anchor gate: a "
            "force rather than a lifted constraint). 0 = disabled."
        ),
    )
    parser.add_argument(
        "--loss-scale-norm-m", type=int, default=0,
        help=(
            "Divide the forecast loss by the variance of the union's top-m "
            "scores, normalized to 1 at step 0. Makes the objective invariant to "
            "how far the tail reaches below the bulk, which the unnormalized "
            "loss can always reduce by pulling the tail in. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--gap-floor-m", type=int, default=0,
        help=(
            "One-sided tail-spread constraint: every consecutive gap among the "
            "union's top-m order statistics (nats) must stay at or above the "
            "corresponding gap of the population OLS line through the anchor "
            "pool's top-m. Makes the corrected tail the smallest feasible tail. "
            "0 = disabled; needs --gap-floor-lambda > 0 and the anchor guards."
        ),
    )
    parser.add_argument(
        "--gap-floor-lambda", type=float, default=0.0,
        help=(
            "Weight of the l1 hinge enforcing the gap floors. An exact penalty: "
            "any value above the largest dual multiplier yields the constrained "
            "optimum itself; certify post hoc via gap_hinge_mean == 0."
        ),
    )
    parser.add_argument(
        "--gap-floor-block", type=int, default=1,
        help=(
            "Window size B for the floors: constrain every B-rank span "
            "v_(r) - v_(r+B) instead of every consecutive gap. B in the tens "
            "lifts the constraint's features above the scoring channel's "
            "pass-to-pass noise, which dominated the B=1 arms. 1 = per-gap."
        ),
    )
    parser.add_argument(
        "--gap-floor-level-gated", action="store_true",
        help=(
            "Open violated spans only by pushing the lower point down; the "
            "upper side is detached. Keeps every constraint force in the "
            "suppression direction, like the improving-only mask."
        ),
    )
    parser.add_argument(
        "--gap-floor-slack", type=float, default=GAP_FLOOR_EPS,
        help=(
            "Slack (nats) subtracted from every floor before the hinge. Set "
            "from the measured pass noise (~2-3x noise_sd*sqrt(2)) for window "
            "arms; the tiny default only covers fp32 rounding."
        ),
    )
    parser.add_argument(
        "--sigma-floor-lambda", type=float, default=0.0,
        help=(
            "Weight of the one-sided floor on the realized involved set's std "
            "(fit top-k + extrapolated eval ranks), referenced to the LINE "
            "through the anchor pool and averaged over the K permutations "
            "before the hinge. Certify post hoc via sigma_hinge_mean == 0."
        ),
    )
    parser.add_argument(
        "--sigma-floor-margin", type=float, default=0.0,
        help=(
            "Nats subtracted from the sigma floor's reference to absorb the "
            "realized set's partition noise (~2 sd / sqrt(K))."
        ),
    )
    parser.add_argument(
        "--sigma-floor-symmetric", action="store_true",
        help=(
            "Two-sided sigma constraint: penalize |sigma_bar - reference| "
            "beyond the margin, pinning the involved set's spread to the "
            "anchor line's from both sides instead of only flooring it."
        ),
    )
    parser.add_argument(
        "--cert-distill-coef", type=float, default=0.0,
        help=(
            "Weight of the squared pull of the anchor's top ranks onto the "
            "certified tail (needs the sigma floor; 0 = off)."
        ),
    )
    parser.add_argument(
        "--cert-distill-ranks", type=int, default=300,
        help="Anchor ranks the certified-tail targets cover (default 300).",
    )
    parser.add_argument(
        "--cert-distill-space", default="log_prob", choices=["log_prob", "logit"],
        help="Space of the certified-tail pull: log-prob nats or event log-odds.",
    )
    parser.add_argument(
        "--forecast-loss-coef", type=float, default=1.0,
        help="Weight on the forecast loss (0 + --cert-distill-coef = pure distillation).",
    )
    parser.add_argument(
        "--cert-distill-shared-pass", action="store_true", default=False,
        help=(
            "Skip channel prompts in the bootstrap grad pass and read the anchor "
            "penalty (and a nonzero forecast loss) from the constraint channel's "
            "pass wherever it covers a prompt (halves the grad-scored set at k50)."
        ),
    )
    parser.add_argument(
        "--sigma-ref-m", type=int, default=250,
        help=(
            "Top anchor ranks the sigma reference straightens; must cover the "
            "deepest rank the involved set reads (default 250, enough for k=10)."
        ),
    )
    parser.add_argument(
        "--target-holdout-full-pools", action="store_true",
        help=(
            "Under target-level hold-out, skip the within-target train/test "
            "halving: train targets train on all their records and held-out "
            "targets are evaluated on all of theirs."
        ),
    )
    parser.add_argument(
        "--heldout-from-end", action="store_true",
        help=(
            "Take the held-out targets from the end of the seeded target "
            "shuffle, so the held-out set is independent of the train count."
        ),
    )
    parser.add_argument(
        "--partition-base-targets", type=int, default=0,
        help=(
            "Partition only the first N bank targets (the same held-out pools "
            "as a bank of just those N) and train on every later target too. "
            "0 = partition all targets."
        ),
    )
    parser.add_argument(
        "--dynamics-dump-dir", default="",
        help=(
            "Write per-step score-space dumps (union log_probs, exact dL/dv "
            "over the union, partitions, per-permutation stats) here for the "
            "learning-dynamics analysis. Diagnostic only; does not change "
            "training and does not block resume."
        ),
    )
    parser.add_argument(
        "--dynamics-dump-every", type=int, default=1,
        help="Dump every N steps (default 1).",
    )

    # Training
    parser.add_argument(
        "--optimizer", choices=["adamw", "sgd"], default="adamw",
        help=(
            "adamw is the canonical recipe. sgd is a mechanism probe: it removes "
            "Adam's per-coordinate normalization, which is the remaining "
            "candidate explanation for the selectivity a real trajectory shows "
            "and the response kernel alone does not. Comparable only at matched "
            "parameter-step size, so an sgd arm needs a much larger --lr."
        ),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--topk-refresh-interval",
        type=int,
        default=10,
        help="Deprecated compatibility flag; fit top-k is recomputed each step.",
    )
    parser.add_argument("--eval-cache-top-b", type=int, default=0,
                        help="Cache top-B eval prompts per target; 0=disabled")
    parser.add_argument("--eval-cache-refresh-interval", type=int, default=10,
                        help="Refresh the eval cache every N steps")
    parser.add_argument(
        "--baseline-eval-cache-dir", type=str, default="",
        help=(
            "Directory for the disk-backed baseline-eval cache. When set, "
            "baseline test eval is hash-keyed on cache-relevant config "
            "fields and reused across runs. \"\" = disabled."
        ),
    )
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # KL regularization
    parser.add_argument("--kl-coef", type=float, default=0.0,
                        help="Weight on KL regularization term (0=disabled)")
    parser.add_argument("--kl-trunc-len", type=int, default=8,
                        help="Tokens to sample for truncated KL estimate")
    parser.add_argument("--kl-trunc-every", type=int, default=5,
                        help="Compute truncated KL every N steps")
    parser.add_argument("--kl-batch-size", type=int, default=16,
                        help="Prompts per KL batch")
    parser.add_argument("--kl-train-datasets", nargs="*", default=["password"],
                        help="Datasets for training KL: 'password' or paths to arrow dirs")
    parser.add_argument("--kl-eval-datasets", nargs="*", default=None,
                        help="Datasets for eval-time KL: 'password', arrow dirs, or a "
                             ".jsonl of {\"p\": text} (the token-union KL sidecar)")
    parser.add_argument("--kl-tokenize-max-len", type=int, default=512,
                        help="Max tokens for KL prompts (use 1024 for long WildChat KL prompts)")
    parser.add_argument("--kl-truncation-side", default="right", choices=["right", "left"],
                        help="Truncation side for KL prompts; 'left' keeps the generation marker "
                             "of long prompts (token_union runs use left).")

    # Logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--adapter-snapshot-every", type=int, default=0,
                        help="also keep the adapter every N steps under snapshots/step_NNNN/ "
                             "(0 = off), for post-hoc held-out scoring via --eval-adapter-from.")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--eval-adapter-from", default="",
                        help="save_pretrained adapter dir (e.g. a run's best/); load it and run "
                             "only the final held-out eval, skipping training.")
    parser.add_argument("--token-union-blocklist-prompt", action="store_true",
                        help="token_union only: put the union word list in a system prompt, "
                             "making the forbidden set context-computable (password-like).")
    parser.add_argument("--targets-per-step", type=int, default=0,
                        help="meta-task batch size: 0 = all train targets per optimizer step "
                             "(canonical); b>0 samples b targets per step.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the untouched-model held-out eval suite, then exit.")

    args = parser.parse_args(argv)
    family_prompt_count_map = _parse_family_prompt_count_map(args.cheap_prompts_per_family_map)

    cfg = LQuantConfig(
        model_id=args.model,
        device=args.device,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(m for m in args.lora_target_modules.split(",") if m),
        task_kind=args.task_kind,
        score_max_len=args.score_max_len,
        score_token_budget=args.score_token_budget,
        gradient_checkpointing=args.gradient_checkpointing,
        password_slots=args.password_slots,
        password_policy=args.password_policy,
        train_target_count=args.train_target_count,
        heldout_target_count=args.heldout_target_count,
        cheap_prompts_per_family_per_target=args.cheap_prompts_per_family_per_target,
        cheap_prompts_per_family_map=family_prompt_count_map,
        train_families=args.train_families or list(DEFAULT_TRAIN_FAMILIES),
        test_families=args.test_families or list(DEFAULT_TEST_FAMILIES),
        fit_families=args.fit_families or [],
        deploy_families=args.deploy_families or [],
        shift_fit_size=args.shift_fit_size,
        shift_deploy_size=args.shift_deploy_size,
        prompt_bank_root=args.prompt_bank_root,
        prompt_bank_seed=args.prompt_bank_seed,
        gcg_enabled=args.gcg_enabled,
        gcg_steps_spectrum=args.gcg_steps_spectrum or [5, 10, 20, 40, 80],
        gcg_prompts_per_level_per_target=args.gcg_prompts_per_level_per_target,
        gcg_prompt_len=args.gcg_prompt_len,
        gcg_top_k=args.gcg_top_k,
        gcg_random_restarts=args.gcg_random_restarts,
        gcg_train_max_steps=args.gcg_train_max_steps,
        gcg_decay_alpha=args.gcg_decay_alpha,
        split_mode=args.split_mode,
        fit_fraction=args.fit_fraction,
        shared_family_test_fraction=args.shared_family_test_fraction,
        target_holdout_full_pools=args.target_holdout_full_pools,
        heldout_from_end=args.heldout_from_end,
        partition_base_targets=args.partition_base_targets,
        ols_top_k=args.ols_top_k,
        plotting_position=args.plotting_position,
        extrapolated_weighting=args.extrapolated_weighting,
        extrapolated_loss_space=args.extrapolated_loss_space,
        log_n_min=args.log_n_min,
        log_n_max=args.log_n_max,
        n_deploy_samples=args.n_deploy_samples,
        fit_size_min=args.fit_size_min,
        fit_size_max=args.fit_size_max,
        fit_subsample_min=args.fit_subsample_min,
        loss_mode=args.loss_mode,
        grad_mode=args.grad_mode,
        improving_only_grad=args.improving_only_grad,
        iog_scope=args.iog_scope,
        iog_worsening_coef=args.iog_worsening_coef,
        best_ckpt_min_step=args.best_ckpt_min_step,
        iog_penalty_coef=args.iog_penalty_coef,
        calibration_aware=args.calibration_aware,
        calibration_mode=args.calibration_mode,
        calibration_min_points=args.calibration_min_points,
        calibration_ridge_alpha=args.calibration_ridge_alpha,
        calibration_ema=args.calibration_ema,
        bootstrap_permutations=args.bootstrap_permutations,
        union_cache_top_c=args.union_cache_top_c,
        union_cache_refresh_interval=args.union_cache_refresh_interval,
        iog_fit_mask=args.iog_fit_mask,
        anchor_gate_margin=args.anchor_gate_margin,
        anchor_penalty_coef=args.anchor_penalty_coef,
        loss_scale_norm_m=args.loss_scale_norm_m,
        gap_floor_m=args.gap_floor_m,
        gap_floor_lambda=args.gap_floor_lambda,
        gap_floor_block=args.gap_floor_block,
        gap_floor_slack=args.gap_floor_slack,
        gap_floor_level_gated=args.gap_floor_level_gated,
        sigma_floor_lambda=args.sigma_floor_lambda,
        sigma_floor_margin=args.sigma_floor_margin,
        sigma_floor_symmetric=args.sigma_floor_symmetric,
        sigma_ref_m=args.sigma_ref_m,
        cert_distill_coef=args.cert_distill_coef,
        cert_distill_ranks=args.cert_distill_ranks,
        cert_distill_space=args.cert_distill_space,
        forecast_loss_coef=args.forecast_loss_coef,
        cert_distill_shared_pass=args.cert_distill_shared_pass,
        dynamics_dump_dir=args.dynamics_dump_dir,
        dynamics_dump_every=args.dynamics_dump_every,
        optimizer=args.optimizer,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        topk_refresh_interval=args.topk_refresh_interval,
        eval_cache_top_b=args.eval_cache_top_b,
        eval_cache_refresh_interval=args.eval_cache_refresh_interval,
        baseline_eval_cache_dir=args.baseline_eval_cache_dir,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        adapter_snapshot_every=args.adapter_snapshot_every,
        resume_checkpoint=args.resume_checkpoint,
        eval_adapter_from=args.eval_adapter_from,
        token_union_blocklist_prompt=args.token_union_blocklist_prompt,
        targets_per_step=args.targets_per_step,
        kl_coef=args.kl_coef,
        kl_trunc_len=args.kl_trunc_len,
        kl_trunc_every=args.kl_trunc_every,
        kl_batch_size=args.kl_batch_size,
        kl_train_datasets=args.kl_train_datasets or ["password"],
        kl_eval_datasets=args.kl_eval_datasets or [],
        kl_tokenize_max_len=args.kl_tokenize_max_len,
        kl_truncation_side=args.kl_truncation_side,
        outdir=args.outdir,
        disable_thinking=args.disable_thinking,
        baseline_only=args.baseline_only,
    )

    if cfg.gcg_enabled:
        print(
            "WARNING: --gcg is deprecated in train_lquant. GCG families are no longer "
            "auto-split by step strength; use explicit train_families/test_families "
            "or a prompt bank instead.",
            flush=True,
        )

    train_lquant(cfg)


if __name__ == "__main__":
    main()
