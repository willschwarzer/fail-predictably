"""Model-free probe of what the forecast-error objective asks of each prompt.

The training loss depends on the model only through one vector per target: the
union pool's per-prompt scores. Everything else the step does -- re-partitioning
the pool, selecting the fit top-k and the extrapolated deploy ranks, fitting the
OLS tail line, applying the improving-only gradient mask, averaging over
bootstrap permutations -- is a deterministic function of that vector and the
config. So the objective's whole behaviour can be replayed on a CPU from a saved
score vector, with no model in the loop.

That separation is the point. A training step's outcome is the composition of two
things: what the objective asks for (this module) and what the model can deliver
(the response kernel, which this module does not measure).
Conflating them is why "the tail collapses" has been hard to attribute.

Every quantity here is computed by calling the same ``ols_utils`` functions
training calls, so the probe cannot drift from the trainer. In particular the
per-prompt gradient is obtained by autograd on a leaf copy of the score vector,
which is exactly the quantity that reaches the model's parameters through the
chain rule.

Vocabulary used throughout:
  * ``v`` -- a prompt's score. Here always a log-probability (nats, negative).
    Higher ``v`` means a more failure-prone prompt.
  * ``x = -log(-v)`` -- the transformed score the OLS line is fit in
    (``score_mode="gumbel_log_prob"``). Increasing in ``v``.
  * *selected* -- the prompt is in some permutation's fit top-k or extrapolated
    deploy ranks, so the loss sees it.
  * *active* -- the prompt is selected AND the improving-only mask let its own
    gradient through, so ``dL/dv_i != 0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple

import math

import numpy as np
import torch

from .ols_utils import (
    count_extrapolated_eval_points,
    differentiable_ols_extrapolated_primal_loss_bootstrap,
    plan_bootstrap_permutations,
    select_top_k_indices,
    fit_ols_tail_from_scores,
)

LN10 = math.log(10.0)

# Slack subtracted from every gap floor. The hinge's subgradient is full-size
# however small the violation, so float32 noise in the scores (~1e-5 nats at a
# level of 60) would otherwise fire spurious O(lambda) gradients on a pool that
# sits exactly on the line. 1e-4 nats is 1-2% of the smallest real floor.
GAP_FLOOR_EPS = 1e-4


def population_line_gaps(
    v: np.ndarray, m: int = 250, plotting_offset: float = 1.0
) -> np.ndarray:
    """Consecutive v-space gaps of the population OLS line through v's top-m.

    These are the floors for the one-sided spread constraint. The pool whose
    top-m order statistics sit exactly on this line is the corrected tail: same
    level and reach as ``v``, zero shape mismatch against the functional form
    the objective's extrapolation assumes. Requiring every consecutive gap of
    the trained tail to stay at or above the line's makes that corrected tail
    the smallest feasible tail, while leaving any wider tail feasible. ``v``
    should be the FROZEN base-model pool, so the target does not move with
    training. Returns m-1 positive gaps, ordered from the top rank down.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    m = max(4, min(int(m), n))
    top = np.sort(v)[::-1][:m]
    x = -np.log(np.maximum(-top, 1e-30))
    y = np.log(np.arange(1, m + 1) / (n + plotting_offset))
    if np.ptp(x) <= 0:
        return np.zeros(m - 1)
    slope, intercept = np.polyfit(x, y, 1)
    if abs(slope) < 1e-12:
        return np.zeros(m - 1)
    x_lin = (y - intercept) / slope
    v_lin = -np.exp(-x_lin)
    return v_lin[:-1] - v_lin[1:]


def linearize(
    v: np.ndarray, m: int = 250, plotting_offset: float = 1.0
) -> np.ndarray:
    """Replace the top-m order statistics with the exact OLS line through them.

    The result has the same level and very nearly the same scale as ``v``, but
    zero shape mismatch: its tail is perfectly straight in ``(x, log p)``, which
    is the functional form the objective assumes. Any loss remaining on this
    vector is sampling error, not model error.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    m = max(4, min(int(m), n))
    order = np.argsort(-v, kind="stable")
    top = v[order[:m]]
    x = -np.log(np.maximum(-top, 1e-30))
    # Global (not subsample) plotting positions: this is the population line.
    y = np.log(np.arange(1, m + 1) / (n + plotting_offset))
    b = np.polyfit(x, y, 1)
    slope, intercept = float(b[0]), float(b[1])
    if abs(slope) < 1e-12:
        return v.copy()
    x_lin = (y - intercept) / slope
    v_lin = -np.exp(-x_lin)
    out = v.copy()
    out[order[:m]] = v_lin
    # The observed tail is curved, so a straight line through it lands below the
    # bulk at the bottom of the range. Clipping there would put a kink in the
    # very reference that is supposed to have none, so displace the bulk instead:
    # ranks past m never enter the loss, and only their ordering matters, which
    # a rigid shift preserves.
    if m < n:
        gap = float(v_lin[-1] - v[order[m]])
        if gap <= 0:
            out[order[m:]] = v[order[m:]] + gap - 1e-6
    return np.minimum(out, -1e-9)


def _involved_sigma_reference(
    anchor: np.ndarray,
    *,
    n_fit: int,
    fit_top_k: int,
    extrap_k: int,
    m: int = 250,
    draws: int = 400,
    seed: int = 1234,
) -> float:
    """E[std of the realized involved set] for the LINE through the anchor pool.

    The involved set is what one step's loss actually reads: the fit
    partition's top-k plus the eval partition's extrapolated top ranks. Its
    std under random partitions is the sigma floor's reference: referencing
    the observed pool instead mis-pins the optimum's scale by however much
    sigma the observed curvature carries (measured +47% canonical / -16%
    enlarged), which lets the loss buy a ceiling-ward contraction of the
    straight tail (canonical) or excludes the corrected tail outright
    (enlarged). The line's own sigma puts the correction target exactly on
    the constraint boundary.
    """
    v = linearize(np.asarray(anchor, dtype=np.float64), m=m)
    n = v.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(draws)
    for d in range(draws):
        fit = rng.choice(n, size=n_fit, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[fit] = True
        ev = np.flatnonzero(~mask)
        pts = np.concatenate([
            np.sort(v[fit])[::-1][:fit_top_k],
            np.sort(v[ev])[::-1][:extrap_k],
        ])
        vals[d] = pts.std(ddof=0)
    return float(vals.mean())


_SIGMA_REF_CACHE: Dict[tuple, float] = {}


def involved_sigma_reference(anchor: np.ndarray, **kw) -> float:
    """Cached ``_involved_sigma_reference`` (deterministic given inputs)."""
    a = np.ascontiguousarray(np.asarray(anchor, dtype=np.float64))
    key = (hash(a.tobytes()), tuple(sorted(kw.items())))
    if key not in _SIGMA_REF_CACHE:
        _SIGMA_REF_CACHE[key] = _involved_sigma_reference(a, **kw)
    return _SIGMA_REF_CACHE[key]


def certified_tail_targets(
    anchor: np.ndarray,
    *,
    n_fit: int,
    fit_top_k: int,
    extrap_k: int,
    sigma_ref: float,
    sigma_margin: float,
    two_sided: bool,
    max_rank: int,
    splits: int = 200,
    seed: int = 4321,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Per-prompt targets for the anchor pool's certified tail.

    The certified tail is the straight ``(x, log p)`` line
    over the top ``max_rank`` anchor ranks that (i) lies at or below every anchor
    order statistic, so assigning its values in anchor rank order raises no
    prompt; (ii) keeps the involved-set sigma within ``sigma_margin`` of
    ``sigma_ref`` (from below only, or from both sides if ``two_sided``); and
    (iii) minimizes the split-averaged abs log10 error of the trainer's OLS
    forecast over random fit/eval partitions of the whole pool. For slope ``s``
    the shallowest safe line has intercept ``min_r(y_r + s x_r)``; deepening it
    multiplies every value, the error and sigma by one common factor, so the
    search is one-dimensional in ``s``.

    Returns ``(idx, target, info)``: the anchor's top-``max_rank`` prompt indices
    in rank order, their target log-probs (each <= its anchor value), and the
    floor diagnostics.
    """
    v = np.asarray(anchor, dtype=np.float64)
    n = v.shape[0]
    order = np.argsort(-v, kind="stable")
    L = int(min(n, max_rank))
    vtop = v[order[:L]]
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(n) for _ in range(splits)])  # rank ids
    fr = np.sort(perms[:, :n_fit], axis=1)[:, :fit_top_k]
    dr = np.sort(perms[:, n_fit:], axis=1)[:, :extrap_k]
    keep = (fr.max(axis=1) < L) & (dr.max(axis=1) < L)
    fr, dr = fr[keep], dr[keep]
    n_eval = n - n_fit
    y = np.log(np.arange(1, L + 1) / (n + 1.0))
    xb = -np.log(np.maximum(-vtop, 1e-30))
    y_fit = np.log(np.arange(1, fit_top_k + 1) / (n_fit + 1.0))
    log_inv_n = math.log(1.0 / n_eval)

    def stats(vals: np.ndarray) -> Tuple[float, float]:
        x = -np.log(np.maximum(-vals[fr], 1e-30))
        xm = x.mean(axis=1, keepdims=True)
        ym = y_fit.mean()
        sxx = ((x - xm) ** 2).sum(axis=1)
        sxy = ((x - xm) * (y_fit - ym)).sum(axis=1)
        ok = sxx > 1e-20
        b = np.where(ok, sxy / np.where(ok, sxx, 1.0), 0.0)
        a = ym - b * xm[:, 0]
        nz = b != 0.0
        xstar = np.where(nz, (log_inv_n - a) / np.where(nz, b, 1.0), x[:, 0])
        with np.errstate(over="ignore"):
            pred = -np.exp(-xstar)
        err = np.abs(pred - vals[dr[:, 0]]).mean() / math.log(10.0)
        inv = np.concatenate([vals[fr], vals[dr]], axis=1)
        return float(err), float(inv.std(axis=1).mean())

    def safe(s: float) -> np.ndarray:
        a = float(np.min(y + s * xb))
        return -np.exp(-(a - y) / s)

    def objective(s: float) -> Tuple[float, float]:
        err, sig = stats(safe(s))
        if two_sided and sig > sigma_ref + sigma_margin:
            return math.inf, 1.0
        c = max(1.0, (sigma_ref - sigma_margin) / max(sig, 1e-12))
        return err * c, c

    b0, _ = np.polyfit(xb, y, 1)
    s0 = -float(b0)
    grid = s0 * np.exp(np.linspace(math.log(0.1), math.log(1000.0), 121))
    vals = [objective(s)[0] for s in grid]
    i = int(np.argmin(vals))
    if not math.isfinite(vals[i]):
        raise RuntimeError("certified_tail_targets: no slope satisfies the sigma band")
    best_val, best_s = vals[i], float(grid[i])
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    for s in np.exp(np.linspace(math.log(lo), math.log(hi), 41)):
        val = objective(float(s))[0]
        if val < best_val:
            best_val, best_s = val, float(s)
    deepen = objective(best_s)[1]
    target = safe(best_s) * deepen
    if not np.all(target <= vtop + 1e-9):
        raise RuntimeError("certified_tail_targets: a target exceeds its anchor value")
    base_err, base_sig = stats(vtop)
    info = {
        "base_abs_log10": base_err, "cert_abs_log10": float(best_val),
        "base_sigma": base_sig, "sigma_ref": float(sigma_ref),
        "slope_ratio": best_s / s0, "deepen": float(deepen),
        "top_anchor_nats": float(vtop[0]), "top_target_nats": float(target[0]),
        "mean_drop_nats": float(np.mean(vtop - target)), "ranks": L,
        "splits_used": int(keep.sum()),
    }
    return order[:L].astype(np.int64), target, info


@dataclass
class ObjectiveSpec:
    """The training objective's configuration, reduced to what the loss uses."""

    n_fit: int
    n_eval: int
    ols_top_k: int = 10
    plotting: str = "weibull"
    score_mode: str = "gumbel_log_prob"
    loss_space: str = "original"
    weighting: str = "deploy_log_uniform"
    grad_mode: str = "both"
    improving_only_grad: bool = True
    iog_scope: str = "both"
    iog_worsening_coef: float = 0.0
    iog_fit_mask: str = "closed_form"
    anchor_margin: float = 0.0
    n_permutations: int = 1
    # >0 divides the loss by the variance of the top-m scores, differentiably,
    # which makes the objective invariant to the tail's scale. The forecast loss
    # is homogeneous of degree two in that scale, so without this the objective
    # can always be reduced by shrinking the tail, whatever its shape. Off by
    # default: every arm in the fleet trained the unnormalized loss.
    scale_normalize_m: int = 0
    # gap_floor_m > 1 with gap_floor_lambda > 0 turns on the one-sided spread
    # constraint: every consecutive v-space gap among the union's top-m order
    # statistics must stay at or above the corresponding gap of the population
    # OLS line through the ANCHOR pool's top-m (``population_line_gaps``). The
    # line is the corrected tail -- same level and reach as the base pool, zero
    # shape mismatch -- and it is the smallest tail the constraint admits, so
    # minimizing the forecast loss over the feasible set lands exactly there:
    # wider tails cost more sampling floor, and every compression move (global
    # rescale, ceiling saturation, plateau-from-below, aggregate redistribution)
    # shrinks some gap below its floor. Implemented as an l1 hinge added to the
    # descended objective: an EXACT penalty, so any lambda above the largest
    # dual multiplier reproduces the constrained optimum, not a lambda-dependent
    # compromise. Needs ``anchor`` to be passed to ``probe_step``.
    gap_floor_m: int = 0
    gap_floor_lambda: float = 0.0
    # Window generalization of the floors: with block B > 1 the constraint is
    # on every B-rank span v_(r) - v_(r+B) instead of every consecutive gap.
    # Spans divide the two noises that killed the per-gap arms: the pass-to-
    # pass scoring noise contributes ~sd*sqrt(2) to a span whose floor grows
    # like B gaps, and order-statistic spacing noise averages down by sqrt(B).
    # The concession is that rearrangement WITHIN a window is unconstrained,
    # so the optimum is a band around the line of width ~one window. B = 1
    # recovers the exact per-gap constraint.
    gap_floor_block: int = 1
    # Slack subtracted from every floor before the hinge. The per-gap arms
    # used GAP_FLOOR_EPS = 1e-4, calibrated to fp32 rounding; the measured
    # scoring channel is ~3 orders of magnitude noisier, so window arms
    # should set this from the measured pass noise (~3*sd*sqrt(2)).
    gap_floor_slack: float = GAP_FLOOR_EPS
    # Level-gated hinge: a violated span opens only by pushing its LOWER
    # point down, never its upper point up (the upper side is detached).
    # The un-gated hinge's upward half was the only strong force in the
    # system not subject to the improving-only mask, and it ratcheted the
    # round-1 window arms into the ceiling, where clipping compresses the
    # very spans the forecast reads. Gating the hinge to the suppression
    # direction mirrors IOG's one-sidedness: nothing in the objective can
    # make a rare behavior more likely, and the anchor gate supplies the
    # restoring force from below.
    gap_floor_level_gated: bool = False
    # One-sided floor on the std of the realized involved set (the fit
    # partition's top-k plus the eval partition's extrapolated ranks),
    # referenced to the LINE through the anchor pool (see
    # ``involved_sigma_reference`` for why the line and not the observed
    # pool), averaged across the step's K permutations BEFORE the hinge so
    # bootstrapping shrinks the partition noise by sqrt(K). ``margin`` is
    # subtracted from the reference.
    sigma_floor_lambda: float = 0.0
    sigma_floor_margin: float = 0.0
    # Two-sided variant: penalize |sigma_bar - reference| beyond the margin,
    # pinning the involved set's spread to the anchor line's from both sides.
    sigma_floor_symmetric: bool = False
    # Top anchor ranks the reference line straightens (mirrors cfg.sigma_ref_m).
    sigma_ref_m: int = 250
    # Probe-only realism knob: iid Gaussian noise (nats, fresh per call) added
    # to the scores the CONSTRAINT terms read, modeling the grad-pass
    # measurement channel that inflated the per-gap hinge ~3x in training.
    # The loss itself stays noise-free so traces remain comparable.
    constraint_noise_sd: float = 0.0

    @property
    def n_union(self) -> int:
        return self.n_fit + self.n_eval

    @property
    def eval_extrap_k(self) -> int:
        k = count_extrapolated_eval_points(
            self.n_fit, self.n_eval, plotting=self.plotting, fit_rank_cutoff=1,
        )
        return max(1, min(k, self.n_eval))

    def label(self) -> str:
        bits = [f"K{self.n_permutations}"]
        if self.improving_only_grad:
            bits.append("iog" if self.iog_scope == "both" else f"iog-{self.iog_scope}")
            if self.iog_fit_mask != "closed_form":
                bits.append(f"fitmask-{self.iog_fit_mask}")
            if self.anchor_margin > 0:
                bits.append(f"anchor{self.anchor_margin:g}")
            elif self.iog_worsening_coef > 0:
                bits.append(f"soft{self.iog_worsening_coef:g}")
        else:
            bits.append("noiog")
        bits.append(self.weighting)
        if self.scale_normalize_m:
            bits.append(f"scalenorm{self.scale_normalize_m}")
        if self.gap_floor_m > 1 and self.gap_floor_lambda > 0:
            tag = f"gapfloor{self.gap_floor_m}@{self.gap_floor_lambda:g}"
            if self.gap_floor_block > 1:
                tag += f"win{self.gap_floor_block}"
            if self.gap_floor_level_gated:
                tag += "lvl"
            bits.append(tag)
        if self.sigma_floor_lambda > 0:
            tag = f"sigfloor@{self.sigma_floor_lambda:g}"
            if self.sigma_floor_symmetric:
                tag += "sym"
            bits.append(tag)
        if self.constraint_noise_sd > 0:
            bits.append(f"cnoise{self.constraint_noise_sd:g}")
        bits.append(f"fit{self.n_fit}/dep{self.n_eval}")
        return " ".join(bits)

    @classmethod
    def from_run_config(cls, cfg: dict, *, n_fit: int, n_eval: int, **overrides):
        """Build a spec from a run's ``config.json`` so probes match a real arm."""
        spec = cls(
            n_fit=int(n_fit),
            n_eval=int(n_eval),
            ols_top_k=int(cfg.get("ols_top_k", 10)),
            plotting=str(cfg.get("plotting_position", "weibull")),
            weighting=str(cfg.get("extrapolated_weighting", "rank_uniform")),
            loss_space=(
                "original" if cfg.get("extrapolated_loss_space") == "log_prob"
                else str(cfg.get("extrapolated_loss_space", "score"))
            ),
            grad_mode=str(cfg.get("grad_mode", "both")),
            improving_only_grad=bool(cfg.get("improving_only_grad", False)),
            iog_scope=str(cfg.get("iog_scope", "both")),
            iog_worsening_coef=float(cfg.get("iog_worsening_coef", 0.0) or 0.0),
            iog_fit_mask=str(cfg.get("iog_fit_mask", "closed_form") or "closed_form"),
            anchor_margin=float(cfg.get("anchor_gate_margin", 0.0) or 0.0),
            n_permutations=max(1, int(cfg.get("bootstrap_permutations", 1) or 1)),
            gap_floor_m=int(cfg.get("gap_floor_m", 0) or 0),
            gap_floor_lambda=float(cfg.get("gap_floor_lambda", 0.0) or 0.0),
            gap_floor_block=int(cfg.get("gap_floor_block", 1) or 1),
            gap_floor_level_gated=bool(cfg.get("gap_floor_level_gated", False)),
            gap_floor_slack=float(cfg.get("gap_floor_slack", GAP_FLOOR_EPS) or GAP_FLOOR_EPS),
            sigma_floor_lambda=float(cfg.get("sigma_floor_lambda", 0.0) or 0.0),
            sigma_floor_margin=float(cfg.get("sigma_floor_margin", 0.0) or 0.0),
            sigma_floor_symmetric=bool(cfg.get("sigma_floor_symmetric", False)),
            sigma_ref_m=int(cfg.get("sigma_ref_m", 250) or 250),
        )
        return replace(spec, **overrides) if overrides else spec


# --------------------------------------------------------------------------
# One step
# --------------------------------------------------------------------------


@dataclass
class StepProbe:
    """Everything one objective step does to one score vector."""

    loss: float
    perm_losses: np.ndarray            # [K]
    score_grad: np.ndarray             # [N] exact dL/dv_i
    fit_partitions: np.ndarray         # [K, n_fit] global union indices
    eval_partitions: np.ndarray        # [K, n_eval]
    fit_selected: np.ndarray           # [K, k] global indices of the fit top-k
    deploy_selected: np.ndarray        # [K, b] global indices of the deploy ranks
    stats: Dict[str, float]            # permutation-mean OLS/mask statistics
    spec: ObjectiveSpec = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def n_union(self) -> int:
        return int(self.score_grad.shape[0])

    def selected_mask(self) -> np.ndarray:
        """[N] bool: prompt entered the loss in at least one permutation."""
        m = np.zeros(self.n_union, dtype=bool)
        m[self.fit_selected.reshape(-1)] = True
        m[self.deploy_selected.reshape(-1)] = True
        return m

    def selection_count(self) -> np.ndarray:
        """[N] int: how many of the K permutations selected each prompt."""
        c = np.zeros(self.n_union, dtype=np.int32)
        np.add.at(c, self.fit_selected.reshape(-1), 1)
        np.add.at(c, self.deploy_selected.reshape(-1), 1)
        return c

    def active_mask(self) -> np.ndarray:
        """[N] bool: prompt's own gradient survived the improving-only mask."""
        return self.score_grad != 0.0


def _selected_per_permutation(
    union_lp: torch.Tensor,
    fit_partitions: torch.Tensor,
    eval_partitions: torch.Tensor,
    spec: ObjectiveSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror the bootstrap loss's per-permutation selection, in global indices."""
    k = max(1, min(int(spec.ols_top_k), int(spec.n_fit)))
    b = spec.eval_extrap_k
    fit_out = np.empty((fit_partitions.shape[0], k), dtype=np.int64)
    dep_out = np.empty((eval_partitions.shape[0], b), dtype=np.int64)
    for p in range(fit_partitions.shape[0]):
        fp = fit_partitions[p]
        ep = eval_partitions[p]
        fit_local = select_top_k_indices(union_lp.index_select(0, fp).detach(), k)
        dep_local = torch.argsort(
            union_lp.index_select(0, ep).detach(), descending=True,
        )[:b]
        fit_out[p] = fp.index_select(0, fit_local).cpu().numpy()
        dep_out[p] = ep.index_select(0, dep_local).cpu().numpy()
    return fit_out, dep_out


def probe_step(
    union_lp: np.ndarray | torch.Tensor,
    spec: ObjectiveSpec,
    *,
    rng_seed: int,
    anchor: np.ndarray | torch.Tensor | None = None,
) -> StepProbe:
    """Replay one objective step on ``union_lp`` and return its full anatomy."""
    v = torch.as_tensor(np.asarray(union_lp, dtype=np.float64), dtype=torch.float32)
    if v.shape[0] != spec.n_union:
        raise ValueError(
            f"union has {v.shape[0]} entries but spec expects "
            f"n_fit+n_eval = {spec.n_union}"
        )
    anch = (
        None if anchor is None
        else torch.as_tensor(np.asarray(anchor, dtype=np.float64), dtype=torch.float32)
    )

    fit_partitions, eval_partitions, _needed = plan_bootstrap_permutations(
        n_union=spec.n_union,
        n_fit=spec.n_fit,
        n_eval=spec.n_eval,
        n_permutations=spec.n_permutations,
        rng_seed=int(rng_seed),
        fit_top_k=spec.ols_top_k,
        eval_extrap_k=spec.eval_extrap_k,
        union_values_detached=v,
        grad_mode=spec.grad_mode,
    )

    leaf = v.clone().requires_grad_(True)
    loss, info = differentiable_ols_extrapolated_primal_loss_bootstrap(
        leaf,
        fit_partitions,
        eval_partitions,
        n_fit=spec.n_fit,
        n_eval=spec.n_eval,
        fit_top_k=spec.ols_top_k,
        plotting=spec.plotting,
        score_mode=spec.score_mode,
        grad_mode=spec.grad_mode,
        extrapolated_weighting=spec.weighting,
        loss_space=spec.loss_space,
        improving_only_grad=spec.improving_only_grad,
        iog_scope=spec.iog_scope,
        iog_worsening_coef=spec.iog_worsening_coef,
        iog_fit_mask=spec.iog_fit_mask,
        union_anchor=anch,
        anchor_margin=spec.anchor_margin,
    )
    # The descended objective and the reported one come apart under scale
    # normalization: ``loss`` below stays the raw forecast error so traces remain
    # comparable across variants, while the gradient is taken of whatever the
    # variant actually descends.
    descended = loss
    sigma2_val = float("nan")
    if spec.scale_normalize_m > 0:
        m = max(2, min(int(spec.scale_normalize_m), spec.n_union))
        top_idx = torch.topk(v, m).indices          # selection is detached, as elsewhere
        sigma2 = leaf.index_select(0, top_idx).var(unbiased=False)
        sigma2_val = float(sigma2.item())
        descended = loss / sigma2.clamp_min(1e-12)

    gap_stats: Dict[str, float] | None = None
    sigma_stats: Dict[str, float] | None = None
    constraint_on = (
        spec.gap_floor_m > 1 and spec.gap_floor_lambda > 0
    ) or spec.sigma_floor_lambda > 0
    # The constraint terms read the scores through the measurement channel:
    # in training that is the second, gradient-enabled scoring pass, whose
    # pass-to-pass noise is far larger than fp32 rounding. The noise shifts
    # where the hinge fires; the gradient still flows to the true scores,
    # exactly as in training.
    v_c, leaf_c = v, leaf
    if constraint_on and spec.constraint_noise_sd > 0:
        gen = torch.Generator().manual_seed(int(rng_seed) ^ 0x5EED)
        eps = torch.randn(v.shape[0], generator=gen) * spec.constraint_noise_sd
        v_c = v + eps
        leaf_c = leaf + eps
    if spec.gap_floor_m > 1 and spec.gap_floor_lambda > 0:
        if anch is None:
            raise ValueError(
                "gap_floor needs the anchor score vector: the floors are the "
                "gaps of the population line through the BASE pool's tail"
            )
        m = min(int(spec.gap_floor_m), spec.n_union)
        g0 = torch.as_tensor(
            population_line_gaps(anch.cpu().numpy(), m=m), dtype=leaf.dtype,
        )
        b = max(1, min(int(spec.gap_floor_block), m - 1))
        # Floors for B-rank spans are the line's own spans: sums of B
        # consecutive line gaps. B = 1 is the per-gap constraint.
        csum = torch.cat([torch.zeros(1, dtype=g0.dtype), torch.cumsum(g0, 0)])
        w0 = csum[b:] - csum[:-b]
        top_idx = torch.topk(v_c, m).indices        # sorted descending by score
        top_vals = leaf_c.index_select(0, top_idx)
        upper = top_vals[:-b].detach() if spec.gap_floor_level_gated else top_vals[:-b]
        spans = upper - top_vals[b:]
        viol = torch.relu(w0 - spec.gap_floor_slack - spans)
        descended = descended + spec.gap_floor_lambda * viol.sum()
        gap_stats = {
            "gap_hinge_sum": float(viol.sum().item()),
            "gap_violation_max": float(viol.max().item()),
            "gap_violation_frac": float((viol > 0).float().mean().item()),
        }
    if spec.sigma_floor_lambda > 0:
        if anch is None:
            raise ValueError(
                "sigma_floor needs the anchor score vector: the floor is the "
                "involved-set sigma of the LINE through the BASE pool's tail"
            )
        sigma_ref = involved_sigma_reference(
            anch.cpu().numpy(),
            n_fit=spec.n_fit,
            fit_top_k=spec.ols_top_k,
            extrap_k=spec.eval_extrap_k,
            m=spec.sigma_ref_m,
        )
        ek = spec.eval_extrap_k
        sigmas = []
        for p in range(fit_partitions.shape[0]):
            fidx = fit_partitions[p]
            eidx = eval_partitions[p]
            fsel = fidx[torch.topk(v_c.index_select(0, fidx), spec.ols_top_k).indices]
            esel = eidx[torch.topk(v_c.index_select(0, eidx), ek).indices]
            pts = leaf_c.index_select(0, torch.cat([fsel, esel]))
            sigmas.append(pts.std(unbiased=False))
        # Average sigma across the K permutations BEFORE the hinge: the
        # realized-set sigma's partition noise then shrinks by sqrt(K),
        # while a mean of K hinges would keep the one-sided bias.
        sigma_bar = torch.stack(sigmas).mean()
        sgap = sigma_bar.new_tensor(sigma_ref) - sigma_bar
        if spec.sigma_floor_symmetric:
            sviol = torch.relu(sgap.abs() - spec.sigma_floor_margin)
        else:
            sviol = torch.relu(sgap - spec.sigma_floor_margin)
        descended = descended + spec.sigma_floor_lambda * sviol
        sigma_stats = {
            "sigma_hinge": float(sviol.item()),
            "sigma_bar": float(sigma_bar.item()),
            "sigma_ref": float(sigma_ref),
        }

    (grad,) = torch.autograd.grad(descended, [leaf], allow_unused=True)
    if grad is None:
        grad = torch.zeros_like(leaf)

    fit_sel, dep_sel = _selected_per_permutation(v, fit_partitions, eval_partitions, spec)
    stats = {
        key: float(val.item())
        for key, val in info.items()
        if torch.is_tensor(val) and val.dim() == 0
    }
    if spec.scale_normalize_m > 0:
        stats["descended_loss"] = float(descended.item())
        stats["scale_norm_sigma2"] = sigma2_val
    if gap_stats is not None:
        stats["descended_loss"] = float(descended.item())
        stats.update(gap_stats)
    if sigma_stats is not None:
        stats["descended_loss"] = float(descended.item())
        stats.update(sigma_stats)
    return StepProbe(
        loss=float(loss.item()),
        perm_losses=info["perm_losses"].cpu().numpy().astype(np.float64),
        score_grad=grad.detach().cpu().numpy().astype(np.float64),
        fit_partitions=fit_partitions.cpu().numpy(),
        eval_partitions=eval_partitions.cpu().numpy(),
        fit_selected=fit_sel,
        deploy_selected=dep_sel,
        stats=stats,
        spec=spec,
    )


# --------------------------------------------------------------------------
# Many steps: the selection measure and the gradient's sampling noise
# --------------------------------------------------------------------------


@dataclass
class ProfileResult:
    """Per-prompt aggregates over many independent steps at a fixed score vector.

    All arrays are length N and indexed by *global rank* order (rank 0 = the
    most failure-prone prompt), so they can be read directly against rank.
    """

    spec: ObjectiveSpec
    n_draws: int
    order: np.ndarray             # [N] union indices sorted by score, descending
    p_selected: np.ndarray        # [N] fraction of steps the prompt was selected
    p_selected_fit: np.ndarray
    p_selected_deploy: np.ndarray
    p_active: np.ndarray          # [N] fraction of steps its own gradient survived
    grad_mean: np.ndarray         # [N] E[dL/dv]
    grad_sd: np.ndarray           # [N] sd over steps
    grad_abs_mean: np.ndarray     # [N] E[|dL/dv|]
    losses: np.ndarray            # [n_draws] per-step loss
    perm_sd: np.ndarray           # [n_draws] within-step sd across permutations
    stat_means: Dict[str, float]

    def summary(self) -> Dict[str, float]:
        g = self.grad_mean
        tot = np.abs(g).sum()
        top10 = np.abs(g[:10]).sum()
        top100 = np.abs(g[:100]).sum()
        return {
            "loss_mean": float(self.losses.mean()),
            "loss_sd": float(self.losses.std(ddof=0)),
            "loss_cv": float(
                self.losses.std(ddof=0) / max(1e-30, abs(self.losses.mean()))
            ),
            "perm_sd_mean": float(self.perm_sd.mean()),
            "grad_l2": float(np.sqrt((g ** 2).sum())),
            "grad_l1": float(tot),
            "grad_share_top10": float(top10 / max(1e-30, tot)),
            "grad_share_top100": float(top100 / max(1e-30, tot)),
            "n_active_expected": float(self.p_active.sum()),
            "downward_share": float(
                g[g > 0].sum() / max(1e-30, np.abs(g).sum())
            ),
            **self.stat_means,
        }


def probe_profile(
    union_lp: np.ndarray,
    spec: ObjectiveSpec,
    *,
    n_draws: int = 200,
    rng_seed: int = 0,
    anchor: np.ndarray | None = None,
) -> ProfileResult:
    """Aggregate ``n_draws`` independent steps at a fixed score vector.

    Each draw is a fresh set of K permutations, exactly as a training step draws
    them, so the spread across draws is the step-to-step sampling noise the
    optimizer actually sees at this point in score space.
    """
    v = np.asarray(union_lp, dtype=np.float64)
    n = v.shape[0]
    order = np.argsort(-v, kind="stable")
    rank_of = np.empty(n, dtype=np.int64)
    rank_of[order] = np.arange(n)

    sel = np.zeros(n)
    sel_fit = np.zeros(n)
    sel_dep = np.zeros(n)
    act = np.zeros(n)
    g_sum = np.zeros(n)
    g_sq = np.zeros(n)
    g_abs = np.zeros(n)
    losses = np.empty(n_draws)
    perm_sd = np.empty(n_draws)
    stat_acc: Dict[str, float] = {}

    for d in range(n_draws):
        pr = probe_step(v, spec, rng_seed=rng_seed + 7919 * d, anchor=anchor)
        losses[d] = pr.loss
        perm_sd[d] = float(pr.perm_losses.std(ddof=0)) if pr.perm_losses.size > 1 else 0.0
        g = pr.score_grad
        g_sum += g
        g_sq += g * g
        g_abs += np.abs(g)
        m = pr.selected_mask()
        sel[m] += 1.0
        fm = np.zeros(n, dtype=bool)
        fm[pr.fit_selected.reshape(-1)] = True
        dm = np.zeros(n, dtype=bool)
        dm[pr.deploy_selected.reshape(-1)] = True
        sel_fit[fm] += 1.0
        sel_dep[dm] += 1.0
        act[pr.active_mask()] += 1.0
        for key, val in pr.stats.items():
            if math.isfinite(val):
                stat_acc[key] = stat_acc.get(key, 0.0) + val

    inv = 1.0 / float(n_draws)
    g_mean = g_sum * inv
    g_var = np.maximum(0.0, g_sq * inv - g_mean ** 2)
    return ProfileResult(
        spec=spec,
        n_draws=n_draws,
        order=order,
        p_selected=(sel * inv)[order],
        p_selected_fit=(sel_fit * inv)[order],
        p_selected_deploy=(sel_dep * inv)[order],
        p_active=(act * inv)[order],
        grad_mean=g_mean[order],
        grad_sd=np.sqrt(g_var)[order],
        grad_abs_mean=(g_abs * inv)[order],
        losses=losses,
        perm_sd=perm_sd,
        stat_means={k: v_ * inv for k, v_ in stat_acc.items()},
    )


# --------------------------------------------------------------------------
# Tail-shape and forecast-error readouts on a raw score vector
# --------------------------------------------------------------------------


def tail_shape(union_lp: np.ndarray, *, k: int = 10, n_top: int = 300) -> Dict[str, float]:
    """Distribution-shape statistics that do not depend on any partition draw."""
    v = np.sort(np.asarray(union_lp, dtype=np.float64))[::-1]
    x = -np.log(np.maximum(-v, 1e-30))
    out = {
        "v_max": float(v[0]),
        "v_median": float(np.median(v)),
        "v_p01": float(np.percentile(v, 99)),
        "spread_top10_log10": float((v[0] - v[min(9, v.size - 1)]) / LN10),
        "spread_top100_log10": float((v[0] - v[min(99, v.size - 1)]) / LN10),
        "spread_topmed_log10": float((v[0] - np.median(v)) / LN10),
        "x_spread_top10": float(x[0] - x[min(9, x.size - 1)]),
        "iqr_log10": float((np.percentile(v, 75) - np.percentile(v, 25)) / LN10),
    }
    # Hill-type tail index on the top n_top order statistics in x-space: the
    # slope of log survival against x. Steeper (more negative) means a thinner
    # tail; -> 0 means the scores are piling up at one value.
    m = min(int(n_top), v.size - 1)
    if m >= 3:
        y = np.log(np.arange(1, m + 1) / (v.size + 1.0))
        xs = x[:m]
        b = np.polyfit(xs, y, 1)[0] if np.ptp(xs) > 0 else 0.0
        out["tail_slope_top%d" % m] = float(b)
    return out


def fixed_split_worst_error(
    union_lp: np.ndarray,
    *,
    n_fit: int,
    k: int = 10,
    plotting: str = "weibull",
    rng_seed: int = 20260401,
) -> Dict[str, float]:
    """The reported held-out metric, on one FIXED split of a score vector.

    Mirrors ``train_lquant.evaluate_on_test``: shuffle once with a fixed seed,
    fit the OLS tail line on the first ``n_fit`` prompts' top-k, forecast at the
    deploy set's size, and compare with the deploy set's worst observed score.
    Kept separate from the training loss because the two are different
    functionals: the training metric is averaged over fresh partitions every
    step, this one is a single fixed partition.
    """
    v = np.asarray(union_lp, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(v.shape[0])
    fit = v[idx[:n_fit]]
    dep = v[idx[n_fit:]]
    pred, a, b = fit_ols_tail_from_scores(
        fit.tolist(), n_top=k, plotting=plotting,
        deploy_queries=int(dep.shape[0]), score_mode="gumbel_log_prob",
    )
    actual = float(dep.max())
    err_log10 = abs(pred - actual) / LN10
    return {
        "pred_worst_log_prob": float(pred),
        "actual_worst_log_prob": actual,
        "signed_log10_error": float((pred - actual) / LN10),
        "abs_log10_error": float(err_log10),
        "sq_log10_error": float(err_log10 ** 2),
        "fit_ols_a": float(a),
        "fit_ols_b": float(b),
    }


def fixed_split_worst_error_mean(
    union_lp: np.ndarray,
    *,
    n_fit: int,
    k: int = 10,
    plotting: str = "weibull",
    n_splits: int = 24,
    rng_seed: int = 20260401,
) -> Dict[str, float]:
    """``fixed_split_worst_error`` averaged over several splits of the same pool.

    One split is a high-variance estimate: whether the pool's worst prompt lands
    in the fit half or the deploy half swings the error by orders of magnitude.
    Averaging over a fixed family of splits keeps the metric comparable across
    steps (the same splits every time) without inheriting that variance.
    """
    rows = [
        fixed_split_worst_error(
            union_lp, n_fit=n_fit, k=k, plotting=plotting, rng_seed=rng_seed + 101 * s,
        )
        for s in range(max(1, int(n_splits)))
    ]
    sq = np.array([r["sq_log10_error"] for r in rows])
    signed = np.array([r["signed_log10_error"] for r in rows])
    return {
        "sq_log10_error_mean": float(sq.mean()),
        "sq_log10_error_median": float(np.median(sq)),
        "signed_log10_error_mean": float(signed.mean()),
        "fit_ols_b_mean": float(np.mean([r["fit_ols_b"] for r in rows])),
        "under_prediction_frac": float((signed > 0).mean()),
    }


def rank_buckets(n: int, n_buckets: int = 24) -> List[Tuple[int, int]]:
    """Log-spaced rank buckets, so the tail is resolved and the bulk is pooled."""
    edges = np.unique(
        np.round(np.geomspace(1, max(2, n), num=n_buckets + 1)).astype(int)
    )
    return [(int(edges[i]) - 1, int(edges[i + 1]) - 1) for i in range(len(edges) - 1)]


def bucket_mean(values: np.ndarray, buckets: Sequence[Tuple[int, int]]) -> np.ndarray:
    out = np.full(len(buckets), np.nan)
    for i, (lo, hi) in enumerate(buckets):
        if hi > lo:
            out[i] = float(np.mean(values[lo:hi]))
    return out
