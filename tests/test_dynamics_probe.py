"""Unit tests for the learning-dynamics instrumentation.

These guard the two things the whole study rests on: that the new objective
options reduce to the canonical recipe at their defaults, and that the
score-space gradient the probe reports is the real derivative of the loss.

Run directly with the project env (pytest not required):
    python -m tests.test_dynamics_probe
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from forecast_experiment.dynamics_probe import (
    ObjectiveSpec,
    population_line_gaps,
    probe_profile,
    probe_step,
)
from forecast_experiment.ols_utils import (
    _extrapolated_rank_weights,
    count_extrapolated_eval_points,
    differentiable_ols_extrapolated_primal_loss,
    differentiable_ols_extrapolated_primal_loss_bootstrap,
    plan_bootstrap_permutations,
    select_top_k_indices,
)


def _pool(n: int = 200, seed: int = 0) -> np.ndarray:
    """A heavy-tailed log-prob pool shaped like the real WildChat scores."""
    rng = np.random.default_rng(seed)
    return -np.exp(rng.normal(1.6, 0.8, size=n)).astype(np.float64)


def _close(a, b, *, rtol=1e-6, atol=1e-9, what=""):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        d = np.max(np.abs(np.asarray(a) - np.asarray(b)))
        raise AssertionError(f"{what}: max abs diff {d:.3g}")


def test_worst_rank_weighting_is_one_hot() -> None:
    w = _extrapolated_rank_weights(
        5, 100, "weibull", weighting="worst_rank",
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.allclose(w, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    w_u = _extrapolated_rank_weights(
        4, 100, "weibull", weighting="rank_uniform",
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert torch.allclose(w_u, torch.full((4,), 0.25))
    print("ok: worst_rank weighting is one-hot; rank_uniform unchanged")


def test_bootstrap_k1_matches_direct_call() -> None:
    """K=1 through the bootstrap wrapper must equal the single-partition loss."""
    v = torch.tensor(_pool(120, seed=3), dtype=torch.float32)
    n_fit, n_eval = 20, 100
    b = max(1, min(count_extrapolated_eval_points(n_fit, n_eval, plotting="weibull"), n_eval))
    fit_p, eval_p, _ = plan_bootstrap_permutations(
        n_union=120, n_fit=n_fit, n_eval=n_eval, n_permutations=1,
        rng_seed=7, fit_top_k=10, eval_extrap_k=b,
        union_values_detached=v, grad_mode="both",
    )
    boot, _ = differentiable_ols_extrapolated_primal_loss_bootstrap(
        v, fit_p, eval_p, n_fit=n_fit, n_eval=n_eval, fit_top_k=10,
        plotting="weibull", score_mode="gumbel_log_prob", grad_mode="both",
        extrapolated_weighting="deploy_log_uniform", loss_space="original",
        improving_only_grad=True,
    )
    fit_vals = v.index_select(0, fit_p[0])
    eval_vals = v.index_select(0, eval_p[0])
    direct, _ = differentiable_ols_extrapolated_primal_loss(
        fit_vals, select_top_k_indices(fit_vals, 10), n_fit,
        eval_vals, torch.argsort(eval_vals, descending=True)[:b], n_eval,
        plotting="weibull", score_mode="gumbel_log_prob", grad_mode="both",
        extrapolated_weighting="deploy_log_uniform", loss_space="original",
        improving_only_grad=True,
    )
    _close(float(boot), float(direct), rtol=1e-6, what="bootstrap K=1 vs direct")
    print(f"ok: bootstrap K=1 == direct call ({float(boot):.6f})")


def test_anchor_gate_endpoints_bracket_the_hard_mask() -> None:
    """The gate interpolates between the hard mask and no mask at all.

    At a fixed suppression depth, a margin far larger than the depth leaves the
    gate near zero (the hard mask), and a margin far smaller saturates it at one
    (no mask). The margin is therefore "how much slack a suppressed prompt earns
    per nat of suppression", which is the knob the fleet varies.
    """
    v_np = _pool(160, seed=5)
    spec = ObjectiveSpec(n_fit=30, n_eval=130, n_permutations=2)
    # Anchor 20 nats above every current score: every prompt reads as already
    # deeply suppressed relative to where it started.
    anchor = v_np + 20.0

    hard = probe_step(v_np, spec, rng_seed=11).score_grad
    no_mask = probe_step(
        v_np, replace(spec, improving_only_grad=False), rng_seed=11,
    ).score_grad
    gate_closed = probe_step(
        v_np, replace(spec, anchor_margin=2e7), rng_seed=11, anchor=anchor,
    ).score_grad
    gate_open = probe_step(
        v_np, replace(spec, anchor_margin=1.0), rng_seed=11, anchor=anchor,
    ).score_grad

    _close(gate_closed, hard, rtol=1e-4, atol=1e-5, what="margin >> depth vs hard mask")
    _close(gate_open, no_mask, rtol=1e-4, atol=1e-8, what="margin << depth vs no mask")
    print("ok: anchor gate brackets {hard mask, no mask}")


def test_anchor_gate_blocks_points_at_their_anchor() -> None:
    """A point sitting exactly at its anchor keeps the hard block."""
    v_np = _pool(160, seed=6)
    spec = ObjectiveSpec(n_fit=30, n_eval=130, n_permutations=2, anchor_margin=5.0)
    at_anchor = probe_step(v_np, spec, rng_seed=13, anchor=v_np.copy()).score_grad
    hard = probe_step(v_np, replace(spec, anchor_margin=0.0), rng_seed=13).score_grad
    _close(at_anchor, hard, rtol=1e-5, what="points at anchor vs hard mask")
    print("ok: anchor gate blocks points at their starting score")


def test_score_grad_matches_finite_differences() -> None:
    """Without the mask, the probe's dL/dv is the loss's actual derivative.

    The mask is excluded here on purpose. It detaches blocked points, so the
    gradient the model receives is deliberately *not* the loss's derivative --
    see ``test_masked_points_receive_no_gradient_but_still_move_the_loss``.
    """
    v_np = _pool(150, seed=9)
    spec = ObjectiveSpec(
        n_fit=28, n_eval=122, n_permutations=3, improving_only_grad=False,
    )
    pr = probe_step(v_np, spec, rng_seed=17)
    active = np.flatnonzero(pr.active_mask())
    assert active.size >= 3, "expected at least a few active prompts"
    eps = 1e-3
    checked = 0
    for i in active[:8]:
        plus, minus = v_np.copy(), v_np.copy()
        plus[i] += eps
        minus[i] -= eps
        lp = probe_step(plus, spec, rng_seed=17)
        lm = probe_step(minus, spec, rng_seed=17)
        # Skip a point whose perturbation reshuffled a selection: the loss is
        # only piecewise smooth, and at a selection boundary the finite
        # difference straddles two branches.
        if not np.array_equal(lp.fit_selected, pr.fit_selected):
            continue
        if not np.array_equal(lp.deploy_selected, pr.deploy_selected):
            continue
        fd = (lp.loss - lm.loss) / (2 * eps)
        rel = abs(fd - pr.score_grad[i]) / max(1e-9, abs(pr.score_grad[i]))
        assert rel < 3e-2, f"prompt {i}: fd {fd:.6g} vs autograd {pr.score_grad[i]:.6g}"
        checked += 1
    assert checked >= 3, f"only {checked} clean finite-difference checks"
    print(f"ok: score_grad matches finite differences on {checked} active prompts")


def test_masked_points_receive_no_gradient_but_still_move_the_loss() -> None:
    """The mask is a biased subgradient, and the probe reports it as such.

    A blocked prompt gets zero applied gradient while the loss remains sensitive
    to it -- so the loss can climb along the direction training takes. Every
    statement in the report about "what the objective asks of prompt i" means
    the applied gradient, not the loss's derivative.
    """
    v_np = _pool(220, seed=41)
    spec = ObjectiveSpec(n_fit=34, n_eval=186, n_permutations=1)
    pr = probe_step(v_np, spec, rng_seed=23)
    blocked = np.flatnonzero(pr.selected_mask() & ~pr.active_mask())
    assert blocked.size >= 1, "expected the mask to block at least one selected prompt"
    eps = 1e-3
    moved = 0
    for i in blocked[:6]:
        plus, minus = v_np.copy(), v_np.copy()
        plus[i] += eps
        minus[i] -= eps
        lp, lm = probe_step(plus, spec, rng_seed=23), probe_step(minus, spec, rng_seed=23)
        if not np.array_equal(lp.deploy_selected, pr.deploy_selected):
            continue
        fd = (lp.loss - lm.loss) / (2 * eps)
        assert pr.score_grad[i] == 0.0, f"blocked prompt {i} still got gradient"
        if abs(fd) > 1e-6:
            moved += 1
    assert moved >= 1, "expected the loss to depend on at least one blocked prompt"
    print(f"ok: {blocked.size} selected prompts blocked; {moved} still move the loss")


def test_exact_fit_mask_closes_the_worsening_leak() -> None:
    """The closed-form fit mask leaks upward gradients; the exact one does not.

    The closed form differentiates the SCORE-space squared error. In the LM
    configuration the loss is in log-probability space, and the two disagree in
    sign on a few percent of active fit points -- which is precisely the mask
    letting through a gradient it exists to block.
    """
    v_np = _pool(935, seed=57)
    spec = ObjectiveSpec(n_fit=44, n_eval=891, loss_space="original")
    leaked = fixed = 0
    for d in range(120):
        cf = probe_step(v_np, spec, rng_seed=900 + d)
        ex = probe_step(v_np, replace(spec, iog_fit_mask="exact"), rng_seed=900 + d)
        fit_cf = np.zeros(v_np.size, dtype=bool)
        fit_cf[cf.fit_selected.reshape(-1)] = True
        leaked += int(((cf.score_grad < 0) & fit_cf).sum())
        fit_ex = np.zeros(v_np.size, dtype=bool)
        fit_ex[ex.fit_selected.reshape(-1)] = True
        fixed += int(((ex.score_grad < 0) & fit_ex).sum())
    assert leaked > 0, "expected the closed form to leak on this pool"
    assert fixed == 0, f"exact mask still leaked {fixed} upward fit gradients"
    # The fix must not quietly change the objective's size or aim.
    a = probe_profile(v_np, spec, n_draws=60, rng_seed=3).summary()
    b = probe_profile(
        v_np, replace(spec, iog_fit_mask="exact"), n_draws=60, rng_seed=3,
    ).summary()
    assert abs(a["grad_l2"] - b["grad_l2"]) / a["grad_l2"] < 0.05
    assert abs(a["loss_mean"] - b["loss_mean"]) < 1e-6
    print(f"ok: closed form leaked {leaked} upward fit gradients, exact leaked {fixed}")


def test_iog_makes_every_gradient_downward() -> None:
    """Under the hard mask no prompt is ever asked to become more failure-prone."""
    v_np = _pool(300, seed=21)
    spec = ObjectiveSpec(n_fit=40, n_eval=260, n_permutations=4)
    on = probe_profile(v_np, spec, n_draws=20, rng_seed=2).summary()["downward_share"]
    off = probe_profile(
        v_np, replace(spec, improving_only_grad=False), n_draws=20, rng_seed=2,
    ).summary()["downward_share"]
    assert on > 0.999, f"IOG on but downward share only {on:.4f}"
    assert off < 0.999, f"IOG off yet downward share {off:.4f}"
    print(f"ok: downward gradient share {on:.4f} with IOG, {off:.4f} without")


def test_more_permutations_reduce_loss_variance() -> None:
    v_np = _pool(400, seed=31)
    sds = []
    for k in (1, 4, 16):
        prof = probe_profile(
            v_np, ObjectiveSpec(n_fit=44, n_eval=356, n_permutations=k),
            n_draws=60, rng_seed=5,
        )
        sds.append(prof.summary()["loss_sd"])
    assert sds[0] > sds[1] > sds[2], f"loss sd not monotone in K: {sds}"
    # Averaging K draws should shrink the sd toward the 1/sqrt(K) rate. Slack of
    # 3x because the draws share one score vector and the mask couples them.
    assert sds[2] < sds[0] / math.sqrt(16) * 3.0, f"variance reduction too weak: {sds}"
    print(f"ok: loss sd falls with K: {[round(s, 4) for s in sds]}")


def test_expected_loss_does_not_depend_on_k() -> None:
    """The bootstrap loss is a plain mean over independent permutations.

    Each permutation's loss depends only on its own partition, so E[L_K] is the
    same for every K and only its variance falls. This is the reason permutation
    resampling cannot move the objective's sampling floor, and it is worth
    pinning down as an invariant rather than an observation.
    """
    v_np = _pool(400, seed=37)
    means = []
    for k in (1, 8, 32):
        prof = probe_profile(
            v_np, ObjectiveSpec(n_fit=44, n_eval=356, n_permutations=k),
            n_draws=400, rng_seed=13,
        )
        s = prof.summary()
        means.append((s["loss_mean"], s["loss_sd"] / math.sqrt(400)))
    ref, ref_sem = means[0]
    for k, (m, sem) in zip((1, 8, 32), means):
        tol = 4.0 * math.sqrt(ref_sem ** 2 + sem ** 2)
        assert abs(m - ref) <= tol, (
            f"E[L] moved with K: K={k} gives {m:.4f} vs {ref:.4f} (tol {tol:.4f})"
        )
    print(f"ok: E[loss] flat in K: {[round(m, 4) for m, _ in means]}")


def test_scale_normalization_is_invariant_to_the_tails_reach() -> None:
    """Dividing by the top-m variance removes the loss's degree-two homogeneity.

    Shrink the pool's reach below its median at fixed shape and the raw loss
    falls roughly as the square of the reach. The normalized objective is
    designed not to, which is what leaves shape as the only thing to descend on.

    The pool here is deliberately not ``_pool``. The homogeneity comes from
    ``x = -log(-v)`` being nearly affine in the deviation from the bulk, which
    needs the reach to be small against the level. The real cell satisfies that
    comfortably -- a level near 70 nats with a reach of a few -- while ``_pool``
    has a reach comparable to its level and does not, so on ``_pool`` halving the
    reach moves the raw loss the other way.
    """
    rng = np.random.default_rng(41)
    v_np = -(60.0 + np.exp(rng.normal(1.0, 0.7, size=400))).astype(np.float64)
    med = float(np.median(v_np))
    half = np.minimum(med + 0.5 * (v_np - med), -1e-9)
    spec = ObjectiveSpec(n_fit=44, n_eval=356, n_permutations=4)
    normed = replace(spec, scale_normalize_m=100)

    def mean_over(vv, sp, key):
        vals = [
            probe_step(vv, sp, rng_seed=97 + 7919 * d).stats.get(key, float("nan"))
            if key else probe_step(vv, sp, rng_seed=97 + 7919 * d).loss
            for d in range(150)
        ]
        return float(np.mean(vals))

    raw_full = mean_over(v_np, spec, None)
    raw_half = mean_over(half, spec, None)
    nrm_full = mean_over(v_np, normed, "descended_loss")
    nrm_half = mean_over(half, normed, "descended_loss")

    raw_ratio = raw_half / raw_full
    nrm_ratio = nrm_half / nrm_full
    assert raw_ratio < 0.6, f"raw loss barely moved with the reach: {raw_ratio:.3f}"
    assert nrm_ratio > raw_ratio * 2.0, (
        f"normalization did not flatten the scale dependence: "
        f"raw {raw_ratio:.3f} vs normalized {nrm_ratio:.3f}"
    )
    # The reported loss must stay the raw one, so traces remain comparable.
    assert abs(mean_over(v_np, normed, None) - raw_full) < 1e-9 * max(1.0, raw_full) + 0.5
    print(
        f"ok: halving the reach scales the raw loss by {raw_ratio:.3f}, "
        f"the normalized objective by {nrm_ratio:.3f}"
    )


def _realistic_pool(n: int = 400, seed: int = 41) -> np.ndarray:
    """Level near 60 nats, reach a few nats: the regime the real cells sit in."""
    rng = np.random.default_rng(seed)
    return -(60.0 + np.exp(rng.normal(1.0, 0.7, size=n))).astype(np.float64)


def _rebuild_top_gaps(v: np.ndarray, m: int, gaps: np.ndarray) -> np.ndarray:
    """A pool whose top-m consecutive gaps are ``gaps``, rank m held fixed."""
    out = np.asarray(v, dtype=np.float64).copy()
    order = np.argsort(-out, kind="stable")
    top = out[order[:m]].copy()
    for r in range(m - 2, -1, -1):
        top[r] = top[r + 1] + gaps[r]
    out[order[:m]] = top
    return np.minimum(out, -1e-9)


def _hinge_at(v: np.ndarray, spec: ObjectiveSpec, anchor: np.ndarray) -> float:
    pr = probe_step(v, spec, rng_seed=3, anchor=anchor)
    return float(pr.stats["gap_hinge_sum"])


def test_gap_floor_hinge_is_one_sided_in_every_gap() -> None:
    """Zero on the line and on wider tails; positive on every compression move.

    The moves include the two that defeat the coarser constraints considered
    first: redistribution (an aggregate like the tail's variance survives while
    the top compresses) and a plateau raised from below (per-rank value floors
    stay satisfied while every gap collapses).
    """
    v = _realistic_pool()
    m = 60
    g0 = population_line_gaps(v, m=m)
    assert (g0 > 0).all(), "the population line's gaps must be positive"
    spec = ObjectiveSpec(
        n_fit=44, n_eval=356, gap_floor_m=m, gap_floor_lambda=1.0,
    )

    line = _rebuild_top_gaps(v, m, g0)
    assert _hinge_at(line, spec, v) < 1e-9, "the line itself must be feasible"
    wider = _rebuild_top_gaps(v, m, g0 * 1.3)
    assert _hinge_at(wider, spec, v) < 1e-9, "extra spread must stay feasible"

    med = float(np.median(line))
    squeezed = np.minimum(med + 0.5 * (line - med), -1e-9)
    assert _hinge_at(squeezed, spec, v) > 0.1, "global rescale must violate"
    ceiling = np.minimum(0.5 * line, -1e-9)
    assert _hinge_at(ceiling, spec, v) > 0.1, "ceiling saturation must violate"
    plateau = line.copy()
    order = np.argsort(-plateau, kind="stable")
    plateau[order[1:m]] = plateau[order[0]] - np.arange(1, m) * 1e-7
    assert _hinge_at(plateau, spec, v) > 0.1, "plateau-from-below must violate"
    redis = g0.copy()
    redis[:5] = 1e-9
    redis[20:] += g0[:5].sum() / (m - 21)
    assert _hinge_at(_rebuild_top_gaps(v, m, redis), spec, v) > 0.1, (
        "redistribution across the constrained set must violate"
    )
    print("ok: gap hinge is zero on/above the line, positive on every compression move")


def test_gap_floor_gradient_opens_violated_gaps_and_leaves_the_loss_alone() -> None:
    v = _realistic_pool()
    m = 60
    med = float(np.median(v))
    comp = np.minimum(med + 0.5 * (v - med), -1e-9)
    base = ObjectiveSpec(n_fit=44, n_eval=356)
    gated = replace(base, gap_floor_m=m, gap_floor_lambda=2.0)

    pr0 = probe_step(comp, base, rng_seed=11)
    pr1 = probe_step(comp, gated, rng_seed=11, anchor=v)
    assert abs(pr0.loss - pr1.loss) < 1e-9, "the reported loss must stay raw"

    extra = pr1.score_grad - pr0.score_grad
    assert np.abs(extra).sum() > 0, "the hinge must contribute gradient"
    h_before = float(pr1.stats["gap_hinge_sum"])
    stepped = comp - 1e-3 * extra
    h_after = _hinge_at(stepped, gated, v)
    assert h_after < h_before, (
        f"descending the hinge gradient must open the gaps: {h_before:.4f} -> {h_after:.4f}"
    )
    print(
        f"ok: hinge gradient opens violated gaps ({h_before:.4f} -> {h_after:.4f}) "
        "and the reported loss stays raw"
    )


def test_gap_floor_requires_the_anchor() -> None:
    v = _realistic_pool()
    spec = ObjectiveSpec(n_fit=44, n_eval=356, gap_floor_m=60, gap_floor_lambda=1.0)
    try:
        probe_step(v, spec, rng_seed=0)
    except ValueError as e:
        assert "anchor" in str(e)
        print("ok: gap floor without an anchor raises")
        return
    raise AssertionError("gap floor without an anchor must raise")


def test_window_floors_generalize_the_gap_floors() -> None:
    """B-rank span floors: same one-sided geometry, features B gaps wide.

    The point of the windows is the measurement channel: a span's floor grows
    ~B-fold while per-score noise on it stays ~sd*sqrt(2), so at B in the tens
    the constraint reads above the scoring noise that dominated the per-gap
    arms. The geometry must be unchanged: zero on the line and on wider tails,
    positive on compression, and B = 1 must reproduce the per-gap hinge.
    """
    v = _realistic_pool()
    m = 60
    g0 = population_line_gaps(v, m=m)
    win = ObjectiveSpec(
        n_fit=44, n_eval=356, gap_floor_m=m, gap_floor_lambda=1.0,
        gap_floor_block=8,
    )
    per = replace(win, gap_floor_block=1)

    line = _rebuild_top_gaps(v, m, g0)
    assert _hinge_at(line, win, v) < 1e-9, "the line must be window-feasible"
    wider = _rebuild_top_gaps(v, m, g0 * 1.3)
    assert _hinge_at(wider, win, v) < 1e-9, "extra spread must stay feasible"
    med = float(np.median(line))
    squeezed = np.minimum(med + 0.5 * (line - med), -1e-9)
    assert _hinge_at(squeezed, win, v) > 0.1, "rescale must violate windows"
    plateau = line.copy()
    order = np.argsort(-plateau, kind="stable")
    plateau[order[1:m]] = plateau[order[0]] - np.arange(1, m) * 1e-7
    assert _hinge_at(plateau, win, v) > 0.1, "plateau must violate windows"

    # noise robustness: per-score noise of 0.05 nats at the line fires the
    # per-gap hinge but leaves windows with a matched relative slack quiet
    noisy_per = replace(per, constraint_noise_sd=0.05)
    noisy_win = replace(
        win, constraint_noise_sd=0.05, gap_floor_slack=2.0 * 0.05 * np.sqrt(2.0),
    )
    h_per = np.mean([_hinge_at(line, noisy_per, v) for _ in range(3)])
    h_win = np.mean([_hinge_at(line, noisy_win, v) for _ in range(3)])
    assert h_per > 0.2, "per-gap hinge must be swamped by the noise channel"
    assert h_win < 0.05 * h_per, (
        f"windows with noise-matched slack must stay quiet: {h_win:.4f} vs {h_per:.4f}"
    )
    print(
        f"ok: window floors keep the gap geometry (B=1 hinge {_hinge_at(squeezed, per, v):.3f}) "
        f"and read above the noise (line hinge {h_win:.4f} vs per-gap {h_per:.4f})"
    )


def test_sigma_floor_binds_below_the_line_reference_and_averages_over_k() -> None:
    """The sigma floor is referenced to the LINE through the anchor pool.

    An observed-pool reference mis-pins the optimum's scale (the observed
    curvature carries sigma the line does not, measured +47%/-16% across the
    two cells), so the reference must be the line's own involved-set sigma:
    then the corrected tail sits exactly on the boundary. The hinge must be
    zero at/above the reference, positive on a ceiling-ward contraction, and
    its permutation average must tighten with K.
    """
    v = _realistic_pool()
    spec = ObjectiveSpec(n_fit=44, n_eval=356, sigma_floor_lambda=5.0)
    pr = probe_step(v, spec, rng_seed=2, anchor=v)
    ref = pr.stats["sigma_ref"]
    assert ref > 0, "the line reference must be positive"

    contracted = np.minimum(0.5 * v, -1e-9)
    prc = probe_step(contracted, spec, rng_seed=2, anchor=v)
    assert prc.stats["sigma_hinge"] > 0.1, "ceiling contraction must violate"
    g_extra = prc.score_grad - probe_step(
        contracted, replace(spec, sigma_floor_lambda=0.0), rng_seed=2, anchor=v,
    ).score_grad
    assert np.abs(g_extra).sum() > 0, "the sigma hinge must contribute gradient"

    # The reference is the line through the anchor's top-250 (the study's tail
    # convention), so "wider than the reference" must widen at that same m.
    widened = _rebuild_top_gaps(v, 250, population_line_gaps(v, m=250) * 1.3)
    prw = probe_step(widened, spec, rng_seed=2, anchor=v)
    assert prw.stats["sigma_hinge"] < 1e-9, "extra spread must stay feasible"

    b1 = [probe_step(v, replace(spec, n_permutations=1), rng_seed=s, anchor=v)
          .stats["sigma_bar"] for s in range(16)]
    b8 = [probe_step(v, replace(spec, n_permutations=8), rng_seed=s, anchor=v)
          .stats["sigma_bar"] for s in range(16)]
    assert np.std(b8) < 0.6 * np.std(b1), (
        f"K must average the partition noise: sd K=1 {np.std(b1):.4f} vs K=8 {np.std(b8):.4f}"
    )
    print(
        f"ok: sigma floor binds below the line reference ({prc.stats['sigma_hinge']:.3f} "
        f"violated when contracted) and K averages its noise "
        f"({np.std(b1):.4f} -> {np.std(b8):.4f})"
    )


def test_level_gated_hinge_only_pushes_down() -> None:
    """The level-gated hinge opens violated spans by suppression only.

    Round 1's un-gated window arms drifted into the ceiling because the
    hinge's upward half was the one strong force outside the improving-only
    mask. Gated, every constraint-gradient component must point down (or be
    zero), and descending it must still close the violation.
    """
    v = _realistic_pool()
    med = float(np.median(v))
    comp = np.minimum(med + 0.5 * (v - med), -1e-9)
    base = ObjectiveSpec(n_fit=44, n_eval=356)
    gated = replace(base, gap_floor_m=60, gap_floor_lambda=2.0,
                    gap_floor_block=8, gap_floor_level_gated=True)
    pr0 = probe_step(comp, base, rng_seed=11)
    pr1 = probe_step(comp, gated, rng_seed=11, anchor=v)
    extra = pr1.score_grad - pr0.score_grad
    assert np.abs(extra).sum() > 0, "the gated hinge must contribute gradient"
    assert extra.min() >= -1e-12, (
        f"level-gated hinge must never push a score up: min grad {extra.min():.2e}"
    )
    h_before = float(pr1.stats["gap_hinge_sum"])
    h_after = _hinge_at(comp - 2e-3 * extra, gated, v)
    assert h_after < h_before, f"gated descent must still close: {h_before:.4f} -> {h_after:.4f}"
    print(
        f"ok: level-gated hinge pushes only down (min extra grad {extra.min():.1e}) "
        f"and still closes violations ({h_before:.4f} -> {h_after:.4f})"
    )


def main() -> None:
    test_worst_rank_weighting_is_one_hot()
    test_bootstrap_k1_matches_direct_call()
    test_anchor_gate_endpoints_bracket_the_hard_mask()
    test_anchor_gate_blocks_points_at_their_anchor()
    test_score_grad_matches_finite_differences()
    test_masked_points_receive_no_gradient_but_still_move_the_loss()
    test_exact_fit_mask_closes_the_worsening_leak()
    test_iog_makes_every_gradient_downward()
    test_more_permutations_reduce_loss_variance()
    test_expected_loss_does_not_depend_on_k()
    test_scale_normalization_is_invariant_to_the_tails_reach()
    test_gap_floor_hinge_is_one_sided_in_every_gap()
    test_gap_floor_gradient_opens_violated_gaps_and_leaves_the_loss_alone()
    test_gap_floor_requires_the_anchor()
    test_window_floors_generalize_the_gap_floors()
    test_sigma_floor_binds_below_the_line_reference_and_averages_over_k()
    test_level_gated_hinge_only_pushes_down()
    print("\nall dynamics-probe tests passed")


if __name__ == "__main__":
    main()
