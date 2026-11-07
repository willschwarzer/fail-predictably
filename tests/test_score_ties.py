"""Exact score ties in the language-model forecast loss.

Training can create exact float32 ties among tail scores, and the forecast loss
is only piecewise smooth in the scores. These tests back the subsection "Ties in
finite-precision scores" of the appendix "Checks beyond the training forecast
and score". They call the loss that the language-model trainer calls for the
paper's recipe, with the arguments the trainer passes: fit set M = 44,
deployment set N = 891, fit top k = 10, Weibull plotting positions,
deploy-log-uniform rank weights, and errors measured in log-probability.

Run directly with the project env (pytest not required):
    python -m tests.test_score_ties
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from forecast_experiment.ols_utils import (
    OLS_SXX_TOL,
    _ols_beta_from_design,
    count_extrapolated_eval_points,
    differentiable_ols_design,
    differentiable_ols_extrapolated_primal_loss,
    differentiable_ols_extrapolated_primal_loss_bootstrap,
    select_top_k_indices,
)

N_FIT, N_DEPLOY, TOP_K = 44, 891, 10

# For this recipe train_lquant.py reaches the loss through its bootstrap path
# (--bootstrap-permutations 1). It calls
# differentiable_ols_extrapolated_primal_loss_bootstrap with `_boot_kwargs`,
# which calls differentiable_ols_extrapolated_primal_loss in ols_utils.py once
# per permutation. These are the values in `_boot_kwargs`. The trainer maps
# --extrapolated-loss-space log_prob to loss_space="original", and the anchor
# and calibration arguments are off in this recipe, so they keep their defaults.
RECIPE = dict(
    plotting="weibull",
    score_mode="gumbel_log_prob",
    grad_mode="both",
    extrapolated_weighting="deploy_log_uniform",
    loss_space="original",
)
# `_boot_kwargs` also carries the improving-only mask settings. The checks run
# without the mask, where the gradient is the loss's own derivative, and with
# the hard mask that the trainer applies under --improving-only-grad.
NO_MASK = dict(improving_only_grad=False)
HARD_MASK = dict(
    improving_only_grad=True,
    iog_scope="both",
    iog_worsening_coef=0.0,
    iog_fit_mask="closed_form",
)
MASKS = {"no mask": NO_MASK, "hard mask": HARD_MASK}

# The loss compares the forecast with the deployment ranks beyond the fit
# regime, 19 of the 891 here, counted the way the bootstrap path counts them.
N_EXTRAP = max(
    1, min(count_extrapolated_eval_points(N_FIT, N_DEPLOY, plotting="weibull"), N_DEPLOY)
)

SEEDS = (0, 1, 2, 3, 4)


def _log_probs(n: int, seed: int) -> torch.Tensor:
    """Float32 log-probabilities with a long lower tail, like the WildChat scores."""
    rng = np.random.default_rng(seed)
    return torch.tensor(-np.exp(rng.normal(1.6, 0.8, size=n)), dtype=torch.float32)


def _pool(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit and deployment scores drawn from one distribution."""
    v = _log_probs(N_FIT + N_DEPLOY, seed)
    return v[:N_FIT].clone(), v[N_FIT:].clone()


def _tie(values: torch.Tensor, rank: int) -> tuple[torch.Tensor, int, int]:
    """Raise the score at rank + 1 to the score at `rank`, where rank 1 is the highest.

    Returns the tied scores and the positions of the two tied entries.
    """
    order = torch.argsort(values, descending=True)
    upper, lower = int(order[rank - 1]), int(order[rank])
    tied = values.clone()
    tied[lower] = tied[upper]
    return tied, upper, lower


# The three ties the paper names: the side each is on and the rank of its
# upper entry.
TIES = {
    "inside the fit top k": ("fit", 4),
    "at the k/k+1 selection boundary": ("fit", TOP_K),
    "at the deployment maximum": ("deploy", 1),
}


def _tied_pool(seed: int, where: str):
    fit, deploy = _pool(seed)
    side, rank = TIES[where]
    if side == "fit":
        fit, a, b = _tie(fit, rank)
    else:
        deploy, a, b = _tie(deploy, rank)
    return fit, deploy, side, a, b


def _select(fit: torch.Tensor, deploy: torch.Tensor):
    """The selection the trainer's bootstrap path makes for each permutation.

    The fit side keeps its top k through torch.topk. The deployment side keeps
    its N_EXTRAP highest scores, in descending order, through argsort.
    """
    fit_idx = select_top_k_indices(fit.detach(), TOP_K)
    deploy_idx = torch.argsort(deploy.detach(), descending=True)[:N_EXTRAP]
    return fit_idx, deploy_idx


def _loss(fit: torch.Tensor, deploy: torch.Tensor, mask: dict = NO_MASK):
    fit_idx, deploy_idx = _select(fit, deploy)
    loss, info = differentiable_ols_extrapolated_primal_loss(
        fit, fit_idx, N_FIT, deploy, deploy_idx, N_DEPLOY, **RECIPE, **mask,
    )
    return loss, info, fit_idx, deploy_idx


@dataclass
class _Result:
    loss: torch.Tensor
    fit_grad: torch.Tensor
    deploy_grad: torch.Tensor
    fit_idx: torch.Tensor
    deploy_idx: torch.Tensor
    info: dict

    @property
    def grad(self) -> torch.Tensor:
        return torch.cat([self.fit_grad, self.deploy_grad])


def _evaluate(fit: torch.Tensor, deploy: torch.Tensor, mask: dict = NO_MASK) -> _Result:
    """Loss and score gradients at the given scores, taken as leaf tensors."""
    fit = fit.detach().clone().requires_grad_(True)
    deploy = deploy.detach().clone().requires_grad_(True)
    loss, info, fit_idx, deploy_idx = _loss(fit, deploy, mask)
    loss.backward()
    return _Result(loss.detach(), fit.grad, deploy.grad, fit_idx, deploy_idx, info)


def _value(t: torch.Tensor) -> float:
    return float(t.detach())


def test_helper_matches_the_trainer_entry_point():
    """The helpers above compute what the trainer computes, bit for bit.

    The trainer calls differentiable_ols_extrapolated_primal_loss_bootstrap on
    the union of fit and deployment scores. With one permutation that keeps the
    fit set first, it must give the same loss and score gradients as the
    direct call in _evaluate, ties included.
    """
    fit_part = torch.arange(N_FIT).unsqueeze(0)
    deploy_part = torch.arange(N_FIT, N_FIT + N_DEPLOY).unsqueeze(0)
    for seed in SEEDS:
        fit, deploy = _pool(seed)
        fit, _, _ = _tie(fit, TOP_K)
        deploy, _, _ = _tie(deploy, 1)
        for mask_name, mask in MASKS.items():
            direct = _evaluate(fit, deploy, mask)
            union = torch.cat([fit, deploy]).requires_grad_(True)
            loss, _ = differentiable_ols_extrapolated_primal_loss_bootstrap(
                union, fit_part, deploy_part,
                n_fit=N_FIT, n_eval=N_DEPLOY, fit_top_k=TOP_K, **RECIPE, **mask,
            )
            loss.backward()
            label = f"seed {seed}, {mask_name}"
            assert torch.equal(loss.detach(), direct.loss), f"{label}: loss differs"
            assert torch.equal(union.grad, direct.grad), f"{label}: gradient differs"


def test_tie_selection_is_deterministic():
    """At an exact tie, repeated selection calls pick the same entries.

    The loss depends on which tied entry the selection keeps, so a tie has to
    resolve the same way every time it is evaluated. At the k/k+1 boundary
    torch.topk keeps one of the two tied fit scores, and at the deployment
    maximum argsort orders the two tied scores. Each selection runs 100 times.
    """
    for seed in SEEDS:
        fit, deploy = _pool(seed)
        fit, a, b = _tie(fit, TOP_K)
        deploy, _, _ = _tie(deploy, 1)
        fit_idx, deploy_idx = _select(fit, deploy)
        kept = fit_idx.tolist()
        assert (a in kept) != (b in kept), f"seed {seed}: boundary tie kept {kept}"
        for _ in range(100):
            fit_again, deploy_again = _select(fit.clone(), deploy.clone())
            assert torch.equal(fit_again, fit_idx), f"seed {seed}: fit top k changed"
            assert torch.equal(deploy_again, deploy_idx), f"seed {seed}: deployment order changed"


def test_ties_keep_loss_and_gradients_finite():
    """Ties inside the fit top k, at the k/k+1 boundary and at the deployment
    maximum give a finite loss and finite score gradients, with and without the
    mask."""
    for where in TIES:
        for seed in SEEDS:
            fit, deploy, _, _, _ = _tied_pool(seed, where)
            for mask_name, mask in MASKS.items():
                r = _evaluate(fit, deploy, mask)
                label = f"tie {where}, seed {seed}, {mask_name}"
                assert torch.isfinite(r.loss), f"{label}: loss {_value(r.loss)}"
                assert torch.isfinite(r.grad).all(), f"{label}: gradient not finite"
                if mask is NO_MASK:
                    assert r.fit_grad.abs().max() > 0, f"{label}: no gradient on the fit side"


EPS = 1e-5  # nats, far below the gaps between neighboring tail scores
RTOL = 1e-3  # relative to the largest gradient entry


def _nudge(fit, deploy, side, pos, delta) -> _Result:
    fit, deploy = fit.clone(), deploy.clone()
    (fit if side == "fit" else deploy)[pos] += delta
    return _evaluate(fit, deploy)


def test_tie_gradient_is_a_one_sided_gradient():
    """At a tie, autograd returns the gradient of the loss from one side.

    The loss is smooth wherever the selected scores are strictly ordered, and
    its smooth pieces meet at ties. The limit of the gradient from either side
    of a tie is a valid subgradient there. Raising the tied entry that the
    selection ranked first by EPS keeps the selection, and the gradient there
    should match the gradient at the tie. Lowering it by EPS swaps the two
    entries' ranks, and the gradient there should be the tie gradient with the
    two entries exchanged. So each tied entry gets the gradient of the rank it
    holds, and a tied entry left out of the fit top k gets exactly zero.

    This runs without the mask, since the hard mask blocks some gradients on
    purpose and its gradient is not the loss's derivative.
    """
    for where in TIES:
        for seed in SEEDS:
            fit, deploy, side, a, b = _tied_pool(seed, where)
            tie = _evaluate(fit, deploy)
            sel = (tie.fit_idx if side == "fit" else tie.deploy_idx).tolist()
            if a in sel and (b not in sel or sel.index(a) < sel.index(b)):
                first, second = a, b
            else:
                first, second = b, a
            offset = 0 if side == "fit" else N_FIT
            i, j = offset + first, offset + second
            label = f"tie {where}, seed {seed}"

            up = _nudge(fit, deploy, side, first, +EPS)
            down = _nudge(fit, deploy, side, first, -EPS)
            sel_up = (up.fit_idx if side == "fit" else up.deploy_idx).tolist()
            sel_down = (down.fit_idx if side == "fit" else down.deploy_idx).tolist()
            swapped_sel = [second if p == first else first if p == second else p for p in sel]
            assert sel_up == sel, f"{label}: raising the first entry changed the selection"
            assert sel_down == swapped_sel, f"{label}: lowering it did not swap the two entries"

            g = tie.grad
            scale = _value(g.abs().max())
            swapped = g.clone()
            swapped[i], swapped[j] = g[j], g[i]
            err_up = _value((up.grad - g).abs().max())
            err_down = _value((down.grad - swapped).abs().max())
            assert err_up <= RTOL * scale, f"{label}: gradient {err_up:.3g} off the side it chose"
            assert err_down <= RTOL * scale, f"{label}: other side {err_down:.3g} off the swap"

            if second not in sel:
                assert _value(g[j]) == 0.0, f"{label}: unselected twin got {_value(g[j])}"
                assert _value(g[i]) != 0.0, f"{label}: selected twin got no gradient"
            else:
                assert _value(g[i]) != _value(g[j]), f"{label}: twins got the same gradient"


LR, STEPS = 3e-3, 30
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def _adam_step(theta: torch.Tensor, m: torch.Tensor, s: torch.Tensor, step: int) -> None:
    """One Adam update with the usual defaults, in place.

    Written out because building torch.optim.Adam imports torch._dynamo, which
    takes longer than everything else in this file.
    """
    g = theta.grad
    m.mul_(BETA1).add_(g, alpha=1 - BETA1)
    s.mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
    m_hat = m / (1 - BETA1**step)
    s_hat = s / (1 - BETA2**step)
    with torch.no_grad():
        theta -= LR * m_hat / (s_hat.sqrt() + ADAM_EPS)
    theta.grad = None


def test_persistent_tie_between_scores_sharing_a_parameter():
    """Two fit scores that read the same parameter stay tied at every step.

    No update can separate them, so every step of the optimization evaluates the
    loss at a tie. Each of the top eleven fit scores is its own parameter except
    the pair, which shares one and starts at ranks 10 and 11 (the k/k+1
    boundary) or at ranks 4 and 5. Adam should lower the loss, with finite
    gradients at every step.
    """
    for rank in (TOP_K, 4):
        for mask_name, mask in MASKS.items():
            for seed in SEEDS:
                fit, deploy = _pool(seed)
                order = torch.argsort(fit, descending=True)
                top, rest = fit[order[: TOP_K + 1]], fit[order[TOP_K + 1 :]]
                theta = torch.cat([top[:rank], top[rank + 1 :]]).requires_grad_(True)
                m, s = torch.zeros_like(theta), torch.zeros_like(theta)
                label = f"pair at ranks {rank} and {rank + 1}, seed {seed}, {mask_name}"
                losses = []
                for step in range(1, STEPS + 1):
                    scores = torch.cat([theta[:rank], theta[rank - 1 : rank], theta[rank:], rest])
                    assert _value(scores[rank - 1]) == _value(scores[rank]), f"{label}: tie broken"
                    loss, _, _, _ = _loss(scores, deploy, mask)
                    loss.backward()
                    assert torch.isfinite(loss), f"{label}, step {step}: loss {_value(loss)}"
                    assert torch.isfinite(theta.grad).all(), f"{label}, step {step}: gradient"
                    if step == 1 and mask is NO_MASK:
                        assert _value(theta.grad[rank - 1]) != 0.0, f"{label}: pair got no gradient"
                    losses.append(_value(loss))
                    _adam_step(theta, m, s, step)
                assert losses[-1] < losses[0], f"{label}: loss {losses[0]:.4g} -> {losses[-1]:.4g}"


def _selected_sxx(fit: torch.Tensor) -> float:
    """sxx, the sum of squared deviations of the selected fit scores after the
    transform, computed the way the loss computes it before comparing it with
    OLS_SXX_TOL."""
    x, y = differentiable_ols_design(
        fit, select_top_k_indices(fit, TOP_K), N_FIT,
        plotting=RECIPE["plotting"], score_mode=RECIPE["score_mode"],
    )
    _, sxx, _ = _ols_beta_from_design(x, y)
    return _value(sxx)


def _fit_top_k_on_one_score(seed: int):
    """Fit scores whose whole top k equals one score, chosen to reach the guard.

    The tied score starts at the highest fit score. The float32 mean of ten
    equal numbers can be off by one unit in the last place, and then sxx comes
    out near 1e-14, far above OLS_SXX_TOL, so the guard is not reached. The
    tied score therefore moves up one float32 step at a time, to the nearest
    value whose transformed score has a mantissa short enough that ten copies
    of it sum without rounding, in any order.
    """
    fit, deploy = _pool(seed)
    top = torch.argsort(fit, descending=True)[:TOP_K]
    v = fit[top[0]].clone()
    for _ in range(1000):
        tied = fit.clone()
        tied[top] = v
        x, _ = differentiable_ols_design(
            tied, select_top_k_indices(tied, TOP_K), N_FIT,
            plotting=RECIPE["plotting"], score_mode=RECIPE["score_mode"],
        )
        scores = x[:, 1]
        mantissa, _ = math.frexp(_value(scores[0]))
        if (mantissa * 2**20).is_integer() and bool((scores == scores[0]).all()):
            return tied, deploy, _value(v)
        v = torch.nextafter(v, torch.zeros(()))
    raise AssertionError(f"seed {seed}: no tied score with a short mantissa nearby")


def test_all_tied_fit_top_k_reaches_the_degenerate_guard():
    """When the whole fit top k ties, the guard keeps the loss finite.

    With every selected fit score equal, the OLS slope is 0/0. When sxx is
    below OLS_SXX_TOL, the loss forecasts the tied score at every deployment
    rank in place of inverting the line.
    """
    for seed in SEEDS:
        fit, deploy, v = _fit_top_k_on_one_score(seed)
        sxx = _selected_sxx(fit)
        assert sxx < OLS_SXX_TOL, f"seed {seed}: sxx {sxx:.3g} misses the guard"
        for mask_name, mask in MASKS.items():
            r = _evaluate(fit, deploy, mask)
            label = f"seed {seed}, {mask_name}"
            assert torch.isfinite(r.loss), f"{label}: loss {_value(r.loss)}"
            assert not torch.isnan(r.grad).any(), f"{label}: NaN in the gradient"
            assert torch.isfinite(r.grad).all(), f"{label}: gradient not finite"
            forecast_max = _value(r.info["predicted_max_value"])
            forecast_min = _value(r.info["predicted_min_value"])
            assert forecast_max == forecast_min, (
                f"{label}: forecast runs from {forecast_max} to {forecast_min}"
            )
            assert abs(forecast_max - v) <= 1e-5 * abs(v), (
                f"{label}: forecast {forecast_max}, tied score {v}"
            )


_TESTS = [
    test_helper_matches_the_trainer_entry_point,
    test_tie_selection_is_deterministic,
    test_ties_keep_loss_and_gradients_finite,
    test_tie_gradient_is_a_one_sided_gradient,
    test_persistent_tie_between_scores_sharing_a_parameter,
    test_all_tied_fit_top_k_reaches_the_degenerate_guard,
]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
    if failed:
        print(f"\n{failed} of {len(_TESTS)} tests failed")
        return 1
    print(f"\nAll {len(_TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
