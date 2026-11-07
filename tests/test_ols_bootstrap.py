"""Unit tests for the bootstrap-permutation OLS extrapolation loss.

Run directly with the project env (pytest not required):
    python -m tests.test_ols_bootstrap
"""
from __future__ import annotations

import math

import torch

from forecast_experiment.ols_utils import (
    count_extrapolated_eval_points,
    differentiable_ols_extrapolated_primal_loss,
    differentiable_ols_extrapolated_primal_loss_bootstrap,
    plan_bootstrap_permutations,
)


def _make_union(n: int, seed: int = 0, requires_grad: bool = True) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Heavy-tailed enough that top-k selection is non-trivial.
    base = torch.randn(n, generator=g) - 2.0
    base = base.to(torch.float32)
    if requires_grad:
        base = base.detach().clone().requires_grad_(True)
    return base


def test_smoke_shape_and_grad():
    n_fit, n_eval = 30, 60
    n_union = n_fit + n_eval
    for n_perm in (1, 4, 16):
        union = _make_union(n_union, seed=7)
        fit_p, eval_p, _ = plan_bootstrap_permutations(
            n_union, n_fit, n_eval, n_perm,
            rng_seed=42,
            fit_top_k=10,
            eval_extrap_k=count_extrapolated_eval_points(n_fit, n_eval),
            union_values_detached=union.detach(),
            grad_mode="fit",
        )
        assert fit_p.shape == (n_perm, n_fit)
        assert eval_p.shape == (n_perm, n_eval)
        # No record can appear in both fit and eval of the same permutation.
        for p in range(n_perm):
            assert len(set(fit_p[p].tolist()) & set(eval_p[p].tolist())) == 0

        loss, info = differentiable_ols_extrapolated_primal_loss_bootstrap(
            union, fit_p, eval_p,
            n_fit=n_fit, n_eval=n_eval, fit_top_k=10,
            score_mode="lm_gumbel",
            grad_mode="fit",
            extrapolated_weighting="rank_uniform",
            loss_space="original",
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad
        loss.backward()
        assert union.grad is not None
        assert union.grad.shape == (n_union,)
        assert torch.isfinite(union.grad).all()


def test_n_perm_1_matches_direct_call():
    """With n_perm=1, the bootstrapped loss should equal a direct call on the
    same partition (using the same fit/eval indices and top-k selection).
    """
    n_fit, n_eval = 30, 60
    n_union = n_fit + n_eval
    union = _make_union(n_union, seed=11)

    fit_p, eval_p, _ = plan_bootstrap_permutations(
        n_union, n_fit, n_eval, 1,
        rng_seed=99,
        fit_top_k=10,
        eval_extrap_k=count_extrapolated_eval_points(n_fit, n_eval),
        union_values_detached=union.detach(),
        grad_mode="fit",
    )

    loss_boot, _ = differentiable_ols_extrapolated_primal_loss_bootstrap(
        union, fit_p, eval_p,
        n_fit=n_fit, n_eval=n_eval, fit_top_k=10,
        score_mode="lm_gumbel",
        grad_mode="fit",
        extrapolated_weighting="rank_uniform",
        loss_space="original",
    )

    # Reproduce the same permutation manually.
    union2 = union.detach().clone().requires_grad_(True)
    fit_vals = union2.index_select(0, fit_p[0])
    eval_vals = union2.index_select(0, eval_p[0])
    _, fit_topk = torch.topk(fit_vals.detach(), 10)
    eval_extrap_k = count_extrapolated_eval_points(n_fit, n_eval)
    eval_topk = torch.argsort(eval_vals.detach(), descending=True)[:eval_extrap_k]
    loss_direct, _ = differentiable_ols_extrapolated_primal_loss(
        fit_vals, fit_topk, n_fit,
        eval_vals, eval_topk, n_eval,
        score_mode="lm_gumbel",
        grad_mode="fit",
        extrapolated_weighting="rank_uniform",
        loss_space="original",
    )

    assert torch.allclose(loss_boot.detach(), loss_direct.detach(), atol=1e-6)


def test_variance_reduction_loss():
    """The bootstrap-averaged loss should have lower variance across RNG seeds
    than the single-draw loss for the same union values.
    """
    n_fit, n_eval = 30, 60
    n_union = n_fit + n_eval
    losses_n1 = []
    losses_n64 = []
    union_vals = _make_union(n_union, seed=2026, requires_grad=False)
    eval_extrap_k = count_extrapolated_eval_points(n_fit, n_eval)
    for seed in range(8):
        for n_perm, target in ((1, losses_n1), (64, losses_n64)):
            u = union_vals.detach().clone().requires_grad_(True)
            fit_p, eval_p, _ = plan_bootstrap_permutations(
                n_union, n_fit, n_eval, n_perm,
                rng_seed=seed * 31 + 1,
                fit_top_k=10,
                eval_extrap_k=eval_extrap_k,
                union_values_detached=u.detach(),
                grad_mode="fit",
            )
            loss, _ = differentiable_ols_extrapolated_primal_loss_bootstrap(
                u, fit_p, eval_p,
                n_fit=n_fit, n_eval=n_eval, fit_top_k=10,
                score_mode="lm_gumbel",
                grad_mode="fit",
                extrapolated_weighting="rank_uniform",
                loss_space="original",
            )
            target.append(loss.item())

    var_1 = float(torch.tensor(losses_n1).var(unbiased=False))
    var_64 = float(torch.tensor(losses_n64).var(unbiased=False))
    assert var_64 < 0.25 * var_1, (
        f"expected n=64 variance to be < 0.25x of n=1; got n=1 var={var_1:.4g}, "
        f"n=64 var={var_64:.4g}"
    )


def test_iog_plumbing():
    n_fit, n_eval = 30, 60
    n_union = n_fit + n_eval
    union = _make_union(n_union, seed=5)
    fit_p, eval_p, _ = plan_bootstrap_permutations(
        n_union, n_fit, n_eval, 4,
        rng_seed=17,
        fit_top_k=10,
        eval_extrap_k=count_extrapolated_eval_points(n_fit, n_eval),
        union_values_detached=union.detach(),
        grad_mode="fit",
    )
    _, info = differentiable_ols_extrapolated_primal_loss_bootstrap(
        union, fit_p, eval_p,
        n_fit=n_fit, n_eval=n_eval, fit_top_k=10,
        score_mode="lm_gumbel",
        grad_mode="fit",
        extrapolated_weighting="rank_uniform",
        loss_space="original",
        improving_only_grad=True,
    )
    for key in ("improving_only_deploy_frac", "improving_only_fit_frac"):
        assert key in info, f"missing IOG telemetry key {key}"
        v = float(info[key].item())
        assert 0.0 <= v <= 1.0, f"{key} out of range: {v}"


def test_needed_grad_idx_only_includes_fit_when_grad_mode_fit():
    n_fit, n_eval = 30, 60
    n_union = n_fit + n_eval
    union_vals = _make_union(n_union, seed=3, requires_grad=False)
    fit_p, eval_p, needed_fit = plan_bootstrap_permutations(
        n_union, n_fit, n_eval, 8,
        rng_seed=1,
        fit_top_k=5,
        eval_extrap_k=count_extrapolated_eval_points(n_fit, n_eval),
        union_values_detached=union_vals,
        grad_mode="fit",
    )
    _, _, needed_both = plan_bootstrap_permutations(
        n_union, n_fit, n_eval, 8,
        rng_seed=1,
        fit_top_k=5,
        eval_extrap_k=count_extrapolated_eval_points(n_fit, n_eval),
        union_values_detached=union_vals,
        grad_mode="both",
    )
    # grad_mode="both" should include at least as many indices as "fit".
    assert needed_fit.numel() <= needed_both.numel()
    # And needed_fit should be a subset of needed_both.
    fit_set = set(needed_fit.tolist())
    both_set = set(needed_both.tolist())
    assert fit_set.issubset(both_set)


_TESTS = [
    test_smoke_shape_and_grad,
    test_n_perm_1_matches_direct_call,
    test_variance_reduction_loss,
    test_iog_plumbing,
    test_needed_grad_idx_only_includes_fit_when_grad_mode_fit,
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
