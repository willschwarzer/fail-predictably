"""Unit tests for the sampling-based (REINFORCE) forecastability estimators.

Covers: (1) startup validation rejects the degenerate n=1 + rollout_mean
combination; (2) the optimal_value baseline yields a nonzero gradient at
n=1; (3) the rollout_mean advantage helper is bit-identical to the legacy
inline expression; (4) estimator B's leave-one-out baseline is independent
of the target task's current sampled regret; (5) the batched rollout
sampler matches exact policy evaluation and the loop sampler
distributionally; (6) the mean of many plugin-estimator gradients aligns
with the exact-path gradient of the same bootstrap forecast loss.

Run directly:
    python -m tests.test_reinforce_estimators
"""
from __future__ import annotations

import random

import torch

from forecast_experiment.rlquant_gridworld import (
    DensePolicyNet,
    GridworldGeneratorConfig,
    build_task_from_layout,
    evaluate_dense_policy_exact,
    generate_layout,
    prepare_task_batch,
    sample_policy_rollouts,
    sample_policy_rollouts_batched,
)
from forecast_experiment.train_rlquant import (
    RLQuantConfig,
    _batch_loo_baseline_value,
    _meta_primal_loss_bootstrap,
    _reinforce_advantages,
    _reinforce_manifest_losses,
    train_rlquant,
)

_CFG = GridworldGeneratorConfig(
    rows=8,
    cols=8,
    horizon=10,
    gamma=1.0,
    step_cost=-0.01,
    goal_reward=1.0,
    catastrophe_penalty=-1.0,
    min_start_goal_distance=5,
)


def _make_tasks(n: int, seed0: int, rare_every: int = 6) -> list:
    tasks: list = []
    offset = 0
    while len(tasks) < n:
        offset += 1
        if offset > 50 * n:
            raise AssertionError("layout generation stalled")
        severity = 0.9 if (len(tasks) % rare_every == rare_every - 1) else 0.0
        layout = generate_layout(
            "trap_open_room",
            severity,
            cfg=_CFG,
            backbone_seed=seed0 + offset,
            overlay_seed=seed0 + 9999 + offset,
        )
        if layout is None:
            continue
        tasks.append(
            build_task_from_layout(layout, split_name="test", task_id=f"t{seed0}_{offset}")
        )
    return tasks


def _make_model(tasks) -> DensePolicyNet:
    return DensePolicyNet(
        tasks[0].feature_tensor.shape[0],
        tasks[0].task_vector.shape[0],
        8,
        8,
        10,
        hidden_channels=8,
        task_hidden_dim=16,
    )


def _meta_cfg(**overrides) -> RLQuantConfig:
    base = dict(
        horizon=10,
        ols_top_k=3,
        meta_loss_mode="ols_primal_extrapolated",
        bootstrap_permutations=1,
        grad_mode="both",
        improving_only_grad=True,
        grad_estimator="reinforce",
        plotting_position="weibull",
        extrapolated_weighting="deploy_log_uniform",
    )
    base.update(overrides)
    return RLQuantConfig(**base)


def test_n1_rollout_mean_validation_raises() -> None:
    cfg = RLQuantConfig(
        grad_estimator="reinforce",
        reinforce_rollouts_per_task=1,
        outdir="/nonexistent/should_never_be_created",
    )
    try:
        train_rlquant(cfg)
    except ValueError as exc:
        assert "identically-zero" in str(exc), str(exc)
    else:
        raise AssertionError("expected ValueError for n=1 + rollout_mean")


def test_rollout_mean_advantages_bit_exact() -> None:
    torch.manual_seed(3)
    returns = torch.randn(7, 5)
    cfg = RLQuantConfig(reinforce_baseline="rollout_mean")
    adv = _reinforce_advantages(returns, cfg=cfg)
    legacy = returns - returns.mean(dim=1, keepdim=True)
    assert torch.equal(adv, legacy)


def test_n1_rollout_mean_advantage_is_identically_zero() -> None:
    torch.manual_seed(4)
    returns = torch.randn(9, 1)
    cfg = RLQuantConfig(reinforce_baseline="rollout_mean")
    adv = _reinforce_advantages(returns, cfg=cfg)
    assert bool((adv == 0).all())


def test_optimal_value_baseline_nonzero_grad_at_n1() -> None:
    torch.manual_seed(5)
    random.seed(5)
    tasks = _make_tasks(6, 300)
    model = _make_model(tasks)
    batch = prepare_task_batch(tasks, "cpu")
    dense_logits = model(batch["features"], batch["task_vectors"])
    returns, logprobs = sample_policy_rollouts_batched(
        tasks, dense_logits, batch, n_rollouts=1, device="cpu"
    )
    cfg = RLQuantConfig(reinforce_baseline="optimal_value")
    adv = _reinforce_advantages(returns, cfg=cfg, optimal_values=batch["optimal_values"])
    loss = -(adv * logprobs).mean()
    loss.backward()
    grad_l1 = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert grad_l1 > 0.0, "optimal_value baseline should give nonzero gradient at n=1"


def test_ema_baseline_updates_after_advantage() -> None:
    cfg = RLQuantConfig(reinforce_baseline="ema", reinforce_ema_decay=0.5)
    ema: dict[str, float] = {"a": 1.0}
    returns = torch.tensor([[3.0, 5.0], [2.0, 2.0]])
    adv = _reinforce_advantages(
        returns, cfg=cfg, task_keys=["a", "b"], ema_state=ema
    )
    # Task 'a': baseline is the PRIOR ema (1.0), not the updated one.
    assert torch.allclose(adv[0], torch.tensor([2.0, 4.0]))
    # Task 'b': first visit falls back to this step's rollout mean (2.0).
    assert torch.allclose(adv[1], torch.tensor([0.0, 0.0]))
    # State updated afterward: a -> 0.5*1 + 0.5*4 = 2.5, b -> 2.0.
    assert abs(ema["a"] - 2.5) < 1e-9 and abs(ema["b"] - 2.0) < 1e-9


def test_batch_loo_baseline_independent_of_target_value() -> None:
    torch.manual_seed(6)
    cfg = _meta_cfg()
    n_fit, n_deploy = 8, 24
    values = torch.rand(n_fit + n_deploy) * 2.0
    u_idx = 5
    kwargs = dict(fit_n_total=n_fit, deploy_n_total=n_deploy, cfg=cfg, rng_seed=777)
    # EMA branch.
    ema = {"k5": 0.37}
    b1 = _batch_loo_baseline_value(values, u_idx, "k5", ema, **kwargs)
    mutated = values.clone()
    mutated[u_idx] = 99.0
    b2 = _batch_loo_baseline_value(mutated, u_idx, "k5", ema, **kwargs)
    assert b1 == b2, (b1, b2)
    # First-visit fallback branch (mean of the OTHER tasks) must also be
    # invariant to the target's own value.
    b3 = _batch_loo_baseline_value(values, u_idx, "unseen", {}, **kwargs)
    b4 = _batch_loo_baseline_value(mutated, u_idx, "unseen", {}, **kwargs)
    assert b3 == b4, (b3, b4)


def test_batched_sampler_matches_exact_and_loop() -> None:
    torch.manual_seed(7)
    random.seed(7)
    tasks = _make_tasks(6, 700)
    model = _make_model(tasks)
    batch = prepare_task_batch(tasks, "cpu")
    with torch.no_grad():
        dense_logits = model(batch["features"], batch["task_vectors"])
        exact = evaluate_dense_policy_exact(dense_logits, batch, horizon=10, gamma=1.0)
        n = 600
        b_ret, b_lp = sample_policy_rollouts_batched(
            tasks, dense_logits, batch, n_rollouts=n, device="cpu"
        )
        l_ret, l_lp = sample_policy_rollouts(tasks, dense_logits, n_rollouts=200, device="cpu")
    # Mean return within 5 standard errors of the exact value.
    se = b_ret.std(dim=1) / (n**0.5)
    diff = (b_ret.mean(dim=1) - exact).abs()
    assert bool((diff <= 5.0 * se + 1e-3).all()), (diff, se)
    # Log-prob sums agree distributionally between the two samplers.
    lp_diff = (b_lp.mean(dim=1) - l_lp.mean(dim=1)).abs()
    lp_scale = b_lp.std(dim=1) / (n**0.5) + l_lp.std(dim=1) / (200**0.5)
    assert bool((lp_diff <= 5.0 * lp_scale + 1e-2).all()), (lp_diff, lp_scale)


def _exact_bootstrap_grad(model, fit_tasks, deploy_tasks, cfg, rng_seed) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    union = list(fit_tasks) + list(deploy_tasks)
    batch = prepare_task_batch(union, "cpu")
    dense_logits = model(batch["features"], batch["task_vectors"])
    returns = evaluate_dense_policy_exact(dense_logits, batch, horizon=cfg.horizon, gamma=cfg.gamma)
    regrets = batch["optimal_values"] - returns
    loss, _ = _meta_primal_loss_bootstrap(
        regrets, len(fit_tasks), len(deploy_tasks), cfg=cfg, rng_seed=rng_seed
    )
    loss.backward()
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in model.parameters()
        ]
    ).clone()


def test_plugin_gradient_aligns_with_exact() -> None:
    torch.manual_seed(8)
    random.seed(8)
    fit_tasks = _make_tasks(10, 1100, rare_every=5)
    deploy_tasks = _make_tasks(30, 3100, rare_every=5)
    model = _make_model(fit_tasks)
    # IOG off for the alignment test: the mask is a deliberate gradient bias
    # shared by both paths, but its threshold flips under sampling noise,
    # which would conflate mask churn with estimator quality.
    cfg = _meta_cfg(
        improving_only_grad=False,
        reinforce_rollouts_per_task=32,
        reinforce_baseline="optimal_value",
    )
    rng_seed = 4242
    g_exact = _exact_bootstrap_grad(model, fit_tasks, deploy_tasks, cfg, rng_seed)
    draws = 40
    acc = torch.zeros_like(g_exact)
    for _ in range(draws):
        model.zero_grad(set_to_none=True)
        result = _reinforce_manifest_losses(
            model,
            fit_tasks,
            deploy_tasks,
            cfg=cfg,
            rng_seed=rng_seed,
            device="cpu",
            ema_return_state={},
            ema_regret_state={},
        )
        assert result is not None
        _, pg_loss, _, _ = result
        pg_loss.backward()
        acc += torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in model.parameters()
            ]
        )
    mean_est = acc / draws
    cos = torch.nn.functional.cosine_similarity(
        mean_est.unsqueeze(0), g_exact.unsqueeze(0)
    ).item()
    assert cos > 0.5, f"mean plugin gradient misaligned with exact gradient: cos={cos:.3f}"


def main() -> None:
    test_n1_rollout_mean_validation_raises()
    print("test_n1_rollout_mean_validation_raises passed")
    test_rollout_mean_advantages_bit_exact()
    print("test_rollout_mean_advantages_bit_exact passed")
    test_n1_rollout_mean_advantage_is_identically_zero()
    print("test_n1_rollout_mean_advantage_is_identically_zero passed")
    test_optimal_value_baseline_nonzero_grad_at_n1()
    print("test_optimal_value_baseline_nonzero_grad_at_n1 passed")
    test_ema_baseline_updates_after_advantage()
    print("test_ema_baseline_updates_after_advantage passed")
    test_batch_loo_baseline_independent_of_target_value()
    print("test_batch_loo_baseline_independent_of_target_value passed")
    test_batched_sampler_matches_exact_and_loop()
    print("test_batched_sampler_matches_exact_and_loop passed")
    test_plugin_gradient_aligns_with_exact()
    print("test_plugin_gradient_aligns_with_exact passed")
    print("all reinforce estimator tests passed")


if __name__ == "__main__":
    main()
