from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch

OLS_SXX_TOL = 1e-20


def select_top_k_indices(values: torch.Tensor, k: int) -> torch.LongTensor:
    """Return detached indices of the top-k largest values."""
    k = max(0, min(int(k), int(values.shape[0])))
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=values.device)
    _, idx = torch.topk(values.detach(), k)
    return idx


def count_extrapolated_eval_points(
    fit_n_total: int,
    eval_n_total: int,
    *,
    plotting: str = "gringorten",
    fit_rank_cutoff: int = 1,
) -> int:
    """Count eval points whose plotting positions lie beyond the fit regime."""
    fit_n_total = int(fit_n_total)
    eval_n_total = int(eval_n_total)
    if fit_n_total <= 0 or eval_n_total <= 0:
        return 0

    fit_rank_cutoff = max(1, min(int(fit_rank_cutoff), fit_n_total))
    threshold = _plotting_position_scalar(fit_rank_cutoff, fit_n_total, plotting)
    plotting = plotting.lower()
    if plotting == "gringorten":
        raw_limit = threshold * (float(eval_n_total) + 0.12) + 0.44
    elif plotting == "weibull":
        raw_limit = threshold * (float(eval_n_total) + 1.0)
    elif plotting == "empirical":
        raw_limit = threshold * float(eval_n_total)
    else:
        raise ValueError("plotting must be one of {'gringorten', 'weibull', 'empirical'}")

    count = max(0, min(eval_n_total, int(math.floor(raw_limit + 1e-12))))
    tol = 1e-12
    while count > 0 and _plotting_position_scalar(count, eval_n_total, plotting) > threshold + tol:
        count -= 1
    while count < eval_n_total and _plotting_position_scalar(count + 1, eval_n_total, plotting) <= threshold + tol:
        count += 1
    return count


def _plotting_positions_tensor(
    k: int,
    n_total: int,
    plotting: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    j = torch.arange(1, k + 1, dtype=dtype, device=device)
    plotting = plotting.lower()
    if plotting == "gringorten":
        return (j - 0.44) / (float(n_total) + 0.12)
    if plotting == "weibull":
        return j / (float(n_total) + 1.0)
    if plotting == "empirical":
        return j / float(n_total)
    raise ValueError("plotting must be one of {'gringorten', 'weibull', 'empirical'}")


def _plotting_position_scalar(rank: int, n_total: int, plotting: str) -> float:
    plotting = plotting.lower()
    if plotting == "gringorten":
        return (float(rank) - 0.44) / (float(n_total) + 0.12)
    if plotting == "weibull":
        return float(rank) / (float(n_total) + 1.0)
    if plotting == "empirical":
        return float(rank) / float(n_total)
    raise ValueError("plotting must be one of {'gringorten', 'weibull', 'empirical'}")


def _plotting_positions_list(k: int, n_total: int, plotting: str) -> list[float]:
    out: list[float] = []
    for j in range(1, k + 1):
        val = _plotting_position_scalar(j, n_total, plotting)
        out.append(min(max(val, 1e-300), 1.0))
    return out


def _extrapolated_rank_weights(
    k: int,
    n_total: int,
    plotting: str,
    *,
    weighting: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized weights over extrapolated ranks.

    `rank_uniform` reproduces the current equal-weight-per-rank behavior.
    The deploy-size options correspond to sampling a deployment size prior and
    snapping to the nearest deploy plotting position (equivalently, the nearest
    survival level), which induces a discrete weight on each rank.
    `worst_rank` puts all the weight on rank 1, so the trained objective is the
    same functional as the reported held-out metric (squared log10 error at the
    worst observed deploy point).
    """
    if k <= 0:
        return torch.zeros((0,), dtype=dtype, device=device)
    if k == 1:
        return torch.ones((1,), dtype=dtype, device=device)

    weighting = weighting.lower()
    if weighting == "rank_uniform":
        return torch.full((k,), 1.0 / float(k), dtype=dtype, device=device)
    if weighting == "worst_rank":
        w = torch.zeros((k,), dtype=dtype, device=device)
        w[0] = 1.0
        return w

    u = _plotting_positions_tensor(
        k,
        n_total,
        plotting,
        device=device,
        dtype=dtype,
    )
    mids = 0.5 * (u[:-1] + u[1:])
    lower = torch.empty_like(u)
    upper = torch.empty_like(u)
    lower[0] = u[0]
    lower[1:] = mids
    upper[:-1] = mids
    upper[-1] = u[-1]

    if weighting == "deploy_log_uniform":
        raw = torch.log(upper / lower)
    elif weighting == "deploy_uniform":
        raw = (1.0 / lower) - (1.0 / upper)
    else:
        raise ValueError(
            "extrapolated_weighting must be one of "
            "{'rank_uniform', 'worst_rank', 'deploy_log_uniform', 'deploy_uniform'}"
        )

    raw = raw.to(dtype=torch.float32)
    total = raw.sum()
    if not torch.isfinite(total) or total.item() <= 0:
        return torch.full((k,), 1.0 / float(k), dtype=dtype, device=device)
    return (raw / total).to(dtype=dtype)


def _transform_tensor(
    values: torch.Tensor,
    *,
    score_mode: str,
    eps: float,
) -> torch.Tensor:
    score_mode = score_mode.lower()
    if score_mode in {"identity", "raw", "raw_regret"}:
        return values.to(torch.float32)
    if score_mode in {"log", "log_regret"}:
        return torch.log(values.clamp_min(eps)).to(torch.float32)
    if score_mode in {"gumbel_log_prob", "lm_gumbel"}:
        return -torch.log((-values).clamp_min(eps)).to(torch.float32)
    if score_mode == "gumbel_prob":
        probs = values.clamp(min=eps, max=1.0 - 1e-16)
        return (-torch.log(-torch.log(probs))).to(torch.float32)
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def _transform_scalar(value: float, *, score_mode: str, eps: float) -> float:
    score_mode = score_mode.lower()
    if score_mode in {"identity", "raw", "raw_regret"}:
        return float(value)
    if score_mode in {"log", "log_regret"}:
        return math.log(max(float(value), eps))
    if score_mode in {"gumbel_log_prob", "lm_gumbel"}:
        return -math.log(max(-float(value), eps))
    if score_mode == "gumbel_prob":
        p = min(max(float(value), eps), 1.0 - 1e-16)
        return -math.log(-math.log(p))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def _inverse_transform_tensor(
    transformed: torch.Tensor,
    *,
    score_mode: str,
) -> torch.Tensor:
    score_mode = score_mode.lower()
    if score_mode in {"identity", "raw", "raw_regret"}:
        return transformed
    if score_mode in {"log", "log_regret"}:
        return torch.exp(transformed)
    if score_mode in {"gumbel_log_prob", "lm_gumbel"}:
        return -torch.exp(-transformed)
    if score_mode == "gumbel_prob":
        return torch.exp(-torch.exp(-transformed))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def _inverse_transform_scalar(value: float, *, score_mode: str) -> float:
    score_mode = score_mode.lower()
    if score_mode in {"identity", "raw", "raw_regret"}:
        return float(value)
    if score_mode in {"log", "log_regret"}:
        return math.exp(float(value))
    if score_mode in {"gumbel_log_prob", "lm_gumbel"}:
        return -math.exp(-float(value))
    if score_mode == "gumbel_prob":
        return math.exp(-math.exp(-float(value)))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def differentiable_ols_design(
    values: torch.Tensor,
    selected_indices: torch.LongTensor,
    n_total: int,
    *,
    plotting: str = "gringorten",
    score_mode: str = "identity",
    score_eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the OLS design matrix X and targets y for selected points."""
    k = int(selected_indices.shape[0])
    if k == 0:
        device = values.device
        nan = torch.full((0, 2), float("nan"), dtype=torch.float32, device=device)
        return nan, torch.full((0,), float("nan"), dtype=torch.float32, device=device)
    selected = values[selected_indices]
    scores = _transform_tensor(selected, score_mode=score_mode, eps=score_eps)
    y = torch.log(
        _plotting_positions_tensor(
            k,
            n_total,
            plotting,
            device=scores.device,
            dtype=torch.float32,
        )
    )
    x = torch.stack([torch.ones_like(scores), scores], dim=-1)
    return x, y


def _ols_beta_from_design(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_scores = x[:, 1]
    x_mean = x_scores.mean()
    y_mean = y.mean()
    x_centered = x_scores - x_mean
    y_centered = y - y_mean
    sxx = (x_centered * x_centered).sum()
    sxy = (x_centered * y_centered).sum()
    # Use epsilon guard instead of branch to keep gradient graph alive
    b = sxy / (sxx + 1e-30)
    a = y_mean - b * x_mean
    beta = torch.stack([a, b])
    return beta, sxx, sxy


def differentiable_ols_coefficients(
    values: torch.Tensor,
    top_k_indices: torch.LongTensor,
    n_total: int,
    *,
    plotting: str = "gringorten",
    score_mode: str = "identity",
    score_eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable OLS tail fit on transformed top-k scores."""
    k = int(top_k_indices.shape[0])
    if k == 0:
        device = values.device
        nan = torch.full((), float("nan"), dtype=torch.float32, device=device)
        return nan, nan

    x, y = differentiable_ols_design(
        values,
        top_k_indices,
        n_total,
        plotting=plotting,
        score_mode=score_mode,
        score_eps=score_eps,
    )
    beta, _sxx, _sxy = _ols_beta_from_design(x, y)
    return beta[0], beta[1]


def differentiable_ols_forecast(
    values: torch.Tensor,
    top_k_indices: torch.LongTensor,
    n_total: int,
    deploy_queries: int | list[int],
    *,
    plotting: str = "gringorten",
    score_mode: str = "identity",
    score_eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable OLS worst-case forecast in the original score domain."""
    k = int(top_k_indices.shape[0])
    if k == 0:
        device = values.device
        scalar = not isinstance(deploy_queries, list)
        nan = torch.full((), float("nan"), dtype=torch.float32, device=device)
        return nan if scalar else nan.unsqueeze(0).expand(len(deploy_queries))

    x, y = differentiable_ols_design(
        values,
        top_k_indices,
        n_total,
        plotting=plotting,
        score_mode=score_mode,
        score_eps=score_eps,
    )
    scores = x[:, 1]
    beta, sxx, _sxy = _ols_beta_from_design(x, y)
    a, b = beta[0], beta[1]

    scalar = not isinstance(deploy_queries, list)
    ms = [deploy_queries] if scalar else deploy_queries
    y_stars = torch.tensor(
        [math.log(1.0 / max(int(m), 1)) for m in ms],
        dtype=torch.float32,
        device=scores.device,
    )

    if sxx.item() < OLS_SXX_TOL:
        forecast = _inverse_transform_tensor(scores[0], score_mode=score_mode)
        return forecast if scalar else forecast.unsqueeze(0).expand(len(ms))

    x_stars = (y_stars - a) / b
    forecast = _inverse_transform_tensor(x_stars, score_mode=score_mode)
    return forecast.squeeze(0) if scalar else forecast


def differentiable_ols_primal_loss(
    fit_values: torch.Tensor,
    fit_indices: torch.LongTensor,
    fit_n_total: int,
    eval_values: torch.Tensor,
    eval_indices: torch.LongTensor,
    eval_n_total: int,
    *,
    plotting: str = "gringorten",
    score_mode: str = "identity",
    score_eps: float = 1e-8,
    grad_mode: str = "both",
    use_weighted_decomposition: bool = False,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact deployment-set MSE for the fit OLS line, with optional decomposition."""
    fit_x, fit_y = differentiable_ols_design(
        fit_values,
        fit_indices,
        fit_n_total,
        plotting=plotting,
        score_mode=score_mode,
        score_eps=score_eps,
    )
    eval_x, eval_y = differentiable_ols_design(
        eval_values,
        eval_indices,
        eval_n_total,
        plotting=plotting,
        score_mode=score_mode,
        score_eps=score_eps,
    )
    if fit_x.shape[0] == 0 or eval_x.shape[0] == 0:
        device = fit_values.device
        nan = torch.full((), float("nan"), dtype=torch.float32, device=device)
        return nan, {
            "fit_a": nan,
            "fit_b": nan,
            "eval_a": nan,
            "eval_b": nan,
            "weighted_coeff_mse": nan,
            "eval_residual_mse": nan,
            "primal_mse": nan,
        }

    beta_fit, _fit_sxx, _fit_sxy = _ols_beta_from_design(fit_x, fit_y)
    beta_eval, _eval_sxx, _eval_sxy = _ols_beta_from_design(eval_x, eval_y)

    n_eval = float(eval_x.shape[0])
    weighted_coeff_mse = ((beta_fit - beta_eval) @ (eval_x.transpose(0, 1) @ eval_x) @ (beta_fit - beta_eval)) / n_eval
    eval_residual_mse = ((eval_x @ beta_eval - eval_y) ** 2).mean()

    if grad_mode == "fit":
        eval_x = eval_x.detach()
        eval_y = eval_y.detach()
        beta_eval = beta_eval.detach()
        weighted_coeff_mse = weighted_coeff_mse.detach() if not use_weighted_decomposition else (
            ((beta_fit - beta_eval) @ (eval_x.transpose(0, 1) @ eval_x) @ (beta_fit - beta_eval)) / n_eval
        )
        eval_residual_mse = eval_residual_mse.detach()
        beta_fit_for_loss = beta_fit
    elif grad_mode == "eval":
        beta_fit_for_loss = beta_fit.detach()
    else:
        beta_fit_for_loss = beta_fit

    primal_mse = ((eval_x @ beta_fit_for_loss - eval_y) ** 2).mean()
    if use_weighted_decomposition:
        diff = beta_fit_for_loss - beta_eval
        weighted_coeff_mse = (diff @ (eval_x.transpose(0, 1) @ eval_x) @ diff) / n_eval
        loss = weighted_coeff_mse + eval_residual_mse
    else:
        loss = primal_mse
    return loss, {
        "fit_a": beta_fit[0],
        "fit_b": beta_fit[1],
        "eval_a": beta_eval[0],
        "eval_b": beta_eval[1],
        "weighted_coeff_mse": weighted_coeff_mse,
        "eval_residual_mse": eval_residual_mse,
        "primal_mse": primal_mse,
    }


def _anchor_worsening_coef(
    values_selected: torch.Tensor,
    anchor_selected: torch.Tensor | None,
    margin: float,
    floor_coef: float,
) -> torch.Tensor:
    """Per-point coefficient on the *worsening* gradient of a selected point.

    A point that sits at (or above) its anchor -- the score it had before
    training started -- keeps the hard block: its worsening gradient is scaled
    by ``floor_coef`` (0 in the canonical recipe). A point already suppressed
    ``margin`` or more below its anchor is free to move back up, coefficient 1.
    In between the coefficient ramps linearly in the suppression depth
    ``anchor - value``, which is why the gate is smooth: nothing about a point's
    treatment changes discontinuously as it crosses its starting score.

    With no anchor (or a non-positive margin) this reduces to the existing
    scalar behaviour, so the canonical mask is bit-identical.
    """
    if anchor_selected is None or margin <= 0.0:
        return torch.full_like(values_selected, float(floor_coef))
    depth = (anchor_selected.to(values_selected.dtype) - values_selected.detach())
    gate = (depth / float(margin)).clamp(0.0, 1.0)
    if floor_coef > 0.0:
        gate = gate.clamp_min(float(floor_coef))
    return gate


def differentiable_ols_extrapolated_primal_loss(
    fit_values: torch.Tensor,
    fit_indices: torch.LongTensor,
    fit_n_total: int,
    eval_values: torch.Tensor,
    eval_indices: torch.LongTensor,
    eval_n_total: int,
    *,
    plotting: str = "gringorten",
    score_mode: str = "identity",
    score_eps: float = 1e-8,
    grad_mode: str = "both",
    extrapolated_weighting: str = "rank_uniform",
    loss_space: str = "score",
    improving_only_grad: bool = False,
    iog_scope: str = "both",
    iog_worsening_coef: float = 0.0,
    iog_fit_mask: str = "closed_form",
    fit_anchor: torch.Tensor | None = None,
    eval_anchor: torch.Tensor | None = None,
    anchor_margin: float = 0.0,
    calibration_alpha: float | None = None,
    calibration_beta: float | None = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct primal loss on extrapolated eval points in score/failureness space.

    We fit the OLS line in ``y = a + b x`` form on the fit tail, where ``x`` is
    transformed failureness and ``y`` is log survival. For each extrapolated eval
    plotting position ``y_j``, we invert the fit line to predict the score
    ``x_hat_j`` at that deployment size and penalize squared error against the
    observed eval score ``x_j``.

    This keeps the deploy-size / plotting-position fixed and measures error in
    failureness, which is the directly aligned forecasting quantity. The current
    implementation still averages equally over the selected extrapolated ranks;
    if we later care about some deployment sizes more than others, that
    weighting should become configurable.

    Calibration-aware training. When ``calibration_alpha`` is given, the
    forecast is affinely corrected, ``Q -> alpha * Q + beta``, before the
    squared error against the deploy actual, mirroring the post-hoc affine
    calibrator that is applied at deployment. ``alpha`` and ``beta`` are treated
    as constants (stop-gradient): the caller fits them on OTHER task-sets
    (leave-one-group-out) so they carry no gradient back to the model here. The
    correction is applied in whichever domain ``loss_space`` selects -- score
    space for an identity-transform setting (e.g. RL regrets), original/value
    space for the LM log-prob setting -- so it matches the domain the deployed
    calibrator is fit in. The improving-only deploy mask fires on the
    *calibrated* under-prediction. ``alpha > 0`` is assumed (the caller's
    shrinkage keeps it near 1); the closed-form fit-side mask sign is unchanged
    by a positive scale.
    """
    fit_x, fit_y = differentiable_ols_design(
        fit_values,
        fit_indices,
        fit_n_total,
        plotting=plotting,
        score_mode=score_mode,
        score_eps=score_eps,
    )
    eval_x, eval_y = differentiable_ols_design(
        eval_values,
        eval_indices,
        eval_n_total,
        plotting=plotting,
        score_mode=score_mode,
        score_eps=score_eps,
    )
    if fit_x.shape[0] == 0 or eval_x.shape[0] == 0:
        device = fit_values.device
        nan = torch.full((), float("nan"), dtype=torch.float32, device=device)
        return nan, {
            "fit_a": nan,
            "fit_b": nan,
            "primal_mse": nan,
            "score_mse": nan,
            "unweighted_score_mse": nan,
            "weighted_score_mae": nan,
            "unweighted_score_mae": nan,
            "weighted_original_mse": nan,
            "unweighted_original_mse": nan,
            "weighted_original_mae": nan,
            "unweighted_original_mae": nan,
            "eval_points_used": nan,
            "fit_regime_log_survival": nan,
            "eval_max_log_survival": nan,
            "eval_min_log_survival": nan,
            "predicted_max_score": nan,
            "predicted_min_score": nan,
            "eval_max_score": nan,
            "eval_min_score": nan,
            "predicted_max_value": nan,
            "predicted_min_value": nan,
            "eval_max_value": nan,
            "eval_min_value": nan,
            "cal_pred_worst": nan,
            "cal_actual_worst": nan,
            "fit_x_spread": nan,
            "eval_x_spread": nan,
            "fit_v_spread": nan,
            "eval_v_spread": nan,
        }

    fit_scores = fit_x[:, 1]
    eval_scores = eval_x[:, 1]
    eval_selected_values = eval_values[eval_indices]
    # Anchors are per-record over the whole partition; the mask only ever sees
    # the selected points, so slice them the same way the values were sliced.
    fit_anchor_selected = None if fit_anchor is None else fit_anchor[fit_indices]
    deploy_anchor_selected = None if eval_anchor is None else eval_anchor[eval_indices]

    # Calibration-aware: constant (stop-gradient) affine correction of the
    # forecast, applied in the loss-space domain (see docstring).
    cal_active = calibration_alpha is not None
    cal_a = float(calibration_alpha) if cal_active else 1.0
    cal_b = float(calibration_beta) if (cal_active and calibration_beta is not None) else 0.0
    loss_space_l = loss_space.lower()
    if loss_space_l not in {"score", "original", "log_prob"}:
        raise ValueError(f"Unsupported loss_space: {loss_space}")
    cal_in_score_space = loss_space_l == "score"

    weights = _extrapolated_rank_weights(
        int(eval_x.shape[0]),
        eval_n_total,
        plotting,
        weighting=extrapolated_weighting,
        device=eval_scores.device,
        dtype=eval_scores.dtype,
    )

    # --- Improving-only gradient masking ---
    # When enabled, block gradients that would push individual scores upward
    # (worsen performance). Deploy-side and fit-side masking are independent:
    #
    # - Deploy mask: detach only the eval-score term for residuals whose local
    #   gradient would increase the eval score. The residual still contributes
    #   to fit-side gradients through the predicted score.
    # - Fit mask: closed-form dL/dx_i sign for fit scores; masked fit points are
    #   treated as constants via a straight-through detach.
    #
    # ``iog_worsening_coef`` (c) softens the mask: blocked points keep a fraction
    # c of their gradient instead of none. The straight-through form
    # ``m * x + (1 - m) * x.detach()`` passes gradient exactly m, so setting the
    # blocked entries to c scales their gradient by c. c=0 is the hard mask (the
    # canonical recipe, and the default); c=1 disables masking entirely.
    if improving_only_grad:
        if iog_scope not in {"both", "deploy_only", "fit_only"}:
            raise ValueError(f"Unsupported iog_scope: {iog_scope!r}")
        if iog_fit_mask not in {"closed_form", "exact"}:
            raise ValueError(f"Unsupported iog_fit_mask: {iog_fit_mask!r}")
        if not 0.0 <= iog_worsening_coef <= 1.0:
            raise ValueError(
                f"iog_worsening_coef must be in [0, 1], got {iog_worsening_coef!r}"
            )
        apply_deploy_mask = iog_scope in {"both", "deploy_only"}
        apply_fit_mask = iog_scope in {"both", "fit_only"}
        with torch.no_grad():
            _beta_det, _sxx_det, _ = _ols_beta_from_design(fit_x, fit_y)
            _b_val = _beta_det[1].item()
            _sxx_val = _sxx_det.item()
            if _sxx_val < OLS_SXX_TOL:
                _pred_det = fit_scores[0].expand_as(eval_scores)
            else:
                _pred_det = (eval_y - _beta_det[0]) / _beta_det[1]

            # Deploy mask: allow where observed >= predicted, i.e. where
            # gradient descent on eval_scores would push scores downward.
            # With calibration, the threshold is the CALIBRATED forecast, in
            # the loss-space domain (score for identity-transform settings,
            # value/original otherwise).
            if apply_deploy_mask:
                if cal_active:
                    if cal_in_score_space:
                        _cal_pred_det = cal_a * _pred_det + cal_b
                        deploy_mask = (eval_scores >= _cal_pred_det).float()
                    else:
                        _pred_val_det = _inverse_transform_tensor(
                            _pred_det, score_mode=score_mode
                        )
                        _cal_pred_val_det = cal_a * _pred_val_det + cal_b
                        deploy_mask = (eval_selected_values >= _cal_pred_val_det).float()
                else:
                    deploy_mask = (eval_scores >= _pred_det).float()
            else:
                deploy_mask = torch.ones_like(eval_scores)

            # Fit mask: the sign of dL/dx_i for each fit point.
            #
            # The closed form below differentiates the SCORE-space squared error.
            # When loss_space is 'original' (the LM recipe: errors in nats of
            # log-probability) that is a different functional, and the two
            # disagree in sign on about 6% of active fit points -- so the mask
            # lets through a worsening gradient it is meant to block. `exact`
            # differentiates whichever loss is actually being optimized, via one
            # tiny autograd pass over a k-element leaf, and blocks it correctly.
            # Cost is negligible; `closed_form` is kept as the default so
            # existing runs stay bit-identical.
            if not apply_fit_mask:
                fit_mask = torch.ones_like(fit_scores)
            elif iog_fit_mask == "exact":
                with torch.enable_grad():
                    _leaf = fit_scores.detach().clone().requires_grad_(True)
                    _fx_leaf = torch.stack([torch.ones_like(_leaf), _leaf], dim=-1)
                    _beta_leaf, _sxx_leaf, _ = _ols_beta_from_design(_fx_leaf, fit_y)
                    if _sxx_leaf.item() < OLS_SXX_TOL:
                        _pred_leaf = _leaf[0].expand_as(eval_scores)
                    else:
                        _pred_leaf = (eval_y - _beta_leaf[0]) / _beta_leaf[1]
                    if cal_in_score_space:
                        _pd = _pred_leaf
                        _ad = eval_scores.detach()
                    else:
                        _pd = _inverse_transform_tensor(_pred_leaf, score_mode=score_mode)
                        _ad = eval_selected_values.detach()
                    if cal_active:
                        _pd = cal_a * _pd + cal_b
                    _l = (weights * (_pd - _ad) ** 2).sum()
                    (_gfit,) = torch.autograd.grad(_l, [_leaf], allow_unused=True)
                fit_mask = (
                    torch.ones_like(fit_scores) if _gfit is None
                    else (_gfit.detach() > 0).float()
                )
            elif _sxx_val >= OLS_SXX_TOL and _b_val ** 2 * _sxx_val > 1e-30:
                _residuals = _pred_det - eval_scores
                _wr = weights * _residuals
                _fy_mean = fit_y.mean()
                _fx_mean = fit_scores.mean()
                _k = float(fit_scores.shape[0])
                _C1 = -((_wr * (eval_y - _fy_mean)).sum()) / (_b_val ** 2 * _sxx_val)
                _C2 = (2.0 / _k) * _wr.sum()
                _per_fit = (
                    2.0 * _C1 * ((fit_y - _fy_mean) - 2.0 * _b_val * (fit_scores - _fx_mean))
                    + _C2
                )
                fit_mask = (_per_fit > 0).float()
            else:
                fit_mask = torch.ones_like(fit_scores)

        # Report the HARD block fractions regardless of softening, so the logged
        # iog_* stats keep meaning "what fraction of points would be blocked".
        deploy_mask_hard = deploy_mask
        fit_mask_hard = fit_mask
        # Softening. The scalar `iog_worsening_coef` applies to every blocked
        # point alike; the anchor gate makes the coefficient per-point, so a
        # point that training has already pushed far below its starting score
        # may be allowed back up while a point at its starting score may not.
        # The two compose by taking the larger coefficient.
        deploy_coef = _anchor_worsening_coef(
            eval_selected_values, deploy_anchor_selected, anchor_margin, iog_worsening_coef,
        )
        fit_coef = _anchor_worsening_coef(
            fit_values[fit_indices], fit_anchor_selected, anchor_margin, iog_worsening_coef,
        )
        anchor_gate_active = anchor_margin > 0.0 and (
            fit_anchor is not None or eval_anchor is not None
        )
        if iog_worsening_coef > 0.0 or anchor_gate_active:
            deploy_mask = deploy_mask + (1.0 - deploy_mask) * deploy_coef
            fit_mask = fit_mask + (1.0 - fit_mask) * fit_coef

        # Straight-through: masked fit scores become constants
        if apply_fit_mask:
            fit_scores = fit_mask * fit_scores + (1.0 - fit_mask) * fit_scores.detach()
            fit_x = torch.stack([torch.ones_like(fit_scores), fit_scores], dim=-1)

    beta_fit, fit_sxx, _fit_sxy = _ols_beta_from_design(fit_x, fit_y)

    if grad_mode == "fit":
        eval_y = eval_y.detach()
        eval_scores = eval_scores.detach()
        beta_fit_for_loss = beta_fit
    elif grad_mode == "eval":
        beta_fit_for_loss = beta_fit.detach()
    else:
        beta_fit_for_loss = beta_fit

    if fit_sxx.item() < OLS_SXX_TOL:
        predicted_scores = fit_scores[0].expand_as(eval_scores)
    else:
        predicted_scores = (eval_y - beta_fit_for_loss[0]) / beta_fit_for_loss[1]

    eval_scores_for_loss = eval_scores
    eval_values_for_loss = eval_selected_values
    if improving_only_grad:
        # Keep the residual term active for fit-side gradients, but detach
        # disallowed eval points so they do not receive worsening gradients.
        eval_scores_for_loss = deploy_mask * eval_scores + (1.0 - deploy_mask) * eval_scores.detach()
        eval_values_for_loss = (
            deploy_mask * eval_selected_values
            + (1.0 - deploy_mask) * eval_selected_values.detach()
        )

    predicted_values = _inverse_transform_tensor(predicted_scores, score_mode=score_mode)

    score_sq_error = (predicted_scores - eval_scores) ** 2
    score_abs_error = (predicted_scores - eval_scores).abs()
    original_sq_error = (predicted_values - eval_selected_values) ** 2
    original_abs_error = (predicted_values - eval_selected_values).abs()

    if cal_in_score_space:
        pred_domain = predicted_scores
        actual_domain_raw = eval_scores
        actual_domain_for_loss = eval_scores_for_loss
    else:
        pred_domain = predicted_values
        actual_domain_raw = eval_selected_values
        actual_domain_for_loss = eval_values_for_loss

    if cal_active:
        cal_pred_domain = cal_a * pred_domain + cal_b
        sq_error_for_loss = (cal_pred_domain - actual_domain_for_loss) ** 2
    else:
        sq_error_for_loss = (pred_domain - actual_domain_for_loss) ** 2

    unweighted_score_mse = score_sq_error.mean()
    score_mse = (weights * score_sq_error).sum()
    unweighted_score_mae = score_abs_error.mean()
    weighted_score_mae = (weights * score_abs_error).sum()
    unweighted_original_mse = original_sq_error.mean()
    weighted_original_mse = (weights * original_sq_error).sum()
    unweighted_original_mae = original_abs_error.mean()
    weighted_original_mae = (weights * original_abs_error).sum()
    primal_loss = (weights * sq_error_for_loss).sum()
    info: dict[str, torch.Tensor] = {
        "fit_a": beta_fit[0],
        "fit_b": beta_fit[1],
        "primal_mse": primal_loss,
        "score_mse": score_mse,
        "unweighted_score_mse": unweighted_score_mse,
        "weighted_score_mae": weighted_score_mae,
        "unweighted_score_mae": unweighted_score_mae,
        "weighted_original_mse": weighted_original_mse,
        "unweighted_original_mse": unweighted_original_mse,
        "weighted_original_mae": weighted_original_mae,
        "unweighted_original_mae": unweighted_original_mae,
        "eval_points_used": torch.tensor(float(eval_x.shape[0]), dtype=torch.float32, device=fit_values.device),
        "fit_regime_log_survival": fit_y[0],
        "eval_max_log_survival": eval_y[0],
        "eval_min_log_survival": eval_y[-1],
        "predicted_max_score": predicted_scores[0],
        "predicted_min_score": predicted_scores[-1],
        "eval_max_score": eval_scores[0],
        "eval_min_score": eval_scores[-1],
        "predicted_max_value": predicted_values[0],
        "predicted_min_value": predicted_values[-1],
        "eval_max_value": eval_selected_values[0],
        "eval_min_value": eval_selected_values[-1],
        # Raw (uncalibrated) worst-rank forecast/actual in the calibration
        # domain (== loss-space domain). The caller feeds these into the
        # leave-one-group-out calibration buffer.
        "cal_pred_worst": pred_domain[0].detach(),
        "cal_actual_worst": actual_domain_raw[0].detach(),
    }
    if cal_active:
        info["calibration_alpha"] = torch.tensor(
            cal_a, dtype=torch.float32, device=fit_values.device
        )
        info["calibration_beta"] = torch.tensor(
            cal_b, dtype=torch.float32, device=fit_values.device
        )
    # Tail-shape diagnostics. Spread -> 0 together with slope -> 0 is the
    # "abolished extrapolation" signature; error falling while both stay finite
    # is "fixed extrapolation". Both spreads are computed on the pre-mask
    # scores, which are already sorted descending by construction.
    info["fit_x_spread"] = (fit_x[:, 1][0] - fit_x[:, 1][-1]).detach()
    info["eval_x_spread"] = (eval_scores[0] - eval_scores[-1]).detach()
    info["fit_v_spread"] = (
        fit_values[fit_indices][0] - fit_values[fit_indices][-1]
    ).detach()
    info["eval_v_spread"] = (
        eval_selected_values[0] - eval_selected_values[-1]
    ).detach()
    if improving_only_grad:
        info["improving_only_deploy_frac"] = deploy_mask_hard.mean()
        info["improving_only_fit_frac"] = fit_mask_hard.mean()
        # Mean gradient weight actually applied to the blocked points. 0 under
        # the canonical hard mask; the anchor gate raises it for points that
        # training has pushed well below where they started.
        info["worsening_coef_deploy_mean"] = (
            deploy_coef * (1.0 - deploy_mask_hard)
        ).sum() / (1.0 - deploy_mask_hard).sum().clamp_min(1.0)
        info["worsening_coef_fit_mean"] = (
            fit_coef * (1.0 - fit_mask_hard)
        ).sum() / (1.0 - fit_mask_hard).sum().clamp_min(1.0)
    return primal_loss, info


def plan_bootstrap_permutations(
    n_union: int,
    n_fit: int,
    n_eval: int,
    n_permutations: int,
    *,
    rng_seed: int,
    fit_top_k: int,
    eval_extrap_k: int,
    union_values_detached: torch.Tensor,
    grad_mode: str = "both",
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Draw bootstrap-permutation partitions and locate records that need grad.

    Returns ``(fit_partitions, eval_partitions, needed_grad_idx)``.

    ``fit_partitions`` has shape ``[P, n_fit]`` and ``eval_partitions`` has
    shape ``[P, n_eval]``; both are LongTensors of indices into the union.
    ``needed_grad_idx`` is the sorted union of indices that any permutation's
    fit-top-k selection touches, plus eval-extrap-top-k indices when
    ``grad_mode != "fit"`` (so the caller knows which records to recompute
    with grad).
    """
    if n_fit + n_eval > n_union:
        raise ValueError(
            f"plan_bootstrap_permutations: n_fit + n_eval ({n_fit + n_eval}) "
            f"exceeds n_union ({n_union})."
        )
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    device = union_values_detached.device
    g = torch.Generator(device="cpu")
    g.manual_seed(int(rng_seed))

    fit_partitions = torch.empty((n_permutations, n_fit), dtype=torch.long)
    eval_partitions = torch.empty((n_permutations, n_eval), dtype=torch.long)
    for p in range(n_permutations):
        perm = torch.randperm(n_union, generator=g)
        fit_partitions[p] = perm[:n_fit]
        eval_partitions[p] = perm[n_fit : n_fit + n_eval]

    fit_partitions = fit_partitions.to(device)
    eval_partitions = eval_partitions.to(device)

    values = union_values_detached
    needed_sets: list[torch.Tensor] = []
    fit_top_k_eff = max(0, min(int(fit_top_k), int(n_fit)))
    eval_extrap_k_eff = max(0, min(int(eval_extrap_k), int(n_eval)))

    if fit_top_k_eff > 0:
        fit_vals = values[fit_partitions]  # [P, n_fit]
        _, fit_topk_local = torch.topk(fit_vals, fit_top_k_eff, dim=1)
        fit_global = fit_partitions.gather(1, fit_topk_local).reshape(-1)
        needed_sets.append(fit_global)

    if grad_mode in ("eval", "both") and eval_extrap_k_eff > 0:
        eval_vals = values[eval_partitions]  # [P, n_eval]
        _, eval_topk_local = torch.topk(eval_vals, eval_extrap_k_eff, dim=1)
        eval_global = eval_partitions.gather(1, eval_topk_local).reshape(-1)
        needed_sets.append(eval_global)

    if needed_sets:
        needed = torch.unique(torch.cat(needed_sets))
    else:
        needed = torch.empty(0, dtype=torch.long, device=device)
    return fit_partitions, eval_partitions, needed


def differentiable_ols_extrapolated_primal_loss_bootstrap(
    union_values: torch.Tensor,
    fit_partitions: torch.LongTensor,
    eval_partitions: torch.LongTensor,
    n_fit: int,
    n_eval: int,
    fit_top_k: int,
    *,
    plotting: str = "gringorten",
    score_mode: str = "identity",
    score_eps: float = 1e-8,
    grad_mode: str = "both",
    extrapolated_weighting: str = "rank_uniform",
    loss_space: str = "score",
    improving_only_grad: bool = False,
    iog_scope: str = "both",
    iog_worsening_coef: float = 0.0,
    iog_fit_mask: str = "closed_form",
    union_anchor: torch.Tensor | None = None,
    anchor_margin: float = 0.0,
    calibration_alpha: float | None = None,
    calibration_beta: float | None = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Permutation-averaged primal loss with low-variance gradient.

    For each permutation, run the existing
    ``differentiable_ols_extrapolated_primal_loss`` on
    ``union_values[fit_partitions[p]]`` (with fit-side top-k re-selected on
    the detached values per permutation) and the corresponding eval slice.
    Returns the mean loss across permutations and a dict of mean-of-stats
    plus ``_std`` companions for the headline scalars, and ``perm_losses``,
    the detached vector of the K per-permutation losses (the direct
    gradient-consistency statistic).

    ``union_anchor`` is a per-union-record anchor score for the anchor-gated
    worsening mask; it is sliced by each permutation's partitions exactly as
    the values are.
    """
    if fit_partitions.shape[0] != eval_partitions.shape[0]:
        raise ValueError(
            "fit_partitions and eval_partitions must have the same number of rows"
        )
    n_perm = int(fit_partitions.shape[0])
    if n_perm == 0:
        device = union_values.device
        nan = torch.full((), float("nan"), dtype=torch.float32, device=device)
        return nan, {"primal_mse": nan}

    eval_extrap_k = count_extrapolated_eval_points(
        n_fit, n_eval, plotting=plotting, fit_rank_cutoff=1
    )
    eval_extrap_k = max(1, min(eval_extrap_k, n_eval))
    fit_top_k_eff = max(1, min(int(fit_top_k), int(n_fit)))

    losses: list[torch.Tensor] = []
    stats_acc: dict[str, list[torch.Tensor]] = {}
    for p in range(n_perm):
        fit_vals_p = union_values.index_select(0, fit_partitions[p])
        eval_vals_p = union_values.index_select(0, eval_partitions[p])

        fit_topk_p = select_top_k_indices(fit_vals_p.detach(), fit_top_k_eff)
        eval_sorted = torch.argsort(eval_vals_p.detach(), descending=True)
        eval_topk_p = eval_sorted[:eval_extrap_k]

        if union_anchor is None:
            fit_anchor_p = None
            eval_anchor_p = None
        else:
            fit_anchor_p = union_anchor.index_select(0, fit_partitions[p])
            eval_anchor_p = union_anchor.index_select(0, eval_partitions[p])

        loss_p, info_p = differentiable_ols_extrapolated_primal_loss(
            fit_vals_p,
            fit_topk_p,
            n_fit,
            eval_vals_p,
            eval_topk_p,
            n_eval,
            plotting=plotting,
            score_mode=score_mode,
            score_eps=score_eps,
            grad_mode=grad_mode,
            extrapolated_weighting=extrapolated_weighting,
            loss_space=loss_space,
            improving_only_grad=improving_only_grad,
            iog_scope=iog_scope,
            iog_worsening_coef=iog_worsening_coef,
            iog_fit_mask=iog_fit_mask,
            fit_anchor=fit_anchor_p,
            eval_anchor=eval_anchor_p,
            anchor_margin=anchor_margin,
            calibration_alpha=calibration_alpha,
            calibration_beta=calibration_beta,
        )
        losses.append(loss_p)
        for key, val in info_p.items():
            stats_acc.setdefault(key, []).append(val)

    stacked_losses = torch.stack(losses)
    loss = stacked_losses.mean()

    info: dict[str, torch.Tensor] = {}
    for key, vals in stats_acc.items():
        stacked = torch.stack([v if v.dim() == 0 else v.mean() for v in vals])
        info[key] = stacked.mean()
        if key in ("primal_mse", "improving_only_deploy_frac", "improving_only_fit_frac"):
            info[f"{key}_std"] = stacked.std(unbiased=False) if stacked.numel() > 1 else torch.zeros_like(stacked[0])
    info["perm_losses"] = stacked_losses.detach()
    info["bootstrap_n_permutations"] = torch.tensor(
        float(n_perm), dtype=torch.float32, device=union_values.device
    )
    return loss, info


def fit_ols_tail_from_scores(
    scores: Sequence[float] | Iterable[float],
    *,
    n_top: int = 10,
    plotting: str = "gringorten",
    deploy_queries: int | None = None,
    score_mode: str = "identity",
    score_eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """Fit an OLS tail line on scores and forecast the worst-case score."""
    values = [float(v) for v in scores]
    n = len(values)
    if n == 0:
        return 0.0, float("nan"), float("nan")

    transformed = [_transform_scalar(v, score_mode=score_mode, eps=score_eps) for v in values]
    transformed.sort(reverse=True)
    k = max(1, min(int(n_top), n))
    x = transformed[:k]
    y = [math.log(v) for v in _plotting_positions_list(k, n, plotting)]

    x_mean = sum(x) / k
    y_mean = sum(y) / k
    sxx = sum((xi - x_mean) ** 2 for xi in x)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    b = 0.0 if sxx <= OLS_SXX_TOL else sxy / sxx
    a = y_mean - b * x_mean

    m = max(1, int(deploy_queries) if deploy_queries is not None else n)
    if b == 0.0:
        x_star = x[0]
    else:
        x_star = (math.log(1.0 / m) - a) / b
    return _inverse_transform_scalar(x_star, score_mode=score_mode), a, b


class AffineCalibrationBuffer:
    """Leave-one-group-out affine calibrator over worst-rank (forecast, actual) pairs.

    Calibration-aware forecastability training needs, for each task-set ``g``,
    the affine map ``actual ~= alpha * forecast + beta`` that the deployed
    post-hoc calibrator would fit -- but fit on *other* task-sets, never on
    ``g`` itself (otherwise ``g`` gets a calibration unavailable at test time).

    This buffer keeps streaming sufficient statistics of detached scalar pairs,
    one most-recent contribution per group, so ``loo_coeffs(g)`` returns the fit
    on every group except ``g`` in O(1) by downdating ``g``'s contribution. The
    statistics span all visited groups (slightly stale, which is harmless since
    the coefficients are stop-gradient), so the calibrator approximates the
    full-train fit rather than a high-variance few-point fit.

    The slope is shrunk toward ``1`` by ``ridge_alpha`` (a dimensionless shrink
    fraction: ``alpha = (alpha_ols + ridge_alpha) / (1 + ridge_alpha)``), which
    both tames small-sample variance and keeps ``alpha`` positive so the
    improving-only mask's sign semantics stay valid. The intercept is left free
    (it is the bias correction, the whole point of calibration). A degenerate or
    nonpositive solve falls back to shift-only (``alpha = 1``,
    ``beta = mean(actual - forecast)``); before ``min_points`` donors it returns
    the identity (raw forecast).
    """

    def __init__(
        self,
        *,
        mode: str = "shift_scale",
        min_points: int = 5,
        ridge_alpha: float = 0.05,
        ema: float = 0.0,
    ) -> None:
        if mode not in {"shift", "shift_scale"}:
            raise ValueError(f"Unsupported calibration mode: {mode!r}")
        if not (0.0 <= ema < 1.0):
            raise ValueError(f"ema must be in [0, 1); got {ema}")
        self.mode = mode
        self.min_points = int(min_points)
        self.ridge_alpha = float(ridge_alpha)
        self.ema = float(ema)
        # Streaming sufficient statistics (group weight w == 1 each).
        self._n = 0
        self._sw = 0.0
        self._swQ = 0.0
        self._swY = 0.0
        self._swQQ = 0.0
        self._swQY = 0.0
        # group_id -> (w, Q, Y) most-recent contribution.
        self._contrib: dict[str, Tuple[float, float, float]] = {}

    def _remove(self, w: float, q: float, y: float) -> None:
        self._n -= 1
        self._sw -= w
        self._swQ -= w * q
        self._swY -= w * y
        self._swQQ -= w * q * q
        self._swQY -= w * q * y

    def _add(self, w: float, q: float, y: float) -> None:
        self._n += 1
        self._sw += w
        self._swQ += w * q
        self._swY += w * y
        self._swQQ += w * q * q
        self._swQY += w * q * y

    def update(self, group_id: str, forecast: float, actual: float, *, weight: float = 1.0) -> None:
        """Record group ``group_id``'s latest worst-rank (forecast, actual) pair."""
        q = float(forecast)
        y = float(actual)
        if not (math.isfinite(q) and math.isfinite(y)):
            return
        prev = self._contrib.get(group_id)
        if prev is not None:
            pw, pq, py = prev
            self._remove(pw, pq, py)
            if self.ema > 0.0:
                q = self.ema * pq + (1.0 - self.ema) * q
                y = self.ema * py + (1.0 - self.ema) * y
        self._add(float(weight), q, y)
        self._contrib[group_id] = (float(weight), q, y)

    def _solve(self, sw: float, swQ: float, swY: float, swQQ: float, swQY: float) -> Tuple[float, float]:
        if sw <= 0.0:
            return 1.0, 0.0
        q_bar = swQ / sw
        y_bar = swY / sw
        sxx = swQQ - sw * q_bar * q_bar
        shift_only = (1.0, y_bar - q_bar)
        if self.mode == "shift" or sxx <= OLS_SXX_TOL:
            return shift_only
        sxy = swQY - sw * q_bar * y_bar
        alpha_ols = sxy / sxx
        alpha = (alpha_ols + self.ridge_alpha) / (1.0 + self.ridge_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            return shift_only
        beta = y_bar - alpha * q_bar
        if not math.isfinite(beta):
            return shift_only
        return alpha, beta

    def loo_coeffs(self, group_id: str) -> Tuple[float, float]:
        """Return ``(alpha, beta)`` fit on all groups except ``group_id``.

        Returns the identity ``(1.0, 0.0)`` (raw forecast) until at least
        ``min_points`` donor groups are available.
        """
        prev = self._contrib.get(group_id)
        if prev is not None:
            pw, pq, py = prev
            n_donor = self._n - 1
            sw = self._sw - pw
            swQ = self._swQ - pw * pq
            swY = self._swY - pw * py
            swQQ = self._swQQ - pw * pq * pq
            swQY = self._swQY - pw * pq * py
        else:
            n_donor = self._n
            sw, swQ, swY, swQQ, swQY = (
                self._sw, self._swQ, self._swY, self._swQQ, self._swQY,
            )
        if n_donor < self.min_points:
            return 1.0, 0.0
        return self._solve(sw, swQ, swY, swQQ, swQY)

    def global_coeffs(self) -> Tuple[float, float]:
        """Return ``(alpha, beta)`` fit on all groups (for final/eval use)."""
        if self._n < self.min_points:
            return 1.0, 0.0
        return self._solve(self._sw, self._swQ, self._swY, self._swQQ, self._swQY)

    @property
    def n_groups(self) -> int:
        return self._n

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "min_points": self.min_points,
            "ridge_alpha": self.ridge_alpha,
            "ema": self.ema,
            "n": self._n,
            "sw": self._sw,
            "swQ": self._swQ,
            "swY": self._swY,
            "swQQ": self._swQQ,
            "swQY": self._swQY,
            "contrib": {k: list(v) for k, v in self._contrib.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.mode = state.get("mode", self.mode)
        self.min_points = int(state.get("min_points", self.min_points))
        self.ridge_alpha = float(state.get("ridge_alpha", self.ridge_alpha))
        self.ema = float(state.get("ema", self.ema))
        self._n = int(state.get("n", 0))
        self._sw = float(state.get("sw", 0.0))
        self._swQ = float(state.get("swQ", 0.0))
        self._swY = float(state.get("swY", 0.0))
        self._swQQ = float(state.get("swQQ", 0.0))
        self._swQY = float(state.get("swQY", 0.0))
        self._contrib = {
            k: (float(v[0]), float(v[1]), float(v[2]))
            for k, v in state.get("contrib", {}).items()
        }
