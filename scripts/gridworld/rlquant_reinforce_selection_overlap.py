#!/usr/bin/env python
"""Selection fidelity of the sampling-based forecastability path.

The implicit curriculum of forecastability training lives in task selection:
fit-side top-k plus extrapolated deploy ranks (the `needed` set from
`plan_bootstrap_permutations`), further filtered by the IOG mask (tasks with
nonzero leaf weight). This diagnostic measures how well the sampled path
reproduces the exact path's selection at a fixed checkpoint, as a function of
rollouts-per-task n. Selection depends only on the no-grad scoring pass, which
is shared by estimators A and B, so cells are indexed by n alone.

Per n, over many independent rollout draws:
  - recall of the exact selected set (overall, fit-side, deploy-side)
  - recall of the exact IOG-retained (nonzero-weight) set
  - draw-to-draw Jaccard of the selected set (churn from sampling noise only;
    the policy is frozen, so training would add churn on top of this)
  - chance-level recall for a same-size uniformly random selection
  - fraction of retained weights with the regret-raising sign (IOG check)

The selection diagnostic in the appendix "Forecastability training with sampled
returns" is run_rlquant_reinforce_selection_overlap.sbatch: seed 11 of the full-IOG
fleet, final checkpoint, the defaults below.

Usage:
  python scripts/gridworld/rlquant_reinforce_selection_overlap.py \
    --seed-dir artifacts/results/<run>/seed_11 \
    --checkpoint final --n-list 1,2,4,8,16,32 --draws 200 \
    --out-json artifacts/results/<out>/selection_overlap_seed11.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecast_experiment import train_rlquant as tr  # noqa: E402
from forecast_experiment.rerun_rlquant_eval import (  # noqa: E402
    _load_cfg,
    _load_manifest_suite,
    _make_model_factory,
)
from forecast_experiment.rlquant_gridworld import (  # noqa: E402
    prepare_task_batch,
    sample_policy_rollouts_batched,
)


def _selection_and_weights(
    regrets_det: torch.Tensor,
    *,
    n_fit: int,
    n_deploy: int,
    eval_extrap_k: int,
    cfg,
    rng_seed: int,
) -> tuple[set[int], set[int], torch.Tensor]:
    """(selected set, IOG-retained set, leaf weights) for one score vector.

    Mirrors the reinforce path: plan on detached scores, then leaf-autograd
    through the bootstrap loss to get the weights the PG pass would use.
    """
    _fit_p, _eval_p, needed_idx = tr.plan_bootstrap_permutations(
        n_union=n_fit + n_deploy,
        n_fit=n_fit,
        n_eval=n_deploy,
        n_permutations=cfg.bootstrap_permutations,
        rng_seed=rng_seed,
        fit_top_k=cfg.ols_top_k,
        eval_extrap_k=eval_extrap_k,
        union_values_detached=regrets_det,
        grad_mode=cfg.grad_mode,
    )
    selected = {int(i) for i in needed_idx.tolist()}
    leaf = regrets_det.detach().clone().requires_grad_(True)
    loss, _ = tr._meta_primal_loss_bootstrap(
        leaf, n_fit, n_deploy, cfg=cfg, rng_seed=rng_seed
    )
    (leaf_grad,) = torch.autograd.grad(loss, [leaf], allow_unused=True)
    weights = torch.zeros_like(leaf) if leaf_grad is None else leaf_grad.detach()
    retained = {int(i) for i in torch.nonzero(weights).flatten().tolist()}
    return selected, retained, weights


def _recall(sample: set[int], reference: set[int]) -> float:
    if not reference:
        return float("nan")
    return len(sample & reference) / len(reference)


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", required=True)
    ap.add_argument("--checkpoint", choices=["pretrained", "final"], default="final")
    ap.add_argument("--manifest-index", type=int, default=0)
    ap.add_argument("--n-list", default="1,2,4,8,16,32")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--rng-seed", type=int, default=20260724)
    ap.add_argument("--improving-only-grad", type=int, default=-1,
                    help="-1: use run config; 0/1: override")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seed_dir = Path(args.seed_dir)
    cfg = _load_cfg(seed_dir, device=args.device)
    if args.improving_only_grad in (0, 1):
        cfg.improving_only_grad = bool(args.improving_only_grad)
    if cfg.bootstrap_permutations < 1 or cfg.meta_loss_mode != "ols_primal_extrapolated":
        raise SystemExit("run config must use the bootstrap ols_primal_extrapolated objective")
    tr._seed_everything(cfg.seed, deterministic_debug=False)
    manifest_suite, materialization_cache = _load_manifest_suite(cfg)
    make_model = _make_model_factory(manifest_suite, cfg)
    model = make_model().to(cfg.device)
    tr._load_checkpoint(model, str(seed_dir / f"{args.checkpoint}_checkpoint.pt"))
    model.eval()

    manifest = manifest_suite.meta_manifests[args.manifest_index]
    fit_tasks = tr._materialize_records_cached(
        list(manifest.fit_tasks), cfg=cfg, split_name="sel_fit", cache=materialization_cache
    )
    deploy_tasks = tr._materialize_records_cached(
        list(manifest.deploy_tasks), cfg=cfg, split_name="sel_deploy", cache=materialization_cache
    )
    n_fit, n_deploy = len(fit_tasks), len(deploy_tasks)
    n_union = n_fit + n_deploy
    union_tasks = list(fit_tasks) + list(deploy_tasks)
    eval_extrap_k = tr.count_extrapolated_eval_points(
        n_fit, n_deploy, plotting=cfg.plotting_position
    )
    print(f"manifest={manifest.name} fit={n_fit} deploy={n_deploy} extrap_k={eval_extrap_k}",
          flush=True)

    # Exact-path reference sets at the same checkpoint.
    with torch.no_grad():
        _, exact_regrets = tr._exact_returns_and_regrets(
            model, union_tasks, cfg=cfg, require_grad=False
        )
    sel_exact, ret_exact, w_exact = _selection_and_weights(
        exact_regrets,
        n_fit=n_fit,
        n_deploy=n_deploy,
        eval_extrap_k=eval_extrap_k,
        cfg=cfg,
        rng_seed=args.rng_seed,
    )
    sel_exact_fit = {i for i in sel_exact if i < n_fit}
    sel_exact_dep = {i for i in sel_exact if i >= n_fit}
    print(
        f"exact: selected={len(sel_exact)} (fit={len(sel_exact_fit)} "
        f"deploy={len(sel_exact_dep)}) iog_retained={len(ret_exact)} "
        f"neg_weights={int((w_exact < 0).sum())}",
        flush=True,
    )

    # One forward pass; only the rollouts are redrawn per draw.
    score_batch = prepare_task_batch(union_tasks, str(cfg.device))
    score_batch = tr._resolve_optimal_values(union_tasks, score_batch, cfg=cfg)
    with torch.no_grad():
        dense_logits = model(score_batch["features"], score_batch["task_vectors"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.rng_seed + 1)

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    results: dict = {
        "seed_dir": str(seed_dir),
        "checkpoint": args.checkpoint,
        "manifest": manifest.name,
        "rng_seed": args.rng_seed,
        "draws": args.draws,
        "improving_only_grad": bool(cfg.improving_only_grad),
        "n_fit": n_fit,
        "n_deploy": n_deploy,
        "exact_selected": len(sel_exact),
        "exact_selected_fit": len(sel_exact_fit),
        "exact_selected_deploy": len(sel_exact_dep),
        "exact_iog_retained": len(ret_exact),
        "cells": [],
    }

    for n in n_list:
        recalls, recalls_fit, recalls_dep, recalls_ret = [], [], [], []
        churns, sizes, neg_fracs = [], [], []
        prev_sel: set[int] | None = None
        for _ in range(args.draws):
            with torch.no_grad():
                returns, _ = sample_policy_rollouts_batched(
                    union_tasks,
                    dense_logits,
                    score_batch,
                    n_rollouts=n,
                    device=str(cfg.device),
                    generator=generator,
                )
            if cfg.reinforce_score == "neg_return":
                regrets = -returns.mean(dim=1)
            else:
                regrets = score_batch["optimal_values"] - returns.mean(dim=1)
            sel, ret, weights = _selection_and_weights(
                regrets,
                n_fit=n_fit,
                n_deploy=n_deploy,
                eval_extrap_k=eval_extrap_k,
                cfg=cfg,
                rng_seed=args.rng_seed,
            )
            recalls.append(_recall(sel, sel_exact))
            recalls_fit.append(_recall({i for i in sel if i < n_fit}, sel_exact_fit))
            recalls_dep.append(_recall({i for i in sel if i >= n_fit}, sel_exact_dep))
            recalls_ret.append(_recall(ret, ret_exact))
            sizes.append(float(len(sel)))
            n_nonzero = int((weights != 0).sum())
            neg_fracs.append(
                float((weights < 0).sum()) / n_nonzero if n_nonzero else float("nan")
            )
            if prev_sel is not None:
                churns.append(_jaccard(sel, prev_sel))
            prev_sel = sel
        cell = {
            "n": n,
            "recall_selected": _mean(recalls),
            "recall_selected_fit": _mean(recalls_fit),
            "recall_selected_deploy": _mean(recalls_dep),
            "recall_iog_retained": _mean(recalls_ret),
            "draw_to_draw_jaccard": _mean(churns),
            "mean_selected_size": _mean(sizes),
            "chance_recall": _mean(sizes) / n_union,
            "neg_weight_frac": _mean(neg_fracs),
        }
        results["cells"].append(cell)
        print(
            f"[n={n}] recall={cell['recall_selected']:.3f} "
            f"(fit={cell['recall_selected_fit']:.3f} "
            f"deploy={cell['recall_selected_deploy']:.3f}) "
            f"iog_recall={cell['recall_iog_retained']:.3f} "
            f"churn_jac={cell['draw_to_draw_jaccard']:.3f} "
            f"chance={cell['chance_recall']:.3f} "
            f"negw={cell['neg_weight_frac']:.4f}",
            flush=True,
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
