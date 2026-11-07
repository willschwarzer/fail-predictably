"""FRLMB tail-fit-quality analysis on a per-prompt score CSV.

Given a CSV of per-prompt risk scores (one column = score), for each (N, M)
cell in a log-spaced grid, repeatedly:

  - Sample N+M scores without replacement from the slice (bootstrap-permutation).
  - Top-K-fit OLS line on the N "fit" scores in (score, log-survival) space.
  - For every extrapolated rank j (j up to count_extrapolated_eval_points(N, M)),
    compute predicted_j = (log(j/(M+1)) - a) / b and compare to actual_j
    = j-th-largest score among the M "deploy" scores.

Sign: delta = predicted - actual; positive = over-pred (line above actual),
negative = under-pred (line below actual).

Output: a per-rank CSV `tail_fit_per_rank.csv` and a per-(N, M) summary
`tail_fit_summary.csv` with mean / median / p10 / p90 / under-pred fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecast_experiment.ols_utils import (  # noqa: E402
    count_extrapolated_eval_points,
    fit_ols_tail_from_scores,
)


def load_scores(csv_path: Path, score_column: str) -> np.ndarray:
    vals: list[float] = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        if score_column not in (r.fieldnames or []):
            raise KeyError(
                f"score-column={score_column!r} not in {csv_path} (have: {r.fieldnames})"
            )
        for row in r:
            v = row[score_column]
            if v == "" or v is None:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    arr = np.asarray(vals, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = arr[finite]
    return arr


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def apply_score_mode(arr: np.ndarray, mode: str, *, eps: float = 1e-8) -> np.ndarray:
    """Mirror ols_utils._transform_scalar/tensor in numpy. Pre-applied here so the
    rest of the analyzer can stay in score-mode='identity'."""
    if mode == "identity":
        return arr.astype(np.float64)
    if mode == "log":
        return np.log(np.clip(arr, eps, None)).astype(np.float64)
    if mode == "gumbel_log_prob":
        return -np.log(np.clip(-arr, eps, None)).astype(np.float64)
    if mode == "gumbel_prob":
        clipped = np.clip(arr, eps, 1.0 - 1e-16)
        return -np.log(-np.log(clipped)).astype(np.float64)
    raise ValueError(f"Unsupported score_mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", type=Path, required=True)
    ap.add_argument("--score-column", required=True, help="Column to use as the risk score.")
    ap.add_argument(
        "--score-mode",
        default="identity",
        choices=("identity", "log", "gumbel_log_prob", "gumbel_prob"),
        help="Transform applied to the raw column before OLS (matches ols_utils _transform).",
    )
    ap.add_argument("--score-eps", type=float, default=1e-8)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cell-name", required=True, help="Label written into output rows.")
    ap.add_argument(
        "--n-grid",
        type=parse_int_list,
        default=[200, 1000, 5000, 25000],
        help="Comma-separated eval (fit) sizes.",
    )
    ap.add_argument(
        "--m-multipliers",
        type=parse_int_list,
        default=[10, 50, 250],
        help="Comma-separated deploy/eval multipliers; M = mult * N (clipped to slice).",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--plotting", choices=("weibull", "gringorten"), default="weibull")
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--max-extrapolated-ranks-per-perm",
        type=int,
        default=0,
        help=">0 caps extrapolated ranks per (perm) to keep CSV bounded; 0 = all.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[load] scores from {args.scores_csv} column={args.score_column} "
        f"score_mode={args.score_mode}",
        flush=True,
    )
    raw_scores = load_scores(args.scores_csv, args.score_column)
    scores = apply_score_mode(raw_scores, args.score_mode, eps=args.score_eps)
    n_total = scores.shape[0]
    print(
        f"[load] n_total={n_total} raw_min={raw_scores.min():.4g} "
        f"raw_max={raw_scores.max():.4g} transformed_min={scores.min():.4g} "
        f"transformed_max={scores.max():.4g}",
        flush=True,
    )
    if n_total < max(args.n_grid) + min(args.m_multipliers) * min(args.n_grid):
        print(
            f"[warn] slice may be smaller than the largest (N,M); cells will be clipped.",
            flush=True,
        )

    rng = np.random.default_rng(args.seed)

    cells: list[tuple[int, int]] = []
    for n in args.n_grid:
        for mult in args.m_multipliers:
            m_target = mult * n
            m = min(m_target, n_total - n)
            if m < args.top_k or n < args.top_k:
                continue
            cells.append((int(n), int(m)))
    if not cells:
        raise RuntimeError("No (N, M) cells fit the slice size.")
    print("[grid] cells (N, M):", cells, flush=True)

    per_rank_path = args.out_dir / "tail_fit_per_rank.csv"
    summary_path = args.out_dir / "tail_fit_summary.csv"
    manifest_path = args.out_dir / "analysis_manifest.json"

    n_extrap_max_seen = 0
    n_rows_written = 0
    summary_rows: list[dict] = []
    t_start = time.time()

    with per_rank_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cell",
                "N",
                "M",
                "perm",
                "rank",
                "n_extrapolated",
                "predicted",
                "actual",
                "delta",
                "ols_a",
                "ols_b",
            ]
        )
        for cell_idx, (n, m) in enumerate(cells):
            n_extrap_full = count_extrapolated_eval_points(
                n, m, plotting=args.plotting, fit_rank_cutoff=1
            )
            cap = args.max_extrapolated_ranks_per_perm
            n_extrap = n_extrap_full if cap <= 0 else min(n_extrap_full, cap)
            n_extrap_max_seen = max(n_extrap_max_seen, n_extrap_full)
            cell_deltas: list[float] = []
            for perm in range(args.n_permutations):
                idx = rng.choice(n_total, size=n + m, replace=False)
                fit_scores = scores[idx[:n]]
                deploy_scores = scores[idx[n:]]
                # Use fit_ols_tail_from_scores to fit the OLS line. We pass
                # deploy_queries=m only to get the worst-rank forecast; for
                # per-rank predictions we compute (a, b) ourselves.
                _forecast_worst, a, b = fit_ols_tail_from_scores(
                    fit_scores.tolist(),
                    n_top=args.top_k,
                    plotting=args.plotting,
                    deploy_queries=m,
                    score_mode="identity",
                )
                if not (math.isfinite(a) and math.isfinite(b)) or b == 0.0:
                    continue
                deploy_desc = np.sort(deploy_scores)[::-1]
                j = np.arange(1, n_extrap + 1, dtype=np.float64)
                if args.plotting == "weibull":
                    y_j = np.log(j / (m + 1.0))
                else:  # gringorten
                    y_j = np.log((j - 0.44) / (m + 0.12))
                predicted = (y_j - a) / b
                actual = deploy_desc[:n_extrap].astype(np.float64)
                delta = predicted - actual
                for ji, dij, pij, aij in zip(j.astype(int), delta, predicted, actual):
                    w.writerow(
                        [
                            args.cell_name,
                            n,
                            m,
                            perm,
                            int(ji),
                            n_extrap_full,
                            f"{pij:.6g}",
                            f"{aij:.6g}",
                            f"{dij:.6g}",
                            f"{a:.6g}",
                            f"{b:.6g}",
                        ]
                    )
                    n_rows_written += 1
                cell_deltas.extend(delta.tolist())

            if cell_deltas:
                arr = np.asarray(cell_deltas)
                summary_rows.append(
                    {
                        "cell": args.cell_name,
                        "N": n,
                        "M": m,
                        "n_extrapolated_per_perm": n_extrap,
                        "n_extrapolated_full": n_extrap_full,
                        "n_permutations": args.n_permutations,
                        "n_deltas": int(arr.size),
                        "mean_delta": float(arr.mean()),
                        "median_delta": float(np.median(arr)),
                        "p10_delta": float(np.percentile(arr, 10)),
                        "p90_delta": float(np.percentile(arr, 90)),
                        "frac_under": float((arr < 0).mean()),
                    }
                )
            el = time.time() - t_start
            print(
                f"[cell {cell_idx + 1}/{len(cells)}] N={n} M={m} extrap={n_extrap_full} "
                f"deltas={len(cell_deltas)} elapsed={el:.1f}s",
                flush=True,
            )

    with summary_path.open("w", newline="") as f:
        if summary_rows:
            wkeys = list(summary_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=wkeys)
            w.writeheader()
            w.writerows(summary_rows)

    manifest = {
        "scores_csv": str(args.scores_csv),
        "score_column": args.score_column,
        "cell_name": args.cell_name,
        "n_total_loaded": n_total,
        "n_grid": args.n_grid,
        "m_multipliers": args.m_multipliers,
        "cells": [{"N": n, "M": m} for n, m in cells],
        "top_k": args.top_k,
        "plotting": args.plotting,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "score_mode": args.score_mode,
        "score_eps": args.score_eps,
        "max_extrapolated_ranks_per_perm": args.max_extrapolated_ranks_per_perm,
        "n_rows_written": n_rows_written,
        "wall_seconds": time.time() - t_start,
        "out": {"per_rank": str(per_rank_path), "summary": str(summary_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n=== summary (mean signed delta per (N, M)) ===", flush=True)
    print(f"{'N':>8} {'M':>10} {'mean':>10} {'median':>10} {'p10':>10} {'p90':>10} {'under':>8}", flush=True)
    for r in summary_rows:
        print(
            f"{r['N']:>8} {r['M']:>10} {r['mean_delta']:>+10.4f} {r['median_delta']:>+10.4f} "
            f"{r['p10_delta']:>+10.4f} {r['p90_delta']:>+10.4f} {r['frac_under']:>8.2%}",
            flush=True,
        )

    print(f"\n[done] wrote {per_rank_path}", flush=True)
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
