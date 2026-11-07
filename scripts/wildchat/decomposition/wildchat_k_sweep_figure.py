"""k-sensitivity of empirical FRLMB error vs. theoretical rank contribution
xi(k, R=10) from the decomposition in the section "Theory: decomposition of
forecast error" (the k-sweep figure of the appendix "Empirical exploration of
the decomposition on WildChat").

For each k in a sweep, reads `tail_fit_summary.csv` from a per-k analyzer
output directory and pulls the median (or mean) forecast error in
transformed score units. Computes theoretical xi(k, R=10) via a fresh
Exp(1) Monte Carlo. Plots empirical curves on length and per-token mean
NLL alongside the theoretical ridge.

The expected reading: at small-to-moderate k the rank term overpredicts
(positive xi), at very large k it sign-flips, and the empirical curves
should track the theoretical ridge up to a curvature offset that depends
on the local q''(y)/q'(y) of each score.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts rather than Type 3
plt.rcParams["ps.fonttype"] = 42
# Legend names of the scores, whose keys in --score-meta also name the sweep cells.
DISPLAY = {"mean_nll": "mean NLL"}
import numpy as np

DIAG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG_DIR))
# Relative "csv" paths in --score-meta resolve against RESULTS_ROOT.
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))

from wildchat_empirical_decomposition import fit_local_quadratic  # noqa: E402
from wildchat_length_hazard import load_score_column  # noqa: E402


def inverse_ols_predict(topk: np.ndarray, a: np.ndarray, r: float) -> np.ndarray:
    a_bar = a.mean()
    x_bar = topk.mean(axis=1)
    cx = topk - x_bar[:, None]
    S_xx = (cx ** 2).sum(axis=1)
    S_ax = ((a - a_bar)[None, :] * cx).sum(axis=1)
    return x_bar + (r - a_bar) * S_xx / S_ax


def sample_max_of_N_exp1(N: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Exact samples of max(N iid Exp(1)) via inverse CDF, numerically stable for large N."""
    U = rng.random(n_trials)
    return -np.log(-np.expm1(np.log(U) / N))


def theoretical_xi(
    k_values: Iterable[int],
    *,
    R: int,
    m: int,
    n_trials: int,
    chunk: int,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float]]:
    """Empirical Exp(1) finite-m FRLMB error / q'(y) at each k. Returns mean and SE."""
    out: dict[int, tuple[float, float]] = {}
    r_log = float(np.log(R))
    for k in k_values:
        if k > m:
            raise ValueError(f"k={k} > m={m}; choose larger m.")
        a = -np.log(np.arange(1, k + 1, dtype=float))
        errs = []
        for i in range(0, n_trials, chunk):
            c = min(chunk, n_trials - i)
            X = rng.exponential(scale=1.0, size=(c, m))
            topk = -np.sort(-np.partition(X, m - k, axis=1)[:, m - k :], axis=1)
            Q_pred = inverse_ols_predict(topk, a, r_log)
            Y_max = sample_max_of_N_exp1(R * m, c, rng)
            errs.append(Q_pred - Y_max)  # q'(y) = 1 for Exp(1)
        errs_concat = np.concatenate(errs)
        mean = float(errs_concat.mean())
        se = float(errs_concat.std(ddof=1) / np.sqrt(errs_concat.size))
        out[int(k)] = (mean, se)
        print(f"  theoretical xi(k={k}, R={R}): {mean:+.4f} +/- {se:.4f}", flush=True)
    return out


def load_empirical(
    sweep_root: Path, score_label: str, k_values: Iterable[int], stat: str
) -> dict[int, tuple[float, float, float]]:
    """For each k, read tail_fit_summary.csv and pull the (stat, p10, p90)."""
    out: dict[int, tuple[float, float, float]] = {}
    for k in k_values:
        sub = sweep_root / f"{score_label}_k{k}"
        path = sub / "tail_fit_summary.csv"
        if not path.exists():
            print(f"  missing {path}; skipping k={k}")
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        # Single-cell sweep; take the first row.
        r = rows[0]
        center = float(r[stat])
        p10 = float(r["p10_delta"])
        p90 = float(r["p90_delta"])
        out[int(k)] = (center, p10, p90)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument(
        "--scores",
        type=str,
        default="length,mean_nll",
        help="Comma-separated score labels matching the sbatch's score IDs.",
    )
    ap.add_argument(
        "--k-values", type=str, default="5,10,20,50,100,200,500"
    )
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--theory-m", type=int, default=5000)
    ap.add_argument("--theory-trials", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260505)
    ap.add_argument(
        "--stat",
        choices=("mean_delta", "median_delta"),
        default="mean_delta",
        help="Empirical statistic to plot. mean_delta is directly comparable to "
        "theoretical xi; median_delta is more outlier-robust.",
    )
    ap.add_argument(
        "--score-meta",
        type=Path,
        required=True,
        help="JSON dict mapping score label -> {csv, score_column, score_mode, m_for_qp}; "
        "relative csv paths resolve against $RESULTS_ROOT.",
    )
    ap.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="Local-quadratic window half-width for q'(y) estimation.",
    )
    ap.add_argument("--out-pdf", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    score_labels = [s.strip() for s in args.scores.split(",") if s.strip()]
    k_values = [int(s) for s in args.k_values.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    score_csv: dict[str, dict[str, str]] = json.loads(args.score_meta.read_text())
    qp_values: dict[str, float] = {}
    for label in score_labels:
        meta = score_csv[label]
        arr = load_score_column(
            RESULTS_ROOT / meta["csv"],
            meta["score_column"],
            score_mode=meta.get("score_mode", "identity"),
        )
        sorted_desc = np.sort(arr)[::-1]
        m_for_qp = int(meta.get("m_for_qp", args.theory_m))
        y_target = float(np.log(m_for_qp))
        qp, qpp, n_window = fit_local_quadratic(
            sorted_desc, y_target, float(args.delta)
        )
        qp_values[label] = qp
        print(
            f"  q'(y) for {label}: {qp:.4g} (q''={qpp:.4g}, "
            f"n_in_window={n_window} at delta={args.delta})",
            flush=True,
        )

    print(
        f"Theoretical xi(k, R={args.R}) via Exp(1) MC at "
        f"m={args.theory_m}, n_trials={args.theory_trials}",
        flush=True,
    )
    theory = theoretical_xi(
        k_values,
        R=args.R,
        m=args.theory_m,
        n_trials=args.theory_trials,
        chunk=args.chunk,
        rng=rng,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]
    series = []
    for i, label in enumerate(score_labels):
        emp = load_empirical(args.sweep_root, label, k_values, args.stat)
        if not emp:
            print(f"[warn] no empirical data for {label}", flush=True)
            continue
        ks_emp = sorted(emp.keys())
        qp = qp_values[label]
        centers = np.array([emp[k][0] for k in ks_emp]) / qp
        p10s = np.array([emp[k][1] for k in ks_emp]) / qp
        p90s = np.array([emp[k][2] for k in ks_emp]) / qp
        color = colors[i % len(colors)]
        ax.plot(ks_emp, centers, "-o", color=color, markersize=6,
                label=f"empirical {DISPLAY.get(label, label)} / q'(y)")
        ax.fill_between(ks_emp, p10s, p90s, color=color, alpha=0.12)
        series.append(
            {
                "label": label,
                "ks": ks_emp,
                "stat": args.stat,
                "qp": qp,
                "centers_normalized": centers.tolist(),
                "p10_normalized": p10s.tolist(),
                "p90_normalized": p90s.tolist(),
            }
        )

    ks_thy = sorted(theory.keys())
    th_means = np.array([theory[k][0] for k in ks_thy])
    th_ses = np.array([theory[k][1] for k in ks_thy])
    ax.plot(
        ks_thy, th_means, "-D", color="black", markersize=5,
        label=rf"theoretical $\xi(k, R={args.R})$ on Exp(1)",
    )
    ax.fill_between(
        ks_thy, th_means - 1.96 * th_ses, th_means + 1.96 * th_ses,
        color="black", alpha=0.10,
    )

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"top-$k$ used for OLS fit (log)")
    ax.set_ylabel(r"forecast error / $q^\prime(y)$ at $R=" + f"{args.R}" + r"$")
    ax.set_title(
        r"$k$-sensitivity: empirical mean error vs. theoretical $\xi(k, R)$"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, framealpha=0.9, loc="best")
    fig.tight_layout()

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight", dpi=200)
    out_png = args.out_png if args.out_png is not None else args.out_pdf.with_suffix(".png")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Wrote {args.out_pdf} (and {out_png})", flush=True)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(
                {
                    "k_values": k_values,
                    "R": args.R,
                    "theory_m": args.theory_m,
                    "theory_trials": args.theory_trials,
                    "stat": args.stat,
                    "theory": {str(k): list(v) for k, v in theory.items()},
                    "empirical": series,
                },
                f,
                indent=2,
            )
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
