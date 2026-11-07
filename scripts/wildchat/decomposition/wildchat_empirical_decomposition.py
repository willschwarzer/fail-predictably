"""Empirical FRLMB decomposition on WildChat scored slices (the decomposition
figure of the appendix "Empirical exploration of the decomposition on WildChat").

For each scored CSV, treat the slice as the population, simulate inverse-OLS
forecasts at (m, N, k, R), and decompose the empirical mean error into:
  - rank contribution from the cached B_INV_REALIZED[R] table at k=10;
  - curvature contribution from a local-quadratic estimator of q'(y) and
    q''(y) on the order statistics of the slice;
  - residual = empirical_total - rank - curvature, which absorbs any
    occupancy / non-smoothness contribution (not separately identifiable
    from a single empirical sample without specifying a mixture model).

Renders the decomposition as a stacked bar chart parallel to the synthetic
decomposition panel of the theory figure (section "Theory: decomposition of
forecast error"), with one column per score variant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts rather than Type 3
plt.rcParams["ps.fonttype"] = 42
import numpy as np

DIAG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG_DIR))
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(1, str(REPO_ROOT / "scripts" / "theory"))  # theory_decomposition_figure
# Relative "csv" paths in --config resolve against RESULTS_ROOT.
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))

from theory_decomposition_figure import (  # noqa: E402
    B_INV_REALIZED,
    Distribution,
    inverse_ols_predict,
)
from wildchat_length_hazard import (  # noqa: E402
    apply_score_mode,
    load_score_column,
)


@dataclass
class PanelSpec:
    label: str
    short: str
    csv: Path
    score_column: str
    score_mode: str = "identity"


def fit_local_quadratic(
    sorted_desc: np.ndarray, y_target: float, delta: float
) -> tuple[float, float, int]:
    """Fit s_(j) = c0 + c1 (y_j - y_target) + 0.5 c2 (y_j - y_target)^2
    on the order-statistic window with y_j in [y_target - delta, y_target + delta].

    Returns (q_prime, q_double_prime, n_in_window). Raises if too few points.
    """
    n = sorted_desc.size
    j = np.arange(1, n + 1, dtype=np.float64)
    y_j = np.log((n + 1.0) / j)  # positive log-1/survival convention
    mask = (y_j >= y_target - delta) & (y_j <= y_target + delta)
    if mask.sum() < 5:
        raise RuntimeError(
            f"local-quadratic window too sparse: {int(mask.sum())} points "
            f"at y_target={y_target:.3f}, delta={delta:.3f}, n={n}"
        )
    yc = y_j[mask] - y_target
    sw = sorted_desc[mask]
    A = np.column_stack([np.ones_like(yc), yc, 0.5 * yc * yc])
    coef, *_ = np.linalg.lstsq(A, sw, rcond=None)
    return float(coef[1]), float(coef[2]), int(mask.sum())


class EmpiricalScoreSlice(Distribution):
    """A Distribution backed by a finite scored sample. Sampling is with
    replacement so trial counts can exceed the slice size; q'(y) and q''(y)
    are estimated via local quadratic on the order statistics."""

    def __init__(
        self,
        spec: PanelSpec,
        *,
        delta: float = 0.5,
    ):
        self.spec = spec
        self.name = spec.label
        self.short = spec.short
        self.delta = delta
        arr = load_score_column(
            spec.csv, spec.score_column, score_mode=spec.score_mode
        )
        if arr.size < 1000:
            raise RuntimeError(
                f"{spec.label}: only {arr.size} usable rows; need >= 1000"
            )
        self._scores = arr
        self._sorted_desc = np.sort(arr)[::-1]
        self.n_total = arr.size
        self._cached_qp: dict[float, tuple[float, float, int]] = {}

    def sample(self, n_trials: int, n_per_trial: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n_total, size=(n_trials, n_per_trial))
        return self._scores[idx]

    def _local_quadratic(self, y: float) -> tuple[float, float, int]:
        if y not in self._cached_qp:
            self._cached_qp[y] = fit_local_quadratic(
                self._sorted_desc, y, self.delta
            )
        return self._cached_qp[y]

    def q_prime(self, y):
        return self._local_quadratic(float(y))[0]

    def q_double_prime(self, y):
        return self._local_quadratic(float(y))[1]


def empirical_decompose(
    slice_dist: EmpiricalScoreSlice,
    *,
    m: int,
    N: int,
    k: int,
    n_trials: int,
    chunk: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Mirror theory_decomposition_figure.decompose, but for empirical slices.

    Avoids mutating theory_decomposition_figure (which would couple to the
    synthetic figure). The math is identical to that of decompose() in the
    synthetic-only case (no mixture branch); occupancy is folded into the
    residual.
    """
    R = N / m
    if int(R) not in B_INV_REALIZED:
        raise ValueError(
            f"Need cached B_INV_REALIZED for R={R}; "
            f"available: {sorted(B_INV_REALIZED.keys())}"
        )
    a = -np.log(np.arange(1, k + 1, dtype=float))
    r = float(np.log(R))
    y = float(np.log(m))

    rank_contrib = float(B_INV_REALIZED[int(R)])

    a_bar = a.mean()
    a2 = a ** 2
    a2_bar = a2.mean()
    a_centered = a - a_bar
    a2_centered = a2 - a2_bar
    slope = (a_centered * a2_centered).sum() / (a_centered ** 2).sum()
    intercept = a2_bar - slope * a_bar
    pred_at_r = slope * r + intercept
    curvature_bracket = pred_at_r - r ** 2

    qp = slice_dist.q_prime(y)
    qpp = slice_dist.q_double_prime(y)
    if qp <= 0:
        # The local quadratic produced a non-positive slope. The score's tail
        # is effectively flat or decreasing in y at this window; bail out
        # rather than divide through.
        raise RuntimeError(
            f"{slice_dist.short}: q'(y)={qp:.4g} <= 0 at y=log(m)={y:.3f}; "
            "local-quadratic estimator is ill-conditioned for this score."
        )
    curvature_contrib = (qpp / qp / 2) * curvature_bracket

    errs = []
    for i in range(0, n_trials, chunk):
        c = min(chunk, n_trials - i)
        fit = slice_dist.sample(c, m, rng)
        deploy = slice_dist.sample(c, N, rng)
        topk = -np.sort(
            -np.partition(fit, m - k, axis=1)[:, m - k :], axis=1
        )
        Q_pred = inverse_ols_predict(topk, a, r)
        Y_max = deploy.max(axis=1)
        errs.append((Q_pred - Y_max) / qp)
    errs_concat = np.concatenate(errs)
    total = float(errs_concat.mean())
    se = float(errs_concat.std(ddof=1) / np.sqrt(errs_concat.size))

    residual = total - (rank_contrib + curvature_contrib)
    return {
        "name": slice_dist.name,
        "short": slice_dist.short,
        "n_slice": slice_dist.n_total,
        "y": y,
        "qp": qp,
        "qpp": qpp,
        "rank": rank_contrib,
        "curvature": curvature_contrib,
        "occupancy": 0.0,  # not separately identifiable; folded into residual
        "remainder": residual,
        "total": total,
        "total_se": se,
        "is_mixture": False,
    }


def render_bars(rows: list[dict], out_pdf: Path) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    n = len(rows)
    x = np.arange(n)
    width = 0.6

    rank_vals = np.array([r["rank"] for r in rows])
    curv_vals = np.array([r["curvature"] for r in rows])
    rem_vals = np.array([r["remainder"] for r in rows])
    total_vals = np.array([r["total"] for r in rows])
    total_ses = np.array([r["total_se"] for r in rows])

    color_rank = "#1f77b4"
    color_curv = "#ff7f0e"
    color_rem = "#7f7f7f"

    contributions = [
        (rank_vals, color_rank),
        (curv_vals, color_curv),
        (rem_vals, color_rem),
    ]
    for i in range(n):
        pos_bottom = 0.0
        neg_bottom = 0.0
        for vals, color in contributions:
            v = vals[i]
            if v >= 0:
                ax.bar(x[i], v, width, bottom=pos_bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                pos_bottom += v
            else:
                ax.bar(x[i], v, width, bottom=neg_bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                neg_bottom += v

    ax.errorbar(
        x, total_vals, yerr=1.96 * total_ses, fmt="D", color="black",
        markersize=7, capsize=4, zorder=10, markeredgecolor="white",
        markeredgewidth=1,
    )

    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["short"] for r in rows], rotation=15, ha="right")
    ax.set_ylabel(r"contribution to forecast error / $q^\prime(y)$")

    legend_elements = [
        Patch(facecolor=color_rank, label=r"rank ($T_\theta / q^\prime(y)$)"),
        Patch(facecolor=color_curv, label=r"curvature ($C_\theta / q^\prime(y)$)"),
        Patch(facecolor=color_rem,
              label=r"residual (occupancy + $o_p$)"),
        Line2D([0], [0], marker="D", color="black", linestyle="None",
               markersize=6, label="empirical total"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"\nWrote {out_pdf} (and .png)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="JSON list of PanelSpec dicts; relative csv paths resolve "
                    "against $RESULTS_ROOT.")
    ap.add_argument("--m", type=int, default=5000)
    ap.add_argument("--N", type=int, default=50000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-pdf", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    if not isinstance(cfg, list) or not cfg:
        raise ValueError("--config must be a non-empty JSON list")

    R = args.N / args.m
    print(
        f"m={args.m}, N={args.N} (R={int(R)}), k={args.k}, "
        f"n_trials={args.n_trials}, delta={args.delta}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    for entry in cfg:
        spec = PanelSpec(
            label=entry["label"],
            short=entry["short"],
            csv=RESULTS_ROOT / entry["csv"],
            score_column=entry["score_column"],
            score_mode=entry.get("score_mode", "identity"),
        )
        t0 = time.time()
        slice_dist = EmpiricalScoreSlice(spec, delta=args.delta)
        try:
            row = empirical_decompose(
                slice_dist,
                m=args.m,
                N=args.N,
                k=args.k,
                n_trials=args.n_trials,
                chunk=args.chunk,
                rng=rng,
            )
        except RuntimeError as e:
            print(f"  {spec.short}: SKIP ({e})", flush=True)
            continue
        rows.append(row)
        print(
            f"  {spec.short:>10}: n_slice={slice_dist.n_total} "
            f"qp={row['qp']:.4g} qpp={row['qpp']:.4g} "
            f"rank={row['rank']:.3f} curv={row['curvature']:.3f} "
            f"rem={row['remainder']:.3f} total={row['total']:.3f}+/-{row['total_se']:.3f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No usable rows after running all panels.")

    print(
        f"\n{'short':>12} | {'rank':>7} | {'curv':>7} | "
        f"{'rem':>7} | {'empirical':>22}"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"{r['short']:>12} | {r['rank']:>7.3f} | {r['curvature']:>7.3f} "
            f"| {r['remainder']:>7.3f} | "
            f"{r['total']:>9.4f} +/- {r['total_se']:.4f}"
        )

    render_bars(rows, args.out_pdf)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=str)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
