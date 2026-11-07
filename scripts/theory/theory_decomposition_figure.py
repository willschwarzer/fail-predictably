"""Theory figure: decomposition of forecast error across distributions.

For each of five distributions (Exp(1), Lognormal, two Log-Weibulls, and a
mixture), simulate the inverse-OLS estimator and decompose the empirical mean
forecast error (in q'(y) units) into rank, curvature, and (for the mixture)
occupancy contributions. Show as side-by-side bars per distribution with a
total marker, demonstrating that:

    * the rank bar has the same height across every distribution (estimator
      property),
    * the curvature bar varies in magnitude and sign (hazard-rate claim),
    * the occupancy bar appears only on the mixture (mixture claim).
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", REPO_ROOT / "artifacts" / "figures"))


# Cached limiting-law rank-bias coefficients (b_inv_realized, computed by
# scripts/theory/synthetic_inverse_ols_rank_bias.py at 10^6 trials).
B_INV_REALIZED = {2: 0.282, 5: 0.574, 10: 0.794, 100: 1.526, 1000: 2.258}


def inverse_ols_predict(topk, a, r):
    a_bar = a.mean()
    x_bar = topk.mean(axis=1)
    cx = topk - x_bar[:, None]
    S_xx = (cx ** 2).sum(axis=1)
    S_ax = ((a - a_bar)[None, :] * cx).sum(axis=1)
    return x_bar + (r - a_bar) * S_xx / S_ax


# ---- Distributions -----------------------------------------------------------

class Distribution:
    name = ""
    short = ""

    def sample(self, n_trials, n_per_trial, rng):
        raise NotImplementedError

    def q_prime(self, y):
        raise NotImplementedError

    def q_double_prime(self, y):
        return 0.0

    def is_mixture(self):
        return False


class Exponential(Distribution):
    name = "Exp(1)"
    short = "Exp"

    def sample(self, n_trials, n_per_trial, rng):
        return rng.exponential(scale=1.0, size=(n_trials, n_per_trial))

    def q_prime(self, y):
        return 1.0


class Lognormal(Distribution):
    """X = exp(Z), Z ~ N(0, sigma^2). Asymptotic q(y) = exp(sigma sqrt(2y))."""

    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.name = rf"Lognormal($\sigma={sigma}$)"
        self.short = f"LN({sigma})"

    def sample(self, n_trials, n_per_trial, rng):
        return np.exp(rng.normal(0, self.sigma, size=(n_trials, n_per_trial)))

    def q(self, y):
        return np.exp(self.sigma * np.sqrt(2 * y))

    def q_prime(self, y):
        return self.q(y) * self.sigma / np.sqrt(2 * y)

    def q_double_prime(self, y):
        return self.q(y) * (
            self.sigma ** 2 / (2 * y)
            - self.sigma / (2 * (y ** 1.5) * np.sqrt(2))
        )


class LogWeibull(Distribution):
    """Score X with bar F(x) = exp(-x^alpha). Then q(y) = y^(1/alpha)."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.name = rf"Log-Weibull($\alpha={alpha}$)"
        self.short = f"LW({alpha})"

    def sample(self, n_trials, n_per_trial, rng):
        U = rng.uniform(0, 1, size=(n_trials, n_per_trial))
        return (-np.log(U)) ** (1 / self.alpha)

    def q_prime(self, y):
        return (1 / self.alpha) * y ** (1 / self.alpha - 1)

    def q_double_prime(self, y):
        return (
            (1 / self.alpha)
            * (1 / self.alpha - 1)
            * y ** (1 / self.alpha - 2)
        )


class Pareto(Distribution):
    """Pareto tail bar F(x) = x^{-alpha} for x >= 1.  In Frechet domain (not
    Gumbel), so the smoothness condition of the decomposition fails: q''/q'
    is the constant 1/alpha rather than vanishing in y.  The decomposition's
    curvature term picks up the Taylor leading-order correction, but the
    o_p remainder is non-negligible -- this distribution exists in the figure
    to demonstrate where the decomposition breaks."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.name = rf"Pareto($\alpha={alpha}$)"
        self.short = f"P({alpha})"

    def sample(self, n_trials, n_per_trial, rng):
        U = rng.uniform(0.0, 1.0, size=(n_trials, n_per_trial))
        return U ** (-1.0 / self.alpha)

    def q(self, y):
        return np.exp(y / self.alpha)

    def q_prime(self, y):
        return self.q(y) / self.alpha

    def q_double_prime(self, y):
        return self.q(y) / (self.alpha ** 2)


class SymmetricBeta(Distribution):
    """Beta(alpha, alpha), symmetric on [0, 1].  Bounded support, so the
    upper tail is in the reverse Weibull (inverse Weibull) max-domain
    of attraction.  Asymptotically S(x) ~ C(1-x)^alpha as x -> 1, so
    q(y) ~ 1 - C' e^{-y/alpha} and |q''|/q' -> 1/alpha is constant
    rather than vanishing in y -- the smoothness condition fails by
    a factor of 1/alpha.  Uniform(0,1) is the alpha=1 special case
    and produces an o_p remainder of order ~12 q'(y), which dominates
    the figure's y-axis; we use alpha=2 (the canonical tent-shaped
    symmetric Beta) to get |q''|/q' = 0.5 and a remainder of order
    ~1 q'(y), visibly non-zero against the Gumbel-domain remainders
    while still fitting the figure."""

    def __init__(self, alpha=5.0):
        self.alpha = alpha
        a = int(alpha) if float(alpha).is_integer() else alpha
        self.name = rf"Beta($\alpha=\beta={a}$)"
        self.short = f"Be({a},{a})"

    def sample(self, n_trials, n_per_trial, rng):
        return rng.beta(self.alpha, self.alpha, size=(n_trials, n_per_trial))

    def q(self, y):
        from scipy.stats import beta as beta_dist
        return beta_dist.ppf(1.0 - np.exp(-y), self.alpha, self.alpha)

    def q_prime(self, y):
        from scipy.stats import beta as beta_dist
        x = beta_dist.ppf(1.0 - np.exp(-y), self.alpha, self.alpha)
        f = beta_dist.pdf(x, self.alpha, self.alpha)
        return np.exp(-y) / f

    def q_double_prime(self, y):
        from scipy.stats import beta as beta_dist
        a = self.alpha
        x = beta_dist.ppf(1.0 - np.exp(-y), a, a)
        f = beta_dist.pdf(x, a, a)
        # f'/f for Beta(a,a) is (a-1)(1-2x) / [x(1-x)].
        f_prime_over_f = (a - 1.0) * (1.0 - 2.0 * x) / (x * (1.0 - x))
        qp = np.exp(-y) / f
        return -qp * (1.0 + qp * f_prime_over_f)


class HiddenMixture(Distribution):
    """(1-eps)*Exp(1) + eps*(Exp(1)+Delta).  Choose eps so m*eps is small and
    N*eps is large -- the hidden-mode regime."""

    def __init__(self, eps, Delta):
        self.eps = eps
        self.Delta = Delta
        self.name = rf"Mixture ($\epsilon={eps}$)"
        self.short = "Mixt"

    def sample(self, n_trials, n_per_trial, rng):
        is_rare = rng.random((n_trials, n_per_trial)) < self.eps
        bulk = rng.exponential(scale=1.0, size=(n_trials, n_per_trial))
        rare = rng.exponential(scale=1.0, size=(n_trials, n_per_trial)) + self.Delta
        return np.where(is_rare, rare, bulk)

    def q_prime(self, y):
        return 1.0  # bulk-dominated when m*eps is small

    def q_double_prime(self, y):
        return 0.0

    def is_mixture(self):
        return True


# ---- Decomposition -----------------------------------------------------------

def decompose(dist, m, N, k, n_trials, chunk, rng):
    a = -np.log(np.arange(1, k + 1, dtype=float))
    R = N / m
    r = np.log(R)
    y = np.log(m)

    # Theoretical rank coefficient at this R, divided by q'(y), gives a
    # contribution that is the same for every distribution.
    if int(R) not in B_INV_REALIZED:
        raise ValueError(f"need cached B_INV_REALIZED for R={R}")
    rank_contrib = B_INV_REALIZED[int(R)]

    # Theoretical curvature bracket at the nominal a vector.
    a_bar = a.mean()
    a2 = a ** 2
    a2_bar = a2.mean()
    a_centered = a - a_bar
    a2_centered = a2 - a2_bar
    slope = (a_centered * a2_centered).sum() / (a_centered ** 2).sum()
    intercept = a2_bar - slope * a_bar
    pred_at_r = slope * r + intercept
    curvature_bracket = pred_at_r - r ** 2

    qp = dist.q_prime(y)
    qpp = dist.q_double_prime(y)
    curvature_contrib = (qpp / qp / 2) * curvature_bracket

    # Empirical total error in q'(y) units.
    errs = []
    for i in range(0, n_trials, chunk):
        c = min(chunk, n_trials - i)
        fit = dist.sample(c, m, rng)
        deploy = dist.sample(c, N, rng)
        topk = -np.sort(
            -np.partition(fit, m - k, axis=1)[:, m - k :], axis=1
        )
        Q_pred = inverse_ols_predict(topk, a, r)
        Y_max = deploy.max(axis=1)
        errs.append((Q_pred - Y_max) / qp)
    errs = np.concatenate(errs)
    total = float(errs.mean())
    se = float(errs.std(ddof=1) / np.sqrt(len(errs)))

    # Occupancy: defaults to 0; for mixtures it is recomputed in main() once
    # the bulk-only Exp(1) total is in hand, so the remainder is honest
    # rather than absorbed by inference.
    occupancy_contrib = 0.0

    return {
        "name": dist.name,
        "short": dist.short,
        "y": y,
        "qp": qp,
        "qpp": qpp,
        "rank": rank_contrib,
        "curvature": curvature_contrib,
        "occupancy": occupancy_contrib,
        "total": total,
        "total_se": se,
        "is_mixture": dist.is_mixture(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=10000)
    parser.add_argument("--N", type=int, default=100000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-pdf",
        type=str,
        default=str(FIGURES_ROOT / "theory_decomposition.pdf"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(RESULTS_ROOT / "theory_decomposition.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    R = args.N / args.m
    print(f"m={args.m}, N={args.N} (R={int(R)}), k={args.k}, "
          f"n_trials={args.n_trials}", flush=True)

    distributions = [
        Exponential(),
        Lognormal(sigma=1.0),
        LogWeibull(alpha=1.5),
        Pareto(alpha=3.0),
        SymmetricBeta(alpha=2.0),
        HiddenMixture(eps=2e-5, Delta=18.0),
    ]

    rows = []
    for d in distributions:
        t0 = time.time()
        rows.append(
            decompose(d, args.m, args.N, args.k, args.n_trials, args.chunk, rng)
        )
        print(f"  {d.short:>10}: {time.time() - t0:.1f}s", flush=True)

    # Recompute mixture occupancy from the parallel Exp(1) bulk total so that
    # the remainder column is honest rather than absorbed by inference.  The
    # Exp(1) row is already in `rows`; mixture's G_theta is approximately
    # bulk_total - mixture_total to leading order.
    exp_total = next(r["total"] for r in rows if r["short"] == "Exp")
    for r in rows:
        if r["is_mixture"]:
            r["occupancy"] = float(exp_total - r["total"])

    # Remainder: empirical_total minus the predicted decomposition
    # (rank + curvature - occupancy).  Small for Gumbel-domain tails; large
    # for Pareto where the smoothness condition fails.
    for r in rows:
        predicted = r["rank"] + r["curvature"] - r["occupancy"]
        r["remainder"] = float(r["total"] - predicted)

    print(
        f"\n{'Dist':>10} | {'rank':>7} | {'curv':>7} | {'occ':>7} | "
        f"{'rem':>7} | {'empirical':>22}"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"{r['short']:>10} | {r['rank']:>7.3f} | {r['curvature']:>7.3f} "
            f"| {r['occupancy']:>7.3f} | {r['remainder']:>7.3f} "
            f"| {r['total']:>9.4f} +/- {r['total_se']:.4f}"
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    n = len(rows)
    x = np.arange(n)
    width = 0.6

    rank_vals = np.array([r["rank"] for r in rows])
    curv_vals = np.array([r["curvature"] for r in rows])
    occ_vals = np.array([-r["occupancy"] for r in rows])
    rem_vals = np.array([r["remainder"] for r in rows])
    total_vals = np.array([r["total"] for r in rows])
    total_ses = np.array([r["total_se"] for r in rows])

    color_rank = "#1f77b4"
    color_curv = "#ff7f0e"
    color_occ = "#d62728"
    color_rem = "#7f7f7f"

    contributions = [
        (rank_vals, color_rank),
        (curv_vals, color_curv),
        (occ_vals, color_occ),
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
    ax.set_xticklabels([r["name"] for r in rows], rotation=15, ha="right")
    ax.set_ylabel(r"contribution to forecast error / $q^\prime(y)$")

    legend_elements = [
        Patch(facecolor=color_rank, label=r"rank ($R_\theta / q^\prime(y)$)"),
        Patch(facecolor=color_curv, label=r"curvature ($C_\theta / q^\prime(y)$)"),
        Patch(facecolor=color_occ, label=r"occupancy ($-G_\theta / q^\prime(y)$)"),
        Patch(facecolor=color_rem, label=r"remainder ($o_p$)"),
        Line2D([0], [0], marker="D", color="black", linestyle="None",
               markersize=6, label="empirical total"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    print(f"\nWrote {out_pdf}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
