"""The decomposition of forecast error on reference distributions, at the WildChat decomposition's setting.

This gives the numbers of the paragraph "Comparison with a bounded reference distribution": the rank,
curvature and residual terms, in units of q'(y), for three tails of the Gumbel domain (Exp(1),
Lognormal(1) and Log-Weibull(1.5)), Pareto(3) and Uniform(0,1), at (M, N, k) = (5,000, 50,000, 10).
It uses decompose() of scripts/theory/theory_decomposition_figure.py, which simulates the inverse-OLS
forecast and takes the curvature term from each distribution's exact q''/q'. Uniform(0,1) is that
script's symmetric Beta with alpha = 1. The residual is the empirical total minus the rank and
curvature terms. The rank term is the script's cached Monte Carlo value, 0.794; the paper prints the
exact value, 0.793. A rerun matches the paper's values within Monte Carlo error.

  python scripts/wildchat/decomposition/reference_decomposition.py [--n-trials 10000] [--seed 42]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "theory"))
from theory_decomposition_figure import (  # noqa: E402
    Exponential,
    Lognormal,
    LogWeibull,
    Pareto,
    SymmetricBeta,
    decompose,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", type=int, default=5000)
    ap.add_argument("--N", type=int, default=50000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    print(f"M = {a.m}, N = {a.N}, k = {a.k}, {a.n_trials} trials; every term in units of q'(y)")
    print(f"{'distribution':>13} | {'rank':>6} | {'curvature':>9} | {'empirical total':>16} | {'residual':>8}")
    for d in (Exponential(), Lognormal(sigma=1.0), LogWeibull(alpha=1.5), Pareto(alpha=3.0), SymmetricBeta(alpha=1.0)):
        r = decompose(d, a.m, a.N, a.k, a.n_trials, a.chunk, rng)
        name = "Uniform(0,1)" if isinstance(d, SymmetricBeta) else r["short"]
        print(f"{name:>13} | {r['rank']:6.3f} | {r['curvature']:+9.3f} | {r['total']:+7.3f} +/- {r['total_se']:.3f} | "
              f"{r['total'] - r['rank'] - r['curvature']:+8.3f}", flush=True)


if __name__ == "__main__":
    main()
