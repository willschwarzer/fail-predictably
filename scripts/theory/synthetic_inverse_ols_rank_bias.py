"""Synthetic validation of the inverse-OLS FRLMB rank-bias constants.

For an exponential score tail with rate 1, q(y) = y so q'(y) = 1 and
q'' = q''' = 0. The smoothness condition in the appendix is trivially
satisfied, and the decomposition predicts that the only non-vanishing term
is the rank term, whose limiting expected error in units of q'(y) is:

    R       b_realized   b_quantile
    2       0.282        0.859
    5       0.574        1.151
    10      0.794        1.371
    100     1.526        2.103
    1000    2.258        2.835

This script samples actual finite-m FRLMB inverse-OLS forecasts on an
Exp(1) tail and compares the empirical mean error to the table.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


THEORY_REALIZED = {2: 0.282, 5: 0.574, 10: 0.794, 100: 1.526, 1000: 2.258}
THEORY_QUANTILE = {2: 0.859, 5: 1.151, 10: 1.371, 100: 2.103, 1000: 2.835}


def sample_max_of_N_exp1(N: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Exact samples of max(N iid Exp(1)) via inverse CDF, numerically stable for large N."""
    U = rng.random(n_trials)
    # max ~ F(x) = (1 - e^{-x})^N. ppf: x = -log(1 - U^{1/N}) = -log(-expm1(log U / N)).
    return -np.log(-np.expm1(np.log(U) / N))


def inverse_ols_predict(topk: np.ndarray, a: np.ndarray, r: float) -> np.ndarray:
    """Inverse-OLS FRLMB prediction at log-rank offset r.

    Regresses log-rank a (dependent) on score x (independent) over the top-k
    pairs, then solves the fitted line for the score at a = r.

    topk : (n_trials, k) descending order stats X_{1:m}^downarrow >= ... >= X_{k:m}^downarrow
    a    : (k,) nominal log-rank offsets a_j = -log j
    r    : log(N/m)
    """
    a_bar = a.mean()
    x_bar = topk.mean(axis=1)
    centered_x = topk - x_bar[:, None]
    S_xx = (centered_x ** 2).sum(axis=1)
    S_ax = ((a - a_bar)[None, :] * centered_x).sum(axis=1)
    return x_bar + (r - a_bar) * S_xx / S_ax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=20000)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    k, m, n_trials = args.k, args.m, args.n_trials
    a = -np.log(np.arange(1, k + 1, dtype=float))

    print(f"k={k}, m={m}, n_trials={n_trials}, chunk={args.chunk}", flush=True)

    t0 = time.time()
    print(f"Generating top-{k} of {m} Exp(1) per trial...", flush=True)
    topk_all = np.empty((n_trials, k), dtype=float)
    for i in range(0, n_trials, args.chunk):
        c = min(args.chunk, n_trials - i)
        X = rng.exponential(scale=1.0, size=(c, m))
        topk_all[i : i + c] = -np.sort(-np.partition(X, m - k, axis=1)[:, m - k :], axis=1)
    t_fit = time.time() - t0
    print(f"  done in {t_fit:.1f}s", flush=True)

    rows = []
    header = (
        f"\n{'R':>5} | {'theory_real':>11} | "
        f"{'empirical_realized':>26} | "
        f"{'theory_quan':>11} | {'empirical_quantile':>26}"
    )
    print(header)
    print("-" * len(header))

    for R, thy_r, thy_q in (
        (R, THEORY_REALIZED[R], THEORY_QUANTILE[R])
        for R in (2, 5, 10, 100, 1000)
    ):
        N = m * R
        r = np.log(R)
        Q_pred = inverse_ols_predict(topk_all, a, r)
        Y_max = sample_max_of_N_exp1(N, n_trials, rng)
        q_pop = np.log(N)
        err_r = Q_pred - Y_max
        err_q = Q_pred - q_pop
        emp_r = float(err_r.mean())
        emp_q = float(err_q.mean())
        se_r = float(err_r.std(ddof=1) / np.sqrt(n_trials))
        se_q = float(err_q.std(ddof=1) / np.sqrt(n_trials))
        gap_r_sigmas = (emp_r - thy_r) / se_r
        gap_q_sigmas = (emp_q - thy_q) / se_q
        print(
            f"{R:>5} | {thy_r:>11.3f} | "
            f"{emp_r:>10.4f} +/- {se_r:.4f} ({gap_r_sigmas:+.1f}sigma) | "
            f"{thy_q:>11.3f} | "
            f"{emp_q:>10.4f} +/- {se_q:.4f} ({gap_q_sigmas:+.1f}sigma)"
        )
        rows.append(
            {
                "R": R,
                "r": float(r),
                "N": N,
                "theory_b_realized": thy_r,
                "theory_b_quantile": thy_q,
                "empirical_b_realized": emp_r,
                "empirical_b_quantile": emp_q,
                "se_realized": se_r,
                "se_quantile": se_q,
                "gap_realized": emp_r - thy_r,
                "gap_quantile": emp_q - thy_q,
                "gap_realized_sigmas": gap_r_sigmas,
                "gap_quantile_sigmas": gap_q_sigmas,
            }
        )

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
