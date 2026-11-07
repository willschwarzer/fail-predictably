"""Does ONE pool-independent suppression map reach the floor on held-out pools?

Fit a monotone map g: base log-prob -> certified-target log-prob on the train
pools' certified tails (isotonic, pooled over pools), apply it to the held-out
pools' base scores, and score the result with the trainer's forecast on 200
splits (k10 and k50, M250/N2050). Baselines: the base pool, the held-out
pool's own certified tail (the floor), and a uniform shift of the bulk drop.
Since g(v) <= v by construction of the targets, applying it raises no prompt.
Uses the 16 rt pools (12 train anchors of rtk16siganch8 seed 11, 4 held-out).
Also repeats with every split of the 16 pools into 12 fit / 4 apply (leave-4-out
over 20 random partitions) so the transfer number is not one partition's luck.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feasible_floor_sweep import REPO_ROOT, load_rt_pools  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))
from forecast_experiment.dynamics_probe import certified_tail_targets, involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

M, K, S = 250, 10, 200


def cert(v):
    e = count_extrapolated_eval_points(M, v.size - M, plotting="weibull")
    sref = involved_sigma_reference(v, n_fit=M, fit_top_k=K, extrap_k=e, m=250)
    idx, tgt, info = certified_tail_targets(v, n_fit=M, fit_top_k=K, extrap_k=e, sigma_ref=sref,
                                            sigma_margin=0.13, two_sided=True, max_rank=300)
    out = v.copy()
    # Below the certified ranks, shift rigidly by the rank-300 drop so order is kept.
    out[np.argsort(-v)[300:]] -= float(v[idx[-1]] - tgt[-1])
    out[idx] = tgt
    return out, idx, tgt


def score(v, seed):
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(v.size) for _ in range(S)])
    e10, _ = split_errors(v, perms, M, k=10)
    e50, _ = split_errors(v, perms, M, k=50)
    return np.abs(e10).mean(), (e10 ** 2).mean(), np.abs(e50).mean(), (e50 ** 2).mean()


def pava(y):
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    vals, wts, cnts = [], [], []
    for yi in y:
        vals.append(float(yi)); wts.append(1.0); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            w = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / w); wts.append(w); cnts.append(c1 + c2)
    return np.repeat(vals, cnts)


def fit_g(pools):
    xs, ys = [], []
    for v in pools:
        c, idx, tgt = cert(v)
        xs.append(v)
        ys.append(c)
    x, y = np.concatenate(xs), np.concatenate(ys)
    o = np.argsort(x, kind="stable")
    xs_sorted, fit = x[o], pava(y[o])
    lo = float(xs_sorted[0])
    shift_lo = float(fit[0] - lo)

    def g(v):
        out = np.interp(v, xs_sorted, fit)
        below = v < lo
        out[below] = v[below] + shift_lo
        return np.minimum(out, v)  # never raise a prompt

    bulk_drop = float(np.median(np.concatenate([a - b for a, b in zip(xs, ys)])))
    return g, bulk_drop


def evaluate(train, test, tag):
    g, bulk_drop = fit_g(list(train.values()))
    rows = []
    for i, (name, v) in enumerate(test.items()):
        c, _, _ = cert(v)
        rows.append((score(v, 7 + i), score(c, 7 + i), score(g(v), 7 + i), score(v - bulk_drop, 7 + i)))
    r = np.array(rows)  # (pools, 4 variants, 4 stats)
    m = r.mean(axis=0)
    labels = ["base", "own certified", "g from train", "uniform shift"]
    print(f"\n== {tag}: {len(test)} pools")
    for j, lab in enumerate(labels):
        print(f"  {lab:14s} k10 abs {m[j, 0]:.3f} mse {m[j, 1]:.3f} (fold {m[0, 1] / m[j, 1]:.2f}x) | "
              f"k50 abs {m[j, 2]:.3f} mse {m[j, 3]:.3f} (fold {m[0, 3] / m[j, 3]:.2f}x)")
    return m


def main():
    pools = load_rt_pools()
    train = {k: v for k, v in pools.items() if k.startswith("train:")}
    held = {k: v for k, v in pools.items() if k.startswith("heldout:")}
    evaluate(train, held, "fit g on the 12 train pools, apply to the 4 held-out pools")
    evaluate(train, train, "in-sample (fit and apply on the 12 train pools)")
    names = list(pools)
    rng = np.random.default_rng(0)
    ms = []
    for rep in range(20):
        perm = rng.permutation(len(names))
        tr = {names[i]: pools[names[i]] for i in perm[:12]}
        te = {names[i]: pools[names[i]] for i in perm[12:]}
        ms.append(evaluate(tr, te, f"partition {rep}") if rep < 2 else _quiet(tr, te))
    ms = np.array(ms)
    print("\n== 20 random 12/4 partitions, mean over partitions (k10 abs / mse fold; k50 abs / mse fold)")
    for j, lab in enumerate(["base", "own certified", "g from train", "uniform shift"]):
        print(f"  {lab:14s} k10 {ms[:, j, 0].mean():.3f} ({(ms[:, 0, 1] / ms[:, j, 1]).mean():.2f}x) "
              f"k50 {ms[:, j, 2].mean():.3f} ({(ms[:, 0, 3] / ms[:, j, 3]).mean():.2f}x)")


def _quiet(tr, te):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate(tr, te, "")


if __name__ == "__main__":
    main()
