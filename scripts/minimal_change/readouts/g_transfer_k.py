"""Honest k-sweep of the fixed-rule transfer (g), and the log-odds saturation test.

Part 1. For each geometry (M, k), certify every rt pool AT THAT k: the sigma
reference linearizes the top L ranks, L = 1.25x the deepest rank any split
reads (+5), instead of the trainer's fixed m = 250 (at k50 the involved set
reaches past rank 250, so a 250-rank reference reads the displaced bulk and is
inflated), and the certified tail covers the same L ranks. Fit one monotone map g (PAVA, pooled) from base log-prob to
certified log-prob on 12 pools, apply it to the other 4 pools' base scores, and
score with the instrument at the SAME k (no cross-k reads). 20 random 12/4
partitions of the 16 rt pools.

Part 2. For the finished seed-11 rt runs, the shift in event log-odds
u = v - log(1 - e^v) between base and trained, binned by base log-prob, on
held-out pools and on train pools (anchor vs last full dump), next to the shift
the k10 g asks for in the same bins. A trained shift that is flat in u across
bins is a near-uniform logit shift, which log-prob space reads as a small drop
at the ceiling.
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feasible_floor_sweep import REPO_ROOT, RUNS, load_rt_pools  # noqa: E402
from g_transfer import pava  # noqa: E402
from drop_by_rank import last_full_dump, traces  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))
from forecast_experiment.dynamics_probe import certified_tail_targets, involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

S = 200
GEOMS = [(250, 10), (250, 20), (250, 50), (500, 100)]
_CERT = {}


def reach(n, M, k, e, splits=200, seed=4321):
    """L for this geometry, from the same permutations certified_tail_targets draws."""
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(n) for _ in range(splits)])
    fr = np.sort(perms[:, :M], axis=1)[:, :k]
    dr = np.sort(perms[:, M:], axis=1)[:, :e]
    used = int(max(fr.max(), dr.max())) + 1
    return min(n, int(math.ceil(1.25 * used)) + 5)


def cert(name, v, M, k):
    key = (name, M, k)
    if key not in _CERT:
        e = count_extrapolated_eval_points(M, v.size - M, plotting="weibull")
        L = reach(v.size, M, k, e)
        sref = involved_sigma_reference(v, n_fit=M, fit_top_k=k, extrap_k=e, m=L)
        idx, tgt, info = certified_tail_targets(v, n_fit=M, fit_top_k=k, extrap_k=e, sigma_ref=sref,
                                                sigma_margin=0.13, two_sided=True, max_rank=L)
        out = v.copy()
        out[np.argsort(-v, kind="stable")[L:]] -= float(v[idx[-1]] - tgt[-1])
        out[idx] = tgt
        _CERT[key] = (out, L, info)
    return _CERT[key]


def fit_g(pairs):
    x = np.concatenate([a for a, _ in pairs])
    y = np.concatenate([b for _, b in pairs])
    o = np.argsort(x, kind="stable")
    xs, fit = x[o], pava(y[o])
    lo, shift_lo = float(xs[0]), float(fit[0] - xs[0])

    def g(v):
        out = np.interp(v, xs, fit)
        out[v < lo] = v[v < lo] + shift_lo
        return np.minimum(out, v)

    bulk = float(np.median(x - y))
    return g, bulk


def score(v, M, k, seed):
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(v.size) for _ in range(S)])
    e, _ = split_errors(v, perms, M, k=k)
    return np.abs(e).mean(), (e ** 2).mean()


def part1(pools):
    names = list(pools)
    print("== Part 1: g transfer, certified and read at the same k (20 random 12/4 partitions of 16 rt pools)")
    for M, k in GEOMS:
        rows = []
        for nm in names:
            c, L, info = cert(nm, pools[nm], M, k)
            rows.append((L, info["base_abs_log10"], info["cert_abs_log10"], info["top_target_nats"], info["sigma_ref"]))
        r = np.array(rows)
        rng = np.random.default_rng(0)
        parts = []
        for _ in range(20):
            perm = rng.permutation(len(names))
            tr, te = [names[i] for i in perm[:12]], [names[i] for i in perm[12:]]
            g, bulk = fit_g([(pools[n], cert(n, pools[n], M, k)[0]) for n in tr])
            m = []
            for j, n in enumerate(te):
                v = pools[n]
                c = cert(n, v, M, k)[0]
                m.append([score(x, M, k, 7 + j) for x in (v, c, g(v), v - bulk)])
            parts.append(np.mean(np.array(m), axis=0))  # (4 variants, abs/mse)
        p = np.array(parts)
        print(f"\nM{M} k{k}: L {int(r[:, 0].min())}-{int(r[:, 0].max())}, sigma_ref {r[:, 4].mean():.2f} nats, "
              f"certified top {r[:, 3].mean():+.2f} nats (in-search abs {r[:, 1].mean():.3f} -> {r[:, 2].mean():.3f})")
        for j, lab in enumerate(["base", "own certified", "g from 12", "uniform shift"]):
            fold = (p[:, 0, 1] / p[:, j, 1])
            print(f"  {lab:14s} abs {p[:, j, 0].mean():.3f}  mse {p[:, j, 1].mean():.3f}  "
                  f"mse fold {fold.mean():.2f}x (partition range {fold.min():.2f}-{fold.max():.2f})")


def logodds(v):
    v = np.minimum(v, -1e-6)
    return v - np.log(-np.expm1(v))


BINS = [-np.inf, -10, -6, -4, -2, -1, -0.3, 0.01]


def binned(base, trained, g):
    du = logodds(base) - logodds(trained)
    dv = base - trained
    dg = logodds(base) - logodds(g(base))
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (base > lo) & (base <= hi)
        out.append((int(m.sum()), float(dv[m].mean()) if m.any() else np.nan,
                    float(du[m].mean()) if m.any() else np.nan, float(dg[m].mean()) if m.any() else np.nan))
    return out


def part2(pools):
    print("\n== Part 2: shift in event log-odds by base log-prob bin (k10 g fit on the 12 seed-11 train pools)")
    train = {n: v for n, v in pools.items() if n.startswith("train:")}
    g, _ = fit_g([(v, cert(n, v, 250, 10)[0]) for n, v in train.items()])
    hdr = "  ".join(f"({BINS[i]:g},{BINS[i + 1]:g}]" for i in range(len(BINS) - 1))
    print(f"  bins of base log-prob: {hdr}")
    for name in ["lqdyn_rtsig-anchpen_seed11", "lqdyn_rtsig-anchpensym_seed11", "lqdyn_rtsig-sym_seed11",
                 "lqdyn_rtk16siganch8_seed11"]:
        run = RUNS / name
        if not (run / "test_forecast_traces.npz").exists():
            continue
        b, t = traces(run, "baseline_"), traces(run, "")
        held = [binned(b[k], t[k], g) for k in b]
        step, tr = last_full_dump(run)
        trn = [binned(a, u, g) for a, u in tr.values()]
        for tag, rows in (("held-out", held), (f"train {step}", trn)):
            a = np.array(rows, dtype=np.float64)  # (pools, bins, 4)
            n = a[:, :, 0].sum(axis=0)
            w = np.nan_to_num(a[:, :, 0])
            mean = lambda c: (np.nansum(a[:, :, c] * w, axis=0) / np.maximum(n, 1))
            dv, du, dg = mean(1), mean(2), mean(3)
            print(f"\n{name} {tag}: n per bin {' '.join(str(int(x)) for x in n)}")
            print("   drop in log-prob (nats): " + "  ".join(f"{x:6.2f}" for x in dv))
            print("   shift in log-odds      : " + "  ".join(f"{x:6.2f}" for x in du))
            print("   g asks (log-odds)      : " + "  ".join(f"{x:6.2f}" for x in dg))


def main():
    pools = load_rt_pools()
    part2(pools)
    part1(pools)


if __name__ == "__main__":
    main()
