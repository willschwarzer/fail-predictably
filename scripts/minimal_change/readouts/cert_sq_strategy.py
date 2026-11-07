"""Certificates under a squared-error objective, and the ceiling strategy built explicitly.

Two questions. (1) dynamics_probe.certified_tail_targets
minimizes the split-averaged ABS log10 error at the deploy worst, while every
readout reports the MSE fold; how much do the reference folds move if the
certificate minimizes squared error instead? (2) On password v2 a single
straight line over the top L ranks cannot keep the GCG successes at the ceiling
above a steepened bulk; what does that two-piece shape reach?

Part A, password v2 (the 0.6B strict-v2 warm-start fleet's 70 held-out pools of
935, M44/N891, k10; loader from ceiling_check.py): pooled MSE fold over base for
the certificate under {abs, sq} x {two-sided, one-sided}, for predict-1, and for
the strategy tail. The strategy keeps every prompt above p = 0.1 at its base
value and puts the ranks below them on the shallowest safe straight line in
(log position, x) whose slope minimizes that pool's split-averaged squared error,
subject to the one-sided floor (involved sigma >= reference - 0.13 nats). Ranks
past L drop by the gap at L (assign), as in the certificates. Part B, v2e2 (32 pools of
2300, M250/N2050, k10 and k50): the same four certificates. Part C prints one
password v2 pool's tail under each construction.

  python cert_sq_strategy.py

Inputs: the password strict-v2 warm-start 0.6B IOG fleet ($RESULTS_ROOT/FLEET below), whose
launcher is not in this release; Part B reads the v2e2 bank and full-pool scores (g_transfer_v2).
"""
import glob
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g_transfer_k as gk  # noqa: E402
from ceiling_check import RES, load_traces  # noqa: E402
from g_transfer_v2 import pools_from_bank, smoke  # noqa: E402
from forecast_experiment.dynamics_probe import involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402

MARGIN = 0.13
LN10 = math.log(10.0)
FLEET = "lquant_40slots_gcg6warm_strictv2_iog_full_bootstrap1_cache296_seed*_20260724/seed_*"
SHOW_RANKS = (1, 2, 3, 4, 5, 10, 21, 50, 100, 200)


class Splits:
    """The certificate's internal splits and statistics (dynamics_probe.certified_tail_targets)."""

    def __init__(self, n, M, k, e, L, splits=200, seed=4321):
        rng = np.random.default_rng(seed)
        perms = np.stack([rng.permutation(n) for _ in range(splits)])  # rank ids
        fr = np.sort(perms[:, :M], axis=1)[:, :k]
        dr = np.sort(perms[:, M:], axis=1)[:, :e]
        keep = (fr.max(axis=1) < L) & (dr.max(axis=1) < L)
        self.fr, self.dr = fr[keep], dr[keep]
        self.y_fit = np.log(np.arange(1, k + 1) / (M + 1.0))
        self.log_inv_n = math.log(1.0 / (n - M))

    def stats(self, vals):
        """Abs and squared log10 error at the deploy worst, and the involved sigma (vals ranked)."""
        x = -np.log(np.maximum(-vals[self.fr], 1e-30))
        xm = x.mean(axis=1, keepdims=True)
        ym = self.y_fit.mean()
        sxx = ((x - xm) ** 2).sum(axis=1)
        sxy = ((x - xm) * (self.y_fit - ym)).sum(axis=1)
        ok = sxx > 1e-20
        b = np.where(ok, sxy / np.where(ok, sxx, 1.0), 0.0)
        a = ym - b * xm[:, 0]
        nz = b != 0.0
        xstar = np.where(nz, (self.log_inv_n - a) / np.where(nz, b, 1.0), x[:, 0])
        with np.errstate(over="ignore", invalid="ignore"):
            pred = -np.exp(-xstar)
            d = (pred - vals[self.dr[:, 0]]) / LN10
            ab, sq = float(np.abs(d).mean()), float((d ** 2).mean())
        inv = np.concatenate([vals[self.fr], vals[self.dr]], axis=1)
        return ab, sq, float(inv.std(axis=1).mean())


def geometry(v, M, k):
    n = v.size
    e = count_extrapolated_eval_points(M, n - M, plotting="weibull")
    L = gk.reach(n, M, k, e)
    sref = involved_sigma_reference(v, n_fit=M, fit_top_k=k, extrap_k=e, m=L)
    vtop = np.sort(v)[::-1][:L]
    y = np.log(np.arange(1, L + 1) / (n + 1.0))
    xb = -np.log(np.maximum(-vtop, 1e-30))
    return n, e, L, sref, vtop, y, xb, Splits(n, M, k, e, L)


def assign(v, tgt):
    """Give the top-len(tgt) prompts the ranked targets; ranks past it drop by the gap at the last one."""
    L = tgt.size
    order = np.argsort(-v, kind="stable")
    out = v.copy()
    out[order[L:]] -= float(v[order[L - 1]] - tgt[-1])
    out[order[:L]] = tgt
    return out


def search(f, s0, lo, hi, npts):
    grid = s0 * np.exp(np.linspace(math.log(lo), math.log(hi), npts))
    vals = [f(float(s)) for s in grid]
    i = int(np.argmin(vals))
    best_val, best_s = vals[i], float(grid[i])
    for s in np.exp(np.linspace(math.log(grid[max(i - 1, 0)]), math.log(grid[min(i + 1, grid.size - 1)]), 41)):
        val = f(float(s))
        if val < best_val:
            best_val, best_s = val, float(s)
    return best_val, best_s


def cert(v, M, k, two_sided, obj):
    """certified_tail_targets with the objective switchable between abs and squared error."""
    n, e, L, sref, vtop, y, xb, sp = geometry(v, M, k)

    def safe(s):
        a = float(np.min(y + s * xb))
        return -np.exp(-(a - y) / s)

    def objective(s):
        ab, sq, sig = sp.stats(safe(s))
        if two_sided and sig > sref + MARGIN:
            return math.inf, 1.0
        c = max(1.0, (sref - MARGIN) / max(sig, 1e-12))  # deepening scales error by c, squared error by c^2
        val = ab * c if obj == "abs" else sq * c * c
        return (val if math.isfinite(val) else math.inf), c

    s0 = -float(np.polyfit(xb, y, 1)[0])
    best_val, best_s = search(lambda s: objective(s)[0], s0, 0.1, 1000.0, 121)
    if not math.isfinite(best_val):
        raise RuntimeError("no slope satisfies the band")
    tgt = safe(best_s) * objective(best_s)[1]
    assert np.all(tgt <= vtop + 1e-9)
    return assign(v, tgt), best_s / s0, sref, e


def strategy(v, M, k, thr=math.log(0.1)):
    """Prompts above p = thr stay at base; the ranks below follow a safe straight line, slope by squared error."""
    n, e, L, sref, vtop, y, xb, sp = geometry(v, M, k)
    g = int((vtop > thr).sum())
    yb, xbb = y[g:], xb[g:]

    def tail(s):
        a = float(np.min(yb + s * xbb))
        out = vtop.copy()
        out[g:] = -np.exp(-(a - yb) / s)
        return out

    def objective(s):
        ab, sq, sig = sp.stats(tail(s))
        if sig < sref - MARGIN or not math.isfinite(sq):
            return math.inf
        return sq

    s0 = -float(np.polyfit(xbb, yb, 1)[0])
    best_val, best_s = search(objective, s0, 0.003, 100.0, 181)
    if not math.isfinite(best_val):
        return None, g, sref, e, float("nan")
    tgt = tail(best_s)
    assert np.all(tgt <= vtop + 1e-9)
    return assign(v, tgt), g, sref, e, best_s / s0


def involved_sigma(v, M, k, e, splits=200, seed=4321):
    rng = np.random.default_rng(seed)
    ranked = np.sort(v)[::-1]
    perms = np.stack([rng.permutation(v.size) for _ in range(splits)])
    fr = np.sort(perms[:, :M], axis=1)[:, :k]
    dr = np.sort(perms[:, M:], axis=1)[:, :e]
    return float(np.concatenate([ranked[fr], ranked[dr]], axis=1).std(axis=1).mean())


def predict_one(v, M, seed):
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(v.size) for _ in range(gk.S)])
    err = -v[perms[:, M:]].max(axis=1) / LN10
    return np.abs(err).mean(), (err ** 2).mean()


def fold_line(label, mse, ab, base_mse, base_ab, extra=""):
    print(f"  {label:<28s} pooled MSE {mse:9.4f}  fold {base_mse / mse:8.1f}x | mean abs {ab:6.3f}  "
          f"abs fold {base_ab / ab:5.2f}x{extra}")


def part_a():
    pools = {}
    for d in sorted(glob.glob(f"{RES}/{FLEET}")):
        for name, v in load_traces(d).items():
            pools[f"{Path(d).name}:{name}"] = v
    M, k = 44, 10
    rows = {key: ([], []) for key in ("base", "abs two", "abs one", "sq two", "sq one", "strategy", "predict-1")}
    strat_sig, strat_g, strat_none, shown = [], [], 0, False
    for j, (n_, v) in enumerate(pools.items()):
        seed = 7 + j
        b = gk.score(v, M, k, seed)
        rows["base"][0].append(b[1]); rows["base"][1].append(b[0])
        tails = {"base": v}
        for obj in ("abs", "sq"):
            for two in (True, False):
                out, sr, sref, e = cert(v, M, k, two, obj)
                c = gk.score(out, M, k, seed)
                key = f"{obj} {'two' if two else 'one'}"
                rows[key][0].append(c[1]); rows[key][1].append(c[0])
                tails[key] = out
        out, g, sref, e, sr = strategy(v, M, k)
        strat_g.append(g)
        if out is None:
            strat_none += 1
            out = tails["sq one"]
        else:
            strat_sig.append(involved_sigma(out, M, k, e) / sref)
        c = gk.score(out, M, k, seed)
        rows["strategy"][0].append(c[1]); rows["strategy"][1].append(c[0])
        tails["strategy"] = out
        p1 = predict_one(v, M, seed)
        rows["predict-1"][0].append(p1[1]); rows["predict-1"][1].append(p1[0])
        if not shown and int((v > math.log(0.5)).sum()) >= 2:
            shown = True
            show = (n_, tails, {key: (rows[key][1][-1], rows[key][0][-1]) for key in rows})
    bm, ba = float(np.mean(rows["base"][0])), float(np.mean(rows["base"][1]))
    print(f"== Part A: password v2, {len(pools)} pools of 935, M{M}/N891, k{k}")
    for key, (m, a) in rows.items():
        extra = ""
        if key == "strategy":
            extra = (f" | kept-at-base prompts per pool {np.mean(strat_g):.2f}; involved sigma / reference "
                     f"median {np.median(strat_sig):.1f} (min {np.min(strat_sig):.2f}); no feasible slope on {strat_none}")
        fold_line(key, float(np.mean(m)), float(np.mean(a)), bm, ba, extra)
    name, tails, errs = show
    print(f"\n== Part C: {name}, log10 p by base rank (columns), and that pool's abs / squared error")
    print("  " + " " * 12 + "".join(f"{r:>8d}" for r in SHOW_RANKS) + "      abs      sq")
    for key, t in tails.items():
        ranked = np.sort(t)[::-1] / LN10
        print(f"  {key:<12s}" + "".join(f"{ranked[r - 1]:8.2f}" for r in SHOW_RANKS)
              + f"  {errs[key][0]:7.3f} {errs[key][1]:7.3f}")


def part_b():
    meta = smoke()
    pools = {n: x for n, x in pools_from_bank(meta).items() if x.size == 2300}
    M = 250
    print(f"\n== Part B: v2e2, {len(pools)} pools of 2300, M{M}/N2050")
    for k in (10, 50):
        acc = {key: ([], []) for key in ("base", "abs two", "abs one", "sq two", "sq one")}
        for j, v in enumerate(pools.values()):
            b = gk.score(v, M, k, 7 + j)
            acc["base"][0].append(b[1]); acc["base"][1].append(b[0])
            for obj in ("abs", "sq"):
                for two in (True, False):
                    out, sr, sref, e = cert(v, M, k, two, obj)
                    c = gk.score(out, M, k, 7 + j)
                    key = f"{obj} {'two' if two else 'one'}"
                    acc[key][0].append(c[1]); acc[key][1].append(c[0])
        bm, ba = float(np.mean(acc["base"][0])), float(np.mean(acc["base"][1]))
        print(f" k{k}:")
        for key, (m, a) in acc.items():
            fold_line(key, float(np.mean(m)), float(np.mean(a)), bm, ba)


def main():
    part_a()
    part_b()


if __name__ == "__main__":
    main()
