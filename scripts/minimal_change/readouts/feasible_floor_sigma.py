"""Sigma-constrained feasible floor.

Follow-up to feasible_floor_sweep.py. There, the best safety-feasible straight
tail sat at the slope-grid edge (10x the base line's slope) for every pool:
under "no prompt raised" alone, the error keeps falling as the tail is
compressed and pushed down, toward the flattened tail -- the degenerate
solution the sigma floor exists to forbid. This script adds the trainer's
sigma floor as a constraint.

Per base pool and geometry (M, N, k), on one shared set of S random splits:
  sigma(v)  = E_split[ std of v over the involved set ], involved set = fit
              top-k U deploy top-e, e = count_extrapolated_eval_points(M, N,
              weibull): the set the trainer's hinge reads (train_lquant.py,
              sigma-floor block of the bootstrap path).
  sigma_ref = sigma(base-level line): dynamics_probe._involved_sigma_reference
              on linearize(anchor), here with the line fit over the top-L ranks.
  For slope s, the shallowest safety-feasible line has intercept
  a_safe(s) = min_r (y_r + s xb_r). Lowering the intercept multiplies every
  value, the forecast error, the null error, and sigma by the same factor
  exp(-da/s), so the best line at slope s under sigma >= target is the safety
  line deepened by max(1, target / sigma_safe(s)), and its error is
  err_safe(s) times that factor. We minimize over s.
Constraints: one-sided margin 0 (sigma >= sigma_ref) and 0.13 nats (the rt
runs), two-sided margin 0.13 (the "sym" arm). Also reported: the fixed-slope
line (s = s0), which is pure suppression with no compression.
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feasible_floor_sweep import EPS, LN10, READOUTS, REPO_ROOT, forecast_signed, load_fullpool, load_rt_pools  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))
from forecast_experiment.dynamics_probe import _involved_sigma_reference, linearize  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402

OUT = READOUTS / "feasible_floor_sigma.json"
S = 200
CONSTRAINTS = {  # name: (margin, two_sided)
    "one_m0": (0.0, False),
    "one_m013": (0.13, False),
    "sym_m013": (0.13, True),
}
GRID = np.exp(np.linspace(math.log(0.1), math.log(1000.0), 121))  # s / s0


def extrap_k(M, N):
    return count_extrapolated_eval_points(M, N, plotting="weibull")


def sample_tables(n, M, N, k, e, rng):
    """Exact selection sampling of a random (fit M, deploy N, unused rest)
    assignment over the ranks of a sorted pool, vectorized over S splits;
    returns each split's fit top-k ranks and deploy top-e ranks."""
    fit_rem = np.full(S, M, dtype=np.int64)
    dep_rem = np.full(S, N, dtype=np.int64)
    fr = np.zeros((S, k), dtype=np.int64)
    dr = np.zeros((S, e), dtype=np.int64)
    nf = np.zeros(S, dtype=np.int64)
    nd = np.zeros(S, dtype=np.int64)
    rows = np.arange(S)
    for r in range(n):
        u = rng.random(S) * (n - r)
        is_fit = u < fit_rem
        is_dep = (~is_fit) & (u < fit_rem + dep_rem)
        need_f = is_fit & (nf < k)
        fr[rows[need_f], nf[need_f]] = r
        nf += need_f
        need_d = is_dep & (nd < e)
        dr[rows[need_d], nd[need_d]] = r
        nd += need_d
        fit_rem -= is_fit
        dep_rem -= is_dep
        if (r & 31) == 31 and (nf >= k).all() and (nd >= e).all():
            break
    assert (nf >= k).all() and (nd >= e).all()
    return fr, dr


def line(y, s, a):
    return -np.exp(-(a - y) / s)


def stats(v, fr, dr, M, N):
    err = forecast_signed(v, fr, dr[:, 0], M, N)
    null = (v[fr[:, 0]] - v[dr[:, 0]]) / LN10
    inv = np.concatenate([v[fr], v[dr]], axis=1)
    return {
        "abs": float(np.abs(err).mean()), "signed": float(err.mean()), "mse": float((err ** 2).mean()),
        "null_abs": float(np.abs(null).mean()), "null_mse": float((null ** 2).mean()),
        "sigma": float(inv.std(axis=1).mean()), "top_nats": float(-v[0]),
    }


def scaled(st, c):
    """Stats of the same line deepened by factor c (every value times c)."""
    out = dict(st)
    for key in ("abs", "signed", "null_abs", "sigma", "top_nats"):
        out[key] = st[key] * c
    for key in ("mse", "null_mse"):
        out[key] = st[key] * c * c
    return out


def skill(st):
    return 1.0 - st["mse"] / st["null_mse"] if st["null_mse"] > 0 else float("nan")


def evaluate_pool(v_sorted, M, N, k, rng):
    n = v_sorted.size
    e = extrap_k(M, N)
    fr, dr = sample_tables(n, M, N, k, e, rng)
    used = int(max(fr.max(), dr.max())) + 1
    L = min(n, int(math.ceil(1.25 * used)) + 5)
    vtop = v_sorted[:L]
    y = np.log(np.arange(1, L + 1) / (n + 1.0))
    xb = -np.log(np.maximum(-vtop, EPS))
    base = stats(vtop, fr, dr, M, N)
    b, a0 = np.polyfit(xb, y, 1)
    s0 = -float(b)
    f0 = stats(line(y, s0, a0), fr, dr, M, N)
    sref = f0["sigma"]

    def safe(ratio):
        s = s0 * ratio
        v = line(y, s, float(np.min(y + s * xb)))
        assert np.all(v <= vtop + 1e-12)
        return stats(v, fr, dr, M, N)

    def objective(st, margin, two_sided):
        if two_sided and st["sigma"] > sref + margin:
            return math.inf, None
        c = max(1.0, (sref - margin) / st["sigma"])
        return st["abs"] * c, c

    coarse = [safe(r) for r in GRID]
    out = {
        "n": n, "e": e, "L": L, "used": used, "sigma_ref": sref,
        "base": base, "floor0": f0, "base_skill": skill(base), "floor0_skill": skill(f0),
        "fixed_slope": safe(1.0),
        "safe_10x": coarse[int(np.argmin(np.abs(GRID - 10.0)))],
        "safe_1000x": coarse[-1],
    }
    for name, (margin, two_sided) in CONSTRAINTS.items():
        vals = [objective(st, margin, two_sided)[0] for st in coarse]
        i = int(np.argmin(vals))
        if not math.isfinite(vals[i]):
            out[name] = None
            continue
        lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, GRID.size - 1)]
        best = (vals[i], GRID[i], coarse[i])
        for r in np.exp(np.linspace(math.log(lo), math.log(hi), 41)):
            st = safe(r)
            val = objective(st, margin, two_sided)[0]
            if val < best[0]:
                best = (val, r, st)
        _, r_best, st = best
        c = objective(st, margin, two_sided)[1]
        res = scaled(st, c)
        res.update({"slope_ratio": float(r_best), "deepen": float(c), "skill": skill(res),
                    "at_edge": bool(i in (0, GRID.size - 1)), "sigma_binds": bool(c > 1.0 + 1e-9)})
        out[name] = res
    return out


# ----------------------------------------------------------------- validate
def validate(pools):
    """(1) the involved-set sigma against the trainer's reference function at
    the rt geometry, replaying its own 400 partitions (seed 1234) on its own
    line (linearize, m = 250); (2) the scaling identity the constrained
    optimum rests on."""
    M, N, k = 250, 2050, 10
    e = extrap_k(M, N)
    for name in list(pools)[:3]:
        p = pools[name]
        n = p.size
        lin = linearize(p, m=250)
        order = np.argsort(-lin, kind="stable")
        rank_of = np.empty(n, dtype=np.int64)
        rank_of[order] = np.arange(n)
        v0 = lin[order]
        rng = np.random.default_rng(1234)
        fr = np.empty((400, k), dtype=np.int64)
        dr = np.empty((400, e), dtype=np.int64)
        for d in range(400):
            fit = rng.choice(n, size=M, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[fit] = True
            fr[d] = np.sort(rank_of[fit])[:k]
            dr[d] = np.sort(rank_of[~mask])[:e]
        mine = stats(v0, fr, dr, M, N)["sigma"]
        lib = _involved_sigma_reference(p, n_fit=M, fit_top_k=k, extrap_k=e)
        print(f"sigma_ref check {name}: mine {mine:.6f} vs trainer reference {lib:.6f} "
              f"(rel diff {mine / lib - 1:+.1e}); e = {e}", flush=True)
        assert abs(mine / lib - 1) < 1e-9
        st = stats(v0, fr, dr, M, N)
        st2 = stats(1.7 * v0, fr, dr, M, N)
        for key in ("abs", "sigma", "null_abs"):
            assert abs(st2[key] / st[key] - 1.7) < 1e-9, key


# --------------------------------------------------------------------- jobs
def rt_cell(args):
    pools, M, N, k, seed = args
    rng = np.random.default_rng(seed)
    return {"pop": "rt", "M": M, "N": N, "k": k,
            "per_pool": {name: evaluate_pool(np.sort(p)[::-1], M, N, k, rng) for name, p in pools.items()}}


def full_cell(args):
    full, M, R, k, seed = args
    N = int(round(R * M))
    n = M + N
    P = int(min(16, max(4, full.size // n)))
    rng = np.random.default_rng(seed)
    rows = [evaluate_pool(np.sort(full[rng.choice(full.size, size=n, replace=False)])[::-1], M, N, k, rng)
            for _ in range(P)]
    return {"pop": "full", "M": M, "N": N, "R": R, "k": k, "P": P, "rows": rows}


def summarize(rows):
    def mean(f):
        vals = [f(r) for r in rows]
        vals = [x for x in vals if x is not None and math.isfinite(x)]
        return float(np.mean(vals)) if vals else float("nan")

    def cons(name, key):
        return mean(lambda r: r[name][key] if r[name] is not None else None)

    d = {
        "base": mean(lambda r: r["base"]["abs"]), "base_sd": float(np.std([r["base"]["abs"] for r in rows])),
        "floor0": mean(lambda r: r["floor0"]["abs"]), "sigma_ref": mean(lambda r: r["sigma_ref"]),
        "base_null": mean(lambda r: r["base"]["null_abs"]), "base_skill": mean(lambda r: r["base_skill"]),
        "fixed": mean(lambda r: r["fixed_slope"]["abs"]), "fixed_top": mean(lambda r: r["fixed_slope"]["top_nats"]),
        "safe10": mean(lambda r: r["safe_10x"]["abs"]), "safe1000": mean(lambda r: r["safe_1000x"]["abs"]),
        "safe10_sig": mean(lambda r: r["safe_10x"]["sigma"] / r["sigma_ref"]),
    }
    for name in CONSTRAINTS:
        d[name] = cons(name, "abs")
        d[name + "_skill"] = cons(name, "skill")
        d[name + "_null"] = cons(name, "null_abs")
        d[name + "_top"] = cons(name, "top_nats")
        d[name + "_slope"] = cons(name, "slope_ratio")
        d[name + "_edge"] = mean(lambda r, nm=name: float(r[nm]["at_edge"]) if r[nm] is not None else None)
        d[name + "_binds"] = mean(lambda r, nm=name: float(r[nm]["sigma_binds"]) if r[nm] is not None else None)
        d[name + "_feasible"] = float(np.mean([r[name] is not None for r in rows]))
        diffs = [r["base"]["abs"] - r[name]["abs"] for r in rows if r[name] is not None]
        d[name + "_gain_sd"] = float(np.std(diffs)) if diffs else float("nan")
    return d


def print_row(tag, d):
    print(f"{tag}  base {d['base']:.3f} (sd {d['base_sd']:.2f}, null {d['base_null']:.2f}, skill {d['base_skill']:+.2f})"
          f"  floor0 {d['floor0']:.3f}  sig_ref {d['sigma_ref']:.2f}"
          f"  | fixed-slope {d['fixed']:.3f} [top {d['fixed_top']:.1f}]"
          f"  safe10x {d['safe10']:.3f} (sig {d['safe10_sig']:.2f}ref) safe1000x {d['safe1000']:.3f}"
          f"  | one_m0 {d['one_m0']:.3f} (x{d['base'] / d['one_m0']:.1f}, skill {d['one_m0_skill']:+.2f}, "
          f"top {d['one_m0_top']:.1f}, slope x{d['one_m0_slope']:.2f}, edge {d['one_m0_edge']:.2f}, "
          f"binds {d['one_m0_binds']:.2f}, gain sd {d['one_m0_gain_sd']:.2f})"
          f"  one_m013 {d['one_m013']:.3f} (x{d['base'] / d['one_m013']:.1f})"
          f"  sym_m013 {d['sym_m013']:.3f} (feas {d['sym_m013_feasible']:.2f})", flush=True)


def main():
    t0 = time.time()
    rt = load_rt_pools()
    print(f"rt pools: {len(rt)} x n={sorted({p.size for p in rt.values()})}", flush=True)
    validate(rt)
    full = load_fullpool()

    rt_geoms = [(250, 2050, k) for k in (5, 10, 20, 50)]
    rt_geoms += [(100, 2200, k) for k in (5, 10, 20)]
    rt_geoms += [(500, 1800, k) for k in (10, 20, 50, 100)]
    rt_geoms += [(1000, 1300, k) for k in (10, 50, 100, 200)]
    rt_geoms += [(100, N, 10) for N in (200, 400, 800, 1600)]
    full_geoms = [(250, 8.2, 10)]
    for M in (44, 100, 250, 500, 1000, 2000, 4000):
        for R in (2, 4, 8, 16):
            if M * (1 + R) > full.size:
                continue
            for k in (5, 10, 20, 50, 100, 200, 400):
                if k <= M / 4:
                    full_geoms.append((M, R, k))

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rt_res = list(ex.map(rt_cell, [(rt, M, N, k, 7 + i) for i, (M, N, k) in enumerate(rt_geoms)]))
        print(f"rt cells done at {time.time() - t0:.0f}s", flush=True)
        full_res = list(ex.map(full_cell, [(full, M, R, k, 1000 + i) for i, (M, R, k) in enumerate(full_geoms)]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rt": rt_res, "full": full_res}, indent=1))

    print("\n== (A) real rt task pools (16 pools, n=2300; mean over pools; abs log10 error, decades)")
    for c in rt_res:
        rows = list(c["per_pool"].values())
        print_row(f"M={c['M']:5d} N={c['N']:5d} R={c['N'] / c['M']:5.1f} k={c['k']:4d} e={rows[0]['e']:3d}", summarize(rows))
        ho = [v for name, v in c["per_pool"].items() if name.startswith("heldout:")]
        print_row("      held-out 4 only                        ", summarize(ho))
    print("\n== (B) random pools from the full 85.6k pool (mean over P pools)")
    for c in full_res:
        print_row(f"M={c['M']:5d} R={c['R']:4} N={c['N']:6d} k={c['k']:4d} P={c['P']:2d}", summarize(c["rows"]))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
