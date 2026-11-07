"""Safety-feasible floor sweep.

Question: at which geometries (fit size M, deploy size N = R*M, top-k) would a
model whose tail is exactly straight -- and which raised no prompt above its
base score -- forecast clearly better than the base model?

Per base pool and geometry, on one shared set of random fit/deploy splits:
  base     : the base pool's split-averaged abs log10 worst-prompt error
  floor0   : the pool with its top-L order statistics replaced by the OLS line
             through them (dynamics_probe.linearize's construction, global
             Weibull positions). This is the base-level straight floor; on the
             rt pools it raises ~40% of the top ranks, so the improving-only
             guarantee cannot reach it.
  feasible : the best straight line lying at or below every base order
             statistic over the top-L ranks, searched over slope. Rank-wise
             dominance is necessary and sufficient for a per-prompt assignment
             that raises no prompt (assign line values in base rank order).
L covers every rank any split's fit top-k or deploy max uses, plus 25%.
Bulk ranks never enter any split's statistics, so they are left implicit.

Populations: (A) the 16 real rt task pools (n = 2300 each; 12 train-target
anchors from a step-0 dump + 4 held-out base traces); (B) random pools of size
M + N drawn from the 85,601-prompt full flagged-EN pool (same score, same
38-token union).
"""
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
RUNS = RESULTS_ROOT / "lquant_dynamics/runs"
READOUTS = RESULTS_ROOT / "lquant_dynamics/readouts"
# Full-pool scores from scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh.
FULLPOOL = Path(os.environ.get("LOGIT_CACHE_ROOT", RESULTS_ROOT / "wildchat_fullpool_logits")) / "Qwen3-0.6B/harmful"
FULLPOOL_CSV = FULLPOOL / "wildchat_harmful_scores.csv"
OUT = READOUTS / "feasible_floor_sweep.json"
LN10 = math.log(10.0)
EPS = 1e-8
S = 200  # splits per pool


# --------------------------------------------------------------------- data
def load_rt_pools():
    run = RUNS / "lqdyn_rtk16siganch8_seed11"
    pools = {}
    z = np.load(run / "dynamics" / "step_0000_rank0.npz", allow_pickle=True)
    tg = sorted({int(x.split("/")[0]) for x in z.files if "/" in x})
    assert all(int(z[f"{t}/union_is_full"]) == 1 for t in tg)
    part = json.loads((run / "target_partition.json").read_text())
    for t in tg:
        pools[f"train:{part['train_targets'][t]}"] = np.asarray(z[f"{t}/union_lp"], dtype=np.float64)
    meta = json.loads((run / "baseline_test_forecast_traces_metadata.json").read_text())
    zb = np.load(run / "baseline_test_forecast_traces.npz")
    for t in meta["targets"]:
        pre = t["array_prefix"]
        pools[f"heldout:{t['target']}"] = np.concatenate(
            [zb[f"{pre}__fit_log_probs"], zb[f"{pre}__deploy_log_probs"]]).astype(np.float64)
    return pools


def load_fullpool():
    with open(FULLPOOL_CSV) as f:
        v = np.array([float(r["log_p_harmful"]) for r in csv.DictReader(f)], dtype=np.float64)
    return v[np.isfinite(v)]


# --------------------------------------------------------------- forecaster
def sample_tables(n, M, N, k, rng):
    """Exact selection sampling of a random (fit M, deploy N, unused rest)
    assignment over the ranks of a sorted pool, vectorized over S splits and
    stopped once every split has its fit top-k and its deploy max."""
    fit_rem = np.full(S, M, dtype=np.int64)
    dep_rem = np.full(S, N, dtype=np.int64)
    fit_ranks = np.zeros((S, k), dtype=np.int64)
    nfound = np.zeros(S, dtype=np.int64)
    dep_rank = np.full(S, -1, dtype=np.int64)
    rows = np.arange(S)
    for r in range(n):
        u = rng.random(S) * (n - r)
        is_fit = u < fit_rem
        is_dep = (~is_fit) & (u < fit_rem + dep_rem)
        need = is_fit & (nfound < k)
        fit_ranks[rows[need], nfound[need]] = r
        nfound += need
        dep_rank[is_dep & (dep_rank < 0)] = r
        fit_rem -= is_fit
        dep_rem -= is_dep
        if (r & 31) == 31 and (nfound >= k).all() and (dep_rank >= 0).all():
            break
    assert (nfound >= k).all() and (dep_rank >= 0).all()
    return fit_ranks, dep_rank


def forecast_signed(vtop, fit_ranks, dep_rank, M, N):
    """Vectorized fit_ols_tail_from_scores (weibull, gumbel_log_prob) on a
    rank-sorted pool; returns signed log10 error (forecast minus actual)."""
    x = -np.log(np.maximum(-vtop[fit_ranks], EPS))  # (S, k), descending
    k = x.shape[1]
    y = np.log(np.arange(1, k + 1) / (M + 1.0))
    xm = x.mean(axis=1, keepdims=True)
    ym = y.mean()
    sxx = ((x - xm) ** 2).sum(axis=1)
    sxy = ((x - xm) * (y - ym)).sum(axis=1)
    ok = sxx > 1e-20
    b = np.where(ok, sxy / np.where(ok, sxx, 1.0), 0.0)
    a = ym - b * xm[:, 0]
    nz = b != 0.0
    xstar = np.where(nz, (math.log(1.0 / N) - a) / np.where(nz, b, 1.0), x[:, 0])
    with np.errstate(over="ignore"):
        pred = -np.exp(-xstar)
    return (pred - vtop[dep_rank]) / LN10


def abs_mean(e):
    return float(np.mean(np.abs(e)))


# --------------------------------------------------------------- the lines
def line_values(vtop, n, s, a):
    y = np.log(np.arange(1, vtop.size + 1) / (n + 1.0))
    return -np.exp(-(a - y) / s)


def base_level_line(vtop, n):
    x = -np.log(np.maximum(-vtop, EPS))
    y = np.log(np.arange(1, vtop.size + 1) / (n + 1.0))
    b, a = np.polyfit(x, y, 1)  # y = b x + a, b < 0
    s = -b
    return line_values(vtop, n, s, a), s


def feasible_line(vtop, n, s):
    xb = -np.log(np.maximum(-vtop, EPS))
    y = np.log(np.arange(1, vtop.size + 1) / (n + 1.0))
    a = float(np.min(y + s * xb))
    v = line_values(vtop, n, s, a)
    return v


def evaluate_pool(v_sorted, M, N, k, rng):
    n = v_sorted.size
    fr, dr = sample_tables(n, M, N, k, rng)
    used = int(max(fr.max(), dr.max())) + 1
    L = min(n, int(math.ceil(1.25 * used)) + 5)
    vtop = v_sorted[:L]
    e_base = forecast_signed(vtop, fr, dr, M, N)
    v0, s0 = base_level_line(vtop, n)
    e0 = forecast_signed(v0, fr, dr, M, N)

    def feas_err(s):
        vf = feasible_line(vtop, n, s)
        return abs_mean(forecast_signed(vf, fr, dr, M, N)), vf

    grid = s0 * np.exp(np.linspace(math.log(0.1), math.log(10.0), 81))
    errs = [feas_err(s)[0] for s in grid]
    i = int(np.nanargmin(errs))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    fine = np.linspace(lo, hi, 41)
    ferrs = [feas_err(s)[0] for s in fine]
    j = int(np.nanargmin(ferrs))
    s_best = float(fine[j])
    best_abs, vf = feas_err(s_best)
    assert np.all(vf <= vtop + 1e-12), "feasible line raised a prompt"
    raised0 = float(np.mean(v0 > vtop))
    return {
        "base": abs_mean(e_base), "base_signed": float(np.mean(e_base)),
        "floor0": abs_mean(e0), "feasible": best_abs,
        "base_worst_log10": float(np.mean(vtop[dr])) / LN10,
        "floor0_top_nats": float(-v0[0]), "feasible_top_nats": float(-vf[0]),
        "feasible_slope_ratio": s_best / s0, "floor0_raised_frac": raised0,
        "L": L, "used": used,
    }


# ----------------------------------------------------------------- validate
def validate(pool):
    from forecast_experiment.dynamics_probe import fixed_split_worst_error
    n = pool.size
    order = np.argsort(-pool, kind="stable")
    rank_of = np.empty(n, dtype=np.int64)
    rank_of[order] = np.arange(n)
    vs = pool[order]
    worst = 0.0
    for sd in range(20):
        ref = fixed_split_worst_error(pool, n_fit=250, k=10, rng_seed=1000 + sd)
        idx = np.random.default_rng(1000 + sd).permutation(n)
        fr = np.sort(rank_of[idx[:250]])[:10][None, :]
        dr = np.array([rank_of[idx[250:]].min()])
        mine = forecast_signed(vs, fr, dr, 250, n - 250)[0]
        worst = max(worst, abs(mine - ref["signed_log10_error"]))
    print(f"validation vs fixed_split_worst_error: max |diff| = {worst:.2e}", flush=True)
    assert worst < 1e-9


# --------------------------------------------------------------------- jobs
def rt_cell(args):
    pools, M, N, k, seed = args
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in pools.items():
        out[name] = evaluate_pool(np.sort(p)[::-1], M, N, k, rng)
    return {"pop": "rt", "M": M, "N": N, "k": k, "per_pool": out}


def full_cell(args):
    full, M, R, k, seed = args
    N = int(round(R * M))
    n = M + N
    P = int(min(16, max(4, full.size // n)))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(P):
        p = full[rng.choice(full.size, size=n, replace=False)]
        rows.append(evaluate_pool(np.sort(p)[::-1], M, N, k, rng))
    return {"pop": "full", "M": M, "N": N, "R": R, "k": k, "P": P, "rows": rows}


def mean_over(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    t0 = time.time()
    rt = load_rt_pools()
    print(f"rt pools: {len(rt)} x n={sorted({p.size for p in rt.values()})}", flush=True)
    validate(next(iter(rt.values())))
    full = load_fullpool()
    print(f"full pool: n={full.size}, max {full.max():.4f}, frac within 0.5 dec of ceiling "
          f"{np.mean(full > math.log(10 ** -0.5)):.5f}", flush=True)

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
        ho = [v for kk, v in c["per_pool"].items() if kk.startswith("heldout:")]
        b, f0, fe = mean_over(rows, "base"), mean_over(rows, "floor0"), mean_over(rows, "feasible")
        print(f"M={c['M']:5d} N={c['N']:5d} R={c['N']/c['M']:5.1f} k={c['k']:4d}  base {b:.3f}  floor0 {f0:.3f}  "
              f"feasible {fe:.3f}  base/feasible {b/fe:5.2f}  | heldout-4: base {mean_over(ho,'base'):.3f} "
              f"feasible {mean_over(ho,'feasible'):.3f}  | feasible top {mean_over(rows,'feasible_top_nats'):.1f} nats "
              f"(floor0 top {mean_over(rows,'floor0_top_nats'):.1f}), floor0 raises {100*mean_over(rows,'floor0_raised_frac'):.0f}% of top-L",
              flush=True)
    print("\n== (B) random pools from the full 85.6k pool (mean over P pools)")
    for c in full_res:
        rows = c["rows"]
        b, f0, fe = mean_over(rows, "base"), mean_over(rows, "floor0"), mean_over(rows, "feasible")
        print(f"M={c['M']:5d} R={c['R']:4} N={c['N']:6d} k={c['k']:4d} P={c['P']:2d}  base {b:.3f} (signed {mean_over(rows,'base_signed'):+.2f}, "
              f"actual worst {mean_over(rows,'base_worst_log10'):+.2f} dec)  floor0 {f0:.3f}  feasible {fe:.3f}  "
              f"base/feasible {b/fe:5.2f}  feasible top {mean_over(rows,'feasible_top_nats'):.1f} nats  "
              f"slope x{mean_over(rows,'feasible_slope_ratio'):.2f}", flush=True)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
