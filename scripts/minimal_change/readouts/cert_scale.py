"""Certified floors at larger fit sizes M, deploy ratios R = N/M and fit top-k (WildChat E2).

The paper's geometry is M = 250 fit prompts and N = 2050 deploy prompts (R = 8.2), with fit
top-k 10 or 50, on the 32 bank pools of 2300 moderation-flagged English prompts (v2e2 bank,
base Qwen3-0.6B log p(E2)). Here we draw pools of U = M + N prompts from the bank's whole
population (its 73,600 prompts, which the bank deduplicates on normalized text) and compute,
per pool and k:

  * the base error of the paper's inverse-OLS forecaster on 200 random fit/deploy splits;
  * the certified floor: the best straight tail inside the two-sided band of +/- 0.13 nats on
    the involved-set sigma, over the k-matched reach L (the construction of
    g_transfer_k.cert), scored on the same 200 splits.

Pooled MSE fold of a cell = mean base MSE over its pools / mean certified MSE. Each row also
records the certified line v_r = -|v_1| r^(1/lambda) (its rate lambda and top value v_1) and the
base pool's split bias and spread, the inputs of the straight-tail floor in floor_theory.py.

Pools larger than about a third of the population overlap heavily, so those cells read closer to
one pool than to many; the CSV keeps the pool count and size so the reader can see it.

  python cert_scale.py OUT_CSV [--smoke] [--workers 8]

floor_theory.py, grid_theory.py, tail_opt_grid.py and verify_tail_opt.py read the full run's CSV
from $RESULTS_ROOT/lquant_dynamics/readouts/cert_scale.csv (cert_scale.sbatch runs there).
"""
import argparse
import csv
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g_transfer_k as gk  # noqa: E402
from g_transfer_v2 import pools_from_bank, smoke  # noqa: E402
from forecast_experiment.dynamics_probe import certified_tail_targets, involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

MARGIN = 0.13
S = 200
LN10 = math.log(10.0)

# (M, R) cells and their k values (k <= M/4). R = 8.2 is the paper's N/M.
CELLS = [
    (250, 8.2, (10, 50)), (250, 30, (10, 50)), (250, 100, (10, 50)), (250, 290, (10, 50)),
    (1000, 8.2, (10, 50, 200)), (1000, 30, (10, 50, 200)), (1000, 70, (10, 50, 200)),
    (4000, 8.2, (10, 50, 200, 1000)),
]
SMOKE_CELLS = [(250, 8.2, (10,)), (1000, 30, (50,))]

POP = None    # the population's base log p(E2), set before the workers fork
BANK = None   # the 32 bank pools, for the reference cell


def n_pools(U):
    return 32 if U <= 10_000 else (16 if U <= 40_000 else 8)


def certify(v, M, k):
    """g_transfer_k.cert without its cache, also returning the certified line itself."""
    e = count_extrapolated_eval_points(M, v.size - M, plotting="weibull")
    L = gk.reach(v.size, M, k, e)
    sref = involved_sigma_reference(v, n_fit=M, fit_top_k=k, extrap_k=e, m=L)
    idx, tgt, info = certified_tail_targets(v, n_fit=M, fit_top_k=k, extrap_k=e, sigma_ref=sref,
                                            sigma_margin=MARGIN, two_sided=True, max_rank=L)
    out = v.copy()
    out[np.argsort(-v, kind="stable")[L:]] -= float(v[idx[-1]] - tgt[-1])
    out[idx] = tgt
    return out, tgt, e, L, sref, info


def involved_sigma(v, M, k, e, splits=200, seed=4321):
    """Split-averaged std of the involved set (fit top-k plus deploy top-e)."""
    rng = np.random.default_rng(seed)
    ranked = np.sort(v)[::-1]
    perms = np.stack([rng.permutation(v.size) for _ in range(splits)])
    fr = np.sort(perms[:, :M], axis=1)[:, :k]
    dr = np.sort(perms[:, M:], axis=1)[:, :e]
    return float(np.concatenate([ranked[fr], ranked[dr]], axis=1).std(axis=1).mean())


def errors(v, perms, M, k):
    e, _ = split_errors(v, perms, M, k=k)
    return e


def run_pool(task):
    cell_id, M, N, ks, pool_id, source = task
    U = M + N
    if source == "bank":
        v = BANK[pool_id]
        name = f"bank{pool_id:02d}"
    else:
        rng = np.random.default_rng([M, N, pool_id, 20261104])
        v = POP[rng.choice(POP.size, size=U, replace=False)]
        name = f"rand{pool_id:02d}"
    rng = np.random.default_rng(7 + pool_id)  # gk.score's seeding: paired base/cert splits
    perms = np.stack([rng.permutation(U) for _ in range(S)])
    rows = []
    for k in ks:
        t0 = time.time()
        eb = errors(v, perms, M, k)
        row = {"cell": cell_id, "source": source, "pool": name, "M": M, "N": N, "R": N / M, "U": U, "k": k,
               "base_top": float(v.max()), "base_abs": float(np.abs(eb).mean()), "base_mse": float((eb ** 2).mean()),
               "base_bias": float(eb.mean()), "base_sd": float(eb.std())}
        try:
            out, tgt, e, L, sref, info = certify(v, M, k)
        except RuntimeError as err:
            row.update({"status": f"cert failed: {err}"})
            rows.append(row)
            continue
        ec = errors(out, perms, M, k)
        r = np.arange(1, tgt.size + 1)
        slope = np.polyfit(np.log(r), np.log(-tgt), 1)[0]  # exact line: log|v_r| = log|v_1| + log(r)/lambda
        lam = 1.0 / slope
        row.update({
            "status": "ok", "e": e, "L": L,
            "cert_abs": float(np.abs(ec).mean()), "cert_mse": float((ec ** 2).mean()),
            "cert_bias": float(ec.mean()), "cert_sd": float(ec.std()),
            "cert_lambda": float(lam), "cert_v1": float(tgt[0]), "cert_spread_top": float(-tgt[0] / lam),
            "base_lambda": float(lam / info["slope_ratio"]), "slope_ratio": info["slope_ratio"],
            "deepen": info["deepen"], "sigma_ref": sref, "sigma_lo": sref - MARGIN,
            "cert_sigma": involved_sigma(out, M, k, e), "base_sigma": involved_sigma(v, M, k, e),
            "top_anchor": info["top_anchor_nats"], "mean_drop": info["mean_drop_nats"],
            "secs": round(time.time() - t0, 1),
        })
        rows.append(row)
    return rows


def main():
    global POP, BANK
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    meta = smoke()
    pools = pools_from_bank(meta)
    names = [n for n, x in pools.items() if x.size == 2300]
    BANK = [pools[n] for n in names]
    POP = np.concatenate(BANK)
    print(f"population: {POP.size} prompts from {len(BANK)} bank pools; top {POP.max():+.3f} nats; "
          f"{int((POP > math.log(0.5)).sum())} above p = 0.5, {int((POP > -7).sum())} above -7 nats", flush=True)

    tasks = []
    cells = SMOKE_CELLS if a.smoke else CELLS
    if not a.smoke:
        for j in range(len(BANK)):
            tasks.append(("bank_M250_R8.2", 250, 2050, (10, 50), j, "bank"))
    for M, R, ks in cells:
        N = int(round(R * M))
        U = M + N
        if U > POP.size:
            print(f"skip M{M} R{R}: pool of {U} exceeds the population")
            continue
        P = 2 if a.smoke else n_pools(U)
        for j in range(P):
            tasks.append((f"M{M}_R{R:g}", M, N, ks, j, "random"))
    # Largest pools first, so the long tasks do not straggle at the end.
    tasks.sort(key=lambda t: -(t[1] + t[2]) * max(t[3]))
    print(f"{len(tasks)} pool tasks on {a.workers} workers", flush=True)

    fields = ["cell", "source", "pool", "M", "N", "R", "U", "k", "status", "e", "L",
              "base_top", "base_abs", "base_mse", "base_bias", "base_sd",
              "cert_abs", "cert_mse", "cert_bias", "cert_sd",
              "cert_lambda", "cert_v1", "cert_spread_top", "base_lambda", "slope_ratio", "deepen",
              "sigma_ref", "sigma_lo", "cert_sigma", "base_sigma", "top_anchor", "mean_drop", "secs"]
    t0 = time.time()
    with open(a.out_csv, "w", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        done = 0
        for rows in pool.imap_unordered(run_pool, tasks):
            for r in rows:
                w.writerow(r)
            f.flush()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} tasks, {time.time() - t0:.0f} s", flush=True)
    summarize(a.out_csv)


def summarize(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)]
    keys = sorted({(r["source"] != "bank", int(r["M"]), float(r["R"]), int(r["k"]), r["cell"]) for r in rows})
    print("\ncell                 k  pools fail     U    L   base mse  cert mse  fold   base bias/sd    "
          "cert lambda  |v1|   spread  deepen  base top")
    for _, _, _, k, cell in keys:
        rs = [r for r in rows if r["cell"] == cell and int(r["k"]) == k]
        ok = [r for r in rs if r["status"] == "ok"]
        if not ok:
            print(f"{cell:18s} {k:4d}  {len(rs):3d}  all failed")
            continue
        g = lambda c: np.array([float(r[c]) for r in ok])
        bm, cm = g("base_mse").mean(), g("cert_mse").mean()
        print(f"{cell:18s} {k:4d}  {len(rs):3d}  {len(rs) - len(ok):3d}  {int(ok[0]['U']):6d} {int(g('L').mean()):5d}"
              f"   {bm:7.3f}  {cm:7.3f}  {bm / cm:5.2f}x  {g('base_bias').mean():+6.2f}/{g('base_sd').mean():5.2f}"
              f"   {g('cert_lambda').mean():6.2f}  {-g('cert_v1').mean():6.2f}  {g('cert_spread_top').mean():6.3f}"
              f"  {g('deepen').mean():5.2f}  {g('base_top').mean():+6.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
    else:
        main()
