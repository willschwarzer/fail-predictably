"""Is the log-prob-0 ceiling distorting the WildChat forecast task?

Per base pool, on S random fit/deploy splits (feasible_floor_sigma machinery):
  base      OLS tail forecast error (abs / signed / mse, log10)
  cert      certified floor (sym_m013: straight, rank-wise <= base, involved
            sigma pinned to the base-level line's within +/-0.13 nats)
  one       trivial "predict p = 1" forecaster: error = -log10 p(deploy max)
  fitmax    trivial "predict the fit max" forecaster (the null)
plus top-5 log10 p and counts above 0.5 / 0.1 / 0.01. Each pool is also
re-evaluated with every prompt above p = 0.1 removed (deploy size shrinks by
the removed count), to measure how much of the base error and the certified
headroom the near-ceiling prompts carry.

Populations:
  rt       16 rt pools (12 train anchors + 4 held-out base traces), M250/N2050
  full_rt  16 random 2300-pools from the 85,601 full flagged pool, M250/N2050
  wc_s17   canonical 0.6B seed-17 held-out pools (7 x 935), M44/N891, no prompt
  wcblk_s17 the same 7 pools under --token-union-blocklist-prompt
  pw2      password strict-v2 warm-start 0.6B, held-out slots, M44/N891

Inputs (under $RESULTS_ROOT):
  {WC}_20260722/seed_17, {WC}_blkklnf05/seed_17  the 0.6B WildChat seed-17 runs, from
      scripts/wildchat/natural_prompts/run_lquant_wildchat_ldnoobw_multiseed.sbatch at SEED=17
      with TAG_DATE=20260722, and with TAG_DATE=blkklnf05 BLOCKLIST=1
  lquant_40slots_gcg6warm_strictv2_iog_full_bootstrap1_cache296_seed*_20260724/seed_*
      the password strict-v2 warm-start 0.6B IOG fleet; its launcher is not in this release
"""
import glob
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feasible_floor_sweep import LN10, READOUTS, RESULTS_ROOT, load_fullpool, load_rt_pools  # noqa: E402
import feasible_floor_sigma as ffs  # noqa: E402

RES = RESULTS_ROOT
OUT = READOUTS / "ceiling_check.json"
WC = "lquant_wildchat_ldnoobw_qwen_qwen3-0_6b_iog_full_boot1_cache296_seed17"


def load_traces(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "baseline_test_forecast_traces_metadata.json").read_text())
    z = np.load(run_dir / "baseline_test_forecast_traces.npz")
    out = {}
    for t in meta["targets"]:
        pre = t["array_prefix"]
        out[t["target"]] = np.concatenate(
            [z[f"{pre}__fit_log_probs"], z[f"{pre}__deploy_log_probs"]]).astype(np.float64)
    return out


def one_pool(args):
    name, v, M, N, seed = args
    rng = np.random.default_rng(seed)
    v = np.sort(v[np.isfinite(v)])[::-1]
    row = {"name": name, "n": int(v.size),
           "top5_log10": [float(x) / LN10 for x in v[:5]],
           "n_gt_0.5": int((v > math.log(0.5)).sum()),
           "n_gt_0.1": int((v > math.log(0.1)).sum()),
           "n_gt_0.01": int((v > math.log(0.01)).sum())}
    for tag, keep in (("all", np.ones(v.size, bool)), ("cens01", v <= math.log(0.1))):
        vv = v[keep]
        n_dep = min(N, vv.size - M)
        e = ffs.extrap_k(M, n_dep)
        fr, dr = ffs.sample_tables(vv.size, M, n_dep, 10, e, rng)
        one = -vv[dr[:, 0]] / LN10
        fitmax = (vv[fr[:, 0]] - vv[dr[:, 0]]) / LN10
        ev = ffs.evaluate_pool(vv, M, n_dep, 10, rng)
        cert = ev["sym_m013"]
        row[tag] = {
            "N": int(n_dep),
            "base_abs": ev["base"]["abs"], "base_signed": ev["base"]["signed"], "base_mse": ev["base"]["mse"],
            "cert_abs": cert["abs"] if cert else None, "cert_mse": cert["mse"] if cert else None,
            "one_abs": float(np.abs(one).mean()), "one_mse": float((one ** 2).mean()),
            "fitmax_abs": float(np.abs(fitmax).mean()), "fitmax_mse": float((fitmax ** 2).mean()),
            "dep_max_log10_mean": float(np.mean(vv[dr[:, 0]])) / LN10,
            "frac_splits_dep_max_gt_0.5": float(np.mean(vv[dr[:, 0]] > math.log(0.5))),
        }
    return row


def main():
    t0 = time.time()
    jobs = []
    rt = load_rt_pools()
    for name, v in rt.items():
        jobs.append(("rt", name, v, 250, 2050))
    full = load_fullpool()
    rng = np.random.default_rng(7)
    for i in range(16):
        jobs.append(("full_rt", f"full{i:02d}", full[rng.choice(full.size, 2300, replace=False)], 250, 2050))
    for pop, d in (("wc_s17", f"{RES}/{WC}_20260722/seed_17"), ("wcblk_s17", f"{RES}/{WC}_blkklnf05/seed_17")):
        for name, v in load_traces(d).items():
            jobs.append((pop, name, v, 44, 891))
    for d in sorted(glob.glob(f"{RES}/lquant_40slots_gcg6warm_strictv2_iog_full_bootstrap1_cache296_seed*_20260724/seed_*")):
        sd = Path(d).name
        for name, v in load_traces(d).items():
            jobs.append(("pw2", f"{sd}:{name}", v, 44, 891))
    print(f"{len(jobs)} pools", flush=True)
    args = [(f"{pop}|{name}", v, M, N, 1000 + i) for i, (pop, name, v, M, N) in enumerate(jobs)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one_pool, args, chunksize=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))

    by = {}
    for r in rows:
        by.setdefault(r["name"].split("|")[0], []).append(r)
    keys = ("base_abs", "cert_abs", "one_abs", "fitmax_abs", "base_mse", "cert_mse", "one_mse", "dep_max_log10_mean",
            "frac_splits_dep_max_gt_0.5")
    for pop, rs in by.items():
        print(f"\n== {pop}: {len(rs)} pools; mean n>0.5 {np.mean([r['n_gt_0.5'] for r in rs]):.2f}, "
              f"n>0.1 {np.mean([r['n_gt_0.1'] for r in rs]):.2f}, n>0.01 {np.mean([r['n_gt_0.01'] for r in rs]):.2f}; "
              f"mean top1 {np.mean([r['top5_log10'][0] for r in rs]):+.2f}")
        for tag in ("all", "cens01"):
            vals = {k: np.mean([r[tag][k] for r in rs if r[tag][k] is not None]) for k in keys}
            print(f"  [{tag}] " + " ".join(f"{k}={vals[k]:.3f}" for k in keys)
                  + f" | abs fold base/cert {vals['base_abs'] / vals['cert_abs']:.2f}x, mse fold "
                    f"{vals['base_mse'] / vals['cert_mse']:.2f}x")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
