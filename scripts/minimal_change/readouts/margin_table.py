"""Training ablation of the band's margin (paper appendix "Forecastability training with minimal change
to the model", "Training at other margins"; runs from scripts/minimal_change/submit_margin_ablation.sh):
the main band arm of each
dataset (anchpen + two-sided band) at seed 43, trained with margins 0.06, 0.13 (the seed-43 main run)
and 0.25 nats, read on the same held-out pools.

Per run: the held-out fold as read (sum of base MSE over sum of trained MSE across the seed's held-out
pools, 200 scoring splits each, with a 95% bootstrap over pools); the spread-normalized fold
(strict_table.py's band clamp and projection) under the paper's 0.13-nat standard and under the run's
own training margin, each with its fraction of the straight-tail certificate at that margin; both
folds relative to the 0.13 run, with a paired 95% bootstrap over the same pools (it resamples held-out
pools for fixed trained models, so it covers evaluation noise only; each margin is one training run,
and the spread across training seeds at one margin, in strict_table*.csv, is the yardstick for
training randomness); the top / median
event drop in nats (fleet_table.py's definition, mean over pools); and the held-out single-token KL
to base from the run's final evaluation (kl_table.py's rule). Password is read at step 200 and
WildChat at step 100, the length of the WildChat ablation runs.

  python margin_table.py [--pw-step 200] [--wc-step 100] [--out margin_table.csv]
"""
import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cert_scale  # noqa: E402
from cert_sq_strategy import involved_sigma  # noqa: E402
from feasibility import project  # noqa: E402
from fleet_table import READOUTS, RUNS, base_errors  # noqa: E402
from heldout_spread import pools  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

STANDARD = 0.13  # the paper's spread-normalized standard (cert_scale.MARGIN)
RUN_LIST = [  # (dataset, training margin, run)
    ("password", 0.06, "lqpw_pw2apsp200m06_seed43"),
    ("password", 0.13, "lqpw_pw2apsp200_seed43"),
    ("password", 0.25, "lqpw_pw2apsp200m25_seed43"),
    ("wildchat", 0.06, "lqdyn_v2e2ap50spm06_seed43"),
    ("wildchat", 0.13, "lqdyn_v2e2ap50sp_seed43"),
    ("wildchat", 0.25, "lqdyn_v2e2ap50spm25_seed43"),
]


def read(run, step, margins):
    """Per held-out pool: base, as-read, and per-margin certificate / spread-normalized MSEs, and drops."""
    ev = RUNS / run / "snapshot_evals"
    f = ev / f"step{step:04d}_test_forecast_traces.npz"
    if not f.exists():
        return None
    base, k = pools(ev / "base_test_forecast_traces.npz")
    trained, _ = pools(f)
    rows = []
    for name, (vb, M, e, perms, eb, _sb, _sref) in sorted(base_errors(base, k).items()):
        vt = trained[name][0]
        et, _ = split_errors(vt, perms, M, k=k)
        r = {"pool": name, "base": float((eb ** 2).mean()), "asread": float((et ** 2).mean()),
             "top": float(vb.max() - vt.max()), "med": float(np.median(vb) - np.median(vt))}
        st = involved_sigma(vt, M, k, e)
        for mg in margins:
            cert_scale.MARGIN = mg  # certify() reads the module's margin at call time
            out, _tgt, _e2, _L, sref, _info = cert_scale.certify(vb, M, k)
            lo, hi = sref - mg, sref + mg
            vc = vt * (min(max(st, lo), hi) / st)
            vf, ok = project(vc, vb, M, k, e, lo, hi)
            ec, _ = split_errors(out, perms, M, k=k)
            ef, _ = split_errors(vf, perms, M, k=k)
            r[("cert", mg)], r[("sn", mg)], r[("ok", mg)] = float((ec ** 2).mean()), float((ef ** 2).mean()), bool(ok)
        rows.append(r)
    cert_scale.MARGIN = STANDARD
    return rows


def final_kl(run):
    d = RUNS / run
    f = d / "kl_eval_results.json"
    if not f.exists():
        return None
    steps = json.load(open(d / "config.json")).get("steps")
    snap = d / "snapshots" / f"step_{int(steps):04d}"
    if not snap.exists() or f.stat().st_mtime < snap.stat().st_mtime:
        return None  # a final eval that has not run yet
    return float(json.load(open(f))["single_token"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw-step", type=int, default=200)
    ap.add_argument("--wc-step", type=int, default=100)
    ap.add_argument("--out", default="margin_table.csv")
    a = ap.parse_args()
    out_rows = []
    for dataset in ("password", "wildchat"):
        step = a.pw_step if dataset == "password" else a.wc_step
        reads = {}
        for ds, mg, run in RUN_LIST:
            if ds != dataset:
                continue
            r = read(run, step, sorted({STANDARD, mg}))
            if r is None:
                print(f"{dataset:8s} margin {mg:.2f} {run}: no read at step {step}")
                continue
            reads[mg] = (run, r)
        if STANDARD not in reads:
            continue
        ref = reads[STANDARD][1]
        names = [p["pool"] for p in ref]
        for mg, (run, r) in reads.items():
            assert [p["pool"] for p in r] == names, (run, "held-out pools differ from the 0.13 run")
            gap = max(abs(p["base"] - q["base"]) / q["base"] for p, q in zip(r, ref))
            if gap > 1e-6:
                print(f"  warning: {run} base MSEs differ from the 0.13 run's by up to {gap:.1e} (relative)")
        print(f"\n{dataset} at step {step} ({len(names)} held-out pools, seed 43)")
        for mg in sorted(reads):
            run, r = reads[mg]
            B = np.array([p["base"] for p in r])
            T = np.array([p["asread"] for p in r])
            T0 = np.array([p["asread"] for p in ref])
            SN = np.array([p[("sn", STANDARD)] for p in r])
            SN0 = np.array([p[("sn", STANDARD)] for p in ref])
            C = np.array([p[("cert", STANDARD)] for p in r])
            SNo = np.array([p[("sn", mg)] for p in r])
            Co = np.array([p[("cert", mg)] for p in r])
            rng = np.random.default_rng([20261105, step, zlib.crc32(f"{dataset}|{mg}".encode())])
            idx = rng.integers(0, len(B), (4000, len(B)))
            q_ar = np.percentile(B[idx].sum(1) / T[idx].sum(1), [2.5, 97.5])
            q_rat = np.percentile(T0[idx].sum(1) / T[idx].sum(1), [2.5, 97.5])
            q_rsn = np.percentile(SN0[idx].sum(1) / SN[idx].sum(1), [2.5, 97.5])
            kl = final_kl(run)
            row = {"dataset": dataset, "train_margin": mg, "run": run, "step": step, "pools": len(B),
                   "asread_fold": B.sum() / T.sum(), "asread_lo": q_ar[0], "asread_hi": q_ar[1],
                   "asread_vs_013": T0.sum() / T.sum(), "asread_vs_013_lo": q_rat[0], "asread_vs_013_hi": q_rat[1],
                   "sn013_fold": B.sum() / SN.sum(), "cert013_fold": B.sum() / C.sum(),
                   "sn013_fraction_of_cert": C.sum() / SN.sum(),
                   "sn013_vs_013": SN0.sum() / SN.sum(), "sn013_vs_013_lo": q_rsn[0], "sn013_vs_013_hi": q_rsn[1],
                   "sn_own_fold": B.sum() / SNo.sum(), "cert_own_fold": B.sum() / Co.sum(),
                   "sn_own_fraction_of_cert": Co.sum() / SNo.sum(),
                   "infeasible_013": sum(not p[("ok", STANDARD)] for p in r),
                   "infeasible_own": sum(not p[("ok", mg)] for p in r),
                   "top_drop": float(np.mean([p["top"] for p in r])),
                   "median_drop": float(np.mean([p["med"] for p in r])),
                   "kl_single_token": kl, "base_mse_sum": B.sum(), "asread_mse_sum": T.sum(),
                   "sn013_mse_sum": SN.sum(), "cert013_mse_sum": C.sum()}
            out_rows.append(row)
            print(f"  margin {mg:.2f}: as read {row['asread_fold']:6.2f}x (95% {q_ar[0]:.2f}-{q_ar[1]:.2f}; "
                  f"{row['asread_vs_013']:.2f} of the 0.13 run's, {q_rat[0]:.2f}-{q_rat[1]:.2f}) | "
                  f"spread-normalized at 0.13 {row['sn013_fold']:5.2f}x ({row['sn013_fraction_of_cert']:.2f} of cert "
                  f"{row['cert013_fold']:.2f}x; {row['sn013_vs_013']:.2f} of the 0.13 run's) | at own margin "
                  f"{row['sn_own_fold']:5.2f}x ({row['sn_own_fraction_of_cert']:.2f} of cert {row['cert_own_fold']:.2f}x) | "
                  f"drop top {row['top_drop']:.1f} median {row['median_drop']:.1f} nats | "
                  f"KL {'n/a' if kl is None else f'{kl:.2e}'}"
                  + (f" | {row['infeasible_013']}/{row['infeasible_own']} pools infeasible"
                     if row["infeasible_013"] or row["infeasible_own"] else ""), flush=True)
    READOUTS.mkdir(parents=True, exist_ok=True)
    with open(READOUTS / a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {READOUTS / a.out}")


if __name__ == "__main__":
    main()
