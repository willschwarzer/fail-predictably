"""The strict-standard table: per arm and seed, the held-out fold inside the certificate's feasible set.

Per run and step this repeats feasibility.py exactly (band clamp to the involved-set sigma band,
then projection onto both certificate constraints, scored on the pool's 200 scoring splits), and
keeps the per-pool MSEs (base, straight-tail certificate, feasible trained, and the trained model as
read before the band clamp and projection). Per arm it reports each
seed's fold and fraction of its certificate, the mean and standard error of the per-seed folds,
and the pooled fold (sum of base MSE over sum of trained MSE across every seed's held-out pools)
with a 95% two-level bootstrap (seeds, then pools within each seed; 4,000 resamples).

  python strict_table.py [--step 100] [--out strict_table.csv]
  python strict_table.py --step 200 --out strict_table_200.csv
  python strict_table.py --steps 50 60 70 80 90 100 120 150 200 --brief --out strict_steps.csv

--out is a file name under $RESULTS_ROOT/lquant_dynamics/readouts/, where the figure scripts read
strict_table.csv and strict_table_200.csv.

The password band arm's seeds 11, 17 and 23 trained 100 steps in `lqpw_pw2apsp_seed*` and continued
to 200 in `lqpw_pw2apsp200_seed*`, so their reads after step 100 come from the second directory. The
same holds for every seed of the two password arms without the band (`lqpw_pw2iog_seed*` ->
`lqpw_pw2iog200_seed*`, `lqpw_pw2nbap_seed*` -> `lqpw_pw2nbap200_seed*`). On WildChat every seed of
the four 2x2 arms trained 100 steps in `lqdyn_<arm>_seed*` and continued to 200 in
`lqdyn_<arm>x200_seed*`, read the same way.
"""
import argparse
import csv
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cert_scale import MARGIN, certify  # noqa: E402
from cert_sq_strategy import involved_sigma  # noqa: E402
from feasibility import project  # noqa: E402
from fleet_table import READOUTS, RUNS, base_errors  # noqa: E402
from heldout_spread import pools  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

SEEDS = (11, 17, 23, 29, 43)
# (dataset, arm label, run name for a seed); the password band arm's seeds 29 and 43 are the
# from-scratch 200-step runs, read at the same step as the others.
ARMS = [
    ("password", "anchpen + band (main)",
     lambda s, t: f"lqpw_pw2apsp_seed{s}" if s in (11, 17, 23) and t <= 100 else f"lqpw_pw2apsp200_seed{s}"),
    ("password", "IOG alone", lambda s, t: f"lqpw_pw2iog_seed{s}" if t <= 100 else f"lqpw_pw2iog200_seed{s}"),
    ("password", "anchpen, no band", lambda s, t: f"lqpw_pw2nbap_seed{s}" if t <= 100 else f"lqpw_pw2nbap200_seed{s}"),
    ("password", "IOG + band", lambda s, t: f"lqpw_pw2iogsb200_seed{s}"),
    ("wildchat", "anchpen + band (main)",
     lambda s, t: f"lqdyn_v2e2ap50sp_seed{s}" if t <= 100 else f"lqdyn_v2e2ap50spx200_seed{s}"),
    ("wildchat", "IOG alone",
     lambda s, t: f"lqdyn_v2e2iogap50_seed{s}" if t <= 100 else f"lqdyn_v2e2iogap50x200_seed{s}"),
    ("wildchat", "anchpen, no band",
     lambda s, t: f"lqdyn_v2e2nbap50_seed{s}" if t <= 100 else f"lqdyn_v2e2nbap50x200_seed{s}"),
    ("wildchat", "IOG + band",
     lambda s, t: f"lqdyn_v2e2iogsbap50sp_seed{s}" if t <= 100 else f"lqdyn_v2e2iogsbap50spx200_seed{s}"),
    ("wildchat", "anchpen + one-sided band", lambda s, t: f"lqdyn_v2e2osap50sp_seed{s}"),
    ("wildchat", "anchpen + band, MLP adapter", lambda s, t: f"lqdyn_v2e2mlpap50sp_seed{s}"),
]


def read(run, step):
    ev = RUNS / run / "snapshot_evals"
    f = ev / f"step{step:04d}_test_forecast_traces.npz"
    if not f.exists():
        return None
    base, k = pools(ev / "base_test_forecast_traces.npz")
    trained, _ = pools(f)
    rows = []
    for name, (vb, M, e, perms, eb, sb, _sref) in base_errors(base, k).items():
        vt = trained[name][0]
        out, tgt, e2, L, sref, info = certify(vb, M, k)
        lo, hi = sref - MARGIN, sref + MARGIN
        st = involved_sigma(vt, M, k, e)
        vc = vt * (min(max(st, lo), hi) / st)
        vf, ok = project(vc, vb, M, k, e, lo, hi)
        ec, _ = split_errors(out, perms, M, k=k)
        ef, _ = split_errors(vf, perms, M, k=k)
        et, _ = split_errors(vt, perms, M, k=k)  # the trained model as read, before the band clamp and projection
        rows.append((float((eb ** 2).mean()), float((ec ** 2).mean()), float((ef ** 2).mean()), float((et ** 2).mean()), ok))
    return np.array([r[:4] for r in rows]), sum(not r[4] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--steps", type=int, nargs="+", help="several steps (overrides --step)")
    ap.add_argument("--brief", action="store_true", help="one line per arm and step")
    ap.add_argument("--out", default="strict_table.csv")
    a = ap.parse_args()
    out_rows = []
    steps = a.steps or [a.step]
    for dataset, label, name, step in [(d, l, n, t) for d, l, n in ARMS for t in steps]:
        per = {}
        for s in SEEDS:
            r = read(name(s, step), step)
            if r is not None:
                per[s] = r
        if not per:
            if not a.brief:
                print(f"\n{dataset:8s} {label:30s}: no reads at step {step}")
            continue
        folds = {s: r[0][:, 0].sum() / r[0][:, 2].sum() for s, r in per.items()}
        caps = {s: r[0][:, 0].sum() / r[0][:, 1].sum() for s, r in per.items()}
        asread = {s: r[0][:, 0].sum() / r[0][:, 3].sum() for s, r in per.items()}
        allr = np.concatenate([r[0] for r in per.values()])
        pooled, pooled_cap = allr[:, 0].sum() / allr[:, 2].sum(), allr[:, 0].sum() / allr[:, 1].sum()
        pooled_ar = allr[:, 0].sum() / allr[:, 3].sum()
        seeds = list(per)
        # One bootstrap stream per arm and step, so an arm's interval moves only when its own reads do.
        rng = np.random.default_rng([20261104, step, zlib.crc32(f"{dataset}|{label}".encode())])
        boot = []
        for _ in range(4000):
            pick = [per[seeds[i]][0] for i in rng.integers(0, len(seeds), len(seeds))]
            res = np.concatenate([p[rng.integers(0, len(p), len(p))] for p in pick])
            boot.append((res[:, 0].sum() / res[:, 2].sum(), res[:, 1].sum() / res[:, 2].sum(),
                         res[:, 0].sum() / res[:, 3].sum()))
        boot = np.array(boot)
        q = np.percentile(boot[:, 0], [2.5, 97.5])
        qf = np.percentile(boot[:, 1], [2.5, 97.5])  # trained fold as a fraction of the certificate's
        qa = np.percentile(boot[:, 2], [2.5, 97.5])  # as read
        f = np.array(list(folds.values()))
        se = f.std(ddof=1) / np.sqrt(len(f)) if len(f) > 1 else float("nan")
        infeas = sum(r[1] for r in per.values())

        def seed_row(s):
            # The per-seed sums of per-pool MSE let a reader pool any subset of seeds.
            m = per[s][0]
            return {"dataset": dataset, "arm": label, "seed": s, "step": step, "run": name(s, step),
                    "fold": folds[s], "cert_fold": caps[s], "fraction_of_cert": folds[s] / caps[s],
                    "pools": len(m), "base_mse_sum": m[:, 0].sum(), "cert_mse_sum": m[:, 1].sum(),
                    "trained_mse_sum": m[:, 2].sum(), "asread_fold": asread[s], "asread_mse_sum": m[:, 3].sum()}

        if a.brief:
            print(f"{dataset:8s} {label:30s} step {step:3d}: pooled {pooled:5.2f}x ({q[0]:.2f}-{q[1]:.2f}), "
                  f"{pooled / pooled_cap:.2f} of cert, as read {pooled_ar:.2f}x | "
                  + " ".join(f"s{s} {folds[s]:5.2f}" for s in seeds), flush=True)
            out_rows.extend(seed_row(s) for s in seeds)
            out_rows.append({"dataset": dataset, "arm": label, "seed": "pooled", "step": step, "run": "",
                             "fold": pooled, "cert_fold": pooled_cap, "fraction_of_cert": pooled / pooled_cap,
                             "pools": len(allr), "per_seed_mean": f.mean(), "per_seed_se": se,
                             "pooled_lo": q[0], "pooled_hi": q[1], "asread_fold": pooled_ar,
                             "asread_lo": qa[0], "asread_hi": qa[1]})
            continue
        print(f"\n{dataset:8s} {label:30s} step {step}: {len(f)} seeds, per-seed mean {f.mean():.2f}x (s.e. {se:.2f}) | "
              f"pooled {pooled:.2f}x (95% {q[0]:.2f}-{q[1]:.2f}), {pooled / pooled_cap:.2f} of the pooled certificate "
              f"{pooled_cap:.2f}x ({qf[0]:.2f}-{qf[1]:.2f}){'' if not infeas else f' | {infeas} pools infeasible'}")
        print(f"   as read (no band clamp or projection): pooled {pooled_ar:.2f}x (95% {qa[0]:.2f}-{qa[1]:.2f}) | "
              + "  ".join(f"s{s} {asread[s]:.2f}x" for s in seeds))
        print("   " + "  ".join(f"s{s} {folds[s]:.2f}x ({folds[s] / caps[s]:.2f} of {caps[s]:.2f}x)" for s in seeds))
        out_rows.extend(seed_row(s) for s in seeds)
        out_rows.append({"dataset": dataset, "arm": label, "seed": "pooled", "step": step, "run": "",
                         "fold": pooled, "cert_fold": pooled_cap, "fraction_of_cert": pooled / pooled_cap,
                         "pools": len(allr), "per_seed_mean": f.mean(), "per_seed_se": se,
                         "pooled_lo": q[0], "pooled_hi": q[1], "asread_fold": pooled_ar,
                         "asread_lo": qa[0], "asread_hi": qa[1]})
    READOUTS.mkdir(parents=True, exist_ok=True)
    with open(READOUTS / a.out, "w", newline="") as fh:
        keys = sorted({k for r in out_rows for k in r}, key=lambda k: list(out_rows[-1]).index(k) if k in out_rows[-1] else 99)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {READOUTS / a.out}")


if __name__ == "__main__":
    main()
