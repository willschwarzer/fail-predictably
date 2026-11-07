"""Where the band's 0.13-nat margin comes from, measured again on the current banks, and how the
spread-normalized read moves with it.

The margin was set during development, on an earlier WildChat bank (paper appendix "Forecastability
training with minimal change to the model", "The margin"), as two standard errors of the training
loss's band estimate. The loss averages the involved-set sigma (the std of the fit partition's top-k plus the
deploy partition's top-e log-probs) over K = 16 random fit/deploy partitions per step, so at the
reference line its partition noise is sd_1 / sqrt(16), where sd_1 is the standard deviation of one
partition's involved-set sigma; the margin is 2 sd_1 / sqrt(16) = sd_1 / 2. On that earlier bank the
real-tail cell measured sd_1 = 0.251 nats (margin 0.13), and the canonical cell 0.485 (margin 0.25).

Part 1 measures sd_1 again on every held-out base pool of the current banks (password v2 at fit
top-k 10, WildChat v2e2 at top-k 50), at the line the training reference straightens (the top
sigma_ref_m ranks, from each run's config) and at the certificate's line (the k-matched reach L).

Part 2 repeats the spread-normalized read (strict_table.read) at one step for every arm, with the
margin set to each of MARGINS. The margin enters the read twice: as the band the trained spread is
clamped into, and as the band the certificate's straight tail must satisfy. The trained models are
fixed (they all trained with 0.13 where they trained with a band), so this measures how the read
depends on the margin, not how training would. At 0.13 it must reproduce strict_table.py.

  python margin_calibration.py [--step 100] [--out margin_calibration.csv]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cert_scale  # noqa: E402
import strict_table as st  # noqa: E402
from cert_sq_strategy import involved_sigma  # noqa: E402
from feasibility import project  # noqa: E402
from fleet_table import READOUTS, RUNS, base_errors  # noqa: E402
import g_transfer_k as gk  # noqa: E402
from heldout_spread import pools  # noqa: E402
from forecast_experiment.dynamics_probe import linearize  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

MARGINS = (0.03, 0.06, 0.13, 0.25, 0.5, 1.0)
K_TRAIN = 16


def one_partition_sigmas(v, M, k, e, draws=400, seed=1234):
    """Involved-set sigma of each of `draws` random fit/deploy partitions of v."""
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        perm = rng.permutation(v.size)
        out[d] = np.concatenate([np.sort(v[perm[:M]])[::-1][:k], np.sort(v[perm[M:]])[::-1][:e]]).std()
    return out


def part1(step):
    rows = []
    for dataset, label, name in st.ARMS:
        if label != "anchpen + band (main)":
            continue
        seen = {}
        ref_m = None
        for s in st.SEEDS:
            run = RUNS / name(s, step)
            f = run / "snapshot_evals" / "base_test_forecast_traces.npz"
            if not f.exists():
                continue
            ref_m = int(json.load(open(run / "config.json"))["sigma_ref_m"])
            base, k = pools(f)
            for pool, (vb, M) in base.items():
                seen.setdefault(pool, (vb, M, k))
        for pool, (vb, M, k) in sorted(seen.items()):
            n = vb.size
            e = count_extrapolated_eval_points(M, n - M, plotting="weibull")
            L = gk.reach(n, M, k, e)
            r = {"dataset": dataset, "pool": pool, "n": n, "M": M, "k": k, "e": e, "reach_L": L,
                 "sigma_ref_m": ref_m}
            for tag, vv in (("train_line", linearize(vb, m=ref_m)), ("cert_line", linearize(vb, m=L)),
                            ("observed", vb)):
                sig = one_partition_sigmas(vv, M, k, e)
                r[f"{tag}_sigma_mean"] = sig.mean()
                r[f"{tag}_sd1"] = sig.std()
                r[f"{tag}_margin_2se_k16"] = 2 * sig.std() / np.sqrt(K_TRAIN)
            rows.append(r)
        sub = [r for r in rows if r["dataset"] == dataset]
        for tag in ("train_line", "cert_line", "observed"):
            sd1 = np.array([r[f"{tag}_sd1"] for r in sub])
            sm = np.array([r[f"{tag}_sigma_mean"] for r in sub])
            mg = sd1 / 2
            print(f"{dataset:8s} {tag:10s} ({len(sub)} pools, k {sub[0]['k']}, e {sub[0]['e']}, ref m {ref_m}, "
                  f"L {sorted({r['reach_L'] for r in sub})}): sigma median {np.median(sm):.3f} "
                  f"({sm.min():.3f}-{sm.max():.3f}) nats | sd_1 median {np.median(sd1):.3f} "
                  f"({sd1.min():.3f}-{sd1.max():.3f}) | 2 s.e. at K=16 median {np.median(mg):.3f} "
                  f"({mg.min():.3f}-{mg.max():.3f}); 0.13 is {0.13 / np.median(mg):.2f}x that, "
                  f"{0.13 / np.median(sm):.1%} of the median sigma | the read's own s.e. (200 splits) "
                  f"median {np.median(sd1) / np.sqrt(200):.3f}", flush=True)
    return rows


def read(run, step, margin):
    """strict_table.read with the margin as an argument, plus the share of pools the band clamps."""
    ev = RUNS / run / "snapshot_evals"
    f = ev / f"step{step:04d}_test_forecast_traces.npz"
    if not f.exists():
        return None
    base, k = pools(ev / "base_test_forecast_traces.npz")
    trained, _ = pools(f)
    rows, clamped, infeasible = [], 0, 0
    for name, (vb, M, e, perms, eb, sb, _sref) in base_errors(base, k).items():
        vt = trained[name][0]
        out, tgt, e2, L, sref, info = cert_scale.certify(vb, M, k)
        lo, hi = sref - margin, sref + margin
        s_t = involved_sigma(vt, M, k, e)
        clamped += not (lo <= s_t <= hi)
        vc = vt * (min(max(s_t, lo), hi) / s_t)
        vf, ok = project(vc, vb, M, k, e, lo, hi)
        infeasible += not ok
        ec, _ = split_errors(out, perms, M, k=k)
        ef, _ = split_errors(vf, perms, M, k=k)
        et, _ = split_errors(vt, perms, M, k=k)
        rows.append((float((eb ** 2).mean()), float((ec ** 2).mean()), float((ef ** 2).mean()), float((et ** 2).mean())))
    return np.array(rows), clamped, infeasible


def part2(step):
    rows = []
    for margin in MARGINS:
        cert_scale.MARGIN = margin  # certify() reads the module global at call time
        for dataset, label, name in st.ARMS:
            per = {}
            for s in st.SEEDS:
                r = read(name(s, step), step, margin)
                if r is not None:
                    per[s] = r
            if not per:
                continue
            allr = np.concatenate([r[0] for r in per.values()])
            b, c, f, t = allr.sum(axis=0)
            clamped = sum(r[1] for r in per.values())
            infeas = sum(r[2] for r in per.values())
            seed_folds = {s: r[0][:, 0].sum() / r[0][:, 2].sum() for s, r in per.items()}
            rows.append({"dataset": dataset, "arm": label, "step": step, "margin": margin, "seeds": len(per),
                         "pools": len(allr), "fold": b / f, "cert_fold": b / c, "fraction_of_cert": c / f,
                         "asread_fold": b / t, "pools_clamped": clamped, "pools_infeasible": infeas,
                         **{f"fold_s{s}": v for s, v in seed_folds.items()}})
            print(f"margin {margin:4.2f} {dataset:8s} {label:30s}: {len(per)} seeds, spread-normalized "
                  f"{b / f:6.2f}x, certificate {b / c:6.2f}x ({c / f:.2f} of it), as read {b / t:8.2f}x | "
                  f"band clamps {clamped}/{len(allr)} pools{f', {infeas} infeasible' if infeas else ''}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--out", default="margin_calibration")
    ap.add_argument("--part", choices=("1", "2", "both"), default="both")
    a = ap.parse_args()
    for part, fn in (("1", part1), ("2", part2)):
        if a.part not in (part, "both"):
            continue
        rows = fn(a.step)
        path = READOUTS / f"{a.out}_part{part}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
