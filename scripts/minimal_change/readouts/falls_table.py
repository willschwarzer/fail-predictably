"""Held-out falls per arm, seed and step: how far training lowered the top and the median risk score.

For each run and step this repeats the snapshot reader's own formulas
(scripts/minimal_change/snapshot_heldout_eval.py), on the same saved held-out traces:
  fall_top    = mean over held-out targets of (base max score - trained max score), nats
  fall_median = mean over held-out targets of (base median score - trained median score), nats
where a target's scores are its fit and deploy log-probabilities together, the base traces are the
run's snapshot_evals/base_test_forecast_traces.npz, and the trained traces are that directory's
stepNNNN_test_forecast_traces.npz. The run for each (arm, seed, step) is the one strict_table.py
reads (its ARMS mapping), so these rows line up with strict_table_200.csv and kl_table.csv.

Per arm and step it also writes a mean row (seed "mean") over the seeds whose traces exist, with
the standard error over seeds and the list of seeds included; a build can assert n_seeds == 5.

  python falls_table.py [--steps 100 200] [--out falls_table.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fleet_table import READOUTS, RUNS  # noqa: E402
from poolscale_readout import load  # noqa: E402
from strict_table import ARMS, SEEDS  # noqa: E402

KEEP = {"anchpen + band (main)", "IOG alone", "anchpen, no band", "IOG + band"}


def falls(run, step):
    ev = RUNS / run / "snapshot_evals"
    if not (ev / f"step{step:04d}_test_forecast_traces.npz").exists():
        return None
    base = load(ev, "base_")
    trained = load(ev, f"step{step:04d}_")
    if set(trained) != set(base):
        raise RuntimeError(f"{run} step {step}: target sets differ between base and trained traces")
    top = float(np.mean([base[t][0].max() - trained[t][0].max() for t in base]))
    med = float(np.mean([np.median(base[t][0]) - np.median(trained[t][0]) for t in base]))
    return top, med, len(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--out", default=str(READOUTS / "falls_table.csv"))
    a = ap.parse_args()
    cols = ["dataset", "arm", "seed", "step", "run", "fall_top", "fall_median", "n_targets",
            "se_top", "se_median", "n_seeds", "seeds"]
    rows = []
    for ds, arm, name in ARMS:
        if arm not in KEEP:
            continue
        for step in a.steps:
            got = []
            for s in SEEDS:
                run = name(s, step)
                r = falls(run, step)
                if r is None:
                    continue
                got.append((s, *r))
                rows.append(dict(dataset=ds, arm=arm, seed=s, step=step, run=run, fall_top=f"{r[0]:.4f}",
                                 fall_median=f"{r[1]:.4f}", n_targets=r[2]))
            if not got:
                continue
            t = np.array([g[1] for g in got])
            m = np.array([g[2] for g in got])
            se = (lambda x: float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan"))
            rows.append(dict(dataset=ds, arm=arm, seed="mean", step=step, run="", fall_top=f"{t.mean():.4f}",
                             fall_median=f"{m.mean():.4f}", n_targets="", se_top=f"{se(t):.4f}",
                             se_median=f"{se(m):.4f}", n_seeds=len(got),
                             seeds=" ".join(str(g[0]) for g in got)))
            print(f"{ds:8s} {arm:22s} step {step:3d}: top {t.mean():7.2f}  median {m.mean():7.2f}  "
                  f"({len(got)} seeds: {' '.join(str(g[0]) for g in got)})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
