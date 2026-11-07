"""Per arm and step: held-out KL with thinking on (the trainer's prompts) and off (the scoring template).

Reads $RESULTS_ROOT/lquant_dynamics/readouts/kl_thinking_off_{password,wildchat}.csv, which
scripts/audits/kl_thinking_off.sbatch writes, and prints, per dataset, arm and step, the geometric mean over
seeds of each KL, the range over seeds, the base and trained models' mean top-1 probabilities in each mode,
and the largest gap between the recomputed thinking-on KL and the trainer's own (kl_eval_results.json, at a
run's last step).

  sbatch scripts/minimal_change/readouts/py_small.sbatch kl_thinking_off_summary.py [--suffix _left]
        (_left: the password rows with left truncation, kl_thinking_off_password_left.csv)
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
READOUTS = RESULTS_ROOT / "lquant_dynamics/readouts"
ORDER = ["anchpen + band (main)", "anchpen, no band", "IOG + band", "IOG alone"]


def gm(v):
    return math.exp(sum(math.log(max(x, 1e-12)) for x in v) / len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    rows = []
    for ds in ("password", "wildchat"):
        f = READOUTS / f"kl_thinking_off_{ds}{a.suffix}.csv"
        if f.exists():
            rows += list(csv.DictReader(open(f)))
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["arm"], int(r["step"]))].append(r)
    for (ds, arm, step) in sorted(by, key=lambda k: (k[0], k[2], ORDER.index(k[1]) if k[1] in ORDER else 9)):
        g = by[(ds, arm, step)]
        on = [float(r["kl_on"]) for r in g]
        off = [float(r["kl_off"]) for r in g]
        gaps = [abs(float(r["kl_on"]) / float(r["kl_on_trainer"]) - 1) for r in g if r["kl_on_trainer"]]
        tt = sum(float(r["trained_top1_off"]) for r in g) / len(g)
        print(f"{ds:8s} step {step:3d} {arm:22s} n={len(g)} | on {gm(on):.3g} ({min(on):.2g}-{max(on):.2g}) | "
              f"off {gm(off):.3g} ({min(off):.2g}-{max(off):.2g}) | off/on {gm(off) / gm(on):.3g} | "
              f"top-1 off base {g[0]['base_top1_off']} trained {tt:.3f} | cut {g[0]['n_cut']}/{g[0]['n_prompts']}"
              + (f" | max |on/trainer-1| {max(gaps):.3f}" if gaps else ""))


if __name__ == "__main__":
    main()
