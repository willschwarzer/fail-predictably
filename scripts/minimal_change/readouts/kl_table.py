"""Held-out single-token KL to the base model, per arm and seed, from each finished run's final eval.

Reads `kl_eval_results.json` (written by train_lquant.py's final evaluation suite, 256 held-out
prompts) in every run directory of the strict table's arms. An extension directory (`*200`) starts
as a copy of its 100-step source, so its KL file is the source's until the extension finishes; we
keep it only when it is newer than the run's step-200 adapter snapshot.

  python kl_table.py            -> $RESULTS_ROOT/lquant_dynamics/readouts/kl_table.csv, and one line per arm on stdout
"""
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
READOUTS = RESULTS_ROOT / "lquant_dynamics/readouts"
RUNS = RESULTS_ROOT / "lquant_dynamics/runs"
ARMS = [("wildchat", "anchpen + band (main)", "lqdyn_v2e2ap50sp"),
        ("wildchat", "anchpen + band (main), 200 steps", "lqdyn_v2e2ap50spx200"),
        ("wildchat", "IOG alone", "lqdyn_v2e2iogap50"),
        ("wildchat", "IOG alone, 200 steps", "lqdyn_v2e2iogap50x200"),
        ("wildchat", "anchpen, no band", "lqdyn_v2e2nbap50"),
        ("wildchat", "anchpen, no band, 200 steps", "lqdyn_v2e2nbap50x200"),
        ("wildchat", "IOG + band", "lqdyn_v2e2iogsbap50sp"),
        ("wildchat", "IOG + band, 200 steps", "lqdyn_v2e2iogsbap50spx200"),
        ("wildchat", "anchpen + one-sided band", "lqdyn_v2e2osap50sp"),
        ("wildchat", "anchpen + band, MLP adapter", "lqdyn_v2e2mlpap50sp"),
        ("password", "anchpen + band (main)", "lqpw_pw2apsp"),
        ("password", "anchpen + band (main), 200 steps", "lqpw_pw2apsp200"),
        ("password", "IOG alone", "lqpw_pw2iog"),
        ("password", "IOG alone, 200 steps", "lqpw_pw2iog200"),
        ("password", "anchpen, no band", "lqpw_pw2nbap"),
        ("password", "anchpen, no band, 200 steps", "lqpw_pw2nbap200"),
        ("password", "IOG + band, 200 steps", "lqpw_pw2iogsb200")]


def main():
    rows = []
    by_arm = defaultdict(list)
    for dataset, label, prefix in ARMS:
        for f in sorted(RUNS.glob(f"{prefix}_seed*/kl_eval_results.json")):
            m = re.fullmatch(rf"{re.escape(prefix)}_seed(\d+)", f.parent.name)
            if not m:
                continue
            steps = json.load(open(f.parent / "config.json")).get("steps")
            snap = f.parent / "snapshots" / f"step_{int(steps):04d}"
            if not snap.exists() or f.stat().st_mtime < snap.stat().st_mtime:
                continue  # an extension's copied KL file, or a run whose final eval has not run yet
            kl = float(json.load(open(f))["single_token"])
            rows.append({"dataset": dataset, "arm": label, "seed": int(m.group(1)), "steps": steps, "kl_single_token": kl})
            by_arm[(dataset, label)].append((int(m.group(1)), kl))
    READOUTS.mkdir(parents=True, exist_ok=True)
    with open(READOUTS / "kl_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for dataset, label, _ in ARMS:
        v = sorted(by_arm.get((dataset, label), []))
        if not v:
            continue
        gm = math.exp(sum(math.log(k) for _, k in v) / len(v))
        print(f"{dataset:8s} {label:34s} geometric mean {gm:.2e} over {len(v)} seeds | "
              + ", ".join(f"s{s} {k:.2e}" for s, k in v))
    print("wrote", READOUTS / "kl_table.csv")


if __name__ == "__main__":
    main()
