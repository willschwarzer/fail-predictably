"""Pooled folds per run (or bank cell) from tail_opt aggregate CSVs: base over each tail, on the
independent evaluation partitions (expected MSE) and on the legacy scoring splits.

  python summarize_tail_opt.py tail_opt_fleet.csv [tail_opt_fleet_fitmax.csv ...]
"""
import csv
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    g = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["run"], r["step"], r["k"], r["band"])
        g[key][r["pool"]][r["tail"]] = r
    print(f"\n{path}: pooled MSE fold over base (sum of base MSE / sum of tail MSE); evaluation partitions | legacy splits")
    print(f"{'run':<26s} {'step':>4s} {'k':>3s} {'pools':>5s} | {'cert':>6s} {'straight':>8s} {'trained':>7s} {'optim':>6s} "
          f"{'opt/cert':>8s} | {'cert':>6s} {'trained':>7s} {'optim':>6s} | {'sig_hi':>6s}")
    for key in sorted(g):
        pools = g[key]
        def s(tail, col):
            vals = [p[tail][col] for p in pools.values() if tail in p]
            return sum(float(x) for x in vals) if len(vals) == len(pools) else float("nan")
        out = []
        for col in ("evaluation_mse", "legacy_mse"):
            b = s("base", col)
            out.append({t: b / s(t, col) for t in ("certificate", "straight_mse", "trained", "optimized")})
        at_hi = sum(abs(float(p["optimized"]["sigma_involved"]) - float(p["optimized"]["sigma_hi"])) < 1e-6 for p in pools.values())
        e, l = out
        print(f"{key[0]:<26s} {key[1]:>4s} {key[2]:>3s} {len(pools):>5d} | {e['certificate']:6.2f} {e['straight_mse']:8.2f} "
              f"{e['trained']:7.2f} {e['optimized']:6.2f} {e['optimized'] / e['certificate']:8.2f} | {l['certificate']:6.2f} "
              f"{l['trained']:7.2f} {l['optimized']:6.2f} | {at_hi:>3d}/{len(pools)}")
