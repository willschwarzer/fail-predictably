"""The normalized error G = MSE / (sigma / ln 10)^2 over the certificate grid: rank-combinatorics
theory for straight tails against the measured certificate and the general-tail benchmark.

For a straight tail v_r = -|v_1| r^(1/lambda) (floor_theory.py), the split-averaged MSE is
(|v_1| / ln 10)^2 F(lambda) and the involved-set sigma is |v_1| S(lambda), with F and S
expectations over random partitions of pool ranks alone (floor_theory.Floor, case A). So

    G_straight(M, N, k; lambda) = F(lambda) / S(lambda)^2   ->   C / D^2  for large lambda,

with C = E[(log r* - log J)^2] and D the mean spread of log pool rank over the involved set: a
function of the geometry and the line's depth only, with no fitted constant. Per grid cell we
report it at each pool's certified line (cert_scale.csv: lambda), next to the measured G of the
certificate and of the optimized general tail (tail_opt grid cell tables; evaluation partitions),
and the ratio G_opt / G_cert, the share of the straight floor that shape alone removes.

  python grid_theory.py CELLS_CSV [CELLS_CSV ...] [--draws 20000]
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from floor_theory import READOUTS, Floor, closed_form, e_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--pools", type=int, default=8, help="first pools per cell, as the grid used")
    a = ap.parse_args()
    rng = np.random.default_rng(20261104)
    certs = [r for r in csv.DictReader(open(READOUTS / "cert_scale.csv")) if r["source"] == "random" and r["status"] == "ok"]
    floors = {}
    for path in a.cells:
        rows = list(csv.DictReader(open(path)))
        band = rows[0]["band"] if rows else "?"
        print(f"\n== {Path(path).name} (band {band}); G on the independent evaluation partitions, mean over pools")
        print("      M       R      k     Dy   C_A     D   G_str(C/D^2)  G_str(F/S^2)  G_cert   G_opt  (cv)   "
              "opt/cert G  cert/base  opt/base  opt/cert fold  pools")
        for r in sorted(rows, key=lambda r: (int(r["M"]), float(r["R"]), int(r["k"]))):
            M, N, k = int(r["M"]), int(r["N"]), int(r["k"])
            e = e_of(M, N)
            key = (M, N, k, e)
            if key not in floors:
                d = max(4000, a.draws // 4) if k >= 1000 else a.draws
                floors[key] = Floor(M, N, k, e, d, rng, "A")
            fl = floors[key]
            pid = {f"rand{i:02d}" for i in range(a.pools)}
            lams = [float(c["cert_lambda"]) for c in certs
                    if c["cell"] == r["cell"] and int(c["k"]) == k and c["pool"] in pid]
            g_fs = float(np.mean([fl.F(l)[0] / fl.sigma(l) ** 2 for l in lams])) if lams else float("nan")
            g_cd = fl.C / fl.Dlog ** 2
            Dy = closed_form(M, N, k, "A")[4]
            gc = float(r["evaluation_certificate_G_mean"])
            go = float(r["evaluation_optimized_G_mean"])
            print(f"   {M:5d} {float(r['R']):7g} {k:6d}  {Dy:5.2f} {fl.C:6.3f} {fl.Dlog:5.3f}"
                  f"   {g_cd:9.4f}    {g_fs:9.4f}    {gc:7.4f} {go:7.4f} ({float(r['evaluation_optimized_G_cv']):4.2f})"
                  f"   {go / gc:6.3f}    {float(r['evaluation_certificate_over_base']):7.2f}  "
                  f"{float(r['evaluation_optimized_over_base']):7.2f}   {float(r['evaluation_optimized_over_certificate']):7.2f}"
                  f"        {r['completed_pools']}/{r['expected_pools']}", flush=True)


if __name__ == "__main__":
    main()
