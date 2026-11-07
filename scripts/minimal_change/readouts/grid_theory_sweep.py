"""How the straight-tail floor depends on the line's depth: G = F(lambda) / S(lambda)^2 over lambda.

For a straight tail v_r = -|v_1| r^(1/lambda) (floor_theory.py), lambda = |v_1| / s is the top's depth
below log p = 0 in units of its spread s at the top (nats of log p per unit of log rank). The
split-averaged MSE is (|v_1| / ln 10)^2 F(lambda) and the involved-set sigma is |v_1| S(lambda), so
in the band (sigma fixed) the floor is MSE = (sigma / ln 10)^2 G with G = F / S^2, which tends to
C / D^2 for deep lines (lambda -> infinity). Per geometry of the certificate grid, this prints G
at a range of lambda, the deep limit, and G at the pools' certified lambda (cert_scale.csv).

  python grid_theory_sweep.py [--draws 20000]
"""
import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from floor_theory import READOUTS, Floor, e_of  # noqa: E402

LAMS = (1.5, 2, 3, 4, 5, 6, 8, 12, 20, 50, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(20261104)
    certs = [r for r in csv.DictReader(open(READOUTS / "cert_scale.csv"))
             if r["source"] == "random" and r["status"] == "ok" and r["pool"] in {f"rand{i:02d}" for i in range(8)}]
    cells = sorted({(int(r["M"]), int(r["N"]), int(r["k"]), r["cell"]) for r in certs})
    print("G = F(lambda) / S(lambda)^2 (straight-tail MSE over (sigma / ln 10)^2), case A, by depth lambda")
    print("    M      R      k    e  " + "".join(f"{l:>8g}" for l in LAMS) + "    C/D^2   lam_cert  G(lam_cert)  S(lam_cert)")
    rows = []
    for M, N, k, cell in cells:
        e = e_of(M, N)
        d = max(4000, a.draws // 4) if k >= 1000 else a.draws
        fl = Floor(M, N, k, e, d, rng, "A")
        g = [fl.F(l)[0] / fl.sigma(l) ** 2 for l in LAMS]
        lams = [float(r["cert_lambda"]) for r in certs if r["cell"] == cell and int(r["k"]) == k]
        lc = float(np.mean(lams))
        gc = float(np.mean([fl.F(l)[0] / fl.sigma(l) ** 2 for l in lams]))
        sc = float(np.mean([fl.sigma(l) for l in lams]))
        print(f"{M:5d} {N / M:6g} {k:6d} {e:4d}  " + "".join(f"{x:8.4f}" for x in g)
              + f"  {fl.C / fl.Dlog ** 2:7.3f}   {lc:6.2f}     {gc:7.4f}     {sc:7.3f}", flush=True)
        rows.append({"M": M, "N": N, "R": N / M, "k": k, "e": e, "C": fl.C, "Dlog": fl.Dlog,
                     "lam_cert": lc, "G_cert_theory": gc, "S_cert": sc,
                     **{f"G_lam{l:g}": x for l, x in zip(LAMS, g)}})
    out = READOUTS / "grid_theory_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
