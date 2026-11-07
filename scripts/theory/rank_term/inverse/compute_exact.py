"""Exact tables for the inverse forecaster's rank term (no Monte Carlo).

Writes exact.json next to this file.  Run on a compute node (a few minutes).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import optimize, special

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_exact as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "exact.json"
KLIST = list(range(2, 21)) + [30, 50, 100]
CELLS = [(10, 44, 891), (10, 143, 2857), (10, 250, 2050)]
CENTERED_FAMS = ("weibull", "hazen", "gringorten", "digamma")  # empirical == weibull centered
RS_APPG = [2, 5, 10, 20, 100, 300, 1000]
KS_APPG = [5, 7, 10, 15, 20, 30, 50]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    res = {}
    t0 = time.time()

    # ------------------------------------------------------------------ w_k table
    log("w_k closed form vs quadrature")
    wtab = {}
    for fam in CENTERED_FAMS:
        wtab[fam] = {}
        for k in KLIST:
            wc, prec = R.w_closed(fam, k)
            if k > 2:
                Q = R.Quad(R.centered_np(fam, k))
                wq = Q.EW()
                w2 = Q.EW2()
                elw = Q.ELW()
            else:
                c = R.centered_np(fam, 2)
                a = c[0] - c[1]
                wq = 1.0 / a
                w2 = 2.0 / a ** 2  # E[V_1^2]/a^2
                elw = 0.5 * 2.0 / a  # L = V_1/2, E[L W] = E[V_1^2]/(2a)
            s = R.s_forward(fam, k)
            wtab[fam][k] = {"w_closed": wc, "w_closed_precdiff": prec, "w_quad": wq,
                            "EW2_quad": w2, "ELW_quad": elw, "s": s,
                            "sdW": math.sqrt(max(w2 - wq ** 2, 0.0))}
        log(f"  {fam} done ({time.time()-t0:.1f}s)")
    res["w_table"] = wtab

    # ---------------------------------------------- Weibull zero contour, k = 2..1000
    log("Weibull limit: w_k, beta_k, R_0(k) for k = 2..1000 (quadrature; closed form k<=150)")
    zc = {}
    j_all = np.arange(1, 1001, dtype=float)
    Hjm1_all = special.digamma(j_all) + R.EULER
    for k in range(2, 1001):
        if k == 2:
            w = 1.0 / math.log(2.0)
            wcl = w
        else:
            w = R.Quad(R.centered_np("weibull", k)).EW()
            wcl = R.w_closed("weibull", k, check=False)[0] if k <= 150 else float("nan")
        s = R.s_forward("weibull", k)
        lk = math.lgamma(k + 1) / k
        beta = w * lk - float(np.mean(Hjm1_all[:k]))
        beta_fwd = s * lk - float(np.mean(Hjm1_all[:k]))
        logR0 = -beta / (w - 1.0)
        zc[k] = {"w": w, "w_closed": wcl, "s": s, "beta": beta, "beta_fwd": beta_fwd,
                 "logR0": logR0, "R0": math.exp(logR0) if logR0 < 700 else float("inf"),
                 "Exi_R1": beta, "Exi_R2": beta + (w - 1.0) * math.log(2.0)}
        if k % 100 == 0:
            log(f"  k={k} w={w:.6f} beta={beta:+.5f} R0={zc[k]['R0']:.4g} ({time.time()-t0:.1f}s)")
    res["weibull_limit"] = zc
    pos_R1 = [k for k in zc if zc[k]["beta"] > 0]
    pos_R2 = [k for k in zc if zc[k]["Exi_R2"] > 0]
    res["weibull_sets"] = {
        "k_all_R_ge_1_positive": [min(pos_R1), max(pos_R1)],
        "k_all_R_ge_1_positive_is_interval": pos_R1 == list(range(min(pos_R1), max(pos_R1) + 1)),
        "k_all_R_ge_2_positive": [min(pos_R2), max(pos_R2)],
        "k_all_R_ge_2_positive_is_interval": pos_R2 == list(range(min(pos_R2), max(pos_R2) + 1)),
        "w_decreasing_in_k_from_3": all(zc[k + 1]["w"] < zc[k]["w"] for k in range(3, 1000)),
        "beta_decreasing_in_k": all(zc[k + 1]["beta"] < zc[k]["beta"] for k in range(2, 1000)),
        "w_3_gt_w_2": zc[3]["w"] > zc[2]["w"],
    }
    # zero contour k_0(R) for R in the figure range
    k0 = {}
    for Rv in [1.0, 1.5, 2, 3, 5, 10, 20, 30, 100, 300, 1000, 3000, 10000]:
        vals = [(k, (zc[k]["w"] - 1.0) * math.log(Rv) + zc[k]["beta"]) for k in range(2, 1001)]
        neg = [k for k, v in vals if v < 0]
        k0[str(Rv)] = {"first_negative_k": min(neg) if neg else None,
                       "negatives_are_tail": (neg == list(range(min(neg), 1001))) if neg else None}
    res["weibull_k0_of_R"] = k0
    log(f"  sets: {res['weibull_sets']}")

    # ---------------------------------------------------------- plotting-positions appendix grid
    log("plotting-positions appendix grid (limit, read-off log N)")
    grid = {}
    for fam in ("weibull", "gringorten", "hazen", "digamma"):
        vals = {}
        for k in KS_APPG:
            w = wtab[fam][k]["w_quad"] if k in wtab[fam] else R.Quad(R.centered_np(fam, k)).EW()
            for Rv in RS_APPG:
                vals[f"{k},{Rv}"] = R.mean_xi_limit(fam, k, Rv, w, rread=math.log(Rv))["Exi"]
        arr = np.array(list(vals.values()))
        own = {}
        for k in KS_APPG:
            w = wtab[fam][k]["w_quad"] if k in wtab[fam] else R.Quad(R.centered_np(fam, k)).EW()
            for Rv in RS_APPG:
                own[f"{k},{Rv}"] = R.mean_xi_limit(fam, k, Rv, w)["Exi"]
        arr2 = np.array(list(own.values()))
        grid[fam] = {"readoff_logN": vals, "min": float(arr.min()), "max": float(arr.max()),
                     "n_negative": int((arr < 0).sum()),
                     "readoff_own": own, "own_min": float(arr2.min()), "own_max": float(arr2.max()),
                     "own_n_negative": int((arr2 < 0).sum())}
        log(f"  {fam}: logN read-off range [{arr.min():+.4f}, {arr.max():+.4f}], neg {int((arr<0).sum())}/49; "
            f"own read-off range [{arr2.min():+.4f}, {arr2.max():+.4f}], neg {int((arr2<0).sum())}/49")
    res["appendixG_grid"] = grid

    # ------------------------------------------------------------ k=10 limit row
    row = {}
    for fam in ("weibull", "hazen", "gringorten", "digamma"):
        w = wtab[fam][10]["w_quad"]
        row[fam] = {}
        for Rv in [1, 2, 5, 10, 20, 100, 300, 1000, 10000]:
            a = R.mean_xi_limit(fam, 10, Rv, w, rread=math.log(Rv))
            b = R.mean_xi_limit(fam, 10, Rv, w)
            row[fam][str(Rv)] = {"readoff_logN": a, "readoff_own": b}
    res["k10_limit"] = row

    # -------------------------------------------------------------- paper cells
    log("paper cells (finite M)")
    cells = {}
    for (k, M, N) in CELLS:
        key = f"{k},{M},{N}"
        cells[key] = {}
        for fam in R.FAMILIES:
            cf = "weibull" if fam == "empirical" else fam
            w = wtab[cf][k]["w_quad"]
            e2 = wtab[cf][k]["EW2_quad"]
            elw = wtab[cf][k]["ELW_quad"]
            out = {}
            for conv, ustar in (("own", None), ("logN", math.log(N)), ("logN1", math.log(N + 1.0))):
                d = R.mean_xi_finite(fam, k, M, N, w, ustar)
                d["var"] = R.var_xi_finite(fam, k, M, N, d["ustar_minus_ubar"], w, e2, elw)
                d["sd"] = math.sqrt(d["var"])
                out[conv] = d
            cells[key][fam] = out
        log(f"  {key}: weibull own {cells[key]['weibull']['own']['Exi']:+.5f}, "
            f"gringorten logN {cells[key]['gringorten']['logN']['Exi']:+.5f}")
    res["cells"] = cells

    # limit decomposition at k = 10, R = 20
    dec = {}
    for fam in ("weibull", "hazen", "gringorten", "digamma"):
        w = wtab[fam][10]["w_quad"]
        dec[fam] = {"logN": R.mean_xi_limit(fam, 10, 20.0, w, rread=math.log(20.0)),
                    "own": R.mean_xi_limit(fam, 10, 20.0, w)}
        e2 = wtab[fam][10]["EW2_quad"]
        elw = wtab[fam][10]["ELW_quad"]
        ab = float(R.positions(fam, 10, None).mean())
        dec[fam]["logN"]["var"] = R.var_xi_finite(fam, 10, None, None, math.log(20.0) - ab, w, e2, elw)
    res["decomp_k10_R20_limit"] = dec

    # ---------------------------------------------- finite-M zero contour (Weibull)
    log("finite-M zero contour N_0(k) at the paper M's (Weibull, read-off log(N+1))")
    fz = {}
    for M in (44, 143, 250):
        fz[M] = {}
        for k in list(range(2, 21)) + [25, 30, 40, 50, 60, 75, 100]:
            if k > M:
                continue
            w = wtab["weibull"][k]["w_quad"] if k in wtab["weibull"] else R.Quad(R.centered_np("weibull", k)).EW()
            u = R.positions("weibull", k, M)
            ub = float(u.mean())
            mb = R.mbar_finite(k, M)

            def f(logN1):
                N = math.exp(logN1) - 1.0
                return mb + (logN1 - ub) * w - float(special.digamma(N + 1.0) + R.EULER)

            lo, hi = math.log(2.0), math.log(1e15)
            flo, fhi = f(lo), f(hi)
            N0 = None
            if flo < 0 < fhi:
                N0 = math.exp(optimize.brentq(f, lo, hi, xtol=1e-12)) - 1.0
            fz[M][k] = {"N0": N0, "R0": (N0 / M) if N0 else None,
                        "Exi_at_N_eq_M": f(math.log(M + 1.0)), "Exi_at_N_eq_2M": f(math.log(2 * M + 1.0)),
                        "Exi_at_N_eq_1e4M": f(math.log(1e4 * M + 1.0))}
    res["finite_zero_contour"] = fz
    k10 = {M: fz[M][10] for M in fz}
    log(f"  k=10: {k10}")

    # ------------------------------------------------------------------ I3 checks
    log("I3: single-N and {3,4},{3,4,5} solvability")
    i3 = {}
    # (a) single N: k=2 and k=3 jointly unbiased at one N with shared positions + read-off
    M, N = 143, 2857
    u12 = R.positions("weibull", 2, M)
    a = u12[0] - u12[1]
    m = np.array([R.harmonic(M) - R.harmonic(j - 1) for j in range(1, 4)])
    HN = R.harmonic(N)
    ustar = u12.mean() + a * (HN - m[:2].mean())

    def bias3(u3):
        u = np.array([u12[0], u12[1], u3])
        c = u - u.mean()
        w3 = R.Quad(c).EW()
        return m.mean() + (ustar - u.mean()) * w3 - HN

    scan = [(u3, bias3(u3)) for u3 in np.linspace(u12[1] - 0.05, u12[1] - 3.0, 60)]
    root = None
    for (x0, f0), (x1, f1) in zip(scan[:-1], scan[1:]):
        if f0 * f1 < 0:
            root = optimize.brentq(bias3, x0, x1, xtol=1e-13)
            break
    i3["single_N"] = {"M": M, "N": N, "u1": float(u12[0]), "u2": float(u12[1]), "ustar": float(ustar),
                      "u3_root": root, "weibull_u3": float(math.log((M + 1) / 3.0)),
                      "bias2_at_root": 0.0, "bias3_at_root": bias3(root) if root else None,
                      "scan": [[float(x), float(y)] for x, y in scan[::6]]}
    # (b) {3,4} jointly unbiased for all N: normalise u1=0, u2=-1, free u3; u4 = ubar3 - 1/w3
    mfull = np.array([-R.harmonic(j - 1) for j in range(1, 6)])

    def w_of(u):
        c = np.asarray(u, float)
        c = c - c.mean()
        return R.Quad(c).EW()

    def F34(u3):
        u = [0.0, -1.0, u3]
        w3 = w_of(u)
        mb3 = mfull[:3].mean()
        ub3 = np.mean(u)
        u4 = ub3 + (mfull[3] - mb3) / w3
        return w_of(u + [u4]) - w3, u4, w3

    grid34 = []
    for u3 in np.concatenate([np.linspace(-1.02, -3.0, 100), np.linspace(-3.05, -8.0, 40)]):
        try:
            v, u4, w3 = F34(u3)
        except ValueError:
            v, u4, w3 = float("nan"), float("nan"), float("nan")
        grid34.append((float(u3), float(v), float(u4), float(w3)))
    roots34 = []
    for (x0, f0, *_), (x1, f1, *_) in zip(grid34[:-1], grid34[1:]):
        if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 < 0:
            r = optimize.brentq(lambda x: F34(x)[0], x0, x1, xtol=1e-13)
            v, u4, w3 = F34(r)
            u = [0.0, -1.0, r, u4]
            # {3,4,5}: u5 forced by the 4-5 level condition; check w5 - w3
            mb4 = mfull[:4].mean()
            ub4 = np.mean(u)
            u5 = ub4 + (mfull[4] - mb4) / w3
            try:
                w5 = w_of(u + [u5])
            except ValueError:
                w5 = float("nan")
            roots34.append({"u3": r, "u4": u4, "w": w3, "w4_minus_w3": v, "u5_forced": float(u5),
                            "w5_minus_w3": float(w5 - w3) if np.isfinite(w5) else None,
                            "decreasing": bool(0 > -1 > r > u4)})
    i3["pair34"] = {"roots": roots34, "scan": grid34[::10]}
    # (c) w_3(digamma) exact
    i3["w3_digamma"] = float(mp.mpf(14) / 9 * mp.log(mp.mpf(5) / 2) - mp.mpf(1) / 3)
    res["I3"] = i3
    log(f"  single-N root u3={root}; pair34 roots={roots34}")

    OUT.write_text(json.dumps(res, indent=1, default=float))
    log(f"wrote {OUT} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
