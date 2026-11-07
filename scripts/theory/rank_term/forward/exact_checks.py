"""Exact (rational / high-precision) checks for the forward-forecaster results.

Everything here is deterministic algebra; no Monte Carlo.  Runs in well
under 2 s.

Conventions (depth units, Exp(1) scores, so sigma = 1, mu = 0):
  m_j = E[D_(j)] = H_M - H_{j-1}           (mean depth of the j-th highest of M)
  forward forecast  F = xbar + (u* - ubar) S_ux / S_uu   (OLS of x on u)
  weights           w_j = 1/k + (u* - ubar)(u_j - ubar)/S_uu, so F = w . x
  E[F] = w . m ;  Var(F) = sum_{l=1}^{M} (W_{min(l,k)} / l)^2,  W_l = w_1+...+w_l
  (Renyi: D_(j) = sum_{l=j}^{M} E_l / l).

Output: exact_checks.json next to this file.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50
OUT = Path(__file__).with_name("exact_checks.json")

CELLS = {  # (k, M, N) of the paper
    "password": (10, 44, 891),
    "wildchat_compliance": (10, 143, 2857),
    "wildchat_minimal_change": (10, 250, 2050),
}


_HCACHE = [Fr(0)]


def H(n):
    while len(_HCACHE) <= n:
        _HCACHE.append(_HCACHE[-1] + Fr(1, len(_HCACHE)))
    return _HCACHE[n]


def Hmp(n):
    return mp.harmonic(n)


def trig_tail(a, M):
    """sum_{l=a}^{M} 1/l^2 = psi'(a) - psi'(M+1)."""
    return mp.polygamma(1, a) - mp.polygamma(1, M + 1)


# ----------------------------------------------------------------------------
# 1. Rational identities used in the proofs
# ----------------------------------------------------------------------------

def check_identities(kmax=40, M=120):
    HM = H(M)
    bad = []
    for k in range(2, kmax + 1):
        m = [HM - H(j - 1) for j in range(1, k + 1)]
        mbar = sum(m) / k
        Smm = sum((x - mbar) ** 2 for x in m)
        if mbar != HM - H(k) + 1:
            bad.append(("mbar", k))
        if Smm != k - H(k):
            bad.append(("Smm", k))
        if m[-1] - mbar != Fr(-(k - 1), k):
            bad.append(("g", k))
        # spacing weights of digamma-OLS read at u* = H_N, N symbolic -> test 2 values
        for N in (M + 7, 5 * M + 3):
            HN = H(N)
            w = [Fr(1, k) + (HN - mbar) * (x - mbar) / Smm for x in m]
            W = list(itertools.accumulate(w))
            A = (HN - mbar) / (k - H(k))
            for l in range(1, k):
                if W[l - 1] / l != Fr(1, k) + A * (H(k) - H(l)):
                    bad.append(("Wl", k, N, l))
            if W[-1] != 1:
                bad.append(("Wk", k, N))
            if sum(wi * mi for wi, mi in zip(w, m)) != HN:
                bad.append(("unbiased", k, N))
    return {"kmax": kmax, "M": M, "failures": bad}


# ----------------------------------------------------------------------------
# 2. One deployment size: enumerate every solution of the unbiasedness system
#    (normalized by the affine group: u_1 = m_1, u_2 = m_2, which forces u* = H_N)
# ----------------------------------------------------------------------------

def quad_coeffs(prefix, k, m, HN):
    """f(v) = S_um (u* - ubar) - (H_N - mbar) S_uu with u_k = v, u* = H_N."""
    mbar = sum(m[:k]) / k
    mc = [m[j] - mbar for j in range(k)]
    P = sum(prefix)
    Q = sum(x * x for x in prefix)
    R = sum(prefix[j] * mc[j] for j in range(k - 1))
    D = HN - mbar
    c0 = R * (HN - P / k) - D * (Q - P * P / k)
    c1 = mc[k - 1] * (HN - P / k) - R / k + 2 * D * P / k
    c2 = -mc[k - 1] / k - D * (1 - mp.mpf(1) / k)
    return c2, c1, c0


def real_roots(c2, c1, c0, tol=mp.mpf("1e-40")):
    if abs(c2) < tol:
        if abs(c1) < tol:
            return None  # degenerate (identity or contradiction)
        return [-c0 / c1]
    disc = c1 * c1 - 4 * c2 * c0
    if disc < -tol:
        return []
    if abs(disc) <= tol:
        return [-c1 / (2 * c2)]
    sq = mp.sqrt(disc)
    return sorted([(-c1 - sq) / (2 * c2), (-c1 + sq) / (2 * c2)])


def exact_bias(u, ustar, m, HN):
    k = len(u)
    ubar = sum(u) / k
    mbar = sum(m[:k]) / k
    Suu = sum((x - ubar) ** 2 for x in u)
    Sum = sum((u[j] - ubar) * (m[j] - mbar) for j in range(k))
    return mbar + (ustar - ubar) * Sum / Suu - HN


def enumerate_one_N(M, N, K):
    m = [Hmp(M) - Hmp(j - 1) for j in range(1, K + 1)]
    HN = Hmp(N)
    branches = [[m[0], m[1]]]
    for k in range(3, K + 1):
        new = []
        for pre in branches:
            roots = real_roots(*quad_coeffs(pre, k, m, HN))
            if roots is None:
                raise RuntimeError("degenerate quadratic")
            for v in roots:
                new.append(pre + [v])
        branches = new
    out = []
    for u in branches:
        maxbias = max(abs(exact_bias(u[:k], HN, m, HN)) for k in range(2, K + 1))
        mono = all(u[j] > u[j + 1] for j in range(K - 1))
        dev = max(abs(u[j] - m[j]) for j in range(K))
        out.append({"u_minus_m": [float(u[j] - m[j]) for j in range(K)],
                    "strictly_decreasing": mono, "is_digamma": dev < 1e-30,
                    "max_abs_bias_over_k": float(maxbias)})
    return out


def second_root_digamma_prefix(k, Delta):
    """Spurious root t_k (u_k = m_k + t_k) when u_1..u_{k-1} = m and u* = H_N."""
    Hk = Hmp(k)
    num = (k - 1) * Delta + k * Hk - 2 * k + 1
    den = (k - 1) * (Delta + Hmp(k - 1) - 1)
    return num / den


# ----------------------------------------------------------------------------
# 3. Two deployment sizes: Jacobian rank of the 2(K-1) equations at the
#    digamma solution (should be K, i.e. a 2-dim solution set = the affine orbit)
# ----------------------------------------------------------------------------

def two_N_jacobian_rank(M, N1, N2, K):
    m = [Hmp(M) - Hmp(j - 1) for j in range(1, K + 1)]
    H1, H2 = Hmp(N1), Hmp(N2)

    def eqs(z):
        u = list(z[:K])
        us1, us2 = z[K], z[K + 1]
        res = []
        for k in range(2, K + 1):
            res.append(exact_bias(u[:k], us1, m, H1))
            res.append(exact_bias(u[:k], us2, m, H2))
        return res

    z0 = m + [H1, H2]
    h = mp.mpf("1e-20")
    J = []
    for i in range(K + 2):
        zp = list(z0); zp[i] += h
        zm = list(z0); zm[i] -= h
        fp, fm = eqs(zp), eqs(zm)
        J.append([(a - b) / (2 * h) for a, b in zip(fp, fm)])
    Jm = mp.matrix(J).T  # rows = equations
    sv = mp.svd_r(Jm, compute_uv=False)
    svs = sorted([float(s) for s in sv], reverse=True)
    rank = sum(1 for s in svs if s > 1e-12 * svs[0])
    return {"n_eq": 2 * (K - 1), "n_unknowns": K + 2, "rank": rank, "singular_values": svs}


# ----------------------------------------------------------------------------
# 4. Variances, biases of linear forecasts in closed form
# ----------------------------------------------------------------------------

def fwd_weights(u, ustar):
    k = len(u)
    ubar = sum(u) / k
    Suu = sum((x - ubar) ** 2 for x in u)
    return [mp.mpf(1) / k + (ustar - ubar) * (x - ubar) / Suu for x in u]


def lin_mean_var(w, M):
    """Mean and variance (sigma = 1) of w . D_(1..k), D = top-k depths of M Exp(1)."""
    k = len(w)
    m = [Hmp(M) - Hmp(j - 1) for j in range(1, k + 1)]
    mean = sum(wi * mi for wi, mi in zip(w, m))
    W = list(itertools.accumulate(w))
    var = sum((W[l - 1] / l) ** 2 for l in range(1, k)) + W[-1] ** 2 * trig_tail(k, M)
    return mean, var


def cov_matrix(k, M):
    return mp.matrix([[trig_tail(max(i, j), M) for j in range(1, k + 1)] for i in range(1, k + 1)])


def positions(name, k, M):
    js = range(1, k + 1)
    if name == "digamma":
        return [Hmp(M) - Hmp(j - 1) for j in js]
    if name == "weibull":
        return [mp.log(mp.mpf(M + 1) / j) for j in js]
    if name == "empirical":
        return [mp.log(mp.mpf(M) / j) for j in js]
    if name == "hazen":
        return [mp.log(mp.mpf(M) / (j - mp.mpf("0.5"))) for j in js]
    if name == "gringorten":
        return [mp.log((M + mp.mpf("0.12")) / (j - mp.mpf("0.44"))) for j in js]
    raise ValueError(name)


def readoff(name, N):
    if name == "digamma":
        return Hmp(N)
    if name == "weibull":
        return mp.log(N + 1)
    if name == "empirical":
        return mp.log(N)
    if name == "hazen":
        return mp.log(mp.mpf(N) / mp.mpf("0.5"))
    if name == "gringorten":
        return mp.log((N + mp.mpf("0.12")) / mp.mpf("0.56"))
    raise ValueError(name)


def cell_table(k, M, N):
    HN, HM = Hmp(N), Hmp(M)
    Dp = HN - HM + Hmp(k - 1)  # H_N - m_k
    varY = mp.polygamma(1, 1) - mp.polygamma(1, N + 1)
    tail = trig_tail(k, M)
    rows = {}
    # BLUE / UMVU
    beta = Dp / (k - 1)
    w_blue = [beta] * (k - 1) + [1 - (k - 1) * beta]
    mean, var = lin_mean_var(w_blue, M)
    rows["BLUE_UMVU"] = {"bias": mean - HN, "var": var,
                         "var_closed_form": tail + Dp ** 2 / (k - 1)}
    # GLS (Lloyd) cross-check: BLUE of mu + sigma*H_N from Cov = C
    C = cov_matrix(k, M)
    m = positions("digamma", k, M)
    A = mp.matrix([[1, mj] for mj in m])
    Ci = C ** -1
    G = (A.T * Ci * A) ** -1 * A.T * Ci  # 2 x k
    w_gls = [G[0, j] + HN * G[1, j] for j in range(k)]
    rows["BLUE_UMVU"]["max_abs_diff_vs_GLS_weights"] = max(abs(a - b) for a, b in zip(w_gls, w_blue))
    # MRE (minimum-MSE equivariant): X_(k) + S * Dp / k
    c = Dp / k
    w_mre = [c] * (k - 1) + [1 - (k - 1) * c]
    mean, var = lin_mean_var(w_mre, M)
    rows["MRE"] = {"bias": mean - HN, "var": var, "bias_closed_form": -Dp / k,
                   "mse_closed_form": tail + Dp ** 2 / k}
    # forward OLS, each convention with its own read-off
    for name in ("digamma", "weibull", "empirical", "hazen", "gringorten"):
        u = positions(name, k, M)
        w = fwd_weights(u, readoff(name, N))
        mean, var = lin_mean_var(w, M)
        d = {"bias": mean - HN, "var": var}
        if name == "digamma":
            Aco = (HN - (HM - Hmp(k) + 1)) / (k - Hmp(k))
            Hs = [Hmp(l) for l in range(1, k)]
            Hb = sum(Hs) / (k - 1)
            d["var_closed_form"] = tail + Dp ** 2 / (k - 1) + Aco ** 2 * sum((h - Hb) ** 2 for h in Hs)
            # cross-check against w' C w
            wv = mp.matrix(w)
            d["var_wCw"] = (wv.T * C * wv)[0]
        rows[f"fwd_{name}"] = d
        # same positions, k-dependent read-off that makes it exactly unbiased (F2 ii)
        if name != "digamma":
            ubar = sum(u) / k
            mbar = sum(m) / k
            Suu = sum((x - ubar) ** 2 for x in u)
            Sum = sum((u[j] - ubar) * (m[j] - mbar) for j in range(k))
            ustar = ubar + (HN - mbar) * Suu / Sum
            w = fwd_weights(u, ustar)
            mean, var = lin_mean_var(w, M)
            rows[f"fwd_{name}_corrected_readoff"] = {"bias": mean - HN, "var": var,
                                                    "ustar": ustar, "ustar_minus_HN_affine_ref": None}
    # digamma positions with quantile read-offs (F4)
    u = positions("digamma", k, M)
    for lab, us in (("logN", mp.log(N)), ("log(N+1)", mp.log(N + 1)), ("logN+gamma", mp.log(N) + mp.euler)):
        w = fwd_weights(u, us)
        mean, var = lin_mean_var(w, M)
        rows[f"fwd_digamma_read_{lab}"] = {"bias": mean - HN, "var": var,
                                          "bias_vs_quantile_logN": mean - mp.log(N)}
    out = {}
    for key, d in rows.items():
        dd = {kk: (float(v) if v is not None else None) for kk, v in d.items()}
        if "var" in d and "bias" in d:
            dd["E_xi2"] = float(d["var"] + varY + d["bias"] ** 2)
            dd["rms_xi"] = math.sqrt(dd["E_xi2"])
        out[key] = dd
    return {"k": k, "M": M, "N": N, "VarY": float(varY), "H_N_minus_m_k": float(Dp), "rows": out}


# ----------------------------------------------------------------------------
# 5. Line-unbiased positions at a fixed k (F3): the sphere, and the rule-tied
#    (k-3)-sphere
# ----------------------------------------------------------------------------

def fixed_k_sphere_examples(M=143):
    k = 3
    m = [Hmp(M) - Hmp(j - 1) for j in range(1, k + 1)]
    mbar = sum(m) / k
    mc = [x - mbar for x in m]
    # circle: uc on sphere |uc - mc/2| = |mc|/2 inside sum-zero plane
    # orthonormal basis of the sum-zero plane: e1 along mc, e2 orthogonal
    nmc = mp.sqrt(sum(x * x for x in mc))
    e1 = [x / nmc for x in mc]
    v = [1, -2, 1]
    v = [x - sum(v) / 3 for x in v]
    dot = sum(a * b for a, b in zip(v, e1))
    v = [a - dot * b for a, b in zip(v, e1)]
    nv = mp.sqrt(sum(x * x for x in v))
    e2 = [x / nv for x in v]
    res = []
    for th in (mp.pi / 7, mp.pi / 3, 2 * mp.pi / 3):
        uc = [nmc / 2 * (e1[j] + mp.cos(th) * e1[j] + mp.sin(th) * e2[j]) for j in range(k)]
        u = [mbar + x for x in uc]
        ubar = sum(u) / k
        Suu = sum((x - ubar) ** 2 for x in u)
        Sum = sum((u[j] - ubar) * (m[j] - mbar) for j in range(k))
        res.append({"theta": float(th), "u_minus_m": [float(u[j] - m[j]) for j in range(k)],
                    "slope_s": float(Sum / Suu), "ubar_minus_mbar": float(ubar - mbar),
                    "strictly_decreasing": bool(u[0] > u[1] > u[2])})
    return res


def main():
    t = {}
    t["identities"] = check_identities()
    # one-N enumeration at the paper cells, K = 10, and a few smaller K
    t["one_N_enumeration"] = {}
    for name, (k, M, N) in CELLS.items():
        for K in (3, 4, 5, 10):
            sols = enumerate_one_N(M, N, K)
            t["one_N_enumeration"][f"{name}_K{K}"] = {
                "n_real_solutions_mod_affine": len(sols),
                "n_strictly_monotone": sum(s["strictly_decreasing"] for s in sols),
                "n_digamma": sum(s["is_digamma"] for s in sols),
                "max_abs_bias_any_solution": max(s["max_abs_bias_over_k"] for s in sols),
                "solutions": sols if K <= 4 else sols[:4],
            }
    # spurious k=3 root closed form vs enumeration; N < M counterexample
    M, N = 100, 80
    Delta = Hmp(N) - Hmp(M)
    t3 = second_root_digamma_prefix(3, Delta)
    sols = enumerate_one_N(M, N, 3)
    t["N_less_than_M_counterexample"] = {
        "M": M, "N": N, "Delta": float(Delta), "t3_closed_form": float(t3),
        "t3_formula_2D+1/2_over_2D+1": float((2 * Delta + mp.mpf(1) / 2) / (2 * Delta + 1)),
        "solutions": sols}
    # bad intervals for one-N monotone uniqueness (digamma prefix)
    t["bad_Delta_intervals"] = {}
    for k in range(3, 21):
        lo = 1 - Hmp(k - 1)
        ck = 2 * k - 3 - (k - 1) * Hmp(k - 1)
        t["bad_Delta_intervals"][k] = [float(lo), float(ck / (k - 2))]
    # two N: Jacobian rank
    t["two_N_jacobian"] = {f"K{K}": two_N_jacobian_rank(143, 2857, 1000, K) for K in (3, 5, 10)}
    # cells
    t["cells"] = {name: cell_table(*c) for name, c in CELLS.items()}
    t["fixed_k3_line_unbiased_circle"] = fixed_k_sphere_examples()
    # F4: H_N - log N - gamma at the cells
    t["F4_quantile_gap"] = {name: {"logN_minus_HN": float(mp.log(N) - Hmp(N)),
                                   "log(N+1)_minus_HN": float(mp.log(N + 1) - Hmp(N)),
                                   "logN+gamma_minus_HN": float(mp.log(N) + mp.euler - Hmp(N))}
                            for name, (k, M, N) in CELLS.items()}
    OUT.write_text(json.dumps(t, indent=1, default=float))
    # brief stdout summary
    print("identity failures:", t["identities"]["failures"][:5])
    for key, d in t["one_N_enumeration"].items():
        print(key, {kk: d[kk] for kk in ("n_real_solutions_mod_affine", "n_strictly_monotone", "n_digamma", "max_abs_bias_any_solution")})
    print("N<M counterexample:", t["N_less_than_M_counterexample"]["t3_closed_form"],
          [(s["u_minus_m"], s["strictly_decreasing"]) for s in t["N_less_than_M_counterexample"]["solutions"]])
    print("two-N jacobian ranks:", {k: v["rank"] for k, v in t["two_N_jacobian"].items()})
    for name, c in t["cells"].items():
        print(name, "VarY=%.5f" % c["VarY"])
        for row, d in c["rows"].items():
            print("   %-34s bias=%+.5f var=%.5f E[xi^2]=%.5f rms=%.4f" % (row, d["bias"], d["var"], d["E_xi2"], d["rms_xi"]),
                  {kk: round(v, 6) for kk, v in d.items() if kk not in ("bias", "var", "E_xi2", "rms_xi") and v is not None})
    print("circle:", t["fixed_k3_line_unbiased_circle"])
    print("F4 gaps:", t["F4_quantile_gap"])


if __name__ == "__main__":
    main()
