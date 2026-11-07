"""Exact (finite-M) theory of the inverse forecaster's rank term.

Depth units: fit depths D_(1) >= ... >= D_(M) are Exp(1) order statistics.
Renyi: V_l = D_(l) - D_(l+1) ~ Exp(rate l), independent, and independent of D_(k).
Top-k centered depths are linear in V_1..V_{k-1}:
    D_(j) - Dbar = sum_l (1[j<=l] - l/k) V_l.
With centered positions c_j = u_j - ubar and b_l = sum_{j<=l} c_j,
    S_uD = b^T V,   S_DD = V^T A V,   A_{ll'} = min(l,l') - l l'/k.
W = S_DD / S_uD.  Inverse forecast: Dbar + (u* - ubar) W.

w_k = E[W] = int_0^inf E[V^T A V exp(-t b^T V)] dt
    = int_0^inf prod_l p_l/(t+p_l) * sum_{l,l'} At_{ll'} /((t+p_l)(t+p_l')) dt,
  p_l = l/b_l,  At_{ll'} = A_{ll'} (1+delta_{ll'}) / (b_l b_l').
Partial fractions (poles -p_l of order <= 3) give the closed form
    w_k = sum_i [ -c1_i log p_i + c2_i/p_i + c3_i/(2 p_i^2) ],   sum_i c1_i = 0.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
from scipy import integrate, special

EULER = 0.57721566490153286060651209008240243
FAMILIES = ("weibull", "empirical", "hazen", "gringorten", "digamma")


# ----------------------------------------------------------------------------
# positions
# ----------------------------------------------------------------------------
def harmonic(n: int) -> float:
    """H_n = psi(n+1) + gamma (exact for integers via digamma)."""
    return float(special.digamma(n + 1.0) + EULER)


def harmonic_mp(n: int):
    return mp.digamma(n + 1) + mp.euler


def positions(family: str, k: int, M: int | None = None) -> np.ndarray:
    """Depth-unit positions u_j = -log S_hat_j, j = 1..k.

    M=None gives the M -> infinity limit offsets u_j - log M."""
    j = np.arange(1, k + 1, dtype=float)
    if family in ("weibull", "empirical"):
        if M is None:
            return -np.log(j)
        return np.log((M + 1.0) / j) if family == "weibull" else np.log(M / j)
    if family == "hazen":
        return -np.log(j - 0.5) if M is None else np.log(M / (j - 0.5))
    if family == "gringorten":
        return -np.log(j - 0.44) if M is None else np.log((M + 0.12) / (j - 0.44))
    if family == "digamma":
        Hjm1 = special.digamma(j) + EULER  # H_{j-1}
        if M is None:
            return EULER - Hjm1  # H_M - H_{j-1} - log M -> gamma - H_{j-1}
        return harmonic(M) - Hjm1
    raise ValueError(family)


def readoff(family: str, N: int | None, M: int | None = None, R: float | None = None) -> float:
    """Own-family deployment position for deployment rank 1, in depth units.

    Finite: N given.  Limit (M=None, R given): read-off minus log M."""
    if N is not None:
        return {
            "weibull": math.log(N + 1.0),
            "empirical": math.log(N),
            "hazen": math.log(N / 0.5),
            "gringorten": math.log((N + 0.12) / 0.56),
            "digamma": harmonic(N),
        }[family]
    lr = math.log(R)
    return {
        "weibull": lr,
        "empirical": lr,
        "hazen": lr + math.log(2.0),
        "gringorten": lr - math.log(0.56),
        "digamma": lr + EULER,
    }[family]


def centered_mp(family: str, k: int):
    """Centered positions c_j = u_j - ubar as mpf (M-free for all five families)."""
    if family in ("weibull", "empirical"):
        u = [-mp.log(j) for j in range(1, k + 1)]
    elif family == "hazen":
        u = [-mp.log(mp.mpf(j) - mp.mpf("0.5")) for j in range(1, k + 1)]
    elif family == "gringorten":
        u = [-mp.log(mp.mpf(j) - mp.mpf("0.44")) for j in range(1, k + 1)]
    elif family == "digamma":
        u = [-harmonic_mp(j - 1) for j in range(1, k + 1)]
    else:
        raise ValueError(family)
    ub = mp.fsum(u) / k
    return [x - ub for x in u]


def centered_np(family: str, k: int) -> np.ndarray:
    u = positions(family, k, None)
    return u - u.mean()


# ----------------------------------------------------------------------------
# closed form for w_k = E[S_DD / S_uD]
# ----------------------------------------------------------------------------
def w_closed_mp(c, dps: int | None = None):
    """Exact w_k from centered positions c (list of mpf, strictly decreasing).

    Returns (w, sum of residues c1 [should be 0])."""
    k = len(c)
    n = k - 1
    old = mp.mp.dps
    if dps is not None:
        mp.mp.dps = dps
    try:
        c = [mp.mpf(x) for x in c]
        b = []
        acc = mp.mpf(0)
        for l in range(n):
            acc += c[l]
            b.append(acc)
        if any(x <= 0 for x in b):
            raise ValueError("partial sums b_l must be positive")
        L = [mp.mpf(l + 1) for l in range(n)]
        p = [L[l] / b[l] for l in range(n)]
        kk = mp.mpf(k)
        Add = [(L[l] - L[l] * L[l] / kk) for l in range(n)]  # A_ll
        Atd = [2 * Add[l] / (b[l] * b[l]) for l in range(n)]  # At_ll
        total = mp.mpf(0)
        c1sum = mp.mpf(0)
        for i in range(n):
            pi = p[i]
            d = [p[l] - pi for l in range(n)]
            R0 = mp.mpf(1)
            S1 = mp.mpf(0)
            S2 = mp.mpf(0)
            z = [mp.mpf(0)] * n  # z_l = 1/(b_l d_l), z_i = 0
            for l in range(n):
                if l == i:
                    continue
                R0 *= p[l] / d[l]
                inv = 1 / d[l]
                S1 += inv
                S2 += inv * inv
                z[l] = inv / b[l]
            R1 = -R0 * S1
            R2 = R0 * (S1 * S1 + S2) / 2
            # (A z)_i and (A z')_i with z'_l = z_l / d_l
            Az_i = mp.mpf(0)
            Azp_i = mp.mpf(0)
            for l in range(n):
                if l == i:
                    continue
                a_il = min(L[i], L[l]) - L[i] * L[l] / kk
                Az_i += a_il * z[l]
                Azp_i += a_il * z[l] / d[l]
            G0 = Az_i / b[i]
            G1 = -Azp_i / b[i]
            # Q0 = z^T A z + sum_l A_ll z_l^2   (z^T A z = centered SS of tail sums)
            Z = [mp.mpf(0)] * (n + 1)  # Z_j = sum_{l>=j} z_l, j = 1..k (Z_k = 0)
            for l in range(n - 1, -1, -1):
                Z[l] = Z[l + 1] + z[l]
            Zb = mp.fsum(Z) / kk
            Q0 = mp.fsum((x - Zb) ** 2 for x in Z) + mp.fsum(Add[l] * z[l] ** 2 for l in range(n))
            Ati = Atd[i]
            c3 = pi * Ati * R0
            c2 = pi * (Ati * R1 + 2 * R0 * G0)
            c1 = pi * (Ati * R2 + 2 * (R1 * G0 + R0 * G1) + R0 * Q0)
            total += -c1 * mp.log(pi) + c2 / pi + c3 / (2 * pi * pi)
            c1sum += c1
        return +total, +c1sum
    finally:
        mp.mp.dps = old


def w_closed(family: str, k: int, dps: int | None = None, check: bool = True):
    """Closed-form w_k at two precisions; returns (w as float, abs diff between precisions)."""
    if k == 2:
        c = centered_np(family, 2)
        return 1.0 / (c[0] - c[1]), 0.0
    d1 = dps or (40 + 2 * k)
    with mp.workdps(d1):
        w1, r1 = w_closed_mp(centered_mp(family, k))
    if not check:
        return float(w1), float("nan")
    with mp.workdps(d1 + 40):
        w2, r2 = w_closed_mp(centered_mp(family, k))
    return float(w2), float(abs(w2 - w1))


# ----------------------------------------------------------------------------
# quadrature (double precision) for E[W], E[W^2], E[L W]
# ----------------------------------------------------------------------------
class Quad:
    """Exponential-tilting integrands for moments of W = S_DD/S_uD."""

    def __init__(self, c: np.ndarray):
        c = np.asarray(c, dtype=float)
        self.k = k = c.size
        self.n = n = k - 1
        self.b = np.cumsum(c)[:-1]
        if np.any(self.b <= 0):
            raise ValueError("b_l must be positive")
        self.l = np.arange(1, k, dtype=float)
        lm = np.minimum.outer(self.l, self.l)
        self.A = lm - np.outer(self.l, self.l) / k
        self.Add = np.diag(self.A).copy()
        self.p = self.l / self.b

    def _g(self, t):
        g = 1.0 / (self.l + t * self.b)
        logphi = np.sum(np.log(self.l * g))
        return g, math.exp(logphi)

    def f1(self, t):  # integrand of E[W]
        g, phi = self._g(t)
        G = np.concatenate([np.cumsum(g[::-1])[::-1], [0.0]])
        q = np.sum((G - G.mean()) ** 2) + np.sum(self.Add * g * g)
        return phi * q

    def f2(self, t):  # integrand of E[W^2] (extra factor t)
        g, phi = self._g(t)
        A = self.A
        a = A @ g
        muAmu = g @ a
        trAS = np.sum(self.Add * g * g)
        g2 = g * g
        trASAS = np.sum((A * A) * np.outer(g2, g2))
        EQ2 = ((muAmu + trAS) ** 2 + 2.0 * trASAS + 6.0 * np.sum(self.Add ** 2 * g2 * g2)
               + 4.0 * np.sum(a * a * g2) + 8.0 * np.sum(a * self.Add * g2 * g))
        return t * phi * EQ2

    def fL(self, t):  # integrand of E[L W], L = (1/k) sum l V_l
        g, phi = self._g(t)
        A = self.A
        a = A @ g
        cvec = self.l / self.k
        muAmu = g @ a
        trAS = np.sum(self.Add * g * g)
        ELQ = (cvec @ g) * (muAmu + trAS) + 2.0 * np.sum(cvec * g * g * a) + 2.0 * np.sum(cvec * self.Add * g ** 3)
        return phi * ELQ

    def integrate(self, f):
        # break points at the pole scales
        pmin, pmax = float(np.min(self.p)), float(np.max(self.p))
        pts = np.concatenate([[0.0], np.geomspace(pmin / 20.0, pmax * 20.0, 30)])
        total = 0.0
        err = 0.0
        for lo, hi in zip(pts[:-1], pts[1:]):
            v, e = integrate.quad(f, lo, hi, limit=200, epsabs=0, epsrel=1e-13)
            total += v
            err += e
        v, e = integrate.quad(f, pts[-1], np.inf, limit=400, epsabs=0, epsrel=1e-13)
        return total + v, err + e

    def EW(self):
        return self.integrate(self.f1)[0]

    def EW2(self):
        return self.integrate(self.f2)[0]

    def ELW(self):
        return self.integrate(self.fL)[0]


def s_forward(family: str, k: int) -> float:
    """s = E[beta_f] = S_um / S_uu (exact; forward slope mean)."""
    c = centered_np(family, k)
    m = centered_np("digamma", k)
    return float(c @ m / (c @ c))


# ----------------------------------------------------------------------------
# exact means and variances of xi
# ----------------------------------------------------------------------------
def mbar_finite(k: int, M: int) -> float:
    j = np.arange(1, k + 1, dtype=float)
    return float(harmonic(M) - np.mean(special.digamma(j) + EULER))


def mean_xi_finite(family: str, k: int, M: int, N: int, w: float, ustar: float | None = None) -> dict:
    """E[xi] = mbar + (u* - ubar) w - H_N, exact at finite M (w = w_k(family))."""
    u = positions(family, k, M)
    ub = float(u.mean())
    if ustar is None:
        ustar = readoff(family, N)
    mb = mbar_finite(k, M)
    HN = harmonic(N)
    s = s_forward(family, k)
    return {
        "Exi": mb + (ustar - ub) * w - HN,
        "Exi_fwd": mb + (ustar - ub) * s - HN,
        "level": mb - ub,
        "spacing": (ustar - ub) * (s - 1.0),
        "eiv": (ustar - ub) * (w - s),
        "readgap": ustar - HN,
        "ustar_minus_ubar": ustar - ub,
        "ustar": ustar,
        "s": s,
        "w": w,
    }


def mean_xi_limit(family: str, k: int, R: float, w: float, rread: float | None = None) -> dict:
    """Limit (M -> inf, N = R M) E[xi] = E[Tbar] + (r_read - abar) w - log R - gamma."""
    a = positions(family, k, None)
    ab = float(a.mean())
    if rread is None:
        rread = readoff(family, None, None, R)
    j = np.arange(1, k + 1, dtype=float)
    ETbar = float(EULER - np.mean(special.digamma(j) + EULER))  # gamma - mean H_{j-1}
    s = s_forward(family, k)
    target = math.log(R) + EULER
    return {
        "Exi": ETbar + (rread - ab) * w - target,
        "Exi_fwd": ETbar + (rread - ab) * s - target,
        "level": ETbar - ab,
        "spacing": (rread - ab) * (s - 1.0),
        "eiv": (rread - ab) * (w - s),
        "readgap": rread - target,
        "s": s,
        "w": w,
    }


def var_xi_finite(family: str, k: int, M: int | None, N: int | None, ustar_minus_ubar: float,
                  EW: float, EW2: float, ELW: float) -> float:
    """Var(xi) = Var D_(k) + Var Dmax_N + Var(L + c W), c = u* - ubar (exact)."""
    c = ustar_minus_ubar
    varL = (k - 1) / k ** 2
    EL = (k - 1) / k
    covLW = ELW - EL * EW
    varW = EW2 - EW ** 2
    vDk = float(special.polygamma(1, k)) - (float(special.polygamma(1, M + 1)) if M else 0.0)
    vDmax = math.pi ** 2 / 6 - (float(special.polygamma(1, N + 1)) if N else 0.0)
    return vDk + vDmax + varL + 2 * c * covLW + c * c * varW
