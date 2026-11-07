"""Independent verification library for the inverse-forecaster rank-term claims.

An independent re-implementation: it does not import or copy any script from ../inverse.

Conventions
-----------
Fit depths D_(1) >= ... >= D_(M): order statistics of M iid Exp(1).
Positions u_1..u_k (depth units, u_j = -log S_hat_j).
Inverse forecast in depth units: Dbar + (y - ubar) * W, W = S_DD / S_uD.
xi = forecast - D^N, D^N = max of N iid Exp(1), E[D^N] = H_N.
"""
import math
import numpy as np
from scipy import integrate
import mpmath as mp

EULER = 0.57721566490153286060651209008240243


# ----------------------------------------------------------------------------- basics
def H(n):
    """Harmonic number H_n (exact float sum for small n, asymptotic for large)."""
    n = int(n)
    if n <= 0:
        return 0.0
    if n < 100000:
        return math.fsum(1.0 / i for i in range(1, n + 1))
    x = float(n)
    return math.log(x) + EULER + 1 / (2 * x) - 1 / (12 * x * x) + 1 / (120 * x ** 4)


def Hvec(k):
    """array [H_0, H_1, ..., H_{k-1}] (index j-1 for rank j)."""
    out = np.zeros(k)
    s = 0.0
    for i in range(1, k):
        s += 1.0 / i
        out[i] = s
    return out


def positions(family, k, M=None):
    """Full positions u_1..u_k at fit size M; if M is None, the M -> inf offsets from log M."""
    j = np.arange(1, k + 1, dtype=float)
    if family == 'weibull':
        return (np.log(M + 1) if M is not None else 0.0) - np.log(j)
    if family == 'empirical':
        return (np.log(M) if M is not None else 0.0) - np.log(j)
    if family == 'hazen':
        return (np.log(M) if M is not None else 0.0) - np.log(j - 0.5)
    if family == 'gringorten':
        return (np.log(M + 0.12) if M is not None else 0.0) - np.log(j - 0.44)
    if family == 'digamma':
        if M is not None:
            return H(M) - Hvec(k)
        return EULER - Hvec(k)   # lim (H_M - log M) - H_{j-1}
    raise ValueError(family)


def readoff(name, family, N, M=None):
    """Read-off depth. If M is None, return the offset from log M in the limit with R = N (N is R)."""
    if M is None:
        R = N
        lr = math.log(R)
        if name in ('logN', 'logN1'):
            return lr
        if name == 'own':
            return {'weibull': lr, 'empirical': lr, 'hazen': lr + math.log(2),
                    'gringorten': lr - math.log(0.56), 'digamma': lr + EULER}[family]
        raise ValueError(name)
    if name == 'logN':
        return math.log(N)
    if name == 'logN1':
        return math.log(N + 1)
    if name == 'own':
        return {'weibull': math.log(N + 1), 'empirical': math.log(N), 'hazen': math.log(2 * N),
                'gringorten': math.log((N + 0.12) / 0.56), 'digamma': H(N)}[family]
    raise ValueError(name)


def centered(u):
    u = np.asarray(u, dtype=float)
    return u - u.mean()


def bvec(u):
    c = centered(u)
    return np.cumsum(c)[:-1]


# ----------------------------------------------------------------------------- w_k by the Laplace integral (double)
def _integrand_w(t, b, l, k):
    rate = l + t * b
    g = 1.0 / rate
    logP = np.sum(np.log(l) - np.log(rate))
    tail = np.cumsum(g[::-1])[::-1]
    gAg = np.dot(tail, tail) - (np.dot(l, g)) ** 2 / k
    Adiag = l - l * l / k
    return math.exp(logP) * (gAg + np.dot(Adiag, g * g))


def w_laplace(u, epsrel=1e-13):
    """E[S_DD/S_uD] via int_0^inf E[S_DD e^{-t S_uD}] dt, quad in x = log t."""
    u = np.asarray(u, dtype=float)
    k = len(u)
    b = bvec(u)
    if np.any(b <= 0):
        raise ValueError('need b_l > 0')
    l = np.arange(1, k, dtype=float)
    f = lambda x: math.exp(x) * _integrand_w(math.exp(x), b, l, k)
    # integrand in x: peak near x ~ log(p) values; integrate over a wide range in pieces
    p = l / b
    lo = math.log(p.min()) - 40.0
    hi = math.log(p.max()) + 40.0 / max(1, k - 1) + 10.0
    pts = np.linspace(lo, hi, 25)
    tot = 0.0
    for a, c in zip(pts[:-1], pts[1:]):
        val, err = integrate.quad(f, a, c, epsabs=0, epsrel=epsrel, limit=400)
        tot += val
    return tot


def w_laplace_inf(u):
    """Same integral, but quad directly on t in [0, inf) (a second double-precision scheme)."""
    u = np.asarray(u, dtype=float)
    k = len(u)
    b = bvec(u)
    l = np.arange(1, k, dtype=float)
    f = lambda t: _integrand_w(t, b, l, k)
    p = l / b
    brk = [0.0] + sorted(set(np.round(np.geomspace(p.min() / 10, p.max() * 10, 12), 12)))
    tot = 0.0
    for a, c in zip(brk[:-1], brk[1:]):
        tot += integrate.quad(f, a, c, epsabs=0, epsrel=1e-13, limit=400)[0]
    tot += integrate.quad(f, brk[-1], np.inf, epsabs=0, epsrel=1e-13, limit=400)[0]
    return tot


# ----------------------------------------------------------------------------- mpmath high-precision Laplace
def w_laplace_mp(u, dps=30):
    with mp.workdps(dps):
        u = [mp.mpf(x) for x in u]
        k = len(u)
        ub = mp.fsum(u) / k
        c = [x - ub for x in u]
        b = []
        s = mp.mpf(0)
        for x in c[:-1]:
            s += x
            b.append(s)
        l = [mp.mpf(i) for i in range(1, k)]

        def f(t):
            g = [1 / (l[i] + t * b[i]) for i in range(k - 1)]
            P = mp.mpf(1)
            for i in range(k - 1):
                P *= l[i] * g[i]
            tail = [mp.mpf(0)] * (k - 1)
            acc = mp.mpf(0)
            for i in range(k - 2, -1, -1):
                acc += g[i]
                tail[i] = acc
            gAg = mp.fsum(x * x for x in tail) - mp.fsum(l[i] * g[i] for i in range(k - 1)) ** 2 / k
            diag = mp.fsum((l[i] - l[i] ** 2 / k) * g[i] ** 2 for i in range(k - 1))
            return P * (gAg + diag)

        p = sorted([l[i] / b[i] for i in range(k - 1)])
        pts = [mp.mpf(0)] + [x for x in p] + [mp.inf]
        return mp.quad(f, pts)


# ----------------------------------------------------------------------------- second moment (Laplace, double)
def _integrand_w2(t, b, l, k, Afull):
    rate = l + t * b
    g = 1.0 / rate
    logP = np.sum(np.log(l) - np.log(rate))
    Ag = Afull @ g
    Ad = np.diag(Afull)
    EQ = g @ Ag + np.dot(Ad, g * g)
    G2 = np.outer(g * g, g * g)
    EQ2 = EQ ** 2 + 2 * np.sum(Afull ** 2 * G2) + 6 * np.dot(Ad ** 2, g ** 4) \
        + 4 * np.dot(Ag ** 2, g ** 2) + 8 * np.dot(Ag * Ad, g ** 3)
    return t * math.exp(logP) * EQ2


def w2_laplace(u):
    u = np.asarray(u, dtype=float)
    k = len(u)
    b = bvec(u)
    l = np.arange(1, k, dtype=float)
    L = l[:, None]
    Lp = l[None, :]
    Afull = np.minimum(L, Lp) - L * Lp / k
    f = lambda x: math.exp(x) * _integrand_w2(math.exp(x), b, l, k, Afull)
    p = l / b
    lo = math.log(p.min()) - 40.0
    hi = math.log(p.max()) + 15.0
    pts = np.linspace(lo, hi, 25)
    return sum(integrate.quad(f, a, c, epsabs=0, epsrel=1e-13, limit=400)[0] for a, c in zip(pts[:-1], pts[1:]))


# ----------------------------------------------------------------------------- Dirichlet (simplex) representation
def W_from_V(V, u):
    """W = S_DD/S_uD computed directly from spacings V (shape (..., k-1)) by building the depths."""
    V = np.asarray(V, dtype=float)
    k = V.shape[-1] + 1
    # D_j - D_k = sum_{l=j}^{k-1} V_l
    D = np.concatenate([np.cumsum(V[..., ::-1], axis=-1)[..., ::-1], np.zeros(V.shape[:-1] + (1,))], axis=-1)
    Dc = D - D.mean(axis=-1, keepdims=True)
    c = centered(u)
    return (Dc * Dc).sum(-1) / (Dc @ c)


def w_dirichlet_k3(u):
    """w_3 = 2 int_0^1 W(V1 = x, V2 = (1-x)/2) dx (E = G*Delta, Delta ~ U(simplex), W 1-homogeneous)."""
    f = lambda x: W_from_V(np.array([x, (1 - x) / 2.0]), u)
    return 2.0 * integrate.quad(f, 0, 1, epsabs=0, epsrel=1e-13, limit=200)[0]


def w_dirichlet_k4(u):
    """w_4 = (k-1) E[W(Delta)] = 3 * 2 * int over triangle of W(V = (x, y/2, (1-x-y)/3))."""
    f = lambda y, x: W_from_V(np.array([x, y / 2.0, (1 - x - y) / 3.0]), u)
    val = integrate.dblquad(f, 0, 1, lambda x: 0.0, lambda x: 1 - x, epsabs=1e-14, epsrel=1e-13)[0]
    return 6.0 * val


# ----------------------------------------------------------------------------- deterministic constants
def s_forward(u):
    """s_k = S_um / S_uu with m_j = -H_{j-1} (centered mean depths)."""
    u = np.asarray(u, dtype=float)
    k = len(u)
    m = -Hvec(k)
    c = centered(u)
    return np.dot(c, m - m.mean()) / np.dot(c, c)


def mbar(k, M):
    return H(M) - H(k) + 1.0


def exact_bias(u_full, w, k, M, N, ystar):
    """E[xi] = mbar + (y - ubar) w - H_N (u_full = finite-M positions, length k)."""
    ub = float(np.mean(u_full))
    return mbar(k, M) + (ystar - ub) * w - H(N)


def limit_bias(a, w, k, rstar, R):
    """limit: mbar_inf = gamma - (H_k - 1); target mean log R + gamma; a = offsets."""
    return (EULER - (H(k) - 1)) + (rstar - float(np.mean(a))) * w - (math.log(R) + EULER)
