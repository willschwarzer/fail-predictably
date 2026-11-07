"""The error floor of a straight tail, from rank combinatorics alone.

Setting. The paper's forecaster takes the fit set's top-k log-probs, regresses
y_j = log(j / (M + 1)) on psi_j = -log(-v_j) (inverse OLS), solves for the psi where the line
reaches log(1 / N), and predicts the deploy set's maximum log-prob from it. Its error is read in
decades (log10 p), averaged over random fit/deploy splits of a pool of U = M + N prompts.

Reduction. Call a pool's top straight when its top order statistics lie on a line in
(psi, log rank). In log-prob that line is v_r = -|v_1| r^(1/lambda) for pool rank r, with
lambda the line's rate (psi per unit of log rank) and v_1 its top value. Inverse OLS is
equivariant under affine maps of psi, so on every split the forecast lands at a continuous pool
rank r* that depends only on which pool ranks fell in the fit set, and the error is

    pred - actual = |v_1| (J^(1/lambda) - r*^(1/lambda))  nats,

with J the deploy set's top pool rank. The split-averaged MSE in decades is therefore

    MSE = (|v_1| / ln 10)^2 F(M, N, k; lambda),   F = E[(J^(1/lambda) - r*^(1/lambda))^2],

and to first order in 1/lambda

    MSE ~= (s / ln 10)^2 C(M, N, k),   C = E[(log r* - log J)^2],   s = |v_1| / lambda,

where s is the line's spread at the top: nats of log p per unit of log rank. The geometry
enters only through C, the tail only through its spread and its depth (lambda = |v_1| / s).

Two versions:
  A, realized pool: the pool's own order statistics lie on the line and only the partition is
     random. The certificate is this case exactly, since it replaces the pool's top L ranks by
     such a line; so A predicts the certified floor with no free parameter.
  B, population: the score distribution has a straight tail and each pool is a fresh sample
     from it. Disjoint random subsets of an i.i.d. sample are independent samples, so the
     fit and deploy sets are independent, and the deploy maximum adds its own Gumbel noise.
     This is the case of a model scored on new prompts.

Closed form (Renyi / Gamma approximation, large k). With Gamma_j the j-th arrival of a unit
Poisson process, the fit set's j-th pool rank is about Gamma_j U / M, and
Cov(log Gamma_i, log Gamma_j) = trigamma(max(i, j)) exactly. Linearizing the OLS slope around
-1 gives log r* - log(U (M+1) / (M N)) ~= sum_j a_j eta_j, eta_j = log(Gamma_j / j),
a_j = (1 - (Dy / Vu) u~_j) / k, with u~_j the centered log j, Vu their variance and
Dy = mean_j log j + log(N / (M + 1)) the extrapolation distance in log rank. Then

    C ~= a' T a + c_dep + bias^2,   a' T a -> 2 (1 + Dy + Dy^2) / k  for large k,

with c_dep = Var(log J) for A (J geometric; about (log 2)^2 / R) and pi^2 / 6 for B, where the
deploy maximum's log-rank also has mean log(1 + 1/R) - gamma, so B carries a bias near
gamma = 0.577 that A does not.

  python floor_theory.py [--certs cert_scale.csv] [--draws 20000]
"""
import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import digamma, polygamma

LN10 = math.log(10.0)
EULER = 0.5772156649015329
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
READOUTS = RESULTS_ROOT / "lquant_dynamics/readouts"


# ---------------------------------------------------------------- rank simulators (lambda = 1)

def sim_partition_exact(M, N, k, e, D, rng, chunk=2000):
    """Pool ranks of the fit set's top k and the deploy set's top e, for D uniformly random
    fit subsets of size M out of U = M + N ranks (sequential hypergeometric draws, exact)."""
    U = M + N
    Rf = np.zeros((D, k), dtype=np.int64)
    Rd = np.zeros((D, e), dtype=np.int64)
    for s in range(0, D, chunk):
        d = min(chunk, D - s)
        rf, rd = Rf[s:s + d], Rd[s:s + d]
        cf = np.zeros(d, dtype=np.int64)
        cd = np.zeros(d, dtype=np.int64)
        r = 0
        while ((cf < k) | (cd < e)).any():
            r += 1
            fit = rng.random(d) < (M - cf) / (U - r + 1)
            m = fit & (cf < k)
            rf[m, cf[m]] = r
            m = ~fit & (cd < e)
            rd[m, cd[m]] = r
            cf += fit
            cd += ~fit
    return Rf.astype(np.float64), Rd.astype(np.float64)


def sim_partition_bernoulli(M, N, k, e, D, rng):
    """Same, with each rank in the fit set independently w.p. M/U (accurate for k << M);
    used where the exact walk is too long."""
    q = M / (M + N)
    kk = k + 20
    Rf = np.cumsum(rng.geometric(q, size=(D, kk)), axis=1)
    w = Rf - np.arange(1, kk + 1)  # non-fit ranks below each fit rank; non-decreasing
    i = np.arange(1, e + 1)
    Rd = np.stack([i + np.searchsorted(w[d], i, side="left") for d in range(D)])
    return Rf[:, :k].astype(np.float64), Rd.astype(np.float64)


def sim_population(M, N, k, e, D, rng):
    """Continuous pool-rank equivalents r = U * survival of the fit top-k (M i.i.d. draws) and
    deploy top-e (N i.i.d. draws, independent of the fit) from a straight population tail."""
    U = M + N

    def top(n, m):
        lo = -np.log(rng.beta(m + 1, n - m, size=D))  # the (m+1)-th largest of n Exp(1)
        z = rng.exponential(size=(D, m)) / np.arange(1, m + 1)
        E = lo[:, None] + np.cumsum(z[:, ::-1], axis=1)[:, ::-1]  # E_(j) = E_(m+1) + sum_{i>=j} z_i / i
        return U * np.exp(-E)

    return top(M, k), top(N, e)


def forecast_logrank(Rf, M, N):
    """log r*: the inverse-OLS forecast of the deploy maximum, as a pool log-rank (lambda = 1)."""
    k = Rf.shape[1]
    x = -np.log(Rf)
    y = np.log(np.arange(1, k + 1) / (M + 1.0))
    xm = x.mean(axis=1, keepdims=True)
    sxx = ((x - xm) ** 2).sum(axis=1)
    sxy = ((x - xm) * (y - y.mean())).sum(axis=1)
    b = sxy / sxx
    a = y.mean() - b * xm[:, 0]
    return -(math.log(1.0 / N) - a) / b


class Floor:
    """Simulated (log r*, deploy ranks, fit ranks) for one geometry; F, C and sigma at any lambda."""

    def __init__(self, M, N, k, e, D, rng, mode):
        self.M, self.N, self.k, self.e, self.mode = M, N, k, e, mode
        if mode == "A":
            U = M + N
            walk = k * U / M * (1 + 4 / math.sqrt(k)) + e
            sim = sim_partition_exact if walk <= 4e4 else sim_partition_bernoulli
            self.sim = sim.__name__
            Rf, Rd = sim(M, N, k, e, D, rng)
        else:
            self.sim = "sim_population"
            Rf, Rd = sim_population(M, N, k, e, D, rng)
        self.lr = forecast_logrank(Rf, M, N)
        self.lj = np.log(Rd[:, 0])
        self.linv = np.log(np.concatenate([Rf, Rd], axis=1))  # involved set, log pool rank
        d = self.lr - self.lj
        self.C, self.bias, self.sd = float((d ** 2).mean()), float(d.mean()), float(d.std())
        self.Dlog = float(self.linv.std(axis=1).mean())

    def F(self, lam):
        d = np.exp(self.lj / lam) - np.exp(self.lr / lam)
        return float((d ** 2).mean()), float(np.abs(d).mean())

    def sigma(self, lam):
        return float(np.exp(self.linv / lam).std(axis=1).mean())


def closed_form(M, N, k, mode):
    j = np.arange(1, k + 1, dtype=np.float64)
    u = np.log(j)
    ut = u - u.mean()
    Vu = float((ut ** 2).mean())
    Dy = float(u.mean() + math.log(N / (M + 1.0)))
    a = (1.0 - (Dy / Vu) * ut) / k
    # a' T a with T_ij = trigamma(max(i, j)), in O(k): the pairs with max(i, j) = m contribute
    # trigamma(m) (A_m^2 - A_{m-1}^2), A_m the partial sums of a.
    A = np.cumsum(a)
    var_fit = float((polygamma(1, j) * (A ** 2 - np.concatenate([[0.0], A[:-1]]) ** 2)).sum())
    mean_fit = float(a @ (digamma(j) - u))
    R = N / M
    if mode == "A":
        q = M / (M + N)
        jj = np.arange(1, 200)
        pj = q ** (jj - 1) * (1 - q)
        mJ = float((pj * np.log(jj)).sum())
        vJ = float((pj * np.log(jj) ** 2).sum() - mJ ** 2)
        mean = math.log((1 + 1 / R) * (1 + 1 / M)) + mean_fit - mJ
        return var_fit + vJ + mean ** 2, var_fit, vJ, mean, Dy
    mean = math.log(1 + 1 / M) + mean_fit + EULER
    vdep = math.pi ** 2 / 6
    return var_fit + vdep + mean ** 2, var_fit, vdep, mean, Dy


def e_of(M, N):
    from forecast_experiment.ols_utils import count_extrapolated_eval_points
    return count_extrapolated_eval_points(M, N, plotting="weibull")


# ---------------------------------------------------------------- reports

def check_certs(path, draws, rng):
    """Predict every certified pool's floor from its line (|v_1|, lambda) and the geometry alone."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    geoms = sorted({(int(r["M"]), int(r["N"]), int(r["k"]), int(r["e"])) for r in rows})
    print("== A: certified floor predicted from (M, N, k) and each pool's certified line, no free parameter")
    print("   cell                      pools   cert mse   theory F   linear C    ratio   B/A at the same line   "
          "C_A    C_B   lambda   spread")
    out = {}
    for M, N, k, e in geoms:
        fa = Floor(M, N, k, e, draws, rng, "A")
        fb = Floor(M, N, k, e, draws, rng, "B")
        rs = [r for r in rows if (int(r["M"]), int(r["N"]), int(r["k"])) == (M, N, k)]
        for src in sorted({r["source"] for r in rs}):
            sr = [r for r in rs if r["source"] == src]
            emp, th, lin, thb = [], [], [], []
            for r in sr:
                lam, v1 = float(r["cert_lambda"]), -float(r["cert_v1"])
                emp.append(float(r["cert_mse"]))
                th.append(v1 ** 2 * fa.F(lam)[0] / LN10 ** 2)
                lin.append((v1 / lam) ** 2 * fa.C / LN10 ** 2)
                thb.append(v1 ** 2 * fb.F(lam)[0] / LN10 ** 2)
            emp, th, lin, thb = map(np.mean, (emp, th, lin, thb))
            lam = np.mean([float(r["cert_lambda"]) for r in sr])
            spr = np.mean([float(r["cert_spread_top"]) for r in sr])
            print(f"   M{M:<5d} N{N:<6d} k{k:<5d} {src:6s} {len(sr):4d}   {emp:8.4f}   {th:8.4f}   {lin:8.4f}   "
                  f"{emp / th:6.3f}   {thb / th:6.2f}x                 {fa.C:6.3f} {fb.C:6.3f}  {lam:6.2f}  {spr:6.3f}")
            out[(M, N, k, src)] = (emp, th, lin, thb)
    return out


def closed_form_check(draws, rng):
    print("\n== C(M, N, k): simulated vs closed form (A: realized pool; B: population)")
    print("      M        N      k    sim     C_A (sim / closed)    C_B (sim / closed)   fit var (closed)   Dy")
    for M, R, k in [(250, 8.2, 10), (250, 8.2, 50), (1000, 8.2, 200), (1000, 30, 50), (4000, 8.2, 1000),
                    (250, 100, 10), (1000, 70, 200)]:
        N = int(round(R * M))
        e = e_of(M, N)
        fa = Floor(M, N, k, e, draws, rng, "A")
        fb = Floor(M, N, k, e, draws, rng, "B")
        ca, va, _, _, Dy = closed_form(M, N, k, "A")
        cb = closed_form(M, N, k, "B")[0]
        print(f"   {M:6d} {N:8d} {k:6d}  {fa.sim[15:]:9s}  {fa.C:7.3f} / {ca:7.3f}      {fb.C:7.3f} / {cb:7.3f}"
              f"        {va:7.3f}        {Dy:5.2f}")


def scale_table(draws, rng):
    print("\n== The straight-tail floor at larger geometry, in units of (s / ln 10)^2, s = the line's spread at the top")
    print("   (C_A: realized pool, the certificate's setting; C_B: population, a model on new prompts;"
          " closed form in brackets)")
    Ms = [250, 1000, 4000, 16000, 100000]
    Rs = [8.2, 30, 100, 1000]
    ks = [10, 50, 200, 1000, 5000]
    print("        M       R      k      C_A [closed]       C_B [closed]")
    for M in Ms:
        for R in Rs:
            for k in ks:
                if k > M // 4:
                    continue
                N = int(round(R * M))
                e = min(e_of(M, N), 50)
                d = max(2000, draws // 4) if k >= 1000 else draws
                fa = Floor(M, N, k, e, d, rng, "A")
                fb = Floor(M, N, k, e, d, rng, "B")
                ca = closed_form(M, N, k, "A")[0] if k <= 5000 else float("nan")
                cb = closed_form(M, N, k, "B")[0] if k <= 5000 else float("nan")
                print(f"   {M:6d} {R:7g} {k:6d}   {fa.C:7.3f} [{ca:7.3f}]   {fb.C:7.3f} [{cb:7.3f}]"
                      f"   ({fa.sim[15:]})", flush=True)


def bernoulli_check(draws, rng):
    """The Bernoulli approximation against the exact walk, where both are affordable."""
    print("\n== Bernoulli membership vs the exact hypergeometric walk, C_A")
    print("        M       R      k    k/M     exact   Bernoulli   ratio")
    for M, R, k in [(250, 8.2, 50), (250, 100, 50), (1000, 8.2, 200), (1000, 30, 200), (4000, 8.2, 1000),
                    (4000, 30, 1000), (1000, 100, 200)]:
        N = int(round(R * M))
        e = min(e_of(M, N), 50)
        ex = sim_partition_exact(M, N, k, e, draws, rng)
        be = sim_partition_bernoulli(M, N, k, e, draws, rng)
        c = []
        for Rf, Rd in (ex, be):
            d = forecast_logrank(Rf, M, N) - np.log(Rd[:, 0])
            c.append(float((d ** 2).mean()))
        print(f"   {M:6d} {R:7g} {k:6d}  {k / M:5.3f}   {c[0]:7.3f}   {c[1]:7.3f}    {c[1] / c[0]:6.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certs", default=str(READOUTS / "cert_scale.csv"))
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--skip-scale", action="store_true")
    ap.add_argument("--bernoulli-check", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(20261104)
    if a.bernoulli_check:
        bernoulli_check(a.draws, rng)
        return
    if Path(a.certs).exists():
        check_certs(a.certs, a.draws, rng)
    closed_form_check(a.draws, rng)
    if not a.skip_scale:
        scale_table(a.draws, rng)


if __name__ == "__main__":
    main()
