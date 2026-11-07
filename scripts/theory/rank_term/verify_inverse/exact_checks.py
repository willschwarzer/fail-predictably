"""Exact (quadrature / closed-form) checks of the inverse-forecaster rank-term results and the appendix numbers.

Independent re-implementation (vlib.py). Writes exact_checks.json and prints a log.
"""
import json, math, time, sys
import numpy as np
from scipy import optimize
import mpmath as mp
from vlib import *

t0 = time.time()
out = {}
FAMS = ['weibull', 'hazen', 'gringorten', 'digamma']

# ---------------------------------------------------------------- 1. w_k table, three engines + Dirichlet
print('== 1. w_k table ==', flush=True)
tab = {}
for fam in FAMS:
    for k in [2, 3, 4, 5, 10, 20, 50]:
        u = positions(fam, k)
        a = w_laplace(u)
        b = w_laplace_inf(u)
        c = float(w_laplace_mp(u, dps=30)) if k <= 50 else float('nan')
        d = w_dirichlet_k3(u) if k == 3 else (w_dirichlet_k4(u) if k == 4 else float('nan'))
        s = s_forward(u)
        tab[f'{fam}_{k}'] = dict(w_log=a, w_inf=b, w_mp=c, w_dir=d, s=s)
        print(f'{fam:10s} k={k:3d} w_log={a:.15f} w_inf={b:.15f} w_mp={c:.15f} w_dir={d:.15f} s={s:.9f}', flush=True)
out['w_table'] = tab

# E[W^2] spot values
print('== E[W^2] ==', flush=True)
w2 = {}
for fam in FAMS:
    for k in [3, 10, 20, 100]:
        u = positions(fam, k)
        m2 = w2_laplace(u)
        w = w_laplace(u)
        w2[f'{fam}_{k}'] = dict(EW2=m2, sd=math.sqrt(m2 - w * w))
        print(f'{fam:10s} k={k:3d} E[W^2]={m2:.9f} sd(W)={math.sqrt(m2 - w*w):.6f}', flush=True)
out['EW2'] = w2

# ---------------------------------------------------------------- 2. exact biases at the paper cells
print('== 2. exact biases at paper cells ==', flush=True)
cells = [(10, 44, 891), (10, 143, 2857), (10, 250, 2050)]
bias = {}
wk = {fam: w_laplace(positions(fam, 10)) for fam in FAMS + ['empirical']}
sk = {fam: s_forward(positions(fam, 10)) for fam in FAMS + ['empirical']}
for (k, M, N) in cells:
    for fam in ['weibull', 'empirical', 'hazen', 'gringorten', 'digamma']:
        for ro in ['own', 'logN', 'logN1']:
            uf = positions(fam, k, M)
            y = readoff(ro, fam, N, M)
            e = exact_bias(uf, wk[fam], k, M, N, y)
            bias[f'{fam}_{ro}_{k}_{M}_{N}'] = e
            print(f'({k},{M},{N}) {fam:10s} {ro:5s} y={y:.6f} E[xi]={e:+.5f}', flush=True)
out['bias_cells'] = bias

# four-way split
print('== four-way split ==', flush=True)
split = {}
for (k, M, N) in cells:
    for ro in ['logN1', 'logN']:
        uf = positions('weibull', k, M)
        y = readoff(ro, 'weibull', N, M)
        ub = uf.mean()
        lev = mbar(k, M) - ub
        spa = (y - ub) * (sk['weibull'] - 1)
        inv = (y - ub) * (wk['weibull'] - sk['weibull'])
        gap = y - H(N)
        split[f'{k}_{M}_{N}_{ro}'] = dict(level=lev, spacing=spa, inversion=inv, readoff=gap,
                                          total=lev + spa + inv + gap, forward=lev + spa + gap)
        print(f'({k},{M},{N}) {ro:5s} level={lev:+.6f} spacing={spa:+.6f} inversion={inv:+.6f} gap={gap:+.6f} '
              f'total={lev+spa+inv+gap:+.6f} forward={lev+spa+gap:+.6f}', flush=True)
# limit k=10, R=20
a = positions('weibull', 10)
lev = (EULER - (H(10) - 1)) - a.mean()
yy = math.log(20)
spa = (yy - a.mean()) * (sk['weibull'] - 1)
inv = (yy - a.mean()) * (wk['weibull'] - sk['weibull'])
gap = -EULER
split['limit_10_R20'] = dict(level=lev, spacing=spa, inversion=inv, readoff=gap, total=lev + spa + inv + gap)
print(f'limit k=10 R=20 level={lev:+.6f} spacing={spa:+.6f} inversion={inv:+.6f} gap={gap:+.6f} total={lev+spa+inv+gap:+.6f}')
out['split'] = split

# ---------------------------------------------------------------- 3. limit at k=10 (Weibull) and the 49-cell grids
print('== 3. limit k=10 Weibull, grids ==', flush=True)
w10 = wk['weibull']
beta10 = w10 * math.lgamma(11) / 10 - (H(10) - 1)
Rs = [2, 5, 10, 20, 100, 300, 1000]
row = [(w10 - 1) * math.log(R) + beta10 for R in Rs]
row2 = [limit_bias(positions('weibull', 10), w10, 10, math.log(R), R) for R in Rs]
print('w10-1 =', w10 - 1, 'beta10 =', beta10)
print('row (formula) ', ['%.4f' % x for x in row])
print('row (generic) ', ['%.4f' % x for x in row2])
out['limit_k10'] = dict(slope=w10 - 1, intercept=beta10, values=row, values_generic=row2)

ks = [5, 7, 10, 15, 20, 30, 50]
grids = {}
wcache = {}
for fam in FAMS:
    for k in ks:
        wcache[(fam, k)] = w_laplace(positions(fam, k))
for fam in FAMS:
    for ro in ['logN', 'own']:
        vals = []
        for k in ks:
            a = positions(fam, k)
            for R in Rs:
                r = readoff(ro, fam, R, None)
                vals.append((k, R, limit_bias(a, wcache[(fam, k)], k, r, R)))
        v = np.array([x[2] for x in vals])
        imin, imax = int(np.argmin(v)), int(np.argmax(v))
        grids[f'{fam}_{ro}'] = dict(min=float(v.min()), argmin=vals[imin][:2], max=float(v.max()),
                                    argmax=vals[imax][:2], n_neg=int((v < 0).sum()), n_pos=int((v > 0).sum()))
        print(f'grid {fam:10s} {ro:4s} min={v.min():+.4f} at {vals[imin][:2]} max={v.max():+.4f} at {vals[imax][:2]} '
              f'neg={int((v<0).sum())}/49', flush=True)
out['grids'] = grids

# Gringorten offsets vs mean depths
j = np.arange(1, 201)
gdev = (EULER - Hvec(200)) - (-np.log(j - 0.44))
print('max |Gringorten offset - mean depth| over j<=200 =', np.abs(gdev).max(), 'at j=', int(np.argmax(np.abs(gdev)) + 1))
out['gringorten_maxdev'] = float(np.abs(gdev).max())

# ---------------------------------------------------------------- 4. zero contour, beta_k, monotonicity up to k=1000
print('== 4. zero contour (Weibull limit) ==', flush=True)
KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
wW = {}
for k in range(2, KMAX + 1):
    wW[k] = w_laplace(positions('weibull', k))
    if k % 200 == 0:
        print('  k', k, 'w', wW[k], 'elapsed', time.time() - t0, flush=True)
beta = {k: wW[k] * math.lgamma(k + 1) / k - (H(k) - 1) for k in wW}
R0 = {k: (math.exp(-beta[k] / (wW[k] - 1))) for k in wW}
for k in [2, 3, 5, 10, 17, 18, 20, 30, 36, 37, 50, 78, 79, 100, 141, 142, 200, 500, 1000]:
    if k in wW:
        print(f'k={k:5d} w={wW[k]:.9f} beta={beta[k]:+.6f} R0={R0[k]:.6g}', flush=True)
ks_all = sorted(wW)
dec_w = all(wW[k + 1] < wW[k] for k in ks_all if k >= 3 and k + 1 in wW)
dec_b = all(beta[k + 1] < beta[k] for k in ks_all if k >= 5 and k + 1 in wW)
first_fail_w = [k for k in ks_all if k >= 3 and k + 1 in wW and not wW[k + 1] < wW[k]]
first_fail_b = [k for k in ks_all if k >= 5 and k + 1 in wW and not beta[k + 1] < beta[k]]
pos_beta = [k for k in ks_all if beta[k] > 0]
neg_beta_low = [k for k in ks_all if k <= 17 and beta[k] < 0]
k1000 = [k for k in ks_all if k <= 1000]
gaps = sorted((abs(wW[a] - wW[b]), a, b) for a in k1000 for b in k1000 if a < b and (abs(a - b) <= 3 or a == 2))
print('w decreasing for 3<=k<=KMAX:', dec_w, first_fail_w[:5])
print('beta decreasing for 5<=k<=KMAX:', dec_b, first_fail_b[:5])
print('k with beta>0:', pos_beta[:30], ' max:', max(pos_beta) if pos_beta else None)
print('smallest gap |w_a-w_b| over 2<=a<b<=1000 (|a-b|<=3 or a=2):', gaps[0])
# R0 increasing for k>=7?
r0inc = [k for k in ks_all if k >= 7 and k + 1 in wW and not R0[k + 1] > R0[k]]
print('R0 not increasing at k>=7:', r0inc[:5])
# k0(R): smallest k with bias < 0 at R
k0 = {}
for R in [2, 3, 5, 10, 20, 100, 1000, 1e4]:
    neg = [k for k in ks_all if (wW[k] - 1) * math.log(R) + beta[k] < 0]
    k0[str(R)] = min(neg) if neg else None
    # check upper set
    if neg:
        kk = min(neg)
        allneg = all((wW[k] - 1) * math.log(R) + beta[k] < 0 for k in ks_all if k >= kk)
    else:
        allneg = None
    print(f'R={R:g}: k0={k0[str(R)]}, negative for all k>=k0 up to KMAX: {allneg}', flush=True)
# 'positive for every R>=2 iff k<=36'
pos_R2 = [k for k in ks_all if (wW[k] - 1) * math.log(2) + beta[k] > 0]
print('k with bias>0 at R=2: max', max(pos_R2), 'contiguous from 2:', pos_R2 == list(range(2, max(pos_R2) + 1)))
# (w_k - 1)(log k - 1) trend
for k in [100, 1000, KMAX]:
    if k in wW:
        print(f'k={k}: (w-1)(log k -1) = {(wW[k]-1)*(math.log(k)-1):.4f}; beta+gamma = {beta[k]+EULER:.4f}')
out['weibull_w'] = {str(k): wW[k] for k in ks_all}
out['weibull_beta'] = {str(k): beta[k] for k in ks_all}
out['weibull_R0'] = {str(k): R0[k] for k in ks_all}
out['k0'] = k0
out['monotone'] = dict(w_dec_from3=dec_w, beta_dec_from5=dec_b, max_k_beta_pos=max(pos_beta), smallest_gap=gaps[0][0])

# ---------------------------------------------------------------- 5. s_k signs to large k (O(1) per k running sums)
print('== 5. s_k signs ==', flush=True)
KS = 10 ** 6
jj = np.arange(1, KS + 1, dtype=np.float64)
Hj1 = np.concatenate([[0.0], np.cumsum(1.0 / jj[:-1])])
m = -Hj1
res = {}
for fam, uu in [('weibull', -np.log(jj)), ('hazen', -np.log(jj - 0.5)), ('gringorten', -np.log(jj - 0.44))]:
    # shift to reduce cancellation: use cumulative centered sums with Welford-like updates via cumsum
    su = np.cumsum(uu); sm = np.cumsum(m); suu = np.cumsum(uu * uu); sum_ = np.cumsum(uu * m)
    kk = jj
    Suu = suu - su * su / kk
    Sum = sum_ - su * sm / kk
    s = Sum[1:] / Suu[1:]   # k = 2..KS
    sm1 = s - 1
    if fam == 'weibull':
        ok = bool(np.all(sm1 > 0)); print('weibull s_k>1 for 2<=k<=1e6:', ok, 'min s-1', sm1.min())
    if fam == 'hazen':
        ok = bool(np.all(sm1 < 0)); print('hazen s_k<1 for 2<=k<=1e6:', ok, 'max s-1', sm1.max())
    if fam == 'gringorten':
        below = np.where(sm1 < 0)[0] + 2
        print('gringorten s_k<1 at k in', below.min(), '..', below.max(), 'count', len(below),
              '; s_10, s_11 =', s[8], s[9], '; s at 1e3,1e4,1e5,1e6:', s[998], s[9998], s[99998], s[-1])
    res[fam] = dict(min=float(sm1.min()), max=float(sm1.max()))
out['s_signs'] = res

# ---------------------------------------------------------------- 6. variances / MSE at (10,143,2857)
print('== 6. MSE at (10,143,2857) ==', flush=True)
from scipy.special import polygamma
k, M, N = 10, 143, 2857
psi1 = lambda x: float(polygamma(1, x))
varDk = psi1(k) - psi1(M + 1)
varY = psi1(1) - psi1(N + 1)
uW = positions('weibull', k)
wq = wk['weibull']; EW2 = w2_laplace(uW); varW = EW2 - wq * wq
for ro in ['logN1', 'logN']:
    y = readoff(ro, 'weibull', N, M)
    cc = y - positions('weibull', k, M).mean()
    var = varDk + varY + (k - 1) / k ** 2 + 2 * cc * wq / k + cc * cc * varW
    e = exact_bias(positions('weibull', k, M), wq, k, M, N, y)
    print(f'inverse Weibull {ro}: mean={e:.5f} var={var:.5f} sd={math.sqrt(var):.4f} MSE={var + e*e:.4f}')
    out[f'mse_inv_weibull_{ro}'] = var + e * e
# forward digamma at H_N: forecast Dbar + c*beta_f, beta_f = S_mD/S_mm linear in V
md = -Hvec(k); mc = md - md.mean(); Smm = float(mc @ mc)
bdig = np.cumsum(mc)[:-1]; l = np.arange(1, k)
c2 = H(N) - mbar(k, M)
coef = l / k + c2 * bdig / Smm        # coefficient of V_l in L + c beta_f
var_fwd = varDk + varY + float(np.sum(coef ** 2 / l ** 2))
print(f'forward digamma at H_N: MSE={var_fwd:.4f}')
a_ = H(N) - H(M) + H(k - 1)
mv = varDk + varY + a_ * a_ / (k - 1)
mmse = varDk + varY + a_ * a_ / k
print(f'UMVU: MSE={mv:.4f}; min-MSE equivariant: {mmse:.4f}; c*={(k-1)*a_/k:.4f}; Var(Y)={varY:.4f}')
out['mse'] = dict(fwd_dig=var_fwd, umvu=mv, minmse=mmse, varY=varY)

# ---------------------------------------------------------------- 7. misc I1/I3a checks
print('== 7. misc ==', flush=True)
# Theorem 3b one-N counterexample
uu = np.array([math.log(144), math.log(72), 3.79844])
ys = 7.04327
b2 = mbar(2, 143) + (ys - uu[:2].mean()) * (1 / (uu[0] - uu[1])) - H(2857)
w3c = w_laplace(uu)
b3 = mbar(3, 143) + (ys - uu.mean()) * w3c - H(2857)
print(f'one-N counterexample: bias k=2 {b2:+.2e}, k=3 {b3:+.2e} (w3={w3c:.6f})')
out['oneN_counterexample'] = dict(b2=b2, b3=b3)
# exact read-off shift for Weibull k=10
for (k, M, N) in cells:
    e = exact_bias(positions('weibull', k, M), wq, k, M, N, math.log(N + 1))
    d = e / wq
    print(f'({k},{M},{N}): unbiased read-off = log(N+1) - {d:.4f}; survival {math.exp(d):.3f}/(N+1)')
# digamma at H_N
e = exact_bias(positions('digamma', 10, 143), wk['digamma'], 10, 143, 2857, H(2857))
print('digamma own at (10,143,2857):', e, ' (w-1)(H_N - mbar) =', (wk['digamma'] - 1) * (H(2857) - mbar(10, 143)))
# rate formula
for (k, M, N) in [(10, 143, 2857), (10, 44, 891)]:
    R = N / M
    fin = exact_bias(positions('weibull', k, M), wq, k, M, N, math.log(N + 1))
    lim = (wq - 1) * math.log(R) + beta10
    print(f'rate ({k},{M},{N}): exact diff {fin - lim:+.6f}, formula -(w-1/2)(1-1/R)/M = {-(wq - 0.5)*(1 - 1/R)/M:+.6f}')
# exact at N=M with read-off log(N+1)
for M in [44, 143, 250]:
    e1 = exact_bias(positions('weibull', 10, M), wq, 10, M, M, math.log(M + 1))
    e2 = exact_bias(positions('weibull', 10, M), wq, 10, M, M, math.log(M))
    print(f'N=M={M}: bias(log(N+1))={e1:.9f} beta10={beta10:.9f}; bias(log N)={e2:.6f}')
# empirical shift
for M in [44, 143]:
    print(f'empirical-minus-Weibull at common read depth, M={M}: {wq*math.log((M+1)/M):.5f}')

json.dump(out, open(__import__('pathlib').Path(__file__).with_name('exact_checks.json'), 'w'), indent=1, default=float)
print('done, elapsed', time.time() - t0)
