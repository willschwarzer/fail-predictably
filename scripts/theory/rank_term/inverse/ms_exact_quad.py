# Exact E[xi^2] for the exactly unbiased choices, from the stored quadrature moments (no MC).
# Run compute_exact.py first; it writes exact.json next to this file.
import json, math, sys
from pathlib import Path
D = str(Path(__file__).resolve().parent)
sys.path.insert(0, D)
import rt_exact as R
ex = json.load(open(D + '/exact.json'))
wt = ex['w_table']
k, M, N = 10, 44, 891
H = lambda n: sum(1.0 / l for l in range(1, n + 1))
HN = H(N)
dbar = sum(H(M) - H(i - 1) for i in range(1, k + 1)) / k
ubarW = sum(math.log((M + 1) / i) for i in range(1, k + 1)) / k
def mom(fam):
    t = wt[fam]; t = t.get(str(k), t.get(k))
    return t['w_quad'], t['EW2_quad'], t['ELW_quad']
for fam, ubar, label in (('digamma', dbar, 'stretched mean depths at H_N (= mean depths at dbar+(H_N-dbar)/w)'),
                         ('weibull', ubarW, 'Weibull at ubar+(H_N-dbar)/w')):
    w, e2, elw = mom(fam)
    c = (HN - dbar) / w
    v = R.var_xi_finite(fam, k, M, N, c, w, e2, elw)
    print(f'{label:70s} w={w:.6f}  E[xi^2]={v:.4f}  (bias 0)')
# checks against stored table values
w, e2, elw = mom('weibull'); c = math.log(N) - ubarW
b = dbar + c * w - HN; print('check Weibull logN: bias %.4f  E[xi^2] %.4f (expect 0.9985, 10.449)' % (b, R.var_xi_finite('weibull', k, M, N, c, w, e2, elw) + b * b))
w, e2, elw = mom('digamma'); c = HN - dbar
b = c * (w - 1); print('check mean depths at H_N: bias %.4f  E[xi^2] %.4f (expect 0.4302, ~8.14)' % (b, R.var_xi_finite('digamma', k, M, N, c, w, e2, elw) + b * b))
print('Var Dmax_N = %.4f' % (math.pi ** 2 / 6 - sum(1.0 / l ** 2 for l in range(N + 1, 10 ** 7))))
