"""Does a run's held-out gain come with a collapsed tail? (involved-set spread vs the band)

The anchpen recipe penalizes (weight 50) the involved set's spread leaving a band of
+/-0.13 nats around its reference: the std, in nats of log p, of the fit top-k plus the
deploy ranks the forecast extrapolates to, averaged over splits, referenced to the
straight line through the base pool's top L ranks (dynamics_probe.involved_sigma_reference,
L = the deepest rank any split reads). The IOG-only arms train without that term. Here,
for RUN_DIR@STEP, on each held-out pool's reader traces (base and step; fit + deploy =
the whole pool), at the run's own k with each pool's own fit size (M250/N2050 on WildChat,
M44/N891 on password), we print the reference, the
base pool's own spread, the step's spread, the step / reference ratio, and whether the
step sits below the floor's edge (reference - 0.13), plus the top and median drops. A
gain earned by flattening the tail shows up as a ratio well under 1. Then the run's
k-own MSE fold over base on the reader's splits three ways: as read; with each pool's
trained log-probs scaled up (c > 1) until its spread reaches the floor's edge, where it
is below it (the certificate's one-sided accounting); and scaled to the base pool's own
spread in either direction (a fold from shape alone).

  python heldout_spread.py RUN_DIR@STEP[:K] [RUN_DIR@STEP[:K] ...]   (K overrides the run's own k)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g_transfer_k as gk  # noqa: E402
from cert_sq_strategy import MARGIN, involved_sigma  # noqa: E402
from forecast_experiment.dynamics_probe import involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402

from poolscale_readout import S, split_errors  # noqa: E402

def pools(npz_path):
    z = np.load(npz_path)
    meta = json.loads(Path(str(npz_path).replace(".npz", "_metadata.json")).read_text())
    out = {}
    for t in meta["targets"]:
        p = t["array_prefix"]
        fit = z[f"{p}__fit_log_probs"]
        out[t["target"]] = (np.concatenate([fit, z[f"{p}__deploy_log_probs"]]).astype(np.float64), int(fit.size))
    return out, int(meta["ols_top_k"])


def run(arg):
    run_dir, _, step = arg.partition("@")
    step, _, k_override = step.partition(":")
    ev = Path(run_dir) / "snapshot_evals"
    base, k = pools(ev / "base_test_forecast_traces.npz")
    if k_override:
        k = int(k_override)
    trained, _ = pools(ev / f"step{int(step):04d}_test_forecast_traces.npz")
    print(f"\n== {Path(run_dir).name} @ {step}, k{k}: per held-out pool, spread in nats")
    print("   pool   reference  base own   step   step/ref  below edge   top drop  median drop")
    ratios, below = [], 0
    mse = {"base": 0.0, "raw": 0.0, "floor": 0.0, "matched": 0.0}
    signed_b, signed_t = [], []
    dec = {"bb": 0.0, "vb": 0.0, "bt": 0.0, "vt": 0.0}
    for name, (vb, M) in base.items():
        vs = trained[name][0]
        n = vb.size
        e = count_extrapolated_eval_points(M, n - M, plotting="weibull")
        L = gk.reach(n, M, k, e)
        sref = involved_sigma_reference(vb, n_fit=M, fit_top_k=k, extrap_k=e, m=L)
        sb = involved_sigma(vb, M, k, e)
        ss = involved_sigma(vs, M, k, e)
        low = ss < sref - MARGIN
        below += int(low)
        ratios.append(ss / sref)
        top = float(np.sort(vb)[::-1][0] - np.sort(vs)[::-1][0])
        print(f"   {name}  {sref:8.2f}  {sb:8.2f}  {ss:7.2f}  {ss / sref:7.2f}   {'yes' if low else 'no':>6s}"
              f"    {top:7.2f}   {float(np.median(vb - vs)):7.2f}")
        # The reader's metric (poolscale_readout: 200 splits seeded by the pool name). Scaling a
        # pool's log-probs by c scales every split's error and its involved spread by exactly c
        # (the inverse-OLS forecast is equivariant under that scaling), so c^2 * MSE is the
        # pool's MSE with its spread moved by c.
        rng = np.random.default_rng(int.from_bytes(name.encode(), "little") % (2**32))
        perms = np.stack([rng.permutation(n) for _ in range(S)])
        eb, _ = split_errors(vb, perms, M, k=k)
        et, _ = split_errors(vs, perms, M, k=k)
        m_t = float((et ** 2).mean())
        signed_b.append(float(eb.mean()))
        signed_t.append(float(et.mean()))
        dec["bb"] += float(eb.mean()) ** 2
        dec["vb"] += float(eb.var())
        dec["bt"] += float(et.mean()) ** 2
        dec["vt"] += float(et.var())
        mse["base"] += float((eb ** 2).mean())
        mse["raw"] += m_t
        mse["floor"] += m_t * max(1.0, (sref - MARGIN) / ss) ** 2
        mse["matched"] += m_t * (sb / ss) ** 2
    print(f"   median step/ref {np.median(ratios):.2f} (range {min(ratios):.2f}-{max(ratios):.2f}); "
          f"{below}/{len(ratios)} pools below the floor's edge (reference line over the top L = {L} ranks, "
          f"{e} extrapolated deploy ranks)")
    print(f"   k{k} MSE fold over base: as read {mse['base'] / mse['raw']:.2f}x | spread raised to the floor's edge "
          f"where below it {mse['base'] / mse['floor']:.2f}x | spread set to the base pool's own "
          f"{mse['base'] / mse['matched']:.2f}x")
    print(f"   mean signed error (decades; negative = forecast below the deploy worst): base {np.mean(signed_b):+.2f}, "
          f"trained {np.mean(signed_t):+.2f}; trained pools over-forecast {sum(s > 0 for s in signed_t)}/{len(signed_t)}")
    n_p = len(signed_t)
    print(f"   MSE (decades^2, mean over pools) = squared per-pool bias + split variance: base {dec['bb'] / n_p:.3f} + "
          f"{dec['vb'] / n_p:.3f}, trained {dec['bt'] / n_p:.3f} + {dec['vt'] / n_p:.3f}; variance-only fold "
          f"{dec['vb'] / dec['vt']:.2f}x")


def main():
    for arg in sys.argv[1:]:
        run(arg)


if __name__ == "__main__":
    main()
