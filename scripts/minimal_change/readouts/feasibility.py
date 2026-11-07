"""Is a trained held-out pool inside the certificate's feasible set, and what does it read once it is?

The certificate (cert_scale.certify) is the best straight tail over the top L ranks with (i) its
involved-set sigma inside the band [sigma_ref - 0.13, sigma_ref + 0.13] and (ii) no prompt raised
above its base log p. A band clamp alone enforces (i) only: password s17 at step 90, band-clamped,
raises 175 of 6,545 held-out prompts above base, so its 4.49x against the 4.29x cap compares
different feasible sets. Per run and step, this script
reports, per held-out pool and pooled:

  raised       prompts above base (by more than 1e-6 nats) after band clamping, and the largest raise;
  feasible     the band-clamped pool projected onto both constraints: raised prompts set to their
               base value, then the pool rescaled into the band (a scale c > 1 only lowers values,
               so it keeps (ii)); if the projection leaves the pool above the band's upper edge,
               scale down and re-clamp, up to 50 rounds; pools still outside are counted;
  folds        band-clamped and feasible pooled folds next to the cap, with a paired bootstrap over
               pools (2,000 resamples) of fold / cap.

  python feasibility.py RUN_GLOB:STEP [RUN_GLOB:STEP ...]   e.g. 'lqpw_pw2apsp_seed17:90'
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cert_scale import MARGIN, certify  # noqa: E402
from fleet_table import RUNS, base_errors  # noqa: E402
from cert_sq_strategy import involved_sigma  # noqa: E402
from heldout_spread import pools  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

TOL = 1e-6


def project(vt, vb, M, k, e, lo, hi, rounds=50):
    """Nearest-in-spirit feasible pool: no prompt above base, involved sigma inside [lo, hi]."""
    v = np.minimum(vt, vb)
    # Relative slack: sigma is homogeneous in v, so a pool scaled onto an edge can land one ulp
    # outside it, which on deep pools (IOG-only arms, values of -1e3 nats) read as infeasible.
    lo_t, hi_t = lo * (1 - 1e-9), hi * (1 + 1e-9)
    for _ in range(rounds):
        s = involved_sigma(v, M, k, e)
        if s < lo_t:
            v = v * (lo / s)  # lowers every value, so no prompt rises
        elif s > hi_t:
            v = np.minimum(v * (hi / s), vb)
        else:
            return v, True
    s = involved_sigma(v, M, k, e)
    return v, lo_t <= s <= hi_t


def main():
    rng = np.random.default_rng(0)
    for spec in sys.argv[1:]:
        pat, step = spec.rsplit(":", 1)
        for d in sorted(RUNS.glob(pat)):
            ev = d / "snapshot_evals"
            f = ev / f"step{int(step):04d}_test_forecast_traces.npz"
            if not f.exists():
                print(f"\n{d.name} step {step}: no read")
                continue
            base, k = pools(ev / "base_test_forecast_traces.npz")
            trained, _ = pools(f)
            be = base_errors(base, k)
            rows = []
            n_raised = n_total = 0
            worst = 0.0
            infeasible = 0
            for name, (vb, M, e, perms, eb, sb, _sref) in be.items():
                vt = trained[name][0]
                assert vt.size == vb.size
                out, tgt, e2, L, sref, info = certify(vb, M, k)
                lo, hi = sref - MARGIN, sref + MARGIN
                st = involved_sigma(vt, M, k, e)
                vc = vt * (min(max(st, lo), hi) / st)  # band-clamped
                up = vc - vb
                n_raised += int((up > TOL).sum())
                n_total += up.size
                worst = max(worst, float(up.max()))
                vf, ok = project(vc, vb, M, k, e, lo, hi)
                infeasible += not ok
                ec, _ = split_errors(out, perms, M, k=k)
                eband, _ = split_errors(vc, perms, M, k=k)
                efeas, _ = split_errors(vf, perms, M, k=k)
                rows.append([float((x ** 2).mean()) for x in (eb, ec, eband, efeas)])
            r = np.array(rows)  # pools x (base, cert, band-clamped, feasible)
            fold = lambda a: a[:, 0].sum() / a[:, 1:].sum(axis=0)
            cap, fb, ff = fold(r)
            boot = np.array([fold(r[rng.integers(0, len(r), len(r))]) for _ in range(2000)])
            qb = np.percentile(boot[:, 1] / boot[:, 0], [2.5, 50, 97.5])
            qf = np.percentile(boot[:, 2] / boot[:, 0], [2.5, 50, 97.5])
            print(f"\n{d.name} step {step} (k{k}, M{M}, {len(r)} pools): cap {cap:.2f}x | band-clamped {fb:.2f}x "
                  f"(over cap {fb / cap:.3f}, 95% bootstrap {qb[0]:.3f}-{qb[2]:.3f}) | feasible {ff:.2f}x "
                  f"(over cap {ff / cap:.3f}, {qf[0]:.3f}-{qf[2]:.3f})")
            print(f"   band-clamped pool raises {n_raised} of {n_total} prompts above base (largest {worst:.2f} nats); "
                  f"{infeasible} of {len(r)} pools could not be projected into the band without raising")
            for (name, _), (b, c, eb_, ef) in zip(be.items(), rows):
                print(f"   {name:>24s}  base {b:8.3f}  cert {c:7.3f}  band {eb_:7.3f}  feasible {ef:7.3f}  "
                      f"| per-pool cap {b / c:5.2f}x band {b / eb_:5.2f}x feasible {b / ef:5.2f}x")


if __name__ == "__main__":
    main()
