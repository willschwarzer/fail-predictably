"""One table of held-out folds for every lqdyn / lqpw run and every scored snapshot step.

For each run directory under $RESULTS_ROOT/lquant_dynamics/runs/ matching the given patterns, and each step its
snapshot reader has scored (snapshot_evals/stepNNNN_ against snapshot_evals/base_),
the run's own-k MSE fold over base on the readers' metric (poolscale_readout: 200
random fit/deploy splits per held-out pool, seeded by the pool name), as read and
with each pool's trained log-probs scaled so its involved-set spread equals its
base pool's own (heldout_spread.py's "spread set to the base pool's own"; exact,
since scaling log p by c scales every split's error by c). Also the median spread
ratio (trained / base own), the mean signed error in decades (negative = forecast
below the deploy worst), and the top / median drops in nats as the readers print
them. Fit size, k and pool count come from each run's own traces, so WildChat
(M250, k10 or k50) and password (M44, k10) runs read alike.

  python fleet_table.py OUT.csv PATTERN [PATTERN ...]   e.g. 'lqpw_pw2*' 'lqdyn_v2e2*ap50*'
"""
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))
from cert_sq_strategy import involved_sigma  # noqa: E402
import g_transfer_k as gk  # noqa: E402
from forecast_experiment.dynamics_probe import involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402
from heldout_spread import pools  # noqa: E402
from poolscale_readout import S, split_errors  # noqa: E402

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
RUNS = RESULTS_ROOT / "lquant_dynamics/runs"
READOUTS = RESULTS_ROOT / "lquant_dynamics/readouts"
BAND_MARGIN = 0.13
STEP_RE = re.compile(r"^step(\d{4})_test_forecast_traces\.npz$")


def base_errors(base, k):
    """Per pool: base values, fit size, split permutations, base split errors, base spread."""
    out = {}
    for name, (vb, M) in base.items():
        n = vb.size
        e = count_extrapolated_eval_points(M, n - M, plotting="weibull")
        rng = np.random.default_rng(int.from_bytes(name.encode(), "little") % (2**32))
        perms = np.stack([rng.permutation(n) for _ in range(S)])
        eb, _ = split_errors(vb, perms, M, k=k)
        # The band's reference (certificate construction: k-matched reach L, straightened top-L).
        sref = involved_sigma_reference(vb, n_fit=M, fit_top_k=k, extrap_k=e, m=gk.reach(n, M, k, e))
        out[name] = (vb, M, e, perms, eb, involved_sigma(vb, M, k, e), sref)
    return out


def read_step(be, trained, k):
    mb = mt = mm = mband = 0.0
    ratio, signed, top, med = [], [], [], []
    below = 0
    for name, (vb, M, e, perms, eb, sb, sref) in be.items():
        vs = trained[name][0]
        et, _ = split_errors(vs, perms, M, k=k)
        ss = involved_sigma(vs, M, k, e)
        m_t = float((et ** 2).mean())
        mb += float((eb ** 2).mean())
        mt += m_t
        mm += m_t * (sb / ss) ** 2
        # Band-clamped: an out-of-band pool scaled to the nearest band edge (BAND_MARGIN nats).
        band = min(max(ss, sref - BAND_MARGIN), sref + BAND_MARGIN)
        mband += m_t * (band / ss) ** 2
        below += ss < sref - BAND_MARGIN
        ratio.append(ss / sb)
        signed.append(float(et.mean()))
        top.append(float(vb.max() - vs.max()))
        med.append(float(np.median(vb) - np.median(vs)))
    return {"fold": mb / mt, "matched": mb / mm, "band": mb / mband, "below_band": below,
            "spread": float(np.median(ratio)),
            "signed": float(np.mean(signed)), "top_drop": float(np.mean(top)),
            "median_drop": float(np.mean(med)), "pools": len(ratio)}


def main():
    out_csv = Path(sys.argv[1])
    runs = sorted({d for pat in sys.argv[2:] for d in RUNS.glob(pat) if (d / "snapshot_evals").is_dir()})
    rows = []
    for d in runs:
        ev = d / "snapshot_evals"
        if not (ev / "base_test_forecast_traces.npz").exists():
            continue
        steps = sorted(int(m.group(1)) for f in ev.iterdir() if (m := STEP_RE.match(f.name)))
        if not steps:
            continue
        base, k = pools(ev / "base_test_forecast_traces.npz")
        be = base_errors(base, k)
        M = next(iter(base.values()))[1]
        for step in steps:
            trained, _ = pools(ev / f"step{step:04d}_test_forecast_traces.npz")
            r = read_step(be, trained, k)
            rows.append({"run": d.name, "step": step, "k": k, "M": M, **r})
            print(f"{d.name:34s} step {step:3d} k{k:<2d} | as read {r['fold']:6.2f}x | band {r['band']:6.2f}x "
                  f"({r['below_band']} below) | spread-matched {r['matched']:6.2f}x "
                  f"| spread {r['spread']:.2f}x base | signed {r['signed']:+.2f} | drop top {r['top_drop']:5.1f} "
                  f"median {r['median_drop']:5.1f} nats", flush=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "step", "k", "M", "pools", "fold", "band", "below_band",
                                           "matched", "spread",
                                           "signed", "top_drop", "median_drop"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
