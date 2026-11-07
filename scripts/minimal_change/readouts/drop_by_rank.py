"""Where did training suppress, by base rank, on train vs held-out pools?

For finished rt runs: per pool, the drop (base - trained, nats) of the prompts at
base ranks 1-2, 3-10, 11-50, 51-300, and the bulk (300+), compared with the drop
the certified tail asks for at the same ranks. Held-out pools come from the
aligned baseline/final test traces; train pools from the anchor and the last
full-union dynamics dump. Also: password-v2 trained tails vs the base pool's
sigma reference (is the password result compressed by the WildChat standard?).

Inputs (under $RESULTS_ROOT): the lqdyn_rt* runs of scripts/minimal_change/submit_lqdyn_fleet.sh
(constraintduel2 and safetypolicy waves), and the password strict-v2 warm-start 0.6B IOG fleet
lquant_40slots_gcg6warm_strictv2_iog_full_bootstrap1_cache296_seed*_20260724/seed_*, whose
launcher is not in this release.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT / "src"))
from forecast_experiment.dynamics_probe import certified_tail_targets, involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
RUNS = RESULTS_ROOT / "lquant_dynamics/runs"
BANDS = [(0, 2), (2, 10), (10, 50), (50, 300), (300, None)]


def traces(run, prefix):
    meta = json.loads((run / f"{prefix}test_forecast_traces_metadata.json").read_text())
    z = np.load(run / f"{prefix}test_forecast_traces.npz")
    return {t["target"]: np.concatenate([z[f"{t['array_prefix']}__fit_log_probs"],
                                         z[f"{t['array_prefix']}__deploy_log_probs"]]).astype(np.float64)
            for t in meta["targets"]}


def last_full_dump(run):
    for f in sorted((run / "dynamics").glob("step_*_rank0.npz"), reverse=True):
        z = np.load(f, allow_pickle=True)
        tg = sorted({int(k.split("/")[0]) for k in z.files if "/" in k})
        if all(int(z[f"{t}/union_is_full"]) == 1 for t in tg):
            return f.name, {str(z["targets"][t]): (z[f"{t}/anchor_lp"].astype(np.float64),
                                                   z[f"{t}/union_lp"].astype(np.float64)) for t in tg}
    return None, {}


def bands(base, trained, cert_drop):
    order = np.argsort(-base)
    drop = (base - trained)[order]
    out = []
    for lo, hi in BANDS:
        sl = slice(lo, hi)
        cd = cert_drop[lo:hi] if hi is not None and hi <= cert_drop.size else (cert_drop[lo:] if lo < cert_drop.size else np.array([0.0]))
        out.append((float(np.mean(drop[sl])), float(np.mean(cd)) if cd.size else 0.0))
    return out


def cert_drop_for(base):
    e = count_extrapolated_eval_points(250, base.size - 250, plotting="weibull")
    sref = involved_sigma_reference(base, n_fit=250, fit_top_k=10, extrap_k=e, m=250)
    idx, tgt, info = certified_tail_targets(base, n_fit=250, fit_top_k=10, extrap_k=e, sigma_ref=sref,
                                            sigma_margin=0.13, two_sided=True, max_rank=300)
    return base[idx] - tgt


def fmt(rows):
    m = np.mean(np.array(rows), axis=0)  # (bands, 2)
    return "  ".join(f"r{lo + 1}-{hi if hi else 'end'}: {m[i, 0]:5.2f} (cert {m[i, 1]:4.2f})"
                     for i, (lo, hi) in enumerate(BANDS))


def main():
    for name in ["lqdyn_rtsig-anchpen_seed11", "lqdyn_rtsig-anchpensym_seed11", "lqdyn_rtsig-sym_seed11",
                 "lqdyn_rtk16siganch8_seed11", "lqdyn_rtk16siganch8_seed17", "lqdyn_rtk16siganch8_seed23"]:
        run = RUNS / name
        if not (run / "test_forecast_traces.npz").exists():
            continue
        b, t = traces(run, "baseline_"), traces(run, "")
        held = [bands(b[k], t[k], cert_drop_for(b[k])) for k in b]
        step, tr = last_full_dump(run)
        train = [bands(a, u, cert_drop_for(a)) for a, u in tr.values()]
        print(f"\n{name}\n  held-out ({len(held)} pools): {fmt(held)}\n  train ({len(train)} pools, {step}): {fmt(train)}")

    print("\n== password v2 0.6B trained tails vs the base pool's sigma reference (M44, k10)")
    rows = []
    for d in sorted(glob.glob(str(RESULTS_ROOT / "lquant_40slots_gcg6warm_strictv2_iog_full_bootstrap1_cache296_seed*_20260724/seed_*"))):
        d = Path(d)
        if not (d / "test_forecast_traces.npz").exists():
            continue
        b, t = traces(d, "baseline_"), traces(d, "")
        for k in b:
            n = b[k].size
            e = count_extrapolated_eval_points(44, n - 44, plotting="weibull")
            sref = involved_sigma_reference(b[k], n_fit=44, fit_top_k=10, extrap_k=e, m=250)
            strain = involved_sigma_reference(t[k], n_fit=44, fit_top_k=10, extrap_k=e, m=250)
            rows.append((sref, strain, float(b[k].max()), float(t[k].max()), float(np.median(t[k]))))
    r = np.array(rows)
    print(f"  {len(rows)} slots: base-line sigma ref {r[:, 0].mean():.2f} nats; trained-line sigma {r[:, 1].mean():.2f} "
          f"(ratio median {np.median(r[:, 1] / r[:, 0]):.2f}, range {np.min(r[:, 1] / r[:, 0]):.2f}-{np.max(r[:, 1] / r[:, 0]):.2f}); "
          f"top {r[:, 2].mean():+.2f} -> {r[:, 3].mean():+.2f} nats, trained median {r[:, 4].mean():+.1f}")


if __name__ == "__main__":
    main()
