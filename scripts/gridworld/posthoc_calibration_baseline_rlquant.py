"""Post-hoc shift / shift-and-scale calibration of the RL FRLMB OLS forecast.

RL analogue of `posthoc_calibration_baseline.py`. The score is per-task regret
(score_mode='raw_regret' = identity), so calibration is in regret units.

Two calibration variants:

    1-param: pred_calibrated = pred_raw + beta
    2-param: pred_calibrated = alpha * pred_raw + beta

Calibration is fit on the META-split (fit, deploy) pairs and applied to the
HELDOUT-split pairs. To calibrate the pretrained model (Cal. in the paper's figures),
use --model pretrained; for the trained model (Ours + cal., RL + cal.), --model meta_learned.

The script reads only `forecast_traces.npz` (and its manifests json); it does
NOT re-run any model. It produces an output JSON with the calibration params
plus per-pair raw + calibrated predictions and aggregate metrics.

Usage:
    python scripts/gridworld/posthoc_calibration_baseline_rlquant.py \\
        --run-dir <run_root>/seed_11 \\
        --out-json <run_root>/seed_11/posthoc_calibration_baseline.json \\
        --model pretrained
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from forecast_experiment.ols_utils import fit_ols_tail_from_scores


def _ols_forecast_for_pair(
    fit_regrets: np.ndarray,
    deploy_regrets: np.ndarray,
    *,
    n_top: int,
    plotting: str,
) -> Tuple[float, float]:
    """Return (predicted_worst_regret, actual_worst_regret) for one pair."""
    pred, _a, _b = fit_ols_tail_from_scores(
        fit_regrets.tolist(),
        n_top=n_top,
        plotting=plotting,
        deploy_queries=int(len(deploy_regrets)),
        score_mode="raw_regret",
    )
    actual = float(deploy_regrets.max())
    return float(pred), actual


def _pairs_for_split(
    npz: Dict[str, np.ndarray],
    manifests: List[Dict[str, Any]],
    *,
    n_top: int,
    plotting: str,
    scale: str = "regret",
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for m in manifests:
        prefix = m["array_prefix"]
        if scale == "neg_return":
            # Return-scale forecasting: score = -return, worst position =
            # lowest return. Avoids the regret-scale degeneracy when a
            # policy drives regrets to ~0 (V* spread keeps -return spread).
            fit = -npz[f"{prefix}__fit_returns"]
            deploy = -npz[f"{prefix}__deploy_returns"]
        else:
            fit = npz[f"{prefix}__fit_regrets"]
            deploy = npz[f"{prefix}__deploy_regrets"]
        pred, actual = _ols_forecast_for_pair(
            fit, deploy, n_top=n_top, plotting=plotting,
        )
        out.append({
            "name": m["name"],
            "predicted_worst_regret": pred,
            "actual_worst_regret": actual,
            "fit_count": int(fit.shape[0]),
            "deploy_count": int(deploy.shape[0]),
        })
    return out


def _fit_calibrations(cal: List[Dict[str, float]]) -> Dict[str, Any]:
    pred = np.array([c["predicted_worst_regret"] for c in cal], dtype=float)
    actual = np.array([c["actual_worst_regret"] for c in cal], dtype=float)
    finite = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[finite]
    actual = actual[finite]
    n = len(pred)
    if n < 2:
        raise RuntimeError(f"Not enough finite calibration points: {n}")

    beta_1 = float(np.mean(actual - pred))

    A = np.column_stack([pred, np.ones_like(pred)])
    coef, _r, _rk, _sv = np.linalg.lstsq(A, actual, rcond=None)
    alpha_2 = float(coef[0])
    beta_2 = float(coef[1])

    pred_1cal = pred + beta_1
    pred_2cal = alpha_2 * pred + beta_2
    return {
        "n_calibration_pairs": int(n),
        "calibration_pred_mean": float(pred.mean()),
        "calibration_pred_std": float(pred.std()),
        "calibration_actual_mean": float(actual.mean()),
        "calibration_actual_std": float(actual.std()),
        "calibration_raw_signed_mean": float(np.mean(actual - pred)),
        "calibration_raw_sq_error_mean": float(np.mean((actual - pred) ** 2)),
        "shift_1param": {
            "beta": beta_1,
            "calibration_sq_error_mean": float(np.mean((pred_1cal - actual) ** 2)),
        },
        "shift_scale_2param": {
            "alpha": alpha_2,
            "beta": beta_2,
            "calibration_sq_error_mean": float(np.mean((pred_2cal - actual) ** 2)),
        },
    }


def _apply_to_heldout(
    heldout_pairs: List[Dict[str, float]],
    calibrations: Dict[str, Any],
) -> Dict[str, Any]:
    beta_1 = calibrations["shift_1param"]["beta"]
    alpha_2 = calibrations["shift_scale_2param"]["alpha"]
    beta_2 = calibrations["shift_scale_2param"]["beta"]

    per_pair: List[Dict[str, Any]] = []
    pred_a, actual_a, cal1_a, cal2_a = [], [], [], []
    for r in heldout_pairs:
        pred_raw = float(r["predicted_worst_regret"])
        actual = float(r["actual_worst_regret"])
        if not (np.isfinite(pred_raw) and np.isfinite(actual)):
            continue
        pred_1 = pred_raw + beta_1
        pred_2 = alpha_2 * pred_raw + beta_2
        per_pair.append({
            "name": r["name"],
            "predicted_worst_regret_raw": pred_raw,
            "actual_worst_regret": actual,
            "predicted_worst_regret_shift_1param": pred_1,
            "predicted_worst_regret_shift_scale_2param": pred_2,
            "signed_error_raw": pred_raw - actual,
            "signed_error_shift_1param": pred_1 - actual,
            "signed_error_shift_scale_2param": pred_2 - actual,
            "sq_error_raw": (pred_raw - actual) ** 2,
            "sq_error_shift_1param": (pred_1 - actual) ** 2,
            "sq_error_shift_scale_2param": (pred_2 - actual) ** 2,
        })
        pred_a.append(pred_raw)
        actual_a.append(actual)
        cal1_a.append(pred_1)
        cal2_a.append(pred_2)

    pred_a = np.array(pred_a)
    actual_a = np.array(actual_a)
    cal1_a = np.array(cal1_a)
    cal2_a = np.array(cal2_a)

    def _summary(p, a):
        signed = p - a
        return {
            "n": int(len(p)),
            "mean_signed_error": float(np.mean(signed)),
            "mean_abs_signed_error": float(np.mean(np.abs(signed))),
            "mean_sq_error": float(np.mean(signed ** 2)),
            "median_signed_error": float(np.median(signed)),
        }

    return {
        "per_pair": per_pair,
        "heldout_summary_no_calibration": _summary(pred_a, actual_a),
        "heldout_summary_shift_1param": _summary(cal1_a, actual_a),
        "heldout_summary_shift_scale_2param": _summary(cal2_a, actual_a),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to a seed_<N>/ dir containing forecast_traces.npz "
                        "and forecast_traces_manifests.json.")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--model", choices=["pretrained", "meta_learned"], required=True,
                   help="Which model's traces to calibrate: 'pretrained' (the "
                        "pretrained model; Cal.) or 'meta_learned' (the trained "
                        "model; Ours + cal. and RL + cal.).")
    p.add_argument("--scale", choices=["regret", "neg_return"], default="regret",
                   help="Forecast scale: 'regret' (paper yardstick) or "
                        "'neg_return' (worst-case return; no V* involved).")
    args = p.parse_args()

    run_dir: Path = args.run_dir
    npz_path = run_dir / "forecast_traces.npz"
    manifests_path = run_dir / "forecast_traces_manifests.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")
    if not manifests_path.exists():
        raise FileNotFoundError(f"missing {manifests_path}")

    raw = np.load(npz_path, allow_pickle=False)
    npz: Dict[str, np.ndarray] = {k: raw[k] for k in raw.files}
    meta = json.loads(manifests_path.read_text())
    n_top = int(meta["ols_top_k"])
    plotting = str(meta["plotting_position"])
    if str(meta.get("ols_score_mode", "raw_regret")) != "raw_regret":
        print(
            f"Warning: ols_score_mode in manifests = {meta.get('ols_score_mode')!r}; "
            "this script assumes raw_regret. Recomputing with raw_regret.",
            file=sys.stderr,
        )
    cohort = meta["models"][args.model]
    meta_manifests = cohort["meta"]
    heldout_manifests = cohort["heldout"]
    print(f"Loaded {len(meta_manifests)} meta + {len(heldout_manifests)} heldout pairs "
          f"for model={args.model}; OLS top-k={n_top}, plotting={plotting}", flush=True)

    cal_pairs = _pairs_for_split(npz, meta_manifests, n_top=n_top, plotting=plotting,
                                 scale=args.scale)
    held_pairs = _pairs_for_split(npz, heldout_manifests, n_top=n_top, plotting=plotting,
                                  scale=args.scale)

    calibrations = _fit_calibrations(cal_pairs)
    heldout_summary = _apply_to_heldout(held_pairs, calibrations)

    output = {
        "run_dir": str(run_dir),
        "model": args.model,
        "scale": args.scale,
        "ols_top_k": n_top,
        "plotting_position": plotting,
        "calibrations": calibrations,
        "heldout_calibrated": heldout_summary,
        "calibration_data_per_pair": cal_pairs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, default=str))

    print(f"\nWrote calibration results to {args.out_json}", flush=True)
    print(f"\n=== Summary (heldout, n={heldout_summary['heldout_summary_no_calibration']['n']}) ===")
    for label, key in [
        ("no calibration", "heldout_summary_no_calibration"),
        ("shift only (1 param)", "heldout_summary_shift_1param"),
        ("shift + scale (2 params)", "heldout_summary_shift_scale_2param"),
    ]:
        s = heldout_summary[key]
        print(
            f"  {label:25s} | mean signed = {s['mean_signed_error']:+.3f} | "
            f"mean |signed| = {s['mean_abs_signed_error']:.3f} | "
            f"mean sq = {s['mean_sq_error']:.3f}",
        )


if __name__ == "__main__":
    main()
