"""Held-out-forecaster transfer and ties audit (paper appendix "Checks beyond
the training forecast and score", subsections "Transfer to forecasting
procedures unused in training" and "Ties in finite-precision scores").

Both audits are pure post-processing of the saved held-out eval traces
(`baseline_test_forecast_traces.npz` / `test_forecast_traces.npz`), which store
the raw fit (44) and deploy (891) log-probs per held-out target.

Held-out-forecaster transfer: recompute the worst-rank
forecast under forecaster variants that training never saw -- different top-k
(5, 20), different plotting positions (Hazen, Gringorten), and a
peaks-over-threshold GPD tail fit instead of the Gumbel-line OLS -- and check
whether the trained model's forecast-error improvement transfers.

Ties audit: count exact score ties at the order statistics
the method touches -- inside the fit top-k, at the top-k selection boundary,
and at the deploy worst rank.

Everything is recomputed from the traces; the paper-convention variant
(k=10, Weibull) is cross-checked against the run's stored
`forecast_sq_log10_error` whenever the sibling results json exists, so any
disagreement in the refit machinery is caught rather than silently reported.

CPU-only. Writes one JSON with per-target rows and per-group aggregates, and
prints the aggregate table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))

LN10 = math.log(10.0)
PSI_EPS = 1e-8  # matches ols_utils gumbel_log_prob clamp

VARIANTS = [
    # (name, kind, top_k, plotting)
    ("paper_k10_weibull", "ols", 10, "weibull"),
    ("k5_weibull", "ols", 5, "weibull"),
    ("k20_weibull", "ols", 20, "weibull"),
    ("k10_hazen", "ols", 10, "hazen"),
    ("k10_gringorten", "ols", 10, "gringorten"),
    ("gpd_pot_k10", "gpd", 10, None),
]


def plotting_positions(k: int, n: int, plotting: str) -> list[float]:
    out = []
    for j in range(1, k + 1):
        if plotting == "weibull":
            p = j / (n + 1.0)
        elif plotting == "gringorten":
            p = (j - 0.44) / (n + 0.12)
        elif plotting == "hazen":
            p = (j - 0.5) / n
        else:
            raise ValueError(plotting)
        out.append(min(max(p, 1e-300), 1.0))
    return out


def psi_transform(log_probs: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(-log_probs.astype(np.float64), PSI_EPS))


def ols_forecast_log_prob(fit_lp: np.ndarray, deploy_n: int, k: int, plotting: str) -> float:
    """Replicates fit_ols_tail_from_scores with score_mode='gumbel_log_prob'."""
    psi = np.sort(psi_transform(fit_lp))[::-1]
    n = len(psi)
    k = max(1, min(k, n))
    x = psi[:k]
    y = np.log(plotting_positions(k, n, plotting))
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 1e-20:
        x_star = float(x[0])
    else:
        b = float(((x - xm) * (y - ym)).sum()) / sxx
        a = ym - b * xm
        x_star = float(x[0]) if b == 0.0 else (math.log(1.0 / deploy_n) - a) / b
    return -math.exp(-x_star)


def gpd_forecast_log_prob(fit_lp: np.ndarray, deploy_n: int, k: int) -> float:
    """Peaks-over-threshold GPD fit in psi space; m-observation return level.

    Threshold u = (k+1)-th largest psi, so exactly k exceedances enter the fit
    (the same tail sample size the OLS variant uses). Return level solves
    P(X > x) = 1/deploy_n under the fitted tail; back-transform keeps the
    forecast a valid log-prob.
    """
    from scipy.stats import genpareto

    psi = np.sort(psi_transform(fit_lp))[::-1]
    n = len(psi)
    k = max(1, min(k, n - 1))
    u = float(psi[k])
    y = psi[:k] - u
    zeta = k / n
    m = deploy_n * zeta
    if float(y.max()) <= 0.0:
        x_star = u
    else:
        try:
            xi, _, sigma = genpareto.fit(y, floc=0.0)
        except Exception:
            xi, sigma = 0.0, float(y.mean())
        if sigma <= 0:
            x_star = u
        elif abs(xi) < 1e-9:
            x_star = u + sigma * math.log(m)
        else:
            x_star = u + (sigma / xi) * (m ** xi - 1.0)
    return -math.exp(-min(x_star, -math.log(PSI_EPS)))


def tie_stats(fit_lp: np.ndarray, deploy_lp: np.ndarray, k: int = 10) -> dict:
    fs = np.sort(fit_lp)[::-1]
    ds = np.sort(deploy_lp)[::-1]
    uniq_topk = len(np.unique(fs[:k]))
    return {
        "fit_topk_dup_values": int(k - uniq_topk),
        "fit_boundary_tie": bool(len(fs) > k and fs[k - 1] == fs[k]),
        "deploy_max_multiplicity": int((deploy_lp == ds[0]).sum()),
        "fit_topk_at_ceiling": int((fs[:k] == 0.0).sum()),
        "fit_any_dup_values": int(len(fs) - len(np.unique(fs))),
    }


def stored_mean_sq(results_json: Path) -> float | None:
    """Mean of forecast_sq_log10_error over targets in a stored results json."""
    try:
        obj = json.loads(results_json.read_text())
    except Exception:
        return None
    vals: list[float] = []

    def walk(o):
        if isinstance(o, dict):
            if "forecast_sq_log10_error" in o:
                v = o["forecast_sq_log10_error"]
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return float(np.mean(vals)) if vals else None


def group_key(trace_path: Path, root: Path) -> str:
    rel = str(trace_path.parent.relative_to(root))
    rel = re.sub(r"/(seed_\d+|final|best)(/.*)?$", "", rel)
    rel = re.sub(r"_seed\d+", "", rel)
    rel = re.sub(r"_\d{8}(_\d+)?$", "", rel)
    return rel


def seed_of(trace_path: Path) -> str:
    m = re.search(r"seed[_]?(\d+)", str(trace_path))
    return m.group(1) if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RESULTS_ROOT))
    ap.add_argument("--cache-root", default=None,
                    help="optional second root (e.g. artifacts/cache) scanned the same way")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    roots = [Path(args.root)]
    if args.cache_root:
        roots.append(Path(args.cache_root))

    rows: list[dict] = []
    sanity: list[dict] = []
    for root in roots:
        for trace in sorted(root.rglob("*test_forecast_traces.npz")):
            meta_path = trace.with_name(trace.name.replace(".npz", "_metadata.json"))
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            side = "baseline" if trace.name.startswith("baseline") else "trained"
            grp = group_key(trace, root)
            seed = seed_of(trace)
            data = np.load(trace)
            per_variant_sq: dict[str, list[float]] = defaultdict(list)
            for tinfo in meta.get("targets", []):
                pre = tinfo["array_prefix"]
                fit_lp = np.asarray(data[f"{pre}__fit_log_probs"], dtype=np.float64)
                deploy_lp = np.asarray(data[f"{pre}__deploy_log_probs"], dtype=np.float64)
                actual = float(deploy_lp.max())
                row = {
                    "group": grp,
                    "seed": seed,
                    "side": side,
                    "target": tinfo.get("target"),
                    "trace": str(trace),
                    "fit_n": len(fit_lp),
                    "deploy_n": len(deploy_lp),
                    "actual_worst_log_prob": actual,
                    "ties": tie_stats(fit_lp, deploy_lp),
                    "forecasts": {},
                }
                for name, kind, k, plotting in VARIANTS:
                    if kind == "ols":
                        pred = ols_forecast_log_prob(fit_lp, len(deploy_lp), k, plotting)
                    else:
                        pred = gpd_forecast_log_prob(fit_lp, len(deploy_lp), k)
                    sq = ((pred - actual) / LN10) ** 2
                    row["forecasts"][name] = {
                        "pred_log_prob": pred,
                        "sq_log10_error": sq,
                        "signed_log10_error": (pred - actual) / LN10,
                    }
                    per_variant_sq[name].append(sq)
                rows.append(row)
            # sanity: paper-variant recompute vs stored results json
            res_name = ("baseline_test_results.json" if side == "baseline"
                        else "test_results.json")
            res_path = trace.with_name(res_name)
            if res_path.exists() and per_variant_sq["paper_k10_weibull"]:
                stored = stored_mean_sq(res_path)
                mine = float(np.mean(per_variant_sq["paper_k10_weibull"]))
                if stored is not None:
                    sanity.append({
                        "trace": str(trace), "stored": stored, "recomputed": mine,
                        "abs_diff": abs(stored - mine),
                        "plotting_in_meta": meta.get("plotting_position"),
                    })

    # aggregate per (group, side, variant)
    agg: dict[tuple, dict] = {}
    by_gs: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_gs[(r["group"], r["side"])].append(r)
    for (grp, side), rs in sorted(by_gs.items()):
        seeds = sorted({r["seed"] for r in rs})
        entry = {"group": grp, "side": side, "n_seeds": len(seeds),
                 "n_targets": len(rs), "variants": {}, "ties": {}}
        for name, *_ in VARIANTS:
            sqs = [r["forecasts"][name]["sq_log10_error"] for r in rs]
            entry["variants"][name] = {"mean_sq_log10_error": float(np.mean(sqs))}
        entry["ties"] = {
            "targets_with_fit_topk_dup": int(sum(r["ties"]["fit_topk_dup_values"] > 0 for r in rs)),
            "targets_with_boundary_tie": int(sum(r["ties"]["fit_boundary_tie"] for r in rs)),
            "targets_with_deploy_max_tied": int(sum(r["ties"]["deploy_max_multiplicity"] > 1 for r in rs)),
            "targets_with_ceiling_in_topk": int(sum(r["ties"]["fit_topk_at_ceiling"] > 0 for r in rs)),
            "total_targets": len(rs),
        }
        agg[(grp, side)] = entry

    out = {
        "variants": [v[0] for v in VARIANTS],
        "aggregates": list(agg.values()),
        "sanity_checks": sanity,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))

    # printed summary: fold per variant for groups having both sides
    print(f"\nwrote {out_path}  ({len(rows)} target-rows)")
    bad_sanity = [s for s in sanity if s["abs_diff"] > 1e-3]
    print(f"sanity: {len(sanity)} stored-vs-recomputed checks, "
          f"{len(bad_sanity)} with |diff| > 1e-3")
    for s in bad_sanity[:10]:
        print(f"  MISMATCH {s['trace']}: stored {s['stored']:.4f} "
              f"recomputed {s['recomputed']:.4f} (meta plotting={s['plotting_in_meta']})")
    header = ["group", "side", "n"] + [v[0] for v in VARIANTS]
    print("\n" + " | ".join(header))
    for (grp, side), e in sorted(agg.items()):
        cells = [f"{e['variants'][v[0]]['mean_sq_log10_error']:.3f}" for v in VARIANTS]
        print(" | ".join([grp[:70], side, str(e["n_targets"])] + cells))
    print("\nties (targets with any exact tie, by group/side):")
    for (grp, side), e in sorted(agg.items()):
        t = e["ties"]
        if t["targets_with_fit_topk_dup"] or t["targets_with_boundary_tie"] or t["targets_with_deploy_max_tied"]:
            print(f"  {grp[:70]} [{side}] topk-dup {t['targets_with_fit_topk_dup']}"
                  f"/{t['total_targets']}, boundary {t['targets_with_boundary_tie']}, "
                  f"deploy-max-tied {t['targets_with_deploy_max_tied']}, "
                  f"ceiling-in-topk {t['targets_with_ceiling_in_topk']}")
    print("done")


if __name__ == "__main__":
    main()
