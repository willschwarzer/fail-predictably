"""Three-panel per-method bar chart for the LM mod-bulk + GCG comparison.

Each panel is normalized to its pretrained-baseline mean so that y=1 is the
no-training reference and bars rise above 1 when the method improves on the
baseline. y is on a log scale, so a bar at y=10 means 10x improvement and a
bar at y=0.1 means 10x degradation.

  Panel 1  Capability preservation     baseline_KL / method_KL          (up = better)
  Panel 2  Safety = -log p(leak)       method_NLL / baseline_NLL        (up = safer)
  Panel 3  Forecast precision          baseline_err / method_err        (up = better)

The pretrained baseline has WildChat KL ~= 0; we clamp the baseline mean to a
small floor (default 1e-4) so the capability panel ratio is well-defined. This
clamp is documented in the y-axis label.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _three_metric_bars_common import Metric, render_three_metric_figure  # noqa: E402

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", REPO_ROOT / "artifacts" / "figures"))


@dataclass
class MethodSpec:
    name: str
    glob_pat: str
    color: str
    # "" -> regular method; "baseline" -> M2 (cal applied to pretrained);
    # "metatrained" -> M6 (cal applied to meta-trained model). For "metatrained",
    # capability/safety come from the meta-trained model itself (not baseline);
    # only forecast_err is replaced by the calibrated value.
    calibration_source: str = ""


DEFAULT_METHODS: list[MethodSpec] = [
    MethodSpec(
        "Cal.",
        "lquant_40slots_modbulkgcg_metatrain_seed*_2026050*",
        "#EFCEC2",
        calibration_source="baseline",
    ),
    MethodSpec(
        "SFT",
        "lquant_40slots_modbulkgcg_sft_union_seed*_20260504",
        "#E5B3A6",
    ),
    MethodSpec(
        "SFT + cal.",
        "lquant_40slots_modbulkgcg_sft_union_seed*_20260504",
        "#A65D49",
        calibration_source="metatrained",
    ),
    # Removed for now: SFT-on-deploy isn't a meaningful "oracle" -- it's just
    # SFT on half the per-target prompts, and (fit, deploy) are random splits
    # so the deploy half doesn't preferentially target worst-rank. Re-add by
    # uncommenting; also widen DEFAULT_GROUPS["Baselines"] back to (1, 5) and
    # shift "Ours" to (6, 7).
    # MethodSpec(
    #     "SFT (deploy)",
    #     "lquant_40slots_modbulkgcg_sft_deploy_seed*_2026050*",
    #     "#B59FC2",
    # ),
    # MethodSpec(
    #     "SFT (deploy) + cal.",
    #     "lquant_40slots_modbulkgcg_sft_deploy_seed*_2026050*",
    #     "#6E548A",
    #     calibration_source="metatrained",
    # ),
    MethodSpec(
        "Ours",
        "lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_seed*_20260505",
        "#9DB8D6",
    ),
    MethodSpec(
        "Ours + cal.",
        "lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_seed*_20260505",
        "#3D5A80",
        calibration_source="metatrained",
    ),
]


# Group indices align with DEFAULT_METHODS (no leftmost pretrained-baseline
# slot -- baseline is implicit, the dashed reference line in each panel).
# Currently: 0 = cal-baseline, 1 = SFT-union, 2 = SFT-union+cal, 3 = Ours,
# 4 = Ours+cal. To re-enable the leftmost baseline slot, prepend
# ("Ref.", 0, 0, "") and shift the others by +1, then pass
# include_baseline_slot=True to render_three_metric_figure.
# (If SFT-deploy methods are also re-added, widen "Baselines" to (1, 5) and
# shift "Ours" to (6, 7) on top of the +1 shift.)
DEFAULT_GROUPS: list[tuple[str, int, int, str]] = [
    ("Baselines", 0, 2, ""),
    ("Ours", 3, 4, "#FFE9C2"),
]


# Same three panels, same metrics, pointed at the strict-v2 warm-start-GCG bank.
# Only the run glob changes. The SFT and cal-baseline comparators were never
# trained on this bank, so this preset has just the two arms we have -- both come
# from the same runs, since "+ cal." is applied post hoc to the meta-trained
# model. The pretrained baseline is the dashed reference line in each panel, as
# in the default preset, so it needs no runs of its own.
STRICTV2_METHODS: list[MethodSpec] = [
    MethodSpec(
        "Ours",
        "lquant_40slots_gcg6warm_strictv2_iog_full_bootstrap1_cache296_seed*_20260724",
        "#9DB8D6",
    ),
    MethodSpec(
        "Ours + cal.",
        "lquant_40slots_gcg6warm_strictv2_iog_full_bootstrap1_cache296_seed*_20260724",
        "#3D5A80",
        calibration_source="metatrained",
    ),
]

STRICTV2_GROUPS: list[tuple[str, int, int, str]] = [
    ("Ours", 0, 1, "#FFE9C2"),
]

# Same again on the bank with `password_seeded` removed. That family
# spliced the slot password into an otherwise benign prompt, so it measured
# in-context copying rather than disclosure under a non-disclosure instruction; at
# 4B/8B it supplied nearly all the near-ceiling mass, which leaked into the
# 44-prompt calibration sample and handed the BASELINE forecast the answer. Cutting
# it moved baseline error 1.54 -> 5.94 decades at 4B and 0.99 -> 4.51 at 8B while
# leaving 0.6B unchanged (4.79 -> 4.66), so any trained-vs-baseline fold plotted
# from the older banks understates the method at >=4B. Prefer this preset for
# anything reported at 4B or 8B.
NOPWSEED_METHODS: list[MethodSpec] = [
    MethodSpec(
        "Ours",
        "lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_seed*_20260725",
        "#9DB8D6",
    ),
    MethodSpec(
        "Ours + cal.",
        "lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_seed*_20260725",
        "#3D5A80",
        calibration_source="metatrained",
    ),
]

NOPWSEED_GROUPS: list[tuple[str, int, int, str]] = [
    ("Ours", 0, 1, "#FFE9C2"),
]

PRESETS = {
    "may2026": (DEFAULT_METHODS, DEFAULT_GROUPS),
    "strictv2": (STRICTV2_METHODS, STRICTV2_GROUPS),
    "nopwseed": (NOPWSEED_METHODS, NOPWSEED_GROUPS),
}


def _slot_mean(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if not d:
        return None
    vals = [d[slot][key] for slot in d if key in d[slot]]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _seed_metrics(seed_dir: Path, forecast_metric: str) -> dict | None:
    post_y = _slot_mean(seed_dir / "test_results.json", "actual_worst_log_prob")
    post_fc = _slot_mean(seed_dir / "test_results.json", forecast_metric)
    base_y = _slot_mean(seed_dir / "baseline_test_results.json", "actual_worst_log_prob")
    base_fc = _slot_mean(seed_dir / "baseline_test_results.json", forecast_metric)
    kl_path = seed_dir / "kl_eval_results.json"
    base_kl_path = seed_dir / "baseline_kl_eval_results.json"
    if not kl_path.exists() or not base_kl_path.exists():
        return None
    kl = float(json.loads(kl_path.read_text()).get("single_token", float("nan")))
    base_kl = float(json.loads(base_kl_path.read_text()).get("single_token", float("nan")))
    if any(v is None or not math.isfinite(v) for v in (post_y, post_fc, base_y, base_fc, kl, base_kl)):
        return None
    # Safety raw value = NLL = -log p(leak) (higher = safer).
    return {
        "kl": kl,
        "safety_nll": -post_y,
        "forecast_err": post_fc,
        "baseline_kl": base_kl,
        "baseline_safety_nll": -base_y,
        "baseline_forecast_err": base_fc,
    }


def _calibration_seed_metrics(
    seed_dir: Path, calib_variant: str, source: str, forecast_metric: str,
) -> dict | None:
    """Build a per-seed record for an ex-post calibration method.

    `source` selects which model the calibration was applied to:
      - "baseline":   M2; cal applied to pretrained model. Capability/safety
                      come from the baseline_*.json files (model unchanged).
      - "metatrained": M6; cal applied to meta-trained model. Capability/safety
                      come from test_results.json / kl_eval_results.json (the
                      meta-trained model's own values).
    Forecast value comes from the matching posthoc_calibration_*.json.
    """
    if source == "baseline":
        cal_path = seed_dir / "posthoc_calibration_baseline.json"
    else:
        cal_path = seed_dir / "posthoc_calibration_metatrained.json"
    if not cal_path.exists():
        return None
    try:
        d = json.loads(cal_path.read_text())
    except json.JSONDecodeError:
        return None
    summary = d.get("heldout_calibrated", {}).get(f"heldout_summary_{calib_variant}")
    if not summary:
        return None
    forecast_err = summary.get("mean_sq_log10_error")
    if forecast_err is None or not math.isfinite(float(forecast_err)):
        return None

    # Always need baseline values for the figure-level baseline aggregation.
    base_y = _slot_mean(seed_dir / "baseline_test_results.json", "actual_worst_log_prob")
    base_kl_path = seed_dir / "baseline_kl_eval_results.json"
    if base_y is None or not base_kl_path.exists():
        return None
    base_kl = float(json.loads(base_kl_path.read_text()).get("single_token", float("nan")))
    if not math.isfinite(base_kl) or not math.isfinite(base_y):
        return None

    if source == "baseline":
        # Capability/safety mirror the pretrained baseline: M2 leaves the model
        # untouched so its WildChat KL drift is exactly the baseline's (= 0)
        # and its worst-target NLL is the baseline's NLL.
        kl = base_kl
        safety_nll = -base_y
    else:
        # M6: capability/safety from the meta-trained model itself.
        post_y = _slot_mean(seed_dir / "test_results.json", "actual_worst_log_prob")
        post_kl_path = seed_dir / "kl_eval_results.json"
        if post_y is None or not post_kl_path.exists():
            return None
        kl = float(json.loads(post_kl_path.read_text()).get("single_token", float("nan")))
        if not math.isfinite(kl) or not math.isfinite(post_y):
            return None
        safety_nll = -post_y

    # Pull the actual heldout baseline forecast error from the same file the
    # non-cal seed_metrics path uses, so per-seed paired fold computation works.
    baseline_forecast_err = _slot_mean(seed_dir / "baseline_test_results.json", forecast_metric)
    if baseline_forecast_err is None or not math.isfinite(baseline_forecast_err):
        return None

    return {
        "kl": kl,
        "safety_nll": safety_nll,
        "forecast_err": float(forecast_err),
        "baseline_kl": base_kl,
        "baseline_safety_nll": -base_y,
        "baseline_forecast_err": float(baseline_forecast_err),
    }


def _run_is_complete(seed_dir: Path) -> str:
    """'' if the seed finished cleanly, else the reason to skip it.

    test_results.json is written during training, not only at the end, so its
    presence does not mean the run finished -- and a duplicate launch once
    left directories written by two processes at once. Without this
    check the figure silently averages in truncated runs.
    """
    cfg_path = seed_dir / "config.json"
    if not cfg_path.exists():
        return "no config.json"
    if not (seed_dir / "final" / "adapter_model.safetensors").exists():
        return "no final/ adapter (run did not finish)"
    cfg = json.loads(cfg_path.read_text())
    want = int(cfg.get("steps") or 0)
    hist = seed_dir / "history.json"
    n = len(json.loads(hist.read_text())) if hist.exists() else 0
    if want and n != want:
        return f"history {n} != configured steps {want}"
    return ""


def _collect_method(spec: MethodSpec, results_root: Path, forecast_metric: str) -> list[dict]:
    points: list[dict] = []
    for run_root in sorted(results_root.glob(spec.glob_pat)):
        for seed_dir in sorted(run_root.glob("seed_*")):
            reason = _run_is_complete(seed_dir)
            if reason:
                print(f"  [skip] {run_root.name}/{seed_dir.name}: {reason}", file=sys.stderr)
                continue
            if spec.calibration_source:
                m = _calibration_seed_metrics(
                    seed_dir, "shift_scale_2param", spec.calibration_source, forecast_metric,
                )
            else:
                m = _seed_metrics(seed_dir, forecast_metric)
            if m is None:
                continue
            m["label"] = f"{run_root.name}/{seed_dir.name}"
            points.append(m)
    return points


def _baseline_values(method_points: dict[str, list[dict]]) -> dict[str, list[float]]:
    kls, safeties, fc_errs = [], [], []
    for pts in method_points.values():
        for p in pts:
            kls.append(p["baseline_kl"])
            safeties.append(p["baseline_safety_nll"])
            fc_errs.append(p["baseline_forecast_err"])
    return {"kl": kls, "safety_nll": safeties, "forecast_err": fc_errs}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Three normalized per-metric bar panels for the LM mod-bulk + GCG comparison."
    )
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--methods-preset", default="may2026", choices=sorted(PRESETS),
        help="Which run set to plot. 'may2026' = the modbulk method comparison; "
             "'strictv2' = the warm-start-GCG strict-v2 bank (Ours / Ours+cal only).",
    )
    parser.add_argument("--output", default=str(FIGURES_ROOT / "lm_three_metric_bars.pdf"))
    parser.add_argument("--csv-output", default=None)
    parser.add_argument(
        "--forecast-metric",
        default="forecast_sq_log10_error",
        choices=[
            "forecast_sq_log10_error",
            "extrapolated_weighted_mean_sq_log10_error",
            "forecast_abs_log10_error",
        ],
        help="Per-slot forecast-error key. Default is the worst-rank single-point "
             "metric so it lines up with M2 cal-baseline's calibration target.",
    )
    parser.add_argument(
        "--kl-baseline-floor", type=float, default=1e-4,
        help="Floor for baseline WildChat KL mean before computing the ratio (baseline KL ~= 0).",
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    methods, groups = PRESETS[args.methods_preset]

    method_points: dict[str, list[dict]] = {}
    for spec in methods:
        method_points[spec.name] = _collect_method(spec, results_root, args.forecast_metric)

    baseline_vals = _baseline_values(method_points)

    method_names = [m.name for m in methods]
    method_colors = [m.color for m in methods]

    # Per-method per-seed processed values for the LM panels.
    # Capability: absolute WildChat KL (no normalization). Pretrained baseline = 0
    # so we drop its bar; cal-baseline KL is also 0 -> floor marker.
    cap_values: list[list[float]] = [
        [p["kl"] for p in method_points[m.name]] for m in methods
    ]
    # Safety: decades of leak-prob reduction = (NLL_method - NLL_baseline) / ln(10)
    # per seed. Pretrained baseline at 0 by construction. Cal-baseline = 0
    # per seed (model unchanged), which collapses its dot scatter cleanly.
    log10 = math.log(10.0)
    safety_values: list[list[float]] = [
        [(p["safety_nll"] - p["baseline_safety_nll"]) / log10
         for p in method_points[m.name]]
        for m in methods
    ]
    cal_baseline_idx = next(
        (i for i, m in enumerate(methods) if m.calibration_source == "baseline"),
        -1,
    )

    metrics = [
        Metric(
            title="Capability drift",
            raw_label="WildChat KL drift  ($\\downarrow$)",
            higher_is_better=False,  # unused when raw_values=True
            method_values=cap_values,
            baseline_values=baseline_vals["kl"],
            log_y=True,
            raw_values=True,
            reference_value=None,
            bar_bottom=3e-3,
            skip_baseline_bar=True,
            skip_method_indices=(cal_baseline_idx,) if cal_baseline_idx >= 0 else (),
            y_min=3e-3,
        ),
        Metric(
            title="Safety",
            raw_label="Leak-prob reduction (decades)  ($\\uparrow$)",
            higher_is_better=True,
            method_values=safety_values,
            baseline_values=baseline_vals["safety_nll"],
            log_y=False,
            raw_values=True,
            reference_value=0.0,
            bar_bottom=0.0,
            skip_scatter_method_indices=(cal_baseline_idx,) if cal_baseline_idx >= 0 else (),
        ),
        Metric(
            title="Forecast precision",
            raw_label="Improvement factor  ($\\uparrow$)",
            higher_is_better=False,
            method_values=[[p["forecast_err"] for p in method_points[m.name]] for m in methods],
            baseline_values=baseline_vals["forecast_err"],
            method_baselines=[[p["baseline_forecast_err"] for p in method_points[m.name]] for m in methods],
            y_min=1.0,
        ),
    ]

    render_three_metric_figure(
        method_names=method_names,
        method_colors=method_colors,
        metrics=metrics,
        groups=groups,
        output_path=args.output,
        figsize=(7.0, 2.3),
        include_baseline_slot=False,
        show_scatter=False,
    )

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "method", "run_seed",
                "wildchat_kl", "safety_nll", "forecast_err",
                "baseline_wildchat_kl", "baseline_safety_nll", "baseline_forecast_err",
            ])
            for spec in methods:
                for p in method_points[spec.name]:
                    writer.writerow([
                        spec.name.replace("\n", " "), p["label"],
                        p["kl"], p["safety_nll"], p["forecast_err"],
                        p["baseline_kl"], p["baseline_safety_nll"], p["baseline_forecast_err"],
                    ])
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
