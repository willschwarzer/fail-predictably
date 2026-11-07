"""Three-panel per-method bar chart for the RL bipolar (sev01) comparison.

Each panel is normalized to its pretrained-baseline mean so that y=1 is the
no-meta-training reference and bars rise above 1 when the method improves on
the pretrained checkpoint. y is on a log scale; a bar at y=10 means 10x
improvement, a bar at y=0.1 means 10x degradation.

  Panel 1  Capability preservation     method_return / baseline_return     (up = better)
  Panel 2  Safety                       baseline_wr / method_wr             (up = safer)
  Panel 3  Forecast precision           baseline_err / method_err           (up = better)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    # "" -> regular method; "baseline" -> M2; "metatrained" -> M6 (cal applied
    # to the meta-trained model itself).
    calibration_source: str = ""
    # SFT runs in this codebase store post-SFT values in BOTH pretrained and
    # meta_learned slots, so their seed_dir.baseline data is corrupted. Mark
    # them with baseline_is_canonical=False to exclude from baseline aggregation.
    baseline_is_canonical: bool = True

    @property
    def is_calibration(self) -> bool:
        return bool(self.calibration_source)


DEFAULT_METHODS: list[MethodSpec] = [
    MethodSpec(
        "Cal.",
        "rlquant_trap_report_bipolar_sev01_metatrain_10seeds_canonicalhparams_*",
        "#EFCEC2",
        calibration_source="baseline",
    ),
    MethodSpec(
        "RL",
        "rlquant_trap_report_bipolar_sev01_baseline_sft_union_10seeds_*",
        "#E5B3A6",
        baseline_is_canonical=False,
    ),
    MethodSpec(
        "RL + cal.",
        "rlquant_trap_report_bipolar_sev01_baseline_sft_union_10seeds_*",
        "#A65D49",
        calibration_source="metatrained",
        baseline_is_canonical=False,
    ),
    # Removed for now: SFT-on-deploy isn't a meaningful "oracle" -- it's just
    # SFT on half the per-target prompts, and (fit, deploy) are random splits
    # so the deploy half doesn't preferentially target worst-rank. Re-add by
    # uncommenting; also widen DEFAULT_GROUPS["Baselines"] back to (1, 5) and
    # shift "Ours" to (6, 7).
    # MethodSpec(
    #     "SFT (deploy)",
    #     "rlquant_trap_report_bipolar_sev01_baseline_sft_deploy_10seeds_*",
    #     "#B59FC2",
    #     baseline_is_canonical=False,
    # ),
    # MethodSpec(
    #     "SFT (deploy) + cal.",
    #     "rlquant_trap_report_bipolar_sev01_baseline_sft_deploy_10seeds_*",
    #     "#6E548A",
    #     calibration_source="metatrained",
    #     baseline_is_canonical=False,
    # ),
    # "[0-9]*seeds" matches the 10-seed and 20-more-seeds fleets that the paper's
    # figure pools (30 seeds), and not the 600-update fleets of the sampled-returns
    # appendix (..._cache296_m600_10seeds_*, ..._m600ext_*), which share the prefix.
    MethodSpec(
        "Ours",
        "rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_[0-9]*seeds_*",
        "#9DB8D6",
    ),
    MethodSpec(
        "Ours + cal.",
        "rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_[0-9]*seeds_*",
        "#3D5A80",
        calibration_source="metatrained",
    ),
]


# Same method roster on the blind-dynamics (dynwind25) bank; only the run
# globs differ. SFT-on-deploy stays excluded for the same reason as bipolar.
DYNWIND_METHODS: list[MethodSpec] = [
    MethodSpec(
        "Cal.",
        "rlquant_trap_report_dynwind25_metatrain_iog_full_bootstrap1_cache296_*seeds_*",
        "#EFCEC2",
        calibration_source="baseline",
    ),
    MethodSpec(
        "RL",
        "rlquant_trap_report_dynwind25_baseline_sft_union_10seeds_*",
        "#E5B3A6",
        baseline_is_canonical=False,
    ),
    MethodSpec(
        "RL + cal.",
        "rlquant_trap_report_dynwind25_baseline_sft_union_10seeds_*",
        "#A65D49",
        calibration_source="metatrained",
        baseline_is_canonical=False,
    ),
    MethodSpec(
        "Ours",
        "rlquant_trap_report_dynwind25_metatrain_iog_full_bootstrap1_cache296_*seeds_*",
        "#9DB8D6",
    ),
    MethodSpec(
        "Ours + cal.",
        "rlquant_trap_report_dynwind25_metatrain_iog_full_bootstrap1_cache296_*seeds_*",
        "#3D5A80",
        calibration_source="metatrained",
    ),
]

# Rebuttal suite: exact-path canonical vs the sampling-based (REINFORCE)
# training arms at n=1 and n=8 rollouts per task, plus SFT-REINFORCE at
# matched n. All bars remain normalized to each run's own pretrained
# baseline; evaluation is exact everywhere.
REINFORCE_METHODS: list[MethodSpec] = [
    MethodSpec(
        "Ours (exact)",
        "rlquant_trap_report_bipolar_sev01_iog_full_bootstrap1_cache296_*seeds_*",
        "#3D5A80",
    ),
    MethodSpec(
        "RL n=1",
        "rlquant_trap_report_bipolar_sev01_baseline_sft_union_reinforce_n1_10seeds_*",
        "#E5B3A6",
        baseline_is_canonical=False,
    ),
    MethodSpec(
        "RL n=8",
        "rlquant_trap_report_bipolar_sev01_baseline_sft_union_reinforce_n8_10seeds_*",
        "#A65D49",
        baseline_is_canonical=False,
    ),
    MethodSpec(
        "Ours A n=1",
        "rlquant_trap_report_bipolar_sev01_reinforce_plugin_n1_lr*_10seeds_*",
        "#9DB8D6",
    ),
    MethodSpec(
        "Ours A n=8",
        "rlquant_trap_report_bipolar_sev01_reinforce_plugin_n8_lr*_10seeds_*",
        "#5F7FA8",
    ),
    MethodSpec(
        "Ours B n=1",
        "rlquant_trap_report_bipolar_sev01_reinforce_batch_n1_lr*_10seeds_*",
        "#B59FC2",
    ),
    MethodSpec(
        "Ours B n=8",
        "rlquant_trap_report_bipolar_sev01_reinforce_batch_n8_lr*_10seeds_*",
        "#6E548A",
    ),
]

METHOD_SUITES: dict[str, list[MethodSpec]] = {
    "bipolar_sev01": DEFAULT_METHODS,
    "dynwind25": DYNWIND_METHODS,
    "bipolar_reinforce": REINFORCE_METHODS,
}

# Group indices align with the method list (no leftmost pretrained-baseline
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

REINFORCE_GROUPS: list[tuple[str, int, int, str]] = [
    ("Exact", 0, 0, ""),
    ("RL (sampled)", 1, 2, ""),
    ("Ours (sampled)", 3, 6, "#FFE9C2"),
]

GROUPS_BY_SUITE: dict[str, list[tuple[str, int, int, str]]] = {
    "bipolar_sev01": DEFAULT_GROUPS,
    "dynwind25": DEFAULT_GROUPS,
    "bipolar_reinforce": REINFORCE_GROUPS,
}


def _seed_metrics(seed_dir: Path) -> dict | None:
    p = seed_dir / "final_results.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    try:
        ml = d["meta_learned"]
        pre = d["pretrained"]
        post_ret = float(ml["split_metrics"]["pretrain_train"]["mean_return"])
        post_wr = float(ml["forecast_pairs"]["heldout"]["actual_worst_regret"])
        post_fc = float(ml["forecast_pairs"]["heldout"]["forecast_sq_error"])
        base_ret = float(pre["split_metrics"]["pretrain_train"]["mean_return"])
        base_wr = float(pre["forecast_pairs"]["heldout"]["actual_worst_regret"])
        base_fc = float(pre["forecast_pairs"]["heldout"]["forecast_sq_error"])
    except (KeyError, ValueError, TypeError):
        return None
    if any(not math.isfinite(v) for v in (post_ret, post_wr, post_fc, base_ret, base_wr, base_fc)):
        return None
    return {
        "preservation": post_ret,
        "worst_regret": post_wr,
        "forecast_err": post_fc,
        "baseline_preservation": base_ret,
        "baseline_worst_regret": base_wr,
        "baseline_forecast_err": base_fc,
    }


def _calibration_seed_metrics(
    seed_dir: Path, calib_variant: str, source: str,
) -> dict | None:
    """Per-seed record for an ex-post calibration method.

    `source`:
      - "baseline":   M2; capability/safety from the pretrained checkpoint.
      - "metatrained": M6; capability/safety from the meta-trained model.
    Forecast value comes from the matching posthoc_calibration_*.json.
    """
    # Two filename conventions: the original bipolar analysis wrote
    # posthoc_calibration_{baseline,metatrained}.json; the slip/dynwind
    # analysis passes write posthoc_calibration_{pretrained,meta_learned}.json
    # (named by the traces cohort). Accept either.
    if source == "baseline":
        candidates = ["posthoc_calibration_baseline.json", "posthoc_calibration_pretrained.json"]
    else:
        candidates = ["posthoc_calibration_metatrained.json", "posthoc_calibration_meta_learned.json"]
    cal_path = next((seed_dir / name for name in candidates if (seed_dir / name).exists()), None)
    if cal_path is None:
        return None
    try:
        d = json.loads(cal_path.read_text())
    except json.JSONDecodeError:
        return None
    summary = d.get("heldout_calibrated", {}).get(f"heldout_summary_{calib_variant}")
    if not summary:
        return None
    forecast_err = summary.get("mean_sq_error")
    if forecast_err is None or not math.isfinite(float(forecast_err)):
        return None

    fp = seed_dir / "final_results.json"
    if not fp.exists():
        return None
    try:
        d2 = json.loads(fp.read_text())
        pre = d2["pretrained"]
        ml = d2["meta_learned"]
        base_ret = float(pre["split_metrics"]["pretrain_train"]["mean_return"])
        base_wr = float(pre["forecast_pairs"]["heldout"]["actual_worst_regret"])
        base_fc = float(pre["forecast_pairs"]["heldout"]["forecast_sq_error"])
        post_ret = float(ml["split_metrics"]["pretrain_train"]["mean_return"])
        post_wr = float(ml["forecast_pairs"]["heldout"]["actual_worst_regret"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if any(not math.isfinite(v) for v in (base_ret, base_wr, base_fc, post_ret, post_wr)):
        return None

    if source == "baseline":
        preservation = base_ret
        worst_regret = base_wr
    else:
        preservation = post_ret
        worst_regret = post_wr

    return {
        "preservation": preservation,
        "worst_regret": worst_regret,
        "forecast_err": float(forecast_err),
        "baseline_preservation": base_ret,
        "baseline_worst_regret": base_wr,
        "baseline_forecast_err": float(base_fc),
    }


def _collect_method(spec: MethodSpec, results_root: Path) -> list[dict]:
    points: list[dict] = []
    for run_root in sorted(results_root.glob(spec.glob_pat)):
        for seed_dir in sorted(run_root.glob("seed_*")):
            if spec.is_calibration:
                m = _calibration_seed_metrics(
                    seed_dir, "shift_scale_2param", spec.calibration_source,
                )
            else:
                m = _seed_metrics(seed_dir)
            if m is None:
                continue
            m["label"] = f"{run_root.name}/{seed_dir.name}"
            points.append(m)
    return points


def _baseline_values(
    method_points: dict[str, list[dict]],
    methods: list[MethodSpec],
) -> dict[str, list[float]]:
    """Aggregate per-seed baseline values, but only from methods whose seed_dir
    baseline data is canonical (i.e., the actual pretrained checkpoint, not a
    post-SFT contamination)."""
    pres, wrs, fcs = [], [], []
    canonical = {m.name for m in methods if m.baseline_is_canonical and not m.is_calibration}
    for name, pts in method_points.items():
        if name not in canonical:
            continue
        for p in pts:
            pres.append(p["baseline_preservation"])
            wrs.append(p["baseline_worst_regret"])
            fcs.append(p["baseline_forecast_err"])
    return {"preservation": pres, "worst_regret": wrs, "forecast_err": fcs}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Three normalized per-metric bar panels for the RL bipolar sev01 comparison."
    )
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--output", default=str(FIGURES_ROOT / "rl_three_metric_bars.pdf"))
    parser.add_argument("--csv-output", default=None)
    parser.add_argument(
        "--suite", choices=sorted(METHOD_SUITES), default="bipolar_sev01",
        help="Which bank's run roster to plot (method labels/colors are shared).",
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    methods = METHOD_SUITES[args.suite]

    method_points: dict[str, list[dict]] = {}
    for spec in methods:
        method_points[spec.name] = _collect_method(spec, results_root)

    # Substitute canonical baselines into non-canonical-baseline runs (e.g., SFT).
    # SFT in this codebase trains during the "pretrain" phase and skips the meta
    # phase, so its run-local pretrained block is not the actual baseline -- it
    # is the SFT model's eval. The canonical Ours runs DO produce a real
    # pretrained baseline per seed, so we look that up by seed number and use it
    # as the baseline for the SFT-style runs.
    import re as _re
    canonical_by_seed: dict[int, dict] = {}
    canonical_names = {m.name for m in methods if m.baseline_is_canonical and not m.is_calibration}
    for name, pts in method_points.items():
        if name not in canonical_names:
            continue
        for p in pts:
            mo = _re.search(r"seed_(\d+)", p["label"])
            if not mo:
                continue
            seed = int(mo.group(1))
            canonical_by_seed.setdefault(seed, {
                "pres": p["baseline_preservation"],
                "wr": p["baseline_worst_regret"],
                "fc": p["baseline_forecast_err"],
            })
    for spec in methods:
        if spec.baseline_is_canonical:
            continue
        kept = []
        for p in method_points[spec.name]:
            mo = _re.search(r"seed_(\d+)", p["label"])
            if not mo:
                continue
            seed = int(mo.group(1))
            ref = canonical_by_seed.get(seed)
            if ref is None:
                continue  # no canonical match; drop this seed for this method
            p["baseline_preservation"] = ref["pres"]
            p["baseline_worst_regret"] = ref["wr"]
            p["baseline_forecast_err"] = ref["fc"]
            kept.append(p)
        method_points[spec.name] = kept

    baseline_vals = _baseline_values(method_points, methods)

    # Cal-on-baseline (M2) leaves the model untouched: its capability and
    # safety values are by construction identical to that seed's pretrained
    # baseline. Mirror per-seed so the per-seed paired fold collapses to
    # exactly 1.0 with zero variance. (Forecast error stays as the calibrated
    # value, since calibration *does* change forecasts.)
    for spec in methods:
        if spec.calibration_source == "baseline":
            for p in method_points[spec.name]:
                p["preservation"] = p["baseline_preservation"]
                p["worst_regret"] = p["baseline_worst_regret"]

    method_names = [m.name for m in methods]
    method_colors = [m.color for m in methods]

    cal_baseline_idx = next(
        (i for i, m in enumerate(methods) if m.calibration_source == "baseline"),
        -1,
    )
    cal_skip = (cal_baseline_idx,) if cal_baseline_idx >= 0 else ()

    # Per-method per-seed baselines for per-seed paired fold across all panels.
    pres_method_baselines = [[p["baseline_preservation"] for p in method_points[m.name]] for m in methods]
    wr_method_baselines = [[p["baseline_worst_regret"] for p in method_points[m.name]] for m in methods]
    fc_method_baselines = [[p["baseline_forecast_err"] for p in method_points[m.name]] for m in methods]
    metrics = [
        Metric(
            title="Capability preservation",
            raw_label="Improvement factor  ($\\uparrow$)",
            higher_is_better=True,
            method_values=[[p["preservation"] for p in method_points[m.name]] for m in methods],
            baseline_values=baseline_vals["preservation"],
            method_baselines=pres_method_baselines,
            log_y=False,
            skip_scatter_method_indices=cal_skip,
        ),
        Metric(
            title="Safety",
            raw_label="Improvement factor  ($\\uparrow$)",
            higher_is_better=False,
            method_values=[[p["worst_regret"] for p in method_points[m.name]] for m in methods],
            baseline_values=baseline_vals["worst_regret"],
            method_baselines=wr_method_baselines,
            log_y=False,
            skip_scatter_method_indices=cal_skip,
        ),
        Metric(
            title="Forecast precision",
            raw_label="Improvement factor  ($\\uparrow$)",
            higher_is_better=False,
            method_values=[[p["forecast_err"] for p in method_points[m.name]] for m in methods],
            baseline_values=baseline_vals["forecast_err"],
            method_baselines=fc_method_baselines,
            log_y=True,
            y_min=1.0,
        ),
    ]

    render_three_metric_figure(
        method_names=method_names,
        method_colors=method_colors,
        metrics=metrics,
        groups=GROUPS_BY_SUITE.get(args.suite, DEFAULT_GROUPS),
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
                "pretrain_mean_return", "heldout_worst_regret", "heldout_forecast_err",
                "baseline_pretrain_mean_return", "baseline_heldout_worst_regret",
                "baseline_heldout_forecast_err",
            ])
            for spec in methods:
                for p in method_points[spec.name]:
                    writer.writerow([
                        spec.name.replace("\n", " "), p["label"],
                        p["preservation"], p["worst_regret"], p["forecast_err"],
                        p["baseline_preservation"], p["baseline_worst_regret"],
                        p["baseline_forecast_err"],
                    ])
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
