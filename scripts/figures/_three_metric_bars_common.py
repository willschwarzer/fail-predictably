"""Shared helpers for the LM and RL three-metric bar charts.

Each "metric" describes one panel in the figure: a list of values per method,
a label, and a direction (whether higher or lower raw values are better).
The renderer normalizes every panel to its pretrained-baseline mean so that
each panel reads as "fold-improvement over baseline" with the baseline anchored
at y=1 on a log scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def apply_paper_style() -> None:
    """Apply a publication-style seaborn whitegrid base + Times-family serif.

    Nimbus Roman is the URW++ free Times clone bundled with most TeX
    distributions, so it matches the body font (`\\usepackage{times}`) the
    paper uses. STIX is requested for math so arrows / sub/superscripts share
    the serif weight.
    """
    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 9,
        # Embed TrueType fonts in PDF/PS so text stays selectable and complies
        # with the standard "no Type 3" requirement of ICML / NeurIPS.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


@dataclass
class Metric:
    title: str
    raw_label: str
    higher_is_better: bool
    method_values: list[list[float]]
    baseline_values: list[float]
    baseline_floor: float = 0.0
    log_y: bool = True
    # If True, method_values are already-processed plot values (raw absolute,
    # additive offsets, etc.) and the render does no fold normalization.
    raw_values: bool = False
    # Where to draw the baseline reference line (None = no line).
    reference_value: float | None = 1.0
    # Bars are drawn from this y up; defaults to reference_value if None.
    bar_bottom: float | None = None
    # If True, do not draw the leftmost pretrained-baseline bar in this panel.
    # Used when "baseline" has no meaningful position (e.g., LM KL = 0 on log y).
    skip_baseline_bar: bool = False
    # Indices into method_values to suppress dot scatter for. Used for
    # methods whose per-seed values mirror the baseline (e.g., cal-baseline in
    # capability/safety panels) where dot spread would mislead.
    skip_scatter_method_indices: tuple[int, ...] = ()
    # Indices into method_values to suppress entirely (no bar, no marker, no
    # scatter). Used for methods whose value is exactly the reference and
    # would otherwise render as a misleading floor marker on log axes.
    skip_method_indices: tuple[int, ...] = ()
    # Hard ylim bottom; None lets matplotlib auto-scale.
    y_min: float | None = None
    # Optional per-method per-seed baselines, aligned 1:1 with method_values.
    # When provided, fold computation pairs each seed's method value with its
    # own seed's baseline (per-seed paired fold). When None, falls back to
    # using a single cross-seed baseline mean for all seeds. Per-seed pairing
    # is the natural choice for designs where each method's seed has its own
    # paired baseline (e.g., Cal-baseline computed against the same baseline
    # that Ours and Ours+cal use per seed).
    method_baselines: list[list[float]] | None = None


def compute_bar_positions(
    groups: list[tuple[str, int, int, str]],
    *,
    gap: float = 0.6,
) -> list[float]:
    """Map original integer bar indices to physically-grouped x positions.

    Bars within a group sit at consecutive integers; an extra `gap` of empty
    space is inserted between adjacent groups.
    """
    by_index: dict[int, float] = {}
    cursor = 0.0
    sorted_groups = sorted(groups, key=lambda g: g[1])
    for gi, (_, lo, hi, _) in enumerate(sorted_groups):
        if gi > 0:
            cursor += gap
        for idx in range(lo, hi + 1):
            by_index[idx] = cursor
            cursor += 1.0
    n_bars = max(by_index.keys()) + 1
    return [by_index[i] for i in range(n_bars)]


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


def fold_improvement(method_value: float, baseline_mean: float, *, higher_is_better: bool) -> float:
    if not math.isfinite(method_value) or not math.isfinite(baseline_mean) or baseline_mean == 0:
        return float("nan")
    if higher_is_better:
        return method_value / baseline_mean
    if method_value == 0:
        return float("inf")
    return baseline_mean / method_value


def _draw_groups(
    ax: plt.Axes,
    groups: list[tuple[str, int, int, str]],
    bar_positions: list[float],
    *,
    span_pad: float = 0.45,
) -> None:
    """Shade group background only for groups whose `bg` color is truthy.
    Group text labels are no longer drawn; group identity is conveyed by
    the shaded background of the highlighted (e.g. "Ours") group alone."""
    for _label, lo, hi, bg in groups:
        if not bg:
            continue
        x_lo = bar_positions[lo] - span_pad
        x_hi = bar_positions[hi] + span_pad
        ax.axvspan(x_lo, x_hi, color=bg, alpha=0.55, zorder=0)


def _bar_with_floor(
    ax: plt.Axes,
    *,
    x: float,
    value: float,
    err: float,
    color: str,
    bottom: float,
    log_y: bool = True,
) -> None:
    """Draw one bar from y=bottom to y=value. On a log axis, non-positive
    values cannot be plotted, so they're shown as a small "x" marker instead.
    On a linear axis, negative or zero values are plotted directly."""
    if not math.isfinite(value):
        bar_height = bottom * 0.6 if log_y else 0.5
        ax.bar(
            [x], [bar_height], bottom=bottom, color="white",
            edgecolor=color, linewidth=1.0, hatch="//", alpha=0.4,
        )
        ax.text(x, bottom, "pending", rotation=90, ha="center", va="bottom",
                fontsize=7.5, color="#666666")
        return
    if log_y and value <= 0:
        ax.scatter(
            [x], [bottom], marker="x", s=80, color=color, zorder=5,
        )
        return
    ax.bar(
        [x], [value - bottom], bottom=bottom,
        color=color, edgecolor="none", linewidth=0,
    )
    if err > 0:
        if log_y:
            lo = max(value - err, 1e-12)
            hi = value + err
        else:
            lo = value - err
            hi = value + err
        ax.errorbar([x], [value], yerr=[[value - lo], [hi - value]],
                    color="#333333", linewidth=0.8, capsize=2, fmt="none", zorder=5)


def render_three_metric_figure(
    *,
    method_names: list[str],
    method_colors: list[str],
    metrics: list[Metric],
    groups: list[tuple[str, int, int, str]],
    suptitle: str | None = None,
    output_path,
    figsize: tuple[float, float] = (14.0, 6.2),
    baseline_label: str = "Pretrained\nbaseline",
    baseline_color: str = "#444444",
    include_baseline_slot: bool = True,
    show_scatter: bool = True,
) -> None:
    """Render a 1-row, len(metrics)-column figure of normalized bar panels.

    Each panel is drawn on a log y axis with the baseline at y=1. Methods that
    beat baseline draw upward; methods that underperform draw downward.

    `include_baseline_slot=False` drops the leftmost pretrained-baseline column
    entirely (no x-tick, no bar). In that case the caller's `groups` indices
    must match `method_names` directly (group 0 = first method, no "Ref" group).
    """
    apply_paper_style()
    if include_baseline_slot:
        bar_labels = [baseline_label] + list(method_names)
        bar_colors = [baseline_color] + list(method_colors)
    else:
        bar_labels = list(method_names)
        bar_colors = list(method_colors)
    bar_positions = compute_bar_positions(groups, gap=0.7)

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        skip_scatter = set(metric.skip_scatter_method_indices)
        bar_bottom = metric.bar_bottom if metric.bar_bottom is not None else (
            metric.reference_value if metric.reference_value is not None else 0.0
        )

        if metric.raw_values:
            # method_values are already in plot-units; no fold computation.
            method_plot: list[list[float]] = [
                [float(v) for v in vlist if math.isfinite(v)]
                for vlist in metric.method_values
            ]
        else:
            method_plot = []
            if metric.method_baselines is not None:
                # Per-seed paired fold: fold = (baseline_per_seed / method_per_seed)
                # for higher_is_better=False, inverse for higher_is_better=True.
                if len(metric.method_baselines) != len(metric.method_values):
                    raise ValueError(
                        "method_baselines length must match method_values length"
                    )
                for vlist, blist in zip(metric.method_values, metric.method_baselines):
                    if len(vlist) != len(blist):
                        raise ValueError(
                            "per-method baselines must align 1:1 with method values"
                        )
                    method_plot.append([
                        fold_improvement(v, b, higher_is_better=metric.higher_is_better)
                        for v, b in zip(vlist, blist)
                        if math.isfinite(v) and math.isfinite(b) and b != 0
                        and (b >= metric.baseline_floor if metric.baseline_floor > 0 else True)
                    ])
            else:
                if metric.baseline_floor > 0:
                    baseline_clamped = [max(v, metric.baseline_floor)
                                        for v in metric.baseline_values if math.isfinite(v)]
                else:
                    baseline_clamped = [v for v in metric.baseline_values
                                        if math.isfinite(v) and v > 0]
                b_mean, _ = mean_sem(baseline_clamped if baseline_clamped else metric.baseline_values)
                if not math.isfinite(b_mean) or b_mean == 0:
                    b_mean = metric.baseline_floor or 1.0
                for vlist in metric.method_values:
                    method_plot.append([
                        fold_improvement(v, b_mean, higher_is_better=metric.higher_is_better)
                        for v in vlist if math.isfinite(v)
                    ])

        _draw_groups(ax, groups, bar_positions)

        method_start = 1 if include_baseline_slot else 0
        if include_baseline_slot and not metric.skip_baseline_bar and metric.reference_value is not None:
            _bar_with_floor(ax, x=bar_positions[0], value=metric.reference_value,
                            err=0.0, color=bar_colors[0], bottom=bar_bottom,
                            log_y=metric.log_y)
        skip_method = set(metric.skip_method_indices)
        for i, vals in enumerate(method_plot, start=method_start):
            method_idx = i - method_start
            if method_idx in skip_method:
                continue
            m, s = mean_sem(vals)
            _bar_with_floor(ax, x=bar_positions[i], value=m, err=s,
                            color=bar_colors[i], bottom=bar_bottom,
                            log_y=metric.log_y)

            if not show_scatter or i in skip_scatter:
                continue
            if metric.log_y:
                finite = [v for v in vals if math.isfinite(v) and v > 0]
            else:
                finite = [v for v in vals if math.isfinite(v)]
            if finite:
                jitter = (np.random.RandomState(i + 1).rand(len(finite)) - 0.5) * 0.32
                ax.scatter(
                    np.full(len(finite), bar_positions[i]) + jitter, finite,
                    color="#222222", s=12, alpha=0.55, zorder=4, edgecolors="none",
                )

        if metric.reference_value is not None:
            ax.axhline(metric.reference_value, color="#666666", linewidth=1.0,
                       linestyle="--", alpha=0.85)
        if metric.log_y:
            ax.set_yscale("log")
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_xlim(min(bar_positions) - 0.7, max(bar_positions) + 0.7)
        if metric.y_min is not None:
            ax.set_ylim(bottom=metric.y_min)
        ax.set_title(metric.title, fontsize=11, pad=4)
        ax.set_ylabel(metric.raw_label, fontsize=9)
        # Slightly stronger bottom (x) axis than the rest, to anchor the bars.
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
    else:
        fig.tight_layout()

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
