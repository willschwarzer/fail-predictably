"""Combined two-panel theory figure: decomposition + 2D rank-bias surface.

Loads the precomputed results from
  $RESULTS_ROOT/theory_decomposition.json
  $RESULTS_ROOT/theory_rank_bias_2d_smooth.json
and renders both plots into a single PDF for direct comparison against the
LaTeX-subfigure version.

Inputs (both produced in the release; the paper's figure used these arguments):
  theory_decomposition.json        theory_decomposition_figure.py, default arguments
  theory_rank_bias_2d_smooth.json  theory_rank_bias_2d_smooth.py --n-trials 200000
                                   --chunk 20000 --n-k 35 --n-r 35
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", REPO_ROOT / "artifacts" / "figures"))

# Bump all default font sizes by 1pt; the colorbar label is bumped by 2pt
# explicitly below.
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decomp-json",
        type=str,
        default=str(RESULTS_ROOT / "theory_decomposition.json"),
    )
    parser.add_argument(
        "--bias2d-json",
        type=str,
        default=str(RESULTS_ROOT / "theory_rank_bias_2d_smooth.json"),
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default=str(FIGURES_ROOT / "theory_combined.pdf"),
    )
    args = parser.parse_args()

    with open(args.decomp_json) as f:
        decomp = json.load(f)
    with open(args.bias2d_json) as f:
        bias2d = json.load(f)

    rows = decomp["rows"]

    fig = plt.figure(figsize=(11.0, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)

    # --- Left panel: decomposition stacked bars -----------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    n = len(rows)
    x = np.arange(n)
    width = 0.6

    rank_vals = np.array([r["rank"] for r in rows])
    curv_vals = np.array([r["curvature"] for r in rows])
    occ_vals = np.array([-r["occupancy"] for r in rows])
    rem_vals = np.array([r["remainder"] for r in rows])
    total_vals = np.array([r["total"] for r in rows])
    total_ses = np.array([r["total_se"] for r in rows])

    color_rank = "#1f77b4"
    color_curv = "#ff7f0e"
    color_occ = "#d62728"
    color_rem = "#7f7f7f"

    contributions = [
        (rank_vals, color_rank),
        (curv_vals, color_curv),
        (occ_vals, color_occ),
        (rem_vals, color_rem),
    ]
    for i in range(n):
        pos_bottom = 0.0
        neg_bottom = 0.0
        for vals, color in contributions:
            v = vals[i]
            if v >= 0:
                ax1.bar(x[i], v, width, bottom=pos_bottom, color=color,
                        edgecolor="white", linewidth=0.5)
                pos_bottom += v
            else:
                ax1.bar(x[i], v, width, bottom=neg_bottom, color=color,
                        edgecolor="white", linewidth=0.5)
                neg_bottom += v
    ax1.errorbar(
        x, total_vals, yerr=1.96 * total_ses, fmt="D", color="black",
        markersize=7, capsize=4, zorder=10, markeredgecolor="white",
        markeredgewidth=1,
    )
    ax1.axhline(0, color="gray", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["name"] for r in rows], rotation=15, ha="right",
                        fontsize=9)
    ax1.set_ylabel(r"contribution to forecast error / $q^\prime(y)$")
    ax1.set_title("(a) decomposition across distributions", fontsize=11)
    legend_elements = [
        Patch(facecolor=color_rank, label="rank"),
        Patch(facecolor=color_curv, label="curvature"),
        Patch(facecolor=color_occ, label="occupancy"),
        Patch(facecolor=color_rem, label=r"remainder ($o_p$)"),
        Line2D([0], [0], marker="D", color="black", linestyle="None",
               markersize=6, label="empirical total"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9,
               framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Right panel: 2D rank-bias surface ----------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ks = np.array(bias2d["ks"])
    log_Rs = np.array(bias2d["log_Rs"])
    Rs = np.array(bias2d["Rs"])
    means = np.array(bias2d["b_inv_mean"])

    log_R_grid, k_grid = np.meshgrid(log_Rs, ks)
    data_min = float(means.min())
    data_max = float(means.max())
    vmax_abs = max(abs(data_min), abs(data_max))
    norm = mcolors.Normalize(vmin=-vmax_abs, vmax=vmax_abs)
    fill_levels = np.linspace(data_min, data_max, 200)
    cf = ax2.contourf(
        log_R_grid, k_grid, means, levels=fill_levels,
        cmap="RdBu_r", norm=norm, extend="neither",
    )
    line_levels_all = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                       4.0, 4.5, 5.0]
    line_levels = [v for v in line_levels_all if data_min < v < data_max]
    linewidths_per_level = [1.4 if v == 0 else 0.6 for v in line_levels]
    cs = ax2.contour(
        log_R_grid, k_grid, means, levels=line_levels,
        colors="black", linewidths=linewidths_per_level,
    )
    ax2.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    R_ticks = [2, 5, 10, 30, 100, 300, 1000, 3000, 10000]
    ax2.set_xticks([np.log(R) for R in R_ticks])
    ax2.set_xticklabels([str(R) for R in R_ticks])
    ax2.set_yscale("log")
    k_ticks = [k for k in [5, 10, 20, 50, 100, 200, 500, 1000]
               if ks.min() <= k <= ks.max()]
    ax2.set_yticks(k_ticks)
    ax2.set_yticklabels([str(k) for k in k_ticks])
    ax2.set_xlabel(r"deployment ratio $R = N/m$")
    ax2.set_ylabel(r"top-$k$ count")
    ax2.set_title("(b) rank-term overprediction surface", fontsize=11)

    star_handle, = ax2.plot(
        np.log(100), 10, marker="*", color="black",
        markersize=18, markeredgecolor="white", markeredgewidth=1.3,
        linestyle="None", zorder=5,
        label="Jones et al., Fig. 1",
    )
    ax2.legend(handles=[star_handle], loc="upper right", fontsize=9,
               framealpha=0.95)

    cbar = plt.colorbar(cf, ax=ax2)
    cbar.ax.set_ylim(data_min, data_max)
    int_ticks = [t for t in range(-10, 11) if data_min <= t <= data_max]
    cbar.set_ticks(int_ticks)
    cbar.set_ticklabels([str(t) for t in int_ticks])
    cbar.set_label(r"rank-term overprediction / $q^\prime(y)$",
                   fontsize=11)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
