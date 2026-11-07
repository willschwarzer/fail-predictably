"""Smooth contour version of the 2D rank-bias surface b_inv_{k, log R}.

A Monte Carlo of the rank term on a dense grid of (k, R) values, rendered as a
filled contour plot with explicit contour lines. Its JSON feeds the right panel
of theory_combined_figure.py; the paper used --n-trials 200000 --chunk 20000
--n-k 35 --n-r 35.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", REPO_ROOT / "artifacts" / "figures"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=300_000)
    parser.add_argument("--chunk", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=1000)
    parser.add_argument("--n-k", type=int, default=40)
    parser.add_argument("--n-r", type=int, default=40)
    parser.add_argument("--log-k", action="store_true", default=True)
    parser.add_argument(
        "--out-pdf",
        type=str,
        default=str(FIGURES_ROOT / "theory_rank_bias_2d_smooth.pdf"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(RESULTS_ROOT / "theory_rank_bias_2d_smooth.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.log_k:
        ks_raw = np.exp(np.linspace(np.log(args.k_min), np.log(args.k_max), args.n_k))
        ks = np.array(sorted(set(int(round(x)) for x in ks_raw)), dtype=int)
    else:
        ks = np.arange(args.k_min, args.k_max + 1, dtype=int)
    log_Rs = np.linspace(np.log(2.0), np.log(10000.0), args.n_r)
    Rs = np.exp(log_Rs)
    print(f"k grid: {len(ks)} values from {ks.min()} to {ks.max()}", flush=True)

    sums = np.zeros((len(ks), len(log_Rs)))
    n_total = 0

    k_max = int(ks.max())
    for chunk_start in range(0, args.n_trials, args.chunk):
        c = min(args.chunk, args.n_trials - chunk_start)
        E_fit = rng.exponential(scale=1.0, size=(c, k_max))
        Gamma = np.cumsum(E_fit, axis=1)
        T = -np.log(Gamma)
        E_dep = rng.exponential(scale=1.0, size=c)
        zeta = -np.log(E_dep)

        for ki, k in enumerate(ks):
            T_k = T[:, :k]
            a = -np.log(np.arange(1, k + 1, dtype=float))
            a_bar = a.mean()
            t_bar = T_k.mean(axis=1)
            centered_t = T_k - t_bar[:, None]
            A = ((a - a_bar)[None, :] * centered_t).sum(axis=1)
            B = (centered_t ** 2).sum(axis=1)

            for ri, r in enumerate(log_Rs):
                I_r = t_bar + (r - a_bar) * B / A
                sums[ki, ri] += (I_r - r - zeta).sum()

        n_total += c
        print(f"  {n_total}/{args.n_trials}", flush=True)

    means = sums / n_total

    # ---- Plot ----
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    log_R_grid, k_grid = np.meshgrid(log_Rs, ks)
    data_min = float(means.min())
    data_max = float(means.max())
    # Symmetric color scaling: |bias| controls intensity on both sides, so the
    # small-magnitude negative regime renders pale rather than oversold.
    vmax_abs = max(abs(data_min), abs(data_max))
    norm = mcolors.Normalize(vmin=-vmax_abs, vmax=vmax_abs)
    fill_levels = np.linspace(data_min, data_max, 200)
    cf = ax.contourf(
        log_R_grid, k_grid, means, levels=fill_levels,
        cmap="RdBu_r", norm=norm, extend="neither",
    )

    line_levels_all = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    line_levels = [v for v in line_levels_all if data_min < v < data_max]
    linewidths_per_level = [1.4 if v == 0 else 0.6 for v in line_levels]
    cs = ax.contour(
        log_R_grid, k_grid, means, levels=line_levels,
        colors="black", linewidths=linewidths_per_level,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    R_ticks = [2, 5, 10, 30, 100, 300, 1000, 3000, 10000]
    ax.set_xticks([np.log(R) for R in R_ticks])
    ax.set_xticklabels([str(R) for R in R_ticks])
    if args.log_k:
        ax.set_yscale("log")
        k_ticks = [5, 10, 20, 50, 100, 200, 500, 1000]
        k_ticks = [k for k in k_ticks if ks.min() <= k <= ks.max()]
        ax.set_yticks(k_ticks)
        ax.set_yticklabels([str(k) for k in k_ticks])
    ax.set_xlabel(r"deployment ratio $R = N/m$")
    ax.set_ylabel(r"top-$k$ count")

    # Mark Jones et al. default with a star -- legend goes below the axes,
    # so the in-axes annotation cannot be confused with a contour line.
    star_handle, = ax.plot(
        np.log(100), 10, marker="*", color="black",
        markersize=20, markeredgecolor="white", markeredgewidth=1.5,
        linestyle="None", zorder=5,
        label="Jones et al., Fig. 1 ($k=10$, $R=100$)",
    )
    ax.legend(
        handles=[star_handle],
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        fontsize=9, framealpha=0.95,
    )

    cbar = plt.colorbar(cf, ax=ax)
    # Clip the visible colorbar to the data range so unused colormap tail
    # (between -vmax_abs and data_min) is not rendered.
    cbar.ax.set_ylim(data_min, data_max)
    int_ticks = [t for t in range(-10, 11) if data_min <= t <= data_max]
    cbar.set_ticks(int_ticks)
    cbar.set_ticklabels([str(t) for t in int_ticks])
    cbar.set_label(r"rank-term overprediction (units of $q^\prime(y)$)")

    fig.tight_layout()
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    print(f"\nWrote {out_pdf}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(
            {
                "config": vars(args),
                "ks": ks.tolist(),
                "Rs": Rs.tolist(),
                "log_Rs": log_Rs.tolist(),
                "b_inv_mean": means.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
