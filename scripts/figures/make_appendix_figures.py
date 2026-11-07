"""Appendix figures of the paper.

Renders four PDFs into $FIGURES_ROOT (default artifacts/figures/) from saved results, with PNG previews
in $FIGURES_ROOT/build/; runs no experiments:
  natural_paired_errors.pdf     per-seed pretrained -> trained errors on natural prompts (SFT and ours)
  pretrained_scale_errors.pdf   median signed forecast error of 11 pretrained models vs deployment size
  sampled_rl_tradeoff.pdf       forecast improvement factor and capability vs rollouts (sampled-return RL)
  shift_paired_errors.pdf       per-seed errors without and with the fit/deployment family shift

make_minimal_change_figure.py draws the figure of the minimal-change appendix with this file's plot style.

Run it as a one-core Slurm CPU job; about half a minute of CPU. Every number that the appendix text
also quotes is asserted against the text's value (hard-coded below), so a data change fails loudly.

Inputs (each is a flag of main(); RESULTS_ROOT defaults to artifacts/results/):
  --natural-json          $RESULTS_ROOT/summaries/natural_results_verified.json. Per (size, seed), the
                          pretrained and trained forecast errors of ours and SFT on natural prompts, from the
                          best-checkpoint evaluations of the runs in scripts/wildchat/natural_prompts/.
                          scripts/wildchat/natural_prompts/collect_natural_results.py writes it.
  --rl-capability-json    $RESULTS_ROOT/summaries/rl_capability_verified.json. Per rollout count (1, 2, 4,
                          8, 16), the mean return on the pretraining tasks of the pretrained and trained
                          policies over ten seeds, from the sampled-return RL runs' final evaluations.
                          scripts/gridworld/collect_sampled_returns.py writes it.
  --distshift-md          $RESULTS_ROOT/summaries/distshift_results.md. A Markdown results file whose table
                          "| seed | baseline | trained (best/) | ..." gives the per-seed forecast errors of the
                          family-shift password runs (test_results.json, baseline_test_results.json).
                          scripts/password_game/collect_distshift_results.py writes it.
  --three-metric-csv      $FIGURES_ROOT/lm_three_metric_bars.csv, the per-seed IID errors. Written by
                          scripts/figures/plot_lquant_three_metric_bars.py
                          --csv-output $FIGURES_ROOT/lm_three_metric_bars.csv (default preset).
  --pretrained-grid-root  $RESULTS_ROOT/fit_deploy_error_grid_fullpool, holding
                          <model>/inverse_weibull_grid.npz. Written by
                          scripts/wildchat/pretrained_scale/fit_deploy_error_grid.py with that --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", REPO_ROOT / "artifacts" / "figures"))
OUT = FIGURES_ROOT
BUILD = FIGURES_ROOT / "build"  # PNG previews
sys.path.insert(0, str(HERE))
from _three_metric_bars_common import apply_paper_style  # noqa: E402

# The inputs listed in the module docstring; main()'s flags override them.
INPUTS = {
    "natural_json": RESULTS_ROOT / "summaries" / "natural_results_verified.json",
    "rl_capability_json": RESULTS_ROOT / "summaries" / "rl_capability_verified.json",
    "distshift_md": RESULTS_ROOT / "summaries" / "distshift_results.md",
    "three_metric_csv": FIGURES_ROOT / "lm_three_metric_bars.csv",
    "pretrained_grid_root": RESULTS_ROOT / "fit_deploy_error_grid_fullpool",
}

# Paper palette (main-text bar charts): baselines warm, ours cool.
NAVY = "#3D5A80"      # ours
BROWN = "#A65D49"     # direct risk minimization (SFT / RL)
GRAY = "#6E6E6E"      # pretrained reference
INK = "#222222"

WIDTH = 5.5  # NeurIPS text width, inches


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def save(fig, name):
    fig.savefig(OUT / name, metadata={"Author": "", "Creator": ""})
    fig.savefig(BUILD / name.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def style_log_axis(ax):
    ax.grid(axis="y", which="major", color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- natural prompts
def natural():
    rows = json.loads(Path(INPUTS["natural_json"]).read_text())
    sizes = ["0.6b", "1.7b", "4b", "8b"]
    labels = {"0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B", "4b": "Qwen3-4B", "8b": "Qwen3-8B"}
    paper_means = {"0.6b": 4.75, "1.7b": 38.9, "4b": 326, "8b": 25.8}
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 2.25), sharey=True, layout="constrained")
    for ax, size in zip(axs, sizes):
        rs = sorted([r for r in rows if r["size"] == size], key=lambda r: r["seed"])
        assert len(rs) == 3
        folds = [r["ours"]["baseline_error"] / r["ours"]["error"] for r in rs]
        close(np.mean(folds), paper_means[size], 0.05 * paper_means[size])
        for r in rs:
            base = r["ours"]["baseline_error"]
            assert abs(base - r["sft"]["baseline_error"]) < 1e-9
            ax.plot([0, 1], [base, r["sft"]["error"]], color=BROWN, lw=1.3, alpha=0.9,
                    marker="o", ms=3.2, solid_capstyle="round")
            ax.plot([0, 1], [base, r["ours"]["error"]], color=NAVY, lw=1.3, alpha=0.9,
                    marker="o", ms=3.2, solid_capstyle="round")
            ax.plot([0], [base], marker="o", ms=4.2, color=GRAY, zorder=4)
        ax.set_yscale("log")
        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([0, 1], ["Pretrained", "Trained"])
        ax.set_title(labels[size], fontsize=9)
        style_log_axis(ax)
    axs[0].set_ylabel("Forecast error (squared log$_{10}$)")
    axs[0].plot([], [], color=NAVY, lw=1.3, label="Ours")
    axs[0].plot([], [], color=BROWN, lw=1.3, label="SFT")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               frameon=False, fontsize=8.5)
    save(fig, "natural_paired_errors.pdf")


# ---------------------------------------------------------------- pretrained scale
def pretrained_scale():
    root = Path(INPUTS["pretrained_grid_root"])
    fams = {
        "Qwen3": ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"],
        "Gemma 3": ["gemma-3-270m-it", "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    }
    table = {"Qwen3-0.6B": 8, "Qwen3-1.7B": 46, "Qwen3-4B": 26, "Qwen3-8B": 26, "Qwen3-14B": 28, "Qwen3-32B": 19,
             "gemma-3-270m-it": 55, "gemma-3-1b-it": 5, "gemma-3-4b-it": 34, "gemma-3-12b-it": 160,
             "gemma-3-27b-it": 246}  # the table of "Forecastability of pretrained models across scale"
    ramps = {"Qwen3": plt.get_cmap("Blues"), "Gemma 3": plt.get_cmap("Oranges")}
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True, layout="constrained")
    all_med = []
    for ax, (fam, models) in zip(axs, fams.items()):
        for i, m in enumerate(models):
            d = np.load(root / m / "inverse_weibull_grid.npz")
            mi = int(np.where(d["m_grid"] == 44)[0][0])
            n = d["n_grid"]
            keep = (n > 44) & d["feasible"][mi]
            med = d["median"][mi][keep]
            all_med.extend(med.tolist())
            factor = 10 ** (-med.min())
            if m in table:
                close(factor, table[m], 0.6)
            print(f"  {m}: worst under-prediction x{factor:.1f} at N={n[keep][med.argmin()]}")
            col = ramps[fam](0.35 + 0.6 * i / max(1, len(models) - 1))
            short = m.replace("gemma-3-", "").replace("-it", "").replace("Qwen3-", "").upper()
            ax.plot(n[keep], med, color=col, lw=1.6, marker="o", ms=2.6, label=short)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhspan(-10, 0, color="#F3F0EC", zorder=0, lw=0)
        ax.set_xscale("log")
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("Deployment size $N$")
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right", title="Model size",
                  title_fontsize=7.5, columnspacing=1.2, handlelength=1.6)
    lo, hi = min(all_med), max(all_med)
    axs[0].set_ylim(lo - 0.35, hi + 1.3)  # headroom so the legends clear the Gemma 1B peak
    axs[0].set_ylabel("Median forecast error (decades)")
    axs[0].text(60, lo - 0.2, "under-predicts", fontsize=7.5, color="#7A6A58", style="italic")
    save(fig, "pretrained_scale_errors.pdf")


# ---------------------------------------------------------------- sampled RL
def sampled_rl():
    n = np.array([1, 2, 4, 8, 16])
    fold = np.array([6.5, 11.5, 10.4, 21.8, 29.7])   # the table of "Forecastability training with sampled returns"
    sem = np.array([2.3, 5.1, 4.7, 8.5, 13.8])
    rl_n, rl_fold, rl_sem = np.array([8, 16]), np.array([7.6, 10.4]), np.array([4.3, 3.0])
    ret = json.loads(Path(INPUTS["rl_capability_json"]).read_text())
    assert [r["rollouts"] for r in ret] == list(n)
    cap = np.array([r["trained_mean"] for r in ret])
    cap_sem = np.array([r["trained_sem"] for r in ret])
    for c, want in zip(cap, [0.16, 0.54, 0.72, 0.81, 0.91]):
        close(c, want, 0.006)
    pre, pre_sem = ret[0]["pretrained_mean"], ret[0]["pretrained_sem"]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.3), layout="constrained")
    ax = axs[0]
    ax.errorbar(n, fold, yerr=sem, color=NAVY, marker="o", ms=4, lw=1.5, capsize=2.5, label="Ours")
    ax.errorbar(rl_n * 1.06, rl_fold, yerr=rl_sem, color=BROWN, marker="s", ms=4, lw=0,
                elinewidth=1.3, capsize=2.5, label="RL")
    ax.axhline(1, color=GRAY, lw=0.8, ls="--")
    ax.set_ylabel("Forecast improvement factor")
    ax.set_ylim(0, 47)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axs[1]
    ax.axhspan(pre - pre_sem, pre + pre_sem, color=GRAY, alpha=0.15, lw=0)
    ax.axhline(pre, color=GRAY, lw=0.9, ls="--")
    ax.text(1.05, pre + 0.035, "pretrained policy", fontsize=7.5, color=GRAY)
    ax.errorbar(n, cap, yerr=cap_sem, color=NAVY, marker="o", ms=4, lw=1.5, capsize=2.5)
    ax.set_ylabel("Mean return, pretraining tasks")
    ax.set_ylim(0, 1)
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xticks(n, [str(v) for v in n])
        ax.set_xlabel("Rollouts per task per update")
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    save(fig, "sampled_rl_tradeoff.pdf")


# ---------------------------------------------------------------- shift
def shift():
    md = Path(INPUTS["distshift_md"]).read_text().splitlines()
    i0 = next(i for i, l in enumerate(md) if l.startswith("| seed | baseline | trained (best/)"))
    sh = {}
    for l in md[i0 + 2:]:
        m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", l)
        if not m:
            break
        sh[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    assert len(sh) == 10, sh
    close(np.mean([b for b, _ in sh.values()]), 13.83, 0.01)
    close(np.mean([t for _, t in sh.values()]), 0.455, 0.002)
    close(np.mean([b / t for b, t in sh.values()]), 58.7, 0.6)
    iid = {}
    for r in csv.DictReader(open(INPUTS["three_metric_csv"])):
        if r["method"] == "Ours":
            seed = int(re.search(r"seed_(\d+)", r["run_seed"]).group(1))
            iid[seed] = (float(r["baseline_forecast_err"]), float(r["forecast_err"]))
    assert sorted(iid) == sorted(sh), (sorted(iid), sorted(sh))
    close(np.mean([b / t for b, t in iid.values()]), 60.6, 0.3)
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH * 0.8, 2.3), sharey=True, layout="constrained")
    for ax, (title, d) in zip(axs, [("Original IID split", iid), ("Family shift", sh)]):
        for seed, (b, t) in sorted(d.items()):
            ax.plot([0, 1], [b, t], color=NAVY, lw=1.1, alpha=0.55, marker="o", ms=2.8)
        gb = math.exp(np.mean([math.log(b) for b, _ in d.values()]))
        gt = math.exp(np.mean([math.log(t) for _, t in d.values()]))
        ax.plot([0, 1], [gb, gt], color=INK, lw=2.0, marker="o", ms=4.5, zorder=5,
                label="geometric mean")
        ax.set_yscale("log")
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1], ["Pretrained", "Trained"])
        ax.set_title(title, fontsize=9)
        style_log_axis(ax)
    axs[0].set_ylabel("Forecast error (squared log$_{10}$)")
    axs[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    save(fig, "shift_paired_errors.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--natural-json", type=Path, default=INPUTS["natural_json"])
    ap.add_argument("--rl-capability-json", type=Path, default=INPUTS["rl_capability_json"])
    ap.add_argument("--distshift-md", type=Path, default=INPUTS["distshift_md"])
    ap.add_argument("--three-metric-csv", type=Path, default=INPUTS["three_metric_csv"])
    ap.add_argument("--pretrained-grid-root", type=Path, default=INPUTS["pretrained_grid_root"])
    INPUTS.update(vars(ap.parse_args()))
    apply_paper_style()
    plt.rcParams.update({"axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    BUILD.mkdir(parents=True, exist_ok=True)
    natural()
    pretrained_scale()
    sampled_rl()
    shift()


if __name__ == "__main__":
    main()
