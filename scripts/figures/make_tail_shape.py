"""Tail shape before and after forecastability training with minimal change to the model.

Draws held-out deployment-set scores in the forecasting method's plane, with depth y = -log S_hat on the
horizontal axis (rarer tasks to the right) and the risk score psi = -log(-log p) on the vertical
axis (riskier up), so the forecast line rises up and to the right. The horizontal axis is labeled in
plain terms, since the reader of the body's tail-shape figure has not yet met depth or S_hat: its ticks give
exp(-y), the fraction of prompts at least as risky, on a log scale, and one shared label names it. The trained model is the
arm of "Forecastability training with minimal change to the model" that replaces the main recipe's improving-only
mask with the anchor penalty and adds the spread constraint (the anchor penalty is a replacement for the mask, not an
added term), Qwen3-0.6B, 200 steps. Runs no experiments; reads saved snapshot traces of the minimal-change runs in
$RESULTS_ROOT/lquant_dynamics/runs/ and the readouts in $RESULTS_ROOT/lquant_dynamics/readouts/.

Writes into $FIGURES_ROOT (default artifacts/figures/):
  tail_shape_body_wildchat.pdf       1x2 (body): one held-out WildChat pool, pretrained | trained
  tail_shape_body_row.pdf            1x4 (body): that WildChat pool and one held-out password, in one row
  tail_shape_password.pdf            1x2: the password, pretrained | trained
  tail_shape_password_wildchat.pdf   2x2: the password row above the WildChat row
  tail_shape_three_recipes.pdf       1x3: the password, pretrained | main recipe | trained
and PNG previews in $FIGURES_ROOT/build/. Every number that the figures' captions quote is asserted below, so a
data change fails loudly. About twenty seconds of CPU; run it as a one-core Slurm CPU job.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", REPO_ROOT / "artifacts" / "figures"))
RUNS = RESULTS_ROOT / "lquant_dynamics" / "runs"
BUILD = FIGURES_ROOT / "build"
BUILD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "minimal_change" / "readouts"))
from _three_metric_bars_common import apply_paper_style  # noqa: E402
from poolscale_readout import S, split_errors  # noqa: E402  (the readers' 200-split metric)

WIDTH = 5.5  # NeurIPS text width, inches
STEP = {"password": 200, "wildchat": 200}  # --wildchat-step 100 reads WildChat at 100 steps (the _wc100 variants)
SEEDS = (11, 17, 23, 29, 43)

# Figure 1's palette: navy = model and deployment set, orange = fit set
# and its forecast, red = forecast error. Gray = pretrained reference.
NAVY = "#1F3A5F"
ORANGE = "#D9822B"
RED = "#922B21"
GRAY = "#ABABAB"
SHADE = "#F2F2F2"
NOTE = "#777777"
GREEN = "#20613D"  # Figure 1's f1green (#2E8B57) at 70%, for "forecast within ..." on trained panels

# Run directories (the run tags of scripts/minimal_change/readouts/strict_table.py's ARMS).
RUN = {
    ("both", "password"): lambda s: f"lqpw_pw2apsp200_seed{s}",
    ("both", "wildchat"): lambda s: f"lqdyn_v2e2ap50spx200_seed{s}",
    ("mask", "password"): lambda s: f"lqpw_pw2iog200_seed{s}",
    ("mask", "wildchat"): lambda s: f"lqdyn_v2e2iogap50x200_seed{s}",
}
RUN100 = {("both", "wildchat"): lambda s: f"lqdyn_v2e2ap50sp_seed{s}",
          ("mask", "wildchat"): lambda s: f"lqdyn_v2e2iogap50_seed{s}"}
# The pools shown; their typicality is printed and asserted.
PASSWORD = ("password", 17, "password_slot_16")
WILDCHAT = ("wildchat", 23, "wc09")
P_TICKS = {
    "password": [0.1, 1e-2, 1e-5, 1e-10, 1e-15],
    "wildchat": [1e-2, 1e-4, 1e-6, 1e-8, 1e-15],
    "three": [0.1, 1e-5, 1e-15, 1e-40],
    "password_row": [0.1, 1e-5, 1e-10, 1e-15],
    "wildchat_row": [1e-2, 1e-4, 1e-8, 1e-15],
}
TRAINED_TITLE = "after minimal-change training"  # the minimal-change appendix's name for the recipe
READOUTS = RESULTS_ROOT / "lquant_dynamics" / "readouts"  # the minimal-change appendix's tables


def psi(lp):
    return -np.log(np.maximum(-np.asarray(lp, dtype=np.float64), 1e-300))


def psi_of_p(p):
    return -math.log(-math.log(p))


def load(npz):
    z = np.load(npz)
    meta = json.loads(Path(str(npz).replace(".npz", "_metadata.json")).read_text())
    out = {}
    for t in meta["targets"]:
        a = t["array_prefix"]
        out[t["target"]] = (z[f"{a}__fit_log_probs"].astype(np.float64),
                            z[f"{a}__deploy_log_probs"].astype(np.float64))
    return out, int(meta["ols_top_k"])


def snapshots(arm, bank, seed):
    step = STEP[bank]
    run = (RUN100 if step <= 100 and (arm, bank) in RUN100 else RUN)[(arm, bank)](seed)
    ev = RUNS / run / "snapshot_evals"
    f = ev / f"step{step:04d}_test_forecast_traces.npz"
    if not f.exists():
        return None
    base, k = load(ev / "base_test_forecast_traces.npz")
    trained, _ = load(f)
    return base, trained, k


def ols(fit_lp, k):
    """The forecaster: OLS of log S_hat_i = a + b psi_(i) on the fit set's top k, S_hat_i = i/(M+1)."""
    M = fit_lp.size
    top = psi(np.sort(fit_lp)[::-1][:k])
    ls = np.log(np.arange(1, k + 1) / (M + 1.0))
    b, a = np.polyfit(top, ls, 1)
    return a, b, top, -ls  # depths of the top k


def qhat(a, b, depth):
    """Forecast score at depth y: the line a + b psi = -y solved for psi."""
    return (-np.asarray(depth) - a) / b


def pool_stats(fb, db, ft, dt, k, name):
    """Errors on the stored split and over the readers' 200 splits (fold = base MSE / trained MSE)."""
    vb, vt = np.concatenate([fb, db]), np.concatenate([ft, dt])
    M = fb.size
    rng = np.random.default_rng(int.from_bytes(name.encode(), "little") % (2**32))
    perms = np.stack([rng.permutation(vb.size) for _ in range(S)])
    eb, _ = split_errors(vb, perms, M, k=k)
    et, _ = split_errors(vt, perms, M, k=k)
    canon = np.arange(vb.size)[None]  # the stored split: fit = the first M prompts
    cb, _ = split_errors(vb, canon, M, k=k)
    ct, _ = split_errors(vt, canon, M, k=k)
    rb, rt = np.sort(db)[::-1], np.sort(dt)[::-1]  # deployment set by rank
    up = rt > rb + 1e-9
    top3 = np.argsort(db)[::-1][:3]  # the pretrained model's three riskiest deployment prompts
    rose = vt > vb + 1e-9  # prompts (fit and deployment) whose log-probability rose
    return {"fold": float((eb ** 2).mean() / (et ** 2).mean()), "err_b": float(cb[0]), "err_t": float(ct[0]),
            "rms_b": float(np.sqrt((eb ** 2).mean())), "rms_t": float(np.sqrt((et ** 2).mean())),
            "mean_b": float(eb.mean()), "mean_t": float(et.mean()),
            # how unusual the drawn (stored) split is among the 200: splits at least as bad before / as good after
            "splits_worse_b": int((np.abs(eb) >= abs(cb[0])).sum()), "splits_better_t": int((np.abs(et) <= abs(ct[0])).sum()),
            "max_prompt_rise": float((vt - vb).max()), "max_p_risen": float(math.exp(vt[rose].max())) if rose.any() else 0.0,
            "dmax_b": float(rb[0]), "dmax_t": float(rt[0]), "med_fall": float(np.median(vb) - np.median(vt)),
            "ranks_up": int(up.sum()), "ranks_up_top100": int(up[:100].sum()),
            "max_rank_rise": float((rt - rb).max()), "max_lp_at_risen_rank": float(rt[up].max()) if up.any() else -np.inf,
            "prompts_up": int((vt > vb + 1e-9).sum()), "n_deploy": int(db.size), "n_pool": int(vb.size),
            "above_01_b": int((db > math.log(0.1)).sum()), "above_01_t": int((dt > math.log(0.1)).sum()),
            "top3_b": db[top3].tolist(), "top3_t": dt[top3].tolist(), "min_prompt_fall": float((vb - vt).min())}


def bank_folds(arm, bank):
    """Per-pool folds for every held-out pool of every seed that has reached STEP."""
    folds, seeds, tops, errs, ups, rises = [], [], [], [], [], []
    for s in SEEDS:
        snap = snapshots(arm, bank, s)
        if snap is None:
            continue
        base, trained, k = snap
        seeds.append(s)
        for name, (fb, db) in base.items():
            ft, dt = trained[name]
            st = pool_stats(fb, db, ft, dt, k, name)
            folds.append(st["fold"])
            errs.append(st["err_b"])  # the pretrained forecast's error on the stored split
            tops.append(float(max(fb.max(), db.max())))  # the pool's riskiest pretrained prompt (log p)
            ups.append(st["prompts_up"])
            rises.append(st["max_prompt_rise"])
    bank_folds.tops = np.array(tops)
    bank_folds.errs = np.array(errs)
    bank_folds.ups = np.array(ups)
    bank_folds.rises = np.array(rises)
    return np.array(folds), seeds


def table_row(bank, step):
    """The minimal-change appendix's pooled row for the both-terms arm (strict_table_200.csv, as-read and spread-normalized)."""
    import csv
    for r in csv.DictReader(open(READOUTS / "strict_table_200.csv")):
        if r["dataset"] == bank and r["arm"] == "anchpen + band (main)" and r["seed"] == "pooled" and int(r["step"]) == step:
            return {key: float(r[key]) for key in ("asread_fold", "asread_lo", "asread_hi", "fold", "pooled_lo", "pooled_hi", "pools")}
    raise KeyError((bank, step))


def kl_geomean(bank):
    """The minimal-change appendix's KL: the geometric mean over seeds of the single-token KL at 200 steps (kl_table.csv)."""
    import csv
    v = [float(r["kl_single_token"]) for r in csv.DictReader(open(READOUTS / "kl_table.csv"))
         if r["dataset"] == bank and r["arm"] == "anchpen + band (main), 200 steps"]
    assert len(v) == 5, (bank, len(v))
    return math.exp(sum(math.log(x) for x in v) / len(v))


def panel(ax, fit, dep, k, *, ref=None, xlim, ylim, fit_reach_label=True):
    M, N = fit.size, dep.size
    d = np.log((N + 1.0) / np.arange(1, N + 1))  # Weibull depth of each deployment rank
    s = psi(np.sort(dep)[::-1])
    a, b, ftop, fd = ols(fit, k)
    d_star = math.log(N + 1.0)
    q_star = float(qhat(a, b, math.log(N)))  # the readers' convention: the line at log S = log(1/N)
    ax.axvspan(math.log(M + 1.0), xlim[1] + 1, color=SHADE, lw=0, zorder=0)
    if fit_reach_label:
        ax.text(math.log(M + 1.0) + 0.1, ylim[0] + 0.03 * (ylim[1] - ylim[0]), "beyond\nthe fit set",
                fontsize=7, color=NOTE, va="bottom", ha="left", linespacing=1.0)
    if ref is not None:
        ax.scatter(*ref, s=5, color=GRAY, lw=0, zorder=1)
    ax.scatter(d, s, s=5, color=NAVY, lw=0, zorder=2, alpha=0.85)
    g = np.linspace(fd.min() - 0.6, d_star + 0.05, 100)
    ax.plot(g, qhat(a, b, g), color=ORANGE, lw=1.1, ls=(0, (4, 2)), zorder=3)
    ax.scatter(fd, ftop, s=16, marker="x", color=ORANGE, lw=1.0, zorder=4)
    ax.scatter([d_star], [q_star], s=34, facecolors="white", edgecolors=ORANGE, lw=1.2, zorder=6)
    xe = d_star + 0.17  # the error bar, just right of the deepest rank
    ax.plot([xe, xe], [q_star, s[0]], color=RED, lw=1.3, zorder=5, solid_capstyle="butt")
    for yy in (q_star, s[0]):
        ax.plot([xe - 0.07, xe + 0.07], [yy, yy], color=RED, lw=1.0, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    frac_ticks(ax, xlim)
    return d, s


def frac_ticks(ax, xlim):
    """Label depth y by exp(-y), the fraction of prompts at least as risky: 1, 10^-1, 10^-2, ... on a log scale."""
    m = [i for i in range(0, 8) if i * math.log(10.0) <= xlim[1]]
    ax.set_xticks([i * math.log(10.0) for i in m])
    ax.set_xticklabels(["1" if i == 0 else rf"$10^{{-{i}}}$" for i in m])


def error_text(ax, err, lines=2, trained=False):
    """The forecast error in words: "too low by 1.8 orders of magnitude" (red) before training, and
    "within 0.3 orders of magnitude" (green) after it."""
    side = "low" if err < 0 else "high"
    if trained:
        txt = (f"forecast within\n{abs(err):.1f} orders\nof magnitude" if lines == 3
               else f"forecast within\n{abs(err):.1f} orders of magnitude")
    elif lines == 3:  # narrow panels
        txt = f"forecast too {side}\nby {abs(err):.1f} orders\nof magnitude"
    else:
        txt = f"forecast too {side} by\n{abs(err):.1f} orders of magnitude"
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, color=GREEN if trained else RED, fontsize=7.5, ha="left",
            va="top", zorder=7, linespacing=1.1)


def p_axis(ax, ticks, label=True):
    sec = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    sec.set_yticks([psi_of_p(p) for p in ticks])
    sec.set_yticklabels([f"{p:g}" if p >= 0.1 else rf"$10^{{{round(math.log10(p))}}}$" for p in ticks])
    sec.tick_params(labelsize=7.5, length=2.5)
    if label:
        sec.set_ylabel(r"probability $p$ (riskier $\uparrow$)", fontsize=9)


def ylimits(pairs, k):
    ys = []
    for fit, dep in pairs:
        ys.append(psi(dep))
        a, b, ftop, _ = ols(fit, k)
        ys.append(np.array([qhat(a, b, math.log(dep.size))]))
        ys.append(ftop)
    ys = np.concatenate(ys)
    lo, hi = float(ys.min()), float(ys.max())
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def legend(fig, k_text, loc="outside lower center"):
    h = [Line2D([], [], ls="", marker="o", ms=3.2, color=NAVY, label="deployment-set scores"),
         Line2D([], [], ls="", marker="o", ms=3.2, color=GRAY, label="pretrained deployment-set scores"),
         Line2D([], [], ls="", marker="x", ms=4.5, mew=1.0, color=ORANGE, label=f"fit set's top {k_text} scores"),
         Line2D([], [], color=ORANGE, lw=1.1, ls=(0, (4, 2)), label="forecast line"),
         Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=ORANGE, mew=1.2,
                label="forecast of the deployment maximum"),
         Line2D([], [], color=RED, lw=1.3, label="forecast error")]
    fig.legend(handles=h, loc=loc, ncol=3, fontsize=7.5, frameon=False,
               handletextpad=0.4, columnspacing=1.2)


def pool(arm, spec):
    bank, seed, name = spec
    base, trained, k = snapshots(arm, bank, seed)
    return base[name], trained[name], k


YLABEL = r"risk score $\psi$ (riskier $\uparrow$)"
XLABEL = r"fraction of prompts at least this risky (rarer $\rightarrow$)"  # shared by a figure's panels (supxlabel)
XLABEL_SHORT = r"fraction at least this risky"  # narrow panels


def draw_pair(ax0, ax1, spec, *, title0, title1, ticks, ylabel=YLABEL, p_label=True, err_lines=2,
              fit_reach_label=True):
    (fb, db), (ft, dt), k = pool("both", spec)
    st = pool_stats(fb, db, ft, dt, k, spec[2])
    xlim = (0.0, math.log(db.size + 1.0) + 0.45)
    ylim = ylimits([(fb, db), (ft, dt)], k)
    ref = panel(ax0, fb, db, k, xlim=xlim, ylim=ylim, fit_reach_label=fit_reach_label)
    panel(ax1, ft, dt, k, ref=ref, xlim=xlim, ylim=ylim, fit_reach_label=fit_reach_label)
    error_text(ax0, st["err_b"], err_lines)
    error_text(ax1, st["err_t"], err_lines, trained=True)
    ax0.set_title(title0)
    ax1.set_title(title1)
    if ylabel:
        ax0.set_ylabel(ylabel)
    ax1.tick_params(labelleft=False)
    p_axis(ax1, ticks, label=p_label)
    return st, k


def wc_suffix():
    return "" if STEP["wildchat"] == 200 else f"_wc{STEP['wildchat']}"


def save(fig, name):
    fig.savefig(FIGURES_ROOT / f"{name}.pdf", metadata={"Author": "", "Creator": ""})
    fig.savefig(BUILD / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", FIGURES_ROOT / f"{name}.pdf")


def two_rows(height):
    """A figure whose top subfigure holds the panels and their shared x label, and whose bottom one the legend."""
    fig = plt.figure(figsize=(WIDTH, height), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.16], hspace=0.0)
    return fig, top, bottom


def fig_body_wildchat():
    """Body figure: the WildChat pool alone, pretrained | trained, one row at full text width."""
    fig, top, bottom = two_rows(2.45)
    axs = top.subplots(1, 2)
    st, k = draw_pair(axs[0], axs[1], WILDCHAT, title0="WildChat: pretrained",
                      title1=f"WildChat: {TRAINED_TITLE}", ticks=P_TICKS["wildchat"])
    top.supxlabel(XLABEL, fontsize=9)
    legend(bottom, str(k), loc="center")
    save(fig, "tail_shape_body_wildchat" + wc_suffix())
    return st


def fig_body_row():
    """Body figure with both banks in one row: the WildChat pair (left) and the password pair (right)."""
    fig = plt.figure(figsize=(WIDTH, 2.35), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.17], hspace=0.0)
    subs = top.subfigures(1, 2, wspace=0.04)
    for sub, spec, label, first in ((subs[0], WILDCHAT, "WildChat", True), (subs[1], PASSWORD, "Password game", False)):
        axs = sub.subplots(1, 2)
        draw_pair(axs[0], axs[1], spec, title0="pretrained", title1="after training", ticks=P_TICKS[spec[0] + "_row"],
                  ylabel=r"risk score $\psi$" if first else None, p_label=not first, err_lines=3,
                  fit_reach_label=False)
        for ax in axs:
            ax.set_xlabel(XLABEL_SHORT, fontsize=8)
            ax.title.set_fontsize(8.5)
        sub.suptitle(label, fontsize=9.5)
    h = [Line2D([], [], ls="", marker="o", ms=3.2, color=NAVY, label="deployment-set scores"),
         Line2D([], [], ls="", marker="o", ms=3.2, color=GRAY, label="pretrained scores"),
         Line2D([], [], ls="", marker="x", ms=4.5, mew=1.0, color=ORANGE, label="fit set's top $k$"),
         Line2D([], [], color=ORANGE, lw=1.1, ls=(0, (4, 2)), label="forecast line"),
         Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=ORANGE, mew=1.2, label="forecast of the maximum"),
         Line2D([], [], color=RED, lw=1.3, label="forecast error")]
    bottom.legend(handles=h, loc="center", ncol=3, fontsize=7.5, frameon=False, handletextpad=0.4,
                  columnspacing=1.2)
    save(fig, "tail_shape_body_row" + wc_suffix())


def fig_password():
    fig, top, bottom = two_rows(2.6)
    axs = top.subplots(1, 2)
    st, k = draw_pair(axs[0], axs[1], PASSWORD, title0="Pretrained", title1=TRAINED_TITLE.capitalize(),
                      ticks=P_TICKS["password"])
    top.supxlabel(XLABEL, fontsize=9)
    legend(bottom, str(k), loc="center")
    save(fig, "tail_shape_password")
    return st


def fig_password_wildchat():
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.0), layout="constrained")
    out = {}
    for r, (spec, label) in enumerate(((PASSWORD, "Password game"), (WILDCHAT, "WildChat"))):
        out[spec[0]], _ = draw_pair(axs[r, 0], axs[r, 1], spec, title0=f"{label}: pretrained",
                                    title1=f"{label}: {TRAINED_TITLE}", ticks=P_TICKS[spec[0]])
    fig.supxlabel(XLABEL, fontsize=9)
    legend(fig, "k")
    save(fig, "tail_shape_password_wildchat" + wc_suffix())
    return out


def fig_three_recipes():
    (fb, db), (ft, dt), k = pool("both", PASSWORD)
    (fb2, db2), (fm, dm), _ = pool("mask", PASSWORD)
    assert np.array_equal(fb, fb2) and np.array_equal(db, db2), "the two arms hold out different pools"
    sm = pool_stats(fb, db, fm, dm, k, PASSWORD[2])
    st = pool_stats(fb, db, ft, dt, k, PASSWORD[2])
    xlim = (0.0, math.log(db.size + 1.0) + 0.45)
    ylim = ylimits([(fb, db), (ft, dt), (fm, dm)], k)
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.45), layout="constrained")
    ref = panel(axs[0], fb, db, k, xlim=xlim, ylim=ylim)
    panel(axs[1], fm, dm, k, ref=ref, xlim=xlim, ylim=ylim, fit_reach_label=False)
    panel(axs[2], ft, dt, k, ref=ref, xlim=xlim, ylim=ylim, fit_reach_label=False)
    for i, (ax, e) in enumerate(zip(axs, (st["err_b"], sm["err_t"], st["err_t"]))):
        error_text(ax, e, lines=3, trained=i > 0)
    fig.supxlabel(XLABEL, fontsize=9)
    for ax, t in zip(axs, ("Pretrained", "Main recipe", "Minimal-change training")):
        ax.set_title(t)
    axs[0].set_ylabel(r"risk score $\psi$ (riskier $\uparrow$)")
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    p_axis(axs[2], P_TICKS["three"])
    legend(fig, str(k))
    save(fig, "tail_shape_three_recipes")
    return sm


def close(a, b, tol, what):
    assert abs(a - b) <= tol, f"{what}: {a} vs expected {b}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wildchat-step", type=int, default=200, choices=(100, 200))
    args = ap.parse_args()
    STEP["wildchat"] = args.wildchat_step
    apply_paper_style()
    plt.rcParams.update({"axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    wc = fig_body_wildchat()
    fig_body_row()
    pw = fig_password()
    both = fig_password_wildchat()
    assert both["wildchat"]["fold"] == wc["fold"] and both["password"]["fold"] == pw["fold"]
    mask = fig_three_recipes()
    pw_folds, pw_seeds = bank_folds("both", "password")
    pw_tops = bank_folds.tops
    wc_folds, wc_seeds = bank_folds("both", "wildchat")
    wc_errs, wc_ups, wc_rises = bank_folds.errs, bank_folds.ups, bank_folds.rises
    for label, st in (("password", pw), ("wildchat", wc)):
        print(f"{label}: prompts up {st['prompts_up']}/{st['n_pool']}, largest rise {st['max_prompt_rise']:.3f} nats, riskiest "
              f"risen prompt ends at p {st['max_p_risen']:.3g}; of 200 random splits, {st['splits_worse_b']} had a pretrained "
              f"error at least as large as the drawn split's and {st['splits_better_t']} a trained error at least as small")
    for label, st in (("password", pw), ("wildchat", wc), ("password, main recipe", mask)):
        print(f"{label}: stored-split error {st['err_b']:+.2f} -> {st['err_t']:+.2f} decades; over 200 splits RMS "
              f"{st['rms_b']:.2f} -> {st['rms_t']:.2f}, mean {st['mean_b']:+.2f} -> {st['mean_t']:+.2f}; 200-split fold "
              f"{st['fold']:.2f}; deployment maximum p {math.exp(st['dmax_b']):.3g} -> {math.exp(st['dmax_t']):.3g}; "
              f"median fall {st['med_fall']:.1f} nats; ranks up {st['ranks_up']}/{st['n_deploy']} "
              f"({st['ranks_up_top100']} of the top 100; max rise {st['max_rank_rise']:.2f} nats, "
              f"highest risen rank at log p {st['max_lp_at_risen_rank']:.1f}); prompts up {st['prompts_up']}/{st['n_pool']}")
    for label, st in (("password", pw), ("wildchat", wc)):
        print(f"{label}: deployment prompts above p = 0.1: {st['above_01_b']} -> {st['above_01_t']}; pretrained top three "
              f"log p {np.round(st['top3_b'], 2).tolist()} -> {np.round(st['top3_t'], 2).tolist()}")
    print(f"main recipe: smallest fall of any prompt {mask['min_prompt_fall']:.1f} nats")
    print(f"password pools: {pw_folds.size} from seeds {pw_seeds}, median fold {np.median(pw_folds):.2f}, "
          f"{(pw_folds > 1).sum()} above 1")
    print(f"wildchat pools: {wc_folds.size} from seeds {wc_seeds}, median fold {np.median(wc_folds):.2f}, "
          f"{(wc_folds > 1).sum()} above 1")

    # The numbers the figures' captions quote.
    close(pw["err_b"], -10.1, 0.05, "password pretrained error")
    close(pw["err_t"], -0.4, 0.05, "password trained error")
    close(pw["fold"], 4.0, 0.05, "password pool fold")
    close(math.exp(pw["dmax_b"]), 0.38, 0.005, "password pretrained deployment maximum")
    close(math.exp(pw["dmax_t"]) * 1e5, 1.3, 0.05, "password trained deployment maximum (x1e-5)")
    close(pw["med_fall"], 14.5, 0.05, "password median fall")
    assert pw["above_01_b"] == 3 and pw["above_01_t"] == 0, (pw["above_01_b"], pw["above_01_t"])
    # the three prompts above p = 0.1 each fell by at least 10 nats; the riskiest stayed the deployment maximum
    assert min(b - t for b, t in zip(pw["top3_b"], pw["top3_t"])) >= 10, (pw["top3_b"], pw["top3_t"])
    assert pw["top3_t"][0] == pw["dmax_t"], (pw["top3_t"], pw["dmax_t"])
    assert pw["max_rank_rise"] <= 0.8 and pw["max_lp_at_risen_rank"] < math.log(1e-7), (pw["max_rank_rise"], pw["max_lp_at_risen_rank"])
    assert pw_folds.min() > 1, pw_folds.min()  # every held-out password pool improved
    # "In 31 of the 35 held-out pools ... above 0.9 ... in this pool it was 0.38, the lowest of the 35"
    assert int((pw_tops > math.log(0.9)).sum()) == 31, int((pw_tops > math.log(0.9)).sum())
    close(float(np.exp(pw_tops.min())), 0.38, 0.005, "lowest pretrained pool maximum")
    close(float(np.exp(pw_tops.min())), math.exp(pw["dmax_b"]), 1e-9, "the chosen pool has the lowest maximum")
    assert pw_folds.size == 35, pw_folds.size
    close(float(np.median(pw_folds)), 4.8, 0.05, "password median pool fold")
    # over 200 random splits of the password pool: RMS error 5.3 -> 2.6 decades; the trained forecast ran 2.2 too high
    close(pw["rms_b"], 5.3, 0.05, "password RMS error, pretrained")
    close(pw["rms_t"], 2.6, 0.05, "password RMS error, trained")
    close(pw["mean_t"], 2.2, 0.05, "password mean signed error, trained")
    # per prompt: 33 of the pool's 935 rose, by up to 5.05 nats; the riskiest of them ended at p = 3.5e-6
    assert pw["prompts_up"] == 33 and pw["n_pool"] == 935, (pw["prompts_up"], pw["n_pool"])
    close(pw["max_prompt_rise"], 5.05, 0.005, "password largest single-prompt rise")
    close(pw["max_p_risen"] * 1e6, 3.5, 0.05, "password riskiest risen prompt, trained p (x1e-6)")
    assert pw["max_p_risen"] < math.exp(pw["dmax_t"]), "a risen prompt ended above the trained deployment maximum"
    # the drawn split is unusual at both ends: its pretrained error exceeded 199 of the 200 random splits' errors,
    # and its trained error was below 192 of them
    assert pw["splits_worse_b"] == 1 and pw["splits_better_t"] == 8, (pw["splits_worse_b"], pw["splits_better_t"])
    tp = table_row("password", 200)  # the minimal-change appendix's pooled fold, as read
    close(tp["asread_fold"], 4.41, 0.005, "password pooled fold")
    close(tp["asread_lo"], 3.52, 0.005, "password pooled fold, lower")
    close(tp["asread_hi"], 5.63, 0.005, "password pooled fold, upper")
    assert int(tp["pools"]) == 35, tp
    close(kl_geomean("password") * 1e4, 4.6, 0.05, "password KL (x1e-4)")
    rank = int((wc_folds > wc["fold"] + 1e-12).sum()) + 1
    err_rank = int((wc_errs < wc["err_b"] - 1e-12).sum()) + 1
    print(f"WildChat pool: fold rank {rank} of {wc_folds.size}, pretrained-error rank {err_rank} (most negative first); "
          f"median fold {np.median(wc_folds):.4f}, {(wc_folds > 1).sum()} above 1, over seeds {wc_seeds}")
    if STEP["wildchat"] == 200:
        assert wc_seeds == list(SEEDS) and wc_folds.size == 40, (wc_seeds, wc_folds.size)
        close(wc["err_b"], -1.8, 0.05, "WildChat pretrained error")
        close(wc["err_t"], -0.3, 0.05, "WildChat trained error")
        close(wc["fold"], 5.2, 0.05, "WildChat pool fold")
        close(wc["rms_b"], 1.5, 0.05, "WildChat RMS error, pretrained")
        close(wc["rms_t"], 0.6, 0.05, "WildChat RMS error, trained")
        close(math.exp(wc["dmax_b"]), 0.032, 0.0005, "WildChat pretrained deployment maximum")
        close(math.exp(wc["dmax_t"]) * 1e4, 4.3, 0.05, "WildChat trained deployment maximum (x1e-4)")
        assert wc["ranks_up"] == 0, "a WildChat deployment rank rose; the caption says none did"
        # per prompt: 12 of the pool's 2,300 rose, by at most 0.38 nats; every held-out pool had some prompt rise,
        # by up to 2.3 nats
        assert wc["prompts_up"] == 12 and wc["n_pool"] == 2300 and wc["max_prompt_rise"] <= 0.38, (
            wc["prompts_up"], wc["n_pool"], wc["max_prompt_rise"])
        assert int((wc_ups > 0).sum()) == 40, int((wc_ups > 0).sum())
        close(float(wc_rises.max()), 2.3, 0.05, "WildChat largest single-prompt rise over the 40 pools")
        assert rank == 12 and err_rank == 9, (rank, err_rank)
        close(float(np.median(wc_folds)), 3.3, 0.05, "WildChat median pool fold")
        assert int((wc_folds > 1).sum()) == 33, int((wc_folds > 1).sum())  # 7 of the 40 pools got worse
        tw = table_row("wildchat", 200)
        close(tw["asread_fold"], 3.17, 0.005, "WildChat pooled fold")
        close(tw["asread_lo"], 2.27, 0.005, "WildChat pooled fold, lower")
        close(tw["asread_hi"], 4.42, 0.005, "WildChat pooled fold, upper")
        assert int(tw["pools"]) == 40, tw
        close(kl_geomean("wildchat") * 1e5, 4.0, 0.05, "WildChat KL (x1e-5)")
    else:  # the _wc100 variants: WildChat at 100 steps
        assert wc_seeds == list(SEEDS) and wc_folds.size == 40, (wc_seeds, wc_folds.size)
        close(wc["err_b"], -1.8, 0.05, "WildChat pretrained error")
        close(wc["err_t"], -0.4, 0.05, "WildChat trained error, 100 steps")
        close(wc["fold"], 3.9, 0.05, "WildChat pool fold, 100 steps")
        assert wc["ranks_up"] == 0 and rank == 19, (wc["ranks_up"], rank)
        close(float(np.median(wc_folds)), 3.8, 0.05, "WildChat median pool fold, 100 steps")
        assert int((wc_folds > 1).sum()) == 33, int((wc_folds > 1).sum())
    close(mask["err_t"], 0.2, 0.05, "main recipe error")
    assert mask["min_prompt_fall"] >= 63, mask["min_prompt_fall"]  # "at least 63 nats"
    print("all asserted numbers match")


if __name__ == "__main__":
    main()
