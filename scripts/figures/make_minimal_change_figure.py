"""The figure and the number checks of the minimal-change appendix.

The appendix is "Forecastability training with minimal change to the model". Its figure plots the improvement
factors of the four arms against the KL divergence from the base model (the geometric mean over seeds), with both banks
read at 200 training steps. A filled marker gives an arm's pooled improvement factor and its 95% interval, an open
marker its spread-normalized factor, and an arrow joins the two where they differ. The code's names keep "fold" for
the improvement factor. The script also checks every number that the appendix's table and
paragraphs take from the readouts, including those of "Training at other margins". Each number is hard-coded below as
printed and asserted against the readouts, so a data change fails loudly.

Inputs: the minimal-change readouts in $RESULTS_ROOT/lquant_dynamics/readouts/ (or --readouts), which
scripts/minimal_change/readouts/ writes: strict_table.csv, strict_table_200.csv, kl_table.csv, falls_table.csv and
margin_table.csv. Output: minimal_change_fold_kl_rebuild.pdf, the file name in the paper's source, in $FIGURES_ROOT
(default artifacts/figures/), with a PNG preview in $FIGURES_ROOT/build/.

Run it as a one-core Slurm CPU job; it takes under a minute.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_appendix_figures as maf  # noqa: E402  (the paper's plot style, output paths and save())

np, plt = maf.np, maf.plt
close = maf.close

# Both banks are read at 200 training steps. The mask with the spread constraint has four seeds on WildChat at
# 200 steps (seed 17 has no 200-step run), and every other arm has five.
STEP = "200"
SEEDS = {("wildchat", "spread"): 4}

# Which KL the appendix reports. "on" formats the held-out KL prompts with Qwen3's default chat template (thinking
# on), where the base model's first token is "<think>"; the trainers logged this KL. "off" uses the same prompts with
# thinking off, as in training and scoring. Both were recomputed from the saved adapters into kl_thinking_off_*.csv;
# the script that writes those files is not part of this release, so only the default ("on" with
# PASSWORD_CUT = "right") runs from the release's readouts.
KL_TEMPLATE = "on"

# Where the password runs' KL prompts are cut to 512 tokens. "right" is the trainers' cut, which removes the
# generation marker from 30 of the 254 prompts, so those are read mid-prompt; "left" reads every prompt at the
# reply's first token. The WildChat runs cut from the left, so this changes only the password numbers.
# "on" with "right" reads the trainers' own values (kl_table.csv, margin_table.csv).
PASSWORD_CUT = "right"

# The appendix's table (tab:minimal-change) as printed: improvement factor, its 95% interval, spread-normalized factor, KL
# (geometric mean over seeds), and the falls at the top and median (nats).
TABLE = {
    ("password", "mask"): ("185", "46", "1157", "0.20", "5.7e-3", "102.5", "68.3"),
    ("password", "anchor"): ("1091", "808", "1483", "0.20", "9.6e-3", "101.5", "67.0"),
    ("password", "spread"): ("0.82", "0.44", "1.36", "0.14", "1.2e-2", "89.5", "68.2"),
    ("password", "both"): ("4.41", "3.52", "5.63", "3.45", "4.6e-4", "6.0", "14.3"),
    ("wildchat", "mask"): ("6.39", "3.23", "19.5", "0.18", "22", "105.6", "90.2"),
    ("wildchat", "anchor"): ("5.63", "2.27", "16.4", "0.33", "1.5e-2", "16.5", "3.7"),
    ("wildchat", "spread"): ("0.73", "0.52", "1.03", "0.69", "9.7", "65.6", "60.1"),
    ("wildchat", "both"): ("3.17", "2.27", "4.42", "3.18", "4.0e-5", "3.9", "2.2"),
}
# KL numbers in the appendix's paragraphs: the phrase of the paper that prints each one, the quantity, and the
# number as printed. Ratios are of geometric means over seeds, at 200 steps unless noted.
TEXT_KL = [
    (r"were $4.6\times10^{-4}$ on the password bank", "password both", "4.6e-4"),
    (r"and $4.0\times10^{-5}$ on WildChat, $12$ and $550{,}000$ times smaller than with the mask alone",
     "wildchat both", "4.0e-5"),
    (r"$12$ and $550{,}000$ times smaller than with the mask alone", "password mask / both", "12"),
    (r"$12$ and $550{,}000$ times smaller than with the mask alone", "wildchat mask / both", "5.5e5"),
    (r"while KL grew to $1.4$ and $1.2$ times its value at $100$ steps", "password both, step 200 / step 100", "1.4"),
    (r"while KL grew to $1.4$ and $1.2$ times its value at $100$ steps", "wildchat both, step 200 / step 100", "1.2"),
    (r"at a KL divergence $1{,}400$ times smaller", "wildchat mask / anchor", "1.4e3"),
    (r"(KL $9.6\times10^{-3}$ against $5.7\times10^{-3}$)", "password anchor", "9.6e-3"),
    (r"(KL $9.6\times10^{-3}$ against $5.7\times10^{-3}$)", "password mask", "5.7e-3"),
    (r"At $100$ steps on the password bank, its KL divergence had been half the mask's",
     "password anchor / mask, step 100", "0.5"),
    (r"at KL divergences $25$ and $240{,}000$ times larger than with both terms", "password spread / both", "25"),
    (r"at KL divergences $25$ and $240{,}000$ times larger than with both terms", "wildchat spread / both", "2.4e5"),
    (r"it moved the model furthest of the four arms (KL $1.2\times10^{-2}$)", "password spread", "1.2e-2"),
]
# Other numbers in the paragraphs: the phrase, then (quantity, number as printed) for each number in it.
# Quantities: "<bank> <arm> <fold|lo|hi|sn|cert|top|median>", "<bank> both fold, step 100",
# "<bank> both sn / cert", "<bank> seed fold <min|max>, <arm>", "password anchor / mask, seeds' geometric mean".
TEXT_NUM = [
    (r"the five seeds' improvement factors ranged from $36$ to $1{,}056$, and their pooled factor was $185$",
     [("password seed fold min, mask", "36"), ("password seed fold max, mask", "1056"), ("password mask fold", "185")]),
    (r"It was $4.67$ on the password bank and $6.24$ on WildChat.",
     [("password both cert", "4.67"), ("wildchat both cert", "6.24")]),
    (r"fell by $6.0$ nats on the password bank and $3.9$ on WildChat, against $102.5$ and $105.6$ with the mask alone",
     [("password both top", "6.0"), ("wildchat both top", "3.9"), ("password mask top", "102.5"),
      ("wildchat mask top", "105.6")]),
    (r"the median fell further ($14.3$ nats) than the highest", [("password both median", "14.3")]),
    (r"by factors of $4.41$ and $3.17$", [("password both fold", "4.41"), ("wildchat both fold", "3.17")]),
    (r"($3.45$ and $3.18$, or $0.74$ and $0.51$ of the best straight tail's factor)",
     [("password both sn", "3.45"), ("wildchat both sn", "3.18"), ("password both sn / cert", "0.74"),
      ("wildchat both sn / cert", "0.51")]),
    (r"the pooled improvement factor rose from $3.33$ to $4.41$ on the password bank and from $2.77$ to $3.17$ on WildChat",
     [("password both fold, step 100", "3.33"), ("password both fold", "4.41"),
      ("wildchat both fold, step 100", "2.77"), ("wildchat both fold", "3.17")]),
    (r"(spread-normalized factors of $0.20$ on the password bank and $0.33$ on WildChat)",
     [("password anchor sn", "0.20"), ("wildchat anchor sn", "0.33")]),
    (r"($16.5$ against $105.6$ nats at the top of the tail, and $3.7$ against $90.2$ at the median)",
     [("wildchat anchor top", "16.5"), ("wildchat mask top", "105.6"), ("wildchat anchor median", "3.7"),
      ("wildchat mask median", "90.2")]),
    (r"its improvement factor was close to the mask's ($5.63$ against $6.39$)",
     [("wildchat anchor fold", "5.63"), ("wildchat mask fold", "6.39")]),
    (r"($101.5$ against $102.5$ nats at the top)", [("password anchor top", "101.5"), ("password mask top", "102.5")]),
    (r"($1{,}091$ against $185$ pooled, or $2.2$ times larger by the geometric mean of the seeds' factors)",
     [("password anchor fold", "1091"), ("password mask fold", "185"),
      ("password anchor / mask, seeds' geometric mean", "2.2")]),
    (r"Its pooled improvement factors were $0.82$ on the password bank ($0.44$--$1.36$, with three seeds above one and two below)"
     r" and $0.73$ on WildChat ($0.52$--$1.03$, with one of its four seeds above one)",
     [("password spread fold", "0.82"), ("password spread lo", "0.44"), ("password spread hi", "1.36"),
      ("wildchat spread fold", "0.73"), ("wildchat spread lo", "0.52"), ("wildchat spread hi", "1.03")]),
    (r"($89.5$ against $102.5$ nats at the top of the tail, and $68.2$ against $68.3$ at the median)",
     [("password spread top", "89.5"), ("password mask top", "102.5"), ("password spread median", "68.2"),
      ("password mask median", "68.3")]),
    (r"it lowered the highest log-probability by $65.6$ nats, against $105.6$ with the mask alone",
     [("wildchat spread top", "65.6"), ("wildchat mask top", "105.6")]),
    (r"lowered the risk most on both banks ($102.5$ and $105.6$ nats at the top of the tail)",
     [("password mask top", "102.5"), ("wildchat mask top", "105.6")]),
]
# "Training at other margins": the lowest and highest KL of each bank's three margin runs, as printed.
MARGIN_KL = {"password": ("4.5e-4", "4.8e-4"), "wildchat": ("3.6e-5", "6.0e-5")}

# x-limits and ticks per panel: the password arms span only 4.6e-4 to 1.2e-2, so that panel is zoomed in to keep
# the three arms near 1e-2 apart.
KL_XLIM = {"password": (2e-4, 2e-2), "wildchat": (1e-5, 1e2)}
KL_TICKS = {"password": [1e-3, 1e-2], "wildchat": [1e-4, 1e-2, 1, 1e2]}
YLIM = (0.08, 3000)
# Arms: color, marker, and the name of the table's row (the legend adds "primary method" to the mask's row, "None").
# Hue marks the spread constraint (blue without it, red and amber with it), and lightness the mask (darker)
# against the anchor constraint (lighter).
ARM = {"mask": ("#2F5C99", "o", "None (primary method)"),
       "anchor": ("#3E8FD1", "s", "Anchor constraint"),
       "spread": ("#B04A3A", "^", "Spread constraint"),
       "both": ("#D08A1E", "D", "Both")}
# Where each panel labels its reference lines: (x, ha) of "best straight tail" (above its line) and of
# "no improvement" (below the line at one), chosen so that no label touches a marker or an interval.
REF_LABEL_AT = {"password": ((1.6e-3, "center"), (1.6e-3, "center")),
                "wildchat": ((1.4e-5, "left"), (1.4e-5, "left"))}


def printed(value, text, what=None):
    """Assert that value rounds to the number printed as text ("185", "0.20", "12.1", "5.7e-3")."""
    mantissa, _, exponent = text.partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    tol = 0.5 * 10.0 ** (int(exponent or 0) - decimals)
    assert abs(value - float(text)) <= tol * (1 + 1e-9), (what, value, text) if what else (value, text)


def gmean(values):
    return math.exp(np.mean([math.log(v) for v in values]))


def kl_source(ds, margins=False):
    """The file and column that hold bank ds's per-run KL under KL_TEMPLATE and PASSWORD_CUT."""
    assert KL_TEMPLATE in ("on", "off") and PASSWORD_CUT in ("right", "left"), (KL_TEMPLATE, PASSWORD_CUT)
    if KL_TEMPLATE == "on" and (ds == "wildchat" or PASSWORD_CUT == "right"):
        return ("margin_table.csv" if margins else "kl_table.csv"), "kl_single_token"
    left = "_left" if ds == "password" and PASSWORD_CUT == "left" else ""
    return f"kl_thinking_off_{'margins' if margins else ds}{left}.csv", "kl_" + KL_TEMPLATE


def read_kl(rd, names):
    """Per-seed KL of the four arms' runs at steps 100 and 200: {(bank, step, arm): {seed: KL}}."""
    kl = defaultdict(dict)
    for ds in ("password", "wildchat"):
        fn, col = kl_source(ds)
        for r in csv.DictReader(open(rd / fn)):
            # kl_table.csv names an arm's 200-step rows "<arm>, 200 steps" and gives the step as "steps".
            arm = r["arm"].removesuffix(", 200 steps")
            if r["dataset"] == ds and arm in names:
                kl[(ds, r["steps"] if fn == "kl_table.csv" else r["step"], names[arm])][int(r["seed"])] = float(r[col])
    return kl


def minimal_change(rd):
    """Check the table and the paragraphs against the readouts in rd, and draw the figure."""
    names = {"anchpen + band (main)": "both", "IOG alone": "mask", "anchpen, no band": "anchor",
             "IOG + band": "spread"}
    steps = {"password": STEP, "wildchat": STEP}
    # Pooled and per-seed folds at 200 steps.
    folds = {}
    for r in csv.DictReader(open(rd / "strict_table_200.csv")):
        if r["dataset"] not in steps or r["step"] != STEP or r["arm"] not in names:
            continue
        f = folds.setdefault((r["dataset"], names[r["arm"]]), {"seeds": {}})
        if r["seed"] == "pooled":
            f.update(fold=float(r["asread_fold"]), lo=float(r["asread_lo"]), hi=float(r["asread_hi"]),
                     sn=float(r["fold"]), cert=float(r["cert_fold"]))
        else:
            f["seeds"][int(r["seed"])] = float(r["asread_fold"])
    # Both terms at step 100 (strict_table.csv), for "Every seed improved between 100 and 200 steps".
    both100 = {}
    for r in csv.DictReader(open(rd / "strict_table.csv")):
        if r["arm"] == "anchpen + band (main)" and r["step"] == "100":
            both100.setdefault(r["dataset"], {})[r["seed"]] = float(r["asread_fold"])
    kl_steps = read_kl(rd, names)
    kl = {(ds, a): kl_steps[(ds, step, a)] for ds, step in steps.items() for a in names.values()
          if (ds, step, a) in kl_steps}
    # Falls at the top and median: mean over seeds of each seed's mean over held-out pools (falls_table.csv).
    falls = {}
    for r in csv.DictReader(open(rd / "falls_table.csv")):
        if r["seed"] == "mean" and r["step"] == steps.get(r["dataset"]) and r["arm"] in names:
            key = (r["dataset"], names[r["arm"]])
            assert int(r["n_seeds"]) == SEEDS.get(key, 5), r
            falls[key] = (float(r["fall_top"]), float(r["fall_median"]))
    assert set(TABLE) == set(folds) == set(kl) == set(falls), (sorted(folds), sorted(kl), sorted(falls))
    for key, (fold, lo, hi, sn, kl_text, top, median) in TABLE.items():
        f = folds[key]
        assert len(kl[key]) == SEEDS.get(key, 5) and set(kl[key]) == set(f["seeds"]), (key, sorted(kl[key]), sorted(f["seeds"]))
        for value, text in [(f["fold"], fold), (f["lo"], lo), (f["hi"], hi), (f["sn"], sn), (gmean(kl[key].values()), kl_text),
                            (falls[key][0], top), (falls[key][1], median)]:
            printed(value, text, key)
    print(f"checked: the appendix's table (both banks at 200 steps) matches strict_table_200.csv, falls_table.csv and"
          f" the KL with thinking {KL_TEMPLATE} (password prompts cut from the {PASSWORD_CUT})")

    # ---- KL numbers in the paragraphs
    def gkl(ds, arm, step=None):
        return gmean(kl_steps[(ds, step or steps[ds], arm)].values())
    q = {f"{ds} {a}": gkl(ds, a) for ds in steps for a in names.values()}
    q.update({"password mask / both": gkl("password", "mask") / gkl("password", "both"),
              "wildchat mask / both": gkl("wildchat", "mask") / gkl("wildchat", "both"),
              "password both, step 200 / step 100": gkl("password", "both") / gkl("password", "both", "100"),
              "wildchat both, step 200 / step 100": gkl("wildchat", "both") / gkl("wildchat", "both", "100"),
              "wildchat mask / anchor": gkl("wildchat", "mask") / gkl("wildchat", "anchor"),
              "password anchor / mask, step 100": gkl("password", "anchor", "100") / gkl("password", "mask", "100"),
              "password spread / both": gkl("password", "spread") / gkl("password", "both"),
              "wildchat spread / both": gkl("wildchat", "spread") / gkl("wildchat", "both")})
    for phrase, key, text in TEXT_KL:
        printed(q[key], text, key)
    # Claims without a number of their own, each under the phrase that makes it.
    # "it moved the model furthest of the four arms"
    assert max(names.values(), key=lambda a: q[f"password {a}"]) == "spread", "another arm moved the password model furthest"
    # "improved the forecast at a KL divergence below $5\times10^{-4}$ on both banks"
    assert max(q["password both"], q["wildchat both"]) < 5e-4, "both terms' KL is no longer below 5e-4 on both banks"
    # "With both terms, the trained models moved least on both banks"
    for ds in steps:
        assert min(names.values(), key=lambda a: q[f"{ds} {a}"]) == "both", (ds, "another arm moved the model least")
    print("checked: the KL numbers in the appendix's paragraphs")

    # ---- other numbers in the paragraphs
    v = {}
    for (ds, arm), f in folds.items():
        v.update({f"{ds} {arm} {c}": f[c] for c in ["fold", "lo", "hi", "sn", "cert"]})
        v[f"{ds} {arm} top"], v[f"{ds} {arm} median"] = falls[(ds, arm)]
        v[f"{ds} seed fold min, {arm}"] = min(f["seeds"].values())
        v[f"{ds} seed fold max, {arm}"] = max(f["seeds"].values())
    for ds in steps:
        v[f"{ds} both fold, step 100"] = both100[ds]["pooled"]
        v[f"{ds} both sn / cert"] = folds[(ds, "both")]["sn"] / folds[(ds, "both")]["cert"]
    v["password anchor / mask, seeds' geometric mean"] = (gmean(folds[("password", "anchor")]["seeds"].values())
                                                          / gmean(folds[("password", "mask")]["seeds"].values()))
    for phrase, numbers in TEXT_NUM:
        for key, text in numbers:
            printed(v[key], text, key)
    # Claims without a number of their own, each under the phrase that makes it.
    # "Every seed improved between $100$ and $200$ steps on both banks"
    for ds in steps:
        seeds100 = {int(s): x for s, x in both100[ds].items() if s != "pooled"}
        assert set(seeds100) == set(folds[(ds, "both")]["seeds"]), (ds, sorted(seeds100))
        assert all(folds[(ds, "both")]["seeds"][s] > x for s, x in seeds100.items()), (ds, "a seed got worse")
    # "On the password bank, the median fell further ($14.3$ nats) than the highest"
    assert v["password both median"] > v["password both top"]
    # "Most or all of that gain remained after spread normalization"
    assert all(0.5 < v[f"{ds} both sn"] / v[f"{ds} both fold"] for ds in steps)
    # "the anchor constraint improved the forecast by large factors, but none of the gain remained after spread normalization" and
    # "although none of its gain remained after spread normalization"
    assert all(v[f"{ds} anchor sn"] < 1 for ds in steps)
    # "with three seeds above one and two below"
    assert sorted(x > 1 for x in folds[("password", "spread")]["seeds"].values()) == [False] * 2 + [True] * 3
    # "with one of its four seeds above one"
    assert sorted(x > 1 for x in folds[("wildchat", "spread")]["seeds"].values()) == [False] * 3 + [True]
    # "the spread constraint gave no forecast gain over the base model"
    assert all(v[f"{ds} spread lo"] < 1 for ds in steps)
    # "the primary method of Section~\ref{sec:method} lowered the risk most on both banks"
    for ds in steps:
        assert max(names.values(), key=lambda a: v[f"{ds} {a} top"]) == "mask", (ds, "another arm lowered the top most")
    # "with nearly the mask's improvement factor at a far smaller change to the model"
    assert v["wildchat anchor fold"] / v["wildchat mask fold"] > 0.8 and q["wildchat mask / anchor"] > 100
    print("checked: the other numbers in the appendix's paragraphs")

    # ---- the figure
    fig, axs = plt.subplots(1, 2, figsize=(maf.WIDTH, 3.1), sharey=True, layout="constrained")
    for ax, ds in zip(axs, ["password", "wildchat"]):
        for arm in ["mask", "anchor", "spread", "both"]:
            col, mk, _ = ARM[arm]
            f = folds[(ds, arm)]
            x = gkl(ds, arm)
            assert KL_XLIM[ds][0] < x < KL_XLIM[ds][1] and YLIM[0] < min(f["sn"], f["lo"]) and f["hi"] < YLIM[1], (ds, arm)
            if abs(math.log10(f["fold"] / f["sn"])) > 0.3:   # arrow from the fold to the spread-normalized fold
                ax.annotate("", xy=(x, f["sn"]), xytext=(x, f["fold"]), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, shrinkA=4.5, shrinkB=4.5,
                                            mutation_scale=7))
            # The open marker is drawn larger and below the filled one, so that a ring shows when they coincide.
            ax.plot(x, f["sn"], marker=mk, ms=8.5, mfc="white", mec=col, mew=1.2, lw=0, zorder=4)
            ax.errorbar(x, f["fold"], yerr=[[f["fold"] - f["lo"]], [f["hi"] - f["fold"]]], color=col, marker=mk,
                        ms=6.5, mec="white", mew=0.9, elinewidth=1.1, capsize=2.5, zorder=5)
        cert = folds[(ds, "both")]["cert"]
        ax.axhline(1, color=maf.GRAY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.axhline(cert, color=maf.GRAY, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
        (cx, cha), (nx, nha) = REF_LABEL_AT[ds]
        ax.text(cx, cert * 1.13, "best straight tail", fontsize=7, color=maf.GRAY, ha=cha, va="bottom")
        ax.text(nx, 1 / 1.13, "no improvement", fontsize=7, color=maf.GRAY, ha=nha, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*KL_XLIM[ds])
        ax.set_ylim(*YLIM)
        ax.set_xticks(KL_TICKS[ds])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_title({"password": "Password bank", "wildchat": "WildChat"}[ds], fontsize=9)
        ax.set_xlabel("KL divergence from the base model")
        ax.grid(axis="y", which="major", color="#EBEBEB", lw=0.6)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("Pooled improvement factor")
    # Up and to the left is better: a larger improvement factor at a smaller change to the model.
    # The password panel's upper left is empty, so the arrow goes there.
    axs[0].annotate("better", xy=(0.05, 0.95), xycoords="axes fraction", xytext=(0.25, 0.70),
                    textcoords="axes fraction", ha="center", va="center", fontsize=8, color=maf.GRAY,
                    arrowprops=dict(arrowstyle="-|>", color=maf.GRAY, lw=1.0, shrinkA=3, shrinkB=0,
                                    mutation_scale=9))
    arms = [plt.Line2D([], [], color=ARM[a][0], marker=ARM[a][1], lw=0, ms=6, mec="white", mew=0.9, label=ARM[a][2])
            for a in ["mask", "anchor", "spread", "both"]]
    reads = [plt.Line2D([], [], color=maf.INK, marker="o", lw=0, ms=5.5, mec="white", mew=0.9, label="Improvement factor (95% interval)"),
             plt.Line2D([], [], color=maf.INK, marker="o", lw=0, ms=6.5, mfc="white", mew=1.2,
                        label="Spread-normalized factor")]
    # One legend in three columns, filled column by column: the arms in the first two, the two reads in the third.
    fig.legend(handles=arms + reads, loc="outside lower center", ncol=3, frameon=False, fontsize=8, handletextpad=0.3,
               columnspacing=2.0)
    maf.save(fig, "minimal_change_fold_kl_rebuild.pdf")
    return folds


def margin_ablation(rd):
    """Check the numbers of "Training at other margins" against margin_table.csv and the KL files in rd."""
    rows = {(r["dataset"], r["train_margin"]): r for r in csv.DictReader(open(rd / "margin_table.csv"))}
    # The margin runs stopped at step 200 on the password bank and step 100 on WildChat, so they are compared
    # with the main arm's five seeds at those steps.
    seed_folds = defaultdict(list)
    for fn, ds, step in [("strict_table_200.csv", "password", "200"), ("strict_table.csv", "wildchat", "100")]:
        for r in csv.DictReader(open(rd / fn)):
            if (r["dataset"], r["step"], r["arm"]) == (ds, step, "anchpen + band (main)") and r["seed"] != "pooled":
                seed_folds[ds].append(float(r["asread_fold"]))
    assert {r["step"] for (d, _), r in rows.items() if d == "wildchat"} == {"100"}, "WildChat margin runs moved"
    # (fold, spread-normalized fold at the 0.13-nat read margin) per (bank, training margin); one seed.
    want = {("password", "0.06"): (4.28, 3.72), ("password", "0.13"): (4.43, 3.59),
            ("password", "0.25"): (4.64, 3.89), ("wildchat", "0.06"): (2.65, 2.99),
            ("wildchat", "0.13"): (2.49, 2.84), ("wildchat", "0.25"): (2.56, 2.82)}
    assert set(rows) == set(want), sorted(set(rows) ^ set(want))
    for key, (fold, sn) in want.items():
        close(float(rows[key]["asread_fold"]), fold, 0.005)
        close(float(rows[key]["sn013_fold"]), sn, 0.005)
    # KL under KL_TEMPLATE and PASSWORD_CUT. The recomputation files identify a margin run by its run name.
    margin_of = {r["run"]: key for key, r in rows.items()}
    kl = {}
    for ds in ("password", "wildchat"):
        fn, col = kl_source(ds, margins=True)
        for r in csv.DictReader(open(rd / fn)):
            key = (ds, r["train_margin"]) if fn == "margin_table.csv" else margin_of.get(r["run"])
            if r["dataset"] == ds and key:
                assert r["step"] == rows[key]["step"], (r["run"], r["step"])
                kl[key] = float(r[col])
    assert set(kl) == set(rows), sorted(kl)
    # "at KL divergences between ... and ...": the lowest and highest KL of each bank's three runs.
    for ds, (lo, hi) in MARGIN_KL.items():
        v = [k for (d, _), k in kl.items() if d == ds]
        printed(min(v), lo, ds)
        printed(max(v), hi, ds)
    # "within 9% of one another and inside the range of the five seeds' folds at 0.13 nats (3.83-6.17, 2.45-4.16)"
    for ds, (lo, hi) in {"password": (3.83, 6.17), "wildchat": (2.45, 4.16)}.items():
        seeds = seed_folds[ds]
        assert len(seeds) == 5, seeds
        close(min(seeds), lo, 0.005)
        close(max(seeds), hi, 0.005)
        for col in ["asread_fold", "sn013_fold"]:
            v = [float(r[col]) for (d, _), r in rows.items() if d == ds]
            assert max(v) / min(v) < 1.09, (ds, col, v)
        v = [float(r["asread_fold"]) for (d, _), r in rows.items() if d == ds]
        assert min(seeds) <= min(v) and max(v) <= max(seeds), (ds, v, seeds)
    print(f"checked: training-margin numbers match margin_table.csv and the KL with thinking {KL_TEMPLATE}"
          f" (password prompts cut from the {PASSWORD_CUT})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--readouts", type=Path, default=maf.RESULTS_ROOT / "lquant_dynamics" / "readouts",
                    help="directory of the minimal-change readouts (default: $RESULTS_ROOT/lquant_dynamics/readouts)")
    rd = ap.parse_args().readouts
    maf.apply_paper_style()
    plt.rcParams.update({"axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    maf.BUILD.mkdir(parents=True, exist_ok=True)
    minimal_change(rd)
    margin_ablation(rd)


if __name__ == "__main__":
    main()
