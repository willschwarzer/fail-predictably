"""The figure of the paragraph "Room for improvement" in the minimal-change appendix.

It plots improvement factors against k, the number of fit scores that the forecaster's line is fit to, for tails
that satisfy the two conditions of the spread-normalized factor (no prompt above its base value, and a spread within
delta of sigma_ref):
  - on base-model WildChat task sets at 21 settings (fit sizes 250, 1,000 and 4,000, deployment sets 8.2 to 290
    times larger, eight task sets each): the best straight tail, measured and predicted from rank statistics alone
    (floor_theory.Floor, computed as in grid_theory.py, with the same seed and draws), and the best tail of any shape
    that the numerical search found;
  - on the two banks' held-out task sets: the best straight tail, the best tail of any shape that the search found
    (pooled over the seeds' task sets), and the spread-normalized factor of our minimal-change training at 200 steps.
Each number that the appendix prints about these is hard-coded as printed and asserted against the readouts.

Inputs, in $RESULTS_ROOT/lquant_dynamics/readouts/ (or --readouts), from step 5 of the minimal-change block in
REPRODUCING.md: tail_opt_grid_cells.csv and tail_opt_fleet.csv (the two searches), cert_scale.csv and
strict_table_200.csv. Output: minimal_change_room.pdf in $FIGURES_ROOT (default artifacts/figures/), with a PNG
preview in $FIGURES_ROOT/build/.

Run it as a one-core Slurm CPU job; it takes about two minutes.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "minimal_change" / "readouts"))
import make_appendix_figures as maf  # noqa: E402  (the paper's plot style, output paths and save())
sys.path.insert(0, str(maf.REPO_ROOT / "src"))
from floor_theory import Floor, e_of  # noqa: E402

np, plt = maf.np, maf.plt
BOTH = "#D08A1E"   # "Both" in the minimal-change figure
SHAPE = "#2A9D8F"  # the best tail of any shape
GUIDE = "#C8C8C8"


def printed(value, text, what):
    decimals = len(text.split(".")[1]) if "." in text else 0
    assert abs(value - float(text)) <= 0.5 * 10.0 ** -decimals * (1 + 1e-9), (what, value, text)


def grid(rd, draws=20000, pools=8):
    """Per geometry (M, N, R, k): measured factors and the predicted straight-tail factor (grid_theory.py's computation)."""
    rng = np.random.default_rng(20261104)
    certs = [r for r in csv.DictReader(open(rd / "cert_scale.csv")) if r["source"] == "random" and r["status"] == "ok"]
    rows = list(csv.DictReader(open(rd / "tail_opt_grid_cells.csv")))
    assert {r["band"] for r in rows} == {"involved"} and len(rows) == 21, len(rows)
    pid = {f"rand{i:02d}" for i in range(pools)}
    floors, out = {}, []
    for r in sorted(rows, key=lambda r: (int(r["M"]), float(r["R"]), int(r["k"]))):
        M, N, k = int(r["M"]), int(r["N"]), int(r["k"])
        assert r["completed_pools"] == r["expected_pools"] == str(pools), r["cell"]
        e = e_of(M, N)
        key = (M, N, k, e)
        if key not in floors:
            floors[key] = Floor(M, N, k, e, max(4000, draws // 4) if k >= 1000 else draws, rng, "A")
        fl = floors[key]
        lams = [float(c["cert_lambda"]) for c in certs if c["cell"] == r["cell"] and int(c["k"]) == k and c["pool"] in pid]
        assert len(lams) == pools, (r["cell"], k, len(lams))
        g_str = float(np.mean([fl.F(l)[0] / fl.sigma(l) ** 2 for l in lams]))
        g_cert = float(r["evaluation_certificate_G_mean"])
        fold = float(r["evaluation_certificate_over_base"])
        out.append(dict(M=M, N=N, R=float(r["R"]), k=k, fold=fold, g_cert=g_cert, g_str=g_str,
                        pred=fold * g_cert / g_str, opt=float(r["evaluation_optimized_over_base"])))
        print(f"  M {M:5d} R {float(r['R']):6g} k {k:5d}: best straight tail {fold:6.2f}, predicted {fold * g_cert / g_str:6.2f}"
              f"  (G measured {g_cert:.4f}, predicted {g_str:.4f}, {g_str / g_cert - 1:+.1%}); best tail of any shape {out[-1]['opt']:.1f}")
    return out


def bank_search(rd):
    """Per bank: the search's best tail of any shape, per seed and pooled over the seeds' task sets."""
    sums = defaultdict(lambda: defaultdict(float))
    rows = defaultdict(dict)
    for r in csv.DictReader(open(rd / "tail_opt_fleet.csv")):
        if r["tail"] in ("base", "optimized") and r["run"] != "wildchat_bank":
            assert r["band"] == "involved"
            sums[(r["run"], r["step"])][r["tail"]] += float(r["evaluation_mse"])
            rows[(r["run"], r["step"])][(r["pool"], r["tail"])] = float(r["evaluation_mse"])
    # The password seed-17 run continued to 200 steps in a second directory; its task sets and so its search results
    # are the same as the first directory's, and are counted once.
    assert rows[("lqpw_pw2apsp200_seed17", "110")] == rows[("lqpw_pw2apsp_seed17", "100")]
    out = {}
    for ds, prefix in (("password", "lqpw_pw2apsp_seed"), ("wildchat", "lqdyn_v2e2ap50sp_seed")):
        per = {int(run[len(prefix):]): s for (run, _), s in sums.items() if run.startswith(prefix)}
        seeds = sorted(per)
        assert seeds == ([11, 17, 23] if ds == "password" else [11, 17, 23, 29, 43]), (ds, seeds)
        f = {s: per[s]["base"] / per[s]["optimized"] for s in seeds}
        pooled = sum(per[s]["base"] for s in seeds) / sum(per[s]["optimized"] for s in seeds)
        out[ds] = (pooled, min(f.values()), max(f.values()))
        print(f"  {ds}: best tail of any shape, per seed " + ", ".join(f"s{s} {v:.2f}" for s, v in f.items())
              + f"; pooled {pooled:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--readouts", type=Path, default=maf.RESULTS_ROOT / "lquant_dynamics" / "readouts",
                    help="directory of the minimal-change readouts (default: $RESULTS_ROOT/lquant_dynamics/readouts)")
    ap.add_argument("--name", default="minimal_change_room.pdf")
    args = ap.parse_args()
    rd = args.readouts
    cells = grid(rd)

    # ---- the paragraph's numbers
    by_k = defaultdict(list)
    for c in cells:
        by_k[c["k"]].append(c["fold"])
    for k, (lo, hi) in {10: ("1.7", "3.7"), 50: ("4", "9"), 200: ("22", "31"), 1000: ("88", "88")}.items():
        printed(min(by_k[k]), lo, f"k={k} low")
        printed(max(by_k[k]), hi, f"k={k} high")
    assert {c["M"] for c in cells} == {250, 1000, 4000}, "fit sizes from 250 to 4,000"
    assert {c["R"] for c in cells} == {8.2, 30.0, 70.0, 100.0, 290.0}, "deployment sets 8.2 to 290 times larger"
    worst = max(abs(c["g_str"] / c["g_cert"] - 1) for c in cells)
    assert worst < 0.04, worst
    assert all(c["opt"] > c["fold"] for c in cells), "the search beat the best straight tail in every setting"
    print(f"checked: 1.7-3.7 (k=10), 4-9 (k=50), 22-31 (k=200), 88 (k=1000); prediction within {worst:.1%} in all 21;"
          f" the best tail of any shape is above the best straight tail in all 21")

    # ---- the banks
    bank = {}
    for r in csv.DictReader(open(rd / "strict_table_200.csv")):
        if r["seed"] == "pooled" and r["step"] == "200" and r["arm"] == "anchpen + band (main)":
            bank[r["dataset"]] = (float(r["cert_fold"]), float(r["fold"]))
    printed(bank["password"][0], "4.67", "password best straight tail")
    printed(bank["wildchat"][0], "6.24", "WildChat best straight tail")
    printed(bank["password"][1], "3.45", "password both, spread-normalized")
    printed(bank["wildchat"][1], "3.18", "WildChat both, spread-normalized")
    search = bank_search(rd)
    printed(search["password"][1], "20", "password any shape, lowest seed")
    printed(search["password"][2], "25", "password any shape, highest seed")
    printed(search["wildchat"][1], "9.4", "WildChat any shape, lowest seed")
    printed(search["wildchat"][2], "12.5", "WildChat any shape, highest seed")
    print("checked: the banks' best straight tails (4.67, 6.24), spread-normalized factors (3.45, 3.18) and the"
          " search's best tails (20 to 25, 9.4 to 12.5 by seed)")

    # ---- the figure
    maf.apply_paper_style()
    plt.rcParams.update({"axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})
    maf.BUILD.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.1), layout="constrained")
    # Spread each k's geometries a little along x, in a fixed order, so that markers do not overlap.
    geoms = sorted({(c["M"], c["R"]) for c in cells})
    off = {g: 1.045 ** (i - (len(geoms) - 1) / 2) for i, g in enumerate(geoms)}
    for c in cells:
        x = c["k"] * off[(c["M"], c["R"])]
        # The prediction is a ring around the measured dot; they agree where the dot sits in the ring.
        ax.plot(x, c["pred"], marker="o", ms=7, mfc="none", mec=maf.NAVY, mew=0.8, alpha=0.6, lw=0, zorder=2)
        ax.plot(x, c["fold"], marker="o", ms=3.6, color=maf.NAVY, lw=0, zorder=3)
        ax.plot(x, c["opt"], marker="s", ms=3.8, mfc="none", mec=SHAPE, mew=0.9, lw=0, zorder=3)
    # The banks, to the left of their k.
    for ds, k, label in [("password", 10, "password\nbank"), ("wildchat", 50, "WildChat\nbank")]:
        cert, sn = bank[ds]
        pooled = search[ds][0]
        x = k * 0.68
        ax.plot([x, x], [sn, pooled], color=GUIDE, lw=0.8, zorder=1)
        ax.plot(x, pooled, marker="s", ms=5, mfc="white", mec=SHAPE, mew=1.1, lw=0, zorder=4)
        ax.plot(x, cert, marker="o", ms=5.5, color=maf.NAVY, mec="white", mew=0.6, lw=0, zorder=4)
        ax.plot(x, sn, marker="D", ms=5, color=BOTH, mec=maf.INK, mew=0.5, lw=0, zorder=4)
        ax.text(x / 1.2, math.sqrt(cert * pooled), label, fontsize=7, color=maf.INK, ha="right", va="center",
                linespacing=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.2, 1700)
    ax.set_ylim(1, 420)
    ax.set_xticks([10, 50, 200, 1000])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Number of fit scores $k$")
    ax.set_ylabel("Improvement factor")
    ax.grid(axis="y", which="major", color="#EBEBEB", lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=maf.NAVY, marker="o", ms=3.6, lw=0, label="Best straight tail"),
               plt.Line2D([], [], marker="o", ms=7, mfc="none", mec=maf.NAVY, mew=0.8, alpha=0.6, lw=0,
                          label="Best straight tail, predicted"),
               plt.Line2D([], [], marker="s", ms=3.8, mfc="none", mec=SHAPE, mew=0.9, lw=0,
                          label="Best tail of any shape found"),
               plt.Line2D([], [], color=BOTH, marker="D", ms=5, lw=0, mec=maf.INK, mew=0.5,
                          label="Our minimal-change training\n(spread-normalized)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, handletextpad=0.4,
               columnspacing=1.0)
    maf.save(fig, args.name)


if __name__ == "__main__":
    main()
