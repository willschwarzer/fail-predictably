"""Forward-direction (fit-size m, deploy-size N) baseline forecast-error grid.

The training-side experiments showed that baseline forecast error depends
strongly on the cell (fit size m, deploy size N) and on the plotting-position
convention. This script maps that dependence directly, with no training: for a
grid of (m, N) cells, subsample disjoint fit/deploy sets from a cached score
pool, fit the FRLMB OLS tail on the fit set's top-k transformed scores,
forecast the deploy max, and compare to the realized deploy max. The outputs
are signed-error heatmaps over (m, N) -- warm = over-prediction (safe), cool =
under-prediction (dangerous) -- one per estimator, so we can see whether the
under-prediction "magically disappears" as the cell grows or the deploy ratio
shrinks, and where the over/under crossover sits per model size.

Because log-prob scores are bounded above by 0, the tail is asymptotically
reversed-Weibull, not Gumbel; the psi-linear extrapolation must eventually
over-predict at large enough N. The grid locates that regime change.

Estimators (all fit on the top ols_k of the m fit scores, transformed to
psi = -log(-log p)):
  inverse_weibull    -- the pipeline estimator (regress log-position on score,
                        invert at survival 1/N), Weibull positions j/(m+1).
                        This is the paper/training convention.
  inverse_gringorten -- same, Gringorten positions (j-0.44)/(m+0.12).
  digamma_forward    -- the closed-form rank-unbiased estimator (see the
                        paper's appendix "Plotting positions"):
                        positions p_j = exp(psi(j) - psi(m+1)), regress score
                        on z_j = log p_j (the classical probability-plot
                        direction, linear in the data), read the fitted line
                        off at z* = psi(1) - psi(N+1). Exactly unbiased on
                        exponential tails, so any error it shows is genuine
                        tail-shape misfit rather than estimator rank bias.

Sampling: per rep, one shared permutation of the pool; fit sets are nested
prefixes (perm[:m]) and deploy sets nested suffixes (perm[n-N:]), disjoint
whenever m + N <= n (exactly the feasible triangle). Sharing the permutation
across cells is common-random-numbers: each cell's marginal distribution is
unchanged, and adjacent cells differ only by the quantity being varied, which
makes the heatmaps smooth. Deploy maxima for every N come from one suffix
cumulative max.

Score sources (pick one):
  --root DIR [--root DIR2 ...]  discover <model>/harmful/per_token_log_p.npz
        (works for both the 3000-prompt tail_shape_len8192 pools and the
        full-pool wildchat_fullpool_logits outputs); score = logsumexp over
        all union columns.
  --scores-csv F --model-name M  the scorer's CSV (log_p_harmful column).
  --logprob-bin F --bin-meta F --token-ids-json harmful_set.json
        recompute a union score from a full-vocab fp16 logit dump.

The inverse estimators are validated against ols_utils.fit_ols_tail_from_scores
at startup (--selfcheck, on by default).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
REPO_SRC = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_SRC))

from forecast_experiment.ols_utils import (  # noqa: E402
    OLS_SXX_TOL,
    fit_ols_tail_from_scores,
)

LN10 = math.log(10.0)
TRANSFORM_EPS = 1e-8  # matches the pipeline's score_eps
SCORE_MODE = "gumbel_log_prob"

DEFAULT_M_GRID = "11,22,44,88,176,352,704,1408,2816,5632,11264,22528"
DEFAULT_N_GRID = "28,56,111,223,446,891,1782,3564,7128,14256,28512,57024,114048"
METHODS = ("inverse_weibull", "inverse_gringorten", "digamma_forward")


# --------------------------------------------------------------------------- #
# Transforms and estimators
# --------------------------------------------------------------------------- #
def psi_transform(logp: np.ndarray | float) -> np.ndarray | float:
    """psi = -log(-log p) with the pipeline's eps clamp (log p in nats, <= 0)."""
    neg = np.maximum(-np.asarray(logp, dtype=np.float64), TRANSFORM_EPS)
    return -np.log(neg)


def psi_inverse(x: np.ndarray | float) -> np.ndarray | float:
    return -np.exp(-np.asarray(x, dtype=np.float64))


def _log_positions_inverse(k: int, m: int, convention: str) -> np.ndarray:
    j = np.arange(1, k + 1, dtype=np.float64)
    if convention == "weibull":
        p = j / (m + 1.0)
    elif convention == "gringorten":
        p = (j - 0.44) / (m + 0.12)
    else:
        raise ValueError(convention)
    return np.log(np.clip(p, 1e-300, 1.0))


def predict_inverse(top_x: np.ndarray, m: int, n_deploy: int,
                    convention: str) -> float:
    """Pipeline estimator: OLS of log-position on transformed score, inverted
    at survival 1/N. Mirrors fit_ols_tail_from_scores exactly (incl. the
    flat-tail branch)."""
    k = top_x.size
    y = _log_positions_inverse(k, m, convention)
    x_mean = top_x.mean()
    y_mean = y.mean()
    xc = top_x - x_mean
    sxx = float(np.dot(xc, xc))
    if sxx <= OLS_SXX_TOL:
        return float(top_x[0])
    b = float(np.dot(xc, y - y_mean)) / sxx
    a = y_mean - b * x_mean
    return (math.log(1.0 / n_deploy) - a) / b


def predict_digamma_forward(top_x: np.ndarray, m: int, n_deploy: int) -> float:
    """Rank-unbiased forward estimator: regress score on z_j = psi(j)-psi(m+1),
    read the fitted line off at z* = psi(1) - psi(n_deploy+1)."""
    from scipy.special import digamma

    k = top_x.size
    j = np.arange(1, k + 1, dtype=np.float64)
    z = digamma(j) - digamma(m + 1.0)
    z_mean = z.mean()
    zc = z - z_mean
    szz = float(np.dot(zc, zc))
    x_mean = top_x.mean()
    if szz <= 0.0:
        return float(x_mean)
    d = float(np.dot(zc, top_x - x_mean)) / szz
    c = x_mean - d * z_mean
    z_star = float(digamma(1.0) - digamma(n_deploy + 1.0))
    return c + d * z_star


def selfcheck_inverse(seed: int = 0) -> None:
    """The local inverse estimator must agree with the pipeline's
    fit_ols_tail_from_scores to float precision."""
    rng = np.random.default_rng(seed)
    for trial in range(20):
        n = int(rng.integers(15, 200))
        logp = -np.abs(rng.normal(8.0, 4.0, size=n)) - 1e-3
        k = 10
        n_dep = int(rng.integers(50, 5000))
        top_x = np.sort(psi_transform(logp))[::-1][:k]
        for conv in ("weibull", "gringorten"):
            ref_pred, _a, _b = fit_ols_tail_from_scores(
                logp.tolist(), n_top=k, plotting=conv,
                deploy_queries=n_dep, score_mode=SCORE_MODE,
            )
            mine = float(psi_inverse(predict_inverse(top_x, n, n_dep, conv)))
            if not math.isclose(mine, float(ref_pred), rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(
                    f"selfcheck failed ({conv}, trial {trial}): {mine} vs {ref_pred}")
    print("[selfcheck] inverse estimators match fit_ols_tail_from_scores", flush=True)


# --------------------------------------------------------------------------- #
# Score loading
# --------------------------------------------------------------------------- #
def union_scores_from_npz(npz_path: Path) -> np.ndarray:
    npz = np.load(npz_path, allow_pickle=True)
    ptlp = np.asarray(npz["per_token_log_p"], dtype=np.float64)
    return np.logaddexp.reduce(ptlp, axis=1)


def union_scores_from_bin(bin_path: Path, meta_path: Path,
                          token_ids: list[int]) -> np.ndarray:
    meta = json.loads(meta_path.read_text())
    n_rows, vocab = meta["shape"]
    mm = np.memmap(bin_path, dtype=np.float16, mode="r", shape=(n_rows, vocab))
    cols = np.asarray(sorted(set(int(t) for t in token_ids)), dtype=np.int64)
    out = np.empty(n_rows, dtype=np.float64)
    for i0 in range(0, n_rows, 4096):
        chunk = np.asarray(mm[i0:i0 + 4096][:, cols], dtype=np.float64)
        out[i0:i0 + chunk.shape[0]] = np.logaddexp.reduce(chunk, axis=1)
    return out


def scores_from_csv(csv_path: Path) -> np.ndarray:
    vals: list[float] = []
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            vals.append(float(row["log_p_harmful"]))
    return np.asarray(vals, dtype=np.float64)


def discover_models(roots: list[Path], models: list[str] | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for root in roots:
        if not root.exists():
            print(f"[warn] root does not exist: {root}", flush=True)
            continue
        for m in sorted(root.iterdir()):
            npz = m / "harmful" / "per_token_log_p.npz"
            if npz.exists() and (models is None or m.name in models):
                if m.name in out:
                    print(f"[warn] {m.name} in multiple roots; keeping first", flush=True)
                    continue
                out[m.name] = union_scores_from_npz(npz)
    return out


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def run_grid(scores: np.ndarray, m_grid: list[int], n_grid: list[int],
             *, ols_k: int, methods: list[str], reps: int, seed: int,
             ceiling_dec: float = 0.5) -> dict:
    n_pool = int(scores.size)
    ms = [m for m in m_grid if ols_k <= m < n_pool]
    ns = [n for n in n_grid if n < n_pool]
    feasible = np.array([[m + n <= n_pool for n in ns] for m in ms])
    err = {meth: np.full((len(ms), len(ns), reps), np.nan) for meth in methods}
    # Headroom: both forecast and actual are <= 0, so |err| <= the
    # distance-to-ceiling of whichever is farther from 0. A cell whose median
    # headroom is below the effect size of interest cannot display a meaningful
    # error -- its low error is ceiling-imposed, not evidence the tail model
    # extrapolates well.
    hdr = {meth: np.full((len(ms), len(ns), reps), np.nan) for meth in methods}

    rng = np.random.default_rng(seed)
    for r in range(reps):
        s_perm = scores[rng.permutation(n_pool)]
        # suffix max: deploy actual for every N in one pass
        suffix_max = np.maximum.accumulate(s_perm[::-1])[::-1]
        for i, m in enumerate(ms):
            fit = s_perm[:m]
            k = min(ols_k, m)
            top_logp = np.partition(fit, m - k)[m - k:]
            top_x = np.sort(psi_transform(top_logp))[::-1]
            for j, n_dep in enumerate(ns):
                if not feasible[i, j]:
                    continue
                actual = float(suffix_max[n_pool - n_dep])
                for meth in methods:
                    if meth == "inverse_weibull":
                        px = predict_inverse(top_x, m, n_dep, "weibull")
                    elif meth == "inverse_gringorten":
                        px = predict_inverse(top_x, m, n_dep, "gringorten")
                    elif meth == "digamma_forward":
                        px = predict_digamma_forward(top_x, m, n_dep)
                    else:
                        raise ValueError(meth)
                    pred_logp = float(psi_inverse(px))
                    err[meth][i, j, r] = (pred_logp - actual) / LN10
                    hdr[meth][i, j, r] = -min(pred_logp, actual) / LN10

    q_ceiling = float(np.mean(scores > -ceiling_dec * LN10))
    out = {"m_grid": ms, "n_grid": ns, "feasible": feasible, "n_pool": n_pool,
           "reps": reps, "ols_k": ols_k, "ceiling_dec": ceiling_dec,
           "q_ceiling": q_ceiling,
           "n_star": (1.0 / q_ceiling if q_ceiling > 0 else float("inf")),
           "methods": {}}
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for meth in methods:
            e = err[meth]
            out["methods"][meth] = {
                "err_decades": e,
                "median": np.nanmedian(e, axis=2),
                "p25": np.nanquantile(e, 0.25, axis=2),
                "p75": np.nanquantile(e, 0.75, axis=2),
                "frac_under": np.nanmean(e < 0.0, axis=2),
                "median_headroom": np.nanmedian(hdr[meth], axis=2),
            }
    return out


def regime_matrix(ms: list[int], ns: list[int], feasible: np.ndarray,
                  ols_k: int) -> np.ndarray:
    """0 = forward extrapolation (N >= m, the deployment question);
    1 = interpolation (m/k <= N < m: the read-off survival level 1/N sits
    inside the fitted ranks' positions); 2 = backward extrapolation
    (N < m/k: 1/N is more common than any fitted rank -- where the tiny-slope
    blowups live); 3 = infeasible (m + N > pool)."""
    reg = np.zeros((len(ms), len(ns)), dtype=int)
    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            if not feasible[i, j]:
                reg[i, j] = 3
            elif n < m / ols_k:
                reg[i, j] = 2
            elif n < m:
                reg[i, j] = 1
    return reg


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _log_edges(vals: list[int]) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float64)
    inner = np.sqrt(v[:-1] * v[1:])
    lo = v[0] ** 2 / inner[0] if inner.size else v[0] / 1.4
    hi = v[-1] ** 2 / inner[-1] if inner.size else v[-1] * 1.4
    return np.concatenate([[lo], inner, [hi]])


def plot_grid(result: dict, method: str, model: str, out_base: Path,
              mark: tuple[int, int] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ms, ns = result["m_grid"], result["n_grid"]
    stats = result["methods"][method]
    regime = regime_matrix(ms, ns, result["feasible"], result["ols_k"])
    shown = regime == 0
    med = np.where(shown, stats["median"], np.nan)
    fu = np.where(shown, stats["frac_under"], np.nan)
    ceiling_dec = result["ceiling_dec"]
    ceil_cells = shown & (stats["median_headroom"] < ceiling_dec)

    xe, ye = _log_edges(ms), _log_edges(ns)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)

    # Region underlay: neutral grays at distinct lightness steps (state, not
    # data); the backward-extrapolation region additionally gets a hatch so the
    # distinction survives grayscale/CVD.
    region_cmap = ListedColormap(["0.78", "0.55", "0.95"])
    region_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], region_cmap.N)
    region_masked = np.ma.masked_where(regime == 0, regime)
    back_masked = np.ma.masked_where(regime != 2, np.ones_like(regime, dtype=float))

    # Color limits from the extrapolation cells only (the masked regimes hold
    # the tiny-slope blowups); extreme cells saturate rather than set the scale.
    absmed = np.abs(med[np.isfinite(med)])
    vmax = float(np.quantile(absmed, 0.98)) if absmed.size else 1.0
    vmax = float(np.clip(vmax, 0.5, 10.0))
    cmap_err = plt.get_cmap("RdBu_r").copy()
    cmap_err.set_bad("none")
    cmap_fu = plt.get_cmap("RdBu").copy()
    cmap_fu.set_bad("none")

    for ax in axes:
        ax.pcolormesh(xe, ye, region_masked.T, cmap=region_cmap, norm=region_norm)
        ax.pcolor(xe, ye, back_masked.T, facecolor="none", edgecolor="0.35",
                  hatch="///", linewidth=0.0)

    pm = axes[0].pcolormesh(xe, ye, med.T, cmap=cmap_err,
                            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    if np.isfinite(med).sum() > 3:
        X, Y = np.meshgrid(ms, ns)
        axes[0].contour(X, Y, np.ma.masked_invalid(med).T, levels=[0.0],
                        colors="k", linewidths=1.0, linestyles="--")
    fig.colorbar(pm, ax=axes[0], label="median signed error (decades)")
    axes[0].set_title("forecast $-$ actual deploy max\n(warm = over-predict, cool = under-predict)")

    pf = axes[1].pcolormesh(xe, ye, fu.T, cmap=cmap_fu, vmin=0.0, vmax=1.0)
    fig.colorbar(pf, ax=axes[1], label="frac. under-predicted")
    axes[1].set_title("fraction of reps under-predicting")

    # Ceiling-limited cells: median headroom (distance-to-ceiling of the
    # farther of forecast/actual, a hard bound on |error|) below ceiling_dec.
    ii, jj = np.where(ceil_cells)
    if ii.size:
        axes[0].plot(np.asarray(ms)[ii], np.asarray(ns)[jj], linestyle="none",
                     marker="o", markersize=4, markerfacecolor="0.1",
                     markeredgecolor="white", markeredgewidth=0.6, zorder=4)

    n_star = result.get("n_star", float("inf"))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("fit size $m$")
        ax.set_ylabel("deploy size $N$")
        if math.isfinite(n_star) and ye[0] < n_star < ye[-1]:
            ax.axhline(n_star, linestyle=":", color="0.25", linewidth=1.2)
        if mark is not None and ms[0] <= mark[0] <= ms[-1] and ns[0] <= mark[1] <= ns[-1]:
            ax.plot(mark[0], mark[1], marker="*", markersize=13, zorder=5,
                    markerfacecolor="gold", markeredgecolor="k", linestyle="none")

    handles = [
        Patch(facecolor="0.78", label="interpolation ($m/k \\leq N < m$)"),
        Patch(facecolor="0.55", hatch="///", edgecolor="0.35",
              label="backward extrap. ($N < m/k$)"),
        Patch(facecolor="0.95", edgecolor="0.8", label="infeasible ($m+N>$ pool)"),
        Line2D([], [], linestyle="none", marker="o", markersize=5,
               markerfacecolor="0.1", markeredgecolor="white",
               label=f"ceiling-limited (headroom < {ceiling_dec:g} dec)"),
    ]
    if math.isfinite(n_star):
        handles.append(Line2D([], [], linestyle=":", color="0.25",
                              label=f"$N = 1/q_{{ceil}}$ ({n_star:,.0f})"))
    if mark is not None:
        handles.append(Line2D([], [], linestyle="none", marker="*", markersize=10,
                              markerfacecolor="gold", markeredgecolor="k",
                              label=f"paper cell ({mark[0]}/{mark[1]})"))
    try:
        fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    except (ValueError, TypeError):
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
                   bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{model} -- {method} (pool {result['n_pool']}, "
                 f"k={result['ols_k']}, {result['reps']} reps)")
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, action="append", default=None,
                    help="Root(s) holding <model>/harmful/per_token_log_p.npz.")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--scores-csv", type=Path, default=None)
    ap.add_argument("--logprob-bin", type=Path, default=None)
    ap.add_argument("--bin-meta", type=Path, default=None)
    ap.add_argument("--token-ids-json", type=Path, default=None,
                    help="harmful_set.json; its harmful_ids define the union "
                         "for --logprob-bin.")
    ap.add_argument("--model-name", default=None,
                    help="Label for --scores-csv / --logprob-bin input.")
    ap.add_argument("--out-dir", type=Path,
                    default=RESULTS_ROOT / "fit_deploy_error_grid")
    ap.add_argument("--m-grid", default=DEFAULT_M_GRID)
    ap.add_argument("--n-grid", default=DEFAULT_N_GRID)
    ap.add_argument("--ols-k", type=int, default=10)
    ap.add_argument("--methods", nargs="*", default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mark-cell", default="44,891",
                    help="'m,N' cell to star on the heatmaps (the paper "
                         "training cell by default); 'none' to disable.")
    ap.add_argument("--ceiling-dec", type=float, default=0.5,
                    help="Ceiling proximity threshold in decades: cells whose "
                         "median headroom (distance-to-ceiling of the farther "
                         "of forecast/actual, a hard bound on |error|) is below "
                         "this are marked ceiling-limited; also sets the "
                         "near-ceiling pool fraction q behind the N = 1/q line.")
    ap.add_argument("--no-selfcheck", action="store_true", default=False)
    ap.add_argument("--smoke", action="store_true", default=False,
                    help="reps=5, first model only")
    args = ap.parse_args(argv)

    if not args.no_selfcheck:
        selfcheck_inverse()

    pools: dict[str, np.ndarray] = {}
    if args.root:
        pools.update(discover_models(args.root, args.models))
    if args.scores_csv is not None:
        pools[args.model_name or args.scores_csv.stem] = scores_from_csv(args.scores_csv)
    if args.logprob_bin is not None:
        if args.bin_meta is None or args.token_ids_json is None:
            raise SystemExit("--logprob-bin needs --bin-meta and --token-ids-json")
        ids = json.loads(args.token_ids_json.read_text())["harmful_ids"]
        pools[args.model_name or args.logprob_bin.parent.name] = union_scores_from_bin(
            args.logprob_bin, args.bin_meta, ids)
    if not pools:
        raise SystemExit("No score sources given (use --root / --scores-csv / --logprob-bin).")

    reps = args.reps
    if args.smoke:
        reps = 5
        pools = dict(list(pools.items())[:1])

    m_grid = [int(x) for x in args.m_grid.split(",")]
    n_grid = [int(x) for x in args.n_grid.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"_meta": {"m_grid": m_grid, "n_grid": n_grid,
                               "ols_k": args.ols_k, "reps": reps,
                               "seed": args.seed, "methods": args.methods}}
    for model, scores in pools.items():
        print(f"[grid] {model}: pool={scores.size}", flush=True)
        res = run_grid(scores, m_grid, n_grid, ols_k=args.ols_k,
                       methods=args.methods, reps=reps, seed=args.seed,
                       ceiling_dec=args.ceiling_dec)
        mdir = args.out_dir / model
        mdir.mkdir(parents=True, exist_ok=True)
        regime = regime_matrix(res["m_grid"], res["n_grid"], res["feasible"],
                               res["ols_k"])
        extrap = regime == 0
        summary[model] = {"q_ceiling": res["q_ceiling"], "n_star": res["n_star"]}
        for meth in args.methods:
            st = res["methods"][meth]
            np.savez_compressed(
                mdir / f"{meth}_grid.npz",
                m_grid=np.asarray(res["m_grid"]), n_grid=np.asarray(res["n_grid"]),
                feasible=res["feasible"], regime=regime,
                err_decades=st["err_decades"],
                median=st["median"], p25=st["p25"], p75=st["p75"],
                frac_under=st["frac_under"],
                median_headroom=st["median_headroom"],
                n_pool=res["n_pool"], ols_k=res["ols_k"], reps=res["reps"],
                ceiling_dec=res["ceiling_dec"], q_ceiling=res["q_ceiling"],
                n_star=res["n_star"],
            )
            mark = None
            if args.mark_cell and args.mark_cell.lower() != "none":
                mark = tuple(int(v) for v in args.mark_cell.split(","))
            plot_grid(res, meth, model, mdir / f"{meth}_heatmap", mark=mark)
            ceil_n = int(np.sum(extrap & (st["median_headroom"] < args.ceiling_dec)))
            # Ranges over forward-extrapolation cells only; the interpolation /
            # backward-extrapolation regimes are masked in the plots too.
            summary[model][meth] = {
                "median_err_decades_range": [
                    float(np.nanmin(np.where(extrap, st["median"], np.nan))),
                    float(np.nanmax(np.where(extrap, st["median"], np.nan))),
                ],
                "frac_under_range": [
                    float(np.nanmin(np.where(extrap, st["frac_under"], np.nan))),
                    float(np.nanmax(np.where(extrap, st["frac_under"], np.nan))),
                ],
                "n_ceiling_limited_cells": ceil_n,
                "n_extrap_cells": int(extrap.sum()),
            }
            print(f"  {meth:18} extrap median err in "
                  f"[{summary[model][meth]['median_err_decades_range'][0]:+.2f}, "
                  f"{summary[model][meth]['median_err_decades_range'][1]:+.2f}] dec; "
                  f"frac_under in "
                  f"[{summary[model][meth]['frac_under_range'][0]:.2f}, "
                  f"{summary[model][meth]['frac_under_range'][1]:.2f}]; "
                  f"ceiling-limited {ceil_n}/{int(extrap.sum())} "
                  f"(q_ceil={res['q_ceiling']:.4f}, N*={res['n_star']:.0f})",
                  flush=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
