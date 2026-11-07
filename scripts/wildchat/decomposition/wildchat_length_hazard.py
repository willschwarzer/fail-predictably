"""Empirical hazard / survival / histogram diagnostic for WildChat risk scores.

Single-panel mode reproduces the original assistant-length plot. Multi-panel
mode (--multi-panel-config) produces a stacked figure across multiple scores
for the appendix's trifecta diagnostic.

For each score:
  - Panel 1: histogram (log-y).
  - Panel 2: empirical survival, log-log (or log-y for signed scores).
  - Panel 3: log hazard vs score. Integer-bin mode (assistant length) uses
    the discrete `h(l) = #{L=l}/#{L>=l}`; log-bin and linear-bin modes use
    the continuous bin-derivative estimator
    `h(s_b) = log(S(s_b)/S(s_{b+1})) / (s_{b+1} - s_b)`.

  Rising hazard => Weibull-class (bounded upper endpoint).
  Flat hazard   => Gumbel-class.
  ~1/s falling  => Pareto/Frechet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Relative "csv" paths in --multi-panel-config resolve against RESULTS_ROOT.
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

def apply_score_mode(arr: np.ndarray, mode: str, *, eps: float = 1e-8) -> np.ndarray:
    """Mirror the analyzer's score-mode transforms, plus ``exp`` for the
    perplexity-from-mean-NLL case."""
    if mode == "identity":
        return arr.astype(np.float64)
    if mode == "log":
        return np.log(np.clip(arr, eps, None)).astype(np.float64)
    if mode == "exp":
        return np.exp(arr).astype(np.float64)
    if mode == "gumbel_log_prob":
        return -np.log(np.clip(-arr, eps, None)).astype(np.float64)
    if mode == "gumbel_prob":
        clipped = np.clip(arr, eps, 1.0 - 1e-16)
        return -np.log(-np.log(clipped)).astype(np.float64)
    raise ValueError(f"Unsupported score mode: {mode}")


def load_score_column(
    csv_path: Path,
    score_column: str,
    *,
    score_mode: str = "identity",
    eps: float = 1e-8,
    as_int: bool = False,
) -> np.ndarray:
    vals: list[float] = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        if score_column not in (r.fieldnames or []):
            raise KeyError(
                f"score_column={score_column!r} not in {csv_path} (have: {r.fieldnames})"
            )
        for row in r:
            v = row[score_column]
            if v == "" or v is None:
                continue
            try:
                if as_int:
                    vals.append(float(int(v)))
                else:
                    vals.append(float(v))
            except ValueError:
                continue
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if score_mode != "identity":
        arr = apply_score_mode(arr, score_mode, eps=eps)
        arr = arr[np.isfinite(arr)]
    return arr


def integer_bin(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer-valued score: one bin per observed value."""
    c = Counter(arr.astype(np.int64).tolist())
    edges = np.asarray(sorted(c.keys()), dtype=np.int64)
    counts = np.asarray([c[int(l)] for l in edges], dtype=np.float64)
    return edges.astype(np.float64), counts


def log_bin(arr: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Geometrically spaced edges; only valid for strictly positive scores."""
    lo = float(arr.min())
    hi = float(arr.max())
    if lo <= 0:
        raise ValueError(
            "log-bin mode requires strictly positive scores; got min={lo!r}".format(lo=lo)
        )
    edges = np.unique(
        np.geomspace(max(lo, 1e-12), max(hi, lo + 1e-9), num=n_bins)
    )
    counts, _ = np.histogram(arr, bins=np.concatenate([edges, [edges[-1] * 1.0001 + 1e-9]]))
    return edges.astype(np.float64), counts.astype(np.float64)


def linear_bin(arr: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced edges; supports signed scores (e.g., logit toxicity)."""
    lo = float(arr.min())
    hi = float(arr.max())
    edges = np.linspace(lo, hi, num=n_bins)
    counts, _ = np.histogram(arr, bins=np.concatenate([edges, [edges[-1] + (edges[-1] - edges[-2])]]))
    return edges.astype(np.float64), counts.astype(np.float64)


def survival_and_hazard(
    edges: np.ndarray,
    counts: np.ndarray,
    n_total: int,
    *,
    hazard_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical survival at left bin edge; hazard via discrete or bin-derivative."""
    cum_at_or_above = np.flip(np.cumsum(np.flip(counts)))
    surv = cum_at_or_above / float(n_total)

    if hazard_mode == "discrete":
        # h(l) = count_in_bin / count_at_or_above_bin (for integer bins).
        with np.errstate(divide="ignore", invalid="ignore"):
            haz = np.where(cum_at_or_above > 0, counts / cum_at_or_above, np.nan)
    elif hazard_mode == "continuous":
        # Bin-derivative: h(s_b) = log(S(s_b)/S(s_{b+1})) / (s_{b+1} - s_b).
        haz = np.full_like(surv, np.nan)
        for i in range(len(edges) - 1):
            ds = edges[i + 1] - edges[i]
            if ds <= 0 or surv[i] <= 0 or surv[i + 1] <= 0:
                continue
            haz[i] = np.log(surv[i] / surv[i + 1]) / ds
    else:
        raise ValueError(f"Unsupported hazard_mode={hazard_mode!r}")
    return surv, haz


def _resolve_x_scale(arr: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested
    return "log" if float(arr.min()) > 0 else "linear"


def panel_hist(ax, arr: np.ndarray, x_label: str, title: str) -> None:
    ax.hist(arr, bins=200, color="C0")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def panel_survival(
    ax, edges: np.ndarray, surv: np.ndarray, x_label: str, x_scale: str
) -> None:
    if x_scale == "log":
        ax.loglog(edges, surv, ".", markersize=4, color="C2")
    else:
        ax.semilogy(edges, surv, ".", markersize=4, color="C2")
    ax.set_xlabel(f"{x_label}{' (log)' if x_scale == 'log' else ''}")
    ax.set_ylabel("survival P(S >= s) (log)")
    ax.set_title("Survival")
    ax.grid(alpha=0.3, which="both")


def panel_hazard(
    ax, edges: np.ndarray, haz: np.ndarray, x_label: str, x_scale: str
) -> None:
    mask = np.isfinite(haz) & (haz > 0)
    if x_scale == "log":
        ax.semilogx(edges[mask], np.log(haz[mask]), ".", markersize=4, color="C3")
        ax.set_xlabel(f"{x_label} (log)")
    else:
        ax.plot(edges[mask], np.log(haz[mask]), ".", markersize=4, color="C3")
        ax.set_xlabel(x_label)
    ax.set_ylabel("log hazard h(s)")
    ax.set_title("Hazard")
    ax.grid(alpha=0.3, which="both")


def render_row(
    axes_row,
    arr: np.ndarray,
    edges: np.ndarray,
    surv: np.ndarray,
    haz: np.ndarray,
    *,
    x_label: str,
    x_scale: str,
    hist_title: str,
) -> None:
    panel_hist(axes_row[0], arr, x_label, hist_title)
    panel_survival(axes_row[1], edges, surv, x_label, x_scale)
    panel_hazard(axes_row[2], edges, haz, x_label, x_scale)


def synthetic_sample(
    name: str, n: int, rng: np.random.Generator, **params
) -> np.ndarray:
    """Reference distributions for the trifecta-figure ideal scenarios."""
    if name == "exponential":
        return rng.exponential(scale=1.0, size=n).astype(np.float64)
    if name == "pareto":
        alpha = float(params.get("alpha", 3.0))
        u = rng.uniform(0.0, 1.0, size=n)
        return (u ** (-1.0 / alpha)).astype(np.float64)
    if name == "uniform":
        # Uniform(0, 1); upper endpoint at 1 -> reverse Weibull (bounded) MDA.
        return rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    raise ValueError(f"Unknown synthetic name: {name}")


def compute_score_panel(
    csv_path: Path,
    score_column: str,
    *,
    score_mode: str,
    bin_mode: str,
    n_log_bins: int,
    n_linear_bins: int,
    x_scale: str,
    synthetic: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if synthetic is not None:
        if rng is None:
            rng = np.random.default_rng(int(synthetic.get("seed", 11)))
        n = int(synthetic.get("n", 200_000))
        params = {k: v for k, v in synthetic.items() if k not in ("name", "n", "seed")}
        arr = synthetic_sample(synthetic["name"], n, rng, **params)
    else:
        as_int = bin_mode == "integer"
        arr = load_score_column(
            csv_path, score_column, score_mode=score_mode, as_int=as_int
        )
    if arr.size == 0:
        raise RuntimeError(
            f"No usable rows in {csv_path} for column {score_column} (mode={score_mode})"
        )
    if bin_mode == "integer":
        edges, counts = integer_bin(arr)
        hazard_mode = "discrete"
    elif bin_mode == "log":
        edges, counts = log_bin(arr, n_log_bins)
        hazard_mode = "continuous"
    elif bin_mode == "linear":
        edges, counts = linear_bin(arr, n_linear_bins)
        hazard_mode = "continuous"
    else:
        raise ValueError(f"Unsupported bin_mode={bin_mode!r}")
    surv, haz = survival_and_hazard(edges, counts, arr.size, hazard_mode=hazard_mode)
    resolved_x_scale = _resolve_x_scale(arr, x_scale)
    return {
        "arr": arr,
        "edges": edges,
        "surv": surv,
        "haz": haz,
        "x_scale": resolved_x_scale,
    }


def report_stats(arr: np.ndarray, label: str) -> None:
    qs = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]
    qv = np.quantile(arr, qs)
    print(
        f"[{label}] n={arr.size} min={arr.min():.4g} median={np.median(arr):.4g} "
        f"mean={arr.mean():.4g} max={arr.max():.4g}",
        flush=True,
    )
    for q, v in zip(qs, qv):
        print(f"  q={q:>7}  s={v:.4g}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--multi-panel-config",
        type=Path,
        default=None,
        help="Path to JSON list describing multiple score panels (relative csv paths "
        "resolve against $RESULTS_ROOT). If set, "
        "single-panel arguments are ignored except --out and --n-log-bins.",
    )
    # Single-panel back-compat args:
    ap.add_argument("--scores-csv", type=Path)
    ap.add_argument("--score-column", default="asst_char_len")
    ap.add_argument(
        "--score-mode",
        default="identity",
        choices=("identity", "log", "exp", "gumbel_log_prob", "gumbel_prob"),
    )
    ap.add_argument(
        "--bin-mode", choices=("integer", "log", "linear"), default="log"
    )
    ap.add_argument(
        "--x-scale", choices=("auto", "log", "linear"), default="auto"
    )
    ap.add_argument("--n-log-bins", type=int, default=80)
    ap.add_argument("--n-linear-bins", type=int, default=120)
    ap.add_argument("--x-label", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.multi_panel_config is not None:
        cfg = json.loads(args.multi_panel_config.read_text())
        if not isinstance(cfg, list) or not cfg:
            raise ValueError("--multi-panel-config must be a non-empty JSON list.")
        n_rows = len(cfg)
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 2.6 * n_rows))
        if n_rows == 1:
            axes = np.asarray([axes])
        for i, panel in enumerate(cfg):
            synthetic = panel.get("synthetic")
            if synthetic is None:
                csv_path = RESULTS_ROOT / panel["csv"]
                score_column = panel["score_column"]
            else:
                csv_path = Path("/dev/null")
                score_column = synthetic.get("name", "synthetic")
            label = panel.get("label", csv_path.stem)
            x_label = panel.get("x_label", score_column)
            score_mode = panel.get("score_mode", "identity")
            bin_mode = panel.get("bin_mode", "log")
            x_scale = panel.get("x_scale", "auto")
            n_log_bins = int(panel.get("n_log_bins", args.n_log_bins))
            n_linear_bins = int(panel.get("n_linear_bins", args.n_linear_bins))
            print(f"[panel {i + 1}/{n_rows}] {label}", flush=True)
            res = compute_score_panel(
                csv_path,
                score_column,
                score_mode=score_mode,
                bin_mode=bin_mode,
                n_log_bins=n_log_bins,
                n_linear_bins=n_linear_bins,
                x_scale=x_scale,
                synthetic=synthetic,
            )
            report_stats(res["arr"], label)
            render_row(
                axes[i],
                res["arr"],
                res["edges"],
                res["surv"],
                res["haz"],
                x_label=x_label,
                x_scale=res["x_scale"],
                hist_title=f"{label} (n={res['arr'].size})",
            )
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"[done] wrote {args.out}", flush=True)
        return

    if args.scores_csv is None:
        ap.error("--scores-csv is required in single-panel mode.")
    res = compute_score_panel(
        args.scores_csv,
        args.score_column,
        score_mode=args.score_mode,
        bin_mode=args.bin_mode,
        n_log_bins=args.n_log_bins,
        n_linear_bins=args.n_linear_bins,
        x_scale=args.x_scale,
    )
    label = args.title or args.score_column
    report_stats(res["arr"], label)
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))
    x_label = args.x_label or args.score_column
    render_row(
        axes,
        res["arr"],
        res["edges"],
        res["surv"],
        res["haz"],
        x_label=x_label,
        x_scale=res["x_scale"],
        hist_title=f"Distribution of {label} (n={res['arr'].size})",
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
