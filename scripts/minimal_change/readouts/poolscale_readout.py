"""Multi-split held-out readout for the poolscale / cert-distill arms.

For every run dir given, rebuilds each held-out pool (fit + deploy traces, n =
2300) at baseline and at the end of training, and scores both on the SAME 200
random fit/deploy splits (M = fit count, N = n - M, k = 10, Weibull, the
trainer's inverse-OLS tail forecast): abs / signed / squared log10 error of the
forecast worst prompt, the trivial "predict p = 1" error, and the top-prompt
depth. Per run it reports pool means, the across-pool SEM of the paired
(trained - base) abs delta, and the MSE fold (base mse / trained mse, pooled
over pools and splits, the paper's fold convention). The same splits are also
read with k = 20 and k = 50, and the row starred with the run's own training k
(`ols_top_k`) is its result. A read at any other k covers ranks the objective
never targeted, so it is a curvature diagnostic,
not a result. Runs that share a seed
share held-out pools under HELDOUT_FROM_END, so the same-seed rows are directly
comparable.

  python poolscale_readout.py RUN_DIR [RUN_DIR ...] [--json OUT]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

LN10 = math.log(10.0)
S = 200
EXTRA_KS = (20, 50)


def load(run_dir: Path, prefix: str):
    meta = json.loads((run_dir / f"{prefix}test_forecast_traces_metadata.json").read_text())
    z = np.load(run_dir / f"{prefix}test_forecast_traces.npz")
    out = {}
    for t in meta["targets"]:
        pre = t["array_prefix"]
        out[t["target"]] = (
            np.concatenate([z[f"{pre}__fit_log_probs"], z[f"{pre}__deploy_log_probs"]]).astype(np.float64),
            int(t["fit_count"]),
        )
    return out


def split_errors(v: np.ndarray, perms: np.ndarray, M: int, k: int = 10):
    """Signed log10 error of the inverse-OLS tail forecast on each split."""
    n = v.size
    fit = v[perms[:, :M]]
    dep = v[perms[:, M:]]
    top = -np.sort(-fit, axis=1)[:, :k]
    x = -np.log(np.maximum(-top, 1e-30))
    y = np.log(np.arange(1, k + 1) / (M + 1.0))
    xm = x.mean(axis=1, keepdims=True)
    sxx = ((x - xm) ** 2).sum(axis=1)
    sxy = ((x - xm) * (y - y.mean())).sum(axis=1)
    ok = sxx > 1e-20
    b = np.where(ok, sxy / np.where(ok, sxx, 1.0), 0.0)
    a = y.mean() - b * xm[:, 0]
    nz = b != 0.0
    xstar = np.where(nz, (math.log(1.0 / (n - M)) - a) / np.where(nz, b, 1.0), x[:, 0])
    with np.errstate(over="ignore"):
        pred = -np.exp(-xstar)
    actual = dep.max(axis=1)
    return (pred - actual) / LN10, -actual / LN10


def readout(run_dir: Path):
    base = load(run_dir, "baseline_")
    final = load(run_dir, "")
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = []
    for target, (vb, M) in base.items():
        vt, Mt = final[target]
        assert Mt == M and vt.size == vb.size
        # One split set per pool, seeded by the pool name so every arm that
        # holds this pool out is scored on identical splits.
        rng = np.random.default_rng(int.from_bytes(target.encode(), "little") % (2**32))
        perms = np.stack([rng.permutation(vb.size) for _ in range(S)])
        eb, ob = split_errors(vb, perms, M)
        et, ot = split_errors(vt, perms, M)
        extra = {}
        for k in EXTRA_KS:
            ebk, _ = split_errors(vb, perms, M, k=k)
            etk, _ = split_errors(vt, perms, M, k=k)
            extra.update({
                f"base_abs_k{k}": float(np.abs(ebk).mean()), f"trained_abs_k{k}": float(np.abs(etk).mean()),
                f"base_mse_k{k}": float((ebk ** 2).mean()), f"trained_mse_k{k}": float((etk ** 2).mean()),
            })
        rows.append({
            "target": target,
            "base_abs": float(np.abs(eb).mean()), "trained_abs": float(np.abs(et).mean()),
            "base_signed": float(eb.mean()), "trained_signed": float(et.mean()),
            "base_mse": float((eb ** 2).mean()), "trained_mse": float((et ** 2).mean()),
            "base_one": float(np.abs(ob).mean()), "trained_one": float(np.abs(ot).mean()),
            "base_top_nats": float(vb.max()), "trained_top_nats": float(vt.max()),
            "base_median_nats": float(np.median(vb)), "trained_median_nats": float(np.median(vt)),
            **extra,
        })
    d = np.array([r["trained_abs"] - r["base_abs"] for r in rows])
    summary = {
        "run": run_dir.name, "seed": cfg.get("seed"), "n_pools": len(rows),
        "train_targets": cfg.get("train_target_count"), "targets_per_step": cfg.get("targets_per_step"),
        "cert_distill_coef": cfg.get("cert_distill_coef", 0.0),
        "forecast_loss_coef": cfg.get("forecast_loss_coef", 1.0),
        "base_abs": float(np.mean([r["base_abs"] for r in rows])),
        "trained_abs": float(np.mean([r["trained_abs"] for r in rows])),
        "delta_abs": float(d.mean()), "delta_abs_sem": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan"),
        "mse_fold": float(np.mean([r["base_mse"] for r in rows]) / np.mean([r["trained_mse"] for r in rows])),
        "ols_top_k": cfg.get("ols_top_k", 10),
        "base_signed": float(np.mean([r["base_signed"] for r in rows])),
        "trained_signed": float(np.mean([r["trained_signed"] for r in rows])),
        "trained_one": float(np.mean([r["trained_one"] for r in rows])),
        "top_drop_nats": float(np.mean([r["base_top_nats"] - r["trained_top_nats"] for r in rows])),
        "median_drop_nats": float(np.mean([r["base_median_nats"] - r["trained_median_nats"] for r in rows])),
        "pools": rows,
    }
    for k in EXTRA_KS:
        dk = np.array([r[f"trained_abs_k{k}"] - r[f"base_abs_k{k}"] for r in rows])
        summary[f"delta_abs_k{k}"] = float(dk.mean())
        summary[f"delta_abs_sem_k{k}"] = float(dk.std(ddof=1) / math.sqrt(len(dk))) if len(dk) > 1 else float("nan")
        summary[f"base_abs_k{k}"] = float(np.mean([r[f"base_abs_k{k}"] for r in rows]))
        summary[f"trained_abs_k{k}"] = float(np.mean([r[f"trained_abs_k{k}"] for r in rows]))
        summary[f"mse_fold_k{k}"] = float(
            np.mean([r[f"base_mse_k{k}"] for r in rows]) / np.mean([r[f"trained_mse_k{k}"] for r in rows])
        )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    out = []
    for r in args.runs:
        if not (r / "test_forecast_traces.npz").exists():
            print(f"{r.name}: no final test traces yet")
            continue
        s = readout(r)
        out.append(s)
        print(
            f"{s['run']:28s} seed {s['seed']} pools {s['n_pools']} | k10{'*' if s['ols_top_k'] == 10 else ''} abs {s['base_abs']:.3f} -> "
            f"{s['trained_abs']:.3f} (delta {s['delta_abs']:+.3f} +/- {s['delta_abs_sem']:.3f}) | "
            f"mse fold {s['mse_fold']:.2f}x | signed {s['base_signed']:+.2f} -> {s['trained_signed']:+.2f} | "
            f"drop top {s['top_drop_nats']:.2f} median {s['median_drop_nats']:.2f} nats | "
            f"trained predict-1 {s['trained_one']:.2f}"
            + "".join(
                f" | k{k}{'*' if k == s['ols_top_k'] else ''} abs {s[f'base_abs_k{k}']:.3f} -> "
                f"{s[f'trained_abs_k{k}']:.3f} (delta {s[f'delta_abs_k{k}']:+.3f} +/- {s[f'delta_abs_sem_k{k}']:.3f}), "
                f"mse fold {s[f'mse_fold_k{k}']:.2f}x"
                for k in EXTRA_KS
            )
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
