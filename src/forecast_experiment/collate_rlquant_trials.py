from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np

from .ols_utils import fit_ols_tail_from_scores


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_last_elapsed(entries: list[dict]) -> float:
    if not entries:
        return 0.0
    return float(entries[-1].get("elapsed_sec", 0.0))


def _summary(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": float(len(finite)),
        "mean": float(statistics.fmean(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _nested_float(payload: dict, *path: str, default: float = float("nan")) -> float:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return float(default)
        cur = cur[key]
    return float(cur)


def _load_return_forecast_metrics(run_dir: Path, config: dict) -> dict[str, float]:
    metadata_path = run_dir / "forecast_traces_manifests.json"
    archive_path = run_dir / "forecast_traces.npz"
    if not metadata_path.exists() or not archive_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    models = metadata.get("models", {})
    plotting = str(config.get("plotting_position", "weibull"))
    ols_top_k = int(config.get("ols_top_k", 10))
    out: dict[str, float] = {}
    with np.load(archive_path) as archive:
        for model_label in ("pretrained", "meta_learned"):
            split_rows = models.get(model_label, {})
            for split_name in ("meta", "heldout"):
                rows = split_rows.get(split_name, [])
                sq_errors: list[float] = []
                abs_errors: list[float] = []
                for row in rows:
                    prefix = str(row["array_prefix"])
                    fit_returns = archive[f"{prefix}__fit_returns"]
                    deploy_returns = archive[f"{prefix}__deploy_returns"]
                    if fit_returns.size == 0 or deploy_returns.size == 0:
                        continue
                    pred_neg_return, _a, _b = fit_ols_tail_from_scores(
                        (-fit_returns).tolist(),
                        n_top=ols_top_k,
                        plotting=plotting,
                        deploy_queries=int(deploy_returns.size),
                        score_mode="identity",
                    )
                    predicted_worst_return = -float(pred_neg_return)
                    actual_worst_return = float(np.min(deploy_returns))
                    error = predicted_worst_return - actual_worst_return
                    abs_errors.append(abs(error))
                    sq_errors.append(error * error)
                abs_key = f"{model_label}_{split_name}_worst_return_forecast_abs_error"
                sq_key = f"{model_label}_{split_name}_worst_return_forecast_sq_error"
                out[abs_key] = float(statistics.fmean(abs_errors)) if abs_errors else float("nan")
                out[sq_key] = float(statistics.fmean(sq_errors)) if sq_errors else float("nan")
    if math.isfinite(out.get("pretrained_meta_worst_return_forecast_abs_error", float("nan"))) and math.isfinite(
        out.get("meta_learned_meta_worst_return_forecast_abs_error", float("nan"))
    ):
        out["delta_meta_worst_return_forecast_abs_error"] = float(
            out["meta_learned_meta_worst_return_forecast_abs_error"]
            - out["pretrained_meta_worst_return_forecast_abs_error"]
        )
    if math.isfinite(out.get("pretrained_meta_worst_return_forecast_sq_error", float("nan"))) and math.isfinite(
        out.get("meta_learned_meta_worst_return_forecast_sq_error", float("nan"))
    ):
        out["delta_meta_worst_return_forecast_sq_error"] = float(
            out["meta_learned_meta_worst_return_forecast_sq_error"]
            - out["pretrained_meta_worst_return_forecast_sq_error"]
        )
    if math.isfinite(out.get("pretrained_heldout_worst_return_forecast_abs_error", float("nan"))) and math.isfinite(
        out.get("meta_learned_heldout_worst_return_forecast_abs_error", float("nan"))
    ):
        out["delta_heldout_worst_return_forecast_abs_error"] = float(
            out["meta_learned_heldout_worst_return_forecast_abs_error"]
            - out["pretrained_heldout_worst_return_forecast_abs_error"]
        )
    if math.isfinite(out.get("pretrained_heldout_worst_return_forecast_sq_error", float("nan"))) and math.isfinite(
        out.get("meta_learned_heldout_worst_return_forecast_sq_error", float("nan"))
    ):
        out["delta_heldout_worst_return_forecast_sq_error"] = float(
            out["meta_learned_heldout_worst_return_forecast_sq_error"]
            - out["pretrained_heldout_worst_return_forecast_sq_error"]
        )
    return out


def _collect_trial(run_dir: Path) -> dict | None:
    final_path = run_dir / "final_results.json"
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"
    if not final_path.exists() or not history_path.exists() or not config_path.exists():
        return None
    final = _load_json(final_path)
    history = _load_json(history_path)
    config = _load_json(config_path)

    pretrained = final["pretrained"]
    learned = final["meta_learned"]
    pretrained_meta_abs = _nested_float(pretrained, "forecast_pairs", "meta", "forecast_abs_error")
    learned_meta_abs = _nested_float(learned, "forecast_pairs", "meta", "forecast_abs_error")
    pretrained_heldout_abs = _nested_float(pretrained, "forecast_pairs", "heldout", "forecast_abs_error")
    learned_heldout_abs = _nested_float(learned, "forecast_pairs", "heldout", "forecast_abs_error")
    metrics = {
        "seed": int(config["seed"]),
        "pretrained_meta_forecast_abs_error": pretrained_meta_abs,
        "meta_learned_meta_forecast_abs_error": learned_meta_abs,
        "delta_meta_forecast_abs_error": float(
            learned_meta_abs - pretrained_meta_abs
        ),
        "pretrained_meta_forecast_sq_error": _nested_float(
            pretrained, "forecast_pairs", "meta", "forecast_sq_error", default=pretrained_meta_abs**2
        ),
        "meta_learned_meta_forecast_sq_error": _nested_float(
            learned, "forecast_pairs", "meta", "forecast_sq_error", default=learned_meta_abs**2
        ),
        "delta_meta_forecast_sq_error": float(
            _nested_float(learned, "forecast_pairs", "meta", "forecast_sq_error", default=learned_meta_abs**2)
            - _nested_float(pretrained, "forecast_pairs", "meta", "forecast_sq_error", default=pretrained_meta_abs**2)
        ),
        "pretrained_heldout_forecast_abs_error": pretrained_heldout_abs,
        "meta_learned_heldout_forecast_abs_error": learned_heldout_abs,
        "delta_heldout_forecast_abs_error": float(
            learned_heldout_abs - pretrained_heldout_abs
        ),
        "pretrained_heldout_forecast_sq_error": _nested_float(
            pretrained, "forecast_pairs", "heldout", "forecast_sq_error", default=pretrained_heldout_abs**2
        ),
        "meta_learned_heldout_forecast_sq_error": _nested_float(
            learned, "forecast_pairs", "heldout", "forecast_sq_error", default=learned_heldout_abs**2
        ),
        "delta_heldout_forecast_sq_error": float(
            _nested_float(learned, "forecast_pairs", "heldout", "forecast_sq_error", default=learned_heldout_abs**2)
            - _nested_float(
                pretrained, "forecast_pairs", "heldout", "forecast_sq_error", default=pretrained_heldout_abs**2
            )
        ),
        "pretrained_meta_extrapolated_weighted_sq_error": _nested_float(
            pretrained, "forecast_pairs", "meta", "extrapolated_weighted_mean_sq_error"
        ),
        "meta_learned_meta_extrapolated_weighted_sq_error": _nested_float(
            learned, "forecast_pairs", "meta", "extrapolated_weighted_mean_sq_error"
        ),
        "delta_meta_extrapolated_weighted_sq_error": float(
            _nested_float(learned, "forecast_pairs", "meta", "extrapolated_weighted_mean_sq_error")
            - _nested_float(pretrained, "forecast_pairs", "meta", "extrapolated_weighted_mean_sq_error")
        ),
        "pretrained_heldout_extrapolated_weighted_sq_error": _nested_float(
            pretrained, "forecast_pairs", "heldout", "extrapolated_weighted_mean_sq_error"
        ),
        "meta_learned_heldout_extrapolated_weighted_sq_error": _nested_float(
            learned, "forecast_pairs", "heldout", "extrapolated_weighted_mean_sq_error"
        ),
        "delta_heldout_extrapolated_weighted_sq_error": float(
            _nested_float(learned, "forecast_pairs", "heldout", "extrapolated_weighted_mean_sq_error")
            - _nested_float(pretrained, "forecast_pairs", "heldout", "extrapolated_weighted_mean_sq_error")
        ),
        "pretrained_heldout_deploy_mean_regret": float(pretrained["split_metrics"]["heldout_deploy"]["mean_regret"]),
        "meta_learned_heldout_deploy_mean_regret": float(learned["split_metrics"]["heldout_deploy"]["mean_regret"]),
        "delta_heldout_deploy_mean_regret": float(
            learned["split_metrics"]["heldout_deploy"]["mean_regret"]
            - pretrained["split_metrics"]["heldout_deploy"]["mean_regret"]
        ),
        "pretrained_heldout_deploy_worst_regret": float(pretrained["split_metrics"]["heldout_deploy"]["worst_regret"]),
        "meta_learned_heldout_deploy_worst_regret": float(learned["split_metrics"]["heldout_deploy"]["worst_regret"]),
        "delta_heldout_deploy_worst_regret": float(
            learned["split_metrics"]["heldout_deploy"]["worst_regret"]
            - pretrained["split_metrics"]["heldout_deploy"]["worst_regret"]
        ),
        "pretrained_pretrain_mean_return": float(pretrained["split_metrics"]["pretrain_train"]["mean_return"]),
        "meta_learned_pretrain_mean_return": float(learned["split_metrics"]["pretrain_train"]["mean_return"]),
        "delta_pretrain_mean_return": float(
            learned["split_metrics"]["pretrain_train"]["mean_return"]
            - pretrained["split_metrics"]["pretrain_train"]["mean_return"]
        ),
        "pretrain_elapsed_sec": _safe_last_elapsed(list(history.get("pretrain", []))),
        "meta_elapsed_sec": _safe_last_elapsed(list(history.get("meta", []))),
        "total_elapsed_sec": _safe_last_elapsed(list(history.get("pretrain", [])))
        + _safe_last_elapsed(list(history.get("meta", []))),
    }
    metrics.update(_load_return_forecast_metrics(run_dir, config))
    return metrics


def collate_trials(run_root: Path, *, out_json: Path, out_csv: Path) -> dict:
    run_dirs = sorted(
        [path for path in run_root.glob("seed_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    rows = [row for row in (_collect_trial(path) for path in run_dirs) if row is not None]
    metric_names = [key for key in rows[0].keys() if key != "seed"] if rows else []
    summary = {
        "run_root": str(run_root),
        "n_trials": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "metrics": {name: _summary([float(row[name]) for row in rows]) for name in metric_names},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["seed"] + metric_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collate RLQuant trial directories into summary JSON/CSV")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    out_json = Path(args.out_json) if args.out_json else run_root / "summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else run_root / "trial_metrics.csv"
    summary = collate_trials(run_root, out_json=out_json, out_csv=out_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
