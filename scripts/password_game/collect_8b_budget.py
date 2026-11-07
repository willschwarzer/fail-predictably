"""Collect the Qwen3-8B training-budget table of "Password-game scaling and the role of the
evaluation bank" (seeds 11, 17 and 23, final checkpoints, original password bank).

The rows and the launchers that wrote their runs:
  200 steps, lr 1e-5, all 33 targets: the first 8B run,
      run_lquant_modbulkgcg_qwen3_8b_iog_full_1gpu.sbatch, continued from its saved
      checkpoints by run_lquant_modbulkgcg_qwen3_8b_iog_full_resume_3gpu.sbatch (seeds 11
      and 17, from steps 44 and 40) and run_lquant_modbulkgcg_qwen3_8b_iog_full_resume.sbatch
      (seed 23, from step 65);
  600 steps, lr 1e-5, all 33: run_lquant_modbulkgcg_qwen3_8b_iog_full_3gpu_orig600.sbatch;
  200 steps, lr 3e-5, all 33: run_lquant_modbulkgcg_qwen3_8b_iog_full_3gpu_fullbatch.sbatch;
  200 steps, lr 3e-5, 8 targets: run_lquant_modbulkgcg_qwen3_8b_iog_full_4gpu_tps8.sbatch.

Writes RESULTS_ROOT/summaries/8b_budget.md. A fold is the pretrained model's mean squared
log10 forecast error over the seven held-out passwords divided by the trained model's; the
table gives each seed's fold and their mean and standard error.

Usage (from the repository root; reads only JSON):
    python scripts/password_game/collect_8b_budget.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
P = "lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_qwen3_8b"
ROWS = [  # (steps, learning rate, targets per update, run directory per seed 11, 17, 23)
    ("200", "1e-5", "all 33", [f"{P}_3gpu_steps200_seed11_superpod_3gpu_20260506",
                               f"{P}_3gpu_steps200_seed17_superpod_3gpu_20260506",
                               f"{P}_1gpu_steps200_seed23_superpod_1gpu_20260506"]),
    ("600", "1e-5", "all 33", [f"{P}_3gpu_origrecipe_steps600_seed{s}_20260728" for s in (11, 17, 23)]),
    ("200", "3e-5", "all 33", [f"{P}_3gpu_fullbatch_lr3e5_seed11_20260727",
                               f"{P}_3gpu_fullbatch_lr3e5_seed17_20260727",
                               f"{P}_3gpu_fullbatch_lr3e5_seed23_20260728"]),
    ("200", "3e-5", "8", [f"{P}_4gpu_tps8_lr3e5_seed{s}_20260727" for s in (11, 17, 23)]),
]
SEEDS = (11, 17, 23)


def mean_error(path: Path) -> float:
    data = json.loads(path.read_text())
    errors = [r["forecast_sq_log10_error"] for r in data.values()
              if isinstance(r, dict) and "forecast_sq_log10_error" in r]
    return statistics.mean(errors)


def main() -> None:
    results_default = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=results_default)
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: RESULTS_ROOT/summaries/8b_budget.md)")
    args = ap.parse_args()
    R = args.results_root
    out = args.out or R / "summaries" / "8b_budget.md"

    lines = [
        "# Training budget for Qwen3-8B on the original password bank",
        "",
        "Written by scripts/password_game/collect_8b_budget.py. Final checkpoints, seeds 11, 17 and 23.",
        "",
        "| steps | learning rate | targets per update | per-seed folds | fold (mean +/- SEM) "
        "| trained error | pretrained errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for steps, lr, targets, runs in ROWS:
        base, trained = [], []
        for run, s in zip(runs, SEEDS):
            d = R / run / f"seed_{s}"
            base.append(mean_error(d / "baseline_test_results.json"))
            trained.append(mean_error(d / "test_results.json"))
        folds = [b / t for b, t in zip(base, trained)]
        sem = statistics.stdev(folds) / math.sqrt(len(folds))
        lines.append(
            f"| {steps} | {lr} | {targets} | {', '.join(f'{f:.2f}' for f in folds)} | "
            f"{statistics.mean(folds):.2f} +/- {sem:.2f} | {statistics.mean(trained):.3f} | "
            f"{', '.join(f'{b:.3f}' for b in base)} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
