"""Collect the table of the appendix section "The improving-only mask and hidden rare modes".

Rows:
  earlier recipe, no mask:  run_lquant_40slots_modbulkgcg_metatrain_multiseed.sbatch (5 seeds)
  earlier recipe, mask:     run_lquant_40slots_modbulkgcg_metatrain_iog_multiseed.sbatch (5 seeds)
  Section 6.1 recipe, mask: run_lquant_modbulkgcg_iog_full_bootstrap_cached_multiseed.sbatch (10 seeds)

Each cell is a mean over seeds of the seed's mean over its seven held-out passwords: the
squared log10 forecast error at the worst-rank prompt, and the log-probability of the password
at that prompt. Pretrained values come from baseline_test_results.json and trained values from
test_results.json.

Usage (from the repository root; reads only JSON):
    python scripts/audits/collect_no_mask_table.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EARLY = [(11, "20260502"), (17, "20260504"), (23, "20260504"), (29, "20260504"), (31, "20260504")]
ROWS = [
    ("earlier recipe", "off", "lquant_40slots_modbulkgcg_metatrain_seed{s}_{d}", EARLY),
    ("earlier recipe", "on", "lquant_40slots_modbulkgcg_metatrain_iog_seed{s}_{d}", EARLY),
    ("Section 6.1 recipe", "on", "lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_seed{s}_{d}",
     [(s, "20260505") for s in (11, 17, 23, 29, 31, 37, 41, 43, 47, 53)]),
]


def seed_means(path: Path) -> tuple[float, float]:
    rows = [r for r in json.loads(path.read_text()).values()
            if isinstance(r, dict) and "forecast_sq_log10_error" in r]
    return (statistics.mean(r["forecast_sq_log10_error"] for r in rows),
            statistics.mean(r["actual_worst_log_prob"] for r in rows))


def main() -> None:
    results_default = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=results_default)
    args = ap.parse_args()
    print("| recipe | mask | seeds | pretrained error | trained error | pretrained log-prob | trained log-prob |")
    print("|---|---|---|---|---|---|---|")
    for recipe, mask, pattern, seeds in ROWS:
        pre, post = [], []
        for s, d in seeds:
            run = args.results_root / pattern.format(s=s, d=d) / f"seed_{s}"
            pre.append(seed_means(run / "baseline_test_results.json"))
            post.append(seed_means(run / "test_results.json"))
        mean = lambda xs, i: statistics.mean(x[i] for x in xs)  # noqa: E731
        print(f"| {recipe} | {mask} | {len(seeds)} | {mean(pre, 0):.3f} | {mean(post, 0):.3f} | "
              f"{mean(pre, 1):.2f} | {mean(post, 1):.2f} |")


if __name__ == "__main__":
    main()
