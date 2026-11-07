"""Collect the numbers of the appendix "Forecastability training with sampled returns".

Reads the ten-seed runs of run_rlquant_bipolar_reinforce_negret_m600_20260725.sh (ours,
at 1, 2, 4, 8, 16 and 32 rollouts per task), the three 32-rollout seeds that
run_rlquant_bipolar_sampled_returns_extras.sh n32_last_stable evaluated at their last stable
checkpoint, and the 600-update runs of the return-maximizing RL baseline from
run_rlquant_bipolar_reinforce_overnight_20260725.sh (at 8 and 16 rollouts), together with the
post-hoc calibration files for the negative-return score that
run_rlquant_posthoc_calibration_sweep.sbatch writes into them, and writes
  - RESULTS_ROOT/summaries/rl_capability_verified.json, the per-seed mean returns on the
    pretraining tasks that make_appendix_figures.py plots, and
  - RESULTS_ROOT/summaries/sampled_returns.md, the improvement factors, mean returns,
    worst-case regret factors and calibrated errors that the appendix reports.

Forecast errors are squared errors of the worst held-out negative return, averaged over each
seed's held-out task sets. The improvement factor is the pretrained policy's error divided by
the trained policy's, per seed; we report its mean and standard error across seeds. The
worst-case regret factor is the pretrained policy's worst held-out regret divided by the
trained policy's, per seed, averaged over seeds.

Usage (from the repository root; reads only JSON):
    python scripts/gridworld/collect_sampled_returns.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 17, 23, 29, 31, 37, 41, 43, 47, 53)
OURS = "rlquant_trap_report_bipolar_sev01_reinforce_plugin_negret_m600_n{n}_lr1e4_prex_10seeds_{d}"
RL = "rlquant_trap_report_bipolar_sev01_baseline_sft_union_reinforce_m600_n{n}_10seeds_{d}"
# The 32-rollout seeds whose policy went non-finite late in training, each evaluated at its last stable
# checkpoint (run_rlquant_bipolar_sampled_returns_extras.sh n32_last_stable). The paper's 32-rollout result
# pools them with the other seeds of the 600-update run.
LAST_STABLE = "rlquant_trap_report_bipolar_sev01_reinforce_plugin_negret_prenan_m*_n32_lr1e4_prex_*_20260726"


def sem(xs: list[float]) -> float:
    return statistics.stdev(xs) / math.sqrt(len(xs))


def sq_error(path: Path, key: str = "heldout_summary_no_calibration") -> float:
    return json.loads(path.read_text())["heldout_calibrated"][key]["mean_sq_error"]


def train_task_return(path: Path, policy: str) -> float:
    return json.loads(path.read_text())[policy]["split_metrics"]["pretrain_train"]["mean_return"]


def worst_regret_factor(seed_dir: Path) -> float:
    d = json.loads((seed_dir / "final_results.json").read_text())
    worst = {p: float(d[p]["forecast_pairs"]["heldout"]["actual_worst_regret"]) for p in ("pretrained", "meta_learned")}
    return worst["pretrained"] / worst["meta_learned"]


def n32_seed_dirs(R: Path, d: str) -> tuple[dict[int, Path], dict[int, int]]:
    """Each seed's 32-rollout directory: its last-stable-checkpoint run if it has one, else the 600-update run."""
    dirs = {s: R / OURS.format(n=32, d=d) / f"seed_{s}" for s in SEEDS}
    stopped = {}
    for run in sorted(R.glob(LAST_STABLE)):
        for s in SEEDS:
            if (run / f"seed_{s}" / "final_results.json").exists():
                dirs[s] = run / f"seed_{s}"
                stopped[s] = int(run.name.split("_prenan_m")[1].split("_")[0])
    return dirs, stopped


def main() -> None:
    results_default = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=results_default)
    ap.add_argument("--tag-date", default="20260725", help="date in the runs' names")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: RESULTS_ROOT/summaries)")
    args = ap.parse_args()
    R, d = args.results_root, args.tag_date
    out_dir = args.out_dir or R / "summaries"

    capability, fold_cells, return_cells, cal_lines, regret_cells = [], [], [], [], []
    pretrained_err = None
    for n in (1, 2, 4, 8, 16):
        run = OURS.format(n=n, d=d)
        pre = [sq_error(R / run / f"seed_{s}" / "posthoc_calibration_pretrained_negret.json") for s in SEEDS]
        ours = [sq_error(R / run / f"seed_{s}" / "posthoc_calibration_meta_learned_negret.json") for s in SEEDS]
        ours_cal = [sq_error(R / run / f"seed_{s}" / "posthoc_calibration_meta_learned_negret.json",
                             "heldout_summary_shift_scale_2param") for s in SEEDS]
        if pretrained_err is None:
            pre_cal = [sq_error(R / run / f"seed_{s}" / "posthoc_calibration_pretrained_negret.json",
                                "heldout_summary_shift_scale_2param") for s in SEEDS]
            pretrained_err = (statistics.mean(pre), statistics.mean(pre_cal))
        folds = [a / b for a, b in zip(pre, ours)]
        trained_ret = [train_task_return(R / run / f"seed_{s}" / "final_results.json", "meta_learned") for s in SEEDS]
        pre_ret = [train_task_return(R / run / f"seed_{s}" / "final_results.json", "pretrained") for s in SEEDS]
        capability.append({
            "rollouts": n,
            "source": run,
            "seeds": list(SEEDS),
            "trained_returns": trained_ret,
            "pretrained_returns": pre_ret,
            "trained_mean": statistics.mean(trained_ret),
            "trained_sem": sem(trained_ret),
            "pretrained_mean": statistics.mean(pre_ret),
            "pretrained_sem": sem(pre_ret),
        })
        fold_cells.append(f"{statistics.mean(folds):.1f} +/- {sem(folds):.1f}")
        regret_cells.append(f"{statistics.mean(worst_regret_factor(R / run / f'seed_{s}') for s in SEEDS):.2f}")
        return_cells.append(f"{statistics.mean(trained_ret):.2f}")
        cal_lines.append(f"| Ours, {n} rollouts | {statistics.mean(ours):.2f} | {statistics.mean(ours_cal):.2f} |")

    # 32 rollouts per task.
    dirs, stopped = n32_seed_dirs(R, d)
    pre = [sq_error(dirs[s] / "posthoc_calibration_pretrained_negret.json") for s in SEEDS]
    ours = [sq_error(dirs[s] / "posthoc_calibration_meta_learned_negret.json") for s in SEEDS]
    folds32 = [a / b for a, b in zip(pre, ours)]
    returns32 = [train_task_return(dirs[s] / "final_results.json", "meta_learned") for s in SEEDS]
    regret_cells.append(f"{statistics.mean(worst_regret_factor(dirs[s]) for s in SEEDS):.2f}")
    stopped_text = ", ".join(f"seed {s} at update {u}" for s, u in sorted(stopped.items())) or "none"

    # The RL baseline. Its run directories store the RL-trained policy in both result slots,
    # so the improvement factor divides the shared pretrained policy's error (from our runs) by the RL error.
    rl_lines = []
    for n in (8, 16):
        pre = [sq_error(R / OURS.format(n=n, d=d) / f"seed_{s}" / "posthoc_calibration_pretrained_negret.json")
               for s in SEEDS]
        rl = [sq_error(R / RL.format(n=n, d=d) / f"seed_{s}" / "posthoc_calibration_meta_learned_negret.json")
              for s in SEEDS]
        rl_cal = [sq_error(R / RL.format(n=n, d=d) / f"seed_{s}" / "posthoc_calibration_meta_learned_negret.json",
                           "heldout_summary_shift_scale_2param") for s in SEEDS]
        folds = [a / b for a, b in zip(pre, rl)]
        rl_lines.append(f"| RL, {n} rollouts | {statistics.mean(folds):.1f} +/- {sem(folds):.1f} |")
        cal_lines.append(f"| RL, {n} rollouts | {statistics.mean(rl):.2f} | {statistics.mean(rl_cal):.2f} |")

    lines = [
        "# Forecastability training with sampled returns",
        "",
        "Written by scripts/gridworld/collect_sampled_returns.py. Ten seeds per setting, 600 updates.",
        "",
        "## Table: training on negative sampled returns",
        "",
        "| rollouts per task | 1 | 2 | 4 | 8 | 16 |",
        "|---|---|---|---|---|---|",
        "| improvement factor (mean +/- SEM) | " + " | ".join(fold_cells) + " |",
        "| mean return on the pretraining tasks | " + " | ".join(return_cells) + " |",
        "",
        f"The pretrained policy's mean return on the pretraining tasks is {capability[0]['pretrained_mean']:.2f}.",
        "",
        "## 32 rollouts per task",
        "",
        "Ten seeds. The seeds whose policy went non-finite late in training are read at their last stable checkpoint",
        f"({stopped_text}), the others at update 600.",
        f"Improvement factor {statistics.mean(folds32):.1f} +/- {sem(folds32):.1f}; "
        f"mean return on the pretraining tasks {statistics.mean(returns32):.2f}.",
        "",
        "## Worst-case held-out regret",
        "",
        "The pretrained policy's worst held-out regret divided by the trained policy's, per seed, averaged over seeds.",
        "",
        "| rollouts per task | 1 | 2 | 4 | 8 | 16 | 32 |",
        "|---|---|---|---|---|---|---|",
        "| worst-case regret factor | " + " | ".join(regret_cells) + " |",
        "",
        "## The return-maximizing RL baseline",
        "",
        "| run | improvement factor (mean +/- SEM) |",
        "|---|---|",
        *rl_lines,
        "",
        "## Calibration on the negative-return score",
        "",
        "| policy | mean error | after calibration |",
        "|---|---|---|",
        f"| pretrained | {pretrained_err[0]:.2f} | {pretrained_err[1]:.2f} |",
        *cal_lines,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rl_capability_verified.json").write_text(json.dumps(capability, indent=1) + "\n")
    (out_dir / "sampled_returns.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir / 'rl_capability_verified.json'} and {out_dir / 'sampled_returns.md'}")


if __name__ == "__main__":
    main()
