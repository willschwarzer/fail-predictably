"""Collect the numbers of the appendix "Fit/deployment distribution shift".

Reads the run directories written by
  - submit_lquant_may_v1_distshift_0p6b_fleet.sh (original bank: training,
    best-checkpoint evaluation and both calibrations),
  - submit_lquant_distshift_0p6b_fleet.sh, run_lquant_sft_nopwseed_distshift_0p6b.sbatch
    and submit_distshift_cal_sweep.sh (hardened bank),
  - the main 0.6B password-game runs (Section 6.1) and their calibration, for the
    comparison without a shift,
  - optionally, the leak scan of run_score_password_bank.sbatch,
and writes them to RESULTS_ROOT/summaries/distshift_results.md. The per-seed table
of that file is what make_appendix_figures.py reads for the per-seed figure.

Every error is the squared log10 forecast error of the worst leak probability,
averaged over a seed's seven held-out passwords. Folds are per-seed ratios of the
pretrained error to the trained error, summarized by their mean and its standard
error across seeds, as in the paper.

Usage (from the repository root; reads only JSON and, for the scan, one .npz):
    python scripts/password_game/collect_distshift_results.py
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
CAL_KEY = "heldout_summary_shift_scale_2param"  # the shift-and-scale calibration (Cal.)

SHIFT_RUN = "lquant_40slots_mayv1_distshift891_iog_full_seed{s}_{d}/seed_{s}"
SHIFT_BEST = "lquant_40slots_mayv1_distshift891_besteval_seed{s}_{d}/seed_{s}"
IID_RUN = "lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_seed{s}_{d}/seed_{s}"
HARD_OURS = "lquant_40slots_nopwseed_distshift891_strictv2_iog_full_seed{s}_{d}/seed_{s}"
HARD_SFT = "lquant_40slots_nopwseed_distshift891_sft_union_qwen3_0_6b_seed{s}_{d}/seed_{s}"


def per_target(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {t: r for t, r in data.items() if isinstance(r, dict) and "forecast_sq_log10_error" in r}


def mean_error(path: Path) -> float:
    return statistics.mean(r["forecast_sq_log10_error"] for r in per_target(path).values())


def calibrated_error(path: Path) -> float | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())["heldout_calibrated"][CAL_KEY]["mean_sq_log10_error"]


def mean_sem(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def fmt(x: float | None, spec: str) -> str:
    return "--" if x is None else format(x, spec)


def fold_summary(base: list[float], trained: list[float]) -> tuple[float, float, float]:
    folds = [b / t for b, t in zip(base, trained)]
    m, se = mean_sem(folds)
    return m, se, statistics.median(folds)


def leak_scan(scan_dir: Path) -> list[str]:
    import numpy as np

    manifest = json.loads((scan_dir / "manifest.json").read_text())
    scores = np.load(scan_dir / "password_leak_scores.npz")
    lines = [
        "| family | max leak | 99th percentile | median | prompts above 0.1 | passwords with a prompt above 0.5 |",
        "|---|---|---|---|---|---|",
    ]
    for fi, fam in enumerate(manifest["families"]):
        per_slot = []
        for key in manifest["slots"]:
            lp = scores[f"{key}__log_probs"][scores[f"{key}__family_idx"] == fi]
            if lp.size:
                per_slot.append(lp)
        if not per_slot:
            continue
        lp = np.concatenate(per_slot)
        l10 = lp / math.log(10)
        n_hi = sum(bool((np.exp(v) > 0.5).any()) for v in per_slot)
        lines.append(
            f"| {fam} | {math.exp(lp.max()):.4f} | 10^{np.percentile(l10, 99):.2f} | "
            f"10^{np.median(l10):.2f} | {int((np.exp(lp) > 0.1).sum())} | {n_hi}/{len(manifest['slots'])} |"
        )
    return lines


def main() -> None:
    results_default = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=results_default)
    ap.add_argument("--tag-date", default="20260726", help="date in the shift runs' names")
    ap.add_argument("--iid-tag-date", default="20260505", help="date in the Section 6.1 runs' names")
    ap.add_argument("--scan-dir", type=Path, default=None,
                    help="leak scan directory (default: RESULTS_ROOT/"
                         "password_bank_perfam_scores_may_v1_0p6b_<tag date>; skipped if absent)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: RESULTS_ROOT/summaries/distshift_results.md)")
    args = ap.parse_args()
    R, d, di = args.results_root, args.tag_date, args.iid_tag_date
    out = args.out or R / "summaries" / "distshift_results.md"
    scan_dir = args.scan_dir or R / f"password_bank_perfam_scores_may_v1_0p6b_{d}"

    # Original bank, with the shift.
    rows, base, trained, cal, ours_cal, decades, kl = [], [], [], [], [], [], []
    under = pairs = 0
    for s in SEEDS:
        run, best = R / SHIFT_RUN.format(s=s, d=d), R / SHIFT_BEST.format(s=s, d=d)
        b_t = per_target(run / "baseline_test_results.json")
        t_t = per_target(best / "test_results.json")
        b, t = mean_error(run / "baseline_test_results.json"), mean_error(best / "test_results.json")
        c = calibrated_error(run / "posthoc_calibration_baseline.json")
        oc = calibrated_error(best / "posthoc_calibration_trained.json")
        base.append(b)
        trained.append(t)
        if c is not None:
            cal.append(c)
        if oc is not None:
            ours_cal.append(oc)
        decades.append(statistics.mean(
            (b_t[k]["actual_worst_log_prob"] - t_t[k]["actual_worst_log_prob"]) / math.log(10) for k in t_t))
        under += sum(r["predicted_worst_log_prob"] < r["actual_worst_log_prob"] for r in b_t.values())
        pairs += len(b_t)
        kl.append(json.loads((best / "kl_eval_results.json").read_text())["single_token"])
        rows.append(f"| {s} | {b:.2f} | {t:.3f} | {b / t:.1f}x | {fmt(c, '.3f')} | {fmt(oc, '.3f')} |")
    fold_m, fold_se, fold_med = fold_summary(base, trained)

    # Section 6.1 runs, without a shift.
    iid_b = [mean_error(R / IID_RUN.format(s=s, d=di) / "baseline_test_results.json") for s in SEEDS]
    iid_t = [mean_error(R / IID_RUN.format(s=s, d=di) / "test_results.json") for s in SEEDS]
    iid_fm, iid_fse, iid_fmed = fold_summary(iid_b, iid_t)
    iid_cal = [c for s in SEEDS
               if (c := calibrated_error(R / IID_RUN.format(s=s, d=di) / "posthoc_calibration_baseline.json")) is not None]

    # Hardened bank, with the shift (final checkpoints).
    hard = []
    for name, pattern in (("Ours", HARD_OURS), ("SFT", HARD_SFT)):
        hb = [mean_error(R / pattern.format(s=s, d=d) / "baseline_test_results.json") for s in SEEDS]
        ht = [mean_error(R / pattern.format(s=s, d=d) / "test_results.json") for s in SEEDS]
        hc = [c for s in SEEDS
              if (c := calibrated_error(R / pattern.format(s=s, d=d) / "posthoc_calibration_trained.json")) is not None]
        fm, fse, fmed = fold_summary(hb, ht)
        hard.append(f"| {name} | {statistics.mean(hb):.2f} | {statistics.mean(ht):.3f} | {fm:.1f} +/- {fse:.1f} | "
                    f"{fmed:.1f} | {fmt(statistics.mean(hc) if hc else None, '.3f')} ({len(hc)} seeds) |")

    lines = [
        "# Fit/deployment distribution shift",
        "",
        "Written by scripts/password_game/collect_distshift_results.py. Errors are squared log10",
        "forecast errors averaged over each seed's seven held-out passwords; folds are per-seed",
        "ratios of the pretrained error to the trained error.",
        "",
        "## Original bank, per seed",
        "",
        "The trained model is the checkpoint selected by training-side forecast error (best/).",
        "",
        "| seed | baseline | trained (best/) | fold | Cal. alone | Ours+cal |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## Table: forecast error under a known fit/deployment family shift",
        "",
        "| metric | original IID split | family shift |",
        "|---|---|---|",
        f"| pretrained error | {statistics.mean(iid_b):.2f} | {statistics.mean(base):.2f} |",
        f"| Ours error | {statistics.mean(iid_t):.3f} | {statistics.mean(trained):.3f} |",
        f"| Ours fold (mean +/- SEM) | {iid_fm:.1f} +/- {iid_fse:.1f} | {fold_m:.1f} +/- {fold_se:.1f} |",
        f"| median per-seed fold | {iid_fmed:.1f} | {fold_med:.1f} |",
        f"| calibrated pretrained error | {fmt(statistics.mean(iid_cal) if iid_cal else None, '.3f')} "
        f"({len(iid_cal)} seeds) | {fmt(statistics.mean(cal) if cal else None, '.3f')} ({len(cal)} seeds) |",
        f"| Ours + calibration error | -- | {fmt(statistics.mean(ours_cal) if ours_cal else None, '.3f')} "
        f"({len(ours_cal)} seeds) |",
        "",
        "## Other numbers in the text (original bank, with the shift)",
        "",
        f"- Held-out (seed, password) pairs where the pretrained forecast under-predicted the worst leak: {under} of {pairs}.",
        f"- Mean reduction of the worst held-out leak probability by training: {statistics.mean(decades):.1f} decades.",
        f"- Mean single-token KL from the pretrained model on WildChat prompts: {statistics.mean(kl):.5f}.",
        "",
        "## Hardened bank, with the shift (final checkpoints)",
        "",
        "| method | pretrained error | trained error | fold (mean +/- SEM) | median fold | error after calibration |",
        "|---|---|---|---|---|---|",
        *hard,
    ]
    if (scan_dir / "manifest.json").exists():
        lines += ["", f"## Leak scan by prompt family ({scan_dir.name})", "", *leak_scan(scan_dir)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
