"""Collect the numbers of the natural-prompt appendix ("Forecastability on natural user prompts")
from the run directories.

For each Qwen3 size and seed it selects the checkpoint of the forecastability run and of the SFT
run with select_natural_checkpoint.py, reads that checkpoint's held-out results, and writes one
row per size and seed to $RESULTS_ROOT/summaries/natural_results_verified.json, the input of
the natural_paired_errors figure (scripts/figures/make_appendix_figures.py). Row fields: run
(the result directory, relative to RESULTS_ROOT), checkpoint (the evaluation's arm label, or
"final" for the endpoint), error and baseline_error (trained and pretrained worst-rank squared
log10 forecast errors, averaged over the seed's held-out tasks), fold (baseline_error / error),
kl (single-token KL on the 256-prompt evaluation set) and risk_reduction (decades by which the
worst-prompt probability fell, averaged over held-out tasks).

It also prints every other number the appendix reports: Table natural-main, the decade means,
pooled folds and KL ranges, the KL-sensitivity table (natural-kl), the two extra KL-2.0 seeds,
the blocklist-prompt control, the post-hoc calibration folds, and the training recipe of every
run. With --compare it checks the rows field by field against another copy of the JSON.

The runs are those of scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh; the
post-hoc calibration files come from scripts/password_game/run_posthoc_calibration_baseline.sbatch
run on each forecastability run. Usage:
  python scripts/wildchat/natural_prompts/collect_natural_results.py [--compare OTHER.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_natural_checkpoint import KL_MAX, RESULTS_ROOT, select  # noqa: E402

SIZES = {"0.6b": "qwen_qwen3-0_6b", "1.7b": "qwen_qwen3-1_7b", "4b": "qwen_qwen3-4b", "8b": "qwen_qwen3-8b"}
SEEDS = (11, 17, 23)
# Recipe of the appendix's forecastability runs, as recorded in each run's config.json.
OURS_RECIPE = {"steps": 200, "lr": 3e-5, "targets_per_step": 8, "lora_r": 8, "lora_alpha": 16,
               "kl_coef": 0.5, "union_cache_refresh_interval": 20, "union_cache_top_c": 296,
               "ols_top_k": 10, "plotting_position": "weibull", "improving_only_grad": True,
               "iog_scope": "both", "grad_clip": 1.0, "token_union_blocklist_prompt": False}
SFT_RECIPE = {"lr": 1e-5, "lora_r": 8, "lora_alpha": 16, "kl_coef": 0.5}
SFT_BUDGET = {"0.6b": (2000, 128), "1.7b": (2000, 128), "4b": (4000, 64), "8b": (8000, 32)}
KL_TRAIN_FILE = "kl_train_nonflagged_en_disjoint.jsonl"


def ours_run(results_root: Path, size: str, seed: int, suffix: str = "mb8klnf05") -> Path:
    return results_root / f"lquant_wildchat_ldnoobw_{SIZES[size]}_iog_full_boot1_cache296_seed{seed}_{suffix}" / f"seed_{seed}"


def sft_run(results_root: Path, size: str, seed: int) -> Path:
    return results_root / f"lquant_wildchat_ldnoobw_sft_union_{SIZES[size]}_klnf05_seed{seed}" / f"seed_{seed}"


def _load(p: Path):
    return json.loads(Path(p).read_text())


def metrics(result_dir: Path) -> dict:
    t = _load(result_dir / "test_results.json")
    b = _load(result_dir / "baseline_test_results.json")
    err = sum(v["forecast_sq_log10_error"] for v in t.values()) / len(t)
    berr = sum(v["forecast_sq_log10_error"] for v in b.values()) / len(b)
    rr = sum((b[k]["actual_worst_log_prob"] - t[k]["actual_worst_log_prob"]) / math.log(10) for k in t) / len(t)
    return {"error": err, "baseline_error": berr, "fold": berr / err,
            "kl": _load(result_dir / "kl_eval_results.json")["single_token"], "risk_reduction": rr,
            "worst_task_error": max(v["forecast_sq_log10_error"] for v in t.values())}


def chosen_row(results_root: Path, run_dir: Path, kl_max: float) -> tuple[dict, dict]:
    """(row fields for the JSON, selection details)."""
    chosen, cands, chain, later = select(run_dir, results_root, kl_max)
    if chosen is None:
        raise SystemExit(f"{run_dir}: no evaluated checkpoint meets KL <= {kl_max}")
    m = metrics(chosen.result_dir)
    row = {"run": os.path.relpath(chosen.result_dir, results_root), "checkpoint": chosen.label,
           "error": m["error"], "baseline_error": m["baseline_error"], "fold": m["fold"],
           "kl": m["kl"], "risk_reduction": m["risk_reduction"]}
    info = {"step": chosen.step, "worst_task_error": m["worst_task_error"],
            "candidates": [(c.label, c.objective, c.kl) for c in sorted(cands, key=lambda c: c.objective)],
            "unevaluated_later": later, "chain_tail": [s for s, _ in chain[-4:]]}
    return row, info


def recipe_check(run_dir: Path, want: dict) -> list[str]:
    cfg = _load(run_dir / "config.json")
    bad = [f"{k}={cfg.get(k)!r} (want {v!r})" for k, v in want.items() if cfg.get(k) != v]
    kl_train = [Path(p).name for p in cfg.get("kl_train_datasets", [])]
    if kl_train != [KL_TRAIN_FILE]:
        bad.append(f"kl_train_datasets={kl_train}")
    return bad


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def se(xs):
    xs = list(xs)
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def compare(rows: list[dict], other: list[dict]) -> None:
    key = lambda r: (r["size"], r["seed"])  # noqa: E731
    ours = {key(r): r for r in rows}
    worst_abs = (0.0, None)
    worst_rel = (0.0, None)
    mismatches = []
    missing = sorted(set(map(key, other)) ^ set(ours))
    for r in other:
        mine = ours.get(key(r))
        if mine is None:
            continue
        for arm in ("ours", "sft"):
            for f, v in r[arm].items():
                w = mine[arm].get(f)
                if isinstance(v, str):
                    norm = lambda s: s.split("artifacts/results/", 1)[-1]  # noqa: E731
                    if w is None or norm(v) != norm(w):
                        mismatches.append((key(r), arm, f, v, w))
                    continue
                d = abs(v - w)
                rel = d / abs(v) if v else d
                if d >= worst_abs[0]:
                    worst_abs = (d, (key(r), arm, f, v, w))
                if rel >= worst_rel[0]:
                    worst_rel = (rel, (key(r), arm, f, v, w))
    print(f"\n[compare] {len(other)} rows; cells missing on one side: {missing or 'none'}")
    print(f"[compare] largest absolute difference: {worst_abs[0]:.3g} at {worst_abs[1]}")
    print(f"[compare] largest relative difference: {worst_rel[0]:.3g} at {worst_rel[1]}")
    print(f"[compare] string mismatches: {mismatches or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <results-root>/summaries/natural_results_verified.json")
    ap.add_argument("--kl-max", type=float, default=KL_MAX)
    ap.add_argument("--compare", type=Path, default=None, help="another natural_results_verified.json")
    args = ap.parse_args()
    root = args.results_root
    out = args.out or root / "summaries" / "natural_results_verified.json"

    rows, info = [], {}
    print(f"[select] KL bound {args.kl_max}; candidates sorted by training-side objective")
    for size in SIZES:
        for seed in SEEDS:
            row = {"size": size, "seed": seed}
            for arm, rd, want in (("ours", ours_run(root, size, seed), OURS_RECIPE),
                                  ("sft", sft_run(root, size, seed), SFT_RECIPE)):
                row[arm], info[(size, seed, arm)] = chosen_row(root, rd, args.kl_max)
                i = info[(size, seed, arm)]
                bad = recipe_check(rd, want)
                if arm == "sft":
                    cfg = _load(rd / "config.json")
                    if (cfg["steps"], cfg["sft_batch_per_step"]) != SFT_BUDGET[size]:
                        bad.append(f"budget {cfg['steps']}x{cfg['sft_batch_per_step']}")
                print(f"  {size:4s} s{seed} {arm:4s} chosen {row[arm]['checkpoint']:18s} "
                      f"kl={row[arm]['kl']:.4g} recipe {'ok' if not bad else bad} "
                      f"| candidates {[(l, round(o, 6), round(k, 4)) for l, o, k in i['candidates']]}"
                      + (f" | best/ steps after the choice without an evaluation: {i['unevaluated_later']}"
                         if i["unevaluated_later"] else ""))
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"[write] {out}")

    print("\nTable natural-main (means over seeds 11, 17, 23)")
    by = {s: [r for r in rows if r["size"] == s] for s in SIZES}
    print("  size  pretrained   SFT err  ours err | SFT folds              | ours folds             | ours fold mean")
    for s, rs in by.items():
        sf = [r["sft"]["fold"] for r in rs]
        of = [r["ours"]["fold"] for r in rs]
        print(f"  {s:4s}  {mean(r['ours']['baseline_error'] for r in rs):9.3f} {mean(r['sft']['error'] for r in rs):9.3f}"
              f" {mean(r['ours']['error'] for r in rs):9.3f} | {', '.join(f'{x:.3g}' for x in sf):22s} "
              f"| {', '.join(f'{x:.3g}' for x in of):22s} | {mean(of):.3g} +- {se(of):.3g}")
    print("\nText numbers")
    for s, rs in by.items():
        print(f"  {s:4s} decades (ours) {mean(r['ours']['risk_reduction'] for r in rs):.1f}"
              f" | pooled fold ours {sum(r['ours']['baseline_error'] for r in rs) / sum(r['ours']['error'] for r in rs):.3g}"
              f" SFT {sum(r['sft']['baseline_error'] for r in rs) / sum(r['sft']['error'] for r in rs):.3g}"
              f" | KL ours {min(r['ours']['kl'] for r in rs):.2g}-{max(r['ours']['kl'] for r in rs):.2g}"
              f" SFT {min(r['sft']['kl'] for r in rs):.2g}-{max(r['sft']['kl'] for r in rs):.2g}")
    n_ours_up = sum(r["ours"]["fold"] > 1 for r in rows)
    n_sft_down = sum(r["sft"]["fold"] < 1 for r in rows)
    sft_best_seed = {s: max(rs, key=lambda r: r["sft"]["fold"])["seed"] for s, rs in by.items()}
    print(f"  ours runs that improved the forecast: {n_ours_up}/{len(rows)}; SFT runs that worsened it:"
          f" {n_sft_down}/{len(rows)}; SFT's best seed per size: {sft_best_seed}")

    print("\nTable natural-kl (8B, seed 17) and the extra KL-2.0 seeds")
    extra = [("0.5", "8b", 17, "mb8klnf05"), ("2.0", "8b", 17, "mb8klnf20"), ("5.0", "8b", 17, "mb8klnf50"),
             ("2.0", "8b", 11, "mb8klnf20"), ("2.0", "8b", 23, "mb8klnf20")]
    for w, size, seed, suffix in extra:
        rd = ours_run(root, size, seed, suffix)
        row, i = chosen_row(root, rd, args.kl_max)
        bad = recipe_check(rd, {**OURS_RECIPE, "kl_coef": float(w)})
        print(f"  weight {w} s{seed}: step {i['step'] if i['step'] is not None else 'endpoint'} ({row['checkpoint']})"
              f" fold {row['fold']:.3g} decades {row['risk_reduction']:.1f} KL {row['kl']:.3g} recipe {'ok' if not bad else bad}"
              f" | candidates {[(l, round(o, 6), round(k, 4)) for l, o, k in i['candidates']]}")

    print("\nBlocklist-prompt control (0.6B, seed 17)")
    rd = ours_run(root, "0.6b", 17, "blkklnf05")
    row, i = chosen_row(root, rd, args.kl_max)
    cfg = _load(rd / "config.json")
    plain = next(r for r in rows if (r["size"], r["seed"]) == ("0.6b", 17))["ours"]["baseline_error"]
    # The appendix's fold divides the pretrained error without the system prompt (the main 0.6B
    # seed-17 cell) by this checkpoint's error; the evaluation's own baseline_test_results.json
    # scores the pretrained model with the blocklist prompt.
    print(f"  step {i['step']} ({row['checkpoint']}) fold {plain / row['error']:.3g} (pretrained without the prompt"
          f" {plain:.4g} / trained {row['error']:.4g}; {row['fold']:.3g} against the pretrained model with the prompt,"
          f" {row['baseline_error']:.4g}) KL {row['kl']:.2g} worst-task error {i['worst_task_error']:.3g} | recipe: blocklist prompt"
          f" {cfg.get('token_union_blocklist_prompt')}, lr {cfg.get('lr')}, targets/step {cfg.get('targets_per_step')},"
          f" KL {cfg.get('kl_coef')} on {[Path(p).name for p in cfg.get('kl_train_datasets', [])]},"
          f" refresh {cfg.get('union_cache_refresh_interval')}, steps {cfg.get('steps')}")

    print("\nPost-hoc calibration of the pretrained forecast (posthoc_calibration_baseline.json in each ours run)")
    shift, affine, missing = [], [], []
    for size in SIZES:
        for seed in SEEDS:
            p = ours_run(root, size, seed) / "posthoc_calibration_baseline.json"
            if not p.exists():
                missing.append((size, seed))
                continue
            h = _load(p)["heldout_calibrated"]
            raw = h["heldout_summary_no_calibration"]["mean_sq_log10_error"]
            shift.append(raw / h["heldout_summary_shift_1param"]["mean_sq_log10_error"])
            affine.append(raw / h["heldout_summary_shift_scale_2param"]["mean_sq_log10_error"])
            base = next(r for r in rows if (r["size"], r["seed"]) == (size, seed))["ours"]["baseline_error"]
            if abs(raw - base) > 1e-9:
                print(f"  note: {size} s{seed} calibration file's uncalibrated error {raw} != baseline {base}")
    if shift:
        print(f"  {len(shift)} runs: constant shift folds {min(shift):.2g}-{max(shift):.2g},"
              f" affine folds {min(affine):.2g}-{max(affine):.2g}; missing: {missing or 'none'}")

    if args.compare:
        compare(rows, _load(args.compare))


if __name__ == "__main__":
    main()
