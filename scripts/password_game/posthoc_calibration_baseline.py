"""Post-hoc shift / shift-and-scale calibration of the FRLMB OLS forecast.

The paper's post-hoc calibration baseline (Cal.). Two calibration variants:

    1-param (additive shift, equivalent to shifting the OLS y-intercept):
        pred_calibrated = pred_raw + beta
    2-param (Platt-style shift+scale, equivalent to a' = alpha*a + beta, b' = alpha*b):
        pred_calibrated = alpha * pred_raw + beta

Calibration parameters are fit on TRAIN-target (fit, deploy) pairs under the
PRETRAINED model (no LoRA adapters trained), then applied to the HELDOUT-target
results from `<run_dir>/baseline_test_results.json`. Output is written to
`<out_json>` for downstream comparison against meta-trained results.

Usage:
    python scripts/password_game/posthoc_calibration_baseline.py \\
        --run-dir artifacts/results/<run tag>/seed_11 \\
        --out-json artifacts/results/<run tag>/seed_11/posthoc_calibration_baseline.json

(scripts/password_game/run_posthoc_calibration_baseline.sbatch wraps this on a GPU.)

The script loads the prompt bank + target partition + cfg from <run_dir> and
computes train-target predicted/actual using the BASE pretrained model (loaded
fresh from cfg.model_id, no LoRA).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
# Make src/ importable when run from anywhere
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from forecast_experiment.lquant_prompt_bank import resolve_prompt_bank
from forecast_experiment.ols_utils import fit_ols_tail_from_scores
from forecast_experiment.train_lquant import (
    LQuantConfig,
    _effective_prompt_bank_seed,
    _filter_target_records,
    _load_prompt_pool_from_bank,
    _load_prompt_pool_from_bank_token_union,
    _resolve_target_partition,
    build_prompt_pool,
    compute_password_logprobs_batched,
    split_pool,
)


def _load_cfg(run_dir: Path) -> LQuantConfig:
    """Reconstruct an LQuantConfig from a run's saved config.json."""
    cfg_path = run_dir / "config.json"
    raw = json.loads(cfg_path.read_text())
    valid_fields = {f.name for f in fields(LQuantConfig)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    cfg = LQuantConfig(**filtered)
    return cfg


def _train_target_predictions(
    cfg: LQuantConfig,
    train_recs: Dict[str, List[Dict]],
    eval_recs: Dict[str, List[Dict]],
    train_targets: List[str],
    *,
    lora_checkpoint: Path | None = None,
    n_partitions: int = 1,
) -> List[Dict[str, float]]:
    """For each train target's (fit, deploy) pair, compute the FRLMB-forecast
    predicted_worst_log_prob and actual_worst_log_prob under the chosen model.

    If lora_checkpoint is None, uses the PRETRAINED model (LoRA at PEFT init,
    equivalent to base). If lora_checkpoint is a path, loads that LoRA adapter
    on top of the base model -- useful for calibrating the META-TRAINED forecast.

    These constitute the calibration set.

    Partition handling. The paper-canonical procedure fits the calibrator on ONE
    (predicted_worst, actual_worst) pair per train target, drawn from the
    deterministic ``cfg.seed + 999`` shuffle that ``split_pool`` produces.
    When ``n_partitions > 1`` we additionally resample ``n_partitions - 1``
    fresh fit/deploy partitions over the same scored union of fit and deploy
    prompts (no extra forward passes -- the union is already scored), yielding
    ``n_partitions`` (predicted, actual) pairs per target and ``n_partitions *
    len(train_targets)`` calibration points overall. This deviates from the
    paper's single-partition calibration; ``n_partitions=1`` reproduces the
    paper exactly.
    """
    from importance_sampling.policy_lm import LoRASetup, PolicyLM

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    label = "PRETRAINED (no adapters trained)" if lora_checkpoint is None else f"BASE + LoRA from {lora_checkpoint}"
    print(f"Loading {label} model {cfg.model_id} on {device} ...", flush=True)
    if lora_checkpoint is None:
        # Load with use_lora=True so the LoRA structure is identical to the
        # meta-train path, but the adapters have not been trained -- they're at
        # the PEFT initialization (typically B=0, A=random), equivalent to base.
        policy = PolicyLM(
            model_id=cfg.model_id,
            device=device,
            lora=LoRASetup(
                use_lora=cfg.use_lora,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
            ),
        )
        model = policy.model.eval()
        tokenizer = policy.tokenizer
    else:
        # Load base + adapter from disk via PEFT.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype="auto").to(device)
        model = PeftModel.from_pretrained(base, str(lora_checkpoint)).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    out: List[Dict[str, float]] = []
    started = time.perf_counter()
    for idx, target in enumerate(train_targets, start=1):
        fit_recs = train_recs.get(target, [])
        deploy_recs = eval_recs.get(target, [])
        if not fit_recs or not deploy_recs:
            continue
        with torch.no_grad():
            fit_lp = compute_password_logprobs_batched(
                model, tokenizer, fit_recs,
                enable_thinking=(not cfg.disable_thinking),
                device=device,
                micro_batch_size=cfg.micro_batch_size,
                token_budget=int(getattr(cfg, "score_token_budget", 0) or 0),
                require_grad=False,
            )
            deploy_lp = compute_password_logprobs_batched(
                model, tokenizer, deploy_recs,
                enable_thinking=(not cfg.disable_thinking),
                device=device,
                micro_batch_size=cfg.micro_batch_size,
                token_budget=int(getattr(cfg, "score_token_budget", 0) or 0),
                require_grad=False,
            )
        fit_log_probs = [float(v) for v in fit_lp.float().tolist()]
        deploy_log_probs = [float(v) for v in deploy_lp.float().tolist()]
        # Partition 0: paper-canonical seed+999 partition (don't touch the order
        # produced by split_pool; reproduces the existing K=1 behavior exactly).
        pred_log_prob, fit_a, fit_b = fit_ols_tail_from_scores(
            fit_log_probs,
            n_top=cfg.ols_top_k,
            plotting=cfg.plotting_position,
            deploy_queries=len(deploy_log_probs),
            score_mode="gumbel_log_prob",
        )
        actual_worst = float(deploy_lp.float().max().item())
        out.append({
            "target": target,
            "partition_idx": 0,
            "fit_count": len(fit_log_probs),
            "deploy_count": len(deploy_log_probs),
            "predicted_worst_log_prob": float(pred_log_prob),
            "actual_worst_log_prob": actual_worst,
            "fit_ols_a": float(fit_a),
            "fit_ols_b": float(fit_b),
        })

        # Additional resampled partitions over the SAME scored prompts (zero
        # extra forward passes). Each partition gets a fresh shuffle of the
        # fit+deploy union; first n_fit go to fit, rest to deploy. Seeded
        # deterministically by (cfg.seed, target, partition_idx).
        if n_partitions > 1:
            import hashlib
            n_fit = len(fit_log_probs)
            union = fit_log_probs + deploy_log_probs
            n_union = len(union)
            for pi in range(1, int(n_partitions)):
                digest = hashlib.sha256(
                    f"{cfg.seed}||{target}||posthoc_partition||{pi}".encode("utf-8")
                ).digest()
                seed = int.from_bytes(digest[:8], "big", signed=False)
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n_union)
                fit_idx = perm[:n_fit]
                deploy_idx = perm[n_fit:]
                fit_lp_p = [union[i] for i in fit_idx]
                deploy_lp_p = [union[i] for i in deploy_idx]
                pred_p, fit_a_p, fit_b_p = fit_ols_tail_from_scores(
                    fit_lp_p,
                    n_top=cfg.ols_top_k,
                    plotting=cfg.plotting_position,
                    deploy_queries=len(deploy_lp_p),
                    score_mode="gumbel_log_prob",
                )
                actual_p = float(max(deploy_lp_p))
                out.append({
                    "target": target,
                    "partition_idx": pi,
                    "fit_count": len(fit_lp_p),
                    "deploy_count": len(deploy_lp_p),
                    "predicted_worst_log_prob": float(pred_p),
                    "actual_worst_log_prob": actual_p,
                    "fit_ols_a": float(fit_a_p),
                    "fit_ols_b": float(fit_b_p),
                })
        if idx % max(1, len(train_targets) // 10) == 0 or idx == len(train_targets):
            print(
                f"  train target {idx}/{len(train_targets)} | target={target} "
                f"fit={len(fit_log_probs)} deploy={len(deploy_log_probs)} "
                f"| elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return out


def _fit_calibrations(cal: List[Dict[str, float]]) -> Dict[str, Any]:
    """Fit 1-param (additive shift) and 2-param (shift+scale) calibrations."""
    pred = np.array([c["predicted_worst_log_prob"] for c in cal], dtype=float)
    actual = np.array([c["actual_worst_log_prob"] for c in cal], dtype=float)
    finite = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[finite]
    actual = actual[finite]
    n = len(pred)
    if n < 2:
        raise RuntimeError(f"Not enough finite calibration points: {n}")

    # 1-param: pred + beta_1 ~= actual, MLE: beta_1 = mean(actual - pred)
    beta_1 = float(np.mean(actual - pred))

    # 2-param: alpha * pred + beta_2 ~= actual, OLS
    A = np.column_stack([pred, np.ones_like(pred)])
    coef, _residual, _rank, _sv = np.linalg.lstsq(A, actual, rcond=None)
    alpha_2 = float(coef[0])
    beta_2 = float(coef[1])

    # Calibration-set fit quality
    pred_1cal = pred + beta_1
    pred_2cal = alpha_2 * pred + beta_2
    return {
        "n_calibration_pairs": int(n),
        "calibration_pred_mean": float(pred.mean()),
        "calibration_pred_std": float(pred.std()),
        "calibration_actual_mean": float(actual.mean()),
        "calibration_actual_std": float(actual.std()),
        "calibration_raw_signed_mean": float(np.mean(actual - pred)),  # = beta_1
        "calibration_raw_sq_error_mean": float(np.mean((actual - pred) ** 2)),
        "shift_1param": {
            "beta": beta_1,
            "calibration_sq_error_mean": float(np.mean((pred_1cal - actual) ** 2)),
        },
        "shift_scale_2param": {
            "alpha": alpha_2,
            "beta": beta_2,
            "calibration_sq_error_mean": float(np.mean((pred_2cal - actual) ** 2)),
        },
    }


def _apply_to_heldout(
    heldout_results: Dict[str, Dict],
    calibrations: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply both calibrations to the heldout results from baseline_test_results.json.

    Returns per-slot calibrated predictions plus summary metrics.
    """
    beta_1 = calibrations["shift_1param"]["beta"]
    alpha_2 = calibrations["shift_scale_2param"]["alpha"]
    beta_2 = calibrations["shift_scale_2param"]["beta"]

    per_slot: Dict[str, Dict] = {}
    raw_pred = []
    raw_actual = []
    cal1_pred = []
    cal2_pred = []
    for slot, r in heldout_results.items():
        pred_raw = float(r["predicted_worst_log_prob"])
        actual = float(r["actual_worst_log_prob"])
        if not (math.isfinite(pred_raw) and math.isfinite(actual)):
            continue
        pred_1 = pred_raw + beta_1
        pred_2 = alpha_2 * pred_raw + beta_2
        per_slot[slot] = {
            "predicted_worst_log_prob_raw": pred_raw,
            "actual_worst_log_prob": actual,
            "predicted_worst_log_prob_shift_1param": pred_1,
            "predicted_worst_log_prob_shift_scale_2param": pred_2,
            "signed_log10_error_raw": (pred_raw - actual) / math.log(10),
            "signed_log10_error_shift_1param": (pred_1 - actual) / math.log(10),
            "signed_log10_error_shift_scale_2param": (pred_2 - actual) / math.log(10),
            "sq_log10_error_raw": ((pred_raw - actual) / math.log(10)) ** 2,
            "sq_log10_error_shift_1param": ((pred_1 - actual) / math.log(10)) ** 2,
            "sq_log10_error_shift_scale_2param": ((pred_2 - actual) / math.log(10)) ** 2,
        }
        raw_pred.append(pred_raw)
        raw_actual.append(actual)
        cal1_pred.append(pred_1)
        cal2_pred.append(pred_2)

    raw_pred_a = np.array(raw_pred)
    raw_actual_a = np.array(raw_actual)
    cal1_pred_a = np.array(cal1_pred)
    cal2_pred_a = np.array(cal2_pred)

    log10 = math.log(10)

    def _summary(pred_a, actual_a):
        signed_log10 = (pred_a - actual_a) / log10
        return {
            "n": int(len(pred_a)),
            "mean_signed_log10_error": float(np.mean(signed_log10)),
            "mean_abs_signed_log10_error": float(np.mean(np.abs(signed_log10))),
            "mean_sq_log10_error": float(np.mean(signed_log10 ** 2)),
            "median_signed_log10_error": float(np.median(signed_log10)),
        }

    return {
        "per_slot": per_slot,
        "heldout_summary_no_calibration": _summary(raw_pred_a, raw_actual_a),
        "heldout_summary_shift_1param": _summary(cal1_pred_a, raw_actual_a),
        "heldout_summary_shift_scale_2param": _summary(cal2_pred_a, raw_actual_a),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="A meta-train run dir (e.g. artifacts/results/.../seed_11/) "
                        "containing config.json, target_partition.json, and "
                        "baseline_test_results.json.")
    p.add_argument("--out-json", type=Path, required=True,
                   help="Path to write the calibration results JSON.")
    p.add_argument("--micro-batch-size-override", type=int, default=0,
                   help="If >0, override cfg.micro_batch_size for the calibration eval pass.")
    p.add_argument("--lora-checkpoint", type=Path, default=None,
                   help="If set, load this LoRA adapter on top of base (e.g. <run_dir>/final/) "
                        "and calibrate the META-TRAINED forecast instead of the pretrained one. "
                        "When set, you almost certainly also want to set --heldout-results-json "
                        "to test_results.json (meta-trained heldout) instead of the default "
                        "baseline_test_results.json (pretrained heldout).")
    p.add_argument("--heldout-results-json", default="baseline_test_results.json",
                   help="Filename within --run-dir to read heldout (predicted, actual) from. "
                        "Use 'baseline_test_results.json' (default) for pretrained heldout, or "
                        "'test_results.json' for meta-trained heldout.")
    p.add_argument("--n-partitions", type=int, default=1,
                   help="How many (fit, deploy) partitions per train target to use when "
                        "fitting the calibrator. 1 = paper-canonical (cfg.seed+999 shuffle "
                        "only). K>1 resamples K-1 additional partitions over the SAME scored "
                        "fit+deploy union (zero extra forward passes), giving K*N_targets "
                        "calibration pairs. This deviates from the paper's single-partition "
                        "calibration.")
    args = p.parse_args()

    run_dir: Path = args.run_dir
    cfg = _load_cfg(run_dir)
    if args.micro_batch_size_override > 0:
        cfg.micro_batch_size = args.micro_batch_size_override

    # Load prompt pool same way the run did
    if cfg.prompt_bank_root:
        bank_seed = _effective_prompt_bank_seed(cfg)
        bank_info = resolve_prompt_bank(cfg.prompt_bank_root, bank_seed)
        # We need a tokenizer here for chat-template signature; use base tokenizer
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model_id)
        bank_task_kind = str(bank_info.metadata.get("task_kind", "password"))
        if bank_task_kind == "token_union":
            pool = _load_prompt_pool_from_bank_token_union(bank_info, cfg=cfg, tokenizer=tok)
        else:
            pool = _load_prompt_pool_from_bank(bank_info, cfg=cfg, tokenizer=tok)
    else:
        # build_prompt_pool needs a model+tokenizer. Easier to just load model now.
        from importance_sampling.policy_lm import LoRASetup, PolicyLM
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg.device = device
        policy = PolicyLM(
            model_id=cfg.model_id,
            device=device,
            lora=LoRASetup(
                use_lora=cfg.use_lora, r=cfg.lora_r, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules,
            ),
        )
        pool = build_prompt_pool(cfg, policy.model, policy.tokenizer)

    train_recs, eval_recs, _test_recs = split_pool(pool, cfg)
    target_partition = json.loads((run_dir / "target_partition.json").read_text())
    train_targets = list(target_partition["train_targets"])
    print(f"Train targets: {len(train_targets)}", flush=True)

    cal_data = _train_target_predictions(
        cfg, train_recs, eval_recs, train_targets,
        lora_checkpoint=args.lora_checkpoint,
        n_partitions=args.n_partitions,
    )
    calibrations = _fit_calibrations(cal_data)

    heldout_path = run_dir / args.heldout_results_json
    heldout_results = json.loads(heldout_path.read_text())
    heldout_summary = _apply_to_heldout(heldout_results, calibrations)

    output = {
        "run_dir": str(run_dir),
        "model_id": cfg.model_id,
        "ols_top_k": cfg.ols_top_k,
        "plotting_position": cfg.plotting_position,
        "lora_checkpoint": str(args.lora_checkpoint) if args.lora_checkpoint else None,
        "heldout_results_source": args.heldout_results_json,
        "n_partitions": int(args.n_partitions),
        "calibrations": calibrations,
        "heldout_calibrated": heldout_summary,
        "calibration_data_per_target": cal_data,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote calibration results to {args.out_json}", flush=True)
    print(f"\n=== Summary (heldout, n={heldout_summary['heldout_summary_no_calibration']['n']}) ===", flush=True)
    for label, key in [
        ("no calibration", "heldout_summary_no_calibration"),
        ("shift only (1 param)", "heldout_summary_shift_1param"),
        ("shift + scale (2 params)", "heldout_summary_shift_scale_2param"),
    ]:
        s = heldout_summary[key]
        print(
            f"  {label:25s} | mean signed log10 = {s['mean_signed_log10_error']:+.3f} | "
            f"mean |signed log10| = {s['mean_abs_signed_log10_error']:.3f} | "
            f"mean sq log10 = {s['mean_sq_log10_error']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
