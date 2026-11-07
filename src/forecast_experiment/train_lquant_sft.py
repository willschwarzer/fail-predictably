"""
train_lquant_sft.py
====================

Direct-risk-minimization SFT baseline for the LM password game. Replaces the
FRLMB-aware meta-loss in train_lquant.py with vanilla gradient descent on
log p(password | prompt), summed over a chosen prompt subset:

    L_sft(theta) = mean_{(prompt, password) in subset} log p_theta(password | prompt)

We minimize L_sft (i.e., push log_prob more negative = make the password less
likely = reduce per-prompt risk score). KL regularization to the base model
is preserved (same as meta-training) to keep the LoRA from degenerating on
benign behavior.

Subset choices (--sft-subset):
- "union":      train_recs[target] + eval_recs[target] for each train target.
                Tests "is the meta-training forecast improvement just a
                side-effect of pushing leak prob down on the same prompts?"
- "deploy_only":  eval_recs[target] only. Cheating-oracle baseline -- the
                training loop sees the actual deploy prompts. Upper bound on
                safety improvement; lower bound on heldout forecast quality.
- "fit_only":   train_recs[target] only. Less cheating; tests SFT on just
                the FRLMB fit-side prompts.

Reuses train_lquant.py's setup pipeline (config, model+LoRA load, prompt-bank
loaders, target-partition resolution) and end-of-training eval suite. Only
the training loop is replaced.

Run via the same CLI module:
    python -m forecast_experiment.run_train_lquant_sft \\
        --model Qwen/Qwen3-0.6B --steps 50 --lr 1e-5 \\
        --sft-subset union --outdir ./sft_out \\
        --prompt-bank-root <bank_dir> --prompt-bank-seed 11 \\
        --password-slots 40 --train-target-count 33 --heldout-target-count 7
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from .evaluate_model import prepare_messages
from .lquant_prompt_bank import resolve_prompt_bank
from .train_lquant import (
    LQuantConfig,
    _validate_fit_deploy_family_shift,
    _apply_chat_template_safe,
    _distributed_barrier,
    _distributed_enabled,
    _distributed_local_rank,
    _distributed_rank,
    _distributed_world_size,
    _effective_prompt_bank_seed,
    _filter_target_records,
    _load_prompt_pool_from_bank,
    _load_prompt_pool_from_bank_token_union,
    _resolve_target_partition,
    _shard_items,
    _step_rng,
    _target_step_rng,
    _tokenize_for_kl,
    _write_history_json,
    build_prompt_pool,
    compute_kl_single_token,
    compute_kl_truncated,
    compute_password_logprobs_batched,
    load_kl_dataset,
    run_evaluation_suite,
    split_pool,
)


def _add_lquant_cli(parser: argparse.ArgumentParser) -> None:
    """Mirror the CLI arguments used by `train_lquant.main` so launchers can be reused.

    We only declare the flags we actually use here. Anything specific to the
    meta-training path (loss-mode, ols-top-k, fit-fraction, eval-cache-*,
    improving-only-grad, bootstrap-permutations, plotting-position, etc.)
    is intentionally absent -- the SFT path doesn't use them.
    """
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--password-slots", type=int, default=19)
    parser.add_argument("--password-policy", default="non_disclosure")
    parser.add_argument("--train-target-count", type=int, default=0)
    parser.add_argument("--heldout-target-count", type=int, default=0)

    parser.add_argument("--cheap-prompts-per-family-per-target", type=int, default=64)
    parser.add_argument("--cheap-prompts-per-family-map", nargs="*", default=None)
    parser.add_argument("--train-families", nargs="*", default=None)
    parser.add_argument("--test-families", nargs="*", default=None)
    parser.add_argument(
        "--fit-families", nargs="*", default=None,
        help="Fit/deploy family shift (see train_lquant.py). With --sft-subset "
             "union, SFT trains on the shifted fit+deploy union of train slots.",
    )
    parser.add_argument("--deploy-families", nargs="*", default=None)
    parser.add_argument("--shift-fit-size", type=int, default=0)
    parser.add_argument("--shift-deploy-size", type=int, default=0)
    parser.add_argument("--shared-family-test-fraction", type=float, default=0.0)
    parser.add_argument("--prompt-bank-root", default="")
    parser.add_argument("--prompt-bank-seed", type=int, default=-1)

    parser.add_argument(
        "--ols-top-k", type=int, default=10,
        help="Eval-only here: OLS tail size for run_evaluation_suite forecasts.",
    )
    parser.add_argument(
        "--plotting-position", default="gringorten",
        choices=["gringorten", "weibull"],
        help="Eval-only here: plotting position for run_evaluation_suite "
             "forecasts. Pass 'weibull' to match the strict-v2 meta-training "
             "runs; the LQuantConfig default (gringorten) is kept as the "
             "argparse default so existing SFT launchers are unchanged.",
    )
    parser.add_argument("--split-mode", default="disjoint",
                        choices=["disjoint", "resample", "nested"])
    parser.add_argument("--fit-fraction", type=float, default=0.5,
                        help="Required for split_pool, but the SFT loop ignores the fit/eval distinction at training time.")

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--kl-trunc-len", type=int, default=8)
    parser.add_argument("--kl-trunc-every", type=int, default=5)
    parser.add_argument("--kl-batch-size", type=int, default=16)
    parser.add_argument("--kl-train-datasets", nargs="*", default=["password"])
    parser.add_argument("--kl-eval-datasets", nargs="*", default=[])

    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--disable-thinking", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    # token_union support -- same semantics as train_lquant.main; the shared
    # bank loader and scoring helper read these off LQuantConfig.
    parser.add_argument("--task-kind", default="password", choices=["password", "token_union"])
    parser.add_argument("--score-max-len", type=int, default=0)
    parser.add_argument("--score-token-budget", type=int, default=0)
    parser.add_argument("--kl-tokenize-max-len", type=int, default=512)
    parser.add_argument("--kl-truncation-side", default="right", choices=["right", "left"])
    parser.add_argument("--baseline-eval-cache-dir", default="")


def _build_cfg_from_args(args) -> LQuantConfig:
    """Produce an LQuantConfig from the SFT CLI args.

    Most fields take their LQuantConfig defaults. We only set the ones the SFT
    loop actually consults (or that are needed by shared helpers like
    split_pool, _resolve_target_partition, run_evaluation_suite).
    """
    cfg = LQuantConfig(
        model_id=args.model,
        device=args.device,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        password_slots=args.password_slots,
        password_policy=args.password_policy,
        train_target_count=args.train_target_count,
        heldout_target_count=args.heldout_target_count,
        cheap_prompts_per_family_per_target=args.cheap_prompts_per_family_per_target,
        train_families=args.train_families,
        test_families=args.test_families,
        fit_families=args.fit_families or [],
        deploy_families=args.deploy_families or [],
        shift_fit_size=args.shift_fit_size,
        shift_deploy_size=args.shift_deploy_size,
        shared_family_test_fraction=args.shared_family_test_fraction,
        prompt_bank_root=args.prompt_bank_root,
        prompt_bank_seed=args.prompt_bank_seed,
        split_mode=args.split_mode,
        fit_fraction=args.fit_fraction,
        ols_top_k=args.ols_top_k,
        plotting_position=args.plotting_position,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
        kl_coef=args.kl_coef,
        kl_trunc_len=args.kl_trunc_len,
        kl_trunc_every=args.kl_trunc_every,
        kl_batch_size=args.kl_batch_size,
        kl_train_datasets=list(args.kl_train_datasets),
        kl_eval_datasets=list(args.kl_eval_datasets),
        log_every=args.log_every,
        save_every=args.save_every,
        outdir=args.outdir,
        disable_thinking=args.disable_thinking,
        gradient_checkpointing=args.gradient_checkpointing,
        task_kind=args.task_kind,
        score_max_len=args.score_max_len,
        score_token_budget=args.score_token_budget,
        kl_tokenize_max_len=args.kl_tokenize_max_len,
        kl_truncation_side=args.kl_truncation_side,
        baseline_eval_cache_dir=args.baseline_eval_cache_dir,
    )
    # cheap_prompts_per_family_map needs special parsing
    if args.cheap_prompts_per_family_map:
        from .train_lquant import _parse_family_prompt_count_map
        cfg.cheap_prompts_per_family_map = _parse_family_prompt_count_map(args.cheap_prompts_per_family_map)
    return cfg


def _select_subset_records(
    *,
    train_recs: Dict[str, List[Dict]],
    eval_recs: Dict[str, List[Dict]],
    target: str,
    sft_subset: str,
) -> List[Dict]:
    """Return the prompt records for `target` that the SFT loop should train on."""
    fit_recs = train_recs.get(target, [])
    deploy_recs = eval_recs.get(target, [])
    if sft_subset == "union":
        return list(fit_recs) + list(deploy_recs)
    elif sft_subset == "deploy_only":
        return list(deploy_recs)
    elif sft_subset == "fit_only":
        return list(fit_recs)
    else:
        raise ValueError(f"Unknown --sft-subset: {sft_subset!r}")


def train_lquant_sft(cfg: LQuantConfig, sft_subset: str, batch_per_step: int) -> Dict:
    """SFT training loop that minimizes log p(password | prompt) on a chosen subset.

    Reuses train_lquant.py setup helpers + run_evaluation_suite for end-of-training
    evaluation. KL-to-base regularization is identical to meta-training.
    """
    from importance_sampling.policy_lm import PolicyLM, LoRASetup
    from transformers import AutoModelForCausalLM

    distributed = _distributed_enabled()
    world_size = _distributed_world_size()
    rank = _distributed_rank()
    local_rank = _distributed_local_rank()
    is_main = rank == 0

    if distributed and not torch.cuda.is_available():
        raise RuntimeError("Distributed SFT requires CUDA.")
    if distributed:
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://",
                                    device_id=torch.device(cfg.device))
    elif not cfg.device:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    outdir = Path(cfg.outdir)
    if is_main:
        outdir.mkdir(parents=True, exist_ok=True)
    _distributed_barrier()

    def log(*args, **kwargs):
        if is_main:
            kwargs.setdefault("flush", True)
            print(*args, **kwargs)

    if is_main:
        with open(outdir / "config.json", "w") as f:
            payload = asdict(cfg)
            payload["sft_subset"] = sft_subset
            payload["sft_batch_per_step"] = batch_per_step
            payload["loss_mode"] = "sft_direct_risk_min"
            json.dump(payload, f, indent=2, default=str)

    log(f"Loading model {cfg.model_id} with LoRA={cfg.use_lora} ...")
    lora_setup = LoRASetup(
        use_lora=cfg.use_lora, r=cfg.lora_r, alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules,
    )
    policy = PolicyLM(model_id=cfg.model_id, device=cfg.device, lora=lora_setup)
    raw_model = policy.model
    if cfg.gradient_checkpointing:
        # Same block as train_lquant: the SFT loop retains the whole
        # sft-batch's activation graphs before its single backward, which at
        # 8B exceeds 80GB without checkpointing. enable_input_require_grads is
        # required so the embedding output carries grad into the LoRA path.
        try:
            raw_model.enable_input_require_grads()
        except AttributeError:
            pass
        raw_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if raw_model.config.use_cache:
            raw_model.config.use_cache = False
        log("Gradient checkpointing enabled (use_reentrant=False, use_cache=False).")
    model = raw_model
    tokenizer = policy.tokenizer

    if cfg.prompt_bank_root:
        bank_seed = _effective_prompt_bank_seed(cfg)
        bank_info = resolve_prompt_bank(cfg.prompt_bank_root, bank_seed)
        log(f"Loading prompt bank seed {bank_info.bank_seed} from {bank_info.bank_dir} ...")
        if str(cfg.task_kind) == "token_union":
            pool = _load_prompt_pool_from_bank_token_union(bank_info, cfg=cfg, tokenizer=tokenizer)
        else:
            pool = _load_prompt_pool_from_bank(bank_info, cfg=cfg, tokenizer=tokenizer)
    else:
        log("Building prompt pool ...")
        pool = build_prompt_pool(cfg, raw_model, tokenizer)
    _validate_fit_deploy_family_shift(cfg)
    train_recs, eval_recs, test_recs = split_pool(pool, cfg)
    target_partition = _resolve_target_partition(list(pool["targets"]), cfg)
    train_targets = list(target_partition["train_targets"])
    evaluation_targets = list(target_partition["evaluation_targets"])
    train_recs_eval = _filter_target_records(train_recs, evaluation_targets)
    eval_recs_eval = _filter_target_records(eval_recs, evaluation_targets)
    test_recs_eval = _filter_target_records(test_recs, evaluation_targets)
    if is_main:
        with open(outdir / "target_partition.json", "w") as f:
            json.dump(target_partition, f, indent=2)

    # Local SFT prompt pool: list of (target, record) pairs across all train_targets
    local_targets = _shard_items(train_targets, rank=rank, world_size=world_size)
    sft_pool: List[Dict] = []
    for target in local_targets:
        for rec in _select_subset_records(
            train_recs=train_recs, eval_recs=eval_recs,
            target=target, sft_subset=sft_subset,
        ):
            # Each rec is a dict; system_prompt and event_token_id are already in it
            sft_pool.append(dict(rec))
    log(f"SFT pool size (local): {len(sft_pool)} prompts across {len(local_targets)} targets (subset={sft_subset!r})")
    if not sft_pool:
        raise RuntimeError("Empty SFT pool. Check --sft-subset and target partition.")

    # KL setup -- mirror train_lquant.py's loader pattern.
    base_model = None
    kl_train_prompts: List[str] = []
    kl_eval_prompts: List[str] = []
    kl_active = cfg.kl_coef > 0 or cfg.kl_eval_datasets
    if kl_active:
        log("Loading frozen base model for KL ...")
        base_dtype = next(raw_model.parameters()).dtype
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=base_dtype,
        ).to(cfg.device).eval()
        for p in base_model.parameters():
            p.requires_grad_(False)
        for ds_name in cfg.kl_train_datasets:
            if ds_name == "password":
                for target in train_targets:
                    for rec in train_recs.get(target, []) + eval_recs.get(target, []):
                        msgs = prepare_messages(
                            str(rec["prompt"]), system_prompt=rec.get("system_prompt"),
                        )
                        kl_train_prompts.append(_apply_chat_template_safe(
                            tokenizer, msgs, not cfg.disable_thinking,
                        ))
            else:
                kl_train_prompts.extend(load_kl_dataset(
                    ds_name, tokenizer, n_samples=cfg.kl_batch_size * 20,
                ))
        log(f"  KL train prompts: {len(kl_train_prompts)}")
        for ds_name in cfg.kl_eval_datasets:
            if ds_name == "password":
                for target in evaluation_targets:
                    for rec in test_recs.get(target, []):
                        msgs = prepare_messages(
                            str(rec["prompt"]), system_prompt=rec.get("system_prompt"),
                        )
                        kl_eval_prompts.append(_apply_chat_template_safe(
                            tokenizer, msgs, not cfg.disable_thinking,
                        ))
            else:
                kl_eval_prompts.extend(load_kl_dataset(
                    ds_name, tokenizer, n_samples=256,
                ))
        log(f"  KL eval prompts: {len(kl_eval_prompts)}")

    optimizer = torch.optim.Adam(
        [p for p in raw_model.parameters() if p.requires_grad], lr=cfg.lr,
    )

    # Baseline evaluation BEFORE any training
    if is_main:
        log("\nEvaluating baseline (pretrained, before SFT) ...")
        baseline = run_evaluation_suite(
            raw_model, tokenizer,
            train_recs_eval, eval_recs_eval, test_recs_eval, cfg,
            outdir=outdir, prefix="baseline_", label="baseline ",
            base_model=base_model, kl_eval_prompts=kl_eval_prompts,
        )

    # Training loop
    history: List[Dict] = []
    # Best-checkpoint selection on the SFT loss, mirroring the meta-training
    # runs' train-side best-checkpoint protocol so the comparison stays fair
    # when a cell reports "full run, best checkpoint" for our method. The
    # per-step batch loss is noisy (a random batch_per_step-record sample), so
    # the selection unit is the interval mean over each save_every window; the
    # held-out split is never consulted.
    best_interval_loss = float("inf")
    log(f"\nStarting SFT for {cfg.steps} steps (subset={sft_subset!r}, batch={batch_per_step}) ...")
    t_start = time.perf_counter()
    for step in range(cfg.steps):
        t_step = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        # Sample a batch of records from the SFT pool (with replacement)
        rng = _step_rng(cfg.seed, step, stream=51)
        batch_recs = [sft_pool[rng.randrange(len(sft_pool))] for _ in range(batch_per_step)]

        # Forward + backward
        log_probs = compute_password_logprobs_batched(
            model, tokenizer, batch_recs,
            enable_thinking=(not cfg.disable_thinking),
            device=cfg.device,
            micro_batch_size=cfg.micro_batch_size,
            require_grad=True,
            score_max_len=cfg.score_max_len,
            token_budget=cfg.score_token_budget,
        )
        # Direct risk minimization: minimize sum log_prob (push log_prob more negative).
        # This is gradient DESCENT on log p(password); equivalent to gradient ASCENT on
        # negative log-likelihood (= MAX_NLL = "untrain" the password completion).
        sft_loss = log_probs.mean()
        if not torch.isfinite(sft_loss):
            log(f"  step {step}: non-finite SFT loss, skipping backward")
            continue
        sft_loss.backward()

        # KL regularization (separate backward, same pattern as meta-training)
        kl_single_val = 0.0
        kl_trunc_val = 0.0
        if cfg.kl_coef > 0 and base_model is not None and kl_train_prompts:
            kl_rng = _step_rng(cfg.seed, step, stream=53)
            kl_batch = kl_rng.sample(
                kl_train_prompts, min(cfg.kl_batch_size, len(kl_train_prompts)),
            )
            kl_ids, kl_mask = _tokenize_for_kl(tokenizer, kl_batch, cfg.device)
            kl_1 = compute_kl_single_token(model, base_model, kl_ids, kl_mask)
            kl_loss = cfg.kl_coef * kl_1
            kl_single_val = float(kl_1.item())
            if cfg.kl_trunc_len > 0 and step % cfg.kl_trunc_every == 0:
                kl_t = compute_kl_truncated(
                    model, base_model, kl_ids, kl_mask, n_tokens=cfg.kl_trunc_len,
                )
                kl_loss = kl_loss + cfg.kl_coef * kl_t
                kl_trunc_val = float(kl_t.item())
            if torch.isfinite(kl_loss) and kl_loss.requires_grad:
                kl_loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
        optimizer.step()

        step_sec = time.perf_counter() - t_step
        # Record per-step stats
        mean_lp = float(log_probs.detach().mean().item())
        worst_lp_in_batch = float(log_probs.detach().max().item())
        history.append({
            "step": step,
            "loss": float(sft_loss.detach().item()),
            "mean_log_prob_in_batch": mean_lp,
            "worst_log_prob_in_batch": worst_lp_in_batch,
            "kl_single": kl_single_val,
            "kl_trunc": kl_trunc_val,
            "step_sec": step_sec,
        })
        if step % cfg.log_every == 0:
            extra = ""
            if cfg.kl_coef > 0:
                extra += f" | kl1={kl_single_val:.4f}"
                if kl_trunc_val > 0:
                    extra += f" klt={kl_trunc_val:.4f}"
            log(f"  step {step:4d} | loss={sft_loss.item():.4f} "
                f"| mean_lp={mean_lp:+.3f} | worst_lp_batch={worst_lp_in_batch:+.3f}"
                f"{extra} | {step_sec:.1f}s")

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0 and is_main:
            _write_history_json(outdir, history)
            # Mid-training LoRA snapshot: overwrite latest_snapshot/ each save.
            # If the run crashes mid-training (preempt, OOM, wall hit), the most
            # recent LoRA state is recoverable for offline eval. Snapshots are
            # NOT used for resume (no resume support in this script), only for
            # checkpoint-as-last-resort.
            snap_dir = outdir / "latest_snapshot"
            snap_dir.mkdir(parents=True, exist_ok=True)
            raw_model.save_pretrained(str(snap_dir))
            interval = history[-cfg.save_every:]
            interval_mean = sum(h["loss"] for h in interval) / len(interval)
            if interval_mean < best_interval_loss:
                best_interval_loss = interval_mean
                best_dir = outdir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                raw_model.save_pretrained(str(best_dir))
                with open(outdir / "best_step.json", "w") as f:
                    json.dump({
                        "step": step,
                        "interval_first_step": interval[0]["step"],
                        "interval_last_step": interval[-1]["step"],
                        "interval_mean_sft_loss": interval_mean,
                        "selection_metric": "interval_mean_sft_loss",
                        "note": "Selected on the SFT training loss only "
                                "(interval mean per save_every window); the "
                                "held-out split was not consulted. Mirrors "
                                "the meta-training runs' train-side "
                                "best-checkpoint protocol.",
                    }, f, indent=2)
                log(f"  [best] interval mean loss {interval_mean:.4f} at step {step}; saved best/")

    elapsed = time.perf_counter() - t_start
    log(f"\nSFT complete in {elapsed:.1f}s")

    # Save final LoRA + history
    if is_main:
        final_dir = outdir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        _write_history_json(outdir, history)

        log("\nEvaluating SFT-trained model on heldout ...")
        final_eval = run_evaluation_suite(
            raw_model, tokenizer,
            train_recs_eval, eval_recs_eval, test_recs_eval, cfg,
            outdir=outdir, prefix="", label="",
            base_model=base_model, kl_eval_prompts=kl_eval_prompts,
        )

        return {
            "history": history,
            "baseline_test_results": baseline["test_results"],
            "test_results": final_eval["test_results"],
            "elapsed_sec": elapsed,
        }
    return {"history": history, "elapsed_sec": elapsed}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Direct-risk-minimization SFT baseline for the LM password game."
    )
    _add_lquant_cli(parser)
    parser.add_argument(
        "--sft-subset", default="union",
        choices=["union", "deploy_only", "fit_only"],
        help="Which prompt subset to SFT on. union = train+eval; deploy_only = eval (cheating oracle); fit_only = train.",
    )
    parser.add_argument(
        "--sft-batch-per-step", type=int, default=32,
        help="Number of (prompt, password) pairs sampled per SFT step (with replacement, from the local target shard).",
    )
    args = parser.parse_args(argv)
    cfg = _build_cfg_from_args(args)
    train_lquant_sft(cfg, sft_subset=args.sft_subset, batch_per_step=args.sft_batch_per_step)


if __name__ == "__main__":
    main()
