"""Disk-backed cache for the LM baseline test evaluation.

The baseline eval scores the BASE model (LoRA at PEFT init, B=0 → no LoRA
contribution) on heldout slots' deploy pools and produces the FRLMB OLS
forecast artifacts. It's a 15-25 minute cost on a small GPU and is
deterministic given a hashable subset of LQuantConfig: model_id, prompt
bank identity, target partition seed, OLS settings, KL eval inputs,
disable_thinking. Multi-seed sweeps that vary only `--seed` re-pay this
cost N times unnecessarily.

This module provides a content-addressed cache: a SHA256 hash over the
relevant config fields keys a directory containing the baseline_*.json/npz
artifacts. On hit, files are copied into the run's outdir; on miss, the
caller computes as usual and saves to the cache for future hits.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


_BASELINE_FILES = (
    "baseline_test_results.json",
    "baseline_test_pool_diagnostic.json",
    "baseline_kl_eval_results.json",
    "baseline_test_forecast_traces.npz",
    "baseline_test_forecast_traces_metadata.json",
)


def _baseline_cache_inputs(cfg: Any) -> dict[str, Any]:
    """Collect the LQuantConfig fields that uniquely determine baseline eval output.

    Excluded: meta-training hyperparameters (do not affect baseline) and
    micro_batch_size (numerical noise only, would cause spurious cache misses).
    """
    def _norm_list(x):
        return sorted(list(x)) if x else []

    def _norm_map(d):
        return sorted([list(item) for item in d.items()]) if d else []

    return {
        "model_id": str(getattr(cfg, "model_id", "")),
        "disable_thinking": bool(getattr(cfg, "disable_thinking", False)),
        "task_kind": str(getattr(cfg, "task_kind", "password")),
        "score_max_len": int(getattr(cfg, "score_max_len", 0)),
        "kl_tokenize_max_len": int(getattr(cfg, "kl_tokenize_max_len", 512)),
        "kl_truncation_side": str(getattr(cfg, "kl_truncation_side", "right")),
        "prompt_bank_root": str(getattr(cfg, "prompt_bank_root", "")),
        "prompt_bank_seed": int(getattr(cfg, "prompt_bank_seed", -1)),
        "train_families": _norm_list(getattr(cfg, "train_families", [])),
        "test_families": _norm_list(getattr(cfg, "test_families", [])),
        "cheap_prompts_per_family_per_target": int(
            getattr(cfg, "cheap_prompts_per_family_per_target", 0)
        ),
        "cheap_prompts_per_family_map": _norm_map(
            getattr(cfg, "cheap_prompts_per_family_map", {}) or {}
        ),
        "shared_family_test_fraction": float(
            getattr(cfg, "shared_family_test_fraction", 0.0)
        ),
        "fit_fraction": float(getattr(cfg, "fit_fraction", 0.0)),
        "split_mode": str(getattr(cfg, "split_mode", "")),
        "password_slots": int(getattr(cfg, "password_slots", 0)),
        "password_policy": str(getattr(cfg, "password_policy", "")),
        "seed": int(getattr(cfg, "seed", 0)),
        "train_target_count": int(getattr(cfg, "train_target_count", 0)),
        "heldout_target_count": int(getattr(cfg, "heldout_target_count", 0)),
        "ols_top_k": int(getattr(cfg, "ols_top_k", 0)),
        "plotting_position": str(getattr(cfg, "plotting_position", "")),
        "extrapolated_weighting": str(getattr(cfg, "extrapolated_weighting", "")),
        "extrapolated_loss_space": str(getattr(cfg, "extrapolated_loss_space", "")),
        "kl_eval_datasets": _norm_list(getattr(cfg, "kl_eval_datasets", [])),
        "kl_batch_size": int(getattr(cfg, "kl_batch_size", 0)),
    }


def _baseline_cache_extras(cfg: Any) -> dict:
    """Key fields added ONLY when non-default.

    Adding a field unconditionally would change the hash of every existing
    config and orphan the whole cache, so opt-in scoring-semantics flags enter
    the key only when set. `token_union_blocklist_prompt` rewrites every scored
    prompt (system-prompt blocklist), so its baselines must never collide with
    the canonical ones.
    """
    extras = {}
    if bool(getattr(cfg, "token_union_blocklist_prompt", False)):
        extras["token_union_blocklist_prompt"] = True
    # Fit/deploy family shift changes the held-out fit/deploy partition, so its
    # baselines must never collide with canonical IID ones.
    fit_families = sorted(getattr(cfg, "fit_families", []) or [])
    deploy_families = sorted(getattr(cfg, "deploy_families", []) or [])
    if fit_families or deploy_families:
        extras["fit_families"] = fit_families
        extras["deploy_families"] = deploy_families
        extras["shift_fit_size"] = int(getattr(cfg, "shift_fit_size", 0))
        extras["shift_deploy_size"] = int(getattr(cfg, "shift_deploy_size", 0))
    return extras


def compute_baseline_cache_key(cfg: Any) -> str:
    """Hash-stable 16-char hex key over the cache-relevant subset of cfg."""
    inputs = _baseline_cache_inputs(cfg)
    inputs.update(_baseline_cache_extras(cfg))
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _resolve_cache_root(cache_root: str | Path | None) -> Path | None:
    if not cache_root:
        return None
    return Path(cache_root)


def cache_entry_dir(cache_root: str | Path, key: str) -> Path:
    return Path(cache_root) / key


def try_load_from_cache(
    cache_root: str | Path | None,
    cfg: Any,
    outdir: Path,
    *,
    log=None,
) -> bool:
    """Look up cached baseline artifacts; on hit, copy to outdir and return True.

    Returns False on miss or when caching is disabled. On hit, the caller can
    use the existing `_load_saved_eval_suite(outdir, "baseline_")` helper to
    materialize the dict.
    """
    root = _resolve_cache_root(cache_root)
    if root is None:
        return False
    key = compute_baseline_cache_key(cfg)
    entry = cache_entry_dir(root, key)
    test_results = entry / "baseline_test_results.json"
    if not test_results.exists():
        if log is not None:
            log(f"[baseline-cache] MISS key={key} (no entry at {entry})", flush=True)
        return False

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fname in _BASELINE_FILES:
        src = entry / fname
        if not src.exists():
            continue
        dst = outdir / fname
        if dst.resolve() == src.resolve():
            continue
        shutil.copy2(src, dst)
        copied += 1
    if log is not None:
        log(f"[baseline-cache] HIT key={key} copied={copied} from={entry}", flush=True)
    return True


def save_to_cache(
    cache_root: str | Path | None,
    cfg: Any,
    outdir: Path,
    *,
    log=None,
) -> None:
    """Copy baseline_* artifacts from outdir into the cache. No-op if disabled."""
    root = _resolve_cache_root(cache_root)
    if root is None:
        return
    key = compute_baseline_cache_key(cfg)
    entry = cache_entry_dir(root, key)
    # Each writer gets its own staging directory. A shared `<key>.tmp` is a race:
    # a fleet launched together misses the cache together, and the second writer's
    # rmtree deletes the first writer's half-copied files mid-copy, killing both
    # jobs (seen 2026-07-30 with the learning-dynamics fleet, two arms lost).
    entry_tmp = entry.with_name(f"{entry.name}.tmp.{os.getpid()}-{uuid.uuid4().hex[:8]}")

    outdir = Path(outdir)
    src_files = [outdir / f for f in _BASELINE_FILES if (outdir / f).exists()]
    if not src_files:
        if log is not None:
            log(
                f"[baseline-cache] WARN: nothing to save (no baseline_* files in {outdir})",
                flush=True,
            )
        return

    entry_tmp.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        shutil.copy2(src, entry_tmp / src.name)

    manifest = {
        "key": key,
        "inputs": _baseline_cache_inputs(cfg),
        "files": sorted(p.name for p in src_files),
    }
    (entry_tmp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
    )

    if entry.exists():
        # Concurrent writer beat us; prefer the existing entry, drop the tmp.
        shutil.rmtree(entry_tmp, ignore_errors=True)
        if log is not None:
            log(
                f"[baseline-cache] race: another writer populated {entry}; "
                f"discarded tmp",
                flush=True,
            )
        return
    try:
        entry_tmp.rename(entry)
    except OSError:
        # Another writer won between the check and the rename. Its entry is as
        # good as ours (the key is content-addressed), so keep theirs.
        shutil.rmtree(entry_tmp, ignore_errors=True)
        if log is not None:
            log(f"[baseline-cache] race on rename to {entry}; kept the winner", flush=True)
        return
    if log is not None:
        log(
            f"[baseline-cache] SAVED key={key} files={len(src_files)} to={entry}",
            flush=True,
        )
