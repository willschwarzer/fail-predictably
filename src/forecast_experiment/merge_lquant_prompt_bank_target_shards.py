from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .generate_lquant_prompt_bank import write_generation_config
from .lquant_prompt_bank import (
    PROMPT_BANK_METADATA_FILENAME,
    PROMPT_BANK_ROWS_FILENAME,
    write_prompt_bank,
)
from .password_game import get_password_slots
from .train_lquant import LQuantConfig


_CONSISTENT_METADATA_KEYS = (
    "bank_seed",
    "generation_seed",
    "model_id",
    "password_slots",
    "password_policy",
    "cheap_prompts_per_family_per_target",
    "cheap_prompts_per_family_map",
    "train_families",
    "test_families",
    "gcg_enabled",
    "gcg_steps_spectrum",
    "gcg_prompts_per_level_per_target",
    "gcg_prompt_len",
    "gcg_top_k",
    "gcg_random_restarts",
    "gcg_train_max_steps",
    "gcg_decay_alpha",
    "disable_thinking",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_consistent(reference: dict[str, Any], candidate: dict[str, Any], shard_dir: Path) -> None:
    for key in _CONSISTENT_METADATA_KEYS:
        if reference.get(key) != candidate.get(key):
            raise RuntimeError(
                f"Inconsistent shard metadata for key {key!r} in {shard_dir}: "
                f"{candidate.get(key)!r} != {reference.get(key)!r}"
            )


def _cfg_from_metadata(meta: dict[str, Any]) -> LQuantConfig:
    return LQuantConfig(
        model_id=str(meta["model_id"]),
        password_slots=int(meta["password_slots"]),
        password_policy=str(meta["password_policy"]),
        cheap_prompts_per_family_per_target=int(meta["cheap_prompts_per_family_per_target"]),
        cheap_prompts_per_family_map={
            str(key): int(value)
            for key, value in dict(meta.get("cheap_prompts_per_family_map", {})).items()
        },
        train_families=[str(x) for x in meta["train_families"]],
        test_families=[str(x) for x in meta["test_families"]],
        gcg_enabled=bool(meta["gcg_enabled"]),
        gcg_steps_spectrum=[int(x) for x in meta["gcg_steps_spectrum"]],
        gcg_prompts_per_level_per_target=int(meta["gcg_prompts_per_level_per_target"]),
        gcg_prompt_len=int(meta["gcg_prompt_len"]),
        gcg_top_k=int(meta["gcg_top_k"]),
        gcg_random_restarts=int(meta["gcg_random_restarts"]),
        gcg_train_max_steps=int(meta["gcg_train_max_steps"]),
        gcg_decay_alpha=float(meta["gcg_decay_alpha"]),
        disable_thinking=bool(meta["disable_thinking"]),
        seed=int(meta["bank_seed"]),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge single-target LQuant prompt-bank shards.")
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    shard_root = Path(args.shard_root)
    shard_dirs = sorted(
        path.parent
        for path in shard_root.glob(f"*/{PROMPT_BANK_METADATA_FILENAME}")
        if path.parent.is_dir()
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No prompt bank shards found under {shard_root}")

    first_meta = _read_json(shard_dirs[0] / PROMPT_BANK_METADATA_FILENAME)
    cfg = _cfg_from_metadata(first_meta)
    expected_targets = list(get_password_slots(cfg.password_slots))

    pool_by_target: dict[str, dict[str, list[dict[str, Any]]]] = {}
    slot_to_password: dict[str, str] = {}
    family_support_by_target: dict[str, dict[str, Any]] = {}
    seen_targets: set[str] = set()

    for shard_dir in shard_dirs:
        meta = _read_json(shard_dir / PROMPT_BANK_METADATA_FILENAME)
        _assert_consistent(first_meta, meta, shard_dir)
        shard_targets = [str(target) for target in meta["targets"]]
        shard_families = [str(family_id) for family_id in meta["families"]]
        rows_path = shard_dir / PROMPT_BANK_ROWS_FILENAME
        if not rows_path.exists():
            raise FileNotFoundError(f"Missing shard rows file: {rows_path}")
        if len(shard_targets) != 1:
            raise RuntimeError(
                f"Expected exactly one target per shard, found {len(shard_targets)} in {shard_dir}"
            )
        target_name = shard_targets[0]
        if target_name in seen_targets:
            raise RuntimeError(f"Duplicate target shard encountered for {target_name}")
        seen_targets.add(target_name)

        slot_to_password.update({str(k): str(v) for k, v in dict(meta["slot_to_password"]).items()})
        family_support_by_target.update(
            {
                str(target): dict(fam_map)
                for target, fam_map in dict(meta.get("family_support_by_target", {})).items()
            }
        )

        target_pool: dict[str, list[dict[str, Any]]] = {}
        with rows_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row_target = shard_targets[int(row["t"])]
                family_id = shard_families[int(row["f"])]
                if row_target != target_name:
                    raise RuntimeError(
                        f"Shard {shard_dir} contains unexpected target {row_target} "
                        f"(expected only {target_name})"
                    )
                target_pool.setdefault(family_id, []).append({"prompt": str(row["p"])})
        pool_by_target[target_name] = target_pool

    missing_targets = [target for target in expected_targets if target not in seen_targets]
    unexpected_targets = sorted(seen_targets.difference(expected_targets))
    if missing_targets or unexpected_targets:
        raise RuntimeError(
            "Shard coverage mismatch | "
            f"missing={missing_targets[:5]}{'...' if len(missing_targets) > 5 else ''} "
            f"unexpected={unexpected_targets[:5]}{'...' if len(unexpected_targets) > 5 else ''}"
        )

    ordered_targets = [target for target in expected_targets if target in pool_by_target]
    pool = {
        "targets": ordered_targets,
        "slot_to_password": {target: slot_to_password[target] for target in ordered_targets},
        "pool_by_target": {target: pool_by_target[target] for target in ordered_targets},
        "family_support_by_target": {
            target: family_support_by_target.get(target, {}) for target in ordered_targets
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bank_info = write_prompt_bank(
        outdir,
        bank_seed=int(first_meta["bank_seed"]),
        generation_seed=int(first_meta["generation_seed"]),
        model_id=str(first_meta["model_id"]),
        password_slots=int(first_meta["password_slots"]),
        password_policy=str(first_meta["password_policy"]),
        cheap_prompts_per_family_per_target=int(first_meta["cheap_prompts_per_family_per_target"]),
        cheap_prompts_per_family_map={
            str(key): int(value)
            for key, value in dict(first_meta.get("cheap_prompts_per_family_map", {})).items()
        },
        train_families=[str(x) for x in first_meta["train_families"]],
        test_families=[str(x) for x in first_meta["test_families"]],
        gcg_enabled=bool(first_meta["gcg_enabled"]),
        gcg_steps_spectrum=[int(x) for x in first_meta["gcg_steps_spectrum"]],
        gcg_prompts_per_level_per_target=int(first_meta["gcg_prompts_per_level_per_target"]),
        gcg_prompt_len=int(first_meta["gcg_prompt_len"]),
        gcg_top_k=int(first_meta["gcg_top_k"]),
        gcg_random_restarts=int(first_meta["gcg_random_restarts"]),
        gcg_train_max_steps=int(first_meta["gcg_train_max_steps"]),
        gcg_decay_alpha=float(first_meta["gcg_decay_alpha"]),
        disable_thinking=bool(first_meta["disable_thinking"]),
        pool=pool,
    )
    write_generation_config(outdir, bank_seed=int(first_meta["bank_seed"]), cfg=cfg)

    merge_summary = {
        "shard_root": str(shard_root),
        "shard_count": len(shard_dirs),
        "target_count": len(ordered_targets),
        "row_count": int(bank_info.metadata["row_count"]),
        "bank_dir": str(bank_info.bank_dir),
    }
    with (outdir / "merge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(merge_summary, f, indent=2, sort_keys=True)
    print(json.dumps(merge_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
