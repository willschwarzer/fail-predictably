from __future__ import annotations

from pathlib import Path

from .generate_lquant_prompt_bank import (
    build_generation_parser,
    config_from_args,
    load_generation_artifacts,
    write_generation_config,
)
from .lquant_prompt_bank import write_prompt_bank
from .password_game import get_password_slots
from .train_lquant import build_prompt_pool


def main(argv: list[str] | None = None) -> None:
    parser = build_generation_parser(description="Generate a single-target shard for a compact LQuant prompt bank.")
    parser.add_argument("--target-index", type=int, required=True)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = config_from_args(args)
    all_targets = get_password_slots(cfg.password_slots)
    if args.target_index < 0 or args.target_index >= len(all_targets):
        raise ValueError(
            f"target-index {args.target_index} is out of range for {len(all_targets)} targets"
        )
    target_name = all_targets[args.target_index]
    print(
        f"Building prompt bank shard | bank_seed={args.bank_seed} "
        f"target_index={args.target_index} target={target_name}",
        flush=True,
    )

    tokenizer, model = load_generation_artifacts(cfg)
    pool = build_prompt_pool(
        cfg,
        model,
        tokenizer,
        generation_seed=args.bank_seed,
        targets_subset=[target_name],
    )
    bank_info = write_prompt_bank(
        outdir,
        bank_seed=args.bank_seed,
        generation_seed=args.bank_seed,
        model_id=cfg.model_id,
        password_slots=cfg.password_slots,
        password_policy=cfg.password_policy,
        cheap_prompts_per_family_per_target=cfg.cheap_prompts_per_family_per_target,
        cheap_prompts_per_family_map=cfg.cheap_prompts_per_family_map,
        train_families=cfg.train_families,
        test_families=cfg.test_families,
        gcg_enabled=cfg.gcg_enabled,
        gcg_steps_spectrum=cfg.gcg_steps_spectrum,
        gcg_prompts_per_level_per_target=cfg.gcg_prompts_per_level_per_target,
        gcg_prompt_len=cfg.gcg_prompt_len,
        gcg_top_k=cfg.gcg_top_k,
        gcg_random_restarts=cfg.gcg_random_restarts,
        gcg_train_max_steps=cfg.gcg_train_max_steps,
        gcg_decay_alpha=cfg.gcg_decay_alpha,
        disable_thinking=cfg.disable_thinking,
        pool=pool,
    )
    write_generation_config(outdir, bank_seed=args.bank_seed, cfg=cfg)
    print(
        "Prompt bank shard complete | "
        f"seed={bank_info.bank_seed} target={target_name} "
        f"rows={bank_info.metadata['row_count']} families={len(bank_info.metadata['families'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
