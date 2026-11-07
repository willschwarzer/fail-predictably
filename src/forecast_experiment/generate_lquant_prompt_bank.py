from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lquant_prompt_bank import write_prompt_bank
from .train_lquant import (
    DEFAULT_TEST_FAMILIES,
    DEFAULT_TRAIN_FAMILIES,
    LQuantConfig,
    _parse_family_prompt_count_map,
    build_prompt_pool,
)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_generation_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bank-seed", type=int, required=True)
    parser.add_argument("--password-slots", type=int, default=5)
    parser.add_argument("--password-policy", default="non_disclosure")
    parser.add_argument("--cheap-prompts-per-family-per-target", type=int, default=64)
    parser.add_argument("--cheap-prompts-per-family-map", nargs="*", default=None)
    parser.add_argument("--train-families", nargs="*", default=None)
    parser.add_argument("--test-families", nargs="*", default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--gcg", dest="gcg_enabled", action="store_true")
    parser.add_argument("--gcg-steps-spectrum", nargs="*", type=int, default=None)
    parser.add_argument("--gcg-prompts-per-level-per-target", type=int, default=16)
    parser.add_argument("--gcg-prompt-len", type=int, default=24)
    parser.add_argument("--gcg-top-k", type=int, default=8)
    parser.add_argument("--gcg-random-restarts", type=int, default=2)
    parser.add_argument("--gcg-train-max-steps", type=int, default=20)
    parser.add_argument("--gcg-decay-alpha", type=float, default=2.0)
    return parser


def config_from_args(args: argparse.Namespace) -> LQuantConfig:
    cheap_prompts_per_family_map = _parse_family_prompt_count_map(args.cheap_prompts_per_family_map)
    return LQuantConfig(
        model_id=args.model,
        device=args.device,
        password_slots=args.password_slots,
        password_policy=args.password_policy,
        cheap_prompts_per_family_per_target=args.cheap_prompts_per_family_per_target,
        cheap_prompts_per_family_map=cheap_prompts_per_family_map,
        train_families=args.train_families or list(DEFAULT_TRAIN_FAMILIES),
        test_families=args.test_families or list(DEFAULT_TEST_FAMILIES),
        gcg_enabled=args.gcg_enabled,
        gcg_steps_spectrum=args.gcg_steps_spectrum or [5, 10, 20, 40, 80],
        gcg_prompts_per_level_per_target=args.gcg_prompts_per_level_per_target,
        gcg_prompt_len=args.gcg_prompt_len,
        gcg_top_k=args.gcg_top_k,
        gcg_random_restarts=args.gcg_random_restarts,
        gcg_train_max_steps=args.gcg_train_max_steps,
        gcg_decay_alpha=args.gcg_decay_alpha,
        disable_thinking=args.disable_thinking,
        seed=args.bank_seed,
    )


def load_generation_artifacts(cfg: LQuantConfig):
    print(f"Loading tokenizer for {cfg.model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = None
    if cfg.gcg_enabled:
        print(f"Loading model for GCG generation on device={cfg.device} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            device_map="auto" if cfg.device == "cuda" else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if cfg.device != "cuda":
            model.to(cfg.device)
    return tokenizer, model


def generation_config_payload(*, bank_seed: int, cfg: LQuantConfig) -> dict[str, Any]:
    return {
        "bank_seed": int(bank_seed),
        "generation_cfg": {
            "model_id": cfg.model_id,
            "password_slots": cfg.password_slots,
            "password_policy": cfg.password_policy,
            "cheap_prompts_per_family_per_target": cfg.cheap_prompts_per_family_per_target,
            "cheap_prompts_per_family_map": cfg.cheap_prompts_per_family_map,
            "train_families": cfg.train_families,
            "test_families": cfg.test_families,
            "gcg_enabled": cfg.gcg_enabled,
            "gcg_steps_spectrum": cfg.gcg_steps_spectrum,
            "gcg_prompts_per_level_per_target": cfg.gcg_prompts_per_level_per_target,
            "gcg_prompt_len": cfg.gcg_prompt_len,
            "gcg_top_k": cfg.gcg_top_k,
            "gcg_random_restarts": cfg.gcg_random_restarts,
            "gcg_train_max_steps": cfg.gcg_train_max_steps,
            "gcg_decay_alpha": cfg.gcg_decay_alpha,
            "disable_thinking": cfg.disable_thinking,
        },
    }


def write_generation_config(outdir: Path, *, bank_seed: int, cfg: LQuantConfig) -> None:
    with open(outdir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(generation_config_payload(bank_seed=bank_seed, cfg=cfg), f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = build_generation_parser(description="Generate a compact LQuant prompt bank.")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = config_from_args(args)
    tokenizer, model = load_generation_artifacts(cfg)

    print(f"Building prompt pool for bank seed {args.bank_seed} ...", flush=True)
    pool = build_prompt_pool(cfg, model, tokenizer, generation_seed=args.bank_seed)
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
        "Prompt bank complete | "
        f"seed={bank_info.bank_seed} rows={bank_info.metadata['row_count']} "
        f"targets={len(bank_info.metadata['targets'])} families={len(bank_info.metadata['families'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
