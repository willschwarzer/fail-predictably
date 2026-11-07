from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .rlquant_gridworld import (
    FamilySlice,
    GridworldGeneratorConfig,
    GridworldTaskSpec,
    sample_unique_task_specs,
)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _severity_in_range(severity: float, lower: float, upper: float) -> bool:
    eps = 1e-9
    return float(lower) - eps <= float(severity) <= float(upper) + eps


def _component_name_for_spec(spec: GridworldTaskSpec, components: list[dict[str, object]]) -> str:
    matches = [
        str(component["name"])
        for component in components
        if _severity_in_range(float(spec.severity), float(component["severity_min"]), float(component["severity_max"]))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one generation component for severity={spec.severity}, found {matches}"
        )
    return matches[0]


def _bank_seed_from_path(out_path: Path) -> int:
    name = out_path.parent.name
    prefix = "bank_seed_"
    if not name.startswith(prefix):
        raise RuntimeError(f"Could not infer bank seed from parent directory name {name}")
    return int(name[len(prefix) :])


def _refill_shortfall(
    *,
    merged: list[GridworldTaskSpec],
    seen: set[str],
    plan: dict[str, object],
    bank_seed: int,
    out_path: Path,
) -> dict[str, object]:
    components = list(plan["components"])
    component_counts = {str(component["name"]): 0 for component in components}
    for spec in merged:
        component_counts[_component_name_for_spec(spec, components)] += 1

    generator_cfg = GridworldGeneratorConfig(
        rows=int(plan["generator_config"]["rows"]),
        cols=int(plan["generator_config"]["cols"]),
        horizon=int(plan["generator_config"]["horizon"]),
        gamma=float(plan["generator_config"]["gamma"]),
        step_cost=float(plan["generator_config"]["step_cost"]),
        goal_reward=float(plan["generator_config"]["goal_reward"]),
        catastrophe_penalty=float(plan["generator_config"]["catastrophe_penalty"]),
        min_start_goal_distance=int(plan["generator_config"]["min_start_goal_distance"]),
        max_generation_attempts_per_task=int(plan["generator_config"]["max_generation_attempts_per_task"]),
    )
    family_id = str(plan["family_id"])
    refill_specs: list[GridworldTaskSpec] = []
    shortfalls_before: dict[str, int] = {}
    generated_counts: dict[str, int] = {}
    for component_idx, component in enumerate(components):
        name = str(component["name"])
        target_total = int(component["target_total"])
        shortfall = max(0, target_total - int(component_counts.get(name, 0)))
        shortfalls_before[name] = shortfall
        if shortfall <= 0:
            generated_counts[name] = 0
            continue
        refill_seed = bank_seed * 1_000_000 + (component_idx + 1) * 104_729 + len(merged) + len(refill_specs)
        generated = sample_unique_task_specs(
            split_name=f"bank_refill_{bank_seed}_{name}",
            slices=[
                FamilySlice(
                    family_id=family_id,
                    count=shortfall,
                    severity_min=float(component["severity_min"]),
                    severity_max=float(component["severity_max"]),
                )
            ],
            cfg=generator_cfg,
            seed=refill_seed,
            globally_seen_hashes=seen,
        )
        refill_specs.extend(generated)
        component_counts[name] += len(generated)
        generated_counts[name] = len(generated)

    refill_dir = out_path.parent / "refills"
    refill_dir.mkdir(parents=True, exist_ok=True)
    refill_path = refill_dir / "refill_specs.jsonl"
    with refill_path.open("w", encoding="utf-8") as f:
        for spec in refill_specs:
            f.write(json.dumps(spec.to_json(), sort_keys=True) + "\n")
    merged.extend(refill_specs)
    return {
        "used": bool(refill_specs),
        "path": str(refill_path),
        "generated_count": int(len(refill_specs)),
        "component_shortfall_before": shortfalls_before,
        "component_generated_counts": generated_counts,
        "component_final_counts": component_counts,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate RLQuant task-bank shards.")
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-total", type=int, default=0)
    parser.add_argument("--generation-plan", default="")
    args = parser.parse_args(argv)

    shard_paths = sorted(Path(p) for p in glob.glob(args.shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched {args.shard_glob}")

    seen: set[str] = set()
    merged: list[GridworldTaskSpec] = []
    duplicate_count = 0
    family_counts: dict[str, int] = {}
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                spec = GridworldTaskSpec.from_json(payload)
                if spec.canonical_hash in seen:
                    duplicate_count += 1
                    continue
                seen.add(spec.canonical_hash)
                merged.append(spec)
                family_counts[spec.family_id] = family_counts.get(spec.family_id, 0) + 1
                if args.target_total > 0 and len(merged) >= args.target_total:
                    break
        if args.target_total > 0 and len(merged) >= args.target_total:
            break

    refill_summary: dict[str, object] = {
        "used": False,
        "generated_count": 0,
    }
    if args.generation_plan and args.target_total > 0 and len(merged) < args.target_total:
        plan = _read_json(Path(args.generation_plan))
        refill_summary = _refill_shortfall(
            merged=merged,
            seen=seen,
            plan=plan,
            bank_seed=_bank_seed_from_path(Path(args.out)),
            out_path=Path(args.out),
        )
        family_counts = {}
        for spec in merged:
            family_counts[spec.family_id] = family_counts.get(spec.family_id, 0) + 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for spec in merged:
            f.write(json.dumps(spec.to_json(), sort_keys=True) + "\n")

    metadata = {
        "out": str(out_path),
        "shard_glob": args.shard_glob,
        "shard_count": len(shard_paths),
        "unique_count": len(merged),
        "target_total": int(args.target_total),
        "duplicate_count": int(duplicate_count),
        "family_counts": family_counts,
        "generation_plan": args.generation_plan,
        "refill": refill_summary,
    }
    with out_path.with_suffix(out_path.suffix + ".meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    if args.target_total > 0 and len(merged) < args.target_total:
        raise RuntimeError(
            f"Merged bank landed short: {len(merged)} unique < target_total={args.target_total}"
        )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
