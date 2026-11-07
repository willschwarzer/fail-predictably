from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rlquant_gridworld import FamilySlice, GridworldGeneratorConfig, sample_unique_task_specs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate one lightweight RLQuant task-bank shard.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-name", default="bank_shard")
    parser.add_argument("--severity-min", type=float, required=True)
    parser.add_argument("--severity-max", type=float, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--step-cost", type=float, default=-0.01)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--catastrophe-penalty", type=float, default=-1.0)
    parser.add_argument("--min-start-goal-distance", type=int, default=5)
    parser.add_argument("--max-generation-attempts-per-task", type=int, default=200)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = GridworldGeneratorConfig(
        rows=args.rows,
        cols=args.cols,
        horizon=args.horizon,
        gamma=args.gamma,
        step_cost=args.step_cost,
        goal_reward=args.goal_reward,
        catastrophe_penalty=args.catastrophe_penalty,
        min_start_goal_distance=args.min_start_goal_distance,
        max_generation_attempts_per_task=args.max_generation_attempts_per_task,
    )
    specs = sample_unique_task_specs(
        split_name=args.split_name,
        slices=[
            FamilySlice(
                family_id=args.family_id,
                count=args.count,
                severity_min=args.severity_min,
                severity_max=args.severity_max,
            )
        ],
        cfg=cfg,
        seed=args.seed,
        globally_seen_hashes=None,
    )
    with out_path.open("w", encoding="utf-8") as f:
        for spec in specs:
            f.write(json.dumps(spec.to_json(), sort_keys=True) + "\n")
    metadata = {
        "out": str(out_path),
        "family_id": args.family_id,
        "count": len(specs),
        "requested_count": int(args.count),
        "seed": int(args.seed),
        "severity_min": float(args.severity_min),
        "severity_max": float(args.severity_max),
        "rows": int(args.rows),
        "cols": int(args.cols),
        "horizon": int(args.horizon),
        "gamma": float(args.gamma),
        "step_cost": float(args.step_cost),
        "goal_reward": float(args.goal_reward),
        "catastrophe_penalty": float(args.catastrophe_penalty),
        "min_start_goal_distance": int(args.min_start_goal_distance),
    }
    with out_path.with_suffix(out_path.suffix + ".meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
