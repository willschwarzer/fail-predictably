from __future__ import annotations

import os
import sys
import time


def _rank_label() -> str:
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if rank is None:
        return "rank=main"
    return f"rank={rank} local_rank={local_rank or '?'} world_size={world_size or '?'}"


def main() -> None:
    launch_started = time.perf_counter()
    print(
        f"Starting LQuant launcher | {_rank_label()} | argv={len(sys.argv) - 1}",
        flush=True,
    )
    import_started = time.perf_counter()
    from .train_lquant import main as train_main

    print(
        "Imported forecast_experiment.train_lquant | "
        f"elapsed={time.perf_counter() - import_started:.1f}s "
        f"(total={time.perf_counter() - launch_started:.1f}s)",
        flush=True,
    )
    train_main(sys.argv[1:])


if __name__ == "__main__":
    main()
