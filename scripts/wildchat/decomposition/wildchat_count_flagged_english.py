"""Count the Cell B slice exactly (English + flagged-or-toxic first-user-turn).

Walks all WildChat shards and prints the counts so we can size the Cell B run
without waiting for the GPU job to finish.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from datasets import Dataset as HFDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wildchat-dir",
        type=Path,
        default=Path(os.environ.get("WILDCHAT_DATASET", "data/wildchat")),
        help="Directory holding the WildChat-1M-Full arrow shards, searched recursively "
        "(default: $WILDCHAT_DATASET, else data/wildchat).",
    )
    args = ap.parse_args()
    shards = sorted(args.wildchat_dir.rglob("*.arrow"))
    n_total = 0
    n_eng = 0
    n_flagged_or_toxic = 0
    n_eng_flagged = 0
    n_eng_toxic_only = 0
    n_eng_flagged_only = 0
    t0 = time.time()
    for shard_idx, shard in enumerate(shards):
        ds = HFDataset.from_file(str(shard))
        for i in range(len(ds)):
            n_total += 1
            r = ds[i]
            conv = r.get("conversation", [])
            if not conv or conv[0].get("role") != "user":
                continue
            is_eng = r.get("language") == "English"
            if is_eng:
                n_eng += 1
            is_toxic = r.get("toxic") is True
            omod = r.get("openai_moderation", [])
            is_flagged = any(isinstance(e, dict) and e.get("flagged") for e in omod)
            if is_toxic or is_flagged:
                n_flagged_or_toxic += 1
            if is_eng and (is_toxic or is_flagged):
                n_eng_flagged += 1
                if is_toxic and not is_flagged:
                    n_eng_toxic_only += 1
                if is_flagged and not is_toxic:
                    n_eng_flagged_only += 1
        print(
            f"[shard {shard_idx + 1}/{len(shards)}] cum n_total={n_total} "
            f"eng={n_eng} flagged_or_toxic={n_flagged_or_toxic} "
            f"eng_AND_flagged_or_toxic={n_eng_flagged} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    print()
    print(f"n_total                       = {n_total}")
    print(f"english_first_user_turn       = {n_eng}")
    print(f"flagged_or_toxic              = {n_flagged_or_toxic}")
    print(f"english AND flagged_or_toxic  = {n_eng_flagged}")
    print(f"  (english AND toxic-only)    = {n_eng_toxic_only}")
    print(f"  (english AND flagged-only)  = {n_eng_flagged_only}")


if __name__ == "__main__":
    main()
