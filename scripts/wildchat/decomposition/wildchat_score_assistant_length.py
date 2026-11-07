"""Cell B (v5) scorer: assistant-turn length on WildChat.

For each English first-user-turn / first-assistant-turn pair, record:
  - asst_char_len: number of characters in the assistant turn.
  - asst_word_count: whitespace-tokenized word count.

Pure CPU. Length distributions in real chat data are reliably heavy-tailed
(Pareto-like over a wide range), so this is a natural non-probability score for
testing whether vanilla FRLMB-OLS-Gumbel-tail extrapolation under-predicts when
the underlying distribution is heavier than Gumbel.

Output: wildchat_assistant_length_scores.csv with one row per kept prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

from datasets import Dataset as HFDataset


def find_shards(arrow_dir: Path) -> list[Path]:
    return sorted(arrow_dir.rglob("*.arrow"))


def is_flagged_row(row: dict) -> bool:
    if row.get("toxic") is True:
        return True
    omod = row.get("openai_moderation", [])
    return any(isinstance(e, dict) and e.get("flagged") for e in omod)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wildchat-dir",
        type=Path,
        default=Path(os.environ.get("WILDCHAT_DATASET", "data/wildchat")),
        help="Directory holding the WildChat-1M-Full arrow shards, searched recursively "
        "(default: $WILDCHAT_DATASET, else data/wildchat).",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--language-source", choices=("top_level", "first_turn"), default="top_level")
    ap.add_argument("--language", default="English")
    ap.add_argument("--require-flagged", action="store_true", default=False)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--subsample-fraction", type=float, default=1.0)
    ap.add_argument("--subsample-seed", type=int, default=11)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "wildchat_assistant_length_scores.csv"
    manifest_path = args.out_dir / "manifest.json"

    shards = find_shards(args.wildchat_dir)
    if not shards:
        raise FileNotFoundError(f"No arrow shards under {args.wildchat_dir}")
    print(f"[load] {len(shards)} shards under {args.wildchat_dir}", flush=True)

    rng = random.Random(args.subsample_seed)
    do_subsample = float(args.subsample_fraction) < 1.0

    n_seen = 0
    n_kept = 0
    n_skipped_no_conv = 0
    n_skipped_role = 0
    n_skipped_lang = 0
    n_skipped_flag = 0
    n_skipped_subsample = 0
    n_skipped_empty = 0

    t_start = time.time()
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "shard_idx",
                "row_idx",
                "language",
                "asst_char_len",
                "asst_word_count",
                "conversation_hash",
            ]
        )
        for shard_idx, shard in enumerate(shards):
            ds = HFDataset.from_file(str(shard))
            n_shard_kept = 0
            for i in range(len(ds)):
                if args.limit_rows and n_seen >= args.limit_rows:
                    break
                n_seen += 1
                r = ds[i]
                conv = r.get("conversation", [])
                if len(conv) < 2:
                    n_skipped_no_conv += 1
                    continue
                first = conv[0]
                second = conv[1]
                if first.get("role") != "user" or second.get("role") != "assistant":
                    n_skipped_role += 1
                    continue
                lang = (
                    r.get("language")
                    if args.language_source == "top_level"
                    else first.get("language")
                )
                if lang != args.language:
                    n_skipped_lang += 1
                    continue
                if args.require_flagged and not is_flagged_row(r):
                    n_skipped_flag += 1
                    continue
                if do_subsample and rng.random() >= args.subsample_fraction:
                    n_skipped_subsample += 1
                    continue
                asst = second.get("content") or ""
                if not asst:
                    n_skipped_empty += 1
                    continue
                char_len = len(asst)
                word_count = len(asst.split())
                w.writerow(
                    [shard_idx, i, lang, char_len, word_count, r.get("conversation_hash") or ""]
                )
                n_kept += 1
                n_shard_kept += 1
            print(
                f"[shard {shard_idx + 1}/{len(shards)}] {shard.name}: kept_this_shard={n_shard_kept} cum_kept={n_kept} elapsed={time.time() - t_start:.1f}s",
                flush=True,
            )
            if args.limit_rows and n_seen >= args.limit_rows:
                break

    manifest = {
        "wildchat_dir": str(args.wildchat_dir),
        "language_source": args.language_source,
        "language": args.language,
        "require_flagged": args.require_flagged,
        "subsample_fraction": args.subsample_fraction,
        "subsample_seed": args.subsample_seed,
        "n_seen": n_seen,
        "n_kept": n_kept,
        "n_skipped_no_conv": n_skipped_no_conv,
        "n_skipped_role": n_skipped_role,
        "n_skipped_lang": n_skipped_lang,
        "n_skipped_flag": n_skipped_flag,
        "n_skipped_subsample": n_skipped_subsample,
        "n_skipped_empty": n_skipped_empty,
        "out_csv": str(out_csv),
        "wall_seconds": time.time() - t_start,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] kept={n_kept}/{n_seen} -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
