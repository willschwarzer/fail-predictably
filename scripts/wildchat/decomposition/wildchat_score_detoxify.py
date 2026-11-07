"""Cell A scorer: extract per-prompt detoxify toxicity scores from WildChat-1M.

Walks every arrow shard in the WildChat directory, filters to first-user-turn
English prompts, and writes per-prompt detoxify toxicity scores to a CSV.

The CSV column `logit_toxicity` is the unbounded transform of the [0,1] toxicity
score:  log(t/(1-t))  with t clamped to [eps, 1-eps].  This is the score we feed
into the FRLMB OLS-tail diagnostic in `wildchat_frlmb_analyze.py`.

The English-filter logic is parameterized; the schema verifier
(`wildchat_schema_verify.py`) reports per-row per-turn language coverage so the
caller can pick the right field. Default: top-level `language == 'English'`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

from datasets import Dataset as HFDataset


def find_shards(arrow_dir: Path) -> list[Path]:
    return sorted(arrow_dir.rglob("*.arrow"))


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
    ap.add_argument(
        "--language-source",
        choices=("top_level", "first_turn"),
        default="top_level",
        help="Field to filter on for the English-prompt slice.",
    )
    ap.add_argument(
        "--language",
        default="English",
        help="Language string to keep (matches WildChat's value, not an ISO code).",
    )
    ap.add_argument(
        "--logit-eps",
        type=float,
        default=1e-6,
        help="Clamp toxicity to [eps, 1-eps] before logit-transform.",
    )
    ap.add_argument("--limit-rows", type=int, default=0, help="Debug: stop after this many rows total (0 = all).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = find_shards(args.wildchat_dir)
    if not shards:
        raise FileNotFoundError(f"No arrow shards under {args.wildchat_dir}")
    print(f"[load] {len(shards)} shards under {args.wildchat_dir}", flush=True)

    out_csv = args.out_dir / "wildchat_detoxify_scores.csv"
    eps = float(args.logit_eps)
    eps_hi = 1.0 - eps

    n_total = 0
    n_kept = 0
    n_skipped_no_conv = 0
    n_skipped_not_english = 0
    n_skipped_not_user_first = 0
    n_skipped_no_detox = 0

    t_start = time.time()
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shard_idx", "row_idx", "language", "toxicity", "logit_toxicity", "conversation_hash"])
        for shard_idx, shard in enumerate(shards):
            ds = HFDataset.from_file(str(shard))
            n_shard = len(ds)
            n_kept_shard = 0
            for i in range(n_shard):
                if args.limit_rows and n_total >= args.limit_rows:
                    break
                n_total += 1
                r = ds[i]
                conv = r.get("conversation", [])
                if not conv:
                    n_skipped_no_conv += 1
                    continue
                first = conv[0]
                if first.get("role") != "user":
                    n_skipped_not_user_first += 1
                    continue
                if args.language_source == "top_level":
                    lang = r.get("language")
                else:
                    lang = first.get("language")
                if lang != args.language:
                    n_skipped_not_english += 1
                    continue
                detox = r.get("detoxify_moderation", [])
                if (
                    not detox
                    or not isinstance(detox[0], dict)
                    or detox[0].get("toxicity") is None
                ):
                    n_skipped_no_detox += 1
                    continue
                t = float(detox[0]["toxicity"])
                t_clamped = min(max(t, eps), eps_hi)
                logit_t = math.log(t_clamped / (1.0 - t_clamped))
                w.writerow([shard_idx, i, lang, f"{t:.10g}", f"{logit_t:.10g}", r.get("conversation_hash", "")])
                n_kept += 1
                n_kept_shard += 1
            el = time.time() - t_start
            print(
                f"[shard {shard_idx + 1}/{len(shards)}] {shard.name}: "
                f"rows={n_shard} kept={n_kept_shard} cum_kept={n_kept} elapsed={el:.1f}s",
                flush=True,
            )
            if args.limit_rows and n_total >= args.limit_rows:
                break

    manifest = {
        "wildchat_dir": str(args.wildchat_dir),
        "n_shards_scanned": len(shards),
        "language_source": args.language_source,
        "language": args.language,
        "logit_eps": eps,
        "n_total_rows_scanned": n_total,
        "n_kept": n_kept,
        "n_skipped_no_conv": n_skipped_no_conv,
        "n_skipped_not_user_first": n_skipped_not_user_first,
        "n_skipped_not_english": n_skipped_not_english,
        "n_skipped_no_detox": n_skipped_no_detox,
        "out_csv": str(out_csv),
        "wall_seconds": time.time() - t_start,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] kept={n_kept}/{n_total} -> {out_csv}", flush=True)
    print(f"[done] manifest -> {args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
