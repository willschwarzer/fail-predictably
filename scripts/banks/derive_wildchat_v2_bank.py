"""Derive a v2 WildChat token_union bank: an event-id subset plus an input-only pool rule.

The v1 event counts any answer whose first response token is one of 38 ids, so
benign answers that open with "Sex ..." (to "What is sex?") or with a subword
such as "Ass" ("Assalam ...", "Assist ...") sit at p ~ 1 and carry the forecast
ceiling. This script

  1. keeps a subset of the event ids, written to the bank's per-target
     `target_event_token_ids` map (so `verify_token_union` still checks the
     recorded 38-id union against the tokenizer, and the trainer scores the
     subset; `resolve_target_event_token_ids` prefers the per-target map);
  2. drops every prompt that contains one of the KEPT event strings, a rule on
     the input fixed before any scoring: `--rule word` matches whole words,
     `--rule prefix` (default) any word that begins with an event string, so
     "sexuality" and "Fuckhead" count (case-insensitive either way); `--rule
     none` keeps every prompt (the event subset alone);
  3. re-chunks the surviving rows, in the source's file order (one seeded
     shuffle, so already random), into pools of `--pool-size`, dropping the
     remainder.

  python scripts/banks/derive_wildchat_v2_bank.py SRC_BANK_SEED_DIR DST_BANK_SEED_DIR \
      --drop ass,Ass,ASS,... [--rule prefix|word|none] [--pool-size 2300]

The paper's bank is step 3 of the chain in
scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch (--rule none, full --drop list).
"""
import argparse
import json
import re
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="source bank_seed_<n> directory")
    ap.add_argument("dst", type=Path, help="destination bank_seed_<n> directory (must not exist)")
    ap.add_argument("--drop", required=True, help="comma-separated event strings to remove from the event")
    ap.add_argument("--rule", choices=("prefix", "word", "none"), default="prefix")
    ap.add_argument("--pool-size", type=int, default=2300)
    args = ap.parse_args()

    meta = json.loads((args.src / "metadata.json").read_text())
    if meta.get("target_event_token_ids"):
        raise SystemExit("source bank already has a per-target event map; derive from a full-union bank")
    union = meta["event_token_union"]
    decoded = {int(k): str(v) for k, v in union["decoded"].items()}
    drop = {s for s in args.drop.split(",") if s}
    unknown = drop - set(decoded.values())
    if unknown:
        raise SystemExit(f"--drop names strings outside the event: {sorted(unknown)}")
    keep_ids = sorted(i for i in union["ids"] if decoded[int(i)] not in drop)
    keep_strings = sorted({decoded[int(i)] for i in keep_ids})
    alts = sorted({re.escape(s.lower()) for s in keep_strings}, key=len, reverse=True)
    tail = r"(?![a-z0-9])" if args.rule == "word" else ""
    rx = re.compile(r"(?<![a-z0-9])(" + "|".join(alts) + r")" + tail, re.IGNORECASE)

    kept, n_src = [], 0
    with open(args.src / "prompt_rows.jsonl", encoding="utf-8") as f:
        for line in f:
            n_src += 1
            row = json.loads(line)
            if args.rule == "none" or not rx.search(row["p"]):
                kept.append(row)
    n_pools = len(kept) // args.pool_size
    if n_pools < 2:
        raise SystemExit(f"only {len(kept)} prompts survive the rule: fewer than two pools")
    width = max(2, len(str(n_pools - 1)))
    targets = [f"wc{j:0{width}d}" for j in range(n_pools)]
    family = meta["families"][0]

    args.dst.mkdir(parents=True, exist_ok=False)
    with open(args.dst / "prompt_rows.jsonl", "w", encoding="utf-8") as out:
        for j in range(n_pools):
            for row in kept[j * args.pool_size:(j + 1) * args.pool_size]:
                row["t"] = j
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    meta["targets"] = targets
    meta["prompts_per_target"] = args.pool_size
    meta["prompt_counts_by_target"] = {t: {family: args.pool_size} for t in targets}
    meta["row_count"] = n_pools * args.pool_size
    meta["target_event_token_ids"] = {t: keep_ids for t in targets}
    meta["provenance"] = dict(meta.get("provenance") or {})
    meta["provenance"]["v2_derived_from"] = {
        "src": str(args.src.resolve()),
        "src_rows": n_src,
        "dropped_event_strings": sorted(drop),
        "kept_event_strings": keep_strings,
        "kept_event_ids": keep_ids,
        "pool_rule": {
            "word": "drop prompts containing any kept event string as a whole word (case-insensitive)",
            "prefix": "drop prompts containing any word that begins with a kept event string (case-insensitive)",
            "none": "keep every prompt",
        }[args.rule],
        "rows_passing_rule": len(kept),
        "pool_size": args.pool_size,
        "rows_unused_remainder": len(kept) - n_pools * args.pool_size,
        "rule": "surviving rows in source file order, chunked consecutively into pools",
    }
    (args.dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    for name in ("kl_eval_nonflagged_en.jsonl", "build_report.json"):
        if (args.src / name).exists():
            shutil.copy2(args.src / name, args.dst / name)
    print(f"{n_src} source rows, {len(kept)} pass the rule, {n_pools} pools x {args.pool_size} -> {args.dst}; "
          f"event keeps {len(keep_ids)} ids: {keep_strings}")


if __name__ == "__main__":
    main()
