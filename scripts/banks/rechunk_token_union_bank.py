"""Split every target of a token_union prompt bank into equal sub-targets.

The WildChat token_union banks are built by one seeded shuffle of the flagged
pool followed by chunking (generate_wildchat_token_union_bank.py), so file order
within a target is already random. Splitting each target's rows into `--parts`
consecutive blocks therefore gives `parts`x as many random disjoint pools of
1/parts the size, with the same prompts, event and family. T16_p4600 -> T32_p2300
doubles the number of 2300-prompt pools available to train on and hold out
(the paper's minimal-change WildChat runs use 250 fit + 2050 deploy per pool).
Step 2 of the chain in scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch.

  python scripts/banks/rechunk_token_union_bank.py SRC_BANK_SEED_DIR DST_BANK_SEED_DIR [--parts 2]
"""
import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="source bank_seed_<n> directory")
    ap.add_argument("dst", type=Path, help="destination bank_seed_<n> directory (must not exist)")
    ap.add_argument("--parts", type=int, default=2)
    args = ap.parse_args()

    meta = json.loads((args.src / "metadata.json").read_text())
    old_targets = list(meta["targets"])
    per_target = int(meta["prompts_per_target"])
    if per_target % args.parts:
        raise SystemExit(f"prompts_per_target={per_target} is not divisible by {args.parts}")
    new_per = per_target // args.parts

    rows_by_target: dict[int, list[str]] = {i: [] for i in range(len(old_targets))}
    with open(args.src / "prompt_rows.jsonl", encoding="utf-8") as f:
        for line in f:
            rows_by_target[int(json.loads(line)["t"])].append(line)
    for i, rows in rows_by_target.items():
        if len(rows) != per_target:
            raise SystemExit(f"target {old_targets[i]} has {len(rows)} rows != {per_target}")

    n_new = len(old_targets) * args.parts
    width = max(2, len(str(n_new - 1)))
    new_targets = [f"wc{j:0{width}d}" for j in range(n_new)]
    family = meta["families"][0]

    args.dst.mkdir(parents=True, exist_ok=False)
    n_written = 0
    with open(args.dst / "prompt_rows.jsonl", "w", encoding="utf-8") as out:
        for i in range(len(old_targets)):
            for part in range(args.parts):
                j = i * args.parts + part
                for line in rows_by_target[i][part * new_per:(part + 1) * new_per]:
                    row = json.loads(line)
                    row["t"] = j
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    n_written += 1

    meta["targets"] = new_targets
    meta["prompts_per_target"] = new_per
    meta["prompt_counts_by_target"] = {t: {family: new_per} for t in new_targets}
    meta["row_count"] = n_written
    meta["provenance"] = dict(meta.get("provenance") or {})
    meta["provenance"]["rechunked_from"] = {
        "src": str(args.src.resolve()),
        "parts": args.parts,
        "rule": "new target i*parts+p = rows [p*n/parts, (p+1)*n/parts) of old target i, file order",
    }
    (args.dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    for name in ("kl_eval_nonflagged_en.jsonl", "build_report.json"):
        if (args.src / name).exists():
            shutil.copy2(args.src / name, args.dst / name)
    print(f"wrote {n_written} rows, {n_new} targets x {new_per} -> {args.dst}")


if __name__ == "__main__":
    main()
