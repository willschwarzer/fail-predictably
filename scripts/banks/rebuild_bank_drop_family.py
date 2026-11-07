"""Rebuild an lquant password bank with one bulk family dropped, pool size preserved.

WHY. The `password_seeded` family is not an extraction attack: its prompts are
benign text with the password word spliced in ("Describe pottery plants in
detail", where 'plants' is the slot password), so it measures in-context copying
rather than disclosure under a non-disclosure instruction. That is why it scales
with model size, and at Qwen3-4B/8B it is the ENTIRE source of near-ceiling bulk
leak mass -- 110 and 199 prompts above 1e-4 per 1864 against 0-10 for every other
bulk family. Its near-ceiling prompts land in the 44-prompt calibration sample often enough to
hand the baseline forecast the answer, which destroys the measurement.

WHY NOT JUST DROP IT AT TRAIN TIME. Passing a shorter `--train-families` list
would cut the per-slot pool from 1870 to 1404, which moves the held-out cell from
the paper's fit 44 / deploy 891 to roughly 33 / 669. The cell is the paper's
held-out evaluation cell, so it must not move as a side effect of a family
change. Instead this script regenerates the three
surviving bulk families at a larger per-family count so the pool stays at 1870.

WHAT IS AND IS NOT REGENERATED. GCG rows are copied verbatim from the source bank
-- they are the expensive, model-specific, warm-started artifact and rebuilding
them would mean re-running the GCG array. Only the template-generated bulk
families are re-sampled, on CPU, in seconds. Family capacities were checked
before choosing the counts: compositional 12600, indirect_extraction 825,
mutate 960 distinct prompts, all above the 622 requested.

Determinism: each (slot, family) draw is seeded from bank_seed and the slot
index, so the same source bank always produces the same output bank.

The paper's four nopwseed banks: scripts/banks/build_nopwseed_strictv2_banks.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))
from forecast_experiment.prompt_generators import sample_unique_password_prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-bank", required=True, help="source bank root (contains bank_seed_N/)")
    ap.add_argument("--dst-bank", required=True, help="destination bank root")
    ap.add_argument("--drop-family", default="password_seeded")
    ap.add_argument("--keep-family", default="gcg_steps80",
                    help="family copied verbatim rather than regenerated")
    ap.add_argument("--bank-seed", type=int, default=11)
    ap.add_argument(
        "--allocate", nargs="*", default=None, metavar="FAMILY=COUNT",
        help=(
            "Explicit per-slot count for each surviving regenerated family. "
            "Without it the freed budget is split evenly, which is the WRONG "
            "default here: `mutate` carries a small residual near-ceiling tail "
            "at Qwen3-4B (10 prompts above 1e-4 per 1864, spread over 3 of 4 "
            "screened slots) while `compositional` and `indirect_extraction` "
            "are clean at every size. Growing mutate to absorb the freed budget "
            "would scale its contamination by the same factor. Prefer sending "
            "the surplus to `compositional`, which is clean everywhere and has "
            "12600 distinct candidates."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_seed_dir = os.path.join(args.src_bank, f"bank_seed_{args.bank_seed}")
    meta = json.load(open(os.path.join(src_seed_dir, "metadata.json")))
    families = list(meta["families"])
    if args.drop_family not in families:
        raise SystemExit(f"{args.drop_family!r} not in source families {families}")

    rows = [json.loads(line) for line in open(os.path.join(src_seed_dir, "prompt_rows.jsonl"))]
    by_slot = defaultdict(list)
    for r in rows:
        by_slot[r["t"]].append(r)
    per_slot_total = len(by_slot[0])
    n_keep = sum(1 for r in by_slot[0] if families[r["f"]] == args.keep_family)

    new_families = [f for f in families if f != args.drop_family]
    regen = [f for f in new_families if f != args.keep_family]
    # Split the freed budget so the per-slot total is unchanged.
    budget = per_slot_total - n_keep
    if args.allocate:
        counts = {}
        for spec in args.allocate:
            fam, _, cnt = spec.partition("=")
            if fam not in regen:
                raise SystemExit(f"--allocate names {fam!r}, not in regenerated families {regen}")
            counts[fam] = int(cnt)
        missing = [f for f in regen if f not in counts]
        if missing:
            raise SystemExit(f"--allocate must cover every regenerated family; missing {missing}")
        if sum(counts.values()) != budget:
            raise SystemExit(
                f"--allocate sums to {sum(counts.values())} but the budget is {budget} "
                f"({per_slot_total} per-slot total minus {n_keep} {args.keep_family})"
            )
    else:
        base, extra = divmod(budget, len(regen))
        counts = {f: base + (1 if i < extra else 0) for i, f in enumerate(regen)}

    print(f"source families : {families}")
    print(f"new families    : {new_families}")
    print(f"per-slot total  : {per_slot_total} (unchanged)")
    print(f"  {args.keep_family}: {n_keep} copied verbatim")
    for f, c in counts.items():
        print(f"  {f}: {c} regenerated (source had "
              f"{sum(1 for r in by_slot[0] if families[r['f']] == f)})")
    if args.dry_run:
        return

    fam_idx = {f: i for i, f in enumerate(new_families)}
    slot_to_password = meta["slot_to_password"]
    out_rows = []
    for slot in sorted(by_slot):
        label = f"password_slot_{slot:02d}"
        password = slot_to_password[label]
        for f in regen:
            rng = random.Random(args.bank_seed * 1_000_003 + slot * 1009 + fam_idx[f])
            prompts = sample_unique_password_prompts(
                counts[f], f, rng, password_text=password
            )
            for p in prompts:
                out_rows.append({"t": slot, "f": fam_idx[f], "p": p})
        for r in by_slot[slot]:
            if families[r["f"]] == args.keep_family:
                out_rows.append({"t": slot, "f": fam_idx[args.keep_family], "p": r["p"]})

    dst_seed_dir = os.path.join(args.dst_bank, f"bank_seed_{args.bank_seed}")
    os.makedirs(dst_seed_dir, exist_ok=True)
    with open(os.path.join(dst_seed_dir, "prompt_rows.jsonl"), "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    new_meta = dict(meta)
    new_meta["families"] = new_families
    new_meta["train_families"] = [f for f in meta.get("train_families", []) if f != args.drop_family]
    new_meta["test_families"] = [f for f in meta.get("test_families", []) if f != args.drop_family]
    new_meta["cheap_prompts_per_family_map"] = counts
    new_meta["derived_from"] = args.src_bank
    new_meta["dropped_family"] = args.drop_family
    new_meta["rebuild_note"] = (
        f"{args.drop_family} removed as a measurement confound (in-context copying, "
        f"not extraction); surviving bulk families re-sampled so the per-slot pool "
        f"stays at {per_slot_total} and the fit/deploy cell is unchanged. "
        f"{args.keep_family} rows copied verbatim from the source bank."
    )
    json.dump(new_meta, open(os.path.join(dst_seed_dir, "metadata.json"), "w"), indent=1)
    for extra_f in ("generation_config.json", "merge_summary.json"):
        srcp = os.path.join(src_seed_dir, extra_f)
        if os.path.exists(srcp):
            shutil.copy2(srcp, dst_seed_dir)

    counts_check = defaultdict(lambda: defaultdict(int))
    for r in out_rows:
        counts_check[r["t"]][new_families[r["f"]]] += 1
    totals = {sum(v.values()) for v in counts_check.values()}
    dupes = sum(
        len(g) - len({r["p"] for r in g})
        for g in [[r for r in out_rows if r["t"] == s] for s in sorted(counts_check)]
    )
    print(f"\nwrote {len(out_rows)} rows over {len(counts_check)} slots -> {dst_seed_dir}")
    print(f"per-slot totals: {totals}  (must be {{{per_slot_total}}})")
    print(f"within-slot duplicate prompts: {dupes}  (must be 0)")
    if totals != {per_slot_total} or dupes:
        raise SystemExit("VALIDATION FAILED")


if __name__ == "__main__":
    main()
