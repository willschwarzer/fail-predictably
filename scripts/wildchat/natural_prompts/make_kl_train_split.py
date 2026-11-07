"""Write the KL training prompts for the natural-prompt runs.

The bank builder writes 2,048 non-flagged English WildChat prompts to
kl_eval_nonflagged_en.jsonl in each bank directory. The trainer measures KL on
256 of them, which it draws with a fixed seed (load_kl_dataset in
train_lquant.py). This script writes the other 1,792 prompts, in their original
order, to kl_train_nonflagged_en_disjoint.jsonl, so that with KL_TRAIN=nonflagged
the KL penalty trains on prompts that the KL evaluation never sees.

    python scripts/wildchat/natural_prompts/make_kl_train_split.py \\
        $TASK_BANK_ROOT/wildchat_ldnoobw_qwen_qwen3-0_6b_T40_p1870/bank_seed_11
"""
import argparse
import json
import random
from pathlib import Path

# These match load_kl_dataset's default seed and the trainer's n_samples=256 for
# --kl-eval-datasets.
EVAL_SEED = 42
EVAL_N = 256


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bank_dir", type=Path, help="bank directory holding the KL eval sidecar")
    ap.add_argument("--eval-name", default="kl_eval_nonflagged_en.jsonl")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: kl_train_nonflagged_en_disjoint.jsonl in bank_dir)",
    )
    args = ap.parse_args()

    # Keep the same rows as load_kl_dataset (non-blank lines with a non-empty
    # prompt), so that the indices below refer to the same list.
    lines = []
    with open(args.bank_dir / args.eval_name, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if obj.get("p") or obj.get("prompt"):
                lines.append(s + "\n")

    if len(lines) > EVAL_N:
        held_out = set(random.Random(EVAL_SEED).sample(range(len(lines)), EVAL_N))
    else:
        held_out = set(range(len(lines)))

    out = args.out or args.bank_dir / "kl_train_nonflagged_en_disjoint.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            if i not in held_out:
                f.write(line)
    print(f"wrote {len(lines) - len(held_out)} of {len(lines)} prompts to {out}")


if __name__ == "__main__":
    main()
