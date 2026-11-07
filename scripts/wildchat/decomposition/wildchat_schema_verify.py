"""Smoke-test inspection of WildChat-1M shard 0.

Confirms the structure assumptions for the FRLMB-on-WildChat experiment:
- Whether `detoxify_moderation[0]` is the user-turn score (vs. assistant or absent).
- The shape of `openai_moderation` and `category_scores`.
- How English first-user-turn prompts can be detected (top-level `language`,
  per-turn `language`, or `conversation[0].role`).
- Rough fraction of rows passing each candidate filter on shard 0.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from datasets import Dataset as HFDataset


def trunc(s: str, n: int = 200) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shard",
        type=Path,
        default=None,
        help="One WildChat arrow shard (default: the first shard, in sorted order, under "
        "$WILDCHAT_DATASET, else data/wildchat).",
    )
    ap.add_argument("--n-sample", type=int, default=10000, help="rows to scan for stats")
    args = ap.parse_args()
    if args.shard is None:
        wildchat_dir = Path(os.environ.get("WILDCHAT_DATASET", "data/wildchat"))
        shards = sorted(wildchat_dir.rglob("*.arrow"))
        if not shards:
            raise FileNotFoundError(f"No arrow shards under {wildchat_dir}")
        args.shard = shards[0]

    print(f"[load] {args.shard}", flush=True)
    ds = HFDataset.from_file(str(args.shard))
    print(f"[load] shard rows = {len(ds)}", flush=True)
    print(f"[schema] columns = {list(ds.column_names)}", flush=True)

    print("\n=== row 0 inspection ===", flush=True)
    r0 = ds[0]
    print(f"top-level language = {r0.get('language')!r}", flush=True)
    print(f"top-level toxic    = {r0.get('toxic')!r}", flush=True)
    conv = r0.get("conversation", [])
    print(f"conversation len   = {len(conv)}", flush=True)
    for i, turn in enumerate(conv[:3]):
        print(
            f"  turn[{i}] role={turn.get('role')!r} language={turn.get('language')!r} "
            f"toxic={turn.get('toxic')!r} content={trunc(turn.get('content', ''), 120)!r}",
            flush=True,
        )

    detox = r0.get("detoxify_moderation", [])
    print(f"detoxify_moderation len = {len(detox)}", flush=True)
    if detox:
        print(f"  detox[0] keys     = {list(detox[0].keys())}", flush=True)
        print(f"  detox[0]          = {detox[0]}", flush=True)

    omod = r0.get("openai_moderation", [])
    print(f"openai_moderation len   = {len(omod)}", flush=True)
    if omod:
        e0 = omod[0]
        print(f"  omod[0] keys      = {list(e0.keys())}", flush=True)
        print(f"  omod[0].flagged   = {e0.get('flagged')!r}", flush=True)
        cats = e0.get("categories", {})
        scores = e0.get("category_scores", {})
        print(f"  omod[0].categories example = {dict(list(cats.items())[:5])}", flush=True)
        print(f"  omod[0].category_scores example = {dict(list(scores.items())[:5])}", flush=True)

    print("\n=== filter stats over first {} rows ===".format(args.n_sample), flush=True)
    n = min(args.n_sample, len(ds))
    n_top_english = 0
    n_first_user = 0
    n_first_user_english = 0
    n_top_lang_match_first_turn = 0
    n_detox_present = 0
    n_detox_aligned_with_first_turn = 0  # if first turn has its toxic flag, detox[0].toxicity is presumably its score
    n_toxic_top = 0
    n_omod_first_flagged = 0
    n_any_flagged_in_omod = 0
    lang_counter: Counter = Counter()

    detox_tox_samples: list[float] = []
    flagged_samples_first_msg: list[str] = []

    for i in range(n):
        r = ds[i]
        lang_top = r.get("language")
        lang_counter[lang_top] += 1
        if lang_top == "English":
            n_top_english += 1
        conv = r.get("conversation", [])
        first = conv[0] if conv else None
        if first and first.get("role") == "user":
            n_first_user += 1
            if first.get("language") == "English":
                n_first_user_english += 1
            if lang_top == first.get("language"):
                n_top_lang_match_first_turn += 1
        detox = r.get("detoxify_moderation", [])
        if detox and isinstance(detox[0], dict) and detox[0].get("toxicity") is not None:
            n_detox_present += 1
            detox_tox_samples.append(float(detox[0]["toxicity"]))
        if r.get("toxic") is True:
            n_toxic_top += 1
        omod = r.get("openai_moderation", [])
        if omod and isinstance(omod[0], dict) and omod[0].get("flagged"):
            n_omod_first_flagged += 1
        if any(isinstance(e, dict) and e.get("flagged") for e in omod):
            n_any_flagged_in_omod += 1
        if (
            (r.get("toxic") is True or any(isinstance(e, dict) and e.get("flagged") for e in omod))
            and first
            and first.get("role") == "user"
            and len(flagged_samples_first_msg) < 3
        ):
            flagged_samples_first_msg.append(trunc(first.get("content", ""), 200))

    def pct(num: int) -> str:
        return f"{num} ({num / max(n, 1):.2%})"

    print(f"top-level language=='English'           : {pct(n_top_english)}", flush=True)
    print(f"conversation[0].role=='user'            : {pct(n_first_user)}", flush=True)
    print(f"  ... and conv[0].language=='English'   : {pct(n_first_user_english)}", flush=True)
    print(f"top-level lang matches conv[0].language : {pct(n_top_lang_match_first_turn)}", flush=True)
    print(f"detoxify_moderation[0].toxicity present : {pct(n_detox_present)}", flush=True)
    print(f"top-level toxic==True                   : {pct(n_toxic_top)}", flush=True)
    print(f"openai_moderation[0].flagged==True      : {pct(n_omod_first_flagged)}", flush=True)
    print(f"any openai_moderation entry flagged     : {pct(n_any_flagged_in_omod)}", flush=True)

    if detox_tox_samples:
        import statistics

        srt = sorted(detox_tox_samples)
        n_d = len(srt)
        print(
            "\ndetoxify[0].toxicity sample stats (first {} rows): n={} min={:.2e} median={:.2e} "
            "p90={:.4f} p99={:.4f} p99.9={:.4f} max={:.4f}".format(
                args.n_sample,
                n_d,
                srt[0],
                srt[n_d // 2],
                srt[int(n_d * 0.90)],
                srt[int(n_d * 0.99)],
                srt[int(n_d * 0.999)] if n_d >= 1000 else float("nan"),
                srt[-1],
            ),
            flush=True,
        )
        # mean + stdev for sanity
        print(
            f"detoxify[0].toxicity mean={statistics.fmean(srt):.4e} "
            f"stdev={statistics.pstdev(srt):.4e}",
            flush=True,
        )

    print("\ntop 10 top-level languages:", flush=True)
    for lang, cnt in lang_counter.most_common(10):
        print(f"  {lang!r}: {cnt} ({cnt / n:.2%})", flush=True)

    print("\nflagged-or-toxic first-user-turn samples:", flush=True)
    for s in flagged_samples_first_msg:
        print(f"  - {s!r}", flush=True)

    print(
        "\n[done] Use these results to confirm:"
        "\n  (a) detox[0] aligns with first user turn (compare turn[0].toxic to detox[0].toxicity > 0.5),"
        "\n  (b) which language field is more reliable for the English filter,"
        "\n  (c) the rough size of the flagged-or-toxic slice (extrapolate from this shard)."
    )


if __name__ == "__main__":
    main()
