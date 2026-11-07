"""Cell B (v4) scorer: NLL of the WildChat assistant turn under base Qwen3-0.6B.

For each row in the slice, score the per-prompt NLL of the actual WildChat
assistant turn given the user's first-turn prompt (with chat template):

  nll = -sum_i log P(t_i | t_<i, prompt+template)

Higher = the recorded continuation is more "surprising" to the model = more
"deployment-fragile" prompt. AR correlations within the assistant turn make
this score a candidate for Frechet-style heavy-tail behavior, in contrast to
single-token log_p scores which are structurally bounded.

Output:
- wildchat_nll_scores.csv  -- per-prompt nll, n_assistant_tokens, mean_nll.
- manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable

import sys

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _modelscoring_common import (  # noqa: E402
    last_k_target_log_probs,
    load_causal_lm,
    normalize_prompt_text,
)


def find_shards(arrow_dir: Path) -> list[Path]:
    return sorted(arrow_dir.rglob("*.arrow"))


def is_flagged_row(row: dict) -> bool:
    if row.get("toxic") is True:
        return True
    omod = row.get("openai_moderation", [])
    return any(isinstance(e, dict) and e.get("flagged") for e in omod)


def stream_rows(
    shards: list[Path],
    *,
    language_source: str,
    language: str,
    require_flagged: bool,
    limit_rows: int,
) -> Iterable[tuple[int, int, str, str, str, str]]:
    """Yield (shard_idx, row_idx, language, conversation_hash, user_msg, assistant_msg)."""
    n_total = 0
    for shard_idx, shard in enumerate(shards):
        ds = HFDataset.from_file(str(shard))
        for i in range(len(ds)):
            if limit_rows and n_total >= limit_rows:
                return
            n_total += 1
            r = ds[i]
            conv = r.get("conversation", [])
            if len(conv) < 2:
                continue
            first = conv[0]
            second = conv[1]
            if first.get("role") != "user" or second.get("role") != "assistant":
                continue
            lang = r.get("language") if language_source == "top_level" else first.get("language")
            if lang != language:
                continue
            if require_flagged and not is_flagged_row(r):
                continue
            user_msg = first.get("content") or ""
            asst_msg = second.get("content") or ""
            if not user_msg or not asst_msg:
                continue
            yield shard_idx, i, lang or "", r.get("conversation_hash") or "", user_msg, asst_msg


def build_full_sequence(
    tokenizer,
    user_msg: str,
    asst_msg: str,
    *,
    max_len: int,
) -> tuple[list[int], int, int]:
    """Return (full_token_ids, prefix_len, asst_len_in_full).

    prefix_len = #tokens before the assistant turn starts (we'll only score
    log-probs at positions >= prefix_len). The assistant span has length
    asst_len_in_full = len(full_token_ids) - prefix_len.

    Truncates the assistant turn to fit within max_len; the prefix is preserved
    as-is. If the prefix alone exceeds max_len-1, we truncate the LEFT side of
    the prefix (keeping the system + tail) so at least one assistant token can
    be scored.
    """
    msgs_prefix = [{"role": "user", "content": user_msg}]
    try:
        prefix_text = tokenizer.apply_chat_template(
            msgs_prefix, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prefix_text = tokenizer.apply_chat_template(
            msgs_prefix, tokenize=False, add_generation_prompt=True
        )
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    asst_ids = tokenizer(asst_msg, add_special_tokens=False).input_ids
    if len(prefix_ids) >= max_len - 1:
        prefix_ids = prefix_ids[-(max_len - 1):]
    asst_budget = max_len - len(prefix_ids)
    asst_ids = asst_ids[:asst_budget]
    if not asst_ids:
        return [], len(prefix_ids), 0
    full = prefix_ids + asst_ids
    return full, len(prefix_ids), len(asst_ids)


@torch.no_grad()
def batch_score_nll(
    model,
    pad_id: int,
    seqs: list[tuple[list[int], int, int]],
    *,
    device: str,
) -> list[tuple[float, int]]:
    """Return [(nll, n_assistant_tokens)] per sequence.

    Uses left-padding so the right end of every sequence is the assistant turn;
    the assistant span occupies the last ``asst_len`` positions. Scoring goes
    through ``last_k_target_log_probs``, which runs the backbone once and applies
    the LM head in small position-chunks, keeping only the target-token log-prob
    per position. Memory is bounded by the head chunk size regardless of the
    response length, so long assistant turns are scored exactly, with no
    truncation-driven cap on the span (essential for Gemma 3's 262k-token
    vocabulary, where full-span logits at 8k tokens would be tens of GB).
    """
    if not seqs:
        return []
    full_lengths = [len(s[0]) for s in seqs]
    max_in = max(full_lengths)
    n = len(seqs)
    padded = torch.full((n, max_in), pad_id, dtype=torch.long)
    attn = torch.zeros((n, max_in), dtype=torch.long)
    for i, (full, _prefix_len, _asst_len) in enumerate(seqs):
        L = len(full)
        offset = max_in - L  # left-pad
        padded[i, offset:] = torch.tensor(full, dtype=torch.long)
        attn[i, offset:] = 1
    max_asst = max(asst_len for _f, _pl, asst_len in seqs)
    enc_ids = padded.to(device)
    enc_attn = attn.to(device)
    # [n, max_asst]: log P(token at position max_in-max_asst+j | everything before).
    lp_window = last_k_target_log_probs(model, enc_ids, enc_attn, max_asst)
    out: list[tuple[float, int]] = []
    for i, (_full, _prefix_len, asst_len) in enumerate(seqs):
        if asst_len == 0:
            out.append((0.0, 0))
            continue
        # This row's assistant tokens are the last asst_len window entries; all
        # of them are real (non-pad) positions by construction.
        nll = -lp_window[i, max_asst - asst_len:].sum().item()
        out.append((float(nll), int(asst_len)))
    return out


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
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    ap.add_argument("--language-source", choices=("top_level", "first_turn"), default="top_level")
    ap.add_argument("--language", default="English")
    ap.add_argument("--require-flagged", action="store_true", default=False)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--max-kept", type=int, default=0,
                    help="Stop after this many kept prompts (deterministic slice). 0 = unlimited.")
    ap.add_argument("--dedup", action="store_true", default=False,
                    help="Keep only the first occurrence of each whitespace-"
                         "normalized prompt text (removes spammed-template duplicates).")
    ap.add_argument("--length-sort-buffer", type=int, default=32)
    ap.add_argument("--subsample-fraction", type=float, default=1.0)
    ap.add_argument("--subsample-seed", type=int, default=11)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "wildchat_nll_scores.csv"
    manifest_path = args.out_dir / "manifest.json"

    print(f"[load] tokenizer + model {args.base_model} dtype={args.dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model = load_causal_lm(args.base_model, dtype=dtype_map[args.dtype], device=args.device)

    shards = find_shards(args.wildchat_dir)
    print(f"[load] {len(shards)} shards under {args.wildchat_dir}", flush=True)

    n_kept = 0
    t_start = time.time()
    last_log = t_start
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    buffer_size = max(args.micro_batch, args.length_sort_buffer * args.micro_batch)
    # Each item: (shard_idx, row_idx, lang, hash, full_ids, prefix_len, asst_len)
    buf: list[tuple[int, int, str, str, list[int], int, int]] = []

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "shard_idx",
                "row_idx",
                "language",
                "nll",
                "n_assistant_tokens",
                "mean_nll",
                "conversation_hash",
            ]
        )

        def score_chunk(items: list[tuple[int, int, str, str, list[int], int, int]]) -> None:
            nonlocal n_kept, last_log
            if not items:
                return
            seqs = [(it[4], it[5], it[6]) for it in items]

            def score_halving(ss):
                # Recursive OOM-halving: keep splitting until the batch fits
                # (down to single prompts), rather than giving up after one split.
                try:
                    return batch_score_nll(model, pad_id, ss, device=args.device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(ss) <= 1:
                        raise
                    mid = len(ss) // 2
                    return score_halving(ss[:mid]) + score_halving(ss[mid:])

            results = score_halving(seqs)
            for (shard_idx, row_idx, lang, chash, _ids, _pl, asst_len), (nll, n_asst) in zip(
                items, results
            ):
                mean_nll = (nll / n_asst) if n_asst > 0 else float("nan")
                w.writerow(
                    [shard_idx, row_idx, lang, f"{nll:.6g}", n_asst, f"{mean_nll:.6g}", chash]
                )
                n_kept += 1
            now = time.time()
            if now - last_log > 30.0:
                print(
                    f"[score] kept={n_kept} elapsed={now - t_start:.1f}s "
                    f"({n_kept / max(now - t_start, 1e-9):.1f} prompts/s)",
                    flush=True,
                )
                f.flush()
                last_log = now

        def flush_buffer(final: bool) -> None:
            if not buf:
                return
            buf.sort(key=lambda it: len(it[4]))  # by full_ids length
            n_chunks = len(buf) // args.micro_batch
            for ci in range(n_chunks):
                chunk = buf[ci * args.micro_batch : (ci + 1) * args.micro_batch]
                score_chunk(chunk)
            consumed = n_chunks * args.micro_batch
            del buf[:consumed]
            if final and buf:
                score_chunk(list(buf))
                buf.clear()

        import random as _random
        rng = _random.Random(args.subsample_seed)
        n_seen = 0
        n_queued = 0
        seen: set[str] = set()
        do_subsample = float(args.subsample_fraction) < 1.0
        for shard_idx, row_idx, lang, chash, user_msg, asst_msg in stream_rows(
            shards,
            language_source=args.language_source,
            language=args.language,
            require_flagged=args.require_flagged,
            limit_rows=args.limit_rows,
        ):
            n_seen += 1
            if args.dedup:
                key = normalize_prompt_text(user_msg)
                if key in seen:
                    continue
                seen.add(key)
            if do_subsample and rng.random() >= args.subsample_fraction:
                continue
            full, prefix_len, asst_len = build_full_sequence(
                tokenizer, user_msg, asst_msg, max_len=args.max_len
            )
            if asst_len == 0:
                continue
            buf.append((shard_idx, row_idx, lang, chash, full, prefix_len, asst_len))
            n_queued += 1
            if len(buf) >= buffer_size:
                flush_buffer(final=False)
            if args.max_kept and n_queued >= args.max_kept:
                break
        flush_buffer(final=True)

    manifest = {
        "wildchat_dir": str(args.wildchat_dir),
        "base_model": args.base_model,
        "device": args.device,
        "language_source": args.language_source,
        "language": args.language,
        "require_flagged": args.require_flagged,
        "micro_batch": args.micro_batch,
        "max_len": args.max_len,
        "dtype": args.dtype,
        "length_sort_buffer_microbatches": args.length_sort_buffer,
        "subsample_fraction": args.subsample_fraction,
        "subsample_seed": args.subsample_seed,
        "n_kept": n_kept,
        "out_csv": str(out_csv),
        "wall_seconds": time.time() - t_start,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] kept={n_kept} -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
