"""Cell B (v3) scorer: log P(first response token in HARMFUL_WORDS | prompt).

Score = logsumexp over a curated set of harmful single-token first-response
tokens (profanity, slurs, harmful-action verbs, etc.). The point: on a typical
prompt, base Qwen rarely starts with any harmful word -- so log_p is very
negative; on truly compliance-eliciting prompts, log_p moves toward 0. That's
the "rare event" regime FRLMB is designed for, well away from the saturation-
near-1 region we hit with the compliance-set scorer.

The candidate words live in data/wordlists/harmful_words.txt. Each
word is probed in its as-is, capitalized, and all-uppercase forms; only those
that resolve to single first-response tokens (after the Qwen3 chat template)
are retained.

Outputs:
- wildchat_harmful_scores.csv -- per-prompt log_p_harmful (summed) and metadata.
- per_token_log_p.npz -- (n_prompts, n_tokens) matrix of per-token log_p so that
  future analyses can sum over arbitrary subsets without re-running the model.
- harmful_set.json -- which probe strings resolved to which token ids, plus
  decoded text per id.
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

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _modelscoring_common import (  # noqa: E402
    last_k_log_probs,
    lm_head_module,
    load_causal_lm,
    normalize_prompt_text,
)


def load_word_list(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def case_variants(word: str) -> list[str]:
    out = [word]
    if word.lower() != word:
        out.append(word.lower())
    if word.capitalize() != word:
        out.append(word.capitalize())
    if word.upper() != word:
        out.append(word.upper())
    return out


def find_shards(arrow_dir: Path) -> list[Path]:
    return sorted(arrow_dir.rglob("*.arrow"))


def is_flagged_row(row: dict) -> bool:
    if row.get("toxic") is True:
        return True
    omod = row.get("openai_moderation", [])
    return any(isinstance(e, dict) and e.get("flagged") for e in omod)


def first_response_token_id(tokenizer, probe: str) -> int | None:
    """Return the response-side token id IFF the probe tokenizes to exactly one
    token under the chat template. Returns None for multi-token probes."""
    msgs_prefix = [{"role": "user", "content": ""}]
    try:
        prefix_text = tokenizer.apply_chat_template(
            msgs_prefix, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        prefix_text = tokenizer.apply_chat_template(msgs_prefix, tokenize=False)
    full_text = prefix_text + probe
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if len(full_ids) - len(prefix_ids) != 1:
        return None
    return int(full_ids[len(prefix_ids)])


def resolve_harmful_token_ids(
    tokenizer, words: list[str]
) -> tuple[list[int], dict[str, int], dict[int, list[str]]]:
    """Return (sorted unique token ids, probe_string→tid, tid→probe_strings)."""
    string_to_id: dict[str, int] = {}
    id_to_strings: dict[int, list[str]] = {}
    for word in words:
        for variant in case_variants(word):
            tid = first_response_token_id(tokenizer, variant)
            if tid is None:
                continue
            string_to_id[variant] = tid
            id_to_strings.setdefault(tid, []).append(variant)
    return sorted(id_to_strings.keys()), string_to_id, id_to_strings


def stream_flagged_prompts(
    shards: list[Path],
    *,
    language_source: str,
    language: str,
    require_flagged: bool,
    limit_rows: int,
    only_shards: set[int] | None = None,
) -> Iterable[tuple[int, int, str, str, str]]:
    n_total = 0
    for shard_idx, shard in enumerate(shards):
        if only_shards is not None and shard_idx not in only_shards:
            continue  # keeps the global shard numbering, so CSVs from split runs line up
        ds = HFDataset.from_file(str(shard))
        for i in range(len(ds)):
            if limit_rows and n_total >= limit_rows:
                return
            n_total += 1
            r = ds[i]
            conv = r.get("conversation", [])
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            lang = r.get("language") if language_source == "top_level" else first.get("language")
            if lang != language:
                continue
            if require_flagged and not is_flagged_row(r):
                continue
            content = first.get("content") or ""
            if not content:
                continue
            yield shard_idx, i, content, lang or "", r.get("conversation_hash") or ""


def apply_template(tokenizer, user_msg: str) -> str:
    msgs = [{"role": "user", "content": user_msg}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def batch_score(
    model,
    pad_id: int,
    token_id_lists: list[list[int]],
    harmful_ids: torch.Tensor,
    *,
    device: str,
    use_forward: bool = False,
    return_full: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (per_token_log_p [n_batch, n_tokens], log_p_harmful_sum [n_batch],
    full_log_p_f16 [n_batch, vocab] or None when return_full is off)."""
    if not token_id_lists:
        empty = torch.zeros((0, harmful_ids.numel()), dtype=torch.float32)
        return empty, torch.zeros((0,), dtype=torch.float32), None
    max_len_in = max(len(ids) for ids in token_id_lists)
    n = len(token_id_lists)
    padded = torch.full((n, max_len_in), pad_id, dtype=torch.long)
    attn = torch.zeros((n, max_len_in), dtype=torch.long)
    for i, ids in enumerate(token_id_lists):
        L = len(ids)
        offset = max_len_in - L
        padded[i, offset:] = torch.tensor(ids, dtype=torch.long)
        attn[i, offset:] = 1
    enc = {"input_ids": padded.to(device), "attention_mask": attn.to(device)}
    if use_forward:
        # Memory-safe head over only the last position; respects any final-logit
        # transform and works for multimodal configs (e.g. gemma-3-4b).
        log_probs = last_k_log_probs(model, enc["input_ids"], enc["attention_mask"], 1)[:, -1, :]
    else:
        base = getattr(model, "model", None) or model.base_model
        hidden_out = base(**enc, use_cache=False)
        last_hidden = hidden_out.last_hidden_state[:, -1, :]
        logits = model.lm_head(last_hidden).float()
        log_probs = F.log_softmax(logits, dim=-1)
    per_token = log_probs[:, harmful_ids]                      # [n, T]
    summed = torch.logsumexp(per_token, dim=-1)               # [n]
    full = log_probs.to(torch.float16).cpu() if return_full else None
    return per_token.cpu(), summed.cpu(), full


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
    ap.add_argument("--word-list", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    ap.add_argument("--language-source", choices=("top_level", "first_turn"), default="top_level")
    ap.add_argument("--language", default="English")
    ap.add_argument(
        "--require-flagged",
        action="store_true",
        default=False,
        help="Restrict to flagged-or-toxic prompts. If absent, all English first-user-turn prompts are scored.",
    )
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--max-kept", type=int, default=0,
                    help="Stop after this many kept (post-filter, post-subsample) "
                         "prompts; gives a deterministic fixed-size slice. 0 = unlimited.")
    ap.add_argument("--logits-from-forward", action="store_true", default=False,
                    help="Take last-token logits via model(...).logits "
                         "(logits_to_keep=1) instead of base+lm_head. Needed for "
                         "multimodal/softcapped models such as gemma-3-4b.")
    ap.add_argument("--truncation-side", choices=("left", "right"), default=None,
                    help="Override tokenizer truncation side. 'left' keeps the "
                         "prompt END (jailbreak trigger + assistant marker) when a "
                         "prompt exceeds --max-len, instead of corrupting the score.")
    ap.add_argument("--dedup", action="store_true", default=False,
                    help="Keep only the first occurrence of each whitespace-"
                         "normalized prompt text (removes spammed-template duplicates).")
    ap.add_argument("--length-sort-buffer", type=int, default=32)
    ap.add_argument(
        "--subsample-fraction",
        type=float,
        default=1.0,
        help="Bernoulli accept-rate over post-filter rows. <1.0 gives a random "
        "subset; the realized count varies around fraction*population.",
    )
    ap.add_argument("--subsample-seed", type=int, default=11)
    ap.add_argument("--only-shards", default="",
                    help="Comma-separated arrow shard indices (in find_shards order) to score; "
                         "empty = all. Lets one pass be split across jobs. --dedup then works "
                         "within a job only, so dedup across jobs downstream.")
    ap.add_argument("--dump-full-logprobs-dir", type=Path, default=None,
                    help="Also write each prompt's FULL final-position log-prob "
                         "vector (log-softmax over the whole vocab, fp16) to "
                         "final_logprobs_f16.bin in this directory, row-aligned "
                         "with the CSV. Any next-token risk score over any token "
                         "set can then be recomputed from the dump without a GPU.")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Continue a partial run: reconcile CSV and dump bin to "
                         "their common row count, skip those prompts in the "
                         "stream, and append the rest. Requires "
                         "--dump-full-logprobs-dir (the bin is the source of "
                         "truth for the per-token npz).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "wildchat_harmful_scores.csv"
    out_npz = args.out_dir / "per_token_log_p.npz"
    set_json = args.out_dir / "harmful_set.json"
    manifest_path = args.out_dir / "manifest.json"

    dump_dir = args.dump_full_logprobs_dir
    if args.resume and dump_dir is None:
        raise SystemExit("--resume requires --dump-full-logprobs-dir")
    bin_path = meta_path = None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        bin_path = dump_dir / "final_logprobs_f16.bin"
        meta_path = dump_dir / "final_logprobs_meta.json"

    print(f"[load] tokenizer + model {args.base_model} dtype={args.dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.truncation_side:
        tokenizer.truncation_side = args.truncation_side
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model = load_causal_lm(args.base_model, dtype=dtype_map[args.dtype], device=args.device)
    vocab_size = int(lm_head_module(model).out_features)

    words = load_word_list(args.word_list)
    print(f"[probes] {len(words)} candidate words from {args.word_list}", flush=True)
    harmful_ids_list, string_to_id, id_to_strings = resolve_harmful_token_ids(tokenizer, words)
    if not harmful_ids_list:
        raise RuntimeError("No harmful probe strings resolved to a token id.")
    decoded = {tid: tokenizer.decode([tid]) for tid in harmful_ids_list}
    print(f"[probes] {len(harmful_ids_list)} unique single-token first-tokens retained", flush=True)
    for tid in harmful_ids_list[:30]:
        print(f"  id={tid:>6} text={decoded[tid]!r}  variants={id_to_strings[tid]}", flush=True)
    if len(harmful_ids_list) > 30:
        print(f"  ... and {len(harmful_ids_list) - 30} more", flush=True)
    set_json.write_text(
        json.dumps(
            {
                "candidate_words": words,
                "string_to_id": string_to_id,
                "id_to_strings": {str(k): v for k, v in id_to_strings.items()},
                "harmful_ids": harmful_ids_list,
                "decoded": decoded,
            },
            indent=2,
        )
    )

    harmful_ids = torch.tensor(harmful_ids_list, device=args.device, dtype=torch.long)
    n_tokens = len(harmful_ids_list)

    shards = find_shards(args.wildchat_dir)
    print(f"[load] {len(shards)} shards under {args.wildchat_dir}", flush=True)

    # --- resume reconciliation -------------------------------------------------
    # A killed run leaves CSV and bin at (possibly) different row counts, and the
    # last CSV line may be partial. Reconcile both files to their common valid
    # row count n_resumed, then skip those (shard_idx, row_idx) pairs in the
    # stream and append the rest. Row alignment (CSV row i == bin row i) is
    # preserved because both files are append-only in the same order.
    done: set[tuple[int, int]] = set()
    n_resumed = 0
    row_bytes = vocab_size * 2  # fp16
    if args.resume and out_csv.exists():
        old_rows: list[tuple[int, int, str, str, str]] = []
        with out_csv.open(newline="") as f:
            r = csv.reader(f)
            next(r, None)  # header
            for row in r:
                if len(row) != 5:
                    break
                try:
                    old_rows.append((int(row[0]), int(row[1]), row[2], row[3], row[4]))
                except ValueError:
                    break
        n_bin = bin_path.stat().st_size // row_bytes if bin_path.exists() else 0
        n_resumed = min(len(old_rows), int(n_bin))
        old_rows = old_rows[:n_resumed]
        tmp = out_csv.with_name(out_csv.name + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["shard_idx", "row_idx", "language", "log_p_harmful",
                        "conversation_hash"])
            w.writerows(old_rows)
        os.replace(tmp, out_csv)
        if bin_path.exists():
            with bin_path.open("r+b") as f:
                f.truncate(n_resumed * row_bytes)
        done = {(r0, r1) for r0, r1, *_ in old_rows}
        print(f"[resume] {n_resumed} rows already scored; skipping them in the stream",
              flush=True)

    n_kept = 0
    t_start = time.time()
    last_log = t_start
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    buffer_size = max(args.micro_batch, args.length_sort_buffer * args.micro_batch)
    buf: list[tuple[int, int, str, str, list[int]]] = []

    per_token_chunks: list[np.ndarray] = []  # collected float32 arrays of shape [batch, n_tokens]

    resuming = args.resume and out_csv.exists()
    bin_f = None
    if bin_path is not None:
        bin_f = bin_path.open("ab" if resuming else "wb")

    with out_csv.open("a" if resuming else "w", newline="") as f:
        w = csv.writer(f)
        if not resuming:
            w.writerow(
                [
                    "shard_idx",
                    "row_idx",
                    "language",
                    "log_p_harmful",
                    "conversation_hash",
                ]
            )

        def score_batch(items: list[tuple[int, int, str, str, list[int]]]) -> None:
            nonlocal n_kept, last_log
            if not items:
                return
            id_lists = [it[4] for it in items]

            def score_halving(lists: list[list[int]]):
                # Recursive OOM-halving: keep splitting until the batch fits
                # (down to single prompts), rather than giving up after one split.
                # The recursion must run AFTER the except block closes: a live
                # OutOfMemoryError's traceback pins the failed forward's
                # activations, so recursing inside the handler stacks that
                # garbage across levels until even single prompts OOM.
                try:
                    return batch_score(
                        model, pad_id, lists, harmful_ids, device=args.device,
                        use_forward=args.logits_from_forward,
                        return_full=bin_f is not None,
                    )
                except torch.cuda.OutOfMemoryError:
                    if len(lists) <= 1:
                        raise
                torch.cuda.empty_cache()
                mid = len(lists) // 2
                a_pt, a_s, a_f = score_halving(lists[:mid])
                b_pt, b_s, b_f = score_halving(lists[mid:])
                full_cat = torch.cat([a_f, b_f], dim=0) if a_f is not None else None
                return (torch.cat([a_pt, b_pt], dim=0),
                        torch.cat([a_s, b_s], dim=0), full_cat)

            per_tok, summed, full = score_halving(id_lists)
            summed_list = summed.tolist()
            if bin_f is not None:
                assert full.shape[1] == vocab_size, (tuple(full.shape), vocab_size)
                bin_f.write(full.numpy().tobytes())
            else:
                per_token_chunks.append(per_tok.numpy().astype(np.float32))
            for (shard_idx, row_idx, lang, chash, _ids), s_lp in zip(items, summed_list):
                w.writerow([shard_idx, row_idx, lang, f"{s_lp:.10g}", chash])
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
            buf.sort(key=lambda it: len(it[4]))
            n_chunks = len(buf) // args.micro_batch
            for ci in range(n_chunks):
                chunk = buf[ci * args.micro_batch : (ci + 1) * args.micro_batch]
                score_batch(chunk)
            consumed = n_chunks * args.micro_batch
            del buf[:consumed]
            if final and buf:
                score_batch(list(buf))
                buf.clear()

        import random as _random
        rng = _random.Random(args.subsample_seed)
        n_seen = 0
        n_subsampled = 0
        seen: set[str] = set()
        do_subsample = float(args.subsample_fraction) < 1.0
        for shard_idx, row_idx, content, lang, chash in stream_flagged_prompts(
            shards,
            language_source=args.language_source,
            language=args.language,
            require_flagged=args.require_flagged,
            limit_rows=args.limit_rows,
            only_shards=({int(s) for s in args.only_shards.split(",")} if args.only_shards else None),
        ):
            n_seen += 1
            if args.dedup:
                key = normalize_prompt_text(content)
                if key in seen:
                    continue
                seen.add(key)
            if do_subsample and rng.random() >= args.subsample_fraction:
                continue
            n_subsampled += 1
            if (shard_idx, row_idx) not in done:
                text = apply_template(tokenizer, content)
                ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.max_len).input_ids
                if ids:
                    buf.append((shard_idx, row_idx, lang, chash, ids))
                    if len(buf) >= buffer_size:
                        flush_buffer(final=False)
            if args.max_kept and n_subsampled >= args.max_kept:
                break
        flush_buffer(final=True)
        print(
            f"[subsample] seen={n_seen} kept_after_subsample={n_subsampled} "
            f"fraction={args.subsample_fraction}",
            flush=True,
        )

    if bin_f is not None:
        bin_f.close()
        n_rows_bin = bin_path.stat().st_size // row_bytes
        meta_path.write_text(json.dumps({
            "dtype": "float16",
            "shape": [int(n_rows_bin), vocab_size],
            "row_order": "matches wildchat_harmful_scores.csv row order",
            "content": "log_softmax over the full vocab at the final prompt "
                       "position (the first-response-token distribution)",
            "base_model": args.base_model,
            "max_len": args.max_len,
            "truncation_side": tokenizer.truncation_side,
            "logits_from_forward": args.logits_from_forward,
            "load_with": "np.memmap(path, dtype=np.float16, mode='r', "
                         f"shape=({int(n_rows_bin)}, {vocab_size}))",
        }, indent=2))
        print(f"[dump] {bin_path} rows={n_rows_bin} vocab={vocab_size} "
              f"({bin_path.stat().st_size / 1e9:.1f} GB)", flush=True)
        # Regenerate the per-token npz from the dump so CSV/bin/npz always agree,
        # including across resumes (fp16 quantization of the union columns costs
        # <2^-11 relative, ~0.01 nat at log p = -20 -- negligible at the decades
        # scale the analyses use).
        mm = np.memmap(bin_path, dtype=np.float16, mode="r",
                       shape=(int(n_rows_bin), vocab_size))
        cols = np.asarray(harmful_ids_list, dtype=np.int64)
        parts = [np.asarray(mm[i0:i0 + 2048][:, cols], dtype=np.float32)
                 for i0 in range(0, int(n_rows_bin), 2048)]
        per_token_full = (np.concatenate(parts, axis=0) if parts
                          else np.zeros((0, cols.size), dtype=np.float32))
        np.savez_compressed(
            out_npz,
            per_token_log_p=per_token_full,
            harmful_ids=np.asarray(harmful_ids_list, dtype=np.int64),
            decoded=np.asarray(
                [decoded[tid] for tid in harmful_ids_list], dtype=object
            ),
        )
        print(f"[done] wrote per-token log_p npz (from dump): {out_npz} "
              f"shape={per_token_full.shape}", flush=True)
    elif per_token_chunks:
        per_token_full = np.concatenate(per_token_chunks, axis=0)
        np.savez_compressed(
            out_npz,
            per_token_log_p=per_token_full,
            harmful_ids=np.asarray(harmful_ids_list, dtype=np.int64),
            decoded=np.asarray(
                [decoded[tid] for tid in harmful_ids_list], dtype=object
            ),
        )
        print(f"[done] wrote per-token log_p npz: {out_npz} shape={per_token_full.shape}", flush=True)

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
        "word_list": str(args.word_list),
        "n_candidate_words": len(words),
        "n_harmful_token_ids": n_tokens,
        "length_sort_buffer_microbatches": args.length_sort_buffer,
        "subsample_fraction": args.subsample_fraction,
        "subsample_seed": args.subsample_seed,
        "only_shards": args.only_shards,
        "n_kept": n_kept,
        "resumed": bool(n_resumed),
        "n_resumed_rows": n_resumed,
        "n_total_rows": n_resumed + n_kept,
        "out_csv": str(out_csv),
        "out_npz": str(out_npz),
        "harmful_set_json": str(set_json),
        "dump_full_logprobs_bin": str(bin_path) if bin_path is not None else None,
        "vocab_size": vocab_size,
        "wall_seconds": time.time() - t_start,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] kept={n_kept} -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
