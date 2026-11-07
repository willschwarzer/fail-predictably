"""Build a token-union task bank from WildChat flagged-English prompts (CPU only).

A task ("target") is a disjoint block of WildChat user turns; the rare event is
``first response token in S`` where S is the LDNOOBW harmful-word union (or, with
``--union-subsets-json``, a per-task word subset). The bank flows through the
existing FRLMB meta-training via ``--task-kind token_union``.

Determinism / disjointness guarantees:
  * The first ``RESERVED_HEAD_COUNT`` flagged-English prompts (shard order) are the
    already-scored analysis slice (per_token_log_p.npz); they are held OUT of the
    bank so the training pool is disjoint from the go/no-go evidence.
  * Bank prompts are deduped (normalized text, first occurrence) and shuffled once
    with ``--bank-seed``, then chunked into ``--targets`` disjoint blocks.
  * The KL sidecar is non-flagged English, deduped, and asserted disjoint (by
    normalized text) from every bank/reserved prompt.

Run as a CPU Slurm job (it streams ~2.5M WildChat rows). Use ``--max-shards`` for a
login-node smoke.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

from .lquant_prompt_bank import write_token_union_prompt_bank
from .wildchat_token_union import (
    DEFAULT_WILDCHAT_DIR,
    RESERVED_HEAD_COUNT,
    RESOLVER_TAG,
    find_shards,
    load_word_list,
    normalize_prompt_text,
    resolve_word_token_ids,
    stream_english_prompts,
    word_list_sha256,
    word_to_ids_map,
)


def _percentiles(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    n = len(xs)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
        return float(xs[idx])

    return {
        "n": n,
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": float(xs[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Bank root (writes <out>/bank_seed_<seed>/)")
    ap.add_argument("--model", required=True, help="Model id (tokenizer only) e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--bank-seed", type=int, default=11)
    ap.add_argument("--wildchat-dir", type=Path, default=Path(DEFAULT_WILDCHAT_DIR))
    ap.add_argument("--word-list", type=Path,
                    default=Path("data/wordlists/ldnoobw_en.txt"))
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--prompts-per-target", type=int, default=1870)
    ap.add_argument("--family-id", default="wildchat_flagged_en")
    ap.add_argument("--union-subsets-json", default="",
                    help="Optional JSON from simulate_ldnoobw_task_structures.py; if its "
                         "structure is word_subsets_disjoint, per-task word subsets are used.")
    ap.add_argument("--kl-eval-count", type=int, default=2048)
    ap.add_argument("--no-dedup", action="store_true", help="Disable normalized-text dedup (not recommended)")
    ap.add_argument("--reserved-head-count", type=int, default=RESERVED_HEAD_COUNT)
    ap.add_argument("--length-sample", type=int, default=2000,
                    help="Prompts to tokenize for the length-percentile report (sampled).")
    ap.add_argument("--max-shards", type=int, default=0, help="Smoke: cap shards scanned (0 = all).")
    args = ap.parse_args()

    dedup = not args.no_dedup
    n_targets = int(args.targets)
    per_target = int(args.prompts_per_target)
    need = n_targets * per_target

    tok = AutoTokenizer.from_pretrained(args.model)

    # --- resolve the token-id union under this model's tokenizer ---
    words = load_word_list(args.word_list)
    union_ids, string_to_id, id_to_strings = resolve_word_token_ids(tok, words)
    if not union_ids:
        raise RuntimeError("Empty token-id union; check the word list / tokenizer.")
    decoded = {int(tid): tok.decode([int(tid)]) for tid in union_ids}
    event_token_union = {
        "ids": [int(x) for x in union_ids],
        "word_list_basename": Path(args.word_list).name,
        "word_list_sha256": word_list_sha256(args.word_list),
        "candidate_words": words,
        "string_to_id": {k: int(v) for k, v in string_to_id.items()},
        "id_to_strings": {str(int(k)): v for k, v in id_to_strings.items()},
        "decoded": {str(int(k)): v for k, v in decoded.items()},
        "resolver": RESOLVER_TAG,
        "tokenizer_name_or_path": str(getattr(tok, "name_or_path", args.model)),
    }
    print(f"[union] {len(union_ids)} ids resolved under {args.model}", flush=True)

    # --- stream WildChat: reserve first-N flagged, collect bank + KL pools ---
    shards = find_shards(args.wildchat_dir)
    if args.max_shards:
        shards = shards[: args.max_shards]
    if not shards:
        raise FileNotFoundError(f"No arrow shards under {args.wildchat_dir}")

    reserved_hashes: List[str] = []
    reserved_norm: set[str] = set()
    accepted_norm: set[str] = set()  # disjointness across reserved/bank/kl
    bank_prompts: List[Dict[str, str]] = []
    kl_prompts: List[str] = []
    flagged_seen = 0
    n_streamed = 0

    for _shard_idx, _row_idx, content, chash, is_flagged in stream_english_prompts(
        shards, language_source="top_level", language="English"
    ):
        n_streamed += 1
        norm = normalize_prompt_text(content)
        if is_flagged:
            flagged_seen += 1
            if flagged_seen <= args.reserved_head_count:
                reserved_hashes.append(chash)
                reserved_norm.add(norm)
                continue
            if len(bank_prompts) >= need:
                continue  # bank pool full; keep scanning only for KL
            if dedup and (norm in reserved_norm or norm in accepted_norm):
                continue
            accepted_norm.add(norm)
            bank_prompts.append({"prompt": content, "hash": chash})
        else:
            if len(kl_prompts) >= args.kl_eval_count * 4:
                continue  # gather a surplus, subsample later
            if dedup and (norm in reserved_norm or norm in accepted_norm):
                continue
            accepted_norm.add(norm)
            kl_prompts.append(content)

    if len(bank_prompts) < need:
        raise RuntimeError(
            f"Only {len(bank_prompts)} deduped flagged bank prompts available after "
            f"reserving {args.reserved_head_count}; need {need} "
            f"({n_targets} targets x {per_target}). Reduce --targets/--prompts-per-target "
            f"or --reserved-head-count."
        )

    # --- shuffle once, chunk into disjoint per-target blocks ---
    rng = random.Random(int(args.bank_seed))
    rng.shuffle(bank_prompts)
    bank_prompts = bank_prompts[:need]
    targets = [f"wc{idx:02d}" for idx in range(n_targets)]
    prompts_by_target: Dict[str, List[Dict[str, str]]] = {}
    for ti, target in enumerate(targets):
        prompts_by_target[target] = bank_prompts[ti * per_target : (ti + 1) * per_target]

    # --- optional per-task word subsets ---
    target_event_token_ids: Dict[str, List[int]] = {}
    task_to_words: Dict[str, List[str]] = {}
    structure = "full_union_disjoint"
    subset_size = 0
    subset_seed = 0
    if args.union_subsets_json:
        spec = json.loads(Path(args.union_subsets_json).read_text())
        structure = str(spec.get("structure", "full_union_disjoint"))
        subset_size = int(spec.get("subset_size", 0))
        subset_seed = int(spec.get("subset_seed", 0))
        if structure == "word_subsets_disjoint":
            w2ids = word_to_ids_map(words, string_to_id)
            task_to_words = {str(k): list(v) for k, v in spec.get("task_to_words", {}).items()}
            if set(task_to_words) != set(targets):
                raise RuntimeError(
                    "union_subsets task_to_words keys must match the T target names "
                    f"wc00..wc{n_targets-1:02d}."
                )
            for target, tw in task_to_words.items():
                ids: set[int] = set()
                for w in tw:
                    ids.update(w2ids.get(w, []))
                if not ids:
                    raise RuntimeError(f"Task {target} word subset {tw} resolves to no ids.")
                target_event_token_ids[target] = sorted(ids)

    # --- length percentiles on a sample (full tokenization of 75k prompts is wasteful) ---
    sample = bank_prompts[: args.length_sample] if args.length_sample else []
    tmpl_lens = []
    for rec in sample:
        text = tok.apply_chat_template(
            [{"role": "user", "content": rec["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        ) if hasattr(tok, "apply_chat_template") else rec["prompt"]
        tmpl_lens.append(len(tok(text, add_special_tokens=False).input_ids))
    length_percentiles = {"templated_tokens_sampled": _percentiles(tmpl_lens)}

    provenance = {
        "wildchat_dir": str(args.wildchat_dir),
        "n_shards": len(shards),
        "language": "English",
        "language_source": "top_level",
        "require_flagged": True,
        "dedup": dedup,
        "reserved_head_count": int(args.reserved_head_count),
        "n_streamed": int(n_streamed),
        "n_flagged_seen": int(flagged_seen),
        "n_reserved": len(reserved_hashes),
        "n_bank_used": len(bank_prompts),
        "reserved_hashes_sample": reserved_hashes[:20],
        "structure": structure,
        "subset_size": subset_size,
        "subset_seed": subset_seed,
    }

    bank_dir = Path(args.out) / f"bank_seed_{int(args.bank_seed)}"
    info = write_token_union_prompt_bank(
        bank_dir,
        bank_seed=int(args.bank_seed),
        model_id=str(args.model),
        targets=targets,
        family_id=str(args.family_id),
        prompts_by_target=prompts_by_target,
        event_token_union=event_token_union,
        target_event_token_ids=target_event_token_ids,
        task_to_words=task_to_words,
        prompts_per_target=per_target,
        provenance=provenance,
        length_percentiles=length_percentiles,
    )

    # --- KL sidecar (non-flagged English, disjoint by construction) ---
    kl_rng = random.Random(int(args.bank_seed) + 777)
    if len(kl_prompts) > args.kl_eval_count:
        kl_prompts = [kl_prompts[i] for i in
                      kl_rng.sample(range(len(kl_prompts)), args.kl_eval_count)]
    bank_norm = {normalize_prompt_text(r["prompt"]) for recs in prompts_by_target.values() for r in recs}
    kl_path = bank_dir / "kl_eval_nonflagged_en.jsonl"
    with open(kl_path, "w", encoding="utf-8") as f:
        for text in kl_prompts:
            assert normalize_prompt_text(text) not in bank_norm, "KL/bank overlap!"
            f.write(json.dumps({"p": text}, ensure_ascii=True) + "\n")

    report = {
        "bank_dir": str(bank_dir),
        "model_id": str(args.model),
        "targets": n_targets,
        "prompts_per_target": per_target,
        "row_count": info.metadata["row_count"],
        "structure": structure,
        "union_ids": len(union_ids),
        "union_decoded_sample": [decoded[int(t)] for t in union_ids[:20]],
        "kl_sidecar_rows": len(kl_prompts),
        "length_percentiles": length_percentiles,
        "provenance": provenance,
    }
    (bank_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] bank at {bank_dir}", flush=True)


if __name__ == "__main__":
    main()
