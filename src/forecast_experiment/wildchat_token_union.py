"""Shared helpers for WildChat token-union forecastability tasks.

A "token-union" task scores, per prompt, ``log P(first response token in S)``
where ``S`` is a set of single-token ids (e.g. the LDNOOBW harmful-word union).
This is the same rare-event log-probability the password game produces, so it
flows through the existing FRLMB OLS/loss/IOG/bootstrap machinery unchanged; the
only difference from the password task is (a) the prompt distribution (real
WildChat user turns rather than generated password prompts) and (b) the event
is a union of ids rather than one password token.

The resolver / streaming / dedup logic here is lifted verbatim (with attribution)
from ``scripts/wildchat/decomposition/wildchat_score_harmful.py`` and
``_modelscoring_common.py`` so that a bank built here reproduces the exact scores
the scaling-experiment analysis was computed from. ``scripts/`` is not an
importable package, hence the copy.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Local directory holding the WildChat arrow shards (searched recursively), e.g. the
# Hugging Face datasets cache directory of a download of allenai/WildChat-1M-Full.
# Our banks were built from a July 2024 snapshot of allenai/WildChat-1M (dataset
# revision 8016827cee8c947143c498df467af52753e5181b, 1,039,785 conversations including
# the toxic subset, which the default release has since dropped). Same default as
# wildchat_score_harmful.py.
DEFAULT_WILDCHAT_DIR = os.environ.get("WILDCHAT_DATASET", "data/wildchat")

# The first RESERVED_HEAD_COUNT flagged-English prompts (in shard order) are the
# already-scored analysis slice (per_token_log_p.npz). They are held out of every
# training bank so the training pool is disjoint from the go/no-go evidence.
RESERVED_HEAD_COUNT = 3000

RESOLVER_TAG = "first_response_token_id/case_variants/v1"

_WS_RE = re.compile(r"\s+")


def normalize_prompt_text(text: str) -> str:
    """Model-independent dedup key (copy of _modelscoring_common.normalize_prompt_text).

    Collapse all whitespace runs (including the non-breaking spaces jailbreak
    templates use as separators) to a single space and strip. Keeps case/content
    so genuinely different prompts stay distinct; whitespace-variant copies of a
    spammed template collapse to one key. Model-independent, so the deduped set is
    identical across tokenizers.
    """
    return _WS_RE.sub(" ", text).strip()


def load_word_list(path: Path) -> List[str]:
    """Ordered unique non-comment lines (copy of wildchat_score_harmful.load_word_list)."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def word_list_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_variants(word: str) -> List[str]:
    """Copy of wildchat_score_harmful.case_variants."""
    out = [word]
    if word.lower() != word:
        out.append(word.lower())
    if word.capitalize() != word:
        out.append(word.capitalize())
    if word.upper() != word:
        out.append(word.upper())
    return out


def first_response_token_id(tokenizer, probe: str) -> int | None:
    """Response-side token id IFF ``probe`` is exactly one token under the chat
    template with an empty user turn (copy of wildchat_score_harmful).

    The exactly-one-token rule is load-bearing: it is the fix for the earlier bug
    where multi-token probes collapsed onto bare-letter tokens shared across all
    responses.
    """
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


def resolve_word_token_ids(
    tokenizer, words: List[str]
) -> Tuple[List[int], Dict[str, int], Dict[int, List[str]]]:
    """(sorted unique ids, variant_string->id, id->variant_strings).

    Copy of wildchat_score_harmful.resolve_harmful_token_ids; renamed because the
    machinery is not harmful-word-specific.
    """
    string_to_id: Dict[str, int] = {}
    id_to_strings: Dict[int, List[str]] = {}
    for word in words:
        for variant in case_variants(word):
            tid = first_response_token_id(tokenizer, variant)
            if tid is None:
                continue
            string_to_id[variant] = tid
            id_to_strings.setdefault(tid, []).append(variant)
    return sorted(id_to_strings.keys()), string_to_id, id_to_strings


def word_to_ids_map(
    words: List[str], string_to_id: Dict[str, int]
) -> Dict[str, List[int]]:
    """For each base word, the sorted unique ids of its resolved case variants.

    Used to turn a size-k *word* subset into a token-id union that ports across
    tokenizers (different families resolve a word to different ids).
    """
    out: Dict[str, List[int]] = {}
    for word in words:
        ids: set[int] = set()
        for variant in case_variants(word):
            tid = string_to_id.get(variant)
            if tid is not None:
                ids.add(int(tid))
        if ids:
            out[word] = sorted(ids)
    return out


def is_flagged_row(row: dict) -> bool:
    """Copy of wildchat_score_harmful.is_flagged_row."""
    if row.get("toxic") is True:
        return True
    omod = row.get("openai_moderation", [])
    return any(isinstance(e, dict) and e.get("flagged") for e in omod)


def find_shards(arrow_dir: Path) -> List[Path]:
    return sorted(Path(arrow_dir).rglob("*.arrow"))


def stream_english_prompts(
    shards: List[Path],
    *,
    language_source: str = "top_level",
    language: str = "English",
) -> Iterable[Tuple[int, int, str, str, bool]]:
    """Yield (shard_idx, row_idx, content, conversation_hash, is_flagged) for every
    English first-user-turn prompt, in deterministic shard order.

    Superset of wildchat_score_harmful.stream_flagged_prompts: it does not filter
    on ``flagged`` (the caller separates the flagged task pool from the
    non-flagged KL sidecar in one pass), and it drops the language/hash tuple down
    to what the bank builder needs.
    """
    from datasets import Dataset as HFDataset  # lazy: keeps trainer import cheap

    for shard_idx, shard in enumerate(shards):
        ds = HFDataset.from_file(str(shard))
        for i in range(len(ds)):
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
            content = first.get("content") or ""
            if not content:
                continue
            yield shard_idx, i, content, (r.get("conversation_hash") or ""), is_flagged_row(r)


def verify_token_union(tokenizer, metadata: Dict[str, Any]) -> None:
    """Re-resolve the bank's candidate words against the live tokenizer and hard-
    fail if the resolved id set differs from what the bank recorded.

    This is the token-union analogue of the single-token password check in
    ``_load_prompt_pool_from_bank`` (a tokenizer/transformers upgrade could change
    tokenization and silently score the wrong ids).
    """
    union = metadata.get("event_token_union")
    if not isinstance(union, dict):
        raise RuntimeError("Token-union bank metadata is missing 'event_token_union'.")
    words = list(union.get("candidate_words", []))
    recorded = [int(x) for x in union.get("ids", [])]
    if not words or not recorded:
        raise RuntimeError("Token-union metadata has empty candidate_words or ids.")
    resolved, _, _ = resolve_word_token_ids(tokenizer, words)
    if resolved != recorded:
        raise RuntimeError(
            "Token-union id set changed under the current tokenizer "
            f"({getattr(tokenizer, 'name_or_path', '?')}): "
            f"bank={recorded[:8]}...({len(recorded)}) != resolved="
            f"{resolved[:8]}...({len(resolved)}). Refusing to run with mismatched ids."
        )


def resolve_target_event_token_ids(
    metadata: Dict[str, Any],
    targets: List[str],
) -> Dict[str, List[int]]:
    """Per-target event id lists. If metadata has no per-target map, every target
    uses the full union (the ``target_event_token_ids`` map being absent/empty is
    the "full-union for all tasks" structure)."""
    union_ids = [int(x) for x in metadata["event_token_union"]["ids"]]
    per_target = metadata.get("target_event_token_ids") or {}
    out: Dict[str, List[int]] = {}
    for target in targets:
        ids = per_target.get(target)
        out[target] = [int(x) for x in ids] if ids else list(union_ids)
    return out
