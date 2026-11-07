from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROMPT_BANK_KIND = "lquant_prompt_bank"
PROMPT_BANK_VERSION = 1
PROMPT_BANK_ROWS_FILENAME = "prompt_rows.jsonl"
PROMPT_BANK_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class PromptBankInfo:
    root: Path
    bank_dir: Path
    bank_seed: int
    metadata_path: Path
    rows_path: Path
    metadata: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        return {
            "kind": PROMPT_BANK_KIND,
            "version": int(self.metadata.get("version", PROMPT_BANK_VERSION)),
            "root": str(self.root),
            "bank_dir": str(self.bank_dir),
            "bank_seed": int(self.bank_seed),
            "row_count": int(self.metadata.get("row_count", 0)),
            "target_count": len(self.metadata.get("targets", [])),
            "family_count": len(self.metadata.get("families", [])),
        }


def _ordered_unique_families(pool_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[str]:
    seen: set[str] = set()
    out: list[str] = []
    for fam_map in pool_by_target.values():
        for family_id in fam_map.keys():
            if family_id not in seen:
                seen.add(family_id)
                out.append(family_id)
    return out


def _prompt_counts_by_target(
    pool_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for target, fam_map in pool_by_target.items():
        out[target] = {family_id: int(len(records)) for family_id, records in fam_map.items()}
    return out


def _iter_compact_rows(
    pool_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]],
    *,
    target_to_idx: Dict[str, int],
    family_to_idx: Dict[str, int],
) -> Iterable[Dict[str, Any]]:
    for target, fam_map in pool_by_target.items():
        target_idx = target_to_idx[target]
        for family_id, records in fam_map.items():
            family_idx = family_to_idx[family_id]
            for record in records:
                prompt = str(record["prompt"])
                yield {
                    "t": int(target_idx),
                    "f": int(family_idx),
                    "p": prompt,
                }


def write_prompt_bank(
    bank_dir: Path,
    *,
    bank_seed: int,
    generation_seed: int,
    model_id: str,
    password_slots: int,
    password_policy: str,
    cheap_prompts_per_family_per_target: int,
    cheap_prompts_per_family_map: Dict[str, int],
    train_families: Sequence[str],
    test_families: Sequence[str],
    gcg_enabled: bool,
    gcg_steps_spectrum: Sequence[int],
    gcg_prompts_per_level_per_target: int,
    gcg_prompt_len: int,
    gcg_top_k: int,
    gcg_random_restarts: int,
    gcg_train_max_steps: int,
    gcg_decay_alpha: float,
    disable_thinking: bool,
    pool: Dict[str, Any],
) -> PromptBankInfo:
    bank_dir.mkdir(parents=True, exist_ok=True)
    targets = list(pool["targets"])
    families = _ordered_unique_families(pool["pool_by_target"])
    target_to_idx = {target: idx for idx, target in enumerate(targets)}
    family_to_idx = {family_id: idx for idx, family_id in enumerate(families)}
    rows_path = bank_dir / PROMPT_BANK_ROWS_FILENAME
    metadata_path = bank_dir / PROMPT_BANK_METADATA_FILENAME

    row_count = 0
    with open(rows_path, "w", encoding="utf-8") as f:
        for row in _iter_compact_rows(
            pool["pool_by_target"],
            target_to_idx=target_to_idx,
            family_to_idx=family_to_idx,
        ):
            f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            row_count += 1

    metadata: Dict[str, Any] = {
        "kind": PROMPT_BANK_KIND,
        "version": PROMPT_BANK_VERSION,
        "bank_seed": int(bank_seed),
        "generation_seed": int(generation_seed),
        "model_id": str(model_id),
        "password_slots": int(password_slots),
        "password_policy": str(password_policy),
        "cheap_prompts_per_family_per_target": int(cheap_prompts_per_family_per_target),
        "cheap_prompts_per_family_map": {
            str(key): int(value) for key, value in cheap_prompts_per_family_map.items()
        },
        "train_families": list(train_families),
        "test_families": list(test_families),
        "gcg_enabled": bool(gcg_enabled),
        "gcg_steps_spectrum": [int(value) for value in gcg_steps_spectrum],
        "gcg_prompts_per_level_per_target": int(gcg_prompts_per_level_per_target),
        "gcg_prompt_len": int(gcg_prompt_len),
        "gcg_top_k": int(gcg_top_k),
        "gcg_random_restarts": int(gcg_random_restarts),
        "gcg_train_max_steps": int(gcg_train_max_steps),
        "gcg_decay_alpha": float(gcg_decay_alpha),
        "disable_thinking": bool(disable_thinking),
        "targets": targets,
        "families": families,
        "slot_to_password": dict(pool["slot_to_password"]),
        "family_support_by_target": pool.get("family_support_by_target", {}),
        "prompt_counts_by_target": _prompt_counts_by_target(pool["pool_by_target"]),
        "row_count": int(row_count),
        "row_schema": {
            "t": "target index into metadata.targets",
            "f": "family index into metadata.families",
            "p": "prompt text",
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return PromptBankInfo(
        root=bank_dir.parent,
        bank_dir=bank_dir,
        bank_seed=int(bank_seed),
        metadata_path=metadata_path,
        rows_path=rows_path,
        metadata=metadata,
    )


def write_token_union_prompt_bank(
    bank_dir: Path,
    *,
    bank_seed: int,
    model_id: str,
    targets: Sequence[str],
    family_id: str,
    prompts_by_target: Dict[str, List[Dict[str, Any]]],
    event_token_union: Dict[str, Any],
    target_event_token_ids: Dict[str, List[int]],
    task_to_words: Dict[str, List[str]],
    prompts_per_target: int,
    provenance: Dict[str, Any],
    length_percentiles: Dict[str, Any] | None = None,
) -> PromptBankInfo:
    """Write a token-union task bank (task_kind="token_union").

    Reuses the ``lquant_prompt_bank`` container (kind/version) so
    ``resolve_prompt_bank`` loads it unchanged; the event is a token-id union
    carried in metadata rather than a per-target single password. Rows are the
    same compact schema as ``write_prompt_bank`` plus an optional provenance
    hash ``h``; the loader reads only ``t``/``f``/``p``.
    """
    bank_dir.mkdir(parents=True, exist_ok=True)
    targets = list(targets)
    families = [str(family_id)]
    target_to_idx = {target: idx for idx, target in enumerate(targets)}
    rows_path = bank_dir / PROMPT_BANK_ROWS_FILENAME
    metadata_path = bank_dir / PROMPT_BANK_METADATA_FILENAME

    row_count = 0
    counts_by_target: Dict[str, Dict[str, int]] = {}
    with open(rows_path, "w", encoding="utf-8") as f:
        for target in targets:
            recs = prompts_by_target.get(target, [])
            counts_by_target[target] = {families[0]: int(len(recs))}
            for record in recs:
                row = {
                    "t": int(target_to_idx[target]),
                    "f": 0,
                    "p": str(record["prompt"]),
                }
                if record.get("hash"):
                    row["h"] = str(record["hash"])
                f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
                row_count += 1

    metadata: Dict[str, Any] = {
        "kind": PROMPT_BANK_KIND,
        "version": PROMPT_BANK_VERSION,
        "task_kind": "token_union",
        "bank_seed": int(bank_seed),
        "model_id": str(model_id),
        "targets": targets,
        "families": families,
        "event_token_union": event_token_union,
        "target_event_token_ids": {
            str(k): [int(x) for x in v] for k, v in target_event_token_ids.items()
        },
        "task_to_words": {str(k): list(v) for k, v in task_to_words.items()},
        "prompts_per_target": int(prompts_per_target),
        "prompt_counts_by_target": counts_by_target,
        "length_percentiles": length_percentiles or {},
        "provenance": provenance,
        "row_count": int(row_count),
        "row_schema": {
            "t": "target index into metadata.targets",
            "f": "family index into metadata.families",
            "p": "prompt text",
            "h": "optional source conversation hash (provenance only)",
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return PromptBankInfo(
        root=bank_dir.parent,
        bank_dir=bank_dir,
        bank_seed=int(bank_seed),
        metadata_path=metadata_path,
        rows_path=rows_path,
        metadata=metadata,
    )


def resolve_prompt_bank(root: str, requested_seed: int) -> PromptBankInfo:
    base = Path(root).expanduser().resolve()
    direct_metadata = base / PROMPT_BANK_METADATA_FILENAME
    direct_rows = base / PROMPT_BANK_ROWS_FILENAME
    if direct_metadata.exists() and direct_rows.exists():
        bank_dir = base
    else:
        bank_dir = base / f"bank_seed_{requested_seed}"
    metadata_path = bank_dir / PROMPT_BANK_METADATA_FILENAME
    rows_path = bank_dir / PROMPT_BANK_ROWS_FILENAME
    if not metadata_path.exists() or not rows_path.exists():
        raise FileNotFoundError(
            f"Prompt bank not found under {base} for seed {requested_seed}."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    kind = str(metadata.get("kind", ""))
    version = int(metadata.get("version", 0))
    if kind != PROMPT_BANK_KIND or version != PROMPT_BANK_VERSION:
        raise RuntimeError(
            f"Unsupported prompt bank format at {bank_dir}: kind={kind!r} version={version}"
        )
    bank_seed = int(metadata.get("bank_seed", requested_seed))
    return PromptBankInfo(
        root=base,
        bank_dir=bank_dir,
        bank_seed=bank_seed,
        metadata_path=metadata_path,
        rows_path=rows_path,
        metadata=metadata,
    )
