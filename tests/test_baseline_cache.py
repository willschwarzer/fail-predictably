"""Unit tests for baseline_cache key stability + sensitivity + on-disk roundtrip.

Run directly:
    python -m tests.test_baseline_cache
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from forecast_experiment.baseline_cache import (
    _BASELINE_FILES,
    _baseline_cache_inputs,
    cache_entry_dir,
    compute_baseline_cache_key,
    save_to_cache,
    try_load_from_cache,
)
from forecast_experiment.train_lquant import LQuantConfig


def test_key_stability_under_dict_reordering():
    cfg1 = LQuantConfig(
        cheap_prompts_per_family_map={"a": 1, "b": 2, "c": 3},
        train_families=["x", "y", "z"],
    )
    cfg2 = LQuantConfig(
        cheap_prompts_per_family_map={"c": 3, "a": 1, "b": 2},
        train_families=["z", "y", "x"],
    )
    assert compute_baseline_cache_key(cfg1) == compute_baseline_cache_key(cfg2)


def test_key_sensitivity_to_baseline_relevant_fields():
    base = LQuantConfig()
    base_k = compute_baseline_cache_key(base)
    for delta_kwargs in [
        dict(seed=base.seed + 1),
        dict(model_id=base.model_id + "-other"),
        dict(prompt_bank_root="/different/path"),
        dict(prompt_bank_seed=base.prompt_bank_seed + 1),
        dict(train_target_count=base.train_target_count + 1),
        dict(heldout_target_count=base.heldout_target_count + 1),
        dict(ols_top_k=base.ols_top_k + 1),
        dict(plotting_position="gringorten" if base.plotting_position == "weibull" else "weibull"),
        dict(disable_thinking=not base.disable_thinking),
    ]:
        other = LQuantConfig(**delta_kwargs)
        assert compute_baseline_cache_key(other) != base_k, (
            f"Expected key change for {delta_kwargs}, got {compute_baseline_cache_key(other)}"
        )


def test_key_insensitivity_to_meta_training_fields():
    base = LQuantConfig()
    base_k = compute_baseline_cache_key(base)
    for delta_kwargs in [
        dict(lr=999.9),
        dict(steps=base.steps + 100),
        dict(kl_coef=base.kl_coef + 0.5),
        dict(improving_only_grad=not base.improving_only_grad),
        dict(iog_scope="deploy_only"),
        dict(bootstrap_permutations=4),
        dict(union_cache_top_c=296),
        dict(micro_batch_size=99),
        dict(grad_clip=base.grad_clip + 1.0),
    ]:
        other = LQuantConfig(**delta_kwargs)
        assert compute_baseline_cache_key(other) == base_k, (
            f"Expected key UNCHANGED for {delta_kwargs}, got "
            f"{compute_baseline_cache_key(other)} vs {base_k}"
        )


def test_save_load_roundtrip_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = Path(tmp) / "cache"
        outdir = Path(tmp) / "run1"
        outdir.mkdir(parents=True)
        # Synthesize the artifacts a real run would produce
        (outdir / "baseline_test_results.json").write_text(json.dumps({"slot_0": {"x": 1}}))
        (outdir / "baseline_test_pool_diagnostic.json").write_text(json.dumps({"y": 2}))
        (outdir / "baseline_kl_eval_results.json").write_text(json.dumps({}))
        (outdir / "baseline_test_forecast_traces.npz").write_bytes(b"\x93NUMPY-fake")
        (outdir / "baseline_test_forecast_traces_metadata.json").write_text(
            json.dumps({"format": "test"})
        )
        cfg = LQuantConfig(seed=11)
        save_to_cache(str(cache_root), cfg, outdir)
        # Hit on a different outdir
        outdir2 = Path(tmp) / "run2"
        outdir2.mkdir()
        hit = try_load_from_cache(str(cache_root), cfg, outdir2)
        assert hit, "Expected cache hit"
        for fname in _BASELINE_FILES:
            assert (outdir2 / fname).exists(), f"Missing {fname} in restored outdir"
        # Content matches
        assert (outdir2 / "baseline_test_results.json").read_text() == \
            (outdir / "baseline_test_results.json").read_text()
        # Manifest written
        key = compute_baseline_cache_key(cfg)
        manifest = json.loads((cache_entry_dir(cache_root, key) / "manifest.json").read_text())
        assert manifest["key"] == key
        assert "files" in manifest


def test_disabled_cache_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = LQuantConfig()
        outdir = Path(tmp) / "run"
        outdir.mkdir()
        # cache_root=None: try_load returns False
        assert try_load_from_cache(None, cfg, outdir) is False
        # save_to_cache with cache_root=None is a no-op
        save_to_cache(None, cfg, outdir)
        # Empty string should also be treated as disabled
        assert try_load_from_cache("", cfg, outdir) is False


def test_miss_when_no_entry():
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = Path(tmp) / "cache"
        cache_root.mkdir()
        outdir = Path(tmp) / "run"
        outdir.mkdir()
        cfg = LQuantConfig()
        assert try_load_from_cache(str(cache_root), cfg, outdir) is False


def main():
    test_key_stability_under_dict_reordering()
    test_key_sensitivity_to_baseline_relevant_fields()
    test_key_insensitivity_to_meta_training_fields()
    test_save_load_roundtrip_on_disk()
    test_disabled_cache_is_noop()
    test_miss_when_no_entry()
    print("all baseline_cache tests passed")


if __name__ == "__main__":
    main()
