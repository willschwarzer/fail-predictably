"""Unit tests for slip-level trap bank support.

Covers: (1) backward-compatible bit-exactness for slip_value=0 records
(old banks without the field must load and materialize identically),
(2) slip>0 record correctness (geometry preserved, hash distinct, DP
optimal value consistent), (3) the task-vector include_slip flag,
(4) the slip-grouped manifest builder in train_rlquant.

Run directly:
    python -m tests.test_slip_bank
"""
from __future__ import annotations

import torch

from forecast_experiment.rlquant_gridworld import (
    GridworldGeneratorConfig,
    TrapLayoutRecord,
    _layout_from_trap_layout_record,
    build_task_from_layout,
    build_trap_layout_record_from_layout,
    canonical_layout_hash,
    compute_optimal_value,
    generate_layout,
    layout_task_vector,
    materialize_task_from_trap_layout_record,
)

_CFG = GridworldGeneratorConfig(
    rows=8,
    cols=8,
    horizon=10,
    gamma=1.0,
    step_cost=-0.01,
    goal_reward=1.0,
    catastrophe_penalty=-1.0,
    min_start_goal_distance=5,
)


def _make_layout(backbone_seed: int, overlay_seed: int, severity: float, slip_value: float):
    # Generation is rejection sampling, so scan forward from the given seeds
    # until a valid layout appears (slip does not affect validity).
    for offset in range(200):
        layout = generate_layout(
            "trap_open_room",
            severity,
            cfg=_CFG,
            backbone_seed=backbone_seed + offset,
            overlay_seed=overlay_seed + offset,
            slip_value=slip_value,
        )
        if layout is not None:
            return layout
    raise AssertionError(f"no valid layout near seeds ({backbone_seed}, {overlay_seed})")


def _some_valid_seed_pairs(n: int, severity: float, slip_value: float):
    pairs = []
    backbone_seed = 1000
    overlay_seed = 2000
    while len(pairs) < n:
        layout = generate_layout(
            "trap_open_room",
            severity,
            cfg=_CFG,
            backbone_seed=backbone_seed,
            overlay_seed=overlay_seed,
            slip_value=slip_value,
        )
        if layout is not None:
            pairs.append((backbone_seed, overlay_seed))
        backbone_seed += 7
        overlay_seed += 13
    return pairs


def test_backcompat_record_without_slip_field_is_bit_exact():
    layout = _make_layout(12345, 67890, severity=0.0, slip_value=0.0)
    record = build_trap_layout_record_from_layout(layout)
    assert record.slip_value == 0.0

    payload = record.to_json()
    assert "slip_value" in payload
    del payload["slip_value"]  # simulate a record written by the pre-slip code
    old_style_record = TrapLayoutRecord.from_json(payload)
    assert old_style_record.slip_value == 0.0

    task = materialize_task_from_trap_layout_record(
        old_style_record, cfg=_CFG, split_name="test", task_id="t0"
    )
    direct_task = build_task_from_layout(layout, split_name="test", task_id="t0")

    assert task.task_vector.shape[0] == 29
    assert torch.equal(task.task_vector, direct_task.task_vector)
    assert torch.equal(task.feature_tensor, direct_task.feature_tensor)
    assert torch.equal(task.transition_next_states, direct_task.transition_next_states)
    assert torch.equal(task.transition_probs, direct_task.transition_probs)
    assert torch.equal(task.transition_rewards, direct_task.transition_rewards)
    assert task.optimal_value == direct_task.optimal_value

    reconstructed = _layout_from_trap_layout_record(old_style_record, cfg=_CFG)
    assert canonical_layout_hash(reconstructed) == record.canonical_hash


def test_slip_record_geometry_hash_and_optimal_value():
    strictly_lower = 0
    for backbone_seed, overlay_seed in _some_valid_seed_pairs(5, severity=0.0, slip_value=0.0):
        base = _make_layout(backbone_seed, overlay_seed, severity=0.0, slip_value=0.0)
        slipped = _make_layout(backbone_seed, overlay_seed, severity=0.0, slip_value=0.3)

        # Same RNG stream: identical geometry, only the slip grid differs.
        assert torch.equal(base.hazard_mask, slipped.hazard_mask)
        assert base.start_pos == slipped.start_pos
        assert torch.equal(base.goal_mask, slipped.goal_mask)
        assert torch.all(slipped.slip_prob == slipped.slip_prob[0, 0])
        assert abs(float(slipped.slip_prob[0, 0].item()) - 0.3) < 1e-6
        assert canonical_layout_hash(base) != canonical_layout_hash(slipped)

        record = build_trap_layout_record_from_layout(slipped)
        assert abs(record.slip_value - 0.3) < 1e-6

        # Record -> layout round trip is exact (hash includes the slip grid).
        reconstructed = _layout_from_trap_layout_record(record, cfg=_CFG)
        assert canonical_layout_hash(reconstructed) == record.canonical_hash

        # DP optimal value recomputed from the materialized transitions
        # matches the stored one, and slip can only hurt the optimum.
        task = materialize_task_from_trap_layout_record(
            record, cfg=_CFG, split_name="test", task_id="t1"
        )
        recomputed = compute_optimal_value(
            task.transition_next_states,
            task.transition_probs,
            task.transition_rewards,
            start_state_index=task.start_state_index,
            horizon=task.horizon,
            gamma=task.gamma,
        )
        assert abs(float(recomputed) - record.optimal_value) < 1e-6

        base_record = build_trap_layout_record_from_layout(base)
        assert record.optimal_value <= base_record.optimal_value + 1e-6
        if record.optimal_value < base_record.optimal_value - 1e-6:
            strictly_lower += 1
    assert strictly_lower >= 1, "slip 0.3 never strictly lowered the optimal value on 5 layouts"


def test_task_vector_include_slip_flag():
    layout = _make_layout(555, 777, severity=0.0, slip_value=0.25)
    vec_default = layout_task_vector(layout)
    vec_slip = layout_task_vector(layout, include_slip=True)
    assert vec_default.shape[0] == 29
    assert vec_slip.shape[0] == 30
    assert torch.equal(vec_slip[:29], vec_default)
    assert abs(float(vec_slip[29].item()) - 0.25) < 1e-6

    task = materialize_task_from_trap_layout_record(
        build_trap_layout_record_from_layout(layout),
        cfg=_CFG,
        split_name="test",
        task_id="t2",
        include_slip_in_task_vector=True,
    )
    assert task.task_vector.shape[0] == 30
    assert abs(float(task.task_vector[29].item()) - 0.25) < 1e-6


def _write_synthetic_slip_bank(path, levels, per_level, tail_per_level):
    """Write a synthetic trap-layout bank jsonl: per_level records per slip level,
    of which tail_per_level have severity 1. Hashes are unique dummies (the
    grouped builder never materializes)."""
    import json as _json

    with open(path, "w", encoding="utf-8") as f:
        idx = 0
        for level in levels:
            for j in range(per_level):
                record = TrapLayoutRecord(
                    family_id="trap_open_room",
                    subfamily_id="trap_open_room_d5",
                    severity=1.0 if j < tail_per_level else 0.0,
                    canonical_hash=f"synthetic{idx:06d}",
                    optimal_value=0.9,
                    start_pos=(0, 0),
                    goal_pos=(7, 7),
                    hazard_bits="00",
                    backbone_seed=idx,
                    overlay_seed=idx,
                    slip_value=level,
                )
                f.write(_json.dumps(record.to_json(), sort_keys=True) + "\n")
                idx += 1


def _grouped_suite(bank_path, *, seed, meta_count=3, heldout_count=2, pretrain=6):
    from pathlib import Path

    from forecast_experiment.train_rlquant import (
        RLQuantConfig,
        TaskBankInfo,
        _build_manifest_suite_from_bank_slip_grouped,
    )

    cfg = RLQuantConfig(
        seed=seed,
        family_suite="trap_open_room_only",
        meta_manifest_count=meta_count,
        heldout_manifest_count=heldout_count,
        trap_pretrain_per_family=pretrain,
        trap_meta_fit_per_family=4,
        trap_meta_deploy_extra_per_family=8,
        trap_heldout_fit_per_family=4,
        trap_heldout_deploy_extra_per_family=8,
        manifest_grouping="slip_level",
        task_bank_root=str(Path(bank_path).parent),
    )
    bank_info = TaskBankInfo(
        root=Path(bank_path).parent,
        bank_dir=Path(bank_path).parent,
        bank_path=Path(bank_path),
        bank_seed=seed,
        record_format="trap_layout",
        submission_meta={},
        bank_meta={},
        shard_meta={},
    )
    return _build_manifest_suite_from_bank_slip_grouped(cfg, seed=seed, bank_info=bank_info), cfg


def test_slip_grouped_manifest_builder():
    import tempfile
    from pathlib import Path

    levels = [0.0, 0.15, 0.3, 0.45, 0.6]
    with tempfile.TemporaryDirectory() as tmp:
        bank_path = Path(tmp) / "bank_layouts.jsonl"
        _write_synthetic_slip_bank(bank_path, levels, per_level=14, tail_per_level=1)

        suite, _cfg = _grouped_suite(bank_path, seed=11)

        assert len(suite.meta_manifests) == 3
        assert len(suite.heldout_manifests) == 2
        assert len(suite.pretrain_train) == 6

        all_hashes: list[str] = [r.canonical_hash for r in suite.pretrain_train]
        meta_levels = set()
        heldout_levels = set()
        for manifest in suite.meta_manifests + suite.heldout_manifests:
            assert len(manifest.fit_tasks) == 4
            assert len(manifest.deploy_tasks) == 8
            slips = {round(r.slip_value, 6) for r in manifest.fit_tasks + manifest.deploy_tasks}
            assert slips == {round(manifest.slip_level, 6)}, (
                f"{manifest.name}: mixed slips {slips} vs level {manifest.slip_level}"
            )
            assert f"slip{manifest.slip_level:.3f}" in manifest.name
            all_hashes.extend(r.canonical_hash for r in manifest.fit_tasks + manifest.deploy_tasks)
            (meta_levels if manifest.name.startswith("meta") else heldout_levels).add(
                round(manifest.slip_level, 6)
            )
        assert len(all_hashes) == len(set(all_hashes)), "tasks reused across partitions"
        assert meta_levels | heldout_levels == {round(level, 6) for level in levels}
        assert not (meta_levels & heldout_levels)

        # Pretrain draws only from meta-level leftovers.
        pretrain_slips = {round(r.slip_value, 6) for r in suite.pretrain_train}
        assert pretrain_slips <= meta_levels
        assert not (pretrain_slips & heldout_levels)

        # Same seed reproduces the exact partition; across seeds the heldout
        # level choice varies.
        suite_again, _ = _grouped_suite(bank_path, seed=11)
        assert [m.slip_level for m in suite_again.heldout_manifests] == [
            m.slip_level for m in suite.heldout_manifests
        ]
        assert [r.canonical_hash for r in suite_again.pretrain_train] == [
            r.canonical_hash for r in suite.pretrain_train
        ]
        heldout_sets = set()
        for seed in range(11, 31):
            suite_s, _ = _grouped_suite(bank_path, seed=seed)
            heldout_sets.add(tuple(sorted(m.slip_level for m in suite_s.heldout_manifests)))
        assert len(heldout_sets) >= 2, "heldout levels never varied across 20 seeds"

        # Loud failures: wrong level count, uneven groups, pretrain too large.
        try:
            _grouped_suite(bank_path, seed=11, meta_count=4)
            raise AssertionError("expected RuntimeError for level-count mismatch")
        except RuntimeError:
            pass
        try:
            _grouped_suite(bank_path, seed=11, pretrain=7)
            raise AssertionError("expected RuntimeError for insufficient leftovers")
        except RuntimeError:
            pass
        uneven_path = Path(tmp) / "bank_uneven.jsonl"
        _write_synthetic_slip_bank(uneven_path, levels, per_level=14, tail_per_level=1)
        with open(uneven_path, "a", encoding="utf-8") as f:
            record = TrapLayoutRecord(
                family_id="trap_open_room",
                subfamily_id="trap_open_room_d5",
                severity=0.0,
                canonical_hash="synthetic_extra",
                optimal_value=0.9,
                start_pos=(0, 0),
                goal_pos=(7, 7),
                hazard_bits="00",
                backbone_seed=999999,
                overlay_seed=999999,
                slip_value=levels[0],
            )
            import json as _json

            f.write(_json.dumps(record.to_json(), sort_keys=True) + "\n")
        try:
            _grouped_suite(uneven_path, seed=11)
            raise AssertionError("expected RuntimeError for uneven groups")
        except RuntimeError:
            pass


def test_wind_record_roundtrip_and_dp_consistency():
    base = _make_layout(4321, 8765, severity=0.0, slip_value=0.0)
    windy = generate_layout(
        "trap_open_room",
        0.0,
        cfg=_CFG,
        backbone_seed=base.backbone_seed,
        overlay_seed=base.overlay_seed,
        slip_value=0.1,
        wind_dir=1,
        wind_strength=0.3,
    )
    assert windy is not None
    # Same RNG stream: wind/slip change dynamics only, never geometry.
    assert torch.equal(base.hazard_mask, windy.hazard_mask)
    assert canonical_layout_hash(base) != canonical_layout_hash(windy)
    slip_only = generate_layout(
        "trap_open_room",
        0.0,
        cfg=_CFG,
        backbone_seed=base.backbone_seed,
        overlay_seed=base.overlay_seed,
        slip_value=0.1,
    )
    assert canonical_layout_hash(slip_only) != canonical_layout_hash(windy)

    record = build_trap_layout_record_from_layout(windy, dynamics_id=7)
    assert record.wind_dir == 1 and abs(record.wind_strength - 0.3) < 1e-9
    assert record.dynamics_id == 7
    payload = record.to_json()
    for key in ("wind_dir", "wind_strength", "dynamics_id"):
        assert key in payload
        del payload[key]
    old_style = TrapLayoutRecord.from_json(payload)  # pre-wind payloads
    assert old_style.wind_strength == 0.0 and old_style.dynamics_id == -1

    reconstructed = _layout_from_trap_layout_record(record, cfg=_CFG)
    assert canonical_layout_hash(reconstructed) == record.canonical_hash

    task = materialize_task_from_trap_layout_record(
        record, cfg=_CFG, split_name="test", task_id="tw"
    )
    assert task.wind_dir == 1 and abs(task.wind_strength - 0.3) < 1e-9
    sums = task.transition_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    recomputed = compute_optimal_value(
        task.transition_next_states,
        task.transition_probs,
        task.transition_rewards,
        start_state_index=task.start_state_index,
        horizon=task.horizon,
        gamma=task.gamma,
    )
    assert abs(float(recomputed) - record.optimal_value) < 1e-6


def test_blind_dynamics_observation():
    layout = _make_layout(999, 111, severity=0.0, slip_value=0.0)
    windy = generate_layout(
        "trap_open_room",
        0.0,
        cfg=_CFG,
        backbone_seed=layout.backbone_seed,
        overlay_seed=layout.overlay_seed,
        slip_value=0.15,
        wind_dir=2,
        wind_strength=0.25,
    )
    record = build_trap_layout_record_from_layout(windy, dynamics_id=3)
    blind_task = materialize_task_from_trap_layout_record(
        record, cfg=_CFG, split_name="test", task_id="tb", blind_dynamics=True
    )
    seeing_task = materialize_task_from_trap_layout_record(
        record, cfg=_CFG, split_name="test", task_id="ts", blind_dynamics=False
    )
    # Channel 10 is the slip channel: zeroed when blind, populated when not.
    assert torch.all(blind_task.feature_tensor[10] == 0.0)
    assert torch.all(seeing_task.feature_tensor[10] > 0.0)
    # The observation is blind but the MDP keeps the true dynamics.
    assert torch.equal(blind_task.transition_probs, seeing_task.transition_probs)
    base_record = build_trap_layout_record_from_layout(
        _make_layout(999, 111, severity=0.0, slip_value=0.0)
    )
    base_task = materialize_task_from_trap_layout_record(
        base_record, cfg=_CFG, split_name="test", task_id="t0", blind_dynamics=True
    )
    assert not torch.equal(blind_task.transition_probs, base_task.transition_probs)
    try:
        materialize_task_from_trap_layout_record(
            record,
            cfg=_CFG,
            split_name="test",
            task_id="tc",
            blind_dynamics=True,
            include_slip_in_task_vector=True,
        )
        raise AssertionError("expected ValueError for blind + slip-in-task-vector")
    except ValueError:
        pass


def test_dynamics_grouped_manifest_builder():
    import json as _json
    import tempfile
    from pathlib import Path

    specs = [(0.0, 0, 0.0), (0.05, 1, 0.1), (0.1, 2, 0.2), (0.15, 3, 0.3), (0.2, 0, 0.4)]
    with tempfile.TemporaryDirectory() as tmp:
        bank_path = Path(tmp) / "bank_layouts.jsonl"
        with open(bank_path, "w", encoding="utf-8") as f:
            idx = 0
            for did, (slip, wdir, wstr) in enumerate(specs):
                for j in range(14):
                    record = TrapLayoutRecord(
                        family_id="trap_open_room",
                        subfamily_id="trap_open_room_d5",
                        severity=1.0 if j == 0 else 0.0,
                        canonical_hash=f"dynsyn{idx:06d}",
                        optimal_value=0.9,
                        start_pos=(0, 0),
                        goal_pos=(7, 7),
                        hazard_bits="00",
                        backbone_seed=idx,
                        overlay_seed=idx,
                        slip_value=slip,
                        wind_dir=wdir,
                        wind_strength=wstr,
                        dynamics_id=did,
                    )
                    f.write(_json.dumps(record.to_json(), sort_keys=True) + "\n")
                    idx += 1

        from forecast_experiment.train_rlquant import (
            RLQuantConfig,
            TaskBankInfo,
            _build_manifest_suite_from_bank_slip_grouped,
        )

        cfg = RLQuantConfig(
            seed=11,
            family_suite="trap_open_room_only",
            meta_manifest_count=3,
            heldout_manifest_count=2,
            trap_pretrain_per_family=6,
            trap_meta_fit_per_family=4,
            trap_meta_deploy_extra_per_family=8,
            trap_heldout_fit_per_family=4,
            trap_heldout_deploy_extra_per_family=8,
            manifest_grouping="dynamics_id",
            task_bank_root=str(Path(tmp)),
        )
        bank_info = TaskBankInfo(
            root=Path(tmp),
            bank_dir=Path(tmp),
            bank_path=bank_path,
            bank_seed=11,
            record_format="trap_layout",
            submission_meta={},
            bank_meta={},
            shard_meta={},
        )
        suite = _build_manifest_suite_from_bank_slip_grouped(cfg, seed=11, bank_info=bank_info)
        assert len(suite.meta_manifests) == 3 and len(suite.heldout_manifests) == 2
        for manifest in suite.meta_manifests + suite.heldout_manifests:
            dids = {r.dynamics_id for r in manifest.fit_tasks + manifest.deploy_tasks}
            assert len(dids) == 1, f"{manifest.name}: mixed dynamics ids {dids}"
            did = dids.pop()
            assert f"dyn{did:02d}" in manifest.name
            expected_slip, _wdir, expected_wstr = specs[did]
            assert f"slip{expected_slip:.3f}" in manifest.name
            assert f"x{expected_wstr:.3f}" in manifest.name

        # A bank without dynamics ids must fail loudly in dynamics_id mode.
        plain_path = Path(tmp) / "bank_plain.jsonl"
        _write_synthetic_slip_bank(plain_path, [0.0, 0.1, 0.2, 0.3, 0.4], per_level=14, tail_per_level=1)
        bank_info_plain = TaskBankInfo(
            root=Path(tmp),
            bank_dir=Path(tmp),
            bank_path=plain_path,
            bank_seed=11,
            record_format="trap_layout",
            submission_meta={},
            bank_meta={},
            shard_meta={},
        )
        try:
            _build_manifest_suite_from_bank_slip_grouped(cfg, seed=11, bank_info=bank_info_plain)
            raise AssertionError("expected RuntimeError for missing dynamics_id")
        except RuntimeError:
            pass


def main() -> None:
    test_backcompat_record_without_slip_field_is_bit_exact()
    print("test_backcompat_record_without_slip_field_is_bit_exact passed")
    test_slip_record_geometry_hash_and_optimal_value()
    print("test_slip_record_geometry_hash_and_optimal_value passed")
    test_task_vector_include_slip_flag()
    print("test_task_vector_include_slip_flag passed")
    test_slip_grouped_manifest_builder()
    print("test_slip_grouped_manifest_builder passed")
    test_wind_record_roundtrip_and_dp_consistency()
    print("test_wind_record_roundtrip_and_dp_consistency passed")
    test_blind_dynamics_observation()
    print("test_blind_dynamics_observation passed")
    test_dynamics_grouped_manifest_builder()
    print("test_dynamics_grouped_manifest_builder passed")
    print("all slip bank tests passed")


if __name__ == "__main__":
    main()
