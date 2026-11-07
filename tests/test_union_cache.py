"""Unit tests for UnionTopCache and its serialization.

Run directly:
    python -m tests.test_union_cache
"""
from __future__ import annotations

import torch

from forecast_experiment.eval_cache import (
    UnionTopCache,
    deserialize_union_caches,
    serialize_union_caches,
)


def test_validity_basic():
    c = UnionTopCache(
        cached_union_indices=[1, 2, 3, 4, 5],
        last_refresh_step=0,
        n_fit_at_refresh=10,
        n_union_at_refresh=100,
    )
    # Same params, within refresh interval: valid
    assert c.is_valid(step=0, n_fit=10, n_union=100, refresh_interval=5)
    assert c.is_valid(step=4, n_fit=10, n_union=100, refresh_interval=5)
    # At refresh interval: invalid
    assert not c.is_valid(step=5, n_fit=10, n_union=100, refresh_interval=5)
    assert not c.is_valid(step=6, n_fit=10, n_union=100, refresh_interval=5)
    # n_fit changed: invalid
    assert not c.is_valid(step=2, n_fit=11, n_union=100, refresh_interval=5)
    # n_union changed: invalid
    assert not c.is_valid(step=2, n_fit=10, n_union=99, refresh_interval=5)
    # Empty cache: invalid
    empty = UnionTopCache(
        cached_union_indices=[], last_refresh_step=0,
        n_fit_at_refresh=10, n_union_at_refresh=100,
    )
    assert not empty.is_valid(step=0, n_fit=10, n_union=100, refresh_interval=5)


def test_record_hit_and_miss():
    c = UnionTopCache(
        cached_union_indices=[1, 2, 3], last_refresh_step=10,
        n_fit_at_refresh=5, n_union_at_refresh=50,
    )
    c.record_hit(step=10)
    c.record_hit(step=12)
    c.record_hit(step=14)
    c.record_miss()
    s = c.summary()
    assert s["hit_count"] == 3
    assert s["miss_count"] == 1
    # cache ages: 0 + 2 + 4 = 6, divided by 3 hits = 2
    assert abs(s["mean_cache_age"] - 2.0) < 1e-9


def test_serialize_roundtrip():
    c1 = UnionTopCache(
        cached_union_indices=[1, 5, 10],
        last_refresh_step=42,
        n_fit_at_refresh=7,
        n_union_at_refresh=200,
        refresh_count=3,
        hit_count=12,
        miss_count=4,
        fit_fallback_count=1,
        deploy_fallback_count=0,
        total_cache_age=20,
    )
    c2 = UnionTopCache(
        cached_union_indices=[2, 3], last_refresh_step=0,
        n_fit_at_refresh=2, n_union_at_refresh=20,
    )
    payload = serialize_union_caches({"alpha": c1, "beta": c2})
    restored = deserialize_union_caches(payload)
    assert set(restored.keys()) == {"alpha", "beta"}
    assert restored["alpha"].cached_union_indices == [1, 5, 10]
    assert restored["alpha"].last_refresh_step == 42
    assert restored["alpha"].refresh_count == 3
    assert restored["alpha"].hit_count == 12
    assert restored["alpha"].fit_fallback_count == 1
    assert restored["beta"].cached_union_indices == [2, 3]


def test_deterministic_deploy_coverage():
    """Hypergeometric guarantee: with C >= B + n_fit, every random partition
    leaves at least B cached items in the deploy half. This is the
    deterministic invariant the cache relies on.
    """
    n_union = 2016
    n_fit = 96
    n_deploy = n_union - n_fit
    c_size = 296  # = B + n_fit with B=200
    g = torch.Generator(device="cpu").manual_seed(7)

    # Cache the "top-C" indices (first C as a proxy for top-C by score)
    cached_set = set(range(c_size))
    n_trials = 200
    min_deploy_in_cache = float("inf")
    for _ in range(n_trials):
        perm = torch.randperm(n_union, generator=g)
        f_set = set(perm[:n_fit].tolist())
        d_set = set(perm[n_fit : n_fit + n_deploy].tolist())
        # Cached items in deploy
        cached_in_deploy = len(cached_set & d_set)
        min_deploy_in_cache = min(min_deploy_in_cache, cached_in_deploy)
    # At C=296, n_fit=96, the worst case is when ALL 96 fit slots happen
    # to land on cached items, leaving C - n_fit = 200 cached in deploy.
    # That's exactly B=200, so coverage is guaranteed.
    assert min_deploy_in_cache >= 200, (
        f"Expected at least 200 cached items in deploy under any partition, "
        f"got minimum {min_deploy_in_cache} across {n_trials} trials"
    )


def main():
    test_validity_basic()
    test_record_hit_and_miss()
    test_serialize_roundtrip()
    test_deterministic_deploy_coverage()
    print("all UnionTopCache tests passed")


if __name__ == "__main__":
    main()
