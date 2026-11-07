from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DeploySubsetCache:
    """Cache of top-scoring deploy/eval indices for a single manifest or target.

    On cache hit the training loop re-evaluates only the cached subset
    instead of scoring the full deploy pool, then refreshes every
    ``refresh_interval`` steps.
    """

    cached_indices: list[int]
    last_refresh_step: int
    refresh_count: int = 0
    total_uses: int = 0
    total_cache_age: int = 0
    total_candidate_count: int = 0
    total_overlap_with_previous: int = 0
    overlap_measurements: int = 0

    def record_use(self, *, step: int) -> None:
        self.total_uses += 1
        self.total_cache_age += max(0, int(step) - int(self.last_refresh_step))
        self.total_candidate_count += len(self.cached_indices)

    def summary(self) -> dict[str, float]:
        return {
            "refresh_count": float(self.refresh_count),
            "total_uses": float(self.total_uses),
            "mean_cache_age": float(self.total_cache_age / self.total_uses) if self.total_uses else 0.0,
            "mean_candidate_count": float(self.total_candidate_count / self.total_uses) if self.total_uses else 0.0,
            "mean_overlap_with_previous": float(
                self.total_overlap_with_previous / self.overlap_measurements
            )
            if self.overlap_measurements
            else 0.0,
            "current_candidate_count": float(len(self.cached_indices)),
            "last_refresh_step": float(self.last_refresh_step),
        }


def serialize_deploy_caches(caches: dict[str, DeploySubsetCache]) -> dict[str, dict[str, object]]:
    return {name: asdict(cache) for name, cache in caches.items()}


def deserialize_deploy_caches(payload: dict[str, object] | None) -> dict[str, DeploySubsetCache]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, DeploySubsetCache] = {}
    for name, value in payload.items():
        if not isinstance(value, dict):
            continue
        out[str(name)] = DeploySubsetCache(
            cached_indices=list(value.get("cached_indices", [])),
            last_refresh_step=int(value.get("last_refresh_step", 0)),
            refresh_count=int(value.get("refresh_count", 0)),
            total_uses=int(value.get("total_uses", 0)),
            total_cache_age=int(value.get("total_cache_age", 0)),
            total_candidate_count=int(value.get("total_candidate_count", 0)),
            total_overlap_with_previous=int(value.get("total_overlap_with_previous", 0)),
            overlap_measurements=int(value.get("overlap_measurements", 0)),
        )
    return out


@dataclass
class UnionTopCache:
    """Cache of top-C union (fit+deploy) indices for bootstrap meta-training.

    Partition-invariant alternative to DeploySubsetCache: caches the top-C
    indices of the FULL fit+deploy union, where indices 0..n_fit-1 are fit
    and n_fit..n_union-1 are deploy under the canonical ordering passed in.

    Bootstrap then re-partitions the union each step; intersecting any
    permutation's fit/deploy split with the cached top-C gives candidates
    for both fit-top-K and deploy-top-B selection. With C >= B + n_fit,
    the deploy top-B is deterministically covered under any partition; the
    fit top-K is covered with overwhelming probability for moderate C.

    Stores indices only (not scores); the caller re-evaluates cached indices
    each step to get fresh scores against the current model.
    """

    cached_union_indices: list[int]
    last_refresh_step: int
    n_fit_at_refresh: int
    n_union_at_refresh: int
    refresh_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    fit_fallback_count: int = 0
    deploy_fallback_count: int = 0
    total_cache_age: int = 0

    def is_valid(self, *, step: int, n_fit: int, n_union: int, refresh_interval: int) -> bool:
        if not self.cached_union_indices:
            return False
        if (step - self.last_refresh_step) >= refresh_interval:
            return False
        if n_fit != self.n_fit_at_refresh:
            return False
        if n_union != self.n_union_at_refresh:
            return False
        return True

    def record_hit(self, *, step: int) -> None:
        self.hit_count += 1
        self.total_cache_age += max(0, int(step) - int(self.last_refresh_step))

    def record_miss(self) -> None:
        self.miss_count += 1

    def summary(self) -> dict[str, float]:
        total_uses = self.hit_count + self.miss_count
        return {
            "refresh_count": float(self.refresh_count),
            "hit_count": float(self.hit_count),
            "miss_count": float(self.miss_count),
            "fit_fallback_count": float(self.fit_fallback_count),
            "deploy_fallback_count": float(self.deploy_fallback_count),
            "mean_cache_age": float(self.total_cache_age / self.hit_count) if self.hit_count else 0.0,
            "current_cache_size": float(len(self.cached_union_indices)),
            "last_refresh_step": float(self.last_refresh_step),
            "fit_fallback_rate": float(self.fit_fallback_count / total_uses) if total_uses else 0.0,
            "deploy_fallback_rate": float(self.deploy_fallback_count / total_uses) if total_uses else 0.0,
        }


def serialize_union_caches(caches: dict[str, UnionTopCache]) -> dict[str, dict[str, object]]:
    return {name: asdict(cache) for name, cache in caches.items()}


def deserialize_union_caches(payload: dict[str, object] | None) -> dict[str, UnionTopCache]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, UnionTopCache] = {}
    for name, value in payload.items():
        if not isinstance(value, dict):
            continue
        out[str(name)] = UnionTopCache(
            cached_union_indices=list(value.get("cached_union_indices", [])),
            last_refresh_step=int(value.get("last_refresh_step", 0)),
            n_fit_at_refresh=int(value.get("n_fit_at_refresh", 0)),
            n_union_at_refresh=int(value.get("n_union_at_refresh", 0)),
            refresh_count=int(value.get("refresh_count", 0)),
            hit_count=int(value.get("hit_count", 0)),
            miss_count=int(value.get("miss_count", 0)),
            fit_fallback_count=int(value.get("fit_fallback_count", 0)),
            deploy_fallback_count=int(value.get("deploy_fallback_count", 0)),
            total_cache_age=int(value.get("total_cache_age", 0)),
        )
    return out
