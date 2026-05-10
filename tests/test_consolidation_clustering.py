"""Tests for `engram.consolidation._clustering`."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engram.consolidation import ClusterAssignment, ClusterParams, cluster, cohesion
from engram.consolidation._clustering import _hdbscan_available


def _unit(*vecs: list[float]) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# ClusterParams
# ---------------------------------------------------------------------------


class TestClusterParams:
    def test_defaults(self) -> None:
        p = ClusterParams()
        assert p.min_cluster_size == 3
        assert p.cohesion_threshold == 0.6
        assert p.method == "auto"

    def test_min_cluster_size_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="min_cluster_size"):
            ClusterParams(min_cluster_size=1)

    def test_cohesion_threshold_bounds(self) -> None:
        with pytest.raises(ValueError, match="cohesion_threshold"):
            ClusterParams(cohesion_threshold=-0.01)
        with pytest.raises(ValueError, match="cohesion_threshold"):
            ClusterParams(cohesion_threshold=1.01)

    def test_method_validated(self) -> None:
        with pytest.raises(ValueError, match="method"):
            ClusterParams(method="bogus")  # type: ignore[arg-type]

    def test_auto_min_n_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="auto_hdbscan_min_n"):
            ClusterParams(auto_hdbscan_min_n=1)


# ---------------------------------------------------------------------------
# cohesion()
# ---------------------------------------------------------------------------


class TestCohesion:
    def test_singleton_is_one(self) -> None:
        assert cohesion(_unit([1.0, 0.0])) == 1.0

    def test_identical_rows_is_one(self) -> None:
        v = _unit([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
        assert math.isclose(cohesion(v), 1.0, abs_tol=1e-6)

    def test_orthogonal_pair_is_zero(self) -> None:
        v = _unit([1.0, 0.0], [0.0, 1.0])
        assert math.isclose(cohesion(v), 0.0, abs_tol=1e-6)

    def test_antiparallel_pair_is_minus_one(self) -> None:
        v = _unit([1.0, 0.0], [-1.0, 0.0])
        assert math.isclose(cohesion(v), -1.0, abs_tol=1e-6)

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match=r"\(N, D\)"):
            cohesion(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Agglomerative clustering
# ---------------------------------------------------------------------------


class TestAgglomerative:
    def test_two_well_separated_groups(self) -> None:
        # First three vectors all close to (1, 0); next three close to (0, 1).
        v = _unit(
            [1.0, 0.05],
            [1.0, 0.0],
            [0.99, 0.05],
            [0.05, 1.0],
            [0.0, 1.0],
            [0.05, 0.99],
        )
        params = ClusterParams(method="agglomerative", cohesion_threshold=0.9, min_cluster_size=2)
        clusters = cluster(v, params=params)
        assert len(clusters) == 2
        assert clusters[0].members == (0, 1, 2)
        assert clusters[1].members == (3, 4, 5)
        # Each cluster has high cohesion.
        for c in clusters:
            assert c.cohesion > 0.99

    def test_threshold_too_high_kills_all_groups(self) -> None:
        v = _unit(
            [1.0, 0.0],
            [0.99, 0.05],
            [0.0, 1.0],
            [0.05, 0.99],
        )
        params = ClusterParams(
            method="agglomerative", cohesion_threshold=0.999999, min_cluster_size=2
        )
        clusters = cluster(v, params=params)
        # Pairs not similar enough -> singletons -> dropped by min_cluster_size.
        assert clusters == []

    def test_drops_clusters_smaller_than_min(self) -> None:
        # Three singletons + a pair; min_cluster_size=3 keeps nothing.
        v = _unit(
            [1.0, 0.0],
            [1.0, 0.05],
            [0.0, 1.0],
            [-1.0, 0.0],
        )
        params = ClusterParams(method="agglomerative", cohesion_threshold=0.9, min_cluster_size=3)
        clusters = cluster(v, params=params)
        assert clusters == []

    def test_n_below_min_returns_empty(self) -> None:
        v = _unit([1.0, 0.0])
        params = ClusterParams(method="agglomerative", min_cluster_size=2)
        assert cluster(v, params=params) == []

    def test_assignment_is_deterministic(self) -> None:
        # Same input + same params -> identical output.
        v = _unit(
            [1.0, 0.05],
            [1.0, 0.0],
            [0.99, 0.05],
            [0.05, 1.0],
            [0.0, 1.0],
            [0.05, 0.99],
        )
        params = ClusterParams(method="agglomerative", cohesion_threshold=0.9, min_cluster_size=2)
        a = cluster(v, params=params)
        b = cluster(v, params=params)
        assert a == b


# ---------------------------------------------------------------------------
# Auto method picker
# ---------------------------------------------------------------------------


class TestAutoMethod:
    def test_auto_uses_agglomerative_for_small_n(self) -> None:
        # Tiny input -> agglomerative regardless of HDBSCAN availability.
        v = _unit([1.0, 0.0], [0.99, 0.05], [0.98, 0.10])
        clusters = cluster(v, params=ClusterParams(method="auto", min_cluster_size=2))
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# HDBSCAN (only run if installed)
# ---------------------------------------------------------------------------


class TestHdbscan:
    @pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
    def test_hdbscan_finds_dense_clusters(self) -> None:
        rng = np.random.default_rng(seed=42)
        # Two well-separated centers in 8-D so the density estimate has
        # room to disambiguate. HDBSCAN may split a region with internal
        # density variation into >1 sub-cluster - we don't assert the
        # exact partition (that's data-dependent), only that:
        #   * we recover at least one substantial cluster from each side
        #   * every returned cluster is internally cohesive
        d = 8
        center_a = np.zeros(d, dtype=np.float64)
        center_a[0] = 1.0
        center_b = np.zeros(d, dtype=np.float64)
        center_b[1] = 1.0
        a = center_a + rng.normal(0, 0.02, (40, d))
        b = center_b + rng.normal(0, 0.02, (40, d))
        v = np.vstack([a, b])
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = (v / norms).astype(np.float32)
        clusters = cluster(v, params=ClusterParams(method="hdbscan", min_cluster_size=5))
        assert clusters, "expected at least one cluster"
        # Each cluster is internally cohesive.
        for c in clusters:
            assert c.cohesion > 0.95
        # Every member index lands in exactly one cluster.
        seen: set[int] = set()
        for c in clusters:
            for idx in c.members:
                assert idx not in seen
                seen.add(idx)
        # Some points may be noise; we expect the substantial majority to
        # be captured.
        assert len(seen) >= 50

    def test_explicit_hdbscan_when_unavailable_raises(self) -> None:
        if _hdbscan_available():
            pytest.skip("hdbscan is installed; cannot test the missing-import path here")
        v = _unit([1.0, 0.0], [0.99, 0.05], [0.98, 0.1])
        with pytest.raises(ImportError):
            cluster(v, params=ClusterParams(method="hdbscan"))


# ---------------------------------------------------------------------------
# Hypothesis: invariants
# ---------------------------------------------------------------------------


_random_unit_vectors = st.builds(
    lambda seed, n, d: _build_unit(seed=seed, n=n, d=d),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n=st.integers(min_value=2, max_value=20),
    d=st.integers(min_value=2, max_value=8),
)


def _build_unit(*, seed: int, n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    arr = rng.normal(size=(n, d)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


@given(vectors=_random_unit_vectors)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_clusters_are_disjoint_and_within_input_range(vectors: np.ndarray) -> None:
    clusters = cluster(vectors, params=ClusterParams(method="agglomerative", min_cluster_size=2))
    seen: set[int] = set()
    n = vectors.shape[0]
    for c in clusters:
        assert isinstance(c, ClusterAssignment)
        for idx in c.members:
            assert 0 <= idx < n
            assert idx not in seen, "clusters must be disjoint"
            seen.add(idx)
        assert len(c.members) >= 2
