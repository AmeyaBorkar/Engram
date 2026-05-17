"""Clustering of normalized embeddings for consolidation.

Two backends, switchable via `ClusterParams.method`:

  `hdbscan`        — density-based, robust to noise, picks cluster sizes
                     automatically. Requires the `hdbscan` package
                     (optional `[consolidation]` extra). For unit-norm
                     vectors we use the Euclidean metric, which is
                     monotonic with cosine distance and lets HDBSCAN use
                     its spatial indexing fast paths.
  `agglomerative`  — single-link agglomerative threshold clustering using
                     numpy + a union-find. Pure Python beyond numpy, no
                     extra deps. O(N^2) memory for the similarity matrix,
                     so this is the fallback for small N (< 50 by default
                     under `method="auto"`).

`cluster()` returns a list of `ClusterAssignment` records. Items not in
any cluster (HDBSCAN noise label, or singletons under the threshold) are
not returned - the engine only acts on items that grouped.

The math is deterministic in both backends: HDBSCAN itself is
deterministic given a fixed input; the agglomerative path uses an
explicit comparison order that doesn't depend on hash randomization.
Replays produce identical cluster assignments.

Asymptotics (audit H-56):
  * `agglomerative`: O(N^2) memory for the (N,N) similarity matrix +
    O(N^2 alpha(N)) work for the union-find merge step (alpha is the
    inverse-Ackermann factor from path-compression union-find).  The
    edge-finding step itself is now fully vectorized via
    `np.argwhere(np.triu(...))`, so the pure-Python loop cost is
    O(#edges_above_threshold) rather than O(N^2).  For typical
    cohesion thresholds (>= 0.6) the edge count is small fraction of
    N^2; for dense corpora it asymptotes back to the prior bound.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

_LOG = logging.getLogger("engram.consolidation.clustering")

# Audit M-54: agglomerative clustering assumes unit-norm rows so the
# similarity matrix == cosine similarity. A caller that hands in raw
# (un-normalized) vectors gets silent garbage. We sample the first row
# at call time and emit a single warning if its norm deviates from 1.0
# by more than this tolerance. The pass continues — the warning is
# diagnostic, not blocking.
_UNIT_NORM_TOL: float = 1e-3
_NORM_WARNING_EMITTED: bool = False

# A 2D float matrix. Engram normalizes embeddings to unit-norm float32 on
# insert, so callers pass float32 - but the math is fine at float64 too,
# which HDBSCAN promotes to internally. `Any` for the shape annotation
# matches the numpy convention here; mypy strict accepts this.
FloatMatrix = npt.NDArray[np.floating[Any]]


_HDBSCAN_AVAILABLE: bool | None = None
_HDBSCAN_LOCK = threading.Lock()


def _hdbscan_available() -> bool:
    """Cached availability check for the optional `hdbscan` extra.

    Audit M-22: the global cache write is double-checked under
    ``_HDBSCAN_LOCK`` so two threads hitting ``cluster_vectors``
    concurrently under no-GIL (PEP 703) builds can't both observe
    ``None``, both run the import, and race the assignment.  Under GIL
    the race is benign (both writes set the same value); under no-GIL
    the check-then-set isn't atomic.
    """
    global _HDBSCAN_AVAILABLE
    if _HDBSCAN_AVAILABLE is not None:
        return _HDBSCAN_AVAILABLE
    with _HDBSCAN_LOCK:
        # Double-checked locking: a concurrent caller may have populated
        # the cache between our first read and the lock acquire.
        if _HDBSCAN_AVAILABLE is None:
            try:
                import hdbscan  # type: ignore[import-untyped]  # noqa: F401

                _HDBSCAN_AVAILABLE = True
            except ImportError:
                _HDBSCAN_AVAILABLE = False
    return _HDBSCAN_AVAILABLE


ClusterMethod = Literal["auto", "hdbscan", "agglomerative"]


@dataclass(frozen=True, slots=True)
class ClusterParams:
    """Parameters of the clustering pass.

    `min_cluster_size` is the smallest acceptable cluster (singletons and
    near-singletons are dropped on both paths). `cohesion_threshold` is
    the cosine-similarity floor for the agglomerative single-link merge -
    items below the floor are never linked. HDBSCAN ignores it (HDBSCAN's
    own `min_cluster_size` does the analogous job).

    `method="auto"` picks HDBSCAN when it is installed and `N >=
    auto_hdbscan_min_n`, agglomerative otherwise. The default
    `auto_hdbscan_min_n=50` reflects HDBSCAN's preference for larger N -
    on tiny inputs the density estimate is too noisy to trust.
    """

    min_cluster_size: int = 3
    cohesion_threshold: float = 0.6
    method: ClusterMethod = "auto"
    auto_hdbscan_min_n: int = 50

    def __post_init__(self) -> None:
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}")
        if not 0.0 <= self.cohesion_threshold <= 1.0:
            raise ValueError(
                f"cohesion_threshold must be in [0, 1], got {self.cohesion_threshold!r}"
            )
        if self.method not in ("auto", "hdbscan", "agglomerative"):
            raise ValueError(f"method must be auto/hdbscan/agglomerative, got {self.method!r}")
        if self.auto_hdbscan_min_n < 2:
            raise ValueError(f"auto_hdbscan_min_n must be >= 2, got {self.auto_hdbscan_min_n}")


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """One cluster as a set of input-vector indices and its cohesion.

    `cohesion` is the average pairwise cosine similarity within the
    cluster, in [0, 1]. Stage 5's storage layer stores this on
    `Cluster.cohesion`.
    """

    members: tuple[int, ...]
    cohesion: float


def cohesion(vectors: FloatMatrix) -> float:
    """Average pairwise cosine similarity of the rows of `vectors`.

    Assumes rows are already unit-norm (Engram normalizes embeddings on
    insert). For singletons returns 1.0 - a single point is perfectly
    self-similar.
    """
    if vectors.ndim != 2:
        raise ValueError(f"expected (N, D) array, got shape {vectors.shape}")
    n = vectors.shape[0]
    if n <= 1:
        return 1.0
    sims = vectors @ vectors.T
    # Sum off-diagonal entries; divide by N * (N - 1) for an unordered
    # average over all i != j pairs.
    total = float(sims.sum() - np.trace(sims))
    avg: float = total / (n * (n - 1))
    # Clamp out fp noise that occasionally pushes the value microscopically
    # outside [-1, 1] for fully-aligned inputs.
    if avg > 1.0:
        return 1.0
    if avg < -1.0:
        return -1.0
    return avg


_DEFAULT_CLUSTER_PARAMS = ClusterParams()


def cluster(
    vectors: FloatMatrix,
    *,
    params: ClusterParams = _DEFAULT_CLUSTER_PARAMS,
) -> list[ClusterAssignment]:
    """Group rows of `vectors` into clusters.

    Returns a list of `ClusterAssignment`. Order is stable: clusters are
    sorted by their smallest member index ascending, and members within
    each cluster are also ascending. Stable order matters for the engine
    (deterministic abstraction prompts) and for the test suite.
    """
    if vectors.ndim != 2:
        raise ValueError(f"expected (N, D) array, got shape {vectors.shape}")
    n = vectors.shape[0]
    if n < params.min_cluster_size:
        return []

    method = params.method
    if method == "auto":
        method = (
            "hdbscan"
            if (_hdbscan_available() and n >= params.auto_hdbscan_min_n)
            else "agglomerative"
        )

    if method == "hdbscan":
        return _cluster_hdbscan(vectors, params)
    return _cluster_agglomerative(vectors, params)


def _cluster_hdbscan(vectors: FloatMatrix, params: ClusterParams) -> list[ClusterAssignment]:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        metric="euclidean",
        # `min_samples=min_cluster_size` is HDBSCAN's recommended default;
        # we surface it explicitly so production tuners know what they're
        # changing. `cluster_selection_method='eom'` is also the default
        # but matters here: it picks the most-stable cluster from the
        # condensed tree rather than aggressively over-splitting.
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(vectors.astype(np.float64))
    return _assignments_from_labels(vectors, labels, params)


def _check_unit_norm(vectors: FloatMatrix) -> None:
    """Audit M-54: one-time warning if the input doesn't look unit-norm.

    Agglomerative + cohesion math assume rows are unit-norm so the
    similarity matrix equals cosine similarity. A caller that hands in
    un-normalized vectors gets silently degraded scores. We sample
    every row's norm via `linalg.norm` along axis=1 and warn (once
    per process) if any row deviates from 1.0 by more than
    `_UNIT_NORM_TOL`. The check is O(N*D) numpy and adds a few
    microseconds even on a 10k-row input.
    """
    global _NORM_WARNING_EMITTED
    if _NORM_WARNING_EMITTED or vectors.shape[0] == 0:
        return
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(np.abs(norms - 1.0) > _UNIT_NORM_TOL):
        worst = float(np.max(np.abs(norms - 1.0)))
        _LOG.warning(
            "consolidation.clustering: input vectors are not unit-norm "
            "(max |norm-1| = %.4f); cohesion math degrades to dot-product. "
            "Normalize before clustering to restore cosine-similarity "
            "semantics.",
            worst,
        )
        _NORM_WARNING_EMITTED = True


def _cluster_agglomerative(vectors: FloatMatrix, params: ClusterParams) -> list[ClusterAssignment]:
    """Single-link agglomerative threshold clustering.

    Audit H-56: the prior implementation walked the upper triangle of
    the (N, N) similarity matrix in pure Python (`for i in range(n):
    for j in range(i+1, n): if sims[i, j] >= t: union(i, j)`). For
    N=10k that's 5*10^7 Python iterations + 5*10^7 bound checks, ~30 s
    of CPU on a typical laptop. The fix vectorizes the edge-discovery
    pass: `np.triu(sims >= threshold, k=1)` produces the boolean
    upper-triangle mask in numpy; `np.argwhere` lifts the `(i, j)`
    coordinates into one array; the Python loop now iterates only
    over edges that actually exceed the threshold (which is the right
    asymptotic — for cohesion >= 0.6 on real corpora it's a small
    fraction of N^2).
    """
    n = vectors.shape[0]
    _check_unit_norm(vectors)
    sims = vectors @ vectors.T  # (N, N), cosine sim for unit-norm rows
    parent = list(range(n))

    def find(x: int) -> int:
        # Path compression.
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Lower index becomes root - keeps the assignment order
            # deterministic across runs.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Vectorize the edge-discovery pass: boolean mask of (i, j) pairs
    # where i < j AND sim >= threshold. `np.triu(..., k=1)` zeroes the
    # diagonal + lower triangle; `np.argwhere` returns the (i, j)
    # coordinates of the True entries as one (#edges, 2) int array.
    # Iteration order matches the prior nested-loop order
    # (lexicographic i ascending, then j ascending) because argwhere
    # returns rows in row-major scan order — same determinism guarantee.
    if n >= 2:
        edge_mask = np.triu(sims >= params.cohesion_threshold, k=1)
        for i, j in np.argwhere(edge_mask):
            union(int(i), int(j))

    labels = np.array([find(idx) for idx in range(n)], dtype=np.int64)
    return _assignments_from_labels(vectors, labels, params)


def _assignments_from_labels(
    vectors: FloatMatrix,
    labels: Sequence[int] | npt.NDArray[np.int64],
    params: ClusterParams,
) -> list[ClusterAssignment]:
    """Bucket indices by `labels` (skip the HDBSCAN noise label `-1`),
    drop buckets smaller than `min_cluster_size`, compute cohesion, and
    return them in stable order."""
    buckets: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        lbl = int(label)
        if lbl == -1:
            continue
        buckets.setdefault(lbl, []).append(idx)

    out: list[ClusterAssignment] = []
    for members in buckets.values():
        if len(members) < params.min_cluster_size:
            continue
        members.sort()
        coh = cohesion(vectors[np.asarray(members)])
        out.append(ClusterAssignment(members=tuple(members), cohesion=coh))
    out.sort(key=lambda c: c.members[0])
    return out
