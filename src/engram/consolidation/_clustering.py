"""Clustering of normalized embeddings for consolidation.

Two backends, switchable via `ClusterParams.method`:

  `hdbscan`        — density-based, robust to noise, picks cluster sizes
                     automatically. Requires the `hdbscan` package
                     (optional `[consolidation]` extra). For unit-norm
                     vectors we use the Euclidean metric, which is
                     monotonic with cosine distance and lets HDBSCAN use
                     its spatial indexing fast paths.
  `agglomerative`  — single-link agglomerative threshold clustering using
                     numpy vectorized triangle traversal + a union-find.
                     Pure Python beyond numpy, no extra deps. O(N^2)
                     memory for the similarity matrix; the per-pair work
                     is vectorized so wall-time is dominated by the
                     matrix multiply rather than a Python double-loop.

`cluster()` returns a list of `ClusterAssignment` records. Items not in
any cluster (HDBSCAN noise label, or singletons under the threshold) are
not returned - the engine only acts on items that grouped.

Post-processing divergence between the two backends: the agglomerative
path uses `cohesion_threshold` directly as the cosine-similarity floor
for the single-link merge, so it is honored both in cluster *formation*
and (via `cohesion()`) in the reported `ClusterAssignment.cohesion`.
HDBSCAN ignores `cohesion_threshold` entirely - its density-based
selection has no analogous knob; tune `min_cluster_size` instead. The
engine's downstream consumers (`extract_abstraction`, `_detect_conflicts`)
see whatever cohesion the backend reports and clamp it for prompt use,
but they do NOT re-apply `cohesion_threshold` as a filter, so HDBSCAN
can surface a cluster with cohesion below the threshold. Callers that
need a hard floor should post-filter the returned list.

The math is deterministic in both backends: HDBSCAN itself is
deterministic given a fixed input; the agglomerative path uses an
explicit comparison order that doesn't depend on hash randomization.
Replays produce identical cluster assignments.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

_LOG = logging.getLogger("engram.consolidation.clustering")

# A 2D float matrix. Engram normalizes embeddings to unit-norm float32 on
# insert, so callers pass float32 - but the math is fine at float64 too,
# which HDBSCAN promotes to internally. `Any` for the shape annotation
# matches the numpy convention here; mypy strict accepts this.
FloatMatrix = npt.NDArray[np.floating[Any]]


_HDBSCAN_AVAILABLE: bool | None = None


def _hdbscan_available() -> bool:
    """Cached availability check for the optional `hdbscan` extra."""
    global _HDBSCAN_AVAILABLE
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


_UNIT_NORM_WARNED = False


def _warn_non_unit_norm_once(deviation: float) -> None:
    """Log a one-shot warning if the agglomerative path is fed
    non-unit-norm rows.

    The cosine-similarity math is `vectors @ vectors.T`, which only
    equals the cosine for unit-norm rows; Engram normalizes embeddings
    at insert so the contract holds in production. A noisy caller (or a
    sibling agent's test fixture) can silently break the math, so we
    surface it once via the logger instead of asserting -- asserting
    would crash debug runs that legitimately fuzz with random vectors.
    """
    global _UNIT_NORM_WARNED
    if _UNIT_NORM_WARNED:
        return
    _UNIT_NORM_WARNED = True
    _LOG.warning(
        "consolidation: agglomerative cluster received non-unit-norm "
        "vectors (max |1 - ||v|||=%.4f); cosine similarities will be "
        "off the [-1, 1] axis. Normalize embeddings before passing.",
        deviation,
    )


def _cluster_agglomerative(vectors: FloatMatrix, params: ClusterParams) -> list[ClusterAssignment]:
    n = vectors.shape[0]
    # One-time guard: if rows aren't unit-norm, the dot product is not
    # cosine similarity and `cohesion_threshold` becomes meaningless.
    # Engram normalizes at insert so this only trips for misconfigured
    # callers/tests. Cheap check: max row-norm deviation from 1.
    norms = np.linalg.norm(vectors, axis=1)
    max_dev = float(np.abs(norms - 1.0).max()) if n > 0 else 0.0
    if max_dev > 1e-3:
        _warn_non_unit_norm_once(max_dev)

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

    # Vectorized triangle walk: pull the (i, j) coords of every
    # above-threshold cell in the strict upper triangle. For N=10k this
    # turns a 10^8-iteration Python loop into one O(N^2) numpy compare
    # plus a single linear pass over the (typically) sparse hit list.
    # Memory is still O(N^2) (the `sims` matrix dominates); the wall
    # time saving is the Python-bytecode cost of the double loop.
    upper_i, upper_j = np.triu_indices(n, k=1)
    above = sims[upper_i, upper_j] >= params.cohesion_threshold
    # Iterating numpy-ordered pairs (i ascending, then j ascending within
    # each i) preserves the original union-find merge order so the
    # deterministic-replay invariant survives the vectorization.
    hit_i = upper_i[above]
    hit_j = upper_j[above]
    for i, j in zip(hit_i.tolist(), hit_j.tolist(), strict=True):
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
