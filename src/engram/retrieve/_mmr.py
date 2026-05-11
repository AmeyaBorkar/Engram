"""Maximal Marginal Relevance (MMR) diversity rerank.

After the cross-encoder rerank, the top-K can be dominated by
near-duplicates -- five separate "[user] What's my graduation degree?"
turns surface as five separate hits because the haystack rephrased the
same exchange. MMR re-orders the pool to balance relevance against
diversity:

    MMR(d) = λ · rel(d, q) - (1 - λ) · max_{d' in S} sim(d, d')

where `S` is the running set of already-selected items. `λ = 1.0` is
pure relevance (= no MMR effect); `λ = 0.0` is pure diversity. The
LongMemEval-friendly sweet spot lives around `0.6 - 0.8`.

Pairwise similarity is the cosine of the stored dense embeddings (the
same vectors retrieval ranked over). The cross-encoder is not re-run
for similarity -- that would be N² rerank calls, an order of magnitude
more expensive than the diversity gain.

Relevance scores arrive in whatever range the upstream reranker
emits (BGE-reranker-v2-m3 produces unbounded logits typically in
[-8, +8]); we min-max normalize them to [0, 1] internally so the
diversity term (cosine, naturally bounded in [0, 1]) and the
relevance term occupy the same dynamic range. That way `λ` actually
controls the trade-off the docstring promises -- without this
normalization, a wide-range relevance score dominates the redundancy
penalty and MMR degenerates to relevance sort.

Vectorized: similarity is a single `matrix @ matrix[selected]` per
greedy step. For a 30-candidate pool that's ~30µs total versus ~10ms
for the Python-loop version.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def mmr_select(
    items: Sequence[T],
    relevance: Sequence[float],
    vectors: Sequence[Sequence[float] | None],
    *,
    k: int,
    lambda_: float = 0.7,
) -> list[T]:
    """Greedy MMR over `items`.

    Items with `None` for their vector are treated as having zero
    similarity to every other item (no diversity pressure either way).
    That's the safest fallback when an embedding lookup failed -- the
    item still gets MMR's relevance bonus from the `λ · rel` term.

    Returns up to `k` items in MMR order. Stable: tie-breaks fall back
    to the original relevance ranking (which itself was stable).
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    n = len(items)
    if n != len(relevance) or n != len(vectors):
        raise ValueError(
            f"items / relevance / vectors length mismatch: "
            f"{n} / {len(relevance)} / {len(vectors)}"
        )
    if n == 0 or k == 0:
        return []

    relevance_raw = np.asarray(relevance, dtype=np.float32)
    # Min-max normalize relevance into [0, 1]. The pairwise diversity
    # term (cosine on unit-norm vectors) is already in [0, 1], so this
    # puts both halves of the MMR objective on the same scale. Without
    # this, BGE-reranker logits in [-8, +8] dwarf the [0, 1] diversity
    # penalty and `λ` becomes a no-op for typical relevance gaps.
    rel_min = float(relevance_raw.min())
    rel_max = float(relevance_raw.max())
    if rel_max > rel_min:
        relevance_np = (relevance_raw - rel_min) / (rel_max - rel_min)
    else:
        # All relevance identical -> normalize to 0 so the diversity
        # term carries the entire selection signal.
        relevance_np = np.zeros_like(relevance_raw)
    valid_mask = np.fromiter(
        (v is not None for v in vectors), dtype=bool, count=n
    )
    # If no vectors at all, fall back to top-k by raw relevance (the
    # normalized version would be order-preserving but easier to read
    # in the raw form when there's nothing else going on).
    if not valid_mask.any():
        order = np.argsort(-relevance_raw, kind="stable")[:k]
        return [items[int(i)] for i in order]

    # Build a (n, dim) matrix; rows for items with None vectors stay
    # zero so their dot product with anything is zero. We normalize
    # per-row so dot product == cosine (vectors are typically already
    # unit-norm from storage but we don't assume).
    dim = 0
    for v in vectors:
        if v is not None:
            dim = len(v)
            break
    matrix = np.zeros((n, dim), dtype=np.float32)
    for i, v in enumerate(vectors):
        if v is not None:
            matrix[i] = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero: zero-norm rows stay zero after divide.
    norms_safe = np.where(norms > 0, norms, 1.0)
    matrix = matrix / norms_safe

    remaining = np.ones(n, dtype=bool)
    redundancy = np.zeros(n, dtype=np.float32)
    selected: list[int] = []
    k_eff = min(k, n)
    while len(selected) < k_eff:
        scores = lambda_ * relevance_np - (1.0 - lambda_) * redundancy
        masked = np.where(remaining, scores, -math.inf)
        best = int(np.argmax(masked))
        if not math.isfinite(float(masked[best])):
            # Nothing remaining; defensive break.
            break
        selected.append(best)
        remaining[best] = False
        # Update each item's redundancy with the cosine to the newly
        # selected item. Zero-norm rows produce sim=0 and stay clamped.
        sims = matrix @ matrix[best]
        # Mask out items that had no vector: they keep their zero
        # redundancy contribution (no diversity pressure).
        sims = np.where(valid_mask, sims, 0.0)
        redundancy = np.maximum(redundancy, sims)

    return [items[i] for i in selected]
