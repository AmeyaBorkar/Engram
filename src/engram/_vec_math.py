"""Pure vector math helpers.

Small, dependency-free utilities for the cosine / dot pipeline that runs
through retrieval, consolidation, and reconciliation. Each function here
is pure (no I/O, no globals beyond `math`) so it can be unit-tested in
isolation and the property tests (Hypothesis) hit the exact code path
production calls.

Why a dedicated module rather than letting each caller carry its own
`_normalize`: every caller had a slightly different opinion about what
to do on the zero-norm and dimension-mismatch edges. The audit found at
least three copies of `_normalize` that silently returned an
unnormalized vector on zero-norm -- which is a load-bearing footgun for
anything downstream that divides by the magnitude later (cosine becomes
NaN).

These helpers raise on the edge cases by default; legacy callers that
want soft behavior pass `raise_on_zero=False` explicitly. New callers
SHOULD prefer this module over rolling their own; module-local copies
in the retrieve / consolidation / reconcile / memory modules will be
migrated in a follow-up pass to keep this audit fix focused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def normalize(
    vec: Sequence[float],
    *,
    raise_on_zero: bool = True,
    expected_dim: int | None = None,
) -> list[float]:
    """L2-normalize `vec`.

    Parameters
    ----------
    vec:
        Input vector. Accepts any sequence of floats; the function copies
        into a fresh list so the caller's input is not aliased.
    raise_on_zero:
        Default True. When the L2 norm rounds to zero (the input is the
        zero vector, or every entry is sub-epsilon), raises `ValueError`
        rather than silently returning the input unchanged. Soft behavior
        was the previous default in scattered `_normalize` copies; it
        produces NaN downstream in cosine-similarity math, so the new
        default is loud. Callers that legitimately need the soft path --
        e.g. test fixtures that allow an all-zero query as a sentinel --
        pass `raise_on_zero=False` and accept the un-normalized vector
        when the norm rounds to zero.
    expected_dim:
        Optional dimension check. When set, the input MUST have this
        length or a `ValueError` is raised. This catches the common bug
        where an embedder returns a shorter vector than expected (an
        empty embedding from a degenerate prompt, a truncated batch),
        which otherwise propagates as a "tried to slice past end" error
        many layers later.

    Returns
    -------
    list[float]
        A new list whose L2 norm is 1 (or, in the soft-zero case, an
        unchanged copy of the input).

    Raises
    ------
    ValueError
        On dimension mismatch (when `expected_dim` is set), or on
        zero norm (when `raise_on_zero=True`).
    """
    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(f"vector dimension {len(vec)} does not match expected_dim {expected_dim}")
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        if raise_on_zero:
            raise ValueError(
                "cannot normalize a zero-norm vector (pass raise_on_zero=False to accept this case)"
            )
        return list(vec)
    return [x / norm for x in vec]


def cosine_similarity(
    a: Sequence[float],
    b: Sequence[float],
    *,
    expected_dim: int | None = None,
) -> float:
    """Cosine similarity in [-1, 1] between two vectors.

    Both inputs are normalized inline (no zero-norm fallback -- a zero
    vector cannot have an angle, so we raise). When the vectors are
    already known to be unit-norm, prefer `dot(a, b)` directly.

    `expected_dim` mirrors `normalize()`: if set, both vectors must have
    that length.
    """
    if expected_dim is not None and (len(a) != expected_dim or len(b) != expected_dim):
        raise ValueError(
            f"vector dimensions ({len(a)}, {len(b)}) do not match expected_dim {expected_dim}"
        )
    if len(a) != len(b):
        raise ValueError(f"cosine_similarity input dimensions differ: {len(a)} vs {len(b)}")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine_similarity is undefined for a zero-norm vector")
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain dot product. Raises on dimension mismatch.

    `zip(strict=True)` enforces equal lengths so a silent truncation
    against a shorter vector cannot occur (which would otherwise
    produce a misleading similarity below the actual value).
    """
    if len(a) != len(b):
        raise ValueError(f"dot product dimensions differ: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


__all__ = [
    "cosine_similarity",
    "dot",
    "normalize",
]
