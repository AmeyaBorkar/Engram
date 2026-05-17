"""Vector helpers shared across the codebase.

Multiple modules used to declare their own pure-Python `_normalize` —
identical implementations of `sqrt(sum(x*x))` over a list, which is
~50x slower than the numpy equivalent on dim=1024 vectors.  The hot
retrieve / observe / consolidate paths all hit one of those copies
per call, so centralizing both removes drift risk and reclaims the
perf.

Three helpers:
  * `normalize(vec, *, raise_on_zero=True, expected_dim=None)` —
    L2-normalize to unit length.  Raises on zero-norm by default
    (the old soft behavior silently returned NaN-prone vectors and
    propagated zeros downstream); pass ``raise_on_zero=False`` to
    opt back into the legacy behavior at a specific call site.
  * `dot(a, b)` — inner product with dimension check.
  * `cosine_similarity(a, b, *, expected_dim=None)` — cos(θ) with
    zero-norm and dimension validation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _check_dim(vec: Sequence[float], expected_dim: int | None, name: str) -> None:
    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(
            f"{name} length {len(vec)} does not match expected_dim {expected_dim}"
        )


def normalize(
    vec: Sequence[float],
    *,
    raise_on_zero: bool = True,
    expected_dim: int | None = None,
) -> list[float]:
    """L2-normalize `vec` to unit length; return a python list.

    By default raises ``ValueError`` on a zero-norm or non-finite-norm
    vector — silently returning a zero/NaN vector lets bad input
    propagate through cosine similarity downstream where it surfaces
    as a much more confusing failure.  Pass ``raise_on_zero=False`` to
    keep the legacy soft contract (returns a copy of the input).

    ``expected_dim`` is an optional shape check; useful at every site
    that pulls a vector out of an `Embedding` row and expects it to
    match the provider's `.dim`.
    """
    _check_dim(vec, expected_dim, "vec")
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not math.isfinite(norm):
        if raise_on_zero:
            raise ValueError(
                f"cannot normalize zero-norm vector (norm={norm}); pass "
                f"raise_on_zero=False if a copy of the input is acceptable"
            )
        return [float(x) for x in vec]
    return (arr / norm).tolist()


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Inner product; raises ``ValueError`` on dimension mismatch."""
    if len(a) != len(b):
        raise ValueError(
            f"dot: dimensions differ ({len(a)} vs {len(b)})"
        )
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    return float(np.dot(arr_a, arr_b))


def cosine_similarity(
    a: Sequence[float],
    b: Sequence[float],
    *,
    expected_dim: int | None = None,
) -> float:
    """Cosine similarity between `a` and `b`.

    Raises on zero-norm inputs or dimension mismatch.  ``expected_dim``
    optionally pins both vectors to a specific length.
    """
    if len(a) != len(b):
        raise ValueError(
            f"cosine_similarity: dimensions differ ({len(a)} vs {len(b)})"
        )
    if expected_dim is not None:
        if len(a) != expected_dim:
            raise ValueError(
                f"cosine_similarity: expected_dim={expected_dim} but inputs "
                f"have length {len(a)}"
            )
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    if norm_a == 0.0 or norm_b == 0.0 or not (math.isfinite(norm_a) and math.isfinite(norm_b)):
        raise ValueError(
            f"cosine_similarity: zero-norm input (norms={norm_a}, {norm_b})"
        )
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


__all__ = ["cosine_similarity", "dot", "normalize"]
