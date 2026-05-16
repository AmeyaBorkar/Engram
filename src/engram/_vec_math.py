"""Vector helpers shared across the codebase.

Multiple modules used to declare their own pure-Python `_normalize` —
identical implementations of `sqrt(sum(x*x))` over a list, which is
~50x slower than the numpy equivalent on dim=1024 vectors.  The hot
retrieve / observe / consolidate paths all hit one of those copies
per call, so centralizing both removes drift risk and reclaims the
perf.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def normalize(vec: Sequence[float]) -> list[float]:
    """L2-normalize `vec` to unit length; return a python list.

    Returns the unmodified input list when the vector is exactly zero
    (`norm == 0`), matching the documented contract of every former
    `_normalize` copy.  Callers that want to reject zero-norm input
    should validate before calling — see memory.observe / decay /
    consolidate which now log on that path.
    """
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not math.isfinite(norm):
        return [float(x) for x in vec]
    return (arr / norm).tolist()
