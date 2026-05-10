"""Engram decay engine.

Memory items strengthen with use and weaken with time. The math is the
README's canonical formula:

    w_{t+1} = clamp01(w_t * alpha^dt + beta*r - delta*x + gamma*c)

where:

  * `alpha` is the per-second decay base, determined from a user-facing
    `half_life_seconds` knob: `alpha = 0.5 ** (1.0 / half_life_seconds)`.
    With the default 30-day half-life, an item left untouched halves its
    weight every 30 days.
  * `r`, `c`, `x` are non-negative integer counts of reinforcement,
    corroboration, and contradiction signals that have arrived since the
    last decay update.
  * `beta`, `gamma`, `delta` are the per-signal gains/losses.

The math is pure, dimensionless, and lives in `engram.decay._math`. The
storage-aware engine that batches updates and prunes cold items lives in
`engram.decay._engine` (Stage 4 will add it on top).
"""

from engram.decay._engine import DecayEngine, PrunePolicy, TickResult
from engram.decay._math import (
    DecayParams,
    apply,
    clamp01,
    is_cold,
)

__all__ = [
    "DecayEngine",
    "DecayParams",
    "PrunePolicy",
    "TickResult",
    "apply",
    "clamp01",
    "is_cold",
]
