"""Pure decay math.

No I/O, no datetimes, no logging. Every function here is a closed-form
expression on dimensionless floats and integers, so this module is the
ground truth for the formula and is the only place we require 100%
coverage.

Formula (per the README and `ROADMAP.md` Stage 4):

    w_{t+1} = clamp01(w_t * alpha^dt + beta*r - delta*x + gamma*c)

Symbols:
  * `w_t`            - current weight, must lie in [0, 1].
  * `alpha`          - per-second decay base, in (0, 1]. Derived from
                       `half_life_seconds` in `DecayParams`.
  * `dt`             - elapsed seconds since the last update; must be >= 0.
  * `r`, `c`, `x`    - non-negative integer counts of reinforcement,
                       corroboration, and contradiction signals received
                       in the elapsed window.
  * `beta`, `gamma`, `delta` - non-negative gains. Sum-of-gains can exceed
                       1 in a single step; the clamp at the end keeps the
                       weight in [0, 1].

The clamp lives at the boundary, not in between, because intermediate
arithmetic on weights is allowed to overshoot - downstream code MUST treat
the function's output as the canonical post-step weight and never re-apply
the formula on a non-clamped value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecayParams:
    """Parameters of the Engram decay formula.

    Defaults are tuned so that an untouched item has a 30-day half-life,
    a single reinforcement is worth ~10% of full weight, corroboration is
    half as strong, and a single contradiction overrides two reinforcements.
    These are starting points, not load-bearing constants - tune per
    workload.

    Bounds (validated in `__post_init__`):
      * `half_life_seconds` > 0
      * `beta`, `gamma`, `delta` >= 0
      * `threshold` in [0, 1]
    """

    half_life_seconds: float = 30.0 * 86400.0  # 30 days
    beta: float = 0.10
    gamma: float = 0.05
    delta: float = 0.20
    threshold: float = 0.05

    def __post_init__(self) -> None:
        if not (self.half_life_seconds > 0 and math.isfinite(self.half_life_seconds)):
            raise ValueError(
                f"half_life_seconds must be a positive finite float, got {self.half_life_seconds!r}"
            )
        for name in ("beta", "gamma", "delta"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be a non-negative finite float, got {value!r}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold!r}")

    @property
    def alpha(self) -> float:
        """Per-second decay base such that `alpha ** half_life_seconds == 0.5`."""
        return float(0.5 ** (1.0 / self.half_life_seconds))


def clamp01(x: float) -> float:
    """Clamp `x` into the closed interval [0, 1].

    `nan` is mapped to 0.0 - the decay engine never observes NaN at its
    inputs (every input is bounds-validated upstream), but if math somehow
    produces one, propagating a NaN weight would silently corrupt every
    downstream comparison. Mapping to 0 fails loudly via the prune path
    instead.
    """
    if math.isnan(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def apply(
    *,
    weight: float,
    dt_seconds: float,
    reinforcement: int = 0,
    corroboration: int = 0,
    contradiction: int = 0,
    params: DecayParams,
) -> float:
    """Apply one step of the decay formula and return the new weight.

    The result is clamped to [0, 1]. Inputs are validated:
      * `weight` must be in [0, 1].
      * `dt_seconds` must be finite and >= 0.
      * signal counts must be non-negative integers.
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight must be in [0, 1], got {weight!r}")
    if not (math.isfinite(dt_seconds) and dt_seconds >= 0.0):
        raise ValueError(f"dt_seconds must be a non-negative finite float, got {dt_seconds!r}")
    for name, count in (
        ("reinforcement", reinforcement),
        ("corroboration", corroboration),
        ("contradiction", contradiction),
    ):
        if count < 0:
            raise ValueError(f"{name} must be >= 0, got {count}")

    decayed = weight * (params.alpha**dt_seconds)
    raw = (
        decayed
        + params.beta * reinforcement
        + params.gamma * corroboration
        - params.delta * contradiction
    )
    return clamp01(raw)


def is_cold(weight: float, params: DecayParams) -> bool:
    """Return True if `weight` is at or below the prune threshold."""
    return weight < params.threshold
