"""Retry primitive — exponential backoff with optional jitter.

Sync and async surfaces. The clock and RNG are injectable so tests are
deterministic.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

_DEFAULT_RNG = random.Random()  # noqa: S311  # jitter is not crypto


@dataclass
class Retry:
    """Retry a callable on configured exceptions, with exponential backoff.

    `delay(attempt) = min(base_delay * 2**attempt, max_delay)` — optionally
    multiplied by `rng.uniform(0.5, 1.5)` when `jitter=True`. The first
    attempt has no preceding delay.

    `sleep` and `asleep` are injectable so tests pass a no-op; `rng` is
    injectable so jitter is reproducible across runs.
    """

    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True
    # Default to the narrow set of transient errors so a misuse of `Retry()`
    # without an explicit `exceptions=` does NOT silently retry on permanent
    # failures (auth, ValueError, AttributeError, etc.).  Provider adapters
    # should pass their concrete transient classes (e.g. RateLimitError,
    # APIConnectionError, InternalServerError, APITimeoutError).
    exceptions: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)

    sleep: Callable[[float], None] = field(default=time.sleep)
    asleep: Callable[[float], Awaitable[None]] = field(default=asyncio.sleep)
    rng: random.Random = field(default_factory=lambda: _DEFAULT_RNG)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.base_delay < 0 or self.max_delay < self.base_delay:
            raise ValueError(
                f"invalid delays: base_delay={self.base_delay}, max_delay={self.max_delay}"
            )

    def call(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Invoke `fn`, retrying on configured exceptions."""
        last_exc: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                return fn(*args, **kwargs)
            except self.exceptions as exc:
                last_exc = exc
                if attempt == self.max_attempts - 1:
                    raise
                self.sleep(self._delay(attempt))
        # Unreachable: the loop either returns or re-raises the last exception.
        raise RuntimeError("retry loop exited without returning") from last_exc

    async def acall(self, fn: Callable[P, Awaitable[T]], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Async variant of `call`."""
        last_exc: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                return await fn(*args, **kwargs)
            except self.exceptions as exc:
                last_exc = exc
                if attempt == self.max_attempts - 1:
                    raise
                await self.asleep(self._delay(attempt))
        raise RuntimeError("retry loop exited without returning") from last_exc

    def _delay(self, attempt: int) -> float:
        delay: float = min(self.base_delay * (2.0**attempt), self.max_delay)
        if self.jitter:
            delay *= self.rng.uniform(0.5, 1.5)
        return delay
