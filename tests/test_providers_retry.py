"""Tests for `engram.providers.Retry`."""

from __future__ import annotations

import asyncio
import random

import pytest

from engram.providers import Retry


def test_succeeds_on_first_try() -> None:
    r = Retry(max_attempts=3, sleep=lambda _: None)
    calls = 0

    def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert r.call(fn) == "ok"
    assert calls == 1


def test_recovers_after_transient_failure() -> None:
    r = Retry(max_attempts=3, exceptions=(RuntimeError,), sleep=lambda _: None)
    calls = 0

    def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("flaky")
        return "ok"

    assert r.call(fn) == "ok"
    assert calls == 3


def test_raises_after_max_attempts() -> None:
    r = Retry(max_attempts=2, exceptions=(RuntimeError,), sleep=lambda _: None)
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("never recovers")

    with pytest.raises(RuntimeError, match="never recovers"):
        r.call(fn)
    assert calls == 2


def test_only_retries_listed_exceptions() -> None:
    r = Retry(max_attempts=5, exceptions=(ValueError,), sleep=lambda _: None)
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("not in retry list")

    with pytest.raises(RuntimeError):
        r.call(fn)
    assert calls == 1


def test_delay_grows_exponentially_without_jitter() -> None:
    delays: list[float] = []
    r = Retry(
        max_attempts=4,
        base_delay=0.1,
        max_delay=10.0,
        jitter=False,
        exceptions=(RuntimeError,),
        sleep=delays.append,
    )

    def fn() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        r.call(fn)

    assert delays == pytest.approx([0.1, 0.2, 0.4])


def test_delay_capped_at_max_delay() -> None:
    delays: list[float] = []
    r = Retry(
        max_attempts=10,
        base_delay=1.0,
        max_delay=3.0,
        jitter=False,
        exceptions=(RuntimeError,),
        sleep=delays.append,
    )

    def fn() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        r.call(fn)

    assert max(delays) == pytest.approx(3.0)


def test_jitter_uses_injected_rng_for_determinism() -> None:
    delays_a: list[float] = []
    delays_b: list[float] = []

    def make_retry(out: list[float]) -> Retry:
        return Retry(
            max_attempts=3,
            base_delay=0.1,
            jitter=True,
            exceptions=(RuntimeError,),
            sleep=out.append,
            rng=random.Random(42),
        )

    def fn() -> None:
        raise RuntimeError("boom")

    for r, out in [(make_retry(delays_a), delays_a), (make_retry(delays_b), delays_b)]:
        with pytest.raises(RuntimeError):
            r.call(fn)
    assert delays_a == delays_b
    assert all(d > 0 for d in delays_a)


def test_async_recovers_after_transient_failure() -> None:
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    r = Retry(
        max_attempts=3, base_delay=0.1, jitter=False, exceptions=(RuntimeError,), asleep=fake_sleep
    )
    calls = 0

    async def afn() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise RuntimeError("flaky")
        return "ok"

    assert asyncio.run(r.acall(afn)) == "ok"
    assert calls == 2
    assert sleeps == pytest.approx([0.1])


def test_async_raises_after_max_attempts() -> None:
    async def afn() -> None:
        raise RuntimeError("boom")

    r = Retry(
        max_attempts=2,
        base_delay=0.0,
        jitter=False,
        exceptions=(RuntimeError,),
        asleep=lambda _d: _noop_async(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(r.acall(afn))


async def _noop_async() -> None:
    return None


def test_invalid_max_attempts_rejected() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        Retry(max_attempts=0)


def test_default_exceptions_are_narrow_transient_only() -> None:
    """A bare `Retry()` must not retry on permanent failures like ValueError.

    Wide defaults (Exception) caused real bugs: auth failures, programming
    errors, JSON parse errors all retried 5 times with backoff, burning the
    budget and turning a clear permanent failure into a slow one.
    """
    r = Retry(max_attempts=5, sleep=lambda _: None)
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        r.call(fn)
    assert calls == 1, "ValueError must not be retried by default"


def test_invalid_delays_rejected() -> None:
    with pytest.raises(ValueError, match="invalid delays"):
        Retry(base_delay=5.0, max_delay=1.0)
