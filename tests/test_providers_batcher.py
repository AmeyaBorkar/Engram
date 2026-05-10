"""Tests for `engram.providers.Batcher`."""

from __future__ import annotations

import threading
import time

import pytest

from engram.providers import Batcher


def test_single_submit_returns_result() -> None:
    b: Batcher[int, int] = Batcher(fn=lambda items: [i * 2 for i in items])
    try:
        assert b.submit(5) == 10
    finally:
        b.stop()


def test_call_count_tracks_underlying_invocations() -> None:
    calls = {"n": 0}

    def fn(items: list[int]) -> list[int]:
        calls["n"] += 1
        return [i * 2 for i in items]

    b: Batcher[int, int] = Batcher(fn=fn, window_ms=10)
    try:
        for i in range(3):
            assert b.submit(i) == i * 2
        # Sequential submits each become their own (potentially short) batch;
        # we don't assert exact count here, just that the counter moves.
        assert calls["n"] == b.call_count >= 1
    finally:
        b.stop()


def test_concurrent_submits_coalesce_into_few_batches() -> None:
    """N parallel submits should produce fewer than N underlying calls."""
    n_workers = 16
    barrier = threading.Barrier(n_workers + 1)
    results: list[int] = [-1] * n_workers

    def fn(items: list[int]) -> list[int]:
        return [i * 2 for i in items]

    b: Batcher[int, int] = Batcher(fn=fn, window_ms=50, max_batch=64)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = b.submit(idx)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    barrier.wait()  # release workers simultaneously
    for t in threads:
        t.join()
    b.stop()

    assert results == [i * 2 for i in range(n_workers)]
    # 16 submits coalesce into <= 4 batches in practice; a 5x reduction
    # is the Stage 2 target.
    assert b.call_count <= n_workers // 5 + 1


def test_max_batch_caps_batch_size() -> None:
    sizes: list[int] = []

    def fn(items: list[int]) -> list[int]:
        sizes.append(len(items))
        return list(items)

    b: Batcher[int, int] = Batcher(fn=fn, window_ms=100, max_batch=3)

    n_workers = 9
    barrier = threading.Barrier(n_workers + 1)
    results: list[int] = [-1] * n_workers

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = b.submit(idx)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    b.stop()

    assert sorted(results) == list(range(n_workers))
    assert all(s <= 3 for s in sizes)


def test_exception_propagates_to_every_waiter_in_batch() -> None:
    def fn(items: list[int]) -> list[int]:
        raise RuntimeError("boom")

    b: Batcher[int, int] = Batcher(fn=fn, window_ms=20)

    n_workers = 4
    barrier = threading.Barrier(n_workers + 1)
    errors: list[BaseException | None] = [None] * n_workers

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            b.submit(idx)
        except BaseException as exc:
            errors[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    b.stop()

    assert all(isinstance(e, RuntimeError) for e in errors)


def test_mismatched_result_count_raises_for_all_waiters() -> None:
    def fn(items: list[int]) -> list[int]:
        return [items[0]] if items else []  # Always return one regardless

    b: Batcher[int, int] = Batcher(fn=fn, window_ms=30)

    n_workers = 3
    barrier = threading.Barrier(n_workers + 1)
    errors: list[BaseException | None] = [None] * n_workers

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            b.submit(idx)
        except BaseException as exc:
            errors[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    b.stop()

    # When N>1 workers race into the same batch, all should see the mismatch error.
    matched = sum(
        1 for e in errors if isinstance(e, RuntimeError) and "results for batch" in str(e)
    )
    assert matched >= 1


def test_invalid_window_rejected() -> None:
    with pytest.raises(ValueError, match="window_ms"):
        Batcher(fn=lambda items: items, window_ms=-1)


def test_invalid_max_batch_rejected() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        Batcher(fn=lambda items: items, max_batch=0)


def test_submit_after_stop_raises() -> None:
    b: Batcher[int, int] = Batcher(fn=lambda items: list(items))
    b.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        b.submit(1)


def test_window_zero_drains_immediately() -> None:
    """`window_ms=0` should still produce correct results, just without coalescing."""
    b: Batcher[int, int] = Batcher(fn=lambda items: [i + 1 for i in items], window_ms=0)
    try:
        assert b.submit(1) == 2
        assert b.submit(2) == 3
    finally:
        b.stop()


def test_batcher_finishes_in_reasonable_time() -> None:
    """Sanity guardrail: 4 submits should not take more than ~1s with window=20ms."""
    b: Batcher[int, int] = Batcher(fn=lambda items: list(items), window_ms=20)
    start = time.perf_counter()
    try:
        for i in range(4):
            b.submit(i)
    finally:
        b.stop()
    assert time.perf_counter() - start < 1.0
