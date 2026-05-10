"""Performance smoke tests for `Memory.observe` and `Memory.retrieve`.

Stage 3 DoD:
  - P50 < 50 ms per observe at 10k events on a laptop SSD.
  - P50 < 100 ms per retrieve at 10k events.

Marked `slow`; run with `pytest -m slow`. Uses the deterministic
`FakeEmbedder` so latency reflects Engram itself (storage + math)
rather than network I/O to a real provider.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from engram import Memory, SqliteStorage
from engram.providers import FakeEmbedder


def _percentile(samples: list[float], pct: float) -> float:
    samples_sorted = sorted(samples)
    idx = int(len(samples_sorted) * pct / 100)
    idx = min(max(idx, 0), len(samples_sorted) - 1)
    return samples_sorted[idx]


@pytest.mark.slow
def test_observe_p50_under_50ms_at_10k(tmp_path: Path) -> None:
    storage = SqliteStorage(tmp_path / "observe-perf.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=128)
    memory = Memory(storage=storage, embedder=embedder)

    # Warm-up corpus: 10k events.
    with storage.transaction():
        for i in range(10_000):
            memory.observe(f"warmup-{i}")

    # Measure 100 observes against the warmed-up store.
    timings_ms: list[float] = []
    for i in range(100):
        start = time.perf_counter()
        memory.observe(f"measured-{i}")
        timings_ms.append((time.perf_counter() - start) * 1000)

    p50 = _percentile(timings_ms, 50)
    p99 = _percentile(timings_ms, 99)
    assert p50 < 50.0, f"observe P50 = {p50:.2f}ms (budget 50ms); P99 = {p99:.2f}ms"
    storage.close()


@pytest.mark.slow
def test_retrieve_p50_under_100ms_at_10k(tmp_path: Path) -> None:
    storage = SqliteStorage(tmp_path / "retrieve-perf.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=128)
    memory = Memory(storage=storage, embedder=embedder)

    with storage.transaction():
        for i in range(10_000):
            memory.observe(f"event-{i}")

    timings_ms: list[float] = []
    for i in range(100):
        start = time.perf_counter()
        memory.retrieve(f"event-{i}", k=10)
        timings_ms.append((time.perf_counter() - start) * 1000)

    p50 = _percentile(timings_ms, 50)
    p99 = _percentile(timings_ms, 99)
    assert p50 < 100.0, f"retrieve P50 = {p50:.2f}ms (budget 100ms); P99 = {p99:.2f}ms"
    storage.close()
