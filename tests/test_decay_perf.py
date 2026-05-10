"""Performance smoke tests for the decay engine.

Stage 4 SCOREBOARD targets:
  * `decay.record` per-signal P50 < 5 ms / P99 < 20 ms
  * `decay.tick` over 10k hot items P50 < 500 ms / P99 < 2000 ms

Marked `slow`; run with `pytest -m slow`. Uses the deterministic
`FakeEmbedder` so latency reflects Engram itself (storage + math)
rather than network I/O to a real provider.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from engram import DecayParams, Memory, SqliteStorage
from engram.providers import FakeEmbedder
from engram.schemas import ItemKind


def _percentile(samples: list[float], pct: float) -> float:
    samples_sorted = sorted(samples)
    idx = int(len(samples_sorted) * pct / 100)
    idx = min(max(idx, 0), len(samples_sorted) - 1)
    return samples_sorted[idx]


@pytest.mark.slow
def test_record_signal_p50_under_5ms(tmp_path: Path) -> None:
    storage = SqliteStorage(tmp_path / "record-perf.db")
    storage.initialize()
    memory = Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=128),
        decay_params=DecayParams(half_life_seconds=1e9, threshold=0.0),
    )

    # Seed 10k events so the per-signal lookup hits a realistic-sized table.
    ids = []
    with storage.transaction():
        for i in range(10_000):
            ev = memory.observe(f"event-{i}")
            ids.append(ev.id)

    timings_ms: list[float] = []
    for i in range(200):
        start = time.perf_counter()
        memory.reinforce(ids[i % len(ids)], ItemKind.EVENT)
        timings_ms.append((time.perf_counter() - start) * 1000)

    p50 = _percentile(timings_ms, 50)
    p99 = _percentile(timings_ms, 99)
    assert p50 < 5.0, f"record P50 = {p50:.2f}ms (budget 5ms); P99 = {p99:.2f}ms"
    storage.close()


@pytest.mark.slow
def test_tick_p50_under_500ms_at_10k(tmp_path: Path) -> None:
    storage = SqliteStorage(tmp_path / "tick-perf.db")
    storage.initialize()
    memory = Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=128),
        decay_params=DecayParams(half_life_seconds=1e9, threshold=0.0),
    )

    with storage.transaction():
        for i in range(10_000):
            memory.observe(f"event-{i}")

    timings_ms: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        memory.tick()
        timings_ms.append((time.perf_counter() - start) * 1000)

    p50 = _percentile(timings_ms, 50)
    assert p50 < 500.0, f"tick P50 = {p50:.2f}ms over 10k items (budget 500ms)"
    storage.close()
