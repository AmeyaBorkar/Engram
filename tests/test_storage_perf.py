"""Performance smoke tests for storage. Marked `slow`; opt-in.

These exist to back the Stage 1 DoD numbers (1M inserts < 30s, last-1k reads
< 50ms). They are noisy by nature — run on the user's machine, not in CI.

    pytest -m slow
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from engram.schemas import Event
from engram.storage import SqliteStorage


@pytest.mark.slow
def test_insert_one_million_events_under_30s(tmp_path: Path) -> None:
    backend = SqliteStorage(tmp_path / "perf.db")
    backend.initialize()
    try:
        batch_size = 10_000
        total = 1_000_000
        start = time.perf_counter()
        with backend.transaction():
            for _ in range(total // batch_size):
                events = [Event(content="x") for _ in range(batch_size)]
                backend.insert_events(events)
        elapsed = time.perf_counter() - start
        assert backend.count_events() == total
        assert elapsed < 30.0, f"1M inserts took {elapsed:.2f}s (budget 30s)"
    finally:
        backend.close()


@pytest.mark.slow
def test_last_1k_reads_under_50ms(tmp_path: Path) -> None:
    backend = SqliteStorage(tmp_path / "read-perf.db")
    backend.initialize()
    try:
        with backend.transaction():
            backend.insert_events(Event(content=f"e{i}") for i in range(100_000))
        start = time.perf_counter()
        events = backend.list_events(limit=1000)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert len(events) == 1000
        assert elapsed_ms < 50.0, f"reading last 1k took {elapsed_ms:.2f}ms (budget 50ms)"
    finally:
        backend.close()
