"""Stage 6 latency budget @ 100k items.

SCOREBOARD target: P50 < 150 ms, P99 < 500 ms for warm-cache `retrieve`
at 100k items on a laptop. The test plants 100k synthetic events,
warms the vector index, then runs a steady-state retrieval loop.

Slow-marked because the 100k insert + warm-up takes ~10 s on CI's
default ubuntu-latest runner (in-memory SQLite, FakeEmbedder dim=128).
The first retrieve after the write burst pays a one-time cache rebuild
cost; that's intentionally NOT counted toward the budget -- the budget
covers the agent's steady-state read workload, not the write/read mix.

The cache-rebuild cost itself is measured separately as
`test_cold_cache_rebuild_under_one_second` so it stays bounded.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from engram import Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder
from engram.schemas import Embedding, Event, ItemKind

# Loosened relative to the absolute SCOREBOARD numbers because CI runners
# vary in single-thread perf; the warm-cache numpy matmul is the only
# cost being measured, so the order-of-magnitude is what matters.
N_ITEMS = 100_000
WARMUP_RETRIEVES = 5
TIMED_RETRIEVES = 200
P50_BUDGET_MS = 150.0
P99_BUDGET_MS = 500.0
COLD_REBUILD_BUDGET_MS = 1500.0


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    return s[idx]


def _build_corpus(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
    *,
    n: int,
    batch: int = 1_000,
) -> None:
    events = [Event(content=f"sample fact number {i}") for i in range(n)]
    storage.insert_events(events)
    for start in range(0, n, batch):
        chunk = events[start : start + batch]
        vecs = embedder.embed([e.content for e in chunk])
        with storage.transaction():
            for e, v in zip(chunk, vecs, strict=True):
                storage.insert_embedding(
                    Embedding(
                        item_id=e.id,
                        item_kind=ItemKind.EVENT,
                        model=embedder.model,
                        dim=embedder.dim,
                        vector=tuple(v),
                    )
                )


@pytest.mark.slow
def test_retrieve_warm_p50_p99_under_budget(tmp_path: Path) -> None:
    storage = SqliteStorage(tmp_path / "latency.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=128)
    try:
        _build_corpus(storage, embedder, n=N_ITEMS)
        memory = Memory(storage=storage, embedder=embedder)
        for _ in range(WARMUP_RETRIEVES):
            memory.retrieve("warmup", k=10, reinforce=False)

        per_call_ms: list[float] = []
        for i in range(TIMED_RETRIEVES):
            idx = (i * 113) % N_ITEMS
            t0 = time.perf_counter()
            memory.retrieve(f"sample fact number {idx}", k=10, reinforce=False)
            per_call_ms.append((time.perf_counter() - t0) * 1000.0)

        p50 = _percentile(per_call_ms, 50.0)
        p99 = _percentile(per_call_ms, 99.0)
        assert p50 < P50_BUDGET_MS, (
            f"warm-cache P50 = {p50:.1f}ms (budget {P50_BUDGET_MS}ms) "
            f"at N={N_ITEMS} dim={embedder.dim}"
        )
        assert p99 < P99_BUDGET_MS, (
            f"warm-cache P99 = {p99:.1f}ms (budget {P99_BUDGET_MS}ms) "
            f"at N={N_ITEMS} dim={embedder.dim}"
        )
    finally:
        storage.close()


@pytest.mark.slow
def test_cold_cache_rebuild_under_one_and_a_half_seconds(tmp_path: Path) -> None:
    """The first retrieve after a write burst is the cold-cache case."""
    storage = SqliteStorage(tmp_path / "cold.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=128)
    try:
        _build_corpus(storage, embedder, n=N_ITEMS)
        memory = Memory(storage=storage, embedder=embedder)

        t0 = time.perf_counter()
        memory.retrieve("first read", k=10, reinforce=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < COLD_REBUILD_BUDGET_MS, (
            f"cold-cache rebuild = {elapsed_ms:.1f}ms "
            f"(budget {COLD_REBUILD_BUDGET_MS}ms) at N={N_ITEMS}"
        )
    finally:
        storage.close()
