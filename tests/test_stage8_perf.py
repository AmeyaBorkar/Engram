"""Stage 8 latency budgets.

Pinned per-call P50/P99 budgets for the new Stage 8 hot paths:

  * `search_memory_item_embeddings_as_of` -- the temporal-aware
    variant of memory-item search. Over-fetches by `candidate_multiplier`
    (default 4x) from the in-memory vector index, then SQL-filters the
    candidates by validity. At 100k items the extra cost should stay
    well under the existing Stage 6 retrieve budget.
  * `Memory.reconcile` -- 3 storage round-trips (get_conflict +
    invalidate_memory_item + resolve_conflict). Should be sub-50 ms
    in steady state.
  * `Memory.list_conflicts` -- one SQL query with a small LIMIT.

All tests are `@pytest.mark.slow` (~10 s warm-up). They run via
`pytest -m slow`; CI gates the slow lane separately so the regular
test suite stays fast.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from engram import (
    Conflict,
    Memory,
    Resolution,
    SqliteStorage,
)
from engram.providers._fake import FakeEmbedder
from engram.schemas import Embedding, ItemKind, Level, MemoryItem

N_ITEMS = 100_000
WARMUP = 5
TIMED_CALLS = 200

# Budgets (laptop-grade). All in milliseconds.
# `as_of` retrieve carries an over-fetch + SQL filter on top of the
# Stage 6 path; budget is loosened ~50% over the Stage 6 budget.
AS_OF_P50_MS = 225.0
AS_OF_P99_MS = 750.0
RECONCILE_P50_MS = 25.0
RECONCILE_P99_MS = 100.0
LIST_CONFLICTS_P50_MS = 10.0
LIST_CONFLICTS_P99_MS = 50.0


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = round((p / 100.0) * (len(s) - 1))
    return s[idx]


def _seed_memory_corpus(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
    *,
    n: int,
    batch: int = 1_000,
) -> list[MemoryItem]:
    """Plant `n` memory items + embeddings. Returns the items so the
    caller can stitch conflicts/invalidations on a few of them."""
    items: list[MemoryItem] = []
    for i in range(n):
        items.append(
            MemoryItem(
                level=Level.SUMMARY,
                content=f"sample summary {i}",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
    storage.insert_memory_items(items)
    for start in range(0, n, batch):
        chunk = items[start : start + batch]
        vecs = embedder.embed([m.content for m in chunk])
        with storage.transaction():
            for m, v in zip(chunk, vecs, strict=True):
                storage.insert_embedding(
                    Embedding(
                        item_id=m.id,
                        item_kind=ItemKind.MEMORY_ITEM,
                        model=embedder.model,
                        dim=embedder.dim,
                        vector=tuple(v),
                    )
                )
    return items


@pytest.mark.slow
def test_search_memory_item_embeddings_as_of_under_budget(
    tmp_path: Path,
) -> None:
    storage = SqliteStorage(tmp_path / "as_of.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=128)
    try:
        items = _seed_memory_corpus(storage, embedder, n=N_ITEMS)
        # Invalidate 1% of items so the SQL filter has actual work.
        for i, item in enumerate(items[:: N_ITEMS // 1000]):
            storage.invalidate_memory_item(
                item.id,
                at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            )

        memory = Memory(storage=storage, embedder=embedder)
        # Warm up the vector index.
        for _ in range(WARMUP):
            memory.retrieve("warmup", k=10, reinforce=False)

        per_call_ms: list[float] = []
        for i in range(TIMED_CALLS):
            idx = (i * 113) % N_ITEMS
            t0 = time.perf_counter()
            memory.retrieve(
                f"sample summary {idx}",
                k=10,
                reinforce=False,
                as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
            per_call_ms.append((time.perf_counter() - t0) * 1000.0)

        p50 = _percentile(per_call_ms, 50.0)
        p99 = _percentile(per_call_ms, 99.0)
        assert p50 < AS_OF_P50_MS, (
            f"as_of retrieve P50 = {p50:.1f} ms (budget {AS_OF_P50_MS} ms)"
        )
        assert p99 < AS_OF_P99_MS, (
            f"as_of retrieve P99 = {p99:.1f} ms (budget {AS_OF_P99_MS} ms)"
        )
    finally:
        storage.close()


@pytest.mark.slow
def test_reconcile_under_budget(tmp_path: Path) -> None:
    """Pre-seed many pending conflicts, reconcile them in a loop, time
    each call. Excludes MERGE (which would dominate timings via the
    LLM call) -- the budget is for the storage round-trip cost."""
    storage = SqliteStorage(tmp_path / "reconcile.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=64)
    try:
        # Seed pairs and conflicts. Each pair = a + b sharing a vector.
        n_pairs = 1_000
        pairs: list[tuple[MemoryItem, MemoryItem]] = []
        items: list[MemoryItem] = []
        for i in range(n_pairs):
            a = MemoryItem(
                level=Level.SUMMARY,
                content=f"a-{i}",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            b = MemoryItem(
                level=Level.SUMMARY,
                content=f"b-{i}",
                created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                valid_from=datetime(2026, 2, 1, tzinfo=timezone.utc),
            )
            items.extend((a, b))
            pairs.append((a, b))
        storage.insert_memory_items(items)
        conflict_ids: list[UUID] = []
        for a, b in pairs:
            c = Conflict(source_item_id=b.id, target_item_id=a.id, similarity=0.9)
            storage.record_conflict(c)
            conflict_ids.append(c.id)
        memory = Memory(storage=storage, embedder=embedder)

        # Measure per-reconcile latency on a subset.
        per_call_ms: list[float] = []
        for cid in conflict_ids[:TIMED_CALLS]:
            t0 = time.perf_counter()
            memory.reconcile(
                cid,
                resolution=Resolution.PREFER_RECENT,
                now=datetime(2026, 5, 1, tzinfo=timezone.utc),
            )
            per_call_ms.append((time.perf_counter() - t0) * 1000.0)

        p50 = _percentile(per_call_ms, 50.0)
        p99 = _percentile(per_call_ms, 99.0)
        assert p50 < RECONCILE_P50_MS, (
            f"reconcile P50 = {p50:.2f} ms (budget {RECONCILE_P50_MS} ms)"
        )
        assert p99 < RECONCILE_P99_MS, (
            f"reconcile P99 = {p99:.2f} ms (budget {RECONCILE_P99_MS} ms)"
        )
    finally:
        storage.close()


@pytest.mark.slow
def test_list_conflicts_under_budget(tmp_path: Path) -> None:
    """list_conflicts is one SQL query; even at 10k rows the slice
    cost should stay under 10 ms P50."""
    storage = SqliteStorage(tmp_path / "list.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=32)
    try:
        n_pairs = 5_000
        items: list[MemoryItem] = []
        for i in range(n_pairs):
            a = MemoryItem(
                level=Level.SUMMARY,
                content=f"a-{i}",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            b = MemoryItem(
                level=Level.SUMMARY,
                content=f"b-{i}",
                created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                valid_from=datetime(2026, 2, 1, tzinfo=timezone.utc),
            )
            items.extend((a, b))
        storage.insert_memory_items(items)
        for i in range(0, len(items), 2):
            c = Conflict(
                source_item_id=items[i + 1].id,
                target_item_id=items[i].id,
                similarity=0.9,
            )
            storage.record_conflict(c)
        memory = Memory(storage=storage, embedder=embedder)
        # Warm-up.
        for _ in range(WARMUP):
            memory.list_conflicts(limit=100)

        per_call_ms: list[float] = []
        for _ in range(TIMED_CALLS):
            t0 = time.perf_counter()
            memory.list_conflicts(limit=100)
            per_call_ms.append((time.perf_counter() - t0) * 1000.0)

        p50 = _percentile(per_call_ms, 50.0)
        p99 = _percentile(per_call_ms, 99.0)
        assert p50 < LIST_CONFLICTS_P50_MS, (
            f"list_conflicts P50 = {p50:.2f} ms "
            f"(budget {LIST_CONFLICTS_P50_MS} ms) at n={n_pairs} pairs"
        )
        assert p99 < LIST_CONFLICTS_P99_MS, (
            f"list_conflicts P99 = {p99:.2f} ms "
            f"(budget {LIST_CONFLICTS_P99_MS} ms) at n={n_pairs} pairs"
        )
    finally:
        storage.close()
