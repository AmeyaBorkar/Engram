"""Stage 8 temporal retrieve tests.

Exercises `Memory.retrieve(query, k, *, as_of=None)` against multi-
version invalidation chains. The DoD here is "temporal queries return
historically-correct state": given X invalidated by X' which is itself
invalidated by X'', three queries at three timestamps return the
right item.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from engram import (
    Conflict,
    Embedding,
    ItemKind,
    Level,
    Memory,
    MemoryItem,
    Resolution,
    SqliteStorage,
    Storage,
)
from engram.providers._fake import FakeEmbedder
from engram.reconcile import Reconciler


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def memory(storage: SqliteStorage, embedder: FakeEmbedder) -> Memory:
    return Memory(storage=storage, embedder=embedder)


def _seed_item(
    storage: Storage,
    embedder: FakeEmbedder,
    content: str,
    *,
    created_at: datetime,
    vec: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
) -> MemoryItem:
    item = MemoryItem(
        level=Level.SUMMARY,
        content=content,
        created_at=created_at,
        valid_from=created_at,
    )
    storage.insert_memory_item(item)
    storage.insert_embedding(
        Embedding(
            item_id=item.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=embedder.model,
            dim=8,
            vector=vec,
        )
    )
    return item


def _record_and_resolve(
    storage: Storage,
    *,
    new_item: MemoryItem,
    invalidates: MemoryItem,
    now: datetime,
) -> None:
    """Record + resolve a CONTRADICT with PREFER_RECENT so `invalidates` loses."""
    c = Conflict(
        source_item_id=new_item.id,
        target_item_id=invalidates.id,
        similarity=1.0,
    )
    storage.record_conflict(c)
    Reconciler(storage).reconcile(
        c.id, resolution=Resolution.PREFER_RECENT, now=now
    )


# ---------------------------------------------------------------------------
# Single invalidation chain
# ---------------------------------------------------------------------------


class TestSingleInvalidation:
    def test_default_excludes_invalidated(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        old = _seed_item(
            storage, embedder, "X is true", created_at=_utc(2026, 1, 1)
        )
        new = _seed_item(
            storage, embedder, "X is false", created_at=_utc(2026, 4, 1)
        )
        _record_and_resolve(
            storage, new_item=new, invalidates=old, now=_utc(2026, 5, 1)
        )
        results = memory.retrieve(
            "X",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
        )
        ids = {r.item_id for r in results}
        assert new.id in ids
        assert old.id not in ids

    def test_as_of_before_invalidation_shows_loser(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        old = _seed_item(
            storage, embedder, "X is true", created_at=_utc(2026, 1, 1)
        )
        new = _seed_item(
            storage, embedder, "X is false", created_at=_utc(2026, 4, 1)
        )
        _record_and_resolve(
            storage, new_item=new, invalidates=old, now=_utc(2026, 5, 1)
        )
        # Mid-March 2026: only `old` had been written; new doesn't exist yet.
        results = memory.retrieve(
            "X",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            as_of=_utc(2026, 3, 15),
        )
        ids = {r.item_id for r in results}
        assert old.id in ids
        assert new.id not in ids  # not yet valid

    def test_as_of_in_overlap_window_shows_both(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        """Between new's creation (2026-04-01) and resolution (2026-05-01),
        both items were valid -- the conflict had been detected but not
        reconciled yet. as_of=2026-04-15 should return both."""
        old = _seed_item(
            storage, embedder, "X is true", created_at=_utc(2026, 1, 1)
        )
        new = _seed_item(
            storage, embedder, "X is false", created_at=_utc(2026, 4, 1)
        )
        _record_and_resolve(
            storage, new_item=new, invalidates=old, now=_utc(2026, 5, 1)
        )
        results = memory.retrieve(
            "X",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            as_of=_utc(2026, 4, 15),
        )
        ids = {r.item_id for r in results}
        assert old.id in ids  # still valid at 2026-04-15
        assert new.id in ids  # already created

    def test_as_of_after_resolution_excludes_loser(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        old = _seed_item(
            storage, embedder, "X is true", created_at=_utc(2026, 1, 1)
        )
        new = _seed_item(
            storage, embedder, "X is false", created_at=_utc(2026, 4, 1)
        )
        _record_and_resolve(
            storage, new_item=new, invalidates=old, now=_utc(2026, 5, 1)
        )
        results = memory.retrieve(
            "X",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            as_of=_utc(2026, 6, 1),
        )
        ids = {r.item_id for r in results}
        assert old.id not in ids
        assert new.id in ids


# ---------------------------------------------------------------------------
# Multi-version chain: X -> X' -> X''
# ---------------------------------------------------------------------------


class TestMultiVersionChain:
    def test_three_snapshots_return_right_item(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        """Three versions: v1 at 2026-01, v2 at 2026-04 (invalidates v1 at
        2026-05), v3 at 2026-07 (invalidates v2 at 2026-08). All three
        share the same embedding so they compete for top-k.

        Snapshots:
          * as_of = 2026-02-15  -> only v1 (v2/v3 not yet)
          * as_of = 2026-06-01  -> only v2 (v1 invalidated at 05-01, v3 not yet)
          * as_of = 2026-09-01  -> only v3 (v2 invalidated at 08-01, v1 long gone)
          * default (no as_of)  -> only v3
        """
        v1 = _seed_item(storage, embedder, "v1", created_at=_utc(2026, 1, 1))
        v2 = _seed_item(storage, embedder, "v2", created_at=_utc(2026, 4, 1))
        _record_and_resolve(
            storage, new_item=v2, invalidates=v1, now=_utc(2026, 5, 1)
        )
        v3 = _seed_item(storage, embedder, "v3", created_at=_utc(2026, 7, 1))
        _record_and_resolve(
            storage, new_item=v3, invalidates=v2, now=_utc(2026, 8, 1)
        )

        def _ids_at(as_of: datetime | None) -> set[UUID]:
            results = memory.retrieve(
                "X",
                k=10,
                prefer="general",
                confidence_threshold=0.0,
                reinforce=False,
                as_of=as_of,
            )
            return {r.item_id for r in results}

        # 2026-02-15: only v1 exists / is valid.
        early = _ids_at(_utc(2026, 2, 15))
        assert v1.id in early
        assert v2.id not in early
        assert v3.id not in early

        # 2026-06-01: v1 invalidated, v2 alive, v3 not yet.
        mid = _ids_at(_utc(2026, 6, 1))
        assert v1.id not in mid
        assert v2.id in mid
        assert v3.id not in mid

        # 2026-09-01: v2 invalidated, v3 alive.
        late = _ids_at(_utc(2026, 9, 1))
        assert v1.id not in late
        assert v2.id not in late
        assert v3.id in late

        # Default (current state): only v3.
        current = _ids_at(None)
        assert v1.id not in current
        assert v2.id not in current
        assert v3.id in current


# ---------------------------------------------------------------------------
# Keep-both: both items stay surfaced after reconcile
# ---------------------------------------------------------------------------


class TestKeepBothNoInvalidation:
    def test_both_visible_after_keep_both(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        a = _seed_item(storage, embedder, "a", created_at=_utc(2026, 1, 1))
        b = _seed_item(storage, embedder, "b", created_at=_utc(2026, 4, 1))
        c = Conflict(source_item_id=b.id, target_item_id=a.id, similarity=0.95)
        storage.record_conflict(c)
        memory.reconcile(
            c.id, resolution=Resolution.KEEP_BOTH, now=_utc(2026, 5, 1)
        )
        results = memory.retrieve(
            "x",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
        )
        ids = {r.item_id for r in results}
        assert a.id in ids
        assert b.id in ids


# ---------------------------------------------------------------------------
# Validity window TTL (no invalidation, just expires)
# ---------------------------------------------------------------------------


class TestValidityWindow:
    def test_expired_excluded_from_default(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        a = _seed_item(storage, embedder, "a", created_at=_utc(2026, 1, 1))
        storage.set_validity_window(
            a.id, valid_until=_utc(2026, 3, 1)
        )
        results = memory.retrieve(
            "x",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            as_of=_utc(2026, 5, 1),
        )
        ids = {r.item_id for r in results}
        assert a.id not in ids

    def test_in_window_visible(
        self, memory: Memory, storage: SqliteStorage, embedder: FakeEmbedder
    ) -> None:
        a = _seed_item(storage, embedder, "a", created_at=_utc(2026, 1, 1))
        storage.set_validity_window(a.id, valid_until=_utc(2026, 6, 1))
        results = memory.retrieve(
            "x",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            as_of=_utc(2026, 4, 15),
        )
        ids = {r.item_id for r in results}
        assert a.id in ids
