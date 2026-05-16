"""SQLite storage CRUD and integrity tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from engram.ids import new_id
from engram.schemas import (
    Cluster,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
)
from engram.storage import SqliteStorage, stats

# --- events ---------------------------------------------------------------


def test_insert_and_get_event(storage: SqliteStorage) -> None:
    e = Event(content="hello", source="user", metadata={"k": "v"})
    storage.insert_event(e)
    fetched = storage.get_event(e.id)
    assert fetched is not None
    assert fetched.id == e.id
    assert fetched.content == "hello"
    assert fetched.source == "user"
    assert fetched.metadata == {"k": "v"}


def test_get_event_missing_returns_none(storage: SqliteStorage) -> None:
    assert storage.get_event(new_id()) is None


def test_insert_events_bulk(storage: SqliteStorage) -> None:
    events = [Event(content=f"e{i}") for i in range(50)]
    n = storage.insert_events(events)
    assert n == 50
    assert storage.count_events() == 50


def test_list_events_orders_newest_first(storage: SqliteStorage) -> None:
    a = Event(content="first")
    b = Event(content="second")
    storage.insert_event(a)
    storage.insert_event(b)
    listed = storage.list_events(limit=10)
    assert listed[0].content == "second"
    assert listed[1].content == "first"


def test_list_events_filters_by_source(storage: SqliteStorage) -> None:
    storage.insert_event(Event(content="a", source="user"))
    storage.insert_event(Event(content="b", source="agent"))
    storage.insert_event(Event(content="c", source="user"))
    user_events = storage.list_events(source="user")
    assert len(user_events) == 2
    assert all(e.source == "user" for e in user_events)


def test_list_events_filters_by_before(storage: SqliteStorage) -> None:
    earlier = Event(
        content="earlier",
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    later = Event(
        content="later",
        created_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    storage.insert_event(earlier)
    storage.insert_event(later)
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    filtered = storage.list_events(before=cutoff)
    assert len(filtered) == 1
    assert filtered[0].content == "earlier"


def test_duplicate_event_id_rejected(storage: SqliteStorage) -> None:
    e = Event(content="x")
    storage.insert_event(e)
    with pytest.raises(sqlite3.IntegrityError):
        storage.insert_event(e)


# --- memory items ---------------------------------------------------------


def test_insert_and_get_memory_item(storage: SqliteStorage) -> None:
    item = MemoryItem(level=Level.SUMMARY, content="cluster summary", weight=0.7)
    storage.insert_memory_item(item)
    fetched = storage.get_memory_item(item.id)
    assert fetched is not None
    assert fetched.level == Level.SUMMARY
    assert fetched.weight == pytest.approx(0.7)


def test_list_memory_items_filters_by_level(storage: SqliteStorage) -> None:
    storage.insert_memory_item(MemoryItem(level=Level.EVENT, content="e"))
    storage.insert_memory_item(MemoryItem(level=Level.SUMMARY, content="s"))
    storage.insert_memory_item(MemoryItem(level=Level.ABSTRACTION, content="a"))
    abstractions = storage.list_memory_items(level=Level.ABSTRACTION)
    assert len(abstractions) == 1
    assert abstractions[0].content == "a"


def test_update_memory_item_weight(storage: SqliteStorage) -> None:
    item = MemoryItem(level=Level.EVENT, content="x", weight=1.0)
    storage.insert_memory_item(item)
    storage.update_memory_item_weight(item.id, 0.5)
    fetched = storage.get_memory_item(item.id)
    assert fetched is not None
    assert fetched.weight == pytest.approx(0.5)


def test_update_weight_rejects_out_of_range(storage: SqliteStorage) -> None:
    item = MemoryItem(level=Level.EVENT, content="x")
    storage.insert_memory_item(item)
    with pytest.raises(ValueError, match=r"not in \[0, 1\]"):
        storage.update_memory_item_weight(item.id, 1.5)


def test_update_weight_unknown_id_raises(storage: SqliteStorage) -> None:
    with pytest.raises(KeyError):
        storage.update_memory_item_weight(new_id(), 0.5)


def test_count_memory_items_by_level(storage: SqliteStorage) -> None:
    storage.insert_memory_item(MemoryItem(level=Level.EVENT, content="e1"))
    storage.insert_memory_item(MemoryItem(level=Level.EVENT, content="e2"))
    storage.insert_memory_item(MemoryItem(level=Level.SUMMARY, content="s"))
    counts = storage.count_memory_items_by_level()
    assert counts[Level.EVENT] == 2
    assert counts[Level.SUMMARY] == 1
    assert counts[Level.ABSTRACTION] == 0


# --- embeddings -----------------------------------------------------------


def test_insert_and_get_embedding(storage: SqliteStorage) -> None:
    item_id = new_id()
    e = Embedding(
        item_id=item_id,
        item_kind=ItemKind.EVENT,
        model="m",
        dim=4,
        vector=(1.0, 2.0, 3.0, 4.0),
    )
    storage.insert_embedding(e)
    got = storage.get_embedding(item_id, ItemKind.EVENT, "m")
    assert got is not None
    assert got.vector == pytest.approx((1.0, 2.0, 3.0, 4.0))


def test_embedding_unique_per_item_kind_model(storage: SqliteStorage) -> None:
    item_id = new_id()
    a = Embedding(
        item_id=item_id,
        item_kind=ItemKind.EVENT,
        model="m",
        dim=2,
        vector=(0.1, 0.2),
    )
    storage.insert_embedding(a)
    duplicate = Embedding(
        item_id=item_id,
        item_kind=ItemKind.EVENT,
        model="m",
        dim=2,
        vector=(0.3, 0.4),
    )
    with pytest.raises(sqlite3.IntegrityError):
        storage.insert_embedding(duplicate)


# --- clusters & memory item linkage ---------------------------------------


def test_cluster_link_set_null_on_delete(storage: SqliteStorage) -> None:
    cluster = Cluster(cohesion=0.8)
    storage.insert_cluster(cluster)
    item = MemoryItem(level=Level.SUMMARY, content="s", cluster_id=cluster.id)
    storage.insert_memory_item(item)

    # Use the private `_connect()` here on purpose: the public API does
    # not expose a `delete_cluster` because the production code never
    # deletes clusters mid-lifetime.  The test must exercise the SQL
    # foreign-key SET NULL behavior directly to pin the schema's
    # ON DELETE clause.  noqa: SLF001
    storage._connect().execute(
        "DELETE FROM clusters WHERE id = ?", (cluster.id.bytes,)
    )
    fetched = storage.get_memory_item(item.id)
    assert fetched is not None
    assert fetched.cluster_id is None


# --- provenance -----------------------------------------------------------


def test_link_provenance_and_traverse(storage: SqliteStorage) -> None:
    event = Event(content="seed")
    storage.insert_event(event)
    item = MemoryItem(level=Level.SUMMARY, content="from-seed")
    storage.insert_memory_item(item)
    link = storage.link_provenance(item.id, event.id, weight=0.9)
    assert link.memory_item_id == item.id
    assert link.event_id == event.id

    supporting = storage.get_supporting_events(item.id)
    assert [e.id for e in supporting] == [event.id]
    supported = storage.get_supported_memory_items(event.id)
    assert [m.id for m in supported] == [item.id]


def test_provenance_cannot_dangle_to_missing_event(storage: SqliteStorage) -> None:
    item = MemoryItem(level=Level.SUMMARY, content="x")
    storage.insert_memory_item(item)
    with pytest.raises(sqlite3.IntegrityError):
        storage.link_provenance(item.id, new_id())


def test_provenance_cannot_dangle_to_missing_memory_item(storage: SqliteStorage) -> None:
    event = Event(content="x")
    storage.insert_event(event)
    with pytest.raises(sqlite3.IntegrityError):
        storage.link_provenance(new_id(), event.id)


def test_provenance_blocks_event_deletion(storage: SqliteStorage) -> None:
    event = Event(content="seed")
    storage.insert_event(event)
    item = MemoryItem(level=Level.SUMMARY, content="from-seed")
    storage.insert_memory_item(item)
    storage.link_provenance(item.id, event.id)

    # Raw `_connect().execute` here: the public API intentionally
    # offers no `delete_event` because production code never deletes
    # events; the test must hit the SQL layer directly to verify the
    # foreign-key contract.  noqa: SLF001
    def _delete_event() -> None:
        storage._connect().execute(
            "DELETE FROM events WHERE id = ?", (event.id.bytes,)
        )

    with pytest.raises(sqlite3.IntegrityError):
        _delete_event()


def test_memory_item_deletion_cascades_provenance(storage: SqliteStorage) -> None:
    event = Event(content="seed")
    storage.insert_event(event)
    item = MemoryItem(level=Level.SUMMARY, content="from-seed")
    storage.insert_memory_item(item)
    storage.link_provenance(item.id, event.id)

    # Same rationale as above: production has no `delete_memory_item`
    # public surface (Stage 4's `delete_cold_items` is the only path),
    # so we drop into raw SQL to pin ON DELETE CASCADE.  noqa: SLF001
    storage._connect().execute(
        "DELETE FROM memory_items WHERE id = ?", (item.id.bytes,)
    )
    assert storage.count_provenance_links() == 0
    assert storage.count_events() == 1


def test_provenance_unique_per_pair(storage: SqliteStorage) -> None:
    event = Event(content="x")
    storage.insert_event(event)
    item = MemoryItem(level=Level.SUMMARY, content="y")
    storage.insert_memory_item(item)
    storage.link_provenance(item.id, event.id)
    with pytest.raises(sqlite3.IntegrityError):
        storage.link_provenance(item.id, event.id)


# --- transactions ---------------------------------------------------------


def test_transaction_rolls_back_on_error(storage: SqliteStorage) -> None:
    def _insert_then_raise() -> None:
        with storage.transaction():
            storage.insert_event(Event(content="x"))
            assert storage.count_events() == 1
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _insert_then_raise()
    assert storage.count_events() == 0


def test_transaction_commits_on_success(storage: SqliteStorage) -> None:
    with storage.transaction():
        storage.insert_event(Event(content="a"))
        storage.insert_event(Event(content="b"))
    assert storage.count_events() == 2


def test_transaction_reentrant(storage: SqliteStorage) -> None:
    with storage.transaction():
        storage.insert_event(Event(content="a"))
        with storage.transaction():  # no-op
            storage.insert_event(Event(content="b"))
    assert storage.count_events() == 2


# --- inspector ------------------------------------------------------------


def test_stats_returns_counts(storage: SqliteStorage) -> None:
    storage.insert_event(Event(content="e"))
    storage.insert_memory_item(MemoryItem(level=Level.SUMMARY, content="s"))
    storage.insert_cluster(Cluster(cohesion=0.5))
    snap = stats(storage)
    assert snap["events"] == 1
    assert snap["memory_items"] == 1
    assert snap["clusters"] == 1
    assert snap["embeddings"] == 0
    assert snap["provenance_links"] == 0
    assert snap["by_level"][Level.SUMMARY.value] == 1


# --- lifecycle ------------------------------------------------------------


def test_initialize_is_idempotent(disk_storage: SqliteStorage) -> None:
    disk_storage.initialize()  # second call should be a no-op
    disk_storage.insert_event(Event(content="x"))
    assert disk_storage.count_events() == 1


def test_storage_context_manager(tmp_path: object) -> None:
    from pathlib import Path as _Path

    p = _Path(str(tmp_path)) / "ctx.db"
    with SqliteStorage(p) as backend:
        backend.insert_event(Event(content="hi"))
        assert backend.count_events() == 1


def test_insert_events_empty_iterable_returns_zero(storage: SqliteStorage) -> None:
    assert storage.insert_events([]) == 0
    assert storage.count_events() == 0


def test_insert_memory_items_bulk(storage: SqliteStorage) -> None:
    items = [MemoryItem(level=Level.EVENT, content=f"m{i}") for i in range(20)]
    n = storage.insert_memory_items(items)
    assert n == 20
    assert storage.count_memory_items() == 20


def test_insert_memory_items_empty_iterable_returns_zero(storage: SqliteStorage) -> None:
    assert storage.insert_memory_items([]) == 0


def test_get_cluster_returns_populated_cluster(storage: SqliteStorage) -> None:
    cluster = Cluster(cohesion=0.42)
    storage.insert_cluster(cluster)
    fetched = storage.get_cluster(cluster.id)
    assert fetched is not None
    assert fetched.id == cluster.id
    assert fetched.cohesion == pytest.approx(0.42)


def test_get_cluster_missing_returns_none(storage: SqliteStorage) -> None:
    assert storage.get_cluster(new_id()) is None


def test_get_embedding_missing_returns_none(storage: SqliteStorage) -> None:
    assert storage.get_embedding(new_id(), ItemKind.EVENT, "m") is None


def test_get_memory_item_missing_returns_none(storage: SqliteStorage) -> None:
    assert storage.get_memory_item(new_id()) is None
