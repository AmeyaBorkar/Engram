"""Storage-level tests for the consolidation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.providers._fake import FakeEmbedder
from engram.schemas import (
    Cluster,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
)
from engram.storage import SqliteStorage


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _seed_event(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
    *,
    content: str,
    created_at: datetime | None = None,
) -> Event:
    event = Event(content=content, created_at=created_at or _now())
    storage.insert_event(event)
    vec = tuple(embedder.embed([content])[0])
    storage.insert_embedding(
        Embedding(
            item_id=event.id,
            item_kind=ItemKind.EVENT,
            model=embedder.model,
            dim=embedder.dim,
            vector=vec,
        )
    )
    return event


# ---------------------------------------------------------------------------
# iter_unconsolidated_events_with_embeddings
# ---------------------------------------------------------------------------


class TestIterUnconsolidated:
    def test_returns_events_without_provenance(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            a = _seed_event(storage, embedder, content="alpha")
            b = _seed_event(storage, embedder, content="beta")

            pairs = list(storage.iter_unconsolidated_events_with_embeddings(model=embedder.model))
            assert {p[0].id for p in pairs} == {a.id, b.id}
            for ev, vec in pairs:
                assert len(vec) == embedder.dim

    def test_skips_events_with_provenance(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            consolidated = _seed_event(storage, embedder, content="x")
            unconsolidated = _seed_event(storage, embedder, content="y")
            mi = MemoryItem(level=Level.SUMMARY, content="summary x")
            storage.insert_memory_item(mi)
            storage.link_provenance(mi.id, consolidated.id)

            pairs = list(storage.iter_unconsolidated_events_with_embeddings(model=embedder.model))
            assert {p[0].id for p in pairs} == {unconsolidated.id}

    def test_skips_cold_events(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            hot = _seed_event(storage, embedder, content="hot")
            cold = _seed_event(storage, embedder, content="cold")
            storage.mark_cold(cold.id, ItemKind.EVENT, at=_now())

            pairs = list(storage.iter_unconsolidated_events_with_embeddings(model=embedder.model))
            assert {p[0].id for p in pairs} == {hot.id}

    def test_orders_by_created_at(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            base = _now()
            second = _seed_event(
                storage, embedder, content="b", created_at=base + timedelta(seconds=2)
            )
            first = _seed_event(
                storage, embedder, content="a", created_at=base + timedelta(seconds=1)
            )
            third = _seed_event(
                storage, embedder, content="c", created_at=base + timedelta(seconds=3)
            )

            pairs = list(storage.iter_unconsolidated_events_with_embeddings(model=embedder.model))
            ordered_ids = [p[0].id for p in pairs]
            assert ordered_ids == [first.id, second.id, third.id]

    def test_respects_limit(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            for i in range(10):
                _seed_event(storage, embedder, content=f"e{i}")

            pairs = list(
                storage.iter_unconsolidated_events_with_embeddings(model=embedder.model, limit=3)
            )
            assert len(pairs) == 3

    def test_skips_events_without_embedding_for_model(self, tmp_path: Path) -> None:
        # An event with no embedding for our model should not appear.
        embedder_a = FakeEmbedder(dim=8, model="model-a")
        embedder_b = FakeEmbedder(dim=8, model="model-b")
        with SqliteStorage(tmp_path / "x.db") as storage:
            _seed_event(storage, embedder_a, content="only-in-a")
            in_b = _seed_event(storage, embedder_b, content="only-in-b")

            pairs = list(storage.iter_unconsolidated_events_with_embeddings(model=embedder_b.model))
            assert {p[0].id for p in pairs} == {in_b.id}

    def test_validates_batch_size(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(ValueError, match="batch_size"):
                list(storage.iter_unconsolidated_events_with_embeddings(model="any", batch_size=0))

    def test_validates_limit(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(ValueError, match="limit"):
                list(storage.iter_unconsolidated_events_with_embeddings(model="any", limit=-1))

    def test_empty_store_yields_nothing(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            pairs = list(storage.iter_unconsolidated_events_with_embeddings(model="any"))
            assert pairs == []


# ---------------------------------------------------------------------------
# insert_memory_item_with_provenance
# ---------------------------------------------------------------------------


class TestInsertMemoryItemWithProvenance:
    def test_atomic_insert_with_links(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            e1 = _seed_event(storage, embedder, content="x")
            e2 = _seed_event(storage, embedder, content="y")

            cluster = Cluster(cohesion=0.8)
            item = MemoryItem(
                level=Level.SUMMARY,
                content="general pattern",
                cluster_id=cluster.id,
            )
            links = storage.insert_memory_item_with_provenance(
                item, [e1.id, e2.id], cluster=cluster
            )
            assert {ln.event_id for ln in links} == {e1.id, e2.id}
            assert storage.get_memory_item(item.id) is not None
            assert storage.get_cluster(cluster.id) is not None
            supports = storage.get_supporting_events(item.id)
            assert {s.id for s in supports} == {e1.id, e2.id}

    def test_atomic_with_embedding(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            e1 = _seed_event(storage, embedder, content="x")
            item = MemoryItem(level=Level.SUMMARY, content="pattern")
            vec = tuple(embedder.embed([item.content])[0])
            embedding = Embedding(
                item_id=item.id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=embedder.model,
                dim=embedder.dim,
                vector=vec,
            )
            storage.insert_memory_item_with_provenance(item, [e1.id], embedding=embedding)
            stored = storage.get_embedding(item.id, ItemKind.MEMORY_ITEM, embedder.model)
            assert stored is not None
            assert stored.dim == embedder.dim

    def test_event_level_allows_zero_supports(self, tmp_path: Path) -> None:
        # A `level=event` memory item is itself the source; no supports needed.
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = MemoryItem(level=Level.EVENT, content="raw")
            storage.insert_memory_item_with_provenance(item, [])
            assert storage.get_memory_item(item.id) is not None
            assert storage.get_supporting_events(item.id) == []

    def test_non_event_without_supports_rejected(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = MemoryItem(level=Level.SUMMARY, content="dangling")
            with pytest.raises(ValueError, match="supporting event"):
                storage.insert_memory_item_with_provenance(item, [])
            # Nothing landed.
            assert storage.get_memory_item(item.id) is None

    def test_provenance_weights_applied(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            e1 = _seed_event(storage, embedder, content="strong")
            e2 = _seed_event(storage, embedder, content="weak")
            item = MemoryItem(level=Level.SUMMARY, content="pattern")
            links = storage.insert_memory_item_with_provenance(
                item,
                [e1.id, e2.id],
                provenance_weights={e1.id: 1.0, e2.id: 0.3},
            )
            by_event = {ln.event_id: ln.weight for ln in links}
            assert by_event[e1.id] == 1.0
            assert by_event[e2.id] == 0.3

    def test_rolls_back_on_failure(self, tmp_path: Path) -> None:
        # Pass a bogus event id that doesn't exist; the FK violation must
        # roll back the memory_item insert too.
        import sqlite3
        from uuid import uuid4

        with SqliteStorage(tmp_path / "x.db") as storage:
            item = MemoryItem(level=Level.SUMMARY, content="X")
            with pytest.raises(sqlite3.IntegrityError):
                storage.insert_memory_item_with_provenance(item, [uuid4()])
            assert storage.get_memory_item(item.id) is None

    def test_embedding_id_must_match_item(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            e1 = _seed_event(storage, embedder, content="x")
            item = MemoryItem(level=Level.SUMMARY, content="P")
            wrong = Embedding(
                item_id=e1.id,  # mismatched
                item_kind=ItemKind.MEMORY_ITEM,
                model=embedder.model,
                dim=embedder.dim,
                vector=tuple([0.0] * embedder.dim),
            )
            with pytest.raises(ValueError, match="item_id"):
                storage.insert_memory_item_with_provenance(item, [e1.id], embedding=wrong)


# ---------------------------------------------------------------------------
# Protocol satisfied
# ---------------------------------------------------------------------------


def test_sqlite_storage_implements_consolidation_protocol(tmp_path: Path) -> None:
    from engram.storage._protocol import Storage

    with SqliteStorage(tmp_path / "x.db") as storage:
        assert isinstance(storage, Storage)
