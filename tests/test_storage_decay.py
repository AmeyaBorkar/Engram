"""Storage-level tests for decay state CRUD.

Stage 4 commit 4: storage exposes `get_decay_state`, `iter_decay_states`,
`update_decay_state`, `mark_cold` / `unmark_cold` / `count_cold`, and
`delete_cold_items`. Higher-level engine tests (read-modify-write loops,
tick scheduling) live in commit 5.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.schemas import DecayState, Event, ItemKind, Level, MemoryItem
from engram.storage import SqliteStorage


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# --- get_decay_state --------------------------------------------------------


class TestGetDecayState:
    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = Event(content="x")
            assert storage.get_decay_state(event.id, ItemKind.EVENT) is None

    def test_returns_state_with_defaults_for_fresh_event(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = Event(content="x")
            storage.insert_event(event)
            state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state is not None
            assert state.item_id == event.id
            assert state.item_kind is ItemKind.EVENT
            # `pytest.approx` for float comparisons: storage's float
            # round-trip is bit-exact today (REAL == IEEE 754 double),
            # but the test contract shouldn't depend on that encoding.
            assert state.weight == pytest.approx(1.0)
            assert state.reinforcement_count == 0
            assert state.corroboration_count == 0
            assert state.contradiction_count == 0
            assert state.last_decayed_at == event.created_at
            assert state.cold_at is None

    def test_returns_state_for_memory_item(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = MemoryItem(level=Level.SUMMARY, content="x", weight=0.4)
            storage.insert_memory_item(item)
            state = storage.get_decay_state(item.id, ItemKind.MEMORY_ITEM)
            assert state is not None
            assert state.item_kind is ItemKind.MEMORY_ITEM
            assert state.weight == pytest.approx(0.4)
            assert state.last_decayed_at == item.updated_at


# --- update_decay_state -----------------------------------------------------


class TestUpdateDecayState:
    def test_round_trip(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = Event(content="x")
            storage.insert_event(event)
            now = _now()
            new_state = DecayState(
                item_id=event.id,
                item_kind=ItemKind.EVENT,
                weight=0.42,
                reinforcement_count=3,
                corroboration_count=1,
                contradiction_count=2,
                last_decayed_at=now,
                cold_at=None,
            )
            storage.update_decay_state(new_state)
            roundtrip = storage.get_decay_state(event.id, ItemKind.EVENT)
            # Compare field-by-field with `approx` for the float so the
            # test doesn't blow up if storage ever changes its float
            # encoding (e.g. fixed-point milliweight).  SQLite's REAL
            # column round-trips IEEE 754 doubles bit-exact today, but
            # the assertion contract should not depend on that.
            assert roundtrip is not None
            assert roundtrip.item_id == new_state.item_id
            assert roundtrip.item_kind is new_state.item_kind
            assert roundtrip.weight == pytest.approx(new_state.weight)
            assert roundtrip.reinforcement_count == new_state.reinforcement_count
            assert roundtrip.corroboration_count == new_state.corroboration_count
            assert roundtrip.contradiction_count == new_state.contradiction_count
            assert roundtrip.last_decayed_at == new_state.last_decayed_at
            assert roundtrip.cold_at == new_state.cold_at

    def test_update_with_cold_at(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = Event(content="x")
            storage.insert_event(event)
            now = _now()
            cold_state = DecayState(
                item_id=event.id,
                item_kind=ItemKind.EVENT,
                weight=0.01,
                last_decayed_at=now,
                cold_at=now,
            )
            storage.update_decay_state(cold_state)
            roundtrip = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert roundtrip is not None
            assert roundtrip.cold_at == now

    def test_unknown_id_raises(self, tmp_path: Path) -> None:
        from uuid import uuid4

        with SqliteStorage(tmp_path / "x.db") as storage:
            ghost = DecayState(
                item_id=uuid4(),
                item_kind=ItemKind.EVENT,
                weight=0.5,
                last_decayed_at=_now(),
            )
            with pytest.raises(KeyError):
                storage.update_decay_state(ghost)


# --- iter_decay_states ------------------------------------------------------


class TestIterDecayStates:
    def test_yields_every_hot_event(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            ids = []
            for i in range(5):
                ev = Event(content=f"event {i}")
                storage.insert_event(ev)
                ids.append(ev.id)
            seen = {s.item_id for s in storage.iter_decay_states(ItemKind.EVENT)}
            assert seen == set(ids)

    def test_skips_cold_unless_include_cold(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            hot = Event(content="hot")
            cold = Event(content="cold")
            storage.insert_event(hot)
            storage.insert_event(cold)
            storage.mark_cold(cold.id, ItemKind.EVENT, at=_now())

            hot_only = {s.item_id for s in storage.iter_decay_states(ItemKind.EVENT)}
            assert hot_only == {hot.id}

            both = {s.item_id for s in storage.iter_decay_states(ItemKind.EVENT, include_cold=True)}
            assert both == {hot.id, cold.id}

    def test_batch_size_validation(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(ValueError, match="batch_size"):
                list(storage.iter_decay_states(ItemKind.EVENT, batch_size=0))

    def test_batched_streaming_against_large_table(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with storage.transaction():
                storage.insert_events(Event(content=f"e{i}") for i in range(2500))
            count = sum(1 for _ in storage.iter_decay_states(ItemKind.EVENT, batch_size=128))
            assert count == 2500


# --- mark_cold / unmark_cold / count_cold -----------------------------------


class TestColdMarkers:
    def test_mark_then_count(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            for i in range(3):
                storage.insert_event(Event(content=f"e{i}"))
            assert storage.count_cold(ItemKind.EVENT) == 0

            chosen = next(iter(storage.iter_decay_states(ItemKind.EVENT)))
            storage.mark_cold(chosen.item_id, ItemKind.EVENT, at=_now())
            assert storage.count_cold(ItemKind.EVENT) == 1

    def test_mark_cold_unknown_raises(self, tmp_path: Path) -> None:
        from uuid import uuid4

        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(KeyError):
                storage.mark_cold(uuid4(), ItemKind.EVENT, at=_now())

    def test_unmark_restores(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            ev = Event(content="x")
            storage.insert_event(ev)
            storage.mark_cold(ev.id, ItemKind.EVENT, at=_now())
            assert storage.count_cold(ItemKind.EVENT) == 1
            storage.unmark_cold(ev.id, ItemKind.EVENT)
            assert storage.count_cold(ItemKind.EVENT) == 0

    def test_unmark_unknown_raises(self, tmp_path: Path) -> None:
        from uuid import uuid4

        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(KeyError):
                storage.unmark_cold(uuid4(), ItemKind.EVENT)

    def test_count_cold_per_kind(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            ev = Event(content="x")
            mi = MemoryItem(level=Level.SUMMARY, content="y")
            storage.insert_event(ev)
            storage.insert_memory_item(mi)
            now = _now()
            storage.mark_cold(ev.id, ItemKind.EVENT, at=now)
            assert storage.count_cold(ItemKind.EVENT) == 1
            assert storage.count_cold(ItemKind.MEMORY_ITEM) == 0
            storage.mark_cold(mi.id, ItemKind.MEMORY_ITEM, at=now)
            assert storage.count_cold(ItemKind.MEMORY_ITEM) == 1


# --- delete_cold_items ------------------------------------------------------


class TestDeleteColdItems:
    def test_deletes_cold_memory_items(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            keep = MemoryItem(level=Level.SUMMARY, content="keep")
            drop = MemoryItem(level=Level.SUMMARY, content="drop")
            storage.insert_memory_item(keep)
            storage.insert_memory_item(drop)
            storage.mark_cold(drop.id, ItemKind.MEMORY_ITEM, at=_now())
            n = storage.delete_cold_items(ItemKind.MEMORY_ITEM)
            assert n == 1
            assert storage.get_memory_item(drop.id) is None
            assert storage.get_memory_item(keep.id) is not None

    def test_deletes_cold_events_without_provenance(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            ev = Event(content="x")
            storage.insert_event(ev)
            storage.mark_cold(ev.id, ItemKind.EVENT, at=_now())
            n = storage.delete_cold_items(ItemKind.EVENT)
            assert n == 1
            assert storage.get_event(ev.id) is None

    def test_refuses_to_delete_cold_event_with_provenance(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            ev = Event(content="x")
            mi = MemoryItem(level=Level.SUMMARY, content="y")
            storage.insert_event(ev)
            storage.insert_memory_item(mi)
            storage.link_provenance(mi.id, ev.id)
            storage.mark_cold(ev.id, ItemKind.EVENT, at=_now())

            with pytest.raises(RuntimeError, match="provenance"):
                storage.delete_cold_items(ItemKind.EVENT)
            # The event survives.
            assert storage.get_event(ev.id) is not None

    def test_delete_returns_zero_when_nothing_cold(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            storage.insert_event(Event(content="x"))
            assert storage.delete_cold_items(ItemKind.EVENT) == 0


# --- protocol satisfaction --------------------------------------------------


class TestProtocolSatisfied:
    def test_sqlite_storage_implements_decay_methods(self, tmp_path: Path) -> None:
        from engram.storage._protocol import Storage

        with SqliteStorage(tmp_path / "x.db") as storage:
            # runtime_checkable protocol checks method names.
            assert isinstance(storage, Storage)


# --- last_decayed_at preserves precision ------------------------------------


class TestTimestampPrecision:
    def test_microsecond_precision_preserved(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = Event(content="x")
            storage.insert_event(event)
            precise = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc)
            state = DecayState(
                item_id=event.id,
                item_kind=ItemKind.EVENT,
                weight=0.5,
                last_decayed_at=precise,
                cold_at=precise + timedelta(seconds=1),
            )
            storage.update_decay_state(state)
            roundtrip = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert roundtrip is not None
            assert roundtrip.last_decayed_at == precise
            assert roundtrip.cold_at == precise + timedelta(seconds=1)
