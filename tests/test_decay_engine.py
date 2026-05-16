"""Tests for `engram.decay.DecayEngine`.

These cover the read-modify-write loops (record + tick), pruning policy,
and the async surface. The pure formula is exercised in
`tests/test_decay_math.py` already.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from engram.decay import DecayEngine, DecayParams, TickResult
from engram.schemas import Event, ItemKind, Level, MemoryItem
from engram.storage import SqliteStorage


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _seed_event(storage: SqliteStorage, content: str = "x") -> Event:
    event = Event(content=content)
    storage.insert_event(event)
    return event


def _seed_memory_item(
    storage: SqliteStorage,
    *,
    weight: float = 1.0,
    level: Level = Level.SUMMARY,
) -> MemoryItem:
    item = MemoryItem(level=level, content="x", weight=weight)
    storage.insert_memory_item(item)
    return item


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            engine = DecayEngine(storage)
            assert engine.params == DecayParams()
            assert engine.prune_policy == "cold"

    def test_custom_params(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            params = DecayParams(half_life_seconds=10.0, threshold=0.2)
            engine = DecayEngine(storage, params=params, prune_policy="delete")
            assert engine.params is params
            assert engine.prune_policy == "delete"

    def test_invalid_prune_policy(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(ValueError, match="prune_policy"):
                DecayEngine(storage, prune_policy="bogus")  # type: ignore[arg-type]

    def test_invalid_batch_size(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(ValueError, match="batch_size"):
                DecayEngine(storage, batch_size=0)

    def test_empty_kinds_rejected(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(ValueError, match="kinds"):
                DecayEngine(storage, kinds=())

    def test_uncallable_clock_falls_back_to_utcnow(self, tmp_path: Path) -> None:
        # Defensive: passing something non-callable for clock should not
        # break the engine, it should just use the default.
        with SqliteStorage(tmp_path / "x.db") as storage:
            engine = DecayEngine(storage, clock=42)  # type: ignore[arg-type]
            event = _seed_event(storage)
            engine.reinforce(event.id, ItemKind.EVENT, count=1)
            state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state is not None


# ---------------------------------------------------------------------------
# record / reinforce / corroborate / contradict
# ---------------------------------------------------------------------------


class TestRecord:
    def test_reinforce_raises_weight_when_room(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            params = DecayParams(half_life_seconds=1e9, beta=0.10, threshold=0.0)
            engine = DecayEngine(storage, params=params)

            # Weight starts at 1.0 (clamped). Drop it manually so we have
            # headroom.
            initial = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert initial is not None
            half = initial.model_copy(update={"weight": 0.5, "last_decayed_at": _now()})
            storage.update_decay_state(half)

            now = _now() + timedelta(seconds=1)
            new_state = engine.reinforce(event.id, ItemKind.EVENT, count=1, now=now)
            # 0.5 + beta = 0.6 (modulo trivial decay over 1s with huge half-life).
            assert new_state.weight > 0.55
            assert new_state.reinforcement_count == 1

    def test_contradiction_lowers_weight(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            params = DecayParams(half_life_seconds=1e9, delta=0.30, threshold=0.0)
            engine = DecayEngine(storage, params=params)

            new_state = engine.contradict(event.id, ItemKind.EVENT, count=1, now=_now())
            # 1.0 - 0.30 = 0.70 (modulo trivial decay).
            assert new_state.weight < 0.75
            assert new_state.contradiction_count == 1

    def test_corroboration_increments_counter_and_weight(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = _seed_memory_item(storage, weight=0.5)
            params = DecayParams(half_life_seconds=1e9, gamma=0.10, threshold=0.0)
            engine = DecayEngine(storage, params=params)
            new_state = engine.corroborate(item.id, ItemKind.MEMORY_ITEM, count=2)
            # 0.5 + 2 * 0.10 = 0.70.
            assert new_state.weight > 0.65
            assert new_state.corroboration_count == 2

    def test_record_requires_a_signal(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            engine = DecayEngine(storage)
            with pytest.raises(ValueError, match="non-zero signal"):
                engine.record(event.id, ItemKind.EVENT)

    def test_record_rejects_negative_counts(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            engine = DecayEngine(storage)
            with pytest.raises(ValueError, match="non-negative"):
                engine.record(event.id, ItemKind.EVENT, reinforcement=-1)

    def test_record_unknown_item(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            engine = DecayEngine(storage)
            with pytest.raises(KeyError):
                engine.reinforce(uuid4(), ItemKind.EVENT)

    def test_cold_item_rejects_signals(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            storage.mark_cold(event.id, ItemKind.EVENT, at=_now())
            engine = DecayEngine(storage)
            with pytest.raises(RuntimeError, match="cold"):
                engine.reinforce(event.id, ItemKind.EVENT)

    def test_record_marks_cold_when_weight_drops_below_threshold(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = _seed_memory_item(storage, weight=0.30)
            params = DecayParams(half_life_seconds=1e9, delta=0.30, threshold=0.10)
            engine = DecayEngine(storage, params=params)
            new_state = engine.contradict(item.id, ItemKind.MEMORY_ITEM, count=1)
            # 0.30 - 0.30 = 0.00 -> cold (threshold 0.10).
            assert new_state.weight == 0.0
            assert new_state.cold_at is not None

    def test_record_atomic_under_clock_skew(self, tmp_path: Path) -> None:
        # If now < last_decayed_at, dt would be negative; engine clamps to
        # zero to keep the formula well-defined.
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            engine = DecayEngine(storage)
            # Pick a moment well in the past.
            past = _now() - timedelta(days=365)
            new_state = engine.reinforce(event.id, ItemKind.EVENT, count=1, now=past)
            # No NaN, no out-of-range weight.
            assert 0.0 <= new_state.weight <= 1.0


# ---------------------------------------------------------------------------
# tick
# ---------------------------------------------------------------------------


class TestTick:
    def test_empty_store(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            engine = DecayEngine(storage)
            result = engine.tick()
            assert result.items_processed == 0
            assert result.items_pruned == 0
            assert result.items_deleted == 0
            assert result.duration_ms >= 0.0

    def test_tick_decays_old_event(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            params = DecayParams(half_life_seconds=10.0, threshold=0.0)
            engine = DecayEngine(storage, params=params)

            # Tick at time = created_at + 20 seconds (two half-lives).
            future = event.created_at + timedelta(seconds=20)
            result = engine.tick(now=future)
            assert result.items_processed >= 1
            state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state is not None
            assert state.weight == pytest.approx(0.25, rel=1e-3)
            assert state.cold_at is None

    def test_tick_marks_cold(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            # Half-life small + tick after many half-lives -> well below threshold.
            params = DecayParams(half_life_seconds=1.0, threshold=0.05)
            engine = DecayEngine(storage, params=params)
            future = event.created_at + timedelta(seconds=20)
            result = engine.tick(now=future)
            assert result.items_pruned >= 1
            state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state is not None
            assert state.cold_at is not None
            assert state.weight < params.threshold

    def test_tick_skips_cold_items(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            storage.mark_cold(event.id, ItemKind.EVENT, at=_now())
            engine = DecayEngine(storage)
            result = engine.tick()
            assert result.items_processed == 0

    def test_delete_policy_purges_memory_items(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = _seed_memory_item(storage, weight=0.30)
            params = DecayParams(half_life_seconds=1.0, threshold=0.05)
            engine = DecayEngine(storage, params=params, prune_policy="delete")
            future = item.updated_at + timedelta(seconds=20)
            result = engine.tick(now=future)
            assert result.items_deleted >= 1
            assert storage.get_memory_item(item.id) is None

    def test_delete_policy_skips_event_with_provenance(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            mi = _seed_memory_item(storage)
            storage.link_provenance(mi.id, event.id)
            params = DecayParams(half_life_seconds=1.0, threshold=0.05)
            engine = DecayEngine(storage, params=params, prune_policy="delete")
            future = event.created_at + timedelta(seconds=30)
            result = engine.tick(now=future)
            # Event marked cold but not deleted (provenance guard).
            assert storage.get_event(event.id) is not None
            cold_event_state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert cold_event_state is not None
            assert cold_event_state.cold_at is not None
            # Either 0 or 1 deleted depending on whether MI also pruned.
            assert result.items_deleted >= 0

    def test_per_kind_summary_present(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            _seed_event(storage)
            _seed_memory_item(storage)
            engine = DecayEngine(storage)
            result = engine.tick(now=_now())
            # Every kind the engine sweeps appears in the per-kind dict.
            assert ItemKind.EVENT in result.per_kind
            assert ItemKind.MEMORY_ITEM in result.per_kind
            assert ItemKind.PROCEDURE in result.per_kind
            # Seeded kinds have at least one processed row; empty kinds
            # (procedures here -- we didn't seed any) report processed=0
            # cleanly.
            assert result.per_kind[ItemKind.EVENT]["processed"] >= 1
            assert result.per_kind[ItemKind.MEMORY_ITEM]["processed"] >= 1
            assert result.per_kind[ItemKind.PROCEDURE]["processed"] == 0

    def test_tick_idempotent_at_same_moment(self, tmp_path: Path) -> None:
        # Calling tick twice with the same `now` should leave the table
        # unchanged on the second call.
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            params = DecayParams(half_life_seconds=10.0, threshold=0.0)
            engine = DecayEngine(storage, params=params)
            future = event.created_at + timedelta(seconds=5)
            engine.tick(now=future)
            state_after_first = storage.get_decay_state(event.id, ItemKind.EVENT)
            engine.tick(now=future)
            state_after_second = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state_after_first == state_after_second


# ---------------------------------------------------------------------------
# tick_async
# ---------------------------------------------------------------------------


class TestTickAsync:
    def test_returns_tickresult(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            _seed_event(storage)
            engine = DecayEngine(storage)
            result = asyncio.run(engine.tick_async(now=_now()))
            assert isinstance(result, TickResult)
            assert result.items_processed >= 1


# ---------------------------------------------------------------------------
# Cold-transition routing (H-69): newly-cold rows go through mark_cold so
# the vector + BM25 indexes are invalidated, not via raw update_decay_state.
# ---------------------------------------------------------------------------


class TestColdTransitionInvalidatesIndexes:
    """Newly-cold rows must invalidate the vector and BM25 indexes.

    `mark_cold` flips the dirty flags; `update_decay_state` does not. A
    previous implementation set `cold_at` via the plain update path, so
    the cold row stayed visible in retrieve until the next index rebuild.
    Route the cold transition through `mark_cold` to fix that.
    """

    def test_record_cold_calls_mark_cold(self, tmp_path: Path) -> None:
        # Spy on storage.mark_cold. The engine must call it when a record
        # transitions the row into the cold pool.
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = _seed_memory_item(storage, weight=0.30)
            params = DecayParams(half_life_seconds=1e9, delta=0.30, threshold=0.10)
            engine = DecayEngine(storage, params=params)
            calls: list[tuple] = []
            real_mark_cold = storage.mark_cold

            def spy_mark_cold(item_id, kind, *, at):  # type: ignore[no-untyped-def]
                calls.append((item_id, kind, at))
                real_mark_cold(item_id, kind, at=at)

            storage.mark_cold = spy_mark_cold  # type: ignore[method-assign]
            engine.contradict(item.id, ItemKind.MEMORY_ITEM, count=1)
            assert len(calls) == 1
            assert calls[0][0] == item.id
            assert calls[0][1] is ItemKind.MEMORY_ITEM

    def test_tick_cold_calls_mark_cold(self, tmp_path: Path) -> None:
        # Spy on storage.mark_cold during tick: rows that cross under
        # threshold must be routed through mark_cold so vector+BM25
        # invalidation kicks in.
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = _seed_event(storage)
            params = DecayParams(half_life_seconds=1.0, threshold=0.05)
            engine = DecayEngine(storage, params=params)
            calls: list[tuple] = []
            real_mark_cold = storage.mark_cold

            def spy_mark_cold(item_id, kind, *, at):  # type: ignore[no-untyped-def]
                calls.append((item_id, kind, at))
                real_mark_cold(item_id, kind, at=at)

            storage.mark_cold = spy_mark_cold  # type: ignore[method-assign]
            future = event.created_at + timedelta(seconds=20)
            result = engine.tick(now=future)
            assert result.items_pruned >= 1
            assert any(c[0] == event.id and c[1] is ItemKind.EVENT for c in calls)


# ---------------------------------------------------------------------------
# Concurrency guard on tick (H-68)
# ---------------------------------------------------------------------------


class TestTickConcurrencyGuard:
    """A second `tick` call while one is in flight raises rather than
    silently corrupting the iterator-then-update sequence.
    """

    def test_concurrent_tick_raises_runtime_error(self, tmp_path: Path) -> None:
        import threading

        with SqliteStorage(tmp_path / "x.db") as storage:
            _seed_event(storage)
            engine = DecayEngine(storage)
            held = threading.Event()
            release = threading.Event()

            # Reach into the engine's threading lock so the helper thread
            # blocks at the top of tick. We acquire it from the test thread,
            # then a worker thread tries tick and observes the contention.
            assert engine._tick_lock.acquire()  # type: ignore[attr-defined]
            held.set()

            failure: dict[str, BaseException] = {}

            def worker() -> None:
                try:
                    engine.tick()
                except BaseException as exc:
                    failure["exc"] = exc

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=2.0)
            engine._tick_lock.release()  # type: ignore[attr-defined]
            release.set()
            t.join()
            assert "exc" in failure
            assert isinstance(failure["exc"], RuntimeError)
            assert "already running" in str(failure["exc"])


# ---------------------------------------------------------------------------
# Streaming iteration (H-66) -- batch_size should bound peak memory.
# ---------------------------------------------------------------------------


class TestTickStreaming:
    def test_tick_handles_large_store_in_chunks(self, tmp_path: Path) -> None:
        # Seed many items; tick with a small batch_size to exercise the
        # chunked drain path. The engine should still process all of them.
        with SqliteStorage(tmp_path / "x.db") as storage:
            n = 25  # small but > 2 * batch_size below
            for i in range(n):
                _seed_event(storage, content=f"event-{i}")
            engine = DecayEngine(storage, batch_size=4)
            result = engine.tick(now=_now())
            assert result.items_processed == n
