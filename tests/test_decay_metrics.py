"""Tests for the decay metrics surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from engram import DecayParams, Memory, SqliteStorage
from engram.decay import DecayEngine, DecayMetrics, KindCounters
from engram.providers._fake import FakeEmbedder
from engram.schemas import ItemKind


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make(tmp_path: Path, **kwargs: object) -> tuple[Memory, SqliteStorage]:
    storage = SqliteStorage(tmp_path / "x.db")
    storage.initialize()
    memory = Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=4),
        decay_params=kwargs.get("decay_params"),  # type: ignore[arg-type]
    )
    return memory, storage


class TestEmptyEngine:
    def test_metrics_on_empty_store(self, tmp_path: Path) -> None:
        memory, storage = _make(tmp_path)
        try:
            m = memory.metrics()
            assert isinstance(m, DecayMetrics)
            assert m.hot_items == 0
            assert m.cold_items == 0
            assert m.reinforcement_total == 0
            assert m.corroboration_total == 0
            assert m.contradiction_total == 0
            assert m.last_tick is None
            assert ItemKind.EVENT in m.per_kind
            assert ItemKind.MEMORY_ITEM in m.per_kind
            ev = m.per_kind[ItemKind.EVENT]
            assert ev.hot_items == 0
            assert ev.cold_items == 0
        finally:
            storage.close()


class TestCountersAggregateAcrossEvents:
    def test_signal_counters_sum_after_record(self, tmp_path: Path) -> None:
        params = DecayParams(half_life_seconds=1e9, threshold=0.0)
        memory, storage = _make(tmp_path, decay_params=params)
        try:
            e1 = memory.observe("a")
            e2 = memory.observe("b")

            memory.reinforce(e1.id, ItemKind.EVENT, count=2)
            memory.reinforce(e2.id, ItemKind.EVENT, count=3)
            memory.contradict(e1.id, ItemKind.EVENT, count=1)
            memory.corroborate(e1.id, ItemKind.EVENT, count=4)

            m = memory.metrics()
            assert m.reinforcement_total == 5  # 2 + 3
            assert m.contradiction_total == 1
            assert m.corroboration_total == 4
            assert m.hot_items == 2
            assert m.cold_items == 0
            ev = m.per_kind[ItemKind.EVENT]
            assert ev.hot_items == 2
            assert ev.reinforcement_total == 5
        finally:
            storage.close()

    def test_cold_items_excluded_from_signal_totals(self, tmp_path: Path) -> None:
        # An item that was reinforced once and then went cold should NOT
        # contribute to the active reinforcement total.
        params = DecayParams(half_life_seconds=1e9, beta=0.10, delta=0.40, threshold=0.10)
        memory, storage = _make(tmp_path, decay_params=params)
        try:
            doomed = memory.observe("doomed")
            keeper = memory.observe("keeper")

            memory.reinforce(keeper.id, ItemKind.EVENT, count=1)

            # Push doomed cold via repeated contradictions.
            memory.contradict(doomed.id, ItemKind.EVENT, count=3)
            assert memory.is_cold(doomed.id, ItemKind.EVENT)

            m = memory.metrics()
            assert m.cold_items == 1
            assert m.hot_items == 1
            # Only `keeper` contributes to the hot reinforcement total.
            assert m.reinforcement_total == 1
            # Contradictions on the now-cold item do not count.
            assert m.contradiction_total == 0
        finally:
            storage.close()


class TestLastTick:
    def test_last_tick_starts_none_then_populated(self, tmp_path: Path) -> None:
        memory, storage = _make(tmp_path)
        try:
            memory.observe("x")
            assert memory.metrics().last_tick is None
            future = _now() + timedelta(seconds=5)
            memory.tick(now=future)
            m = memory.metrics()
            assert m.last_tick is not None
            assert m.last_tick.items_processed >= 1
            assert m.last_tick.duration_ms >= 0
            assert m.last_tick.started_at == future
        finally:
            storage.close()


class TestPerKindBreakdown:
    def test_per_kind_separates_events_and_memory_items(self, tmp_path: Path) -> None:
        from engram.schemas import Level, MemoryItem

        params = DecayParams(half_life_seconds=1e9, threshold=0.0)
        memory, storage = _make(tmp_path, decay_params=params)
        try:
            e = memory.observe("an event")
            mi = MemoryItem(level=Level.SUMMARY, content="a memory item", weight=0.5)
            storage.insert_memory_item(mi)

            memory.reinforce(e.id, ItemKind.EVENT, count=2)
            memory.corroborate(mi.id, ItemKind.MEMORY_ITEM, count=3)

            m = memory.metrics()
            ev_counters = m.per_kind[ItemKind.EVENT]
            mi_counters = m.per_kind[ItemKind.MEMORY_ITEM]
            assert isinstance(ev_counters, KindCounters)
            assert ev_counters.reinforcement_total == 2
            assert ev_counters.corroboration_total == 0
            assert mi_counters.reinforcement_total == 0
            assert mi_counters.corroboration_total == 3
        finally:
            storage.close()


class TestDirectEngineMetrics:
    def test_engine_metrics_match_memory_metrics(self, tmp_path: Path) -> None:
        memory, storage = _make(tmp_path)
        try:
            memory.observe("x")
            engine = DecayEngine(storage)
            m_via_memory = memory.metrics()
            m_via_engine = engine.metrics()
            # Same hot/cold breakdown; last_tick may differ since each
            # engine tracks its own.
            assert m_via_memory.hot_items == m_via_engine.hot_items
            assert m_via_memory.cold_items == m_via_engine.cold_items
            assert m_via_memory.reinforcement_total == m_via_engine.reinforcement_total
        finally:
            storage.close()
