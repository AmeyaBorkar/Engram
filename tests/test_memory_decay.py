"""End-to-end memory decay tests.

`Memory` exposes the decay surface (`reinforce`, `corroborate`,
`contradict`, `tick`) and `retrieve` filters cold items by default. These
tests exercise the public API rather than the engine directly.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engram import DecayParams, Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder
from engram.schemas import ItemKind


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_memory(
    tmp_path: Path,
    *,
    decay_params: DecayParams | None = None,
    prune_policy: str = "cold",
) -> tuple[Memory, SqliteStorage]:
    storage = SqliteStorage(tmp_path / "x.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=8)
    memory = Memory(
        storage=storage,
        embedder=embedder,
        decay_params=decay_params,
        prune_policy=prune_policy,  # type: ignore[arg-type]
    )
    return memory, storage


# ---------------------------------------------------------------------------
# observe / retrieve still work
# ---------------------------------------------------------------------------


class TestObserveRetrieve:
    def test_observe_and_retrieve_cold_filter(self, tmp_path: Path) -> None:
        memory, storage = _make_memory(tmp_path)
        try:
            hot = memory.observe("hot fact")
            cold = memory.observe("cold fact")
            # Mark `cold` cold via low-level storage.
            storage.mark_cold(cold.id, ItemKind.EVENT, at=_now())

            # Default: cold filtered out.
            results = memory.retrieve("hot fact", k=10)
            assert {r.item_id for r in results} == {hot.id}

            # Explicit override: include cold.
            with_cold = memory.retrieve("hot fact", k=10, include_cold=True)
            assert {r.item_id for r in with_cold} == {hot.id, cold.id}
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# reinforce / corroborate / contradict
# ---------------------------------------------------------------------------


class TestSignalSurface:
    def test_reinforce_increments_counter_and_weight(self, tmp_path: Path) -> None:
        # Use a long half-life so decay over the test interval is negligible.
        params = DecayParams(half_life_seconds=1e9, beta=0.10, threshold=0.0)
        memory, storage = _make_memory(tmp_path, decay_params=params)
        try:
            event = memory.observe("x")
            # Drop weight manually so reinforcement has headroom.
            initial = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert initial is not None
            storage.update_decay_state(
                initial.model_copy(update={"weight": 0.5, "last_decayed_at": _now()})
            )
            new_state = memory.reinforce(event.id, ItemKind.EVENT)
            assert new_state.reinforcement_count == 1
            assert new_state.weight > 0.55
        finally:
            storage.close()

    def test_contradict_eventually_marks_cold(self, tmp_path: Path) -> None:
        params = DecayParams(half_life_seconds=1e9, delta=0.40, threshold=0.10)
        memory, storage = _make_memory(tmp_path, decay_params=params)
        try:
            event = memory.observe("x")
            # Three contradictions: 1.0 - 3*0.4 = -0.2 -> clamp to 0 -> cold.
            for _ in range(3):
                memory.contradict(event.id, ItemKind.EVENT)
            assert memory.is_cold(event.id, ItemKind.EVENT)
        finally:
            storage.close()

    def test_corroborate_targets_memory_item_by_default(self, tmp_path: Path) -> None:
        # corroboration is a memory-item signal in the README; verify the
        # default kind agrees.
        params = DecayParams(half_life_seconds=1e9, gamma=0.10, threshold=0.0)
        memory, storage = _make_memory(tmp_path, decay_params=params)
        try:
            from engram.schemas import Level, MemoryItem

            item = MemoryItem(level=Level.SUMMARY, content="x", weight=0.5)
            storage.insert_memory_item(item)
            new_state = memory.corroborate(item.id, count=1)
            assert new_state.item_kind is ItemKind.MEMORY_ITEM
            assert new_state.corroboration_count == 1
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# tick / tick_async
# ---------------------------------------------------------------------------


class TestTickSurface:
    def test_tick_returns_tickresult(self, tmp_path: Path) -> None:
        memory, storage = _make_memory(tmp_path)
        try:
            for i in range(3):
                memory.observe(f"e{i}")
            result = memory.tick()
            assert result.items_processed >= 3
        finally:
            storage.close()

    def test_tick_async_runs(self, tmp_path: Path) -> None:
        memory, storage = _make_memory(tmp_path)
        try:
            memory.observe("x")
            result = asyncio.run(memory.tick_async())
            assert result.items_processed >= 1
        finally:
            storage.close()

    def test_tick_after_long_idle_marks_cold(self, tmp_path: Path) -> None:
        params = DecayParams(half_life_seconds=1.0, threshold=0.05)
        memory, storage = _make_memory(tmp_path, decay_params=params)
        try:
            event = memory.observe("x")
            future = event.created_at + timedelta(seconds=20)
            memory.tick(now=future)
            assert memory.is_cold(event.id, ItemKind.EVENT)
            results = memory.retrieve("x", k=5)
            # Cold filtered out by default.
            assert all(r.item_id != event.id for r in results)
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# is_cold helper
# ---------------------------------------------------------------------------


class TestIsCold:
    def test_unknown_item_is_not_cold(self, tmp_path: Path) -> None:
        from uuid import uuid4

        memory, storage = _make_memory(tmp_path)
        try:
            assert memory.is_cold(uuid4()) is False
        finally:
            storage.close()

    def test_below_threshold_without_cold_at_is_cold(self, tmp_path: Path) -> None:
        # Defensive: if a row has weight below threshold but `cold_at` is
        # NULL (e.g. an external writer bypassed the engine), `is_cold`
        # still answers True so retrievers can act.
        params = DecayParams(threshold=0.5)
        memory, storage = _make_memory(tmp_path, decay_params=params)
        try:
            event = memory.observe("x")
            state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state is not None
            storage.update_decay_state(state.model_copy(update={"weight": 0.1}))
            assert memory.is_cold(event.id, ItemKind.EVENT)
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_no_decay_args_uses_default_params(self, tmp_path: Path) -> None:
        memory, storage = _make_memory(tmp_path)
        try:
            assert memory.decay.params == DecayParams()
            assert memory.decay.prune_policy == "cold"
        finally:
            storage.close()

    def test_observe_does_not_require_decay_call(self, tmp_path: Path) -> None:
        memory, storage = _make_memory(tmp_path)
        try:
            event = memory.observe("hello")
            results = memory.retrieve("hello", k=1)
            assert results
            assert results[0].item_id == event.id
        finally:
            storage.close()
