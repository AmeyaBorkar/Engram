"""Engine-level Hypothesis property tests.

These cover the Stage 4 DoD invariants end-to-end (engine + storage),
which the math-only properties in `tests/test_decay_math.py` already
cover at the formula level. The point here is to catch leaks between
the layers: e.g. a counter that round-trips through SQLite as a string,
or a `cold_at` that flips back to NULL by accident.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engram import DecayParams, Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder
from engram.schemas import ItemKind

# Operation strategy: each step is a tuple (kind, count). For "tick", count is
# unused. dt_seconds is a non-negative float advanced between operations.
_OP_KIND = st.sampled_from(["reinforce", "corroborate", "contradict", "tick"])
_OP_COUNT = st.integers(min_value=1, max_value=5)
_OP_STEP = st.tuples(
    _OP_KIND,
    _OP_COUNT,
    st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
)


def _params() -> DecayParams:
    # Tunable but fixed across one property run so we can reason about the
    # invariants. Half-life of 60s + low threshold means decay-only ticks
    # show measurable change without immediately pruning.
    return DecayParams(
        half_life_seconds=60.0,
        beta=0.10,
        gamma=0.05,
        delta=0.20,
        threshold=0.05,
    )


def _make_memory(tmp_path: Path) -> tuple[Memory, SqliteStorage]:
    storage = SqliteStorage(tmp_path / "x.db")
    storage.initialize()
    memory = Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=4),
        decay_params=_params(),
    )
    return memory, storage


@given(steps=st.lists(_OP_STEP, min_size=0, max_size=20))
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_weight_stays_in_unit_interval(tmp_path: Path, steps: list[tuple]) -> None:
    """Across any sequence of signal/tick operations, every stored weight
    stays in [0, 1] - no leak through the storage round-trip."""
    memory, storage = _make_memory(tmp_path)
    try:
        event = memory.observe("x")
        cursor = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for kind, count, dt in steps:
            cursor += timedelta(seconds=dt)
            try:
                if kind == "reinforce":
                    memory.reinforce(event.id, ItemKind.EVENT, count=count, now=cursor)
                elif kind == "contradict":
                    memory.contradict(event.id, ItemKind.EVENT, count=count, now=cursor)
                elif kind == "corroborate":
                    memory.corroborate(event.id, ItemKind.EVENT, count=count, now=cursor)
                else:  # tick
                    memory.tick(now=cursor)
            except RuntimeError:
                # Item went cold mid-sequence; further signals are
                # rejected. That's by-design behavior.
                pass

            state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state is not None
            assert 0.0 <= state.weight <= 1.0
    finally:
        storage.close()


@given(steps=st.lists(_OP_STEP, min_size=0, max_size=20))
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_signal_counters_are_non_decreasing(tmp_path: Path, steps: list[tuple]) -> None:
    """The three signal counters never go backwards across a run."""
    memory, storage = _make_memory(tmp_path)
    try:
        event = memory.observe("x")
        cursor = datetime(2026, 1, 1, tzinfo=timezone.utc)
        last = (0, 0, 0)
        for kind, count, dt in steps:
            cursor += timedelta(seconds=dt)
            try:
                if kind == "reinforce":
                    memory.reinforce(event.id, ItemKind.EVENT, count=count, now=cursor)
                elif kind == "contradict":
                    memory.contradict(event.id, ItemKind.EVENT, count=count, now=cursor)
                elif kind == "corroborate":
                    memory.corroborate(event.id, ItemKind.EVENT, count=count, now=cursor)
                else:
                    memory.tick(now=cursor)
            except RuntimeError:
                pass
            state = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert state is not None
            current = (
                state.reinforcement_count,
                state.corroboration_count,
                state.contradiction_count,
            )
            assert current[0] >= last[0]
            assert current[1] >= last[1]
            assert current[2] >= last[2]
            last = current
    finally:
        storage.close()


@given(
    dts=st.lists(
        st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    )
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_pure_decay_is_monotonic(tmp_path: Path, dts: list[float]) -> None:
    """Repeated tick()s without intervening signals never raise weight."""
    memory, storage = _make_memory(tmp_path)
    try:
        event = memory.observe("x")
        cursor = datetime(2026, 1, 1, tzinfo=timezone.utc)
        prev_state = storage.get_decay_state(event.id, ItemKind.EVENT)
        assert prev_state is not None
        for dt in dts:
            cursor += timedelta(seconds=dt)
            memory.tick(now=cursor)
            current = storage.get_decay_state(event.id, ItemKind.EVENT)
            assert current is not None
            assert current.weight <= prev_state.weight + 1e-12
            prev_state = current
    finally:
        storage.close()


@given(
    initial_weight=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
    reinforcements=st.integers(min_value=1, max_value=5),
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_reinforcement_with_headroom_strictly_raises_weight(
    tmp_path: Path, initial_weight: float, reinforcements: int
) -> None:
    """If we manually set the weight under 1.0 and reinforce with
    `now == last_decayed_at` (no decay-since-last), the new weight is
    strictly greater (or pinned to 1.0)."""
    # Use a huge half-life so the dt~0 path stays exactly at the same
    # weight before the reinforcement signal.
    params = DecayParams(half_life_seconds=1e15, beta=0.10, threshold=0.0)
    storage = SqliteStorage(tmp_path / "x.db")
    storage.initialize()
    memory = Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=4),
        decay_params=params,
    )
    try:
        event = memory.observe("x")
        state = storage.get_decay_state(event.id, ItemKind.EVENT)
        assert state is not None
        seeded = state.model_copy(
            update={"weight": initial_weight, "last_decayed_at": state.last_decayed_at}
        )
        storage.update_decay_state(seeded)
        new_state = memory.reinforce(
            event.id, ItemKind.EVENT, count=reinforcements, now=state.last_decayed_at
        )
        assert new_state.weight > initial_weight or new_state.weight == 1.0
    finally:
        storage.close()
