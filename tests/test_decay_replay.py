"""Replayability test for the decay engine.

Stage 4 DoD: "given a fixed event stream and clock, weights are
bit-identical across runs." The math is pure, the engine is stateless
beyond its parameters, and the storage round-trips floats losslessly -
so the only sources of nondeterminism we have to control are:

  * the wall clock (we inject one)
  * the UUIDs assigned to new events (we pre-seed them)

Embedding generation is deterministic on the fake provider, and storage
ordering is keyed on the seeded UUIDs. With those two pinned, the same
operation sequence MUST produce identical decay states down to the bit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from engram import DecayParams, Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder
from engram.schemas import DecayState, Event, ItemKind


@dataclass(frozen=True, slots=True)
class _Op:
    kind: str  # "observe" | "reinforce" | "contradict" | "corroborate" | "tick"
    target_idx: int = -1


def _build_clock(start: datetime, step: timedelta) -> Callable[[], datetime]:
    cursor = [start]

    def clock() -> datetime:
        out = cursor[0]
        cursor[0] = out + step
        return out

    return clock


_SEED_IDS = (
    UUID("11111111-1111-7111-8111-111111111111"),
    UUID("22222222-2222-7222-8222-222222222222"),
    UUID("33333333-3333-7333-8333-333333333333"),
    UUID("44444444-4444-7444-8444-444444444444"),
)


def _run(
    db_path: Path,
    *,
    operations: list[_Op],
    params: DecayParams,
) -> list[DecayState]:
    """Run `operations` against a fresh database and return the final
    decay states sorted by item_id (so callers can compare run-to-run)."""

    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    clock = _build_clock(start=start, step=timedelta(seconds=37))
    embedder = FakeEmbedder(dim=8)

    with SqliteStorage(db_path) as storage:
        memory = Memory(
            storage=storage,
            embedder=embedder,
            decay_params=params,
            clock=clock,
        )

        observed: list[UUID] = []
        for op in operations:
            now = clock()
            if op.kind == "observe":
                seed = _SEED_IDS[len(observed)]
                # Pin the timestamp too, since `Event` defaults pull from
                # the wall clock.
                event = Event(id=seed, content=f"event-{len(observed)}", created_at=now)
                memory.observe(event)
                observed.append(seed)
            elif op.kind == "reinforce":
                memory.reinforce(observed[op.target_idx], ItemKind.EVENT, now=now)
            elif op.kind == "contradict":
                memory.contradict(observed[op.target_idx], ItemKind.EVENT, now=now)
            elif op.kind == "corroborate":
                memory.corroborate(observed[op.target_idx], ItemKind.EVENT, now=now)
            elif op.kind == "tick":
                memory.tick(now=now)
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown op {op.kind}")

        return sorted(
            storage.iter_decay_states(ItemKind.EVENT, include_cold=True),
            key=lambda s: s.item_id.bytes,
        )


def test_replay_is_bit_identical(tmp_path: Path) -> None:
    operations = [
        _Op("observe"),
        _Op("observe"),
        _Op("observe"),
        _Op("reinforce", 0),
        _Op("contradict", 1),
        _Op("tick"),
        _Op("reinforce", 0),
        _Op("corroborate", 2),
        _Op("tick"),
        _Op("contradict", 1),
        _Op("contradict", 1),
        _Op("tick"),
    ]
    params = DecayParams(half_life_seconds=600.0, threshold=0.05)

    a = _run(tmp_path / "a.db", operations=operations, params=params)
    b = _run(tmp_path / "b.db", operations=operations, params=params)

    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert sa.item_id == sb.item_id
        # Bit-identical floats. `==` on float is the right comparison here:
        # the formula is pure and the storage round-trip preserves the
        # exact float bytes, so any mismatch is a real determinism bug.
        assert sa.weight == sb.weight, (
            f"weight diverged: {sa.weight!r} != {sb.weight!r} for {sa.item_id}"
        )
        assert sa.reinforcement_count == sb.reinforcement_count
        assert sa.corroboration_count == sb.corroboration_count
        assert sa.contradiction_count == sb.contradiction_count
        assert sa.last_decayed_at == sb.last_decayed_at
        assert sa.cold_at == sb.cold_at


def test_replay_with_cold_marker_is_stable(tmp_path: Path) -> None:
    # Different operation set: many contradictions push at least one item
    # cold. Verify the cold_at timestamp is also bit-identical across runs.
    operations = [
        _Op("observe"),
        _Op("contradict", 0),
        _Op("contradict", 0),
        _Op("contradict", 0),
        _Op("tick"),
    ]
    params = DecayParams(half_life_seconds=1e9, delta=0.40, threshold=0.10)

    a = _run(tmp_path / "a.db", operations=operations, params=params)
    b = _run(tmp_path / "b.db", operations=operations, params=params)

    assert len(a) == 1 == len(b)
    assert a[0] == b[0]
    assert a[0].cold_at is not None
