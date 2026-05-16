"""Storage-aware decay engine.

The engine is the read-modify-write loop that ties the pure math in
`engram.decay._math` to the per-row state living in `engram.storage`.

Two entry points cover the whole life cycle:

  `record(item_id, kind, ...)` is called by `Memory.reinforce` /
  `corroborate` / `contradict`. It applies decay-since-last-update plus
  the new signal in one atomic transaction, bumps the per-signal counter,
  and pushes the item into the cold pool if the new weight falls under
  the threshold. The eager apply means consumers immediately see the
  reinforced weight - we do not buffer signals for the next tick.

  `tick(now=...)` is the periodic sweep. It iterates every hot item,
  applies pure decay (no signals - those were applied at record time),
  updates the row, and marks newly-cold items. The delete prune policy
  defers the physical purge to tick rather than to record so per-signal
  cost stays bounded.

Both paths take an injectable clock. Replays with a fixed clock are
bit-identical to the original run, by construction (the math is pure and
no other source of nondeterminism enters the engine).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from engram.decay._math import DecayParams, apply, is_cold
from engram.decay._metrics import DecayMetrics, KindCounters
from engram.schemas import DecayState, ItemKind
from engram.storage._protocol import Storage

# Mode for what to do with items whose weight has dropped below the prune
# threshold. `"cold"` marks `cold_at = now` and leaves the row in place
# (auditable). `"delete"` marks `cold_at = now` and physically removes the
# row at the next `tick`.
PrunePolicy = Literal["cold", "delete"]

# Kinds the engine sweeps in `tick`. Frozen so a user can't smuggle a
# bogus item kind in by mutating a default. Stage 7 adds PROCEDURE;
# the storage layer's per-kind SQL templates handle it identically to
# the others, so no extra wiring is required.
_DEFAULT_KINDS: tuple[ItemKind, ...] = (
    ItemKind.EVENT,
    ItemKind.MEMORY_ITEM,
    ItemKind.PROCEDURE,
)


from engram._time import utcnow as _utcnow  # noqa: E402


@dataclass(frozen=True, slots=True)
class TickResult:
    """Outcome of one `DecayEngine.tick` call.

    `items_processed` counts every hot item the tick observed, regardless
    of whether the weight actually changed. `items_pruned` counts those
    that crossed under the threshold during this tick (it does not include
    items that were already cold). `items_deleted` is non-zero only under
    the `delete` prune policy - and only counts rows actually deleted (not
    rows refused by the provenance guard).
    """

    started_at: datetime
    items_processed: int
    items_pruned: int
    items_deleted: int
    duration_ms: float
    per_kind: dict[ItemKind, dict[str, int]] = field(default_factory=dict)


class DecayEngine:
    """Sync decay engine over a `Storage` backend.

    The engine itself is stateless beyond its parameters; every read and
    write goes through storage so two engines pointed at the same database
    behave identically. The clock is injectable for tests / replays.
    """

    def __init__(
        self,
        storage: Storage,
        *,
        params: DecayParams | None = None,
        prune_policy: PrunePolicy = "cold",
        clock: object | None = None,
        kinds: tuple[ItemKind, ...] = _DEFAULT_KINDS,
        batch_size: int = 1000,
    ) -> None:
        if prune_policy not in ("cold", "delete"):
            raise ValueError(f"prune_policy must be 'cold' or 'delete', got {prune_policy!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not kinds:
            raise ValueError("kinds must be non-empty")

        self._storage = storage
        self._params = params if params is not None else DecayParams()
        self._prune_policy: PrunePolicy = prune_policy
        # `clock` is intentionally a callable returning `datetime`.  We
        # accept `object | None` only to keep the signature open to
        # lambdas and functools.partial.  Previously a caller passing
        # `clock=datetime.now()` (a datetime, not a callable returning
        # one) silently fell back to wall-clock without warning —
        # producing a non-deterministic tick under what looked like a
        # frozen-clock test.  Raise instead so the misuse surfaces.
        if clock is None:
            self._clock = _utcnow
        elif callable(clock):
            self._clock = clock  # type: ignore[assignment]
        else:
            raise TypeError(
                f"DecayEngine.clock must be a callable returning datetime; "
                f"got {type(clock).__name__}.  Pass `clock=lambda: my_dt` "
                f"to inject a fixed time."
            )
        self._kinds = tuple(kinds)
        self._batch_size = batch_size
        self._last_tick: TickResult | None = None

    # --- introspection ------------------------------------------------------

    @property
    def params(self) -> DecayParams:
        return self._params

    @property
    def prune_policy(self) -> PrunePolicy:
        return self._prune_policy

    # --- record -------------------------------------------------------------

    def record(
        self,
        item_id: UUID,
        kind: ItemKind,
        *,
        reinforcement: int = 0,
        corroboration: int = 0,
        contradiction: int = 0,
        now: datetime | None = None,
    ) -> DecayState:
        """Apply decay-since-last + new signal(s) to the item, atomically.

        Raises `KeyError` if the item does not exist, `RuntimeError` if it
        is currently cold (cold items must be unmarked before they accept
        new signals), and `ValueError` on invalid signal counts.
        """
        if reinforcement < 0 or corroboration < 0 or contradiction < 0:
            raise ValueError("signal counts must be non-negative")
        if reinforcement == 0 and corroboration == 0 and contradiction == 0:
            raise ValueError("record requires at least one non-zero signal")

        moment = now if now is not None else self._clock()

        with self._storage.transaction():
            state = self._storage.get_decay_state(item_id, kind)
            if state is None:
                raise KeyError(f"{kind.value} {item_id} not found")
            if state.cold_at is not None:
                raise RuntimeError(f"{kind.value} {item_id} is cold; call unmark_cold first")

            dt = max(0.0, (moment - state.last_decayed_at).total_seconds())
            new_weight = apply(
                weight=state.weight,
                dt_seconds=dt,
                reinforcement=reinforcement,
                corroboration=corroboration,
                contradiction=contradiction,
                params=self._params,
            )
            now_cold = is_cold(new_weight, self._params)
            new_state = state.model_copy(
                update={
                    "weight": new_weight,
                    "reinforcement_count": state.reinforcement_count + reinforcement,
                    "corroboration_count": state.corroboration_count + corroboration,
                    "contradiction_count": state.contradiction_count + contradiction,
                    "last_decayed_at": moment,
                    "cold_at": moment if now_cold else None,
                }
            )
            self._storage.update_decay_state(new_state)
            return new_state

    def reinforce(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.EVENT,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        return self.record(item_id, kind, reinforcement=count, now=now)

    def corroborate(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.EVENT,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        return self.record(item_id, kind, corroboration=count, now=now)

    def contradict(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.EVENT,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        return self.record(item_id, kind, contradiction=count, now=now)

    # --- tick ---------------------------------------------------------------

    def tick(self, *, now: datetime | None = None) -> TickResult:
        """Sweep every hot item; apply pure decay; prune below threshold."""
        moment = now if now is not None else self._clock()
        wall_started = time.perf_counter()
        total_processed = 0
        total_pruned = 0
        total_deleted = 0
        per_kind: dict[ItemKind, dict[str, int]] = {}

        # Wrap the whole sweep in a single transaction so a tick is
        # atomic across kinds.  Previously each kind opened its own
        # transaction — a failure mid-tick committed earlier kinds but
        # rolled back later ones, leaving the database at inconsistent
        # `last_decayed_at` values across kinds and producing replay
        # drift on re-run.  The storage layer treats nested
        # `transaction()` blocks as re-entrant no-ops, so per-call
        # transactions inside `update_decay_state` and
        # `delete_cold_items` continue to work.
        with self._storage.transaction():
            for kind in self._kinds:
                kind_processed = 0
                kind_pruned = 0
                kind_deleted = 0
                # Materialize the hot snapshot so we can update during the
                # same transaction without disturbing the iterator's view.
                states = list(self._storage.iter_decay_states(kind, batch_size=self._batch_size))
                for state in states:
                    kind_processed += 1
                    dt = max(0.0, (moment - state.last_decayed_at).total_seconds())
                    new_weight = apply(weight=state.weight, dt_seconds=dt, params=self._params)
                    became_cold = is_cold(new_weight, self._params)
                    if (
                        new_weight == state.weight
                        and state.last_decayed_at == moment
                        and not became_cold
                    ):
                        # Nothing changed. Skip the write.
                        continue

                    new_state = state.model_copy(
                        update={
                            "weight": new_weight,
                            "last_decayed_at": moment,
                            "cold_at": moment if became_cold else None,
                        }
                    )
                    self._storage.update_decay_state(new_state)
                    if became_cold:
                        kind_pruned += 1

                if self._prune_policy == "delete":
                    try:
                        kind_deleted = self._storage.delete_cold_items(kind)
                    except RuntimeError:
                        # Cold events with provenance can't be hard-deleted;
                        # they remain cold (auditable). Counts as 0 deleted.
                        kind_deleted = 0

                per_kind[kind] = {
                    "processed": kind_processed,
                    "pruned": kind_pruned,
                    "deleted": kind_deleted,
                }
                total_processed += kind_processed
                total_pruned += kind_pruned
                total_deleted += kind_deleted

        duration_ms = (time.perf_counter() - wall_started) * 1000.0
        result = TickResult(
            started_at=moment,
            items_processed=total_processed,
            items_pruned=total_pruned,
            items_deleted=total_deleted,
            duration_ms=duration_ms,
            per_kind=per_kind,
        )
        self._last_tick = result
        return result

    async def tick_async(self, *, now: datetime | None = None) -> TickResult:
        """Async wrapper around `tick`.

        Runs the sync tick on the default thread pool so an asyncio event
        loop is not blocked. Stage 9 will introduce a native async storage
        backend; until then this is the canonical async surface.
        """
        return await asyncio.to_thread(self.tick, now=now)

    # --- metrics ------------------------------------------------------------

    def metrics(self) -> DecayMetrics:
        """Snapshot every counter the engine exposes.

        Reads aggregates from storage (one cheap query per kind) and folds
        in the cached `last_tick` so dashboards can show tick latency
        without re-running the sweep.
        """
        per_kind: dict[ItemKind, KindCounters] = {}
        hot_items = 0
        cold_items = 0
        reinforcement_total = 0
        corroboration_total = 0
        contradiction_total = 0
        for kind in self._kinds:
            totals = self._storage.decay_totals(kind)
            counters = KindCounters(
                kind=kind,
                hot_items=totals["hot_items"],
                cold_items=totals["cold_items"],
                reinforcement_total=totals["reinforcement_total"],
                corroboration_total=totals["corroboration_total"],
                contradiction_total=totals["contradiction_total"],
            )
            per_kind[kind] = counters
            hot_items += counters.hot_items
            cold_items += counters.cold_items
            reinforcement_total += counters.reinforcement_total
            corroboration_total += counters.corroboration_total
            contradiction_total += counters.contradiction_total
        return DecayMetrics(
            hot_items=hot_items,
            cold_items=cold_items,
            reinforcement_total=reinforcement_total,
            corroboration_total=corroboration_total,
            contradiction_total=contradiction_total,
            last_tick=self._last_tick,
            per_kind=per_kind,
        )
