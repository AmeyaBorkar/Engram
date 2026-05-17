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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from engram.decay._math import DecayParams, apply, is_cold
from engram.decay._metrics import DecayMetrics, KindCounters
from engram.schemas import DecayState, ItemKind
from engram.storage._protocol import Storage

# Message-substring marker that storage layers use when they refuse to
# hard-delete cold events because they participate in provenance links.
# The decay engine catches RuntimeError narrowly by message so unrelated
# RuntimeErrors (transaction conflicts, etc.) propagate instead of being
# silently counted as 0 deletions.
_COLD_PROVENANCE_MARKER = "cold event(s) with provenance"

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


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


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
        # `clock` is intentionally a callable returning `datetime`. We accept
        # `object | None` only to keep the signature open to lambdas and
        # functools.partial; the runtime check below is what matters.
        self._clock = clock if callable(clock) else _utcnow
        self._kinds = tuple(kinds)
        self._batch_size = batch_size
        self._last_tick: TickResult | None = None
        # Concurrency guards on the sweep. Two threads calling `tick`
        # concurrently would race on the per-kind iterator + update inside
        # one transaction (the storage layer's transaction is per-engine,
        # but the iterator-then-update pattern is not safe to interleave).
        # The threading lock protects the sync path; an asyncio Lock is
        # lazy-created on first `tick_async` so we never construct one
        # outside a running event loop.
        self._tick_lock = threading.Lock()
        self._tick_async_lock: asyncio.Lock | None = None

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
            # `update_decay_state` does not invalidate the vector/BM25
            # indexes, but `mark_cold` does. When a record() call drops
            # the weight below threshold we must go through mark_cold so
            # the cold row is excluded from future vector / BM25 queries
            # immediately; otherwise a retrieve in the same transaction
            # window would surface a row the user just contradicted into
            # the cold pool. Update first to commit the new weight/counts
            # (cold_at stays None on that write), then mark_cold to flip
            # the cold pointer and invalidate the indexes in one step.
            new_state = state.model_copy(
                update={
                    "weight": new_weight,
                    "reinforcement_count": state.reinforcement_count + reinforcement,
                    "corroboration_count": state.corroboration_count + corroboration,
                    "contradiction_count": state.contradiction_count + contradiction,
                    "last_decayed_at": moment,
                    "cold_at": None,
                }
            )
            self._storage.update_decay_state(new_state)
            if now_cold:
                self._storage.mark_cold(item_id, kind, at=moment)
                new_state = new_state.model_copy(update={"cold_at": moment})
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
        """Sweep every hot item; apply pure decay; prune below threshold.

        Concurrency: protected by `self._tick_lock`. Two threads calling
        `tick` concurrently is a programming error -- the iterator + per-row
        write pattern cannot be safely interleaved against the same storage
        instance. We acquire under a `Lock.acquire(blocking=False)`-style
        check: a second concurrent caller raises `RuntimeError` rather than
        silently serializing (which would mask the misuse).
        """
        if not self._tick_lock.acquire(blocking=False):
            raise RuntimeError(
                "DecayEngine.tick is already running in another thread; "
                "do not call tick concurrently against the same engine"
            )
        try:
            return self._tick_locked(now=now)
        finally:
            self._tick_lock.release()

    def _tick_locked(self, *, now: datetime | None) -> TickResult:
        """Inner tick body; runs under `self._tick_lock`."""
        moment = now if now is not None else self._clock()
        wall_started = time.perf_counter()
        total_processed = 0
        total_pruned = 0
        total_deleted = 0
        per_kind: dict[ItemKind, dict[str, int]] = {}

        for kind in self._kinds:
            kind_processed, kind_pruned, kind_deleted = self._tick_kind(kind, moment=moment)
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

    def _tick_kind(self, kind: ItemKind, *, moment: datetime) -> tuple[int, int, int]:
        """Sweep one kind; return (processed, pruned, deleted) for it.

        Reads the iterator into per-batch chunks and flushes each chunk
        inside one transaction before moving to the next. A previous
        implementation materialized the entire iterator into a list,
        which defeated streaming and forced a million-row decay sweep to
        hold every DecayState in memory. Chunking keeps peak memory
        proportional to `batch_size` (default 1000 rows) regardless of
        store size, while still emitting one transaction per chunk to
        keep the iterator's read snapshot separate from the writes.

        Cold transitions go through `mark_cold` rather than embedding
        `cold_at` in `update_decay_state` -- `mark_cold` invalidates the
        vector and BM25 indexes; `update_decay_state` does not. Without
        the routing, a row that newly crossed under threshold would
        still appear in retrieve until the next index rebuild.
        """
        kind_processed = 0
        kind_pruned = 0
        kind_deleted = 0

        # Drain the iterator chunk by chunk so peak memory is O(batch_size)
        # rather than O(store size). Each chunk is one transaction so
        # mid-sweep failures cap the loss to a single chunk's worth of
        # rolled-back writes.
        chunk: list[DecayState] = []
        iterator = self._storage.iter_decay_states(kind, batch_size=self._batch_size)
        try:
            while True:
                # Pull at most batch_size states into a list, then close
                # the iterator's cursor while we mutate. We do not hold
                # the SELECT cursor open across writes -- a sqlite cursor
                # walking the same table that is being updated in the
                # same connection can return inconsistent rows under WAL.
                chunk.clear()
                try:
                    for _ in range(self._batch_size):
                        chunk.append(next(iterator))
                except StopIteration:
                    pass
                if not chunk:
                    break
                p, k = self._flush_chunk(chunk, kind=kind, moment=moment)
                kind_processed += p
                kind_pruned += k
                if len(chunk) < self._batch_size:
                    break
        finally:
            # Close the iterator if the storage backend exposes one
            # (`Iterator` is a generator on SqliteStorage; close releases
            # the cursor immediately). Generators always have .close().
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

        # Delete-policy pass: only after the per-chunk mark_cold writes
        # have committed, so the cold pool includes anything we just
        # transitioned in this sweep.
        if self._prune_policy == "delete":
            try:
                with self._storage.transaction():
                    kind_deleted = self._storage.delete_cold_items(kind)
            except RuntimeError as exc:
                # Narrowly catch the storage-layer's
                # "cannot delete cold event(s) with provenance" case;
                # any other RuntimeError is a real bug and MUST
                # propagate. Match the documented marker string from
                # the storage layer rather than swallowing every
                # RuntimeError.
                if _COLD_PROVENANCE_MARKER not in str(exc):
                    raise
                # Provenance-protected cold events stay cold (auditable).
                # Counts as 0 deleted.
                kind_deleted = 0
        return kind_processed, kind_pruned, kind_deleted

    def _flush_chunk(
        self,
        chunk: list[DecayState],
        *,
        kind: ItemKind,
        moment: datetime,
    ) -> tuple[int, int]:
        """Apply decay to one chunk and write back. Returns (processed, pruned).

        Runs inside one transaction so the writes flush atomically per
        chunk; a million-row sweep with batch_size=1000 commits 1000
        times instead of once-at-end. That trades durability granularity
        for memory boundedness, which is the right call when a tick has
        already been split across many minutes.
        """
        processed = 0
        pruned = 0
        with self._storage.transaction():
            for state in chunk:
                processed += 1
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
                        "cold_at": None,
                    }
                )
                self._storage.update_decay_state(new_state)
                if became_cold:
                    # mark_cold invalidates vector + BM25 indexes; the
                    # plain update_decay_state above does not. Route the
                    # cold transition through it so the row is excluded
                    # from retrieve immediately after this transaction
                    # commits.
                    self._storage.mark_cold(state.item_id, kind, at=moment)
                    pruned += 1
        return processed, pruned

    async def tick_async(self, *, now: datetime | None = None) -> TickResult:
        """Async wrapper around `tick`.

        Runs the sync tick on the default thread pool so an asyncio event
        loop is not blocked. Concurrent `await tick_async()` coroutines on
        the same engine serialize via `self._tick_async_lock` so we do not
        schedule two `to_thread` workers that race each other on the inner
        `self._tick_lock`. Stage 9 will introduce a native async storage
        backend; until then this is the canonical async surface.
        """
        # Lazy-create the asyncio.Lock inside a running loop so we never
        # construct one outside a loop context.
        if self._tick_async_lock is None:
            self._tick_async_lock = asyncio.Lock()
        async with self._tick_async_lock:
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
