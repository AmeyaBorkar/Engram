"""Reconciler engine.

The Stage 5 contradiction detector flags pairs of memory items that
disagree (status=OPEN `Conflict` rows). The reconciler resolves them:

  * Pick a winner per the chosen `Resolution` policy.
  * Invalidate the loser through storage (sets `invalidated_at` +
    `invalidated_by`). Default retrieve drops invalidated items; `as_of`
    queries can still surface them when the timestamp predates the
    invalidation.
  * Flip the conflict row OPEN -> RESOLVED with the winner id,
    resolution, and timestamp recorded for audit.

Policies:

  * `PREFER_RECENT`: pick the item with the later `created_at`. Tie ->
    deterministic lexicographic id compare.
  * `PREFER_TRUSTED`: pick the item with the higher `source_trust`
    (None treated as 0.0). Tie -> falls back to PREFER_RECENT.
  * `PREFER_FREQUENT`: pick the item with the higher
    `corroboration_count` (from decay state). Tie -> falls back to
    PREFER_RECENT.
  * `KEEP_BOTH`: no winner; both items stay valid. The conflict is
    still marked RESOLVED so it stops surfacing on audits.
  * `MANUAL`: caller picks the winner via `manual_winner_id` (must be
    source or target of the conflict).

Stateless after construction; the storage handle is the source of
truth. The engine is per-`Memory` instance.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import UUID

from engram.schemas import (
    Conflict,
    ConflictStatus,
    ItemKind,
    MemoryItem,
    Resolution,
)
from engram.storage._protocol import Storage


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Reconciler:
    """Conflict resolver."""

    def __init__(
        self,
        storage: Storage,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage = storage
        self._clock: Callable[[], datetime] = clock or _utcnow

    def reconcile(
        self,
        conflict_id: UUID,
        *,
        resolution: Resolution,
        manual_winner_id: UUID | None = None,
        now: datetime | None = None,
    ) -> Conflict:
        """Resolve `conflict_id` per `resolution`.

        Raises:
          KeyError: the conflict id does not exist (or, defensively,
            one of the referenced memory items was deleted).
          RuntimeError: the conflict is already resolved. Callers
            should filter on `status=OPEN` first if they want skip
            semantics instead.
          ValueError: invalid `manual_winner_id` for `Resolution.MANUAL`,
            or an unsupported `resolution` value.
        """
        conflict = self._storage.get_conflict(conflict_id)
        if conflict is None:
            raise KeyError(f"conflict {conflict_id} not found")
        if conflict.status is ConflictStatus.RESOLVED:
            raise RuntimeError(
                f"conflict {conflict_id} is already resolved "
                f"(resolution={conflict.resolution})"
            )

        when = now if now is not None else self._clock()
        winner_id = self._pick_winner(
            conflict, resolution, manual_winner_id=manual_winner_id
        )
        loser_id = self._loser(conflict, winner_id) if winner_id is not None else None
        if loser_id is not None and winner_id is not None:
            self._storage.invalidate_memory_item(loser_id, at=when, by=winner_id)
        return self._storage.resolve_conflict(
            conflict_id,
            resolution=resolution,
            resolved_winner_id=winner_id,
            resolved_at=when,
        )

    def _pick_winner(
        self,
        conflict: Conflict,
        resolution: Resolution,
        *,
        manual_winner_id: UUID | None,
    ) -> UUID | None:
        if resolution is Resolution.KEEP_BOTH:
            return None
        if resolution is Resolution.MANUAL:
            if manual_winner_id is None:
                raise ValueError(
                    "manual_winner_id is required when resolution=MANUAL"
                )
            if manual_winner_id not in (
                conflict.source_item_id,
                conflict.target_item_id,
            ):
                raise ValueError(
                    "manual_winner_id must equal source_item_id or target_item_id"
                )
            return manual_winner_id
        source = self._fetch_or_raise(conflict.source_item_id)
        target = self._fetch_or_raise(conflict.target_item_id)
        if resolution is Resolution.PREFER_RECENT:
            return _pick_by_recency(source, target)
        if resolution is Resolution.PREFER_TRUSTED:
            ts = source.source_trust if source.source_trust is not None else 0.0
            tt = target.source_trust if target.source_trust is not None else 0.0
            if ts != tt:
                return source.id if ts > tt else target.id
            return _pick_by_recency(source, target)
        if resolution is Resolution.PREFER_FREQUENT:
            cs = self._corroboration(source.id)
            ct = self._corroboration(target.id)
            if cs != ct:
                return source.id if cs > ct else target.id
            return _pick_by_recency(source, target)
        # Defensive: unreachable when Resolution enum stays exhaustive.
        raise ValueError(f"unsupported resolution: {resolution!r}")  # pragma: no cover

    def _loser(self, conflict: Conflict, winner_id: UUID) -> UUID:
        if winner_id == conflict.source_item_id:
            return conflict.target_item_id
        return conflict.source_item_id

    def _fetch_or_raise(self, item_id: UUID) -> MemoryItem:
        item = self._storage.get_memory_item(item_id)
        if item is None:  # pragma: no cover - dangling FK (cascade racy)
            raise KeyError(f"memory_item {item_id} referenced by conflict not found")
        return item

    def _corroboration(self, item_id: UUID) -> int:
        state = self._storage.get_decay_state(item_id, ItemKind.MEMORY_ITEM)
        return 0 if state is None else state.corroboration_count


def _pick_by_recency(source: MemoryItem, target: MemoryItem) -> UUID:
    """Tie-break helper: later `created_at` wins; equal -> lex by id."""
    if source.created_at != target.created_at:
        return source.id if source.created_at > target.created_at else target.id
    return source.id if source.id.bytes > target.id.bytes else target.id
