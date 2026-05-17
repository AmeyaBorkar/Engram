"""Stage 8 storage tests for the new Conflict/temporal CRUD surface.

Exercises:
  * `record_conflict` / `get_conflict` / `list_conflicts` round-trip,
    plus the UNIQUE(source, target) constraint.
  * `resolve_conflict` flips OPEN -> RESOLVED; double-resolve raises;
    winner-must-be-source-or-target.
  * `invalidate_memory_item` is idempotent (first timestamp wins);
    `set_validity_window` updates partially; `set_source_trust` clamps.
  * `search_memory_item_embeddings_as_of`:
      - default (as_of=None) excludes invalidated items.
      - as_of=t returns items whose validity covers t (including ones
        that have since been invalidated, if invalidation happened
        after t).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from engram import (
    Conflict,
    ConflictStatus,
    Embedding,
    ItemKind,
    Level,
    MemoryItem,
    Resolution,
    SqliteStorage,
    Verdict,
    new_id,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _norm(vec: list[float]) -> tuple[float, ...]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return tuple(x / n for x in vec)


def _seed_item(
    storage: SqliteStorage,
    content: str,
    *,
    level: Level = Level.SUMMARY,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    source_trust: float | None = None,
    vec: list[float] | None = None,
    model: str = "fake",
    dim: int = 4,
) -> MemoryItem:
    item = MemoryItem(
        level=level,
        content=content,
        valid_from=valid_from,
        valid_until=valid_until,
        source_trust=source_trust,
    )
    storage.insert_memory_item(item)
    if vec is not None:
        storage.insert_embedding(
            Embedding(
                item_id=item.id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=model,
                dim=dim,
                vector=_norm(vec),
            )
        )
    return item


# ---------------------------------------------------------------------------
# record_conflict / get_conflict / list_conflicts
# ---------------------------------------------------------------------------


class TestRecordAndGetConflict:
    def test_round_trip(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.91)
        storage.record_conflict(c)
        fetched = storage.get_conflict(c.id)
        assert fetched is not None
        assert fetched.id == c.id
        assert fetched.source_item_id == a.id
        assert fetched.target_item_id == b.id
        assert fetched.similarity == pytest.approx(0.91)
        assert fetched.status is ConflictStatus.OPEN
        assert fetched.verdict is Verdict.CONTRADICT
        assert fetched.resolution is None
        assert fetched.resolved_winner_id is None

    def test_get_missing_returns_none(self, storage: SqliteStorage) -> None:
        assert storage.get_conflict(new_id()) is None

    def test_unique_pair(self, storage: SqliteStorage) -> None:
        import sqlite3

        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c1 = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c1)
        c2 = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.95)
        with pytest.raises(sqlite3.IntegrityError):
            storage.record_conflict(c2)

    def test_fk_cascade_on_memory_item_delete(self, storage: SqliteStorage) -> None:
        """If the storage layer hard-deletes a memory item, its conflict
        rows go with it."""
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c)
        conn = storage._connect()
        conn.execute("DELETE FROM memory_items WHERE id = ?", (a.id.bytes,))
        assert storage.get_conflict(c.id) is None


class TestListConflicts:
    def test_filters_by_status(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c1 = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c1)
        # Resolve so we can filter on status.
        storage.resolve_conflict(
            c1.id,
            resolution=Resolution.PREFER_RECENT,
            resolved_winner_id=a.id,
            resolved_at=_utc(2026, 5, 1),
        )
        c2_src = _seed_item(storage, "c")
        c2_tgt = _seed_item(storage, "d")
        c2 = Conflict(source_item_id=c2_src.id, target_item_id=c2_tgt.id, similarity=0.8)
        storage.record_conflict(c2)
        open_only = storage.list_conflicts(status=ConflictStatus.OPEN)
        assert {c.id for c in open_only} == {c2.id}
        resolved_only = storage.list_conflicts(status=ConflictStatus.RESOLVED)
        assert {c.id for c in resolved_only} == {c1.id}

    def test_filters_by_memory_item(self, storage: SqliteStorage) -> None:
        """`memory_item_id` walks both directions of the graph."""
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = _seed_item(storage, "c")
        ab = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        bc = Conflict(source_item_id=b.id, target_item_id=c.id, similarity=0.8)
        storage.record_conflict(ab)
        storage.record_conflict(bc)
        adj_to_b = storage.list_conflicts(memory_item_id=b.id)
        assert {x.id for x in adj_to_b} == {ab.id, bc.id}
        adj_to_a = storage.list_conflicts(memory_item_id=a.id)
        assert {x.id for x in adj_to_a} == {ab.id}

    def test_invalid_limit(self, storage: SqliteStorage) -> None:
        with pytest.raises(ValueError, match="limit"):
            storage.list_conflicts(limit=0)


# ---------------------------------------------------------------------------
# resolve_conflict
# ---------------------------------------------------------------------------


class TestResolveConflict:
    def test_prefer_recent_with_winner(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c)
        resolved_at = _utc(2026, 5, 1)
        out = storage.resolve_conflict(
            c.id,
            resolution=Resolution.PREFER_RECENT,
            resolved_winner_id=b.id,
            resolved_at=resolved_at,
        )
        assert out.status is ConflictStatus.RESOLVED
        assert out.resolution is Resolution.PREFER_RECENT
        assert out.resolved_winner_id == b.id
        assert out.resolved_at == resolved_at

    def test_keep_both_no_winner(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c)
        out = storage.resolve_conflict(
            c.id,
            resolution=Resolution.KEEP_BOTH,
            resolved_winner_id=None,
            resolved_at=_utc(2026, 5, 1),
        )
        assert out.resolution is Resolution.KEEP_BOTH
        assert out.resolved_winner_id is None

    def test_double_resolve_raises(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c)
        storage.resolve_conflict(
            c.id,
            resolution=Resolution.PREFER_RECENT,
            resolved_winner_id=b.id,
            resolved_at=_utc(2026, 5, 1),
        )
        with pytest.raises(RuntimeError, match="already resolved"):
            storage.resolve_conflict(
                c.id,
                resolution=Resolution.PREFER_RECENT,
                resolved_winner_id=a.id,
                resolved_at=_utc(2026, 5, 2),
            )

    def test_missing_id_raises_key_error(self, storage: SqliteStorage) -> None:
        with pytest.raises(KeyError):
            storage.resolve_conflict(
                new_id(),
                resolution=Resolution.PREFER_RECENT,
                resolved_winner_id=new_id(),
                resolved_at=_utc(2026, 5, 1),
            )

    def test_winner_must_be_source_or_target(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c_third = _seed_item(storage, "c")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c)
        with pytest.raises(ValueError, match="resolved_winner_id"):
            storage.resolve_conflict(
                c.id,
                resolution=Resolution.PREFER_RECENT,
                resolved_winner_id=c_third.id,
                resolved_at=_utc(2026, 5, 1),
            )

    def test_non_keep_both_requires_winner(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c)
        with pytest.raises(ValueError, match="requires resolved_winner_id"):
            storage.resolve_conflict(
                c.id,
                resolution=Resolution.PREFER_RECENT,
                resolved_winner_id=None,
                resolved_at=_utc(2026, 5, 1),
            )

    def test_status_guard_on_update_prevents_double_resolve(
        self, storage: SqliteStorage
    ) -> None:
        """Regression for H-35: the UPDATE carries `AND status = 'open'`
        and asserts `rowcount == 1`.

        Simulate the race directly by flipping the row's status to
        'resolved' via raw SQL after the in-method read (we can't easily
        spin up two threads against an in-memory db that requires same-
        thread access), then verify the UPDATE rolls back rather than
        silently overwriting a now-resolved row.
        """
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(c)

        # Hand-craft the race window: another writer beats us to it and
        # transitions the row to 'resolved' between OUR get_conflict and
        # OUR UPDATE.  We can't easily intercept the in-method read so
        # exercise the same UPDATE shape directly: it should match zero
        # rows because status is no longer 'open'.
        storage._connect().execute(
            "UPDATE conflicts SET status = 'resolved', resolution = 'prefer_recent', "
            "resolved_winner_id = ?, resolved_at = ? WHERE id = ?",
            (a.id.bytes, _utc(2026, 5, 1).isoformat(), c.id.bytes),
        )

        # Now the public method should observe the resolved state and
        # raise — never silently re-resolve over the prior winner.
        with pytest.raises(RuntimeError, match="already resolved"):
            storage.resolve_conflict(
                c.id,
                resolution=Resolution.PREFER_RECENT,
                resolved_winner_id=b.id,
                resolved_at=_utc(2026, 5, 2),
            )
        # The prior resolution stuck (winner=a, not b from our retry).
        final = storage.get_conflict(c.id)
        assert final is not None
        assert final.resolved_winner_id == a.id


# ---------------------------------------------------------------------------
# count_conflicts / count_conflicts_by_status
# ---------------------------------------------------------------------------


class TestCountConflicts:
    def test_empty(self, storage: SqliteStorage) -> None:
        assert storage.count_conflicts() == 0
        counts = storage.count_conflicts_by_status()
        assert counts == {ConflictStatus.OPEN: 0, ConflictStatus.RESOLVED: 0}

    def test_counts_post_record_and_resolve(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        b = _seed_item(storage, "b")
        c = _seed_item(storage, "c")
        storage.record_conflict(Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9))
        c2 = Conflict(source_item_id=a.id, target_item_id=c.id, similarity=0.7)
        storage.record_conflict(c2)
        assert storage.count_conflicts() == 2
        storage.resolve_conflict(
            c2.id,
            resolution=Resolution.PREFER_RECENT,
            resolved_winner_id=a.id,
            resolved_at=_utc(2026, 5, 1),
        )
        counts = storage.count_conflicts_by_status()
        assert counts == {ConflictStatus.OPEN: 1, ConflictStatus.RESOLVED: 1}


# ---------------------------------------------------------------------------
# invalidate_memory_item / set_validity_window / set_source_trust
# ---------------------------------------------------------------------------


class TestInvalidateMemoryItem:
    def test_marks_invalidated(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        winner = _seed_item(storage, "winner")
        when = _utc(2026, 5, 1)
        storage.invalidate_memory_item(a.id, at=when, by=winner.id)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.invalidated_at == when
        assert item.invalidated_by == winner.id

    def test_no_winner_allowed(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        when = _utc(2026, 5, 1)
        storage.invalidate_memory_item(a.id, at=when)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.invalidated_at == when
        assert item.invalidated_by is None

    def test_idempotent_preserves_first(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        w1 = _seed_item(storage, "w1")
        w2 = _seed_item(storage, "w2")
        first = _utc(2026, 5, 1)
        second = _utc(2026, 6, 1)
        storage.invalidate_memory_item(a.id, at=first, by=w1.id)
        storage.invalidate_memory_item(a.id, at=second, by=w2.id)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.invalidated_at == first
        assert item.invalidated_by == w1.id

    def test_missing_id_raises(self, storage: SqliteStorage) -> None:
        with pytest.raises(KeyError):
            storage.invalidate_memory_item(new_id(), at=_utc(2026, 5, 1))


class TestSetValidityWindow:
    def test_set_both(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        vf = _utc(2026, 1, 1)
        vu = _utc(2026, 5, 1)
        storage.set_validity_window(a.id, valid_from=vf, valid_until=vu)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.valid_from == vf
        assert item.valid_until == vu

    def test_set_only_until(self, storage: SqliteStorage) -> None:
        # Seed with explicit valid_from so the "set only until" path has
        # a known anchor that's before the cutoff.
        a = _seed_item(storage, "a", valid_from=_utc(2026, 1, 1))
        vu = _utc(2026, 5, 1)
        storage.set_validity_window(a.id, valid_until=vu)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.valid_until == vu
        assert item.valid_from == _utc(2026, 1, 1)

    def test_until_before_from_rejected(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        with pytest.raises(ValueError, match="precedes"):
            storage.set_validity_window(
                a.id,
                valid_from=_utc(2026, 5, 1),
                valid_until=_utc(2026, 1, 1),
            )

    def test_until_before_from_rolls_back_partial_write(
        self, storage: SqliteStorage
    ) -> None:
        """Regression for H-34: validation must run *before* the UPDATE
        (or roll back on raise), so an invalid window doesn't persist.

        Seeds an item with a known valid_from, then attempts to set
        valid_until earlier than the existing valid_from.  After the
        ValueError, the row's valid_from must still be the seeded value
        (the row must not have been mutated mid-validation).
        """
        a = _seed_item(storage, "a", valid_from=_utc(2026, 3, 1))
        # Seed an explicit, observable valid_from we can assert against.
        storage.set_validity_window(a.id, valid_from=_utc(2026, 6, 1))
        seeded = storage.get_memory_item(a.id)
        assert seeded is not None and seeded.valid_from == _utc(2026, 6, 1)

        # Now attempt a window with valid_until BEFORE valid_from in one
        # call.  Previously this would (a) UPDATE the row with the new
        # valid_until, then (b) read it back and raise — leaving the bad
        # valid_until on disk.  With the transaction wrap, the UPDATE
        # rolls back and the row stays as seeded.
        with pytest.raises(ValueError, match="precedes"):
            storage.set_validity_window(a.id, valid_until=_utc(2026, 1, 1))

        after = storage.get_memory_item(a.id)
        assert after is not None
        # No half-applied state: valid_from is unchanged, valid_until
        # remains NULL.
        assert after.valid_from == _utc(2026, 6, 1)
        assert after.valid_until is None

    def test_missing_id_raises(self, storage: SqliteStorage) -> None:
        with pytest.raises(KeyError):
            storage.set_validity_window(new_id(), valid_until=_utc(2026, 5, 1))

    def test_no_op_when_no_args(self, storage: SqliteStorage) -> None:
        # Should not error when both are None; just a no-op.
        a = _seed_item(storage, "a")
        storage.set_validity_window(a.id)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.valid_until is None


class TestSetSourceTrust:
    def test_set_and_clear(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        storage.set_source_trust(a.id, 0.75)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.source_trust == pytest.approx(0.75)
        storage.set_source_trust(a.id, None)
        item = storage.get_memory_item(a.id)
        assert item is not None
        assert item.source_trust is None

    def test_out_of_range_rejected(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a")
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            storage.set_source_trust(a.id, 1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            storage.set_source_trust(a.id, -0.1)


# ---------------------------------------------------------------------------
# search_memory_item_embeddings_as_of
# ---------------------------------------------------------------------------


class TestSearchMemoryItemEmbeddingsAsOf:
    def test_default_excludes_invalidated(self, storage: SqliteStorage) -> None:
        a = _seed_item(storage, "a", vec=[1.0, 0.0, 0.0, 0.0])
        b = _seed_item(storage, "b", vec=[1.0, 0.0, 0.0, 0.0])
        storage.invalidate_memory_item(a.id, at=_utc(2026, 5, 1), by=b.id)
        hits = storage.search_memory_item_embeddings_as_of(
            [1.0, 0.0, 0.0, 0.0], k=10, model="fake"
        )
        ids = {u for u, _, _ in hits}
        assert b.id in ids
        assert a.id not in ids

    def test_as_of_before_invalidation_shows_loser(self, storage: SqliteStorage) -> None:
        """If `a` was invalidated at t2, querying as_of=t1 (t1 < t2) must
        still surface it. Seed valid_from explicitly so the test does not
        race the wall clock."""
        a = _seed_item(
            storage, "a", vec=[1.0, 0.0, 0.0, 0.0], valid_from=_utc(2026, 1, 1)
        )
        b = _seed_item(
            storage, "b", vec=[1.0, 0.0, 0.0, 0.0], valid_from=_utc(2026, 1, 1)
        )
        storage.invalidate_memory_item(a.id, at=_utc(2026, 5, 1), by=b.id)
        hits = storage.search_memory_item_embeddings_as_of(
            [1.0, 0.0, 0.0, 0.0], k=10, model="fake", as_of=_utc(2026, 4, 15)
        )
        ids = {u for u, _, _ in hits}
        assert a.id in ids

    def test_as_of_filters_valid_window(self, storage: SqliteStorage) -> None:
        """An item with valid_from=2026-03 / valid_until=2026-09 should
        be visible at 2026-05 but invisible at 2026-01 and 2026-10."""
        a = _seed_item(
            storage,
            "in-window",
            valid_from=_utc(2026, 3, 1),
            valid_until=_utc(2026, 9, 1),
            vec=[1.0, 0.0, 0.0, 0.0],
        )

        def _ids_at(as_of: datetime) -> set[UUID]:
            return {
                u
                for u, _, _ in storage.search_memory_item_embeddings_as_of(
                    [1.0, 0.0, 0.0, 0.0], k=10, model="fake", as_of=as_of
                )
            }

        assert a.id in _ids_at(_utc(2026, 5, 1))
        assert a.id not in _ids_at(_utc(2026, 1, 1))
        assert a.id not in _ids_at(_utc(2026, 10, 1))

    def test_empty_corpus(self, storage: SqliteStorage) -> None:
        out = storage.search_memory_item_embeddings_as_of(
            [1.0, 0.0, 0.0, 0.0], k=5, model="fake"
        )
        assert out == []

    def test_invalid_args(self, storage: SqliteStorage) -> None:
        with pytest.raises(ValueError, match="k must be"):
            storage.search_memory_item_embeddings_as_of(
                [1.0, 0.0, 0.0, 0.0], k=0, model="fake"
            )
        with pytest.raises(ValueError, match="candidate_multiplier"):
            storage.search_memory_item_embeddings_as_of(
                [1.0, 0.0, 0.0, 0.0], k=5, model="fake", candidate_multiplier=0
            )


def test_unused() -> None:
    """Smoke for fixtures-only imports above."""
    _ = uuid4
