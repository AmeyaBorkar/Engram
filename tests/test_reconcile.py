"""Stage 8 reconciler tests.

Exercises every policy in `Resolution`, the loser-invalidation
side-effect, and the `Memory.reconcile` / `Memory.list_conflicts`
public surface.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from engram import (
    Conflict,
    ConflictStatus,
    DecayState,
    ItemKind,
    Level,
    Memory,
    MemoryItem,
    Resolution,
    SqliteStorage,
    Storage,
    new_id,
)
from engram.providers._fake import FakeEmbedder
from engram.reconcile import Reconciler


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


def _seed_pair(
    storage: Storage,
    *,
    older_at: datetime,
    newer_at: datetime,
    older_trust: float | None = None,
    newer_trust: float | None = None,
    older_corroboration: int = 0,
    newer_corroboration: int = 0,
) -> tuple[MemoryItem, MemoryItem, Conflict]:
    older = MemoryItem(
        level=Level.SUMMARY,
        content="older",
        created_at=older_at,
        valid_from=older_at,
        source_trust=older_trust,
    )
    newer = MemoryItem(
        level=Level.SUMMARY,
        content="newer",
        created_at=newer_at,
        valid_from=newer_at,
        source_trust=newer_trust,
    )
    storage.insert_memory_item(older)
    storage.insert_memory_item(newer)
    if older_corroboration:
        _seed_corroboration(storage, older.id, older_corroboration, older_at)
    if newer_corroboration:
        _seed_corroboration(storage, newer.id, newer_corroboration, newer_at)
    conflict = Conflict(
        source_item_id=newer.id,
        target_item_id=older.id,
        similarity=0.95,
    )
    storage.record_conflict(conflict)
    return older, newer, conflict


def _seed_corroboration(storage: Storage, item_id: UUID, count: int, at: datetime) -> None:
    existing = storage.get_decay_state(item_id, ItemKind.MEMORY_ITEM)
    assert existing is not None, "Memory item should have a default decay state"
    storage.update_decay_state(
        DecayState(
            item_id=item_id,
            item_kind=ItemKind.MEMORY_ITEM,
            weight=existing.weight,
            reinforcement_count=existing.reinforcement_count,
            corroboration_count=count,
            contradiction_count=existing.contradiction_count,
            last_decayed_at=at,
            cold_at=existing.cold_at,
        )
    )


# ---------------------------------------------------------------------------
# Per-policy: PREFER_RECENT
# ---------------------------------------------------------------------------


class TestPreferRecent:
    def test_newer_wins(self, storage: SqliteStorage) -> None:
        older, newer, conflict = _seed_pair(
            storage, older_at=_utc(2026, 1, 1), newer_at=_utc(2026, 4, 1)
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_RECENT,
            now=_utc(2026, 5, 1),
        )
        assert out.status is ConflictStatus.RESOLVED
        assert out.resolution is Resolution.PREFER_RECENT
        assert out.resolved_winner_id == newer.id

        loser = storage.get_memory_item(older.id)
        assert loser is not None
        assert loser.invalidated_at == _utc(2026, 5, 1)
        assert loser.invalidated_by == newer.id

        winner = storage.get_memory_item(newer.id)
        assert winner is not None
        assert winner.invalidated_at is None


# ---------------------------------------------------------------------------
# Per-policy: PREFER_TRUSTED
# ---------------------------------------------------------------------------


class TestPreferTrusted:
    def test_higher_trust_wins(self, storage: SqliteStorage) -> None:
        older, _newer, conflict = _seed_pair(
            storage,
            older_at=_utc(2026, 1, 1),
            newer_at=_utc(2026, 4, 1),
            older_trust=0.9,
            newer_trust=0.3,
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_TRUSTED,
            now=_utc(2026, 5, 1),
        )
        assert out.resolved_winner_id == older.id

    def test_trust_tie_falls_back_to_recent(self, storage: SqliteStorage) -> None:
        _older, newer, conflict = _seed_pair(
            storage,
            older_at=_utc(2026, 1, 1),
            newer_at=_utc(2026, 4, 1),
            older_trust=0.5,
            newer_trust=0.5,
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_TRUSTED,
            now=_utc(2026, 5, 1),
        )
        assert out.resolved_winner_id == newer.id

    def test_none_trust_treated_as_zero(self, storage: SqliteStorage) -> None:
        older, _newer, conflict = _seed_pair(
            storage,
            older_at=_utc(2026, 1, 1),
            newer_at=_utc(2026, 4, 1),
            older_trust=0.5,
            newer_trust=None,
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_TRUSTED,
            now=_utc(2026, 5, 1),
        )
        assert out.resolved_winner_id == older.id


# ---------------------------------------------------------------------------
# Per-policy: PREFER_FREQUENT
# ---------------------------------------------------------------------------


class TestPreferFrequent:
    def test_higher_corroboration_wins(self, storage: SqliteStorage) -> None:
        older, _newer, conflict = _seed_pair(
            storage,
            older_at=_utc(2026, 1, 1),
            newer_at=_utc(2026, 4, 1),
            older_corroboration=5,
            newer_corroboration=1,
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_FREQUENT,
            now=_utc(2026, 5, 1),
        )
        assert out.resolved_winner_id == older.id

    def test_corroboration_tie_falls_back_to_recent(self, storage: SqliteStorage) -> None:
        _older, newer, conflict = _seed_pair(
            storage,
            older_at=_utc(2026, 1, 1),
            newer_at=_utc(2026, 4, 1),
            older_corroboration=3,
            newer_corroboration=3,
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_FREQUENT,
            now=_utc(2026, 5, 1),
        )
        assert out.resolved_winner_id == newer.id


# ---------------------------------------------------------------------------
# KEEP_BOTH
# ---------------------------------------------------------------------------


class TestKeepBoth:
    def test_no_winner_no_invalidation(self, storage: SqliteStorage) -> None:
        older, newer, conflict = _seed_pair(
            storage, older_at=_utc(2026, 1, 1), newer_at=_utc(2026, 4, 1)
        )
        out = Reconciler(storage).reconcile(
            conflict.id, resolution=Resolution.KEEP_BOTH, now=_utc(2026, 5, 1)
        )
        assert out.status is ConflictStatus.RESOLVED
        assert out.resolution is Resolution.KEEP_BOTH
        assert out.resolved_winner_id is None
        # Neither item is invalidated.
        for item_id in (older.id, newer.id):
            item = storage.get_memory_item(item_id)
            assert item is not None
            assert item.invalidated_at is None


# ---------------------------------------------------------------------------
# MANUAL
# ---------------------------------------------------------------------------


class TestManual:
    def test_with_valid_winner(self, storage: SqliteStorage) -> None:
        older, newer, conflict = _seed_pair(
            storage, older_at=_utc(2026, 1, 1), newer_at=_utc(2026, 4, 1)
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.MANUAL,
            manual_winner_id=older.id,
            now=_utc(2026, 5, 1),
        )
        assert out.resolved_winner_id == older.id
        loser = storage.get_memory_item(newer.id)
        assert loser is not None
        assert loser.invalidated_at == _utc(2026, 5, 1)

    def test_missing_winner_raises(self, storage: SqliteStorage) -> None:
        _older, _newer, conflict = _seed_pair(
            storage, older_at=_utc(2026, 1, 1), newer_at=_utc(2026, 4, 1)
        )
        with pytest.raises(ValueError, match="manual_winner_id"):
            Reconciler(storage).reconcile(conflict.id, resolution=Resolution.MANUAL)

    def test_winner_not_party_raises(self, storage: SqliteStorage) -> None:
        _older, _newer, conflict = _seed_pair(
            storage, older_at=_utc(2026, 1, 1), newer_at=_utc(2026, 4, 1)
        )
        third = new_id()
        with pytest.raises(ValueError, match="manual_winner_id"):
            Reconciler(storage).reconcile(
                conflict.id,
                resolution=Resolution.MANUAL,
                manual_winner_id=third,
            )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_conflict_raises(self, storage: SqliteStorage) -> None:
        with pytest.raises(KeyError):
            Reconciler(storage).reconcile(new_id(), resolution=Resolution.PREFER_RECENT)

    def test_already_resolved_raises(self, storage: SqliteStorage) -> None:
        _older, _newer, conflict = _seed_pair(
            storage, older_at=_utc(2026, 1, 1), newer_at=_utc(2026, 4, 1)
        )
        Reconciler(storage).reconcile(
            conflict.id, resolution=Resolution.PREFER_RECENT, now=_utc(2026, 5, 1)
        )
        with pytest.raises(RuntimeError, match="already resolved"):
            Reconciler(storage).reconcile(
                conflict.id,
                resolution=Resolution.PREFER_RECENT,
                now=_utc(2026, 5, 2),
            )


# ---------------------------------------------------------------------------
# Recency tie-break: same created_at -> deterministic id order
# ---------------------------------------------------------------------------


class TestRecencyTieBreak:
    def test_identical_timestamps_use_id_order(self, storage: SqliteStorage) -> None:
        same = _utc(2026, 3, 15)
        a = MemoryItem(level=Level.SUMMARY, content="a", created_at=same, valid_from=same)
        b = MemoryItem(level=Level.SUMMARY, content="b", created_at=same, valid_from=same)
        storage.insert_memory_item(a)
        storage.insert_memory_item(b)
        conflict = Conflict(source_item_id=a.id, target_item_id=b.id, similarity=0.9)
        storage.record_conflict(conflict)
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_RECENT,
            now=_utc(2026, 5, 1),
        )
        expected = a.id if a.id.bytes > b.id.bytes else b.id
        assert out.resolved_winner_id == expected


# ---------------------------------------------------------------------------
# Memory.reconcile public surface
# ---------------------------------------------------------------------------


class TestMemoryReconcile:
    def test_delegates_to_reconciler(self, memory: Memory) -> None:
        older, newer, conflict = _seed_pair(
            memory.storage,
            older_at=_utc(2026, 1, 1),
            newer_at=_utc(2026, 4, 1),
        )
        out = memory.reconcile(
            conflict.id,
            resolution=Resolution.PREFER_RECENT,
            now=_utc(2026, 5, 1),
        )
        assert out.resolved_winner_id == newer.id
        loser = memory.storage.get_memory_item(older.id)
        assert loser is not None
        assert loser.invalidated_at == _utc(2026, 5, 1)

    def test_list_conflicts(self, memory: Memory) -> None:
        a, b, ab = _seed_pair(
            memory.storage,
            older_at=_utc(2026, 1, 1),
            newer_at=_utc(2026, 4, 1),
        )
        all_conflicts = memory.list_conflicts()
        assert {c.id for c in all_conflicts} == {ab.id}
        open_only = memory.list_conflicts(status=ConflictStatus.OPEN)
        assert {c.id for c in open_only} == {ab.id}
        resolved_only = memory.list_conflicts(status=ConflictStatus.RESOLVED)
        assert resolved_only == []
        # Walking via memory_item_id from either side finds it.
        from_a = memory.list_conflicts(memory_item_id=a.id)
        assert {c.id for c in from_a} == {ab.id}
        from_b = memory.list_conflicts(memory_item_id=b.id)
        assert {c.id for c in from_b} == {ab.id}


# ---------------------------------------------------------------------------
# Integration: reconcile + temporal as_of
# ---------------------------------------------------------------------------


class TestReconcileWithAsOf:
    def test_loser_visible_before_invalidation(self, storage: SqliteStorage) -> None:
        from engram.schemas import Embedding

        embedder = FakeEmbedder(dim=8)
        # Older + newer items with the same embedding.
        same = (1.0,) + (0.0,) * 7
        older = MemoryItem(
            level=Level.SUMMARY,
            content="older",
            created_at=_utc(2026, 1, 1),
            valid_from=_utc(2026, 1, 1),
        )
        newer = MemoryItem(
            level=Level.SUMMARY,
            content="newer",
            created_at=_utc(2026, 4, 1),
            valid_from=_utc(2026, 4, 1),
        )
        for item in (older, newer):
            storage.insert_memory_item(item)
            storage.insert_embedding(
                Embedding(
                    item_id=item.id,
                    item_kind=ItemKind.MEMORY_ITEM,
                    model=embedder.model,
                    dim=8,
                    vector=same,
                )
            )

        conflict = Conflict(source_item_id=newer.id, target_item_id=older.id, similarity=1.0)
        storage.record_conflict(conflict)

        # Resolve at 2026-05-01 -- newer wins, older gets invalidated.
        Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_RECENT,
            now=_utc(2026, 5, 1),
        )

        # Default retrieve (no as_of) -- older should be gone.
        default_hits = storage.search_memory_item_embeddings_as_of(same, k=10, model=embedder.model)
        ids = {u for u, _, _ in default_hits}
        assert newer.id in ids
        assert older.id not in ids

        # as_of before invalidation -- older still visible.
        historical_hits = storage.search_memory_item_embeddings_as_of(
            same, k=10, model=embedder.model, as_of=_utc(2026, 4, 15)
        )
        hist_ids = {u for u, _, _ in historical_hits}
        assert older.id in hist_ids
        assert newer.id in hist_ids


# ---------------------------------------------------------------------------
# Wallclock smoke
# ---------------------------------------------------------------------------


def test_default_clock_used(storage: SqliteStorage) -> None:
    """Reconciler defaults to wallclock UTC when `now` is not passed."""
    _older, _newer, conflict = _seed_pair(
        storage, older_at=_utc(2026, 1, 1), newer_at=_utc(2026, 4, 1)
    )
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    out = Reconciler(storage).reconcile(conflict.id, resolution=Resolution.PREFER_RECENT)
    after = datetime.now(timezone.utc) + timedelta(seconds=1)
    assert out.resolved_at is not None
    assert before <= out.resolved_at <= after
