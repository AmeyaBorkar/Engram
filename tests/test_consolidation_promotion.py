"""Tests for promotion (summary -> abstraction)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from engram import Memory, SqliteStorage
from engram.consolidation import (
    ConsolidationParams,
    PromotionParams,
    PromotionResult,
)
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.schemas import ItemKind, Level, MemoryItem


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make(tmp_path: Path, **kwargs: object) -> tuple[Memory, SqliteStorage]:
    storage = SqliteStorage(tmp_path / "x.db")
    storage.initialize()
    memory = Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=8),
        chat=FakeChat(default="{}"),
        consolidation_params=ConsolidationParams(**kwargs),  # type: ignore[arg-type]
    )
    return memory, storage


def _seed_summary(
    storage: SqliteStorage,
    *,
    content: str,
    weight: float = 0.6,
    metadata: dict | None = None,
) -> MemoryItem:
    item = MemoryItem(
        level=Level.SUMMARY,
        content=content,
        weight=weight,
        metadata=metadata or {"consolidation": {"conflicts": []}},
    )
    storage.insert_memory_item(item)
    return item


# ---------------------------------------------------------------------------
# PromotionParams
# ---------------------------------------------------------------------------


class TestPromotionParams:
    def test_defaults_disabled(self) -> None:
        p = PromotionParams()
        assert p.enabled is False
        assert p.min_corroboration == 3
        assert p.max_contradiction == 0

    def test_min_corroboration_bound(self) -> None:
        with pytest.raises(ValueError, match="min_corroboration"):
            PromotionParams(min_corroboration=0)

    def test_max_contradiction_non_negative(self) -> None:
        with pytest.raises(ValueError, match="max_contradiction"):
            PromotionParams(max_contradiction=-1)

    def test_min_weight_bounds(self) -> None:
        with pytest.raises(ValueError, match="min_weight"):
            PromotionParams(min_weight=-0.1)
        with pytest.raises(ValueError, match="min_weight"):
            PromotionParams(min_weight=1.1)


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------


class TestDisabledDefault:
    def test_default_promote_does_nothing(self, tmp_path: Path) -> None:
        memory, storage = _make(tmp_path)
        try:
            _seed_summary(storage, content="x")
            result = memory.promote()
            assert isinstance(result, PromotionResult)
            assert result.candidates_examined == 0
            assert result.promoted == 0
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# Promotes when criteria met
# ---------------------------------------------------------------------------


class TestPromotesWhenEligible:
    def test_corroborated_summary_promotes(self, tmp_path: Path) -> None:
        memory, storage = _make(
            tmp_path,
            promotion_params=PromotionParams(enabled=True, min_corroboration=2, min_weight=0.0),
        )
        try:
            item = _seed_summary(storage, content="stable pattern", weight=1.0)
            # Three corroboration signals.
            for _ in range(3):
                memory.corroborate(item.id, ItemKind.MEMORY_ITEM)
            result = memory.promote()
            assert result.candidates_examined == 1
            assert result.promoted == 1
            after = storage.get_memory_item(item.id)
            assert after is not None
            assert after.level is Level.ABSTRACTION
        finally:
            storage.close()

    def test_below_threshold_does_not_promote(self, tmp_path: Path) -> None:
        memory, storage = _make(
            tmp_path,
            promotion_params=PromotionParams(enabled=True, min_corroboration=5, min_weight=0.0),
        )
        try:
            item = _seed_summary(storage, content="undercooked", weight=1.0)
            for _ in range(2):
                memory.corroborate(item.id, ItemKind.MEMORY_ITEM)
            result = memory.promote()
            assert result.promoted == 0
            after = storage.get_memory_item(item.id)
            assert after is not None
            assert after.level is Level.SUMMARY
        finally:
            storage.close()

    def test_contradiction_blocks_promotion(self, tmp_path: Path) -> None:
        memory, storage = _make(
            tmp_path,
            promotion_params=PromotionParams(enabled=True, min_corroboration=2, min_weight=0.0),
        )
        try:
            item = _seed_summary(storage, content="contested", weight=1.0)
            for _ in range(3):
                memory.corroborate(item.id, ItemKind.MEMORY_ITEM)
            memory.contradict(item.id, ItemKind.MEMORY_ITEM)
            result = memory.promote()
            assert result.promoted == 0
        finally:
            storage.close()

    def test_low_weight_blocks_promotion(self, tmp_path: Path) -> None:
        memory, storage = _make(
            tmp_path,
            promotion_params=PromotionParams(enabled=True, min_corroboration=2, min_weight=0.5),
        )
        try:
            item = _seed_summary(storage, content="weak", weight=0.3)
            for _ in range(3):
                memory.corroborate(item.id, ItemKind.MEMORY_ITEM)
            result = memory.promote()
            assert result.promoted == 0
        finally:
            storage.close()

    def test_open_conflicts_block_promotion(self, tmp_path: Path) -> None:
        """Audit H-57: an OPEN row in the conflicts table — not a
        metadata snapshot — is what blocks promotion. A resolved
        conflict no longer blocks (covered by the sibling test).
        """
        from engram.schemas import Conflict

        memory, storage = _make(
            tmp_path,
            promotion_params=PromotionParams(
                enabled=True, min_corroboration=2, min_weight=0.0
            ),
        )
        try:
            item = _seed_summary(storage, content="conflicted", weight=1.0)
            sibling = _seed_summary(storage, content="other", weight=1.0)
            # Plant an OPEN conflict naming the candidate.
            storage.record_conflict(
                Conflict(
                    source_item_id=item.id,
                    target_item_id=sibling.id,
                    similarity=0.95,
                )
            )
            for _ in range(3):
                memory.corroborate(item.id, ItemKind.MEMORY_ITEM)
            result = memory.promote()
            assert result.promoted == 0
            assert result.candidates_examined == 2
            after = storage.get_memory_item(item.id)
            assert after is not None
            assert after.level is Level.SUMMARY
        finally:
            storage.close()

    def test_resolved_conflicts_do_not_block_promotion(
        self, tmp_path: Path
    ) -> None:
        """Audit H-57: once the reconciler flips OPEN -> RESOLVED, the
        item is eligible for promotion again. The old metadata-snapshot
        gate never cleared so a once-contradicted summary could never
        be promoted; the new gate consults the persistent conflict
        table's status.
        """
        from engram.schemas import Conflict, Resolution

        memory, storage = _make(
            tmp_path,
            promotion_params=PromotionParams(
                enabled=True, min_corroboration=2, min_weight=0.0
            ),
        )
        try:
            item = _seed_summary(storage, content="reconciled", weight=1.0)
            sibling = _seed_summary(storage, content="other", weight=1.0)
            conflict = Conflict(
                source_item_id=item.id,
                target_item_id=sibling.id,
                similarity=0.95,
            )
            storage.record_conflict(conflict)
            # Resolve via KEEP_BOTH so the conflict row flips to RESOLVED.
            storage.resolve_conflict(
                conflict.id,
                resolution=Resolution.KEEP_BOTH,
                resolved_winner_id=None,
                resolved_at=_now(),
            )
            for _ in range(3):
                memory.corroborate(item.id, ItemKind.MEMORY_ITEM)
            result = memory.promote()
            # Both summaries clear the bar (no open conflicts after
            # the resolve).  The corroborated one promotes.
            assert result.promoted == 1
            after = storage.get_memory_item(item.id)
            assert after is not None
            assert after.level is Level.ABSTRACTION
        finally:
            storage.close()

    def test_already_abstraction_skipped(self, tmp_path: Path) -> None:
        # Items at level=abstraction are not candidates.
        memory, storage = _make(
            tmp_path,
            promotion_params=PromotionParams(enabled=True, min_corroboration=1, min_weight=0.0),
        )
        try:
            ab = MemoryItem(level=Level.ABSTRACTION, content="already there", weight=1.0)
            storage.insert_memory_item(ab)
            for _ in range(3):
                memory.corroborate(ab.id, ItemKind.MEMORY_ITEM)
            result = memory.promote()
            assert result.candidates_examined == 0
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# Storage seam: update_memory_item_level + iter_memory_items
# ---------------------------------------------------------------------------


class TestStorageSeams:
    def test_update_level_round_trip(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = MemoryItem(level=Level.SUMMARY, content="x")
            storage.insert_memory_item(item)
            storage.update_memory_item_level(item.id, Level.ABSTRACTION)
            after = storage.get_memory_item(item.id)
            assert after is not None
            assert after.level is Level.ABSTRACTION

    def test_update_level_unknown_raises(self, tmp_path: Path) -> None:
        from uuid import uuid4

        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(KeyError):
                storage.update_memory_item_level(uuid4(), Level.ABSTRACTION)

    def test_iter_memory_items_streams_in_order(self, tmp_path: Path) -> None:
        from datetime import timedelta

        with SqliteStorage(tmp_path / "x.db") as storage:
            base = _now()
            ids = []
            for i in range(5):
                item = MemoryItem(
                    level=Level.SUMMARY,
                    content=f"i{i}",
                    created_at=base + timedelta(seconds=i),
                )
                storage.insert_memory_item(item)
                ids.append(item.id)
            seen = [item.id for item in storage.iter_memory_items(level=Level.SUMMARY)]
            assert seen == ids

    def test_iter_memory_items_skips_cold_by_default(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            hot = MemoryItem(level=Level.SUMMARY, content="hot")
            cold = MemoryItem(level=Level.SUMMARY, content="cold")
            storage.insert_memory_item(hot)
            storage.insert_memory_item(cold)
            storage.mark_cold(cold.id, ItemKind.MEMORY_ITEM, at=_now())
            seen = [item.id for item in storage.iter_memory_items(level=Level.SUMMARY)]
            assert seen == [hot.id]
            with_cold = [
                item.id
                for item in storage.iter_memory_items(level=Level.SUMMARY, include_cold=True)
            ]
            assert set(with_cold) == {hot.id, cold.id}

    def test_iter_memory_items_validates_batch_size(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            with pytest.raises(ValueError, match="batch_size"):
                list(storage.iter_memory_items(batch_size=0))
