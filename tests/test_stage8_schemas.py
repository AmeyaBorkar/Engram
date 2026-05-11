"""Stage 8 schema tests.

Exercises the new public types -- `Source`, `Resolution`,
`ConflictStatus`, `Verdict`, `Conflict` -- and the temporal / invalidation
fields added to `MemoryItem`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from engram import (
    Conflict,
    ConflictStatus,
    Level,
    MemoryItem,
    Resolution,
    Source,
    Verdict,
    new_id,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class TestSource:
    def test_round_trip(self) -> None:
        s = Source(name="user@example.com", trust=0.9)
        assert s.name == "user@example.com"
        assert s.trust == 0.9

    def test_trust_bounds(self) -> None:
        Source(name="x", trust=0.0)
        Source(name="x", trust=1.0)
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Source(name="x", trust=-0.1)
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Source(name="x", trust=1.1)

    def test_frozen(self) -> None:
        s = Source(name="x", trust=0.5)
        with pytest.raises(ValueError, match=r"frozen|Instance is frozen|immutable"):
            s.trust = 0.6


# ---------------------------------------------------------------------------
# Resolution / ConflictStatus / Verdict enums
# ---------------------------------------------------------------------------


class TestResolutionEnum:
    def test_values(self) -> None:
        assert Resolution.PREFER_RECENT.value == "prefer_recent"
        assert Resolution.PREFER_TRUSTED.value == "prefer_trusted"
        assert Resolution.PREFER_FREQUENT.value == "prefer_frequent"
        assert Resolution.KEEP_BOTH.value == "keep_both"
        assert Resolution.MANUAL.value == "manual"

    def test_round_trip(self) -> None:
        assert Resolution("prefer_recent") is Resolution.PREFER_RECENT


class TestConflictStatusEnum:
    def test_values(self) -> None:
        assert ConflictStatus.OPEN.value == "open"
        assert ConflictStatus.RESOLVED.value == "resolved"


class TestVerdictEnum:
    def test_values(self) -> None:
        assert Verdict.AGREE.value == "agree"
        assert Verdict.CONTRADICT.value == "contradict"
        assert Verdict.UNRELATED.value == "unrelated"


# ---------------------------------------------------------------------------
# Conflict
# ---------------------------------------------------------------------------


class TestConflictDefaults:
    def test_defaults_to_open_contradict(self) -> None:
        a, b = new_id(), new_id()
        c = Conflict(source_item_id=a, target_item_id=b, similarity=0.85)
        assert c.status is ConflictStatus.OPEN
        assert c.verdict is Verdict.CONTRADICT
        assert c.resolution is None
        assert c.resolved_at is None
        assert c.resolved_winner_id is None
        assert c.id is not None
        assert isinstance(c.detected_at, datetime)


class TestConflictInvariants:
    def test_similarity_bounds(self) -> None:
        a, b = new_id(), new_id()
        Conflict(source_item_id=a, target_item_id=b, similarity=-1.0)
        Conflict(source_item_id=a, target_item_id=b, similarity=1.0)
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Conflict(source_item_id=a, target_item_id=b, similarity=1.5)
        with pytest.raises(ValueError, match="greater than or equal to -1"):
            Conflict(source_item_id=a, target_item_id=b, similarity=-1.5)

    def test_source_and_target_must_differ(self) -> None:
        a = new_id()
        with pytest.raises(ValueError, match="must differ"):
            Conflict(source_item_id=a, target_item_id=a, similarity=0.9)

    def test_resolved_requires_resolution_and_resolved_at(self) -> None:
        a, b = new_id(), new_id()
        with pytest.raises(ValueError, match="requires a resolution"):
            Conflict(
                source_item_id=a,
                target_item_id=b,
                similarity=0.9,
                status=ConflictStatus.RESOLVED,
            )
        with pytest.raises(ValueError, match="requires resolved_at"):
            Conflict(
                source_item_id=a,
                target_item_id=b,
                similarity=0.9,
                status=ConflictStatus.RESOLVED,
                resolution=Resolution.PREFER_RECENT,
                resolved_winner_id=a,
            )

    def test_resolved_non_keep_both_requires_winner(self) -> None:
        a, b = new_id(), new_id()
        with pytest.raises(ValueError, match="requires resolved_winner_id"):
            Conflict(
                source_item_id=a,
                target_item_id=b,
                similarity=0.9,
                status=ConflictStatus.RESOLVED,
                resolution=Resolution.PREFER_RECENT,
                resolved_at=_utc(2026, 5, 1),
            )

    def test_resolved_keep_both_no_winner(self) -> None:
        a, b = new_id(), new_id()
        c = Conflict(
            source_item_id=a,
            target_item_id=b,
            similarity=0.9,
            status=ConflictStatus.RESOLVED,
            resolution=Resolution.KEEP_BOTH,
            resolved_at=_utc(2026, 5, 1),
        )
        assert c.resolved_winner_id is None

    def test_resolved_winner_must_be_source_or_target(self) -> None:
        a, b, c_id = new_id(), new_id(), new_id()
        with pytest.raises(ValueError, match="resolved_winner_id"):
            Conflict(
                source_item_id=a,
                target_item_id=b,
                similarity=0.9,
                status=ConflictStatus.RESOLVED,
                resolution=Resolution.PREFER_RECENT,
                resolved_winner_id=c_id,
                resolved_at=_utc(2026, 5, 1),
            )

    def test_open_must_not_have_resolution_fields(self) -> None:
        a, b = new_id(), new_id()
        with pytest.raises(ValueError, match="open conflict"):
            Conflict(
                source_item_id=a,
                target_item_id=b,
                similarity=0.9,
                resolution=Resolution.PREFER_RECENT,
            )
        with pytest.raises(ValueError, match="open conflict"):
            Conflict(
                source_item_id=a,
                target_item_id=b,
                similarity=0.9,
                resolved_winner_id=a,
            )
        with pytest.raises(ValueError, match="open conflict"):
            Conflict(
                source_item_id=a,
                target_item_id=b,
                similarity=0.9,
                resolved_at=_utc(2026, 5, 1),
            )


# ---------------------------------------------------------------------------
# MemoryItem temporal extensions
# ---------------------------------------------------------------------------


class TestMemoryItemTemporal:
    def test_valid_from_defaults_to_created_at(self) -> None:
        created = _utc(2026, 1, 1)
        m = MemoryItem(level=Level.SUMMARY, content="x", created_at=created)
        assert m.valid_from == created

    def test_valid_from_explicit(self) -> None:
        m = MemoryItem(
            level=Level.SUMMARY,
            content="x",
            valid_from=_utc(2026, 1, 1),
            created_at=_utc(2026, 2, 1),
        )
        assert m.valid_from == _utc(2026, 1, 1)

    def test_valid_until_after_valid_from(self) -> None:
        m = MemoryItem(
            level=Level.SUMMARY,
            content="x",
            valid_from=_utc(2026, 1, 1),
            valid_until=_utc(2026, 3, 1),
        )
        assert m.valid_until == _utc(2026, 3, 1)

    def test_valid_until_before_valid_from_rejected(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            MemoryItem(
                level=Level.SUMMARY,
                content="x",
                valid_from=_utc(2026, 3, 1),
                valid_until=_utc(2026, 1, 1),
            )

    def test_invalidated_by_requires_invalidated_at(self) -> None:
        winner = new_id()
        with pytest.raises(ValueError, match="invalidated_by"):
            MemoryItem(
                level=Level.SUMMARY,
                content="x",
                invalidated_by=winner,
            )

    def test_invalidated_pair_round_trip(self) -> None:
        winner = new_id()
        when = _utc(2026, 5, 1)
        m = MemoryItem(
            level=Level.SUMMARY,
            content="x",
            invalidated_at=when,
            invalidated_by=winner,
        )
        assert m.invalidated_at == when
        assert m.invalidated_by == winner

    def test_invalidated_at_without_by_is_allowed(self) -> None:
        """E.g. a TTL-driven invalidation that has no replacement."""
        when = _utc(2026, 5, 1)
        m = MemoryItem(level=Level.SUMMARY, content="x", invalidated_at=when)
        assert m.invalidated_at == when
        assert m.invalidated_by is None

    def test_source_trust_bounds(self) -> None:
        MemoryItem(level=Level.SUMMARY, content="x", source_trust=0.0)
        MemoryItem(level=Level.SUMMARY, content="x", source_trust=1.0)
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            MemoryItem(level=Level.SUMMARY, content="x", source_trust=-0.1)
        with pytest.raises(ValueError, match="less than or equal to 1"):
            MemoryItem(level=Level.SUMMARY, content="x", source_trust=1.1)

    def test_legacy_construction_still_works(self) -> None:
        """Pre-Stage-8 callers omit the new fields; verify back-compat."""
        m = MemoryItem(level=Level.SUMMARY, content="x")
        assert m.valid_from == m.created_at
        assert m.valid_until is None
        assert m.invalidated_at is None
        assert m.invalidated_by is None
        assert m.source_trust is None


def test_uuid_typing_on_conflict() -> None:
    """`source_item_id` etc are typed as UUID; pydantic should coerce strings too."""
    a, b = new_id(), new_id()
    c = Conflict(source_item_id=a, target_item_id=b, similarity=0.5)
    assert isinstance(c.source_item_id, UUID)
    assert isinstance(c.target_item_id, UUID)


def test_detected_at_within_test_window() -> None:
    a, b = new_id(), new_id()
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    c = Conflict(source_item_id=a, target_item_id=b, similarity=0.5)
    after = datetime.now(timezone.utc) + timedelta(seconds=1)
    assert before <= c.detected_at <= after
