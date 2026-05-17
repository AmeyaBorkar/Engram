"""Preference layer tests (Stage 9 / E.6).

Covers:
  * `is_preference(text)` heuristic catches common preference patterns
    and doesn't false-positive on plain facts.
  * Migration 0007 widens `memory_items.level` CHECK to accept
    'preference'; pre-7 rows round-trip cleanly.
  * `Memory.record_preference` creates Event + Level.PREFERENCE
    item with provenance.
  * `Memory.retrieve_preferences` returns only preference-level items.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from uuid import uuid4

import pytest

from engram import Memory, SqliteStorage
from engram._preference import is_preference
from engram.providers._fake import FakeEmbedder
from engram.schemas import Level
from engram.storage.migrations import list_migrations

# ---------------------------------------------------------------------------
# Heuristic detector
# ---------------------------------------------------------------------------


class TestIsPreference:
    @pytest.mark.parametrize(
        "text",
        [
            "I love pineapple on pizza.",
            "I prefer Python over Java.",
            "I really like dogs.",
            "I hate Mondays.",
            "I dislike crowded places.",
            "I can't stand slow elevators.",
            "I always take the stairs.",
            "I never skip leg day.",
            "I usually drink coffee in the morning.",
            "My favorite color is blue.",
            "I'd rather go hiking than to a club.",
            "I'm a huge fan of jazz.",
            "I'm a big fan of opera.",
            "I would prefer not to.",
            "Not a fan of horror movies.",
        ],
    )
    def test_catches_preference_text(self, text: str) -> None:
        assert is_preference(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "The user logged in at 9am.",
            "It is raining today.",
            "The deploy completed successfully.",
            "Database query returned 42 rows.",
            "Pi equals approximately 3.14159.",
            "",
            "   ",
        ],
    )
    def test_does_not_false_positive_on_facts(self, text: str) -> None:
        assert not is_preference(text), text


# ---------------------------------------------------------------------------
# Migration 0007
# ---------------------------------------------------------------------------


class TestMigration0007Listing:
    def test_includes_0007(self) -> None:
        versions = [v for v, _ in list_migrations()]
        assert {1, 2, 3, 4, 5, 6, 7} <= set(versions)


def _pre_0007_db(path: Path) -> sqlite3.Connection:
    """Apply 0001..0006 only."""
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version INTEGER PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)"
        ")"
    )
    pkg = resources.files("engram.storage.migrations")
    for name in (
        "0001_initial.sql",
        "0002_decay.sql",
        "0003_procedures.sql",
        "0004_temporal_conflicts.sql",
        "0005_resolution_merge.sql",
        "0006_multi_tenant.sql",
    ):
        conn.executescript((pkg / name).read_text(encoding="utf-8"))
    return conn


class TestMigration0007Fresh:
    def test_preference_level_accepted(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            mid = uuid4().bytes
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, created_at, updated_at, "
                " last_decayed_at, valid_from) "
                "VALUES (?, 'preference', 'I love tea', "
                "'2026-01-01', '2026-01-01', '2026-01-01', '2026-01-01')",
                (mid,),
            )

    def test_existing_levels_still_accepted(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            for level in ("event", "summary", "abstraction"):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, ?, 'x', '2026-01-01', '2026-01-01', "
                    " '2026-01-01', '2026-01-01')",
                    (uuid4().bytes, level),
                )

    def test_bogus_level_still_rejected(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, 'bogus', 'x', '2026-01-01', "
                    "'2026-01-01', '2026-01-01', '2026-01-01')",
                    (uuid4().bytes,),
                )


class TestMigration0007Upgrade:
    def test_pre_0007_rows_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "v6.db"
        conn = _pre_0007_db(path)
        try:
            mid = uuid4().bytes
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, created_at, updated_at, "
                " last_decayed_at, valid_from) "
                "VALUES (?, 'summary', 'pre-7 row', '2026-01-01', "
                "'2026-01-01', '2026-01-01', '2026-01-01')",
                (mid,),
            )
        finally:
            conn.close()
        with SqliteStorage(path) as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT level, content FROM memory_items WHERE id = ?",
                (mid,),
            ).fetchone()
            assert row is not None
            assert row["level"] == "summary"
            assert row["content"] == "pre-7 row"


# ---------------------------------------------------------------------------
# Memory.record_preference / retrieve_preferences
# ---------------------------------------------------------------------------


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


class TestRecordPreference:
    def test_creates_event_and_preference_item(self, memory: Memory) -> None:
        event, pref = memory.record_preference("I love pineapple pizza.")
        # Event landed.
        assert memory.storage.get_event(event.id) is not None
        # Preference item landed at Level.PREFERENCE.
        fetched = memory.storage.get_memory_item(pref.id)
        assert fetched is not None
        assert fetched.level is Level.PREFERENCE
        # Provenance links the preference item to the event.
        supporting = memory.storage.get_supporting_events(pref.id)
        assert len(supporting) == 1
        assert supporting[0].id == event.id

    def test_metadata_records_source_event(self, memory: Memory) -> None:
        event, pref = memory.record_preference("My favorite is X")
        fetched = memory.storage.get_memory_item(pref.id)
        assert fetched is not None
        assert fetched.metadata["preference"]["source_event_id"] == str(event.id)

    def test_tenant_id_propagates(self, storage: SqliteStorage) -> None:
        memory = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            tenant_id="acme",
        )
        event, pref = memory.record_preference("I love TLDs.")
        assert event.tenant_id == "acme"
        assert pref.tenant_id == "acme"


class TestRetrievePreferences:
    def test_returns_only_preference_level(self, memory: Memory) -> None:
        # Mix of regular observe + record_preference.
        memory.observe("I love this pizza.")
        memory.record_preference("I love this pizza.")
        results = memory.retrieve_preferences("love pizza", k=5, reinforce=False)
        # Every result is Level.PREFERENCE; the raw event doesn't surface.
        assert results
        for r in results:
            assert r.level is Level.PREFERENCE

    def test_empty_preference_layer(self, memory: Memory) -> None:
        memory.observe("just an event, no preference")
        results = memory.retrieve_preferences("anything", k=5)
        assert results == []

    def test_invalid_k_raises(self, memory: Memory) -> None:
        with pytest.raises(ValueError, match="k must be"):
            memory.retrieve_preferences("x", k=0)
