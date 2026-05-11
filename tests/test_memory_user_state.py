"""Tests for the aggregate user-state (Level.GLOBAL) layer."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from uuid import uuid4

import pytest

from engram import Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder
from engram.schemas import Level
from engram.storage.migrations import list_migrations

# ---------------------------------------------------------------------------
# Migration 0008
# ---------------------------------------------------------------------------


class TestMigration0008Listing:
    def test_includes_0008(self) -> None:
        versions = [v for v, _ in list_migrations()]
        assert {1, 2, 3, 4, 5, 6, 7, 8} <= set(versions)


class TestMigration0008Fresh:
    def test_topic_and_global_levels_accepted(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            for level in ("topic", "global"):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, ?, 'x', '2026-01-01', '2026-01-01', "
                    " '2026-01-01', '2026-01-01')",
                    (uuid4().bytes, level),
                )

    def test_pre_0008_levels_still_accepted(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            for level in ("event", "summary", "abstraction", "preference"):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, ?, 'x', '2026-01-01', '2026-01-01', "
                    " '2026-01-01', '2026-01-01')",
                    (uuid4().bytes, level),
                )


def _pre_0008_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY,"
        " applied_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)"
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
        "0007_preference_level.sql",
    ):
        conn.executescript((pkg / name).read_text(encoding="utf-8"))
    return conn


class TestMigration0008Upgrade:
    def test_existing_preference_rows_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "v7.db"
        conn = _pre_0008_db(path)
        try:
            mid = uuid4().bytes
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, created_at, updated_at, "
                " last_decayed_at, valid_from) "
                "VALUES (?, 'preference', 'I love X', '2026-01-01', "
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
            assert row["level"] == "preference"
            assert row["content"] == "I love X"


# ---------------------------------------------------------------------------
# User-state methods
# ---------------------------------------------------------------------------


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


class TestSetUserStateFresh:
    def test_first_call_creates_global_item(self, memory: Memory) -> None:
        item = memory.set_user_state("the user is a Python dev based in NYC")
        assert item.level is Level.GLOBAL
        fetched = memory.get_user_state()
        assert fetched is not None
        assert fetched.id == item.id

    def test_metadata_flag_present(self, memory: Memory) -> None:
        item = memory.set_user_state("x")
        assert item.metadata.get("engram_user_state") is True

    def test_provenance_seeded(self, memory: Memory) -> None:
        item = memory.set_user_state("x")
        events = memory.storage.get_supporting_events(item.id)
        assert len(events) == 1

    def test_explicit_supporting_events(self, memory: Memory) -> None:
        event = memory.observe("user is a dev")
        item = memory.set_user_state("the user is a dev", supporting_event_ids=[event.id])
        supporting = memory.storage.get_supporting_events(item.id)
        assert [e.id for e in supporting] == [event.id]


class TestSetUserStateUpdate:
    def test_second_call_replaces_first(self, memory: Memory) -> None:
        first = memory.set_user_state("the user is X")
        second = memory.set_user_state("the user is Y")
        assert second.id != first.id  # new item
        # Only one global item with the flag remains.
        current = memory.get_user_state()
        assert current is not None
        assert current.id == second.id
        assert current.content == "the user is Y"
        # Sanity: first item no longer findable.
        assert memory.storage.get_memory_item(first.id) is None

    def test_existing_provenance_preserved_on_update(self, memory: Memory) -> None:
        original_event = memory.observe("seed")
        memory.set_user_state("v1", supporting_event_ids=[original_event.id])
        v2 = memory.set_user_state("v2")  # no explicit ids
        events = memory.storage.get_supporting_events(v2.id)
        # The original event id is reused.
        assert any(e.id == original_event.id for e in events)


class TestUserStateTenantScoping:
    def test_get_user_state_filters_by_tenant(
        self, storage: SqliteStorage
    ) -> None:
        # Two tenants on the same storage; each gets its own user-state.
        m_a = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            tenant_id="a",
        )
        m_b = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            tenant_id="b",
        )
        item_a = m_a.set_user_state("a-state")
        item_b = m_b.set_user_state("b-state")
        # Each tenant sees only their own.
        sa = m_a.get_user_state()
        sb = m_b.get_user_state()
        assert sa is not None
        assert sb is not None
        assert sa.id == item_a.id
        assert sb.id == item_b.id

    def test_untenanted_memory_sees_only_untenanted_state(
        self, storage: SqliteStorage
    ) -> None:
        m_untagged = Memory(storage=storage, embedder=FakeEmbedder(dim=8))
        m_tagged = Memory(
            storage=storage, embedder=FakeEmbedder(dim=8), tenant_id="x"
        )
        untagged = m_untagged.set_user_state("u")
        tagged = m_tagged.set_user_state("t")
        u = m_untagged.get_user_state()
        t = m_tagged.get_user_state()
        assert u is not None
        assert t is not None
        assert u.id == untagged.id
        assert t.id == tagged.id
