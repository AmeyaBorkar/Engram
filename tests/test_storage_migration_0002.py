"""Tests for migration 0002 (decay state).

Stage 4 adds weight + four decay-state columns to events and memory_items.
Existing rows must survive the upgrade (backfill `last_decayed_at` from the
row's existing timestamp). New rows must populate `last_decayed_at` on
insert.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from engram.schemas import Event, ItemKind, Level, MemoryItem
from engram.storage import SqliteStorage
from engram.storage.migrations import list_migrations


def _v1_only_db(path: Path) -> sqlite3.Connection:
    """Create a connection that only has the v1 schema applied.

    Useful for testing the 0002 migration in isolation against rows that
    pre-date the new columns.
    """
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    # Bootstrap schema_migrations + apply migration 0001 only.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version    INTEGER PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)"
        ")"
    )
    from importlib import resources

    pkg = resources.files("engram.storage.migrations")
    sql = (pkg / "0001_initial.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    return conn


class TestMigrationListing:
    def test_includes_0002(self) -> None:
        versions = [v for v, _ in list_migrations()]
        assert 1 in versions
        assert 2 in versions
        # Strictly increasing.
        assert versions == sorted(set(versions))


class TestFreshUpgrade:
    def test_columns_present_after_full_apply(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
            for col in (
                "weight",
                "reinforcement_count",
                "corroboration_count",
                "contradiction_count",
                "last_decayed_at",
                "cold_at",
            ):
                assert col in cols, f"events.{col} missing"

            cols = {row["name"] for row in conn.execute("PRAGMA table_info(memory_items)")}
            for col in (
                "reinforcement_count",
                "corroboration_count",
                "contradiction_count",
                "last_decayed_at",
                "cold_at",
            ):
                assert col in cols, f"memory_items.{col} missing"

    def test_indexes_created(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            indexes = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
            assert "idx_events_weight" in indexes
            assert "idx_events_cold_at" in indexes
            assert "idx_memory_items_cold_at" in indexes


class TestBackfillFromV1:
    def test_existing_event_row_gets_decay_columns_with_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "v1.db"
        # Create db with v1 schema only and insert a row.
        conn = _v1_only_db(path)
        try:
            event_id = uuid4()
            created = "2025-01-01T00:00:00+00:00"
            conn.execute(
                "INSERT INTO events (id, content, metadata, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_id.bytes, "old event", "{}", None, created),
            )
        finally:
            conn.close()

        # Now reopen via SqliteStorage which will run migration 0002.
        with SqliteStorage(path) as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT weight, reinforcement_count, corroboration_count, "
                "contradiction_count, last_decayed_at, cold_at, created_at "
                "FROM events WHERE id = ?",
                (event_id.bytes,),
            ).fetchone()

            assert row is not None
            assert row["weight"] == 1.0
            assert row["reinforcement_count"] == 0
            assert row["corroboration_count"] == 0
            assert row["contradiction_count"] == 0
            assert row["last_decayed_at"] == row["created_at"]
            assert row["cold_at"] is None

    def test_existing_memory_item_row_backfilled(self, tmp_path: Path) -> None:
        path = tmp_path / "v1.db"
        conn = _v1_only_db(path)
        try:
            item_id = uuid4()
            created = "2025-01-01T00:00:00+00:00"
            updated = "2025-01-02T00:00:00+00:00"
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, weight, cluster_id, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id.bytes, "summary", "old summary", 0.5, None, "{}", created, updated),
            )
        finally:
            conn.close()

        with SqliteStorage(path) as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT reinforcement_count, corroboration_count, contradiction_count, "
                "last_decayed_at, cold_at, updated_at FROM memory_items WHERE id = ?",
                (item_id.bytes,),
            ).fetchone()
            assert row is not None
            assert row["reinforcement_count"] == 0
            assert row["last_decayed_at"] == row["updated_at"]
            assert row["cold_at"] is None


class TestNewInsertsPopulateDecayColumns:
    def test_insert_event_sets_last_decayed_at_to_created_at(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            event = Event(content="hello")
            storage.insert_event(event)
            row = (
                storage._connect()
                .execute(
                    "SELECT weight, last_decayed_at, cold_at FROM events WHERE id = ?",
                    (event.id.bytes,),
                )
                .fetchone()
            )
            assert row["weight"] == 1.0
            assert row["last_decayed_at"] is not None
            assert row["cold_at"] is None

    def test_insert_memory_item_sets_last_decayed_at(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            item = MemoryItem(level=Level.SUMMARY, content="x")
            storage.insert_memory_item(item)
            row = (
                storage._connect()
                .execute(
                    "SELECT last_decayed_at, cold_at FROM memory_items WHERE id = ?",
                    (item.id.bytes,),
                )
                .fetchone()
            )
            assert row["last_decayed_at"] is not None
            assert row["cold_at"] is None


class TestSearchExcludesColdByDefault:
    def test_cold_event_hidden_unless_include_cold(self, tmp_path: Path) -> None:
        from engram.providers._fake import FakeEmbedder
        from engram.schemas import Embedding

        embedder = FakeEmbedder(dim=8)
        with SqliteStorage(tmp_path / "x.db") as storage:
            hot = Event(content="hot event")
            cold = Event(content="cold event")
            for ev in (hot, cold):
                storage.insert_event(ev)
                vec = tuple(embedder.embed([ev.content])[0])
                storage.insert_embedding(
                    Embedding(
                        item_id=ev.id,
                        item_kind=ItemKind.EVENT,
                        model=embedder.model,
                        dim=embedder.dim,
                        vector=vec,
                    )
                )

            # Manually mark the cold one cold via raw SQL (the engine API
            # lands in commit 4; this commit only verifies the filter).
            now = datetime.now(tz=timezone.utc).isoformat()
            storage._connect().execute(
                "UPDATE events SET cold_at = ? WHERE id = ?",
                (now, cold.id.bytes),
            )

            qvec = tuple(embedder.embed(["hot"])[0])
            hot_only = storage.search_event_embeddings(qvec, k=10, model=embedder.model)
            assert {row[0] for row in hot_only} == {hot.id}

            both = storage.search_event_embeddings(
                qvec, k=10, model=embedder.model, include_cold=True
            )
            assert {row[0] for row in both} == {hot.id, cold.id}
