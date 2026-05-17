"""Tests for migration 0004 (temporal validity + conflicts table).

Stage 8 promotes contradiction detection into a first-class storage
entity and adds temporal validity / invalidation columns to
`memory_items`. The migration also exercises a `valid_from` backfill on
existing rows: rows from a v3 database must end up with
`valid_from = created_at` after upgrade.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from uuid import uuid4

import pytest

from engram.storage import SqliteStorage
from engram.storage.migrations import list_migrations


def _pre_0004_db(path: Path) -> sqlite3.Connection:
    """Apply migrations 0001..0003 only -- the state before 0004."""
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version    INTEGER PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)"
        ")"
    )
    pkg = resources.files("engram.storage.migrations")
    for name in ("0001_initial.sql", "0002_decay.sql", "0003_procedures.sql"):
        sql = (pkg / name).read_text(encoding="utf-8")
        conn.executescript(sql)
    return conn


class TestMigrationListing:
    def test_includes_0004(self) -> None:
        versions = [v for v, _ in list_migrations()]
        assert {1, 2, 3, 4} <= set(versions)
        assert versions == sorted(set(versions))


class TestFreshUpgrade:
    def test_conflicts_table_present(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conflicts'"
            ).fetchone()
            assert row is not None, "conflicts table missing after migration 0004"

    def test_conflicts_columns(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(conflicts)")}
            expected = {
                "id",
                "source_item_id",
                "target_item_id",
                "similarity",
                "verdict",
                "status",
                "resolution",
                "resolved_winner_id",
                "resolved_at",
                "detected_at",
            }
            assert expected <= cols, f"missing columns: {expected - cols}"

    def test_conflicts_indexes(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            indexes = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
            assert "idx_conflicts_status" in indexes
            assert "idx_conflicts_source_item" in indexes
            assert "idx_conflicts_target_item" in indexes
            assert "idx_conflicts_resolved_winner" in indexes

    def test_memory_items_new_columns(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(memory_items)")}
            expected = {
                "valid_from",
                "valid_until",
                "invalidated_at",
                "invalidated_by",
                "source_trust",
            }
            assert expected <= cols, f"missing columns: {expected - cols}"

    def test_memory_items_new_indexes(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            indexes = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
            assert "idx_memory_items_valid_until" in indexes
            assert "idx_memory_items_invalidated_at" in indexes
            assert "idx_memory_items_source_trust" in indexes

    def test_conflicts_status_check(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            # Seed two memory items so the FK targets exist.
            a, b = uuid4().bytes, uuid4().bytes
            for mid in (a, b):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, 'summary', 'x', '2026-01-01T00:00:00+00:00', "
                    "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', "
                    "'2026-01-01T00:00:00+00:00')",
                    (mid,),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO conflicts "
                    "(id, source_item_id, target_item_id, similarity, "
                    " status, detected_at) "
                    "VALUES (?, ?, ?, 0.9, 'bogus', '2026-05-01T00:00:00+00:00')",
                    (uuid4().bytes, a, b),
                )

    def test_conflicts_resolution_check(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            a, b = uuid4().bytes, uuid4().bytes
            for mid in (a, b):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, 'summary', 'x', '2026-01-01T00:00:00+00:00', "
                    "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', "
                    "'2026-01-01T00:00:00+00:00')",
                    (mid,),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO conflicts "
                    "(id, source_item_id, target_item_id, similarity, "
                    " status, resolution, resolved_at, detected_at) "
                    "VALUES (?, ?, ?, 0.9, 'resolved', 'invalid_policy', "
                    " '2026-05-01T00:00:00+00:00', '2026-05-01T00:00:00+00:00')",
                    (uuid4().bytes, a, b),
                )

    def test_conflicts_source_target_distinct(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            a = uuid4().bytes
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, created_at, updated_at, "
                " last_decayed_at, valid_from) "
                "VALUES (?, 'summary', 'x', '2026-01-01T00:00:00+00:00', "
                "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', "
                "'2026-01-01T00:00:00+00:00')",
                (a,),
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO conflicts "
                    "(id, source_item_id, target_item_id, similarity, detected_at) "
                    "VALUES (?, ?, ?, 0.9, '2026-05-01T00:00:00+00:00')",
                    (uuid4().bytes, a, a),
                )

    def test_source_trust_bounds(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            mid = uuid4().bytes
            # In-bounds (and NULL) values accepted.
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, created_at, updated_at, last_decayed_at, "
                " valid_from, source_trust) "
                "VALUES (?, 'summary', 'x', '', '', '', '', 0.5)",
                (mid,),
            )
            # Out-of-bounds rejected.
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, last_decayed_at, "
                    " valid_from, source_trust) "
                    "VALUES (?, 'summary', 'x', '', '', '', '', 1.5)",
                    (uuid4().bytes,),
                )


class TestUpgradeFromV3:
    def test_existing_memory_items_get_valid_from_backfilled(self, tmp_path: Path) -> None:
        """Migration 0004 backfills valid_from = created_at for rows that
        existed before the upgrade."""
        path = tmp_path / "v3.db"
        conn = _pre_0004_db(path)
        try:
            mid = uuid4().bytes
            created = "2025-06-01T12:00:00+00:00"
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, created_at, updated_at, last_decayed_at) "
                "VALUES (?, 'summary', 'pre-existing', ?, ?, ?)",
                (mid, created, created, created),
            )
        finally:
            conn.close()

        with SqliteStorage(path) as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT valid_from, valid_until, invalidated_at, "
                "invalidated_by, source_trust FROM memory_items WHERE id = ?",
                (mid,),
            ).fetchone()
            assert row is not None
            assert row["valid_from"] == created
            assert row["valid_until"] is None
            assert row["invalidated_at"] is None
            assert row["invalidated_by"] is None
            assert row["source_trust"] is None

    def test_existing_data_round_trips_through_python_layer(self, tmp_path: Path) -> None:
        """A pre-0004 row read via SqliteStorage maps to a coherent
        MemoryItem (valid_from defaulted to created_at, others None)."""
        path = tmp_path / "v3.db"
        conn = _pre_0004_db(path)
        try:
            mid = uuid4().bytes
            created = "2025-06-01T12:00:00+00:00"
            conn.execute(
                "INSERT INTO memory_items "
                "(id, level, content, created_at, updated_at, last_decayed_at) "
                "VALUES (?, 'summary', 'pre-existing', ?, ?, ?)",
                (mid, created, created, created),
            )
        finally:
            conn.close()

        with SqliteStorage(path) as storage:
            from uuid import UUID

            item = storage.get_memory_item(UUID(bytes=mid))
            assert item is not None
            assert item.valid_from == item.created_at
            assert item.valid_until is None
            assert item.invalidated_at is None
            assert item.invalidated_by is None
            assert item.source_trust is None
