"""Tests for migration 0003 (procedures).

Stage 7 introduces the `procedures` table and extends the `embeddings`
CHECK constraint to allow `item_kind='procedure'`. The migration also
exercises SQLite's table-rebuild pattern for the embeddings CHECK
change, so existing embedding rows must survive the upgrade.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from engram.storage import SqliteStorage
from engram.storage.migrations import list_migrations


def _pre_0003_db(path: Path) -> sqlite3.Connection:
    """Apply migrations 0001 and 0002 only -- the state before 0003."""
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version    INTEGER PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)"
        ")"
    )
    from importlib import resources

    pkg = resources.files("engram.storage.migrations")
    for name in ("0001_initial.sql", "0002_decay.sql"):
        sql = (pkg / name).read_text(encoding="utf-8")
        conn.executescript(sql)
    return conn


class TestMigrationListing:
    def test_includes_0003(self) -> None:
        versions = [v for v, _ in list_migrations()]
        assert {1, 2, 3} <= set(versions)
        assert versions == sorted(set(versions))


class TestFreshUpgrade:
    def test_procedures_table_present(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='procedures'"
            ).fetchone()
            assert row is not None, "procedures table missing after migration 0003"

    def test_procedures_columns(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(procedures)")}
            expected = {
                "id",
                "situation",
                "action",
                "outcome",
                "weight",
                "reinforcement_count",
                "corroboration_count",
                "contradiction_count",
                "last_decayed_at",
                "cold_at",
                "metadata",
                "created_at",
                "updated_at",
            }
            assert expected <= cols, f"missing columns: {expected - cols}"

    def test_procedures_indexes(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            indexes = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
            assert "idx_procedures_created_at" in indexes
            assert "idx_procedures_weight" in indexes
            assert "idx_procedures_outcome" in indexes
            assert "idx_procedures_cold_at" in indexes

    def test_outcome_check_constraint(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            # Valid outcomes accepted.
            for outcome in ("success", "partial", "failure", "unknown"):
                pid = uuid4().bytes
                conn.execute(
                    "INSERT INTO procedures "
                    "(id, situation, action, outcome, last_decayed_at, "
                    " created_at, updated_at) VALUES (?, 's', 'a', ?, '', '', '')",
                    (pid, outcome),
                )
            # Invalid outcome rejected.
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO procedures "
                    "(id, situation, action, outcome, last_decayed_at, "
                    " created_at, updated_at) "
                    "VALUES (?, 's', 'a', 'wrong', '', '', '')",
                    (uuid4().bytes,),
                )

    def test_embeddings_check_allows_procedure(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            pid = uuid4().bytes
            # Insert a procedure first so the embeddings row has a target.
            conn.execute(
                "INSERT INTO procedures "
                "(id, situation, action, last_decayed_at, created_at, updated_at) "
                "VALUES (?, 's', 'a', '', '', '')",
                (pid,),
            )
            # Embedding insert should succeed with item_kind='procedure'.
            conn.execute(
                "INSERT INTO embeddings "
                "(id, item_id, item_kind, model, dim, vector, created_at) "
                "VALUES (?, ?, 'procedure', 'fake', 4, ?, '')",
                (uuid4().bytes, pid, b"\x00" * 16),
            )

    def test_embeddings_check_still_rejects_unknown_kind(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO embeddings "
                    "(id, item_id, item_kind, model, dim, vector, created_at) "
                    "VALUES (?, ?, 'bogus', 'fake', 4, ?, '')",
                    (uuid4().bytes, uuid4().bytes, b"\x00" * 16),
                )


class TestUpgradeFromV2:
    def test_existing_embedding_rows_survive_rebuild(self, tmp_path: Path) -> None:
        """The 0003 migration rebuilds the embeddings table to widen its
        CHECK constraint. Pre-existing event/memory_item embedding rows
        must round-trip unchanged."""
        path = tmp_path / "v2.db"
        conn = _pre_0003_db(path)
        try:
            event_id = uuid4().bytes
            emb_id = uuid4().bytes
            blob = (1.0).hex().encode()[:32].ljust(32, b"\0")
            # Insert a v1-shaped event + its embedding.
            conn.execute(
                "INSERT INTO events "
                "(id, content, metadata, source, created_at, last_decayed_at) "
                "VALUES (?, 'hello', '{}', NULL, '2025-01-01T00:00:00+00:00', "
                "'2025-01-01T00:00:00+00:00')",
                (event_id,),
            )
            conn.execute(
                "INSERT INTO embeddings "
                "(id, item_id, item_kind, model, dim, vector, created_at) "
                "VALUES (?, ?, 'event', 'fake', 8, ?, '2025-01-01T00:00:00+00:00')",
                (emb_id, event_id, blob),
            )
        finally:
            conn.close()

        # Reopen via SqliteStorage which will apply migration 0003.
        with SqliteStorage(path) as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT id, item_id, item_kind, model, dim, vector FROM embeddings WHERE id = ?",
                (emb_id,),
            ).fetchone()
            assert row is not None
            assert bytes(row["id"]) == emb_id
            assert bytes(row["item_id"]) == event_id
            assert row["item_kind"] == "event"
            assert row["model"] == "fake"
            assert row["dim"] == 8
            assert bytes(row["vector"]) == blob

    def test_embeddings_unique_constraint_preserved(self, tmp_path: Path) -> None:
        """The UNIQUE(item_id, item_kind, model) constraint must survive
        the table rebuild."""
        path = tmp_path / "v2.db"
        conn = _pre_0003_db(path)
        conn.close()
        with SqliteStorage(path) as storage:
            conn = storage._connect()
            # Insert a procedure target then try to double-insert its
            # embedding -- the second INSERT must fail.
            pid = uuid4().bytes
            conn.execute(
                "INSERT INTO procedures "
                "(id, situation, action, last_decayed_at, created_at, updated_at) "
                "VALUES (?, 's', 'a', '', '', '')",
                (pid,),
            )
            conn.execute(
                "INSERT INTO embeddings "
                "(id, item_id, item_kind, model, dim, vector, created_at) "
                "VALUES (?, ?, 'procedure', 'fake', 4, ?, '')",
                (uuid4().bytes, pid, b"\x00" * 16),
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO embeddings "
                    "(id, item_id, item_kind, model, dim, vector, created_at) "
                    "VALUES (?, ?, 'procedure', 'fake', 4, ?, '')",
                    (uuid4().bytes, pid, b"\x00" * 16),
                )
