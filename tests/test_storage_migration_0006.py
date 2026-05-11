"""Tests for migration 0006 (multi-tenant tenant_id columns)."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from uuid import uuid4

from engram.storage import SqliteStorage
from engram.storage.migrations import list_migrations


def _pre_0006_db(path: Path) -> sqlite3.Connection:
    """Apply migrations 0001..0005 only -- the state before 0006."""
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
    for name in (
        "0001_initial.sql",
        "0002_decay.sql",
        "0003_procedures.sql",
        "0004_temporal_conflicts.sql",
        "0005_resolution_merge.sql",
    ):
        sql = (pkg / name).read_text(encoding="utf-8")
        conn.executescript(sql)
    return conn


class TestMigrationListing:
    def test_includes_0006(self) -> None:
        versions = [v for v, _ in list_migrations()]
        assert {1, 2, 3, 4, 5, 6} <= set(versions)
        assert versions == sorted(set(versions))


class TestFreshUpgrade:
    def test_tenant_id_columns_present(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            for table in ("events", "memory_items", "procedures"):
                cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                assert "tenant_id" in cols, f"{table} missing tenant_id column"

    def test_tenant_id_indexes_present(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            indexes = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
            for expected in (
                "idx_events_tenant_id",
                "idx_memory_items_tenant_id",
                "idx_procedures_tenant_id",
            ):
                assert expected in indexes

    def test_tenant_id_round_trip_on_event(self, tmp_path: Path) -> None:
        from engram.schemas import Event

        with SqliteStorage(tmp_path / "x.db") as storage:
            e = Event(content="x", tenant_id="acme-corp")
            storage.insert_event(e)
            fetched = storage.get_event(e.id)
            assert fetched is not None
            assert fetched.tenant_id == "acme-corp"

    def test_tenant_id_round_trip_on_memory_item(self, tmp_path: Path) -> None:
        from engram.schemas import Level, MemoryItem

        with SqliteStorage(tmp_path / "x.db") as storage:
            item = MemoryItem(level=Level.SUMMARY, content="x", tenant_id="tenant-a")
            storage.insert_memory_item(item)
            fetched = storage.get_memory_item(item.id)
            assert fetched is not None
            assert fetched.tenant_id == "tenant-a"

    def test_tenant_id_round_trip_on_procedure(self, tmp_path: Path) -> None:
        from engram.schemas import Outcome, Procedure

        with SqliteStorage(tmp_path / "x.db") as storage:
            p = Procedure(
                situation="s",
                action="a",
                outcome=Outcome.SUCCESS,
                tenant_id="alpha",
            )
            storage.insert_procedure(p)
            fetched = storage.get_procedure(p.id)
            assert fetched is not None
            assert fetched.tenant_id == "alpha"

    def test_default_tenant_id_is_none(self, tmp_path: Path) -> None:
        """Existing callers that don't set tenant_id end up untenanted."""
        from engram.schemas import Event

        with SqliteStorage(tmp_path / "x.db") as storage:
            e = Event(content="x")
            storage.insert_event(e)
            fetched = storage.get_event(e.id)
            assert fetched is not None
            assert fetched.tenant_id is None


class TestUpgradeFromV5:
    def test_existing_rows_have_null_tenant(self, tmp_path: Path) -> None:
        """Pre-0006 rows end up untenanted after upgrade."""
        path = tmp_path / "v5.db"
        conn = _pre_0006_db(path)
        try:
            eid = uuid4().bytes
            conn.execute(
                "INSERT INTO events "
                "(id, content, metadata, created_at, last_decayed_at) "
                "VALUES (?, 'pre', '{}', '2025-01-01', '2025-01-01')",
                (eid,),
            )
        finally:
            conn.close()
        with SqliteStorage(path) as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT tenant_id FROM events WHERE id = ?", (eid,)
            ).fetchone()
            assert row is not None
            assert row["tenant_id"] is None

    def test_new_rows_can_set_tenant(self, tmp_path: Path) -> None:
        from engram.schemas import Event

        path = tmp_path / "v5.db"
        _pre_0006_db(path).close()
        with SqliteStorage(path) as storage:
            e = Event(content="x", tenant_id="t1")
            storage.insert_event(e)
            fetched = storage.get_event(e.id)
            assert fetched is not None
            assert fetched.tenant_id == "t1"
