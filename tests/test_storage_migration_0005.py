"""Tests for migration 0005 (widen conflicts.resolution to include 'merge')."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from uuid import uuid4

from engram.storage import SqliteStorage
from engram.storage.migrations import list_migrations


def _pre_0005_db(path: Path) -> sqlite3.Connection:
    """Apply migrations 0001..0004 only -- the state before 0005."""
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
    ):
        sql = (pkg / name).read_text(encoding="utf-8")
        conn.executescript(sql)
    return conn


class TestMigrationListing:
    def test_includes_0005(self) -> None:
        versions = [v for v, _ in list_migrations()]
        assert {1, 2, 3, 4, 5} <= set(versions)
        assert versions == sorted(set(versions))


class TestFreshUpgrade:
    def test_conflicts_table_still_present(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conflicts'"
            ).fetchone()
            assert row is not None

    def test_merge_resolution_accepted(self, tmp_path: Path) -> None:
        """The whole point of 0005: insert a resolved conflict with
        resolution='merge' and have it succeed."""
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            a, b = uuid4().bytes, uuid4().bytes
            for mid in (a, b):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, 'summary', 'x', '', '', '', '')",
                    (mid,),
                )
            conn.execute(
                "INSERT INTO conflicts "
                "(id, source_item_id, target_item_id, similarity, "
                " status, resolution, resolved_at, detected_at) "
                "VALUES (?, ?, ?, 0.9, 'resolved', 'merge', "
                " '2026-05-01T00:00:00+00:00', '2026-05-01T00:00:00+00:00')",
                (uuid4().bytes, a, b),
            )

    def test_other_resolutions_still_accepted(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            for resolution in (
                "prefer_recent",
                "prefer_trusted",
                "prefer_frequent",
                "keep_both",
                "manual",
            ):
                a, b = uuid4().bytes, uuid4().bytes
                for mid in (a, b):
                    conn.execute(
                        "INSERT INTO memory_items "
                        "(id, level, content, created_at, updated_at, "
                        " last_decayed_at, valid_from) "
                        "VALUES (?, 'summary', 'x', '', '', '', '')",
                        (mid,),
                    )
                conn.execute(
                    "INSERT INTO conflicts "
                    "(id, source_item_id, target_item_id, similarity, "
                    " status, resolution, resolved_at, resolved_winner_id, "
                    " detected_at) "
                    "VALUES (?, ?, ?, 0.9, 'resolved', ?, "
                    " '2026-05-01T00:00:00+00:00', ?, "
                    " '2026-05-01T00:00:00+00:00')",
                    (uuid4().bytes, a, b, resolution, a),
                )

    def test_bogus_resolution_still_rejected(self, tmp_path: Path) -> None:
        with SqliteStorage(tmp_path / "x.db") as storage:
            conn = storage._connect()
            a, b = uuid4().bytes, uuid4().bytes
            for mid in (a, b):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, 'summary', 'x', '', '', '', '')",
                    (mid,),
                )
            try:
                conn.execute(
                    "INSERT INTO conflicts "
                    "(id, source_item_id, target_item_id, similarity, "
                    " status, resolution, resolved_at, detected_at) "
                    "VALUES (?, ?, ?, 0.9, 'resolved', 'bogus_policy', "
                    " '2026-05-01T00:00:00+00:00', '2026-05-01T00:00:00+00:00')",
                    (uuid4().bytes, a, b),
                )
            except sqlite3.IntegrityError:
                pass
            else:  # pragma: no cover
                raise AssertionError("CHECK accepted bogus resolution post-0005")


class TestUpgradeFromV4:
    def test_existing_rows_survive_table_rebuild(self, tmp_path: Path) -> None:
        path = tmp_path / "v4.db"
        conn = _pre_0005_db(path)
        try:
            a, b = uuid4().bytes, uuid4().bytes
            for mid in (a, b):
                conn.execute(
                    "INSERT INTO memory_items "
                    "(id, level, content, created_at, updated_at, "
                    " last_decayed_at, valid_from) "
                    "VALUES (?, 'summary', 'x', '', '', '', '')",
                    (mid,),
                )
            cid = uuid4().bytes
            conn.execute(
                "INSERT INTO conflicts "
                "(id, source_item_id, target_item_id, similarity, status, "
                " detected_at) "
                "VALUES (?, ?, ?, 0.85, 'open', '2026-04-01T00:00:00+00:00')",
                (cid, a, b),
            )
        finally:
            conn.close()

        # Reopen via SqliteStorage which applies 0005.
        with SqliteStorage(path) as storage:
            conn = storage._connect()
            row = conn.execute(
                "SELECT id, source_item_id, target_item_id, similarity, status "
                "FROM conflicts WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row is not None
            assert bytes(row["id"]) == cid
            assert bytes(row["source_item_id"]) == a
            assert bytes(row["target_item_id"]) == b
            assert row["similarity"] == 0.85
            assert row["status"] == "open"

    def test_indexes_recreated(self, tmp_path: Path) -> None:
        path = tmp_path / "v4.db"
        _pre_0005_db(path).close()
        with SqliteStorage(path) as storage:
            conn = storage._connect()
            indexes = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
            for expected in (
                "idx_conflicts_status",
                "idx_conflicts_source_item",
                "idx_conflicts_target_item",
                "idx_conflicts_resolved_winner",
            ):
                assert expected in indexes, f"missing index {expected}"
