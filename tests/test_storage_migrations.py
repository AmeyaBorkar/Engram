"""Tests for the migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from engram.storage import SqliteStorage
from engram.storage.migrations import (
    applied_versions,
    apply_migrations,
    list_migrations,
)


def test_list_migrations_sorted_by_version() -> None:
    migrations = list_migrations()
    assert migrations, "expected at least one migration"
    versions = [v for v, _ in migrations]
    assert versions == sorted(versions)
    assert versions[0] == 1


def test_apply_migrations_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    conn = sqlite3.connect(db)
    conn.isolation_level = None
    try:
        first = apply_migrations(conn)
        second = apply_migrations(conn)
        assert first
        assert second == []
    finally:
        conn.close()


def test_applied_versions_after_initialize(disk_storage: SqliteStorage) -> None:
    conn = disk_storage._connect()
    versions = applied_versions(conn)
    assert 1 in versions


def test_reopened_storage_skips_already_applied_migrations(tmp_path: Path) -> None:
    db = tmp_path / "reopen.db"
    s1 = SqliteStorage(db)
    s1.initialize()
    s1.close()

    s2 = SqliteStorage(db)
    s2.initialize()
    try:
        conn = s2._connect()
        rows = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()
        assert rows[0] == 1, "no duplicate migration recorded"
    finally:
        s2.close()


def test_missing_version_record_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A migration that fails to record its own version is a runner error."""
    from engram.storage import migrations as mig

    bad_sql = "BEGIN; CREATE TABLE noop_table (x INT); COMMIT;"

    def fake_list() -> list[tuple[int, str]]:
        return [(9999, "9999_bad.sql")]

    class FakePath:
        def __truediv__(self, _name: str) -> FakePath:
            return self

        def read_text(self, encoding: str = "utf-8") -> str:
            return bad_sql

    monkeypatch.setattr(mig, "list_migrations", fake_list)
    monkeypatch.setattr(mig.resources, "files", lambda _: FakePath())

    db = tmp_path / "bad.db"
    conn = sqlite3.connect(db)
    conn.isolation_level = None
    try:
        with pytest.raises(RuntimeError, match="did not record version"):
            mig.apply_migrations(conn)
    finally:
        conn.close()
