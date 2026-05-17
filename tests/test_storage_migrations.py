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
        assert rows[0] == len(list_migrations()), "no duplicate migration recorded"
    finally:
        s2.close()


def test_apply_migrations_rejects_outer_transaction(tmp_path: Path) -> None:
    """Regression for H-43: `apply_migrations` MUST refuse to run inside an
    outer transaction.

    `conn.executescript()` issues an unconditional `COMMIT;` before each
    script — a caller that opened a transaction would have it silently
    committed mid-flight.  The runner asserts the precondition rather
    than letting the subtle partial-commit go undetected.
    """
    db = tmp_path / "outer_tx.db"
    conn = sqlite3.connect(db)
    conn.isolation_level = None  # autocommit; we drive BEGIN manually
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert conn.in_transaction is True
        with pytest.raises(RuntimeError, match="inside an outer transaction"):
            apply_migrations(conn)
        # Tidy up so close() doesn't fail.
        conn.execute("ROLLBACK")
    finally:
        conn.close()


def test_applied_versions_uses_immediate_transaction(tmp_path: Path) -> None:
    """Regression for H-42: bootstrapping schema_migrations + reading the
    applied versions runs under BEGIN IMMEDIATE so two parallel openers
    can't both observe the table as empty and both proceed to apply v1.

    We can't easily race two real connections at this layer, so verify
    the contract: `applied_versions` leaves the connection in autocommit
    state (i.e. it COMMITted its own transaction before returning).
    """
    db = tmp_path / "ave.db"
    conn = sqlite3.connect(db)
    conn.isolation_level = None
    try:
        out = applied_versions(conn)
        # Fresh db -> no versions applied yet.
        assert out == set()
        # The bootstrap transaction must have COMMITted before we got
        # control back — apply_migrations would otherwise raise on the
        # in-transaction precondition above.
        assert conn.in_transaction is False
    finally:
        conn.close()


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
