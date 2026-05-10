"""Migration runner.

Migrations are numbered SQL files in this package: `0001_initial.sql`,
`0002_…`, etc. Each file is responsible for its own `INSERT INTO
schema_migrations (version) VALUES (N)`, wrapped in `BEGIN ... COMMIT;` so
that schema change and version record are atomic.

The runner bootstraps the `schema_migrations` table, reads applied versions,
and runs pending migrations in order.
"""

from __future__ import annotations

import re
import sqlite3
from importlib import resources

_MIGRATION_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def list_migrations() -> list[tuple[int, str]]:
    """Return `[(version, filename), ...]` sorted by version."""
    pkg = resources.files(__name__)
    migrations: list[tuple[int, str]] = []
    for entry in pkg.iterdir():
        if not entry.is_file():
            continue
        match = _MIGRATION_RE.match(entry.name)
        if match is None:
            continue
        migrations.append((int(match.group(1)), entry.name))
    migrations.sort(key=lambda item: item[0])
    return migrations


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Return the set of migration versions already applied in `conn`.

    Bootstraps the `schema_migrations` table if absent.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version    INTEGER PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)"
        ")"
    )
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply every pending migration in order. Return the versions applied."""
    applied = applied_versions(conn)
    pending = [
        (version, filename) for version, filename in list_migrations() if version not in applied
    ]
    if not pending:
        return []

    pkg = resources.files(__name__)
    newly_applied: list[int] = []
    for version, filename in pending:
        sql = (pkg / filename).read_text(encoding="utf-8")
        conn.executescript(sql)
        recorded = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        if version not in recorded:
            raise RuntimeError(
                f"migration {filename} did not record version {version} in "
                f"schema_migrations - every migration script must INSERT its own version"
            )
        newly_applied.append(version)
    return newly_applied
