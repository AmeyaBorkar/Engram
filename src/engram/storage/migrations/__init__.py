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
    """Return `[(version, filename), ...]` sorted by version.

    Raises if two migration files share the same numeric prefix
    (e.g. \`0005_a.sql\` + \`0005_b.sql\`) — the runner would treat one
    as the canonical 'v5' and silently skip the other, which has
    silently lost real DDL in tooling-mistake scenarios.
    """
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
    seen: dict[int, str] = {}
    for version, filename in migrations:
        if version in seen:
            raise RuntimeError(
                f"duplicate migration version {version}: "
                f"{seen[version]!r} and {filename!r}"
            )
        seen[version] = filename
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
    versions = {int(row[0]) for row in rows}
    # Sanity-check: applied versions must be a contiguous prefix [1..N].
    # A gap (1, 3 without 2) signals manual db surgery or a partial-
    # restore — applying 2 now risks DDL conflicts against the schema
    # that 3 already produced.  Raise loudly so an operator
    # investigates.
    if versions:
        ordered = sorted(versions)
        expected = list(range(ordered[0], ordered[0] + len(ordered)))
        if ordered != expected:
            missing = [v for v in expected if v not in versions]
            raise RuntimeError(
                f"schema_migrations has gaps: applied {ordered}, missing {missing}"
            )
    return versions


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply every pending migration in order. Return the versions applied.

    Some migrations (0007, 0008) toggle `PRAGMA foreign_keys = OFF` to
    rebuild a table.  If the migration body raises before reaching the
    trailing `PRAGMA foreign_keys = ON`, the connection inherits an
    OFF state that lets subsequent inserts violate FK constraints
    silently.  We wrap the apply loop in a try/finally and restore the
    pre-migration setting so a partial failure can't poison the
    connection's FK enforcement.
    """
    applied = applied_versions(conn)
    pending = [
        (version, filename) for version, filename in list_migrations() if version not in applied
    ]
    if not pending:
        return []

    fk_was_on = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    pkg = resources.files(__name__)
    newly_applied: list[int] = []
    try:
        for version, filename in pending:
            sql = (pkg / filename).read_text(encoding="utf-8")
            try:
                conn.executescript(sql)
            except sqlite3.Error as exc:
                # Enrich the error with the migration filename and version
                # so a typo in 0010_*.sql doesn't surface as a bare SQL
                # syntax error with no context.
                raise RuntimeError(
                    f"migration {filename} (version {version}) failed: {exc}"
                ) from exc
            recorded = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            if version not in recorded:
                raise RuntimeError(
                    f"migration {filename} did not record version {version} in "
                    f"schema_migrations - every migration script must INSERT its own version"
                )
            newly_applied.append(version)
    finally:
        # Restore the FK state regardless of success/failure.  If a
        # migration left foreign_keys OFF mid-rebuild, this resets to the
        # pre-migration value so downstream connection users don't
        # silently insert orphan rows.
        conn.execute(f"PRAGMA foreign_keys = {'ON' if fk_was_on else 'OFF'}")
    return newly_applied
