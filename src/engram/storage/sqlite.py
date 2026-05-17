"""SQLite storage backend.

WAL mode, foreign keys on, per-thread connections. Single-process,
single-machine.  A multi-tenant Postgres backend implementing the
same Storage protocol is a roadmap item, not on disk yet.

Threading model: each thread gets its own connection on first use. The
connection is closed when `SqliteStorage.close()` is called by *that* thread,
or when the storage is dropped. Cross-thread sharing of a connection is not
supported (and `check_same_thread` enforces this).
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID

import numpy as np

from engram.schemas import (
    Cluster,
    Conflict,
    ConflictStatus,
    DecayState,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
    Outcome,
    Procedure,
    ProvenanceLink,
    Resolution,
    Verdict,
)
from engram.retrieve._bm25 import BM25Index
from engram.storage._serialize import (
    dumps_metadata,
    iso,
    loads_metadata,
    pack_vector,
    parse_iso,
    unpack_vector,
)
from engram.storage._vector_index import VectorIndex
from engram.storage.migrations import apply_migrations


class ProvenanceProtectedError(RuntimeError):
    """Raised by `delete_cold_items(EVENT)` when blocked by provenance links.

    Subclass of RuntimeError for backwards compatibility — callers that
    used to catch the bare RuntimeError still catch this — while letting
    new code catch this exact class to distinguish 'cold-protected by
    provenance' from any other RuntimeError surfacing from storage.
    """


def _validate_path(path: str | Path) -> str:
    """Normalize and validate a storage path string.

    ``:memory:`` is accepted verbatim as the canonical in-memory marker.
    Everything else must look like a filesystem path: the SQLite C API
    accepts magic URI forms (``file:foo?mode=memory``, ``file::memory:``,
    ``file:foo.db?mode=ro&cache=shared``) when ``uri=True`` is passed to
    ``sqlite3.connect``, but they short-circuit our durability assumptions
    (WAL, foreign_keys, the path-on-disk that the bench / inspector
    prints).  Reject them at construction so the caller gets a clear
    error instead of a `:memory:` database masquerading as a file path.
    """
    raw = str(path)
    if raw == ":memory:":
        return raw
    # Anything starting with `file:` is an explicit URI; SQLite parses it
    # only when uri=True is passed to connect, but we never opt in, so
    # this would silently produce a relative-path file named ``file:foo``
    # on disk.  That's user-hostile; reject up front.
    lowered = raw.lower()
    if lowered.startswith("file:"):
        raise ValueError(
            f"SqliteStorage path may not be a URI ({raw!r}); pass a filesystem "
            "path or ':memory:' instead"
        )
    # Likewise, embedded ':memory:' segments outside the canonical sentinel
    # are almost certainly a user mistake — they want either the literal
    # filesystem path or the bare ':memory:' marker.
    if ":memory:" in raw:
        raise ValueError(
            f"SqliteStorage path contains ':memory:' fragment ({raw!r}); "
            "pass exactly ':memory:' for an in-memory store or a filesystem path"
        )
    return raw


def _row_to_event(row: sqlite3.Row) -> Event:
    keys = row.keys()
    tenant_id = row["tenant_id"] if "tenant_id" in keys else None
    return Event(
        id=UUID(bytes=row["id"]),
        content=row["content"],
        metadata=loads_metadata(row["metadata"]),
        source=row["source"],
        created_at=parse_iso(row["created_at"]),
        tenant_id=tenant_id,
    )


def _row_to_memory_item(row: sqlite3.Row) -> MemoryItem:
    keys = row.keys()
    valid_from_raw = row["valid_from"]
    valid_until_raw = row["valid_until"]
    invalidated_at_raw = row["invalidated_at"]
    invalidated_by_raw = row["invalidated_by"]
    tenant_id = row["tenant_id"] if "tenant_id" in keys else None
    return MemoryItem(
        id=UUID(bytes=row["id"]),
        level=Level(row["level"]),
        content=row["content"],
        weight=row["weight"],
        cluster_id=UUID(bytes=row["cluster_id"]) if row["cluster_id"] else None,
        metadata=loads_metadata(row["metadata"]),
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
        valid_from=parse_iso(valid_from_raw) if valid_from_raw else None,
        valid_until=parse_iso(valid_until_raw) if valid_until_raw else None,
        invalidated_at=parse_iso(invalidated_at_raw) if invalidated_at_raw else None,
        invalidated_by=UUID(bytes=invalidated_by_raw) if invalidated_by_raw else None,
        source_trust=row["source_trust"],
        tenant_id=tenant_id,
    )


def _memory_item_insert_row(item: MemoryItem) -> tuple[Any, ...]:
    """Flatten a MemoryItem into the SQL row tuple for INSERT.

    `valid_from` is guaranteed non-None by the Pydantic model validator;
    the fallback here is defensive only.
    """
    valid_from = item.valid_from if item.valid_from is not None else item.created_at
    return (
        item.id.bytes,
        item.level.value,
        item.content,
        item.weight,
        item.cluster_id.bytes if item.cluster_id else None,
        dumps_metadata(item.metadata),
        iso(item.created_at),
        iso(item.updated_at),
        iso(item.updated_at),
        iso(valid_from),
        iso(item.valid_until) if item.valid_until else None,
        iso(item.invalidated_at) if item.invalidated_at else None,
        item.invalidated_by.bytes if item.invalidated_by else None,
        item.source_trust,
        item.tenant_id,
    )


def _row_to_embedding(row: sqlite3.Row) -> Embedding:
    dim = int(row["dim"])
    return Embedding(
        id=UUID(bytes=row["id"]),
        item_id=UUID(bytes=row["item_id"]),
        item_kind=ItemKind(row["item_kind"]),
        model=row["model"],
        dim=dim,
        vector=unpack_vector(row["vector"], dim),
        created_at=parse_iso(row["created_at"]),
    )


def _row_to_cluster(row: sqlite3.Row) -> Cluster:
    return Cluster(
        id=UUID(bytes=row["id"]),
        cohesion=row["cohesion"],
        created_at=parse_iso(row["created_at"]),
    )


def _row_to_procedure(row: sqlite3.Row) -> Procedure:
    keys = row.keys()
    tenant_id = row["tenant_id"] if "tenant_id" in keys else None
    return Procedure(
        id=UUID(bytes=row["id"]),
        situation=row["situation"],
        action=row["action"],
        outcome=Outcome(row["outcome"]),
        weight=row["weight"],
        metadata=loads_metadata(row["metadata"]),
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
        tenant_id=tenant_id,
    )


def _row_to_conflict(row: sqlite3.Row) -> Conflict:
    resolution_raw = row["resolution"]
    resolved_at_raw = row["resolved_at"]
    winner_raw = row["resolved_winner_id"]
    return Conflict(
        id=UUID(bytes=row["id"]),
        source_item_id=UUID(bytes=row["source_item_id"]),
        target_item_id=UUID(bytes=row["target_item_id"]),
        similarity=row["similarity"],
        verdict=Verdict(row["verdict"]),
        status=ConflictStatus(row["status"]),
        resolution=Resolution(resolution_raw) if resolution_raw else None,
        resolved_winner_id=UUID(bytes=winner_raw) if winner_raw else None,
        resolved_at=parse_iso(resolved_at_raw) if resolved_at_raw else None,
        detected_at=parse_iso(row["detected_at"]),
    )


def _row_to_provenance_link(row: sqlite3.Row) -> ProvenanceLink:
    return ProvenanceLink(
        id=UUID(bytes=row["id"]),
        memory_item_id=UUID(bytes=row["memory_item_id"]),
        event_id=UUID(bytes=row["event_id"]),
        weight=row["weight"],
        created_at=parse_iso(row["created_at"]),
    )


def _row_to_decay_state(row: sqlite3.Row, kind: ItemKind) -> DecayState:
    cold_at_raw = row["cold_at"]
    return DecayState(
        item_id=UUID(bytes=row["id"]),
        item_kind=kind,
        weight=row["weight"],
        reinforcement_count=int(row["reinforcement_count"]),
        corroboration_count=int(row["corroboration_count"]),
        contradiction_count=int(row["contradiction_count"]),
        last_decayed_at=parse_iso(row["last_decayed_at"]),
        cold_at=parse_iso(cold_at_raw) if cold_at_raw is not None else None,
    )


# Per-kind SQL lookup tables. We pre-build every decay-state statement so
# that the runtime path never interpolates a table name into SQL - that
# keeps `ruff` S608 honest, and the closed `ItemKind` enum is a safer
# boundary than an inline switch on the enum value.
_DECAY_COLS = (
    "id, weight, reinforcement_count, corroboration_count, "
    "contradiction_count, last_decayed_at, cold_at"
)
_DECAY_TABLES: dict[ItemKind, str] = {
    ItemKind.EVENT: "events",
    ItemKind.MEMORY_ITEM: "memory_items",
    ItemKind.PROCEDURE: "procedures",
}
_GET_DECAY_STATE_SQL: dict[ItemKind, str] = {
    kind: f"SELECT {_DECAY_COLS} FROM {table} WHERE id = ?"  # noqa: S608
    for kind, table in _DECAY_TABLES.items()
}
_ITER_DECAY_STATES_HOT_SQL: dict[ItemKind, str] = {
    kind: f"SELECT {_DECAY_COLS} FROM {table} WHERE cold_at IS NULL ORDER BY id"  # noqa: S608
    for kind, table in _DECAY_TABLES.items()
}
_ITER_DECAY_STATES_ALL_SQL: dict[ItemKind, str] = {
    kind: f"SELECT {_DECAY_COLS} FROM {table} ORDER BY id"  # noqa: S608
    for kind, table in _DECAY_TABLES.items()
}
def _build_update_decay_sql(kind: ItemKind, table: str) -> str:
    # memory_items / procedures carry `updated_at`; events do not.  Bump
    # it alongside the decay-state write so audit logs that watch
    # `updated_at` reflect every reinforce / corroborate / tick.
    if kind is ItemKind.EVENT:
        return (
            f"UPDATE {table} SET weight = ?, reinforcement_count = ?, "  # noqa: S608
            "corroboration_count = ?, contradiction_count = ?, "
            "last_decayed_at = ?, cold_at = ? WHERE id = ?"
        )
    return (
        f"UPDATE {table} SET weight = ?, reinforcement_count = ?, "  # noqa: S608
        "corroboration_count = ?, contradiction_count = ?, "
        "last_decayed_at = ?, cold_at = ?, updated_at = ? WHERE id = ?"
    )


_UPDATE_DECAY_STATE_SQL: dict[ItemKind, str] = {
    kind: _build_update_decay_sql(kind, table)
    for kind, table in _DECAY_TABLES.items()
}
_MARK_COLD_SQL: dict[ItemKind, str] = {
    kind: f"UPDATE {table} SET cold_at = ? WHERE id = ?"  # noqa: S608
    for kind, table in _DECAY_TABLES.items()
}
_UNMARK_COLD_SQL: dict[ItemKind, str] = {
    kind: f"UPDATE {table} SET cold_at = NULL WHERE id = ?"  # noqa: S608
    for kind, table in _DECAY_TABLES.items()
}
_COUNT_COLD_SQL: dict[ItemKind, str] = {
    kind: f"SELECT COUNT(*) FROM {table} WHERE cold_at IS NOT NULL"  # noqa: S608
    for kind, table in _DECAY_TABLES.items()
}
_DELETE_COLD_SQL: dict[ItemKind, str] = {
    kind: f"DELETE FROM {table} WHERE cold_at IS NOT NULL"  # noqa: S608
    for kind, table in _DECAY_TABLES.items()
}
_DECAY_TOTALS_SQL: dict[ItemKind, str] = {
    kind: (
        "SELECT "  # noqa: S608
        "SUM(CASE WHEN cold_at IS NULL THEN 1 ELSE 0 END) AS hot_items, "
        "SUM(CASE WHEN cold_at IS NOT NULL THEN 1 ELSE 0 END) AS cold_items, "
        "COALESCE(SUM(CASE WHEN cold_at IS NULL THEN reinforcement_count ELSE 0 END), 0) "
        "  AS reinforcement_total, "
        "COALESCE(SUM(CASE WHEN cold_at IS NULL THEN corroboration_count ELSE 0 END), 0) "
        "  AS corroboration_total, "
        "COALESCE(SUM(CASE WHEN cold_at IS NULL THEN contradiction_count ELSE 0 END), 0) "
        "  AS contradiction_total "
        f"FROM {table}"
    )
    for kind, table in _DECAY_TABLES.items()
}


# Per-shard SELECT for the in-memory vector index. Returns the four
# columns the index expects: `item_id`, `vector`, `cold` (0/1),
# `level` (string). `cold` and `level` are computed at SQL time so the
# shard can filter without a join.
_INDEX_REBUILD_SQL: dict[str, str] = {
    "event": (
        "SELECT emb.item_id AS item_id, "
        "       emb.vector  AS vector, "
        "       (e.cold_at IS NOT NULL) AS cold, "
        "       'event' AS level "
        "FROM embeddings emb "
        "JOIN events e ON emb.item_id = e.id "
        "WHERE emb.item_kind = 'event' AND emb.model = ?"
    ),
    "memory_item": (
        # NOTE: the shared in-memory shard intentionally includes
        # invalidated rows so that `search_memory_item_embeddings_as_of`
        # can find them when looking at historical timestamps.  The
        # non-as_of variant filters invalidated rows at the SQL step (see
        # search_memory_item_embeddings).
        "SELECT emb.item_id AS item_id, "
        "       emb.vector  AS vector, "
        "       (mi.cold_at IS NOT NULL) AS cold, "
        "       mi.level    AS level "
        "FROM embeddings emb "
        "JOIN memory_items mi ON emb.item_id = mi.id "
        "WHERE emb.item_kind = 'memory_item' AND emb.model = ?"
    ),
    # Procedures piggyback on the same kind/model index shape. The
    # `level` slot stores the procedure's outcome so retrieve_procedures
    # can filter by outcome (success/failure/...) through the existing
    # `levels=` kwarg on VectorIndex.search.
    "procedure": (
        "SELECT emb.item_id AS item_id, "
        "       emb.vector  AS vector, "
        "       (p.cold_at IS NOT NULL) AS cold, "
        "       p.outcome   AS level "
        "FROM embeddings emb "
        "JOIN procedures p ON emb.item_id = p.id "
        "WHERE emb.item_kind = 'procedure' AND emb.model = ?"
    ),
}


class SqliteStorage:
    """SQLite-backed `Storage` implementation."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = _validate_path(path)
        self._lock = threading.Lock()
        # `_local` holds the per-thread sqlite3.Connection.  Using a
        # threading.local instead of a tid-keyed dict means a recycled
        # OS thread id can't hand a new thread the connection of a dead
        # one — the previous design cached connections by
        # `threading.get_ident()`, but tids are recycled, so a dead
        # thread's bound connection could be returned to a new thread
        # and explode on `check_same_thread`.
        #
        # `_all_connections` is a sidecar set for close() / __exit__ to
        # find every live connection.  Holds strong references, so a
        # connection survives until close() is called; that mirrors the
        # previous semantics (caller-owned lifecycle).
        self._local = threading.local()
        self._all_connections: list[sqlite3.Connection] = []
        self._initialized = False
        self._vector_index = VectorIndex()
        # Lexical BM25 index over event content. Lazy-built on the
        # first `bm25_search_events` call; rebuilt from scratch when
        # the event corpus changes (insert / mark_cold / unmark_cold /
        # delete_cold_items) OR when the caller asks for different
        # k1 / b hyperparameters than the cached index has. BM25 is
        # independent of the embedding model, so one index covers every
        # retrieve call regardless of which embedder is plugged in.
        self._bm25_events: BM25Index[UUID] | None = None
        self._bm25_events_dirty: bool = True
        self._bm25_k1: float = 1.5
        self._bm25_b: float = 0.75
        # Track which corpus the cached index covers.  If a caller flips
        # `include_cold`, the previous index is wrong for them — see
        # H-30: the first call with include_cold=False used to poison the
        # cache so a follow-up include_cold=True silently dropped cold
        # rows from the search.
        self._bm25_include_cold: bool = False

    # --- lifecycle ----------------------------------------------------------

    @property
    def path(self) -> str:
        """The database path, as the constructor received it.

        Diagnostic surface — a downstream tool wanting to print which
        database an instance is talking to (for log enrichment, error
        messages, manifest fields) can read this without poking
        `_path` directly.  Returns ``':memory:'`` for in-memory stores.
        """
        return self._path

    def __enter__(self) -> SqliteStorage:
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def initialize(self) -> None:
        # Double-checked locking: the fast path is a lock-free read, the
        # slow path serializes migration application so two threads
        # racing on first-use can't both apply migrations and collide on
        # the schema_migrations UNIQUE constraint after partial DDL.
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            apply_migrations(self._connect_unlocked())
            self._initialized = True

    def close(self) -> None:
        # close() can be invoked from any thread, but `sqlite3.Connection`
        # with check_same_thread=True only allows `.close()` from its
        # creating thread.  Wrap each close in suppress() so a
        # cross-thread close call against an orphaned connection (e.g.,
        # close() called from main after a worker thread exited) does
        # not raise; the process exit reaps OS resources anyway.
        with self._lock:
            for conn in list(self._all_connections):
                with contextlib.suppress(sqlite3.Error, sqlite3.ProgrammingError):
                    conn.close()
            self._all_connections.clear()
            # threading.local has no `clear()` reachable across threads;
            # rebind the calling thread's slot if present.
            if getattr(self._local, "conn", None) is not None:
                self._local.conn = None
            self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            return self._connect_unlocked()

    def _connect_unlocked(self) -> sqlite3.Connection:
        """Get-or-create the current thread's connection.

        Caller must hold `self._lock`.  Split out from `_connect` so
        `initialize()` (which already holds the lock for its
        double-checked-locking pattern) can fetch a connection without
        re-acquiring and deadlocking on a non-reentrant Lock.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(
            self._path,
            isolation_level=None,
            check_same_thread=True,
        )
        conn.row_factory = sqlite3.Row
        # PRAGMA journal_mode = WAL returns the mode SQLite actually settled
        # on.  Some filesystems (NFS, sshfs, read-only mounts) silently fall
        # back to DELETE/MEMORY — which kills the concurrency story without
        # any indication.  Read the result and warn so misconfigured paths
        # surface loudly instead of looking like a working WAL deployment.
        mode_row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
        actual_mode = mode_row[0] if mode_row else ""
        if (
            isinstance(actual_mode, str)
            and actual_mode.lower() != "wal"
            # `:memory:` cannot use WAL by design and returns "memory" — no
            # need to warn on an unavoidable behavior.
            and self._path != ":memory:"
        ):
            warnings.warn(
                f"SqliteStorage requested journal_mode=WAL but SQLite settled "
                f"on {actual_mode!r} for {self._path!r}.  Concurrent writers "
                f"may serialize through file-level locks instead of WAL; "
                f"check the underlying filesystem.",
                RuntimeWarning,
                stacklevel=4,
            )
        conn.execute("PRAGMA foreign_keys = ON")
        # synchronous=NORMAL is the standard recommendation under WAL: writes
        # land in the WAL synchronously (so a single-row commit is durable
        # against process crashes / kernel panics), but the periodic
        # checkpoint that moves data from WAL into the main db file is
        # async.  Durability window: a power loss in the ~1s between the
        # commit and the next checkpoint can lose the most recent
        # transaction.  Acceptable for memory / bench / agent workloads;
        # production deployments with stricter durability requirements
        # should bump to FULL via a custom subclass.
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        # Wait up to 5s for a writer lock instead of failing immediately
        # with SQLITE_BUSY.  Combined with BEGIN IMMEDIATE in
        # transaction(), this gives concurrent writers a real shot at
        # serializing through the WAL instead of surfacing flaky
        # OperationalError to the caller.
        conn.execute("PRAGMA busy_timeout = 5000")
        # cache_size: negative -> KB; 64 MB of page cache for hot
        # PK lookups. The bench ingests 500 events per question and
        # then drills against them; a smaller cache flushes the
        # working set on every haystack switch.
        conn.execute("PRAGMA cache_size = -65536")
        # mmap_size: 256 MB. No-op for :memory: but a real win for
        # disk-backed stores -- B-tree pages get faulted in once
        # and stay resident across queries.
        conn.execute("PRAGMA mmap_size = 268435456")
        # `wal_autocheckpoint=1000` is the SQLite default; the
        # bench writes everything in one big transaction per
        # haystack so we don't tune this knob.
        self._local.conn = conn
        self._all_connections.append(conn)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[None]:
        conn = self._connect()
        if conn.in_transaction:
            yield
            return
        # BEGIN IMMEDIATE acquires the write lock at transaction start
        # instead of deferring it until the first write.  Without this,
        # two concurrent writers can both pass BEGIN, both attempt to
        # write, and one gets SQLITE_BUSY mid-transaction — discarding
        # any work done in the block.
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            # ROLLBACK itself can raise if the connection is in a broken
            # state.  Suppress so the original exception (which is more
            # actionable) is what the caller sees.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        else:
            conn.execute("COMMIT")

    # --- events -------------------------------------------------------------

    def insert_event(self, event: Event) -> None:
        # `last_decayed_at` defaults to `created_at` so the first decay tick
        # has a sane dt. `weight` defaults to 1.0 via the column default;
        # we don't override it here because Stage 4 ships a single insert
        # path that always starts items hot.
        self._connect().execute(
            "INSERT INTO events "
            "(id, content, metadata, source, created_at, last_decayed_at, "
            " tenant_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.id.bytes,
                event.content,
                dumps_metadata(event.metadata),
                event.source,
                iso(event.created_at),
                iso(event.created_at),
                event.tenant_id,
            ),
        )
        self._bm25_events_dirty = True

    def insert_events(self, events: Iterable[Event]) -> int:
        rows = [
            (
                e.id.bytes,
                e.content,
                dumps_metadata(e.metadata),
                e.source,
                iso(e.created_at),
                iso(e.created_at),
                e.tenant_id,
            )
            for e in events
        ]
        if not rows:
            return 0
        self._connect().executemany(
            "INSERT INTO events "
            "(id, content, metadata, source, created_at, last_decayed_at, "
            " tenant_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._bm25_events_dirty = True
        return len(rows)

    def get_event(self, event_id: UUID) -> Event | None:
        row = (
            self._connect()
            .execute("SELECT * FROM events WHERE id = ?", (event_id.bytes,))
            .fetchone()
        )
        return _row_to_event(row) if row is not None else None

    def list_events(
        self,
        limit: int = 100,
        before: datetime | None = None,
        source: str | None = None,
    ) -> list[Event]:
        sql = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if before is not None:
            sql += " AND created_at < ?"
            params.append(iso(before))
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._connect().execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]

    def count_events(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM events").fetchone()[0])

    # --- memory items -------------------------------------------------------

    def insert_memory_item(self, item: MemoryItem) -> None:
        self._connect().execute(
            "INSERT INTO memory_items "
            "(id, level, content, weight, cluster_id, metadata, "
            "created_at, updated_at, last_decayed_at, "
            "valid_from, valid_until, invalidated_at, invalidated_by, source_trust, "
            "tenant_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _memory_item_insert_row(item),
        )

    def insert_memory_items(self, items: Iterable[MemoryItem]) -> int:
        rows = [_memory_item_insert_row(i) for i in items]
        if not rows:
            return 0
        self._connect().executemany(
            "INSERT INTO memory_items "
            "(id, level, content, weight, cluster_id, metadata, "
            "created_at, updated_at, last_decayed_at, "
            "valid_from, valid_until, invalidated_at, invalidated_by, source_trust, "
            "tenant_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def get_memory_item(self, item_id: UUID) -> MemoryItem | None:
        row = (
            self._connect()
            .execute("SELECT * FROM memory_items WHERE id = ?", (item_id.bytes,))
            .fetchone()
        )
        return _row_to_memory_item(row) if row is not None else None

    def list_memory_items(
        self,
        level: Level | None = None,
        cluster_id: UUID | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        sql = "SELECT * FROM memory_items WHERE 1=1"
        params: list[Any] = []
        if level is not None:
            sql += " AND level = ?"
            params.append(level.value)
        if cluster_id is not None:
            sql += " AND cluster_id = ?"
            params.append(cluster_id.bytes)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._connect().execute(sql, params).fetchall()
        return [_row_to_memory_item(r) for r in rows]

    def update_memory_item_weight(self, item_id: UUID, weight: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight {weight} not in [0, 1]")
        # Bump `last_decayed_at` alongside `weight` so the next decay
        # tick computes `dt` from this manual adjustment rather than
        # from the previous tick's timestamp.  Without this, an admin
        # call that bumped weight to 1.0 would be immediately decayed
        # by the accumulated `dt` between the prior tick and now,
        # quietly clawing back the boost.
        now = iso(datetime.now(tz=timezone.utc))
        cursor = self._connect().execute(
            "UPDATE memory_items SET weight = ?, updated_at = ?, "
            "last_decayed_at = ? WHERE id = ?",
            (weight, now, now, item_id.bytes),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"memory_item {item_id} not found")

    def update_memory_item_level(self, item_id: UUID, level: Level) -> None:
        cursor = self._connect().execute(
            "UPDATE memory_items SET level = ?, updated_at = ? WHERE id = ?",
            (level.value, iso(datetime.now(tz=timezone.utc)), item_id.bytes),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"memory_item {item_id} not found")
        self._vector_index.mark_dirty(kind=ItemKind.MEMORY_ITEM.value)

    def iter_memory_items(
        self,
        *,
        level: Level | None = None,
        include_cold: bool = False,
        batch_size: int = 1000,
    ) -> Iterator[MemoryItem]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        sql = "SELECT * FROM memory_items WHERE 1=1"
        params: list[Any] = []
        if level is not None:
            sql += " AND level = ?"
            params.append(level.value)
        if not include_cold:
            sql += " AND cold_at IS NULL"
        sql += " ORDER BY created_at ASC, id ASC"
        cursor = self._connect().execute(sql, params)
        try:
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    return
                for row in rows:
                    yield _row_to_memory_item(row)
        finally:
            cursor.close()

    def count_memory_items(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM memory_items").fetchone()[0])

    def count_memory_items_by_level(self) -> dict[Level, int]:
        rows = (
            self._connect()
            .execute("SELECT level, COUNT(*) AS n FROM memory_items GROUP BY level")
            .fetchall()
        )
        result: dict[Level, int] = dict.fromkeys(Level, 0)
        for row in rows:
            result[Level(row["level"])] = int(row["n"])
        return result

    # --- temporal validity & invalidation (Stage 8) ------------------------

    def invalidate_memory_item(
        self,
        item_id: UUID,
        *,
        at: datetime,
        by: UUID | None = None,
    ) -> None:
        # Only set invalidated_at if currently NULL; preserve the first
        # invalidation timestamp on re-calls (as_of queries rely on this).
        cursor = self._connect().execute(
            "UPDATE memory_items "
            "SET invalidated_at = COALESCE(invalidated_at, ?), "
            "    invalidated_by = COALESCE(invalidated_by, ?), "
            "    updated_at = ? "
            "WHERE id = ?",
            (
                iso(at),
                by.bytes if by is not None else None,
                iso(datetime.now(tz=timezone.utc)),
                item_id.bytes,
            ),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"memory_item {item_id} not found")
        # Validity affects retrieve results, so the vector index needs
        # to know that a row's surface-visibility state has changed.
        self._vector_index.mark_dirty(kind=ItemKind.MEMORY_ITEM.value)

    def set_validity_window(
        self,
        item_id: UUID,
        *,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> None:
        if valid_from is None and valid_until is None:
            return
        # Read-then-validate-then-UPDATE inside a single transaction.  The
        # old code wrote first and validated via read-back, which left the
        # bad row persisted when the validation raised.  Now: SELECT
        # current values under BEGIN IMMEDIATE (writer lock), merge the
        # caller's overrides, check the invariant, then UPDATE — so a
        # raise rolls back the unwritten transaction without leaving a
        # malformed window on disk.
        with self.transaction():
            conn = self._connect()
            row = conn.execute(
                "SELECT valid_from, valid_until FROM memory_items WHERE id = ?",
                (item_id.bytes,),
            ).fetchone()
            if row is None:
                raise KeyError(f"memory_item {item_id} not found")
            # Merge: caller's override wins; otherwise keep the stored value.
            new_vf = valid_from if valid_from is not None else (
                parse_iso(row["valid_from"]) if row["valid_from"] else None
            )
            new_vu = valid_until if valid_until is not None else (
                parse_iso(row["valid_until"]) if row["valid_until"] else None
            )
            if new_vf is not None and new_vu is not None and new_vu < new_vf:
                raise ValueError(
                    f"valid_until {new_vu.isoformat()} precedes valid_from {new_vf.isoformat()}"
                )
            sets: list[str] = []
            params: list[Any] = []
            if valid_from is not None:
                sets.append("valid_from = ?")
                params.append(iso(valid_from))
            if valid_until is not None:
                sets.append("valid_until = ?")
                params.append(iso(valid_until))
            sets.append("updated_at = ?")
            params.append(iso(datetime.now(tz=timezone.utc)))
            params.append(item_id.bytes)
            sql = f"UPDATE memory_items SET {', '.join(sets)} WHERE id = ?"  # noqa: S608
            conn.execute(sql, params)
        self._vector_index.mark_dirty(kind=ItemKind.MEMORY_ITEM.value)

    def set_source_trust(self, item_id: UUID, trust: float | None) -> None:
        if trust is not None and not 0.0 <= trust <= 1.0:
            raise ValueError(f"trust {trust} not in [0, 1]")
        cursor = self._connect().execute(
            "UPDATE memory_items SET source_trust = ?, updated_at = ? WHERE id = ?",
            (trust, iso(datetime.now(tz=timezone.utc)), item_id.bytes),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"memory_item {item_id} not found")

    def search_memory_item_embeddings_as_of(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
        as_of: datetime | None = None,
        levels: Sequence[Level] | None = None,
        exclude_ids: Sequence[UUID] = (),
        include_cold: bool = False,
        candidate_multiplier: int = 4,
    ) -> list[tuple[UUID, str, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if candidate_multiplier < 1:
            raise ValueError(f"candidate_multiplier must be >= 1, got {candidate_multiplier}")
        # Over-fetch from the vector index, then SQL-filter by validity.
        # If the filter eliminates more candidates than expected (a dataset
        # heavy with invalidated rows or narrow validity windows), expand
        # the over-fetch and try again so we still meet `k`.  Capped at
        # 4 doublings (~16x the original ask) so a corpus that's mostly
        # historical doesn't spin forever.
        level_values = [level.value for level in levels] if levels else None
        excl_bytes = [iid.bytes for iid in exclude_ids]
        as_of_iso = iso(as_of) if as_of is not None else None
        attempts = 0
        max_attempts = 4
        fetch_multiplier = candidate_multiplier
        filtered: list[tuple[UUID, str, float]] = []
        while True:
            hits = self._vector_index.search(
                self._connect(),
                query_vec,
                kind=ItemKind.MEMORY_ITEM.value,
                model=model,
                rebuild_sql=_INDEX_REBUILD_SQL["memory_item"],
                levels=level_values,
                exclude_ids=excl_bytes,
                include_cold=include_cold,
                k=k * fetch_multiplier,
            )
            if not hits:
                return []
            ids = [u for u, _, _ in hits]
            placeholders = ",".join("?" for _ in ids)
            if as_of is None:
                # Default: current-state. Exclude any row that has been
                # invalidated, regardless of when.
                sql = (
                    "SELECT id, content FROM memory_items "  # noqa: S608
                    f"WHERE id IN ({placeholders}) "
                    "AND invalidated_at IS NULL"
                )
                params: list[Any] = [u.bytes for u in ids]
            else:
                # As-of mode: surface items whose validity covers `as_of`.
                sql = (
                    "SELECT id, content FROM memory_items "  # noqa: S608
                    f"WHERE id IN ({placeholders}) "
                    "  AND (valid_from IS NULL OR valid_from <= ?) "
                    "  AND (valid_until IS NULL OR valid_until > ?) "
                    "  AND (invalidated_at IS NULL OR invalidated_at > ?)"
                )
                params = [*(u.bytes for u in ids), as_of_iso, as_of_iso, as_of_iso]
            rows = self._connect().execute(sql, params).fetchall()
            content: dict[bytes, str] = {bytes(r["id"]): r["content"] for r in rows}
            # Preserve vector-search ordering for items that passed the filter.
            filtered = [
                (u, content[u.bytes], score)
                for u, _, score in hits
                if u.bytes in content
            ]
            attempts += 1
            # Exit when we have k results, when we've already over-fetched
            # the whole shard (further expansion can't help), or after
            # the bounded retry budget.
            if (
                len(filtered) >= k
                or len(hits) < k * fetch_multiplier
                or attempts >= max_attempts
            ):
                break
            fetch_multiplier *= 2
        return filtered[:k]

    # --- conflicts (Stage 8) -----------------------------------------------

    def record_conflict(self, conflict: Conflict) -> None:
        # Re-validate the cross-column invariants that the Pydantic model
        # enforces on construction.  Storage callers can build a Conflict
        # via `model_validate({...})` from external JSON which would
        # produce a row that an OPEN status with resolution/winner set —
        # the schema CHECK constraints don't span columns so a malformed
        # caller bypasses Pydantic's _check_status_invariants.
        is_open = conflict.status == ConflictStatus.OPEN
        has_resolution = (
            conflict.resolution is not None
            or conflict.resolved_winner_id is not None
            or conflict.resolved_at is not None
        )
        if is_open and has_resolution:
            raise ValueError(
                "open conflict must not carry resolution / resolved_winner_id / resolved_at"
            )
        if not is_open and not has_resolution:
            raise ValueError(
                f"{conflict.status.value} conflict must carry resolution + resolved_at"
            )
        # NOTE: the conflicts table has UNIQUE(source_item_id,
        # target_item_id) — directional.  The contradiction detector
        # emitting (A, B) on one pass and (B, A) on another can produce
        # two distinct rows for the same logical pair.  Callers
        # interested in dedup-by-pair should canonicalize at the
        # detection layer (sort the ids lexicographically before
        # recording) or query both directions when reading.  Cross-
        # checking at the storage boundary is too invasive given
        # existing callers; deferred to a future schema migration that
        # changes the unique constraint to (LEAST(a, b), GREATEST(a, b))
        # or a content-hash key.
        self._connect().execute(
            "INSERT INTO conflicts "
            "(id, source_item_id, target_item_id, similarity, verdict, "
            " status, resolution, resolved_winner_id, resolved_at, "
            " detected_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conflict.id.bytes,
                conflict.source_item_id.bytes,
                conflict.target_item_id.bytes,
                conflict.similarity,
                conflict.verdict.value,
                conflict.status.value,
                conflict.resolution.value if conflict.resolution else None,
                conflict.resolved_winner_id.bytes if conflict.resolved_winner_id else None,
                iso(conflict.resolved_at) if conflict.resolved_at else None,
                iso(conflict.detected_at),
            ),
        )

    def get_conflict(self, conflict_id: UUID) -> Conflict | None:
        row = (
            self._connect()
            .execute("SELECT * FROM conflicts WHERE id = ?", (conflict_id.bytes,))
            .fetchone()
        )
        return _row_to_conflict(row) if row is not None else None

    def list_conflicts(
        self,
        *,
        status: ConflictStatus | None = None,
        memory_item_id: UUID | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[Conflict]:
        """List conflicts with optional filters.

        `tenant_id` filters to conflicts whose participating memory items
        both belong to that tenant.  The conflicts table itself doesn't
        carry tenant_id (Stage 9 deferred a schema migration), so we join
        through memory_items twice — once per side — and intersect.
        Untagged tenant_id (None) returns conflicts across all tenants,
        same as the pre-fix behavior.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        sql = "SELECT * FROM conflicts WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        if memory_item_id is not None:
            sql += " AND (source_item_id = ? OR target_item_id = ?)"
            params.extend((memory_item_id.bytes, memory_item_id.bytes))
        if tenant_id is not None:
            # Both sides must match the tenant.  Subselect rather than
            # JOIN so the source/target filters above stay simple.
            sql += (
                " AND EXISTS (SELECT 1 FROM memory_items mi_s "
                "             WHERE mi_s.id = conflicts.source_item_id "
                "               AND mi_s.tenant_id = ?)"
                " AND EXISTS (SELECT 1 FROM memory_items mi_t "
                "             WHERE mi_t.id = conflicts.target_item_id "
                "               AND mi_t.tenant_id = ?)"
            )
            params.extend((tenant_id, tenant_id))
        sql += " ORDER BY detected_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._connect().execute(sql, params).fetchall()
        return [_row_to_conflict(r) for r in rows]

    def resolve_conflict(
        self,
        conflict_id: UUID,
        *,
        resolution: Resolution,
        resolved_winner_id: UUID | None,
        resolved_at: datetime,
    ) -> Conflict:
        # Wrap the read-validate-UPDATE in a transaction (BEGIN IMMEDIATE
        # grabs the writer lock at the start, see `transaction()`).  The
        # UPDATE itself is guarded by `WHERE status = 'open'` so a racing
        # worker that already resolved this conflict can't be silently
        # overwritten — we detect rowcount == 0 and raise.  The earlier
        # code read-then-UPDATEd without a transaction or status guard,
        # letting two concurrent resolvers both pass the status check and
        # both write, with the second silently winning.
        with self.transaction():
            existing = self.get_conflict(conflict_id)
            if existing is None:
                raise KeyError(f"conflict {conflict_id} not found")
            if existing.status is ConflictStatus.RESOLVED:
                raise RuntimeError(
                    f"conflict {conflict_id} is already resolved "
                    f"(resolution={existing.resolution})"
                )
            # KEEP_BOTH and MERGE legitimately leave the winner field NULL.
            if (
                resolution not in (Resolution.KEEP_BOTH, Resolution.MERGE)
                and resolved_winner_id is None
            ):
                raise ValueError(
                    f"resolution={resolution.value} requires resolved_winner_id"
                )
            if resolved_winner_id is not None and resolved_winner_id not in (
                existing.source_item_id,
                existing.target_item_id,
            ):
                raise ValueError(
                    "resolved_winner_id must equal source_item_id or target_item_id"
                )
            cursor = self._connect().execute(
                "UPDATE conflicts SET status = 'resolved', resolution = ?, "
                "resolved_winner_id = ?, resolved_at = ? "
                "WHERE id = ? AND status = 'open'",
                (
                    resolution.value,
                    resolved_winner_id.bytes if resolved_winner_id is not None else None,
                    iso(resolved_at),
                    conflict_id.bytes,
                ),
            )
            if cursor.rowcount != 1:
                # Status guard caught a concurrent resolver that beat us
                # to the UPDATE.  We already loaded `existing` and saw
                # OPEN, so a rowcount of 0 here means another writer
                # flipped it under us.
                raise RuntimeError(
                    f"conflict {conflict_id} was resolved by a concurrent writer"
                )
            result = self.get_conflict(conflict_id)
        if result is None:  # pragma: no cover - raced delete
            raise KeyError(conflict_id)
        return result

    def count_conflicts(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM conflicts").fetchone()[0])

    def count_conflicts_by_status(self) -> dict[ConflictStatus, int]:
        rows = (
            self._connect()
            .execute("SELECT status, COUNT(*) AS n FROM conflicts GROUP BY status")
            .fetchall()
        )
        result: dict[ConflictStatus, int] = dict.fromkeys(ConflictStatus, 0)
        for row in rows:
            result[ConflictStatus(row["status"])] = int(row["n"])
        return result

    # --- procedures ---------------------------------------------------------

    def insert_procedure(self, procedure: Procedure) -> None:
        self._connect().execute(
            "INSERT INTO procedures "
            "(id, situation, action, outcome, weight, metadata, "
            " last_decayed_at, created_at, updated_at, tenant_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                procedure.id.bytes,
                procedure.situation,
                procedure.action,
                procedure.outcome.value,
                procedure.weight,
                dumps_metadata(procedure.metadata),
                iso(procedure.created_at),
                iso(procedure.created_at),
                iso(procedure.updated_at),
                procedure.tenant_id,
            ),
        )

    def get_procedure(self, procedure_id: UUID) -> Procedure | None:
        row = (
            self._connect()
            .execute(
                "SELECT id, situation, action, outcome, weight, metadata, "
                "       created_at, updated_at, tenant_id "
                "FROM procedures WHERE id = ?",
                (procedure_id.bytes,),
            )
            .fetchone()
        )
        return _row_to_procedure(row) if row is not None else None

    def list_procedures(
        self,
        *,
        outcome: Outcome | None = None,
        limit: int = 100,
    ) -> list[Procedure]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        sql = (
            "SELECT id, situation, action, outcome, weight, metadata, "
            "       created_at, updated_at, tenant_id "
            "FROM procedures"
        )
        params: list[Any] = []
        if outcome is not None:
            sql += " WHERE outcome = ?"
            params.append(outcome.value)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._connect().execute(sql, params).fetchall()
        return [_row_to_procedure(r) for r in rows]

    def update_procedure_outcome(self, procedure_id: UUID, outcome: Outcome) -> None:
        cursor = self._connect().execute(
            "UPDATE procedures SET outcome = ?, updated_at = ? WHERE id = ?",
            (outcome.value, iso(datetime.now(tz=timezone.utc)), procedure_id.bytes),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"procedure {procedure_id} not found")
        # The outcome change is part of the procedure's "level" slot in
        # the vector index (so callers can filter by outcome). Invalidate.
        self._vector_index.mark_dirty(kind=ItemKind.PROCEDURE.value)

    def count_procedures(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM procedures").fetchone()[0])

    def count_procedures_by_outcome(self) -> dict[Outcome, int]:
        rows = (
            self._connect()
            .execute("SELECT outcome, COUNT(*) AS n FROM procedures GROUP BY outcome")
            .fetchall()
        )
        result: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
        for row in rows:
            result[Outcome(row["outcome"])] = int(row["n"])
        return result

    # --- embeddings ---------------------------------------------------------

    def insert_embedding(self, embedding: Embedding) -> None:
        self._connect().execute(
            "INSERT INTO embeddings "
            "(id, item_id, item_kind, model, dim, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                embedding.id.bytes,
                embedding.item_id.bytes,
                embedding.item_kind.value,
                embedding.model,
                embedding.dim,
                pack_vector(embedding.vector),
                iso(embedding.created_at),
            ),
        )
        self._vector_index.mark_dirty(kind=embedding.item_kind.value, model=embedding.model)

    def get_embedding(self, item_id: UUID, item_kind: ItemKind, model: str) -> Embedding | None:
        row = (
            self._connect()
            .execute(
                "SELECT * FROM embeddings WHERE item_id = ? AND item_kind = ? AND model = ?",
                (item_id.bytes, item_kind.value, model),
            )
            .fetchone()
        )
        return _row_to_embedding(row) if row is not None else None

    def count_embeddings(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    # --- provenance ---------------------------------------------------------

    def link_provenance(
        self, memory_item_id: UUID, event_id: UUID, weight: float = 1.0
    ) -> ProvenanceLink:
        link = ProvenanceLink(
            memory_item_id=memory_item_id,
            event_id=event_id,
            weight=weight,
        )
        self._connect().execute(
            "INSERT INTO provenance_links "
            "(id, memory_item_id, event_id, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                link.id.bytes,
                link.memory_item_id.bytes,
                link.event_id.bytes,
                link.weight,
                iso(link.created_at),
            ),
        )
        return link

    def get_supporting_events(self, memory_item_id: UUID) -> list[Event]:
        rows = (
            self._connect()
            .execute(
                "SELECT events.* FROM events "
                "JOIN provenance_links ON provenance_links.event_id = events.id "
                "WHERE provenance_links.memory_item_id = ? "
                "ORDER BY provenance_links.weight DESC, events.created_at DESC",
                (memory_item_id.bytes,),
            )
            .fetchall()
        )
        return [_row_to_event(r) for r in rows]

    def get_supported_memory_items(self, event_id: UUID) -> list[MemoryItem]:
        rows = (
            self._connect()
            .execute(
                "SELECT memory_items.* FROM memory_items "
                "JOIN provenance_links ON provenance_links.memory_item_id = memory_items.id "
                "WHERE provenance_links.event_id = ? "
                "ORDER BY memory_items.weight DESC",
                (event_id.bytes,),
            )
            .fetchall()
        )
        return [_row_to_memory_item(r) for r in rows]

    def count_provenance_links(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM provenance_links").fetchone()[0])

    # --- clusters -----------------------------------------------------------

    def insert_cluster(self, cluster: Cluster) -> None:
        self._connect().execute(
            "INSERT INTO clusters (id, cohesion, created_at) VALUES (?, ?, ?)",
            (cluster.id.bytes, cluster.cohesion, iso(cluster.created_at)),
        )

    def get_cluster(self, cluster_id: UUID) -> Cluster | None:
        row = (
            self._connect()
            .execute("SELECT * FROM clusters WHERE id = ?", (cluster_id.bytes,))
            .fetchone()
        )
        return _row_to_cluster(row) if row is not None else None

    def count_clusters(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM clusters").fetchone()[0])

    # --- search -------------------------------------------------------------

    def search_event_embeddings(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        hits = self._vector_index.search(
            self._connect(),
            query_vec,
            kind=ItemKind.EVENT.value,
            model=model,
            rebuild_sql=_INDEX_REBUILD_SQL["event"],
            include_cold=include_cold,
            k=k,
        )
        return self._fetch_event_content(hits)

    def search_memory_item_embeddings(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
        levels: Sequence[Level] | None = None,
        exclude_ids: Sequence[UUID] = (),
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        level_values = [level.value for level in levels] if levels else None
        excl_bytes = [iid.bytes for iid in exclude_ids]
        hits = self._vector_index.search(
            self._connect(),
            query_vec,
            kind=ItemKind.MEMORY_ITEM.value,
            model=model,
            rebuild_sql=_INDEX_REBUILD_SQL["memory_item"],
            levels=level_values,
            exclude_ids=excl_bytes,
            include_cold=include_cold,
            k=k,
        )
        return self._fetch_memory_item_content(hits)

    def _fetch_event_content(
        self, hits: Sequence[tuple[UUID, int, float]]
    ) -> list[tuple[UUID, str, float]]:
        if not hits:
            return []
        ids = [u for u, _, _ in hits]
        placeholders = ",".join("?" for _ in ids)
        rows = (
            self._connect()
            .execute(
                f"SELECT id, content FROM events WHERE id IN ({placeholders})",  # noqa: S608
                [u.bytes for u in ids],
            )
            .fetchall()
        )
        content: dict[bytes, str] = {bytes(r["id"]): r["content"] for r in rows}
        return [(u, content[u.bytes], score) for u, _, score in hits if u.bytes in content]

    def _fetch_memory_item_content(
        self, hits: Sequence[tuple[UUID, int, float]]
    ) -> list[tuple[UUID, str, float]]:
        if not hits:
            return []
        ids = [u for u, _, _ in hits]
        placeholders = ",".join("?" for _ in ids)
        # Filter invalidated rows here.  The in-memory vector shard keeps
        # them (so search_memory_item_embeddings_as_of can look them up
        # historically) and the non-as_of caller drops them at content-
        # fetch time.  Without this, the non-as_of search variant would
        # surface items that have already been retired by reconciliation.
        rows = (
            self._connect()
            .execute(
                f"SELECT id, content FROM memory_items "  # noqa: S608
                f"WHERE invalidated_at IS NULL AND id IN ({placeholders})",
                [u.bytes for u in ids],
            )
            .fetchall()
        )
        content: dict[bytes, str] = {bytes(r["id"]): r["content"] for r in rows}
        return [(u, content[u.bytes], score) for u, _, score in hits if u.bytes in content]

    def search_procedure_embeddings(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
        outcomes: Sequence[Outcome] | None = None,
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        outcome_values = [o.value for o in outcomes] if outcomes else None
        hits = self._vector_index.search(
            self._connect(),
            query_vec,
            kind=ItemKind.PROCEDURE.value,
            model=model,
            rebuild_sql=_INDEX_REBUILD_SQL["procedure"],
            levels=outcome_values,
            include_cold=include_cold,
            k=k,
        )
        return self._fetch_procedure_situation(hits)

    def _fetch_procedure_situation(
        self, hits: Sequence[tuple[UUID, int, float]]
    ) -> list[tuple[UUID, str, float]]:
        """Fetch the `situation` text for top-k procedure hits.

        The retrieve_procedures Memory method needs the full Procedure
        row, but storage.search_*_embeddings keeps the
        `(item_id, content, score)` shape across kinds. For procedures
        we return the situation as the `content` so the surface stays
        uniform; callers fetch the full Procedure separately by id.
        """
        if not hits:
            return []
        ids = [u for u, _, _ in hits]
        placeholders = ",".join("?" for _ in ids)
        rows = (
            self._connect()
            .execute(
                f"SELECT id, situation FROM procedures WHERE id IN ({placeholders})",  # noqa: S608
                [u.bytes for u in ids],
            )
            .fetchall()
        )
        content: dict[bytes, str] = {bytes(r["id"]): r["situation"] for r in rows}
        return [(u, content[u.bytes], score) for u, _, score in hits if u.bytes in content]

    def bm25_search_events(
        self,
        query: str,
        *,
        k: int,
        k1: float = 1.5,
        b: float = 0.75,
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        """Lexical top-k events via BM25 over their `content` field.

        Lazy-builds an in-memory inverted index over (id, content) on
        the first call and on every event corpus change
        (insert/cold/unmark/delete) or hyperparameter change (k1, b).
        Returns `(id, content, score)` triples sorted by score desc.
        Empty corpus or empty query -> [].

        BM25 is corpus-relative -- the same document scored against
        the same query may differ across haystacks. That's the point:
        the LongMemEval cleaned dataset has one fresh haystack per
        question, so BM25 weights are calibrated to that haystack's
        token statistics.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        rebuild = (
            self._bm25_events is None
            or self._bm25_events_dirty
            or self._bm25_k1 != k1
            or self._bm25_b != b
            or self._bm25_include_cold != include_cold
        )
        if rebuild:
            self._rebuild_bm25_events(k1=k1, b=b, include_cold=include_cold)
        assert self._bm25_events is not None
        if not query.strip() or len(self._bm25_events) == 0:
            return []
        hits = self._bm25_events.search(query, k=k)
        if not hits:
            return []
        # Fetch content for the returned ids in a single SQL round-trip,
        # then preserve BM25 ordering when assembling the response.
        ids = [doc_id for doc_id, _ in hits]
        placeholders = ",".join("?" for _ in ids)
        rows = (
            self._connect()
            .execute(
                f"SELECT id, content FROM events WHERE id IN ({placeholders})",  # noqa: S608
                [u.bytes for u in ids],
            )
            .fetchall()
        )
        content: dict[bytes, str] = {bytes(r["id"]): r["content"] for r in rows}
        return [
            (doc_id, content[doc_id.bytes], score)
            for doc_id, score in hits
            if doc_id.bytes in content
        ]

    def _rebuild_bm25_events(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        include_cold: bool = False,
    ) -> None:
        """Materialize the BM25 index from current event content.

        Reads (id, content) tuples for the active event corpus and
        feeds them into a fresh `BM25Index`. The cold filter mirrors
        the dense path: cold events are excluded by default so a
        single-flag retrieve call sees a consistent surface across
        lexical and dense rankings.
        """
        sql = "SELECT id, content FROM events"
        if not include_cold:
            sql += " WHERE cold_at IS NULL"
        # Insertion order = (created_at, id) is not guaranteed without an
        # explicit ORDER BY, but BM25 doesn't depend on insert order --
        # the inverted index is a hash, and tie-breaks fall back to
        # doc_idx (= the natural insertion order from this scan). Any
        # stable scan is fine.
        rows = self._connect().execute(sql).fetchall()
        index: BM25Index[UUID] = BM25Index(k1=k1, b=b)
        for row in rows:
            index.add_doc(UUID(bytes=row["id"]), row["content"])
        self._bm25_events = index
        self._bm25_k1 = k1
        self._bm25_b = b
        self._bm25_include_cold = include_cold
        self._bm25_events_dirty = False

    def get_embeddings_batch(
        self,
        items: Sequence[tuple[UUID, ItemKind]],
        *,
        model: str,
    ) -> dict[UUID, list[float]]:
        """Batch-fetch embedding vectors for the given (id, kind) pairs.

        One SQL round-trip per distinct ItemKind. Returns a dict mapping
        item UUID to the unpacked float list. Missing items are simply
        absent from the dict -- the caller decides what "missing" means
        (MMR treats it as no diversity pressure, the recency boost
        treats it as no recency contribution).
        """
        if not items:
            return {}
        # Bucket by kind so each SQL is type-monomorphic.
        by_kind: dict[ItemKind, list[bytes]] = {}
        for item_id, kind in items:
            by_kind.setdefault(kind, []).append(item_id.bytes)
        out: dict[UUID, list[float]] = {}
        for kind, id_bytes_list in by_kind.items():
            placeholders = ",".join("?" for _ in id_bytes_list)
            sql = (
                f"SELECT item_id, vector, dim FROM embeddings "  # noqa: S608
                f"WHERE model = ? AND item_kind = ? AND item_id IN ({placeholders})"
            )
            rows = self._connect().execute(
                sql, (model, kind.value, *id_bytes_list)
            ).fetchall()
            for row in rows:
                uid = UUID(bytes=row["item_id"])
                out[uid] = list(unpack_vector(row["vector"], int(row["dim"])))
        return out

    def get_created_at_batch(
        self,
        items: Sequence[tuple[UUID, ItemKind]],
    ) -> dict[UUID, datetime]:
        """Batch-fetch `created_at` for the given (id, kind) pairs.

        One SQL round-trip per distinct ItemKind (events / memory_items
        / procedures). The recency-boost path benefits most -- the
        previous one-SQL-per-candidate code did N round trips per
        retrieve. Missing items are absent from the returned dict.
        """
        if not items:
            return {}
        by_kind: dict[ItemKind, list[bytes]] = {}
        for item_id, kind in items:
            by_kind.setdefault(kind, []).append(item_id.bytes)
        out: dict[UUID, datetime] = {}
        for kind, id_bytes_list in by_kind.items():
            table = _DECAY_TABLES.get(kind)
            if table is None:
                continue
            placeholders = ",".join("?" for _ in id_bytes_list)
            sql = (
                f"SELECT id, created_at FROM {table} "  # noqa: S608
                f"WHERE id IN ({placeholders})"
            )
            rows = self._connect().execute(sql, id_bytes_list).fetchall()
            for row in rows:
                out[UUID(bytes=row["id"])] = parse_iso(row["created_at"])
        return out

    def list_recent_events(
        self,
        *,
        k: int,
        include_cold: bool = False,
    ) -> list[tuple[UUID, str]]:
        """Top-K events by `created_at` desc. Cheap candidate source
        for the recent-window hybrid retrieval path. Cold events are
        excluded by default. Returns `(id, content)` so callers can
        plug straight into a fusion ranking.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        sql = "SELECT id, content FROM events"
        if not include_cold:
            sql += " WHERE cold_at IS NULL"
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        rows = self._connect().execute(sql, (k,)).fetchall()
        return [(UUID(bytes=row["id"]), row["content"]) for row in rows]

    def score_events_by_ids(
        self,
        query_vec: Sequence[float],
        event_ids: Sequence[UUID],
        *,
        model: str,
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        ids_list = list(event_ids)
        if not ids_list:
            return []
        # `id IN (...)` with up to ~1000 ids fits within sqlite's default
        # 32k variable limit; the drill path emits at most a few hundred,
        # so we stay well clear. This path is small enough to bypass the
        # vector index cache and just read its candidates inline.
        placeholders = ",".join("?" for _ in ids_list)
        sql = (
            "SELECT e.id AS event_id, e.content AS content, "
            "       emb.vector AS vector, emb.dim AS dim "
            "FROM embeddings emb "
            "JOIN events e ON emb.item_id = e.id "
            "WHERE emb.item_kind = 'event' AND emb.model = ?"
        )
        sql += f" AND e.id IN ({placeholders})"
        if not include_cold:
            sql += " AND e.cold_at IS NULL"
        params: list[Any] = [model, *(eid.bytes for eid in ids_list)]
        rows = self._connect().execute(sql, params).fetchall()
        if not rows:
            return []
        dim = int(rows[0]["dim"])
        if len(query_vec) != dim:
            raise ValueError(
                f"query_vec dim {len(query_vec)} does not match stored embedding dim {dim}"
            )
        raw = b"".join(row["vector"] for row in rows)
        vecs = np.frombuffer(raw, dtype=np.float32, count=len(rows) * dim).reshape(len(rows), dim)
        q = np.asarray(query_vec, dtype=np.float32)
        scores = vecs @ q
        order = np.argsort(-scores, kind="stable")
        return [
            (
                UUID(bytes=rows[i]["event_id"]),
                str(rows[i]["content"]),
                float(scores[i]),
            )
            for i in order
        ]

    # --- decay state --------------------------------------------------------

    def get_decay_state(self, item_id: UUID, kind: ItemKind) -> DecayState | None:
        sql = _GET_DECAY_STATE_SQL[kind]
        row = self._connect().execute(sql, (item_id.bytes,)).fetchone()
        return _row_to_decay_state(row, kind) if row is not None else None

    def iter_decay_states(
        self,
        kind: ItemKind,
        *,
        include_cold: bool = False,
        batch_size: int = 1000,
    ) -> Iterator[DecayState]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        sql = _ITER_DECAY_STATES_ALL_SQL[kind] if include_cold else _ITER_DECAY_STATES_HOT_SQL[kind]
        cursor = self._connect().execute(sql)
        try:
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    return
                for row in rows:
                    yield _row_to_decay_state(row, kind)
        finally:
            cursor.close()

    def update_decay_state(self, state: DecayState) -> None:
        sql = _UPDATE_DECAY_STATE_SQL[state.item_kind]
        params: tuple[Any, ...] = (
            state.weight,
            state.reinforcement_count,
            state.corroboration_count,
            state.contradiction_count,
            iso(state.last_decayed_at),
            iso(state.cold_at) if state.cold_at is not None else None,
        )
        # memory_items and procedures also carry `updated_at`; bump it
        # so audit logs reflect the decay-state write.  Events have no
        # `updated_at` column.
        if state.item_kind is not ItemKind.EVENT:
            params = (*params, iso(state.last_decayed_at))
        params = (*params, state.item_id.bytes)
        cursor = self._connect().execute(sql, params)
        if cursor.rowcount == 0:
            raise KeyError(f"{state.item_kind.value} {state.item_id} not found")

    def mark_cold(self, item_id: UUID, kind: ItemKind, *, at: datetime) -> None:
        sql = _MARK_COLD_SQL[kind]
        cursor = self._connect().execute(sql, (iso(at), item_id.bytes))
        if cursor.rowcount == 0:
            raise KeyError(f"{kind.value} {item_id} not found")
        self._vector_index.mark_dirty(kind=kind.value)
        if kind is ItemKind.EVENT:
            self._bm25_events_dirty = True

    def unmark_cold(self, item_id: UUID, kind: ItemKind) -> None:
        sql = _UNMARK_COLD_SQL[kind]
        cursor = self._connect().execute(sql, (item_id.bytes,))
        if cursor.rowcount == 0:
            raise KeyError(f"{kind.value} {item_id} not found")
        self._vector_index.mark_dirty(kind=kind.value)
        if kind is ItemKind.EVENT:
            self._bm25_events_dirty = True

    def count_cold(self, kind: ItemKind) -> int:
        sql = _COUNT_COLD_SQL[kind]
        return int(self._connect().execute(sql).fetchone()[0])

    def delete_cold_items(self, kind: ItemKind) -> int:
        # For events, refuse to delete rows that participate in provenance
        # links - a foreign key with ON DELETE RESTRICT would raise a generic
        # IntegrityError; we'd rather give the caller an actionable message.
        #
        # The provenance-count check + DELETE run in a single transaction
        # (BEGIN IMMEDIATE grabs the writer lock).  Without it, a second
        # writer could insert a provenance_links row pointing at a cold
        # event between our check and our DELETE — the DELETE would then
        # fail with a bare ON DELETE RESTRICT IntegrityError instead of
        # our typed ProvenanceProtectedError, *and* it would have left
        # the other rows in the cold set already deleted.
        with self.transaction():
            if kind is ItemKind.EVENT:
                blockers = (
                    self._connect()
                    .execute(
                        "SELECT COUNT(*) FROM events e "
                        "JOIN provenance_links p ON p.event_id = e.id "
                        "WHERE e.cold_at IS NOT NULL"
                    )
                    .fetchone()[0]
                )
                if blockers:
                    raise ProvenanceProtectedError(
                        f"cannot delete {blockers} cold event(s) with provenance links; "
                        "use the 'cold' prune policy instead"
                    )
            sql = _DELETE_COLD_SQL[kind]
            cursor = self._connect().execute(sql)
            deleted = int(cursor.rowcount)
        if deleted:
            self._vector_index.mark_dirty(kind=kind.value)
            # BM25 covers events only; if we just dropped cold event rows
            # the cached inverted index now points at ids that the
            # post-fetch SQL will silently drop, returning fewer than k
            # hits with no error to the caller.  Invalidate so the next
            # search rebuilds against the surviving corpus.
            if kind is ItemKind.EVENT:
                self._bm25_events_dirty = True
        return deleted

    def decay_totals(self, kind: ItemKind) -> dict[str, int]:
        sql = _DECAY_TOTALS_SQL[kind]
        row = self._connect().execute(sql).fetchone()
        # SUM over an empty table returns NULL; COALESCE in the SQL takes
        # care of the *_total fields, but the bare CASE-SUM hot/cold
        # gauges fall through as None when there are no rows at all.
        return {
            "hot_items": int(row["hot_items"] or 0),
            "cold_items": int(row["cold_items"] or 0),
            "reinforcement_total": int(row["reinforcement_total"] or 0),
            "corroboration_total": int(row["corroboration_total"] or 0),
            "contradiction_total": int(row["contradiction_total"] or 0),
        }

    # --- consolidation helpers ---------------------------------------------

    def iter_unconsolidated_events_with_embeddings(
        self,
        *,
        model: str,
        limit: int | None = None,
        batch_size: int = 256,
    ) -> Iterator[tuple[Event, list[float]]]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")

        sql = (
            "SELECT e.id AS id, e.content AS content, e.metadata AS metadata, "
            "       e.source AS source, e.created_at AS created_at, "
            "       e.tenant_id AS tenant_id, "
            "       emb.vector AS vector, emb.dim AS dim "
            "FROM events e "
            "JOIN embeddings emb ON emb.item_id = e.id AND emb.item_kind = 'event' "
            "WHERE emb.model = ? "
            "  AND e.cold_at IS NULL "
            "  AND NOT EXISTS (SELECT 1 FROM provenance_links p WHERE p.event_id = e.id) "
            "ORDER BY e.created_at ASC, e.id ASC"
        )
        params: tuple[Any, ...] = (model,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (model, limit)
        cursor = self._connect().execute(sql, params)
        try:
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    return
                for row in rows:
                    # Propagate tenant_id so consolidation produces tenant-
                    # scoped MemoryItems instead of silently dropping the
                    # tag and landing every consolidated row as global.
                    event = Event(
                        id=UUID(bytes=row["id"]),
                        content=row["content"],
                        metadata=loads_metadata(row["metadata"]),
                        source=row["source"],
                        created_at=parse_iso(row["created_at"]),
                        tenant_id=row["tenant_id"],
                    )
                    dim = int(row["dim"])
                    vec = list(unpack_vector(row["vector"], dim))
                    yield event, vec
        finally:
            cursor.close()

    def insert_memory_item_with_provenance(
        self,
        item: MemoryItem,
        supporting_event_ids: Sequence[UUID],
        *,
        cluster: Cluster | None = None,
        embedding: Embedding | None = None,
        provenance_weights: Mapping[UUID, float] | None = None,
    ) -> list[ProvenanceLink]:
        if item.level is not Level.EVENT and not supporting_event_ids:
            raise ValueError(
                f"memory item at level={item.level.value} requires at least one "
                "supporting event id; none given"
            )
        if embedding is not None and embedding.item_id != item.id:
            raise ValueError(
                f"embedding.item_id {embedding.item_id} does not match item.id {item.id}"
            )
        weights = dict(provenance_weights) if provenance_weights else {}
        links: list[ProvenanceLink] = []
        with self.transaction():
            if cluster is not None:
                self.insert_cluster(cluster)
            self.insert_memory_item(item)
            if embedding is not None:
                self.insert_embedding(embedding)
            for event_id in supporting_event_ids:
                weight = weights.get(event_id, 1.0)
                links.append(self.link_provenance(item.id, event_id, weight))
        return links
