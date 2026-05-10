"""SQLite storage backend.

WAL mode, foreign keys on, per-thread connections. Single-process, single-
machine — Stage 9 brings the multi-tenant Postgres backend against the same
protocol.

Threading model: each thread gets its own connection on first use. The
connection is closed when `SqliteStorage.close()` is called by *that* thread,
or when the storage is dropped. Cross-thread sharing of a connection is not
supported (and `check_same_thread` enforces this).
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID

from engram.schemas import (
    Cluster,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
    ProvenanceLink,
)
from engram.storage._serialize import (
    dumps_metadata,
    iso,
    loads_metadata,
    pack_vector,
    parse_iso,
    unpack_vector,
)
from engram.storage.migrations import apply_migrations


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=UUID(bytes=row["id"]),
        content=row["content"],
        metadata=loads_metadata(row["metadata"]),
        source=row["source"],
        created_at=parse_iso(row["created_at"]),
    )


def _row_to_memory_item(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=UUID(bytes=row["id"]),
        level=Level(row["level"]),
        content=row["content"],
        weight=row["weight"],
        cluster_id=UUID(bytes=row["cluster_id"]) if row["cluster_id"] else None,
        metadata=loads_metadata(row["metadata"]),
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
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


def _row_to_provenance_link(row: sqlite3.Row) -> ProvenanceLink:
    return ProvenanceLink(
        id=UUID(bytes=row["id"]),
        memory_item_id=UUID(bytes=row["memory_item_id"]),
        event_id=UUID(bytes=row["event_id"]),
        weight=row["weight"],
        created_at=parse_iso(row["created_at"]),
    )


class SqliteStorage:
    """SQLite-backed `Storage` implementation."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._initialized = False

    # --- lifecycle ----------------------------------------------------------

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
        if self._initialized:
            return
        apply_migrations(self._connect())
        self._initialized = True

    def close(self) -> None:
        with self._lock:
            for conn in list(self._connections.values()):
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._connections.clear()
            self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        tid = threading.get_ident()
        with self._lock:
            conn = self._connections.get(tid)
            if conn is not None:
                return conn
            conn = sqlite3.connect(
                self._path,
                isolation_level=None,
                check_same_thread=True,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            self._connections[tid] = conn
            return conn

    @contextmanager
    def transaction(self) -> Iterator[None]:
        conn = self._connect()
        if conn.in_transaction:
            yield
            return
        conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # --- events -------------------------------------------------------------

    def insert_event(self, event: Event) -> None:
        self._connect().execute(
            "INSERT INTO events (id, content, metadata, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                event.id.bytes,
                event.content,
                dumps_metadata(event.metadata),
                event.source,
                iso(event.created_at),
            ),
        )

    def insert_events(self, events: Iterable[Event]) -> int:
        rows = [
            (
                e.id.bytes,
                e.content,
                dumps_metadata(e.metadata),
                e.source,
                iso(e.created_at),
            )
            for e in events
        ]
        if not rows:
            return 0
        self._connect().executemany(
            "INSERT INTO events (id, content, metadata, source, created_at) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
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
            "(id, level, content, weight, cluster_id, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.id.bytes,
                item.level.value,
                item.content,
                item.weight,
                item.cluster_id.bytes if item.cluster_id else None,
                dumps_metadata(item.metadata),
                iso(item.created_at),
                iso(item.updated_at),
            ),
        )

    def insert_memory_items(self, items: Iterable[MemoryItem]) -> int:
        rows = [
            (
                i.id.bytes,
                i.level.value,
                i.content,
                i.weight,
                i.cluster_id.bytes if i.cluster_id else None,
                dumps_metadata(i.metadata),
                iso(i.created_at),
                iso(i.updated_at),
            )
            for i in items
        ]
        if not rows:
            return 0
        self._connect().executemany(
            "INSERT INTO memory_items "
            "(id, level, content, weight, cluster_id, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
        cursor = self._connect().execute(
            "UPDATE memory_items SET weight = ?, updated_at = ? WHERE id = ?",
            (weight, iso(datetime.now(tz=timezone.utc)), item_id.bytes),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"memory_item {item_id} not found")

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
