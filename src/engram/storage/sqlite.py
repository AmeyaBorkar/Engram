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
    DecayState,
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
_UPDATE_DECAY_STATE_SQL: dict[ItemKind, str] = {
    kind: (
        f"UPDATE {table} SET weight = ?, reinforcement_count = ?, "  # noqa: S608
        "corroboration_count = ?, contradiction_count = ?, "
        "last_decayed_at = ?, cold_at = ? WHERE id = ?"
    )
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
        # `last_decayed_at` defaults to `created_at` so the first decay tick
        # has a sane dt. `weight` defaults to 1.0 via the column default;
        # we don't override it here because Stage 4 ships a single insert
        # path that always starts items hot.
        self._connect().execute(
            "INSERT INTO events "
            "(id, content, metadata, source, created_at, last_decayed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.id.bytes,
                event.content,
                dumps_metadata(event.metadata),
                event.source,
                iso(event.created_at),
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
                iso(e.created_at),
            )
            for e in events
        ]
        if not rows:
            return 0
        self._connect().executemany(
            "INSERT INTO events "
            "(id, content, metadata, source, created_at, last_decayed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
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
            "(id, level, content, weight, cluster_id, metadata, "
            "created_at, updated_at, last_decayed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.id.bytes,
                item.level.value,
                item.content,
                item.weight,
                item.cluster_id.bytes if item.cluster_id else None,
                dumps_metadata(item.metadata),
                iso(item.created_at),
                iso(item.updated_at),
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
                iso(i.updated_at),
            )
            for i in items
        ]
        if not rows:
            return 0
        self._connect().executemany(
            "INSERT INTO memory_items "
            "(id, level, content, weight, cluster_id, metadata, "
            "created_at, updated_at, last_decayed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def update_memory_item_level(self, item_id: UUID, level: Level) -> None:
        cursor = self._connect().execute(
            "UPDATE memory_items SET level = ?, updated_at = ? WHERE id = ?",
            (level.value, iso(datetime.now(tz=timezone.utc)), item_id.bytes),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"memory_item {item_id} not found")

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
        sql = (
            "SELECT e.id AS event_id, e.content AS content, "
            "       emb.vector AS vector, emb.dim AS dim "
            "FROM embeddings emb "
            "JOIN events e ON emb.item_id = e.id "
            "WHERE emb.item_kind = 'event' AND emb.model = ?"
        )
        if not include_cold:
            sql += " AND e.cold_at IS NULL"
        rows = self._connect().execute(sql, (model,)).fetchall()
        if not rows:
            return []

        dim = int(rows[0]["dim"])
        if len(query_vec) != dim:
            raise ValueError(
                f"query_vec dim {len(query_vec)} does not match stored embedding dim {dim}"
            )

        n = len(rows)
        vecs = np.empty((n, dim), dtype=np.float32)
        for i, row in enumerate(rows):
            vecs[i] = np.frombuffer(row["vector"], dtype=np.float32, count=dim)
        q = np.asarray(query_vec, dtype=np.float32)
        scores = vecs @ q  # cosine sim if both sides are unit-norm

        k_eff = min(k, n)
        if k_eff == n:
            order = np.argsort(-scores)
        else:
            cand = np.argpartition(-scores, k_eff - 1)[:k_eff]
            order = cand[np.argsort(-scores[cand])]

        return [
            (UUID(bytes=rows[i]["event_id"]), str(rows[i]["content"]), float(scores[i]))
            for i in order
        ]

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
        sql = (
            "SELECT mi.id AS item_id, mi.content AS content, "
            "       emb.vector AS vector, emb.dim AS dim "
            "FROM embeddings emb "
            "JOIN memory_items mi ON emb.item_id = mi.id "
            "WHERE emb.item_kind = 'memory_item' AND emb.model = ?"
        )
        params: list[Any] = [model]
        if not include_cold:
            sql += " AND mi.cold_at IS NULL"
        if levels:
            placeholders = ",".join("?" for _ in levels)
            sql += f" AND mi.level IN ({placeholders})"
            params.extend(level.value for level in levels)
        if exclude_ids:
            placeholders = ",".join("?" for _ in exclude_ids)
            sql += f" AND mi.id NOT IN ({placeholders})"
            params.extend(item_id.bytes for item_id in exclude_ids)
        rows = self._connect().execute(sql, params).fetchall()
        if not rows:
            return []

        dim = int(rows[0]["dim"])
        if len(query_vec) != dim:
            raise ValueError(
                f"query_vec dim {len(query_vec)} does not match stored embedding dim {dim}"
            )
        n = len(rows)
        vecs = np.empty((n, dim), dtype=np.float32)
        for i, row in enumerate(rows):
            vecs[i] = np.frombuffer(row["vector"], dtype=np.float32, count=dim)
        q = np.asarray(query_vec, dtype=np.float32)
        scores = vecs @ q

        k_eff = min(k, n)
        if k_eff == n:
            order = np.argsort(-scores)
        else:
            cand = np.argpartition(-scores, k_eff - 1)[:k_eff]
            order = cand[np.argsort(-scores[cand])]
        return [
            (UUID(bytes=rows[i]["item_id"]), str(rows[i]["content"]), float(scores[i]))
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
        cursor = self._connect().execute(
            sql,
            (
                state.weight,
                state.reinforcement_count,
                state.corroboration_count,
                state.contradiction_count,
                iso(state.last_decayed_at),
                iso(state.cold_at) if state.cold_at is not None else None,
                state.item_id.bytes,
            ),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"{state.item_kind.value} {state.item_id} not found")

    def mark_cold(self, item_id: UUID, kind: ItemKind, *, at: datetime) -> None:
        sql = _MARK_COLD_SQL[kind]
        cursor = self._connect().execute(sql, (iso(at), item_id.bytes))
        if cursor.rowcount == 0:
            raise KeyError(f"{kind.value} {item_id} not found")

    def unmark_cold(self, item_id: UUID, kind: ItemKind) -> None:
        sql = _UNMARK_COLD_SQL[kind]
        cursor = self._connect().execute(sql, (item_id.bytes,))
        if cursor.rowcount == 0:
            raise KeyError(f"{kind.value} {item_id} not found")

    def count_cold(self, kind: ItemKind) -> int:
        sql = _COUNT_COLD_SQL[kind]
        return int(self._connect().execute(sql).fetchone()[0])

    def delete_cold_items(self, kind: ItemKind) -> int:
        # For events, refuse to delete rows that participate in provenance
        # links - a foreign key with ON DELETE RESTRICT would raise a generic
        # IntegrityError; we'd rather give the caller an actionable message.
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
                raise RuntimeError(
                    f"cannot delete {blockers} cold event(s) with provenance links; "
                    "use the 'cold' prune policy instead"
                )
        sql = _DELETE_COLD_SQL[kind]
        cursor = self._connect().execute(sql)
        return int(cursor.rowcount)

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
                    event = Event(
                        id=UUID(bytes=row["id"]),
                        content=row["content"],
                        metadata=loads_metadata(row["metadata"]),
                        source=row["source"],
                        created_at=parse_iso(row["created_at"]),
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
