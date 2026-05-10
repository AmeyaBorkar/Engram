"""In-memory vector index for fast top-k cosine similarity.

The SQLite path that materializes 100k+ embedding rows into a numpy
matrix is dominated by the per-row data transfer cost (Row object
construction, blob marshalling). At 100k items / dim=128 that's ~180 ms
just to fetch -- well over the Stage 6 P50 budget of 150 ms.

This module sits behind `SqliteStorage`'s search methods and caches
the (n, d) matrix in process memory, keyed by `(item_kind, model)`.
Cache mechanics:

  * Lazy build. On first `search`, the index runs one SELECT and
    materializes the matrix.
  * Dirty flag. Writes (`mark_dirty`) invalidate the cache; the next
    `search` call rebuilds.
  * Rebuild cost is paid once per write burst, not once per query.

The trade is simple: doubled memory (the matrix mirrors the embeddings
table) for ~50x faster retrieval at scale. Stage 9's Postgres backend
will swap this for `pgvector`; sqlite-vec is the natural Stage-N upgrade
once the extension is widely deployable.

NOT thread-safe by itself; `SqliteStorage` already serializes per-thread
connection usage and only one rebuild happens per dirty cycle (subsequent
threads wait on the lock and find a fresh cache).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.floating[Any]]


@dataclass(slots=True)
class _IndexShard:
    """One per (item_kind, model) pair."""

    matrix: FloatMatrix | None = None  # shape (n, d), float32
    ids: list[bytes] = field(default_factory=list)  # parallel; UUID bytes
    cold: npt.NDArray[np.bool_] | None = None  # shape (n,)
    levels: list[str] = field(default_factory=list)  # parallel; 'event'/'summary'/...
    dim: int = 0
    dirty: bool = True


class VectorIndex:
    """Per-storage in-memory cache for cosine top-k.

    Pure cache -- the source of truth stays in SQLite. The class never
    holds a connection; callers pass it in on `search`. Mutations to the
    underlying tables go through `mark_dirty(kind, model)`; the next
    `search` triggers a rebuild from SQL.
    """

    def __init__(self) -> None:
        self._shards: dict[tuple[str, str], _IndexShard] = {}
        self._lock = threading.Lock()

    def _shard(self, kind: str, model: str) -> _IndexShard:
        key = (kind, model)
        shard = self._shards.get(key)
        if shard is None:
            shard = _IndexShard()
            self._shards[key] = shard
        return shard

    def mark_dirty(self, kind: str | None = None, model: str | None = None) -> None:
        """Invalidate matching shards. Both args None -> mark every shard."""
        with self._lock:
            for (k, m), shard in self._shards.items():
                if kind is not None and k != kind:
                    continue
                if model is not None and m != model:
                    continue
                shard.dirty = True

    def search(
        self,
        conn: sqlite3.Connection,
        query_vec: Sequence[float],
        *,
        kind: str,
        model: str,
        rebuild_sql: str,
        levels: Sequence[str] | None = None,
        exclude_ids: Sequence[bytes] = (),
        include_cold: bool = False,
        k: int,
    ) -> list[tuple[UUID, int, float]]:
        """Return up to `k` (UUID, shard_idx, score) triples.

        The shard index lets the caller fetch content via a follow-up
        `SELECT ... WHERE id IN (...)` without re-running the matmul.

        `rebuild_sql` is the SELECT that materializes the shard. It MUST
        return four columns: `item_id BLOB`, `vector BLOB`, `cold INTEGER`
        (0/1), `level TEXT` (use `'event'` for the events shard). Bind
        parameter: `(model,)`.

        `levels` filters shard rows to a subset of level strings. `None`
        means any. `exclude_ids` is a small set of UUID byte blobs to
        skip (used by contradiction detection). `include_cold` lets cold
        items through (audit reads).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        with self._lock:
            shard = self._shard(kind, model)
            if shard.dirty or shard.matrix is None:
                _rebuild_shard(shard, conn, rebuild_sql, model)

        if shard.matrix is None or shard.matrix.shape[0] == 0:
            return []

        if len(query_vec) != shard.dim:
            raise ValueError(f"query_vec dim {len(query_vec)} does not match shard dim {shard.dim}")
        q = np.asarray(query_vec, dtype=np.float32)
        scores = shard.matrix @ q

        # Mask out filtered rows by setting their score to -inf.
        if not include_cold and shard.cold is not None:
            scores = np.where(shard.cold, -np.inf, scores)
        if levels is not None:
            level_set = set(levels)
            level_mask = np.asarray([lv in level_set for lv in shard.levels], dtype=np.bool_)
            scores = np.where(level_mask, scores, -np.inf)
        if exclude_ids:
            excl = set(exclude_ids)
            id_mask = np.asarray([iid in excl for iid in shard.ids], dtype=np.bool_)
            scores = np.where(id_mask, -np.inf, scores)

        n = scores.shape[0]
        k_eff = min(k, n)
        if k_eff == n:
            order = np.argsort(-scores, kind="stable")
        else:
            cand = np.argpartition(-scores, k_eff - 1)[:k_eff]
            order = cand[np.argsort(-scores[cand], kind="stable")]
        out: list[tuple[UUID, int, float]] = []
        for i in order:
            score = float(scores[i])
            if not np.isfinite(score):
                continue
            out.append((UUID(bytes=shard.ids[i]), int(i), score))
        return out


def _rebuild_shard(
    shard: _IndexShard,
    conn: sqlite3.Connection,
    sql: str,
    model: str,
) -> None:
    """One full materialization. Holds the shard lock; safe to call repeatedly."""
    rows = conn.execute(sql, (model,)).fetchall()
    if not rows:
        shard.matrix = np.zeros((0, max(shard.dim, 1)), dtype=np.float32)
        shard.ids = []
        shard.cold = np.zeros((0,), dtype=np.bool_)
        shard.levels = []
        shard.dirty = False
        return
    ids: list[bytes] = []
    cold_flags: list[bool] = []
    levels: list[str] = []
    chunks: list[bytes] = []
    for r in rows:
        ids.append(bytes(r["item_id"]))
        chunks.append(bytes(r["vector"]))
        cold_flags.append(bool(r["cold"]))
        levels.append(str(r["level"]))
    raw = b"".join(chunks)
    if shard.dim == 0:
        shard.dim = len(chunks[0]) // 4  # float32 = 4 bytes
    matrix = np.frombuffer(raw, dtype=np.float32, count=len(ids) * shard.dim).reshape(
        len(ids), shard.dim
    )
    shard.matrix = matrix
    shard.ids = ids
    shard.cold = np.asarray(cold_flags, dtype=np.bool_)
    shard.levels = levels
    shard.dirty = False
