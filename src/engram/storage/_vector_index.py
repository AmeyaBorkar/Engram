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
table) for ~50x faster retrieval at scale.  A future Postgres backend
would swap this for `pgvector`, and `sqlite-vec` is a natural upgrade
once the extension is widely deployable — both are roadmap items, not
shipped.

Concurrency model
-----------------

`SqliteStorage` already serializes per-thread connection usage (each
thread has its own sqlite3.Connection), so `conn.execute` inside
`_rebuild_shard` is single-threaded against the connection.  But
multiple search threads share the in-memory matrix.

Two refinements over the original "single global lock" design:

  * Per-shard RLock.  Searches on the `event` shard don't block searches
    on the `memory_item` shard.  The lookup-to-shard mapping (the
    `_shards` dict) is guarded by an index-wide lock, but that lock is
    released the moment we hold a reference to the shard.  Rebuilds —
    250-500 ms at 100k rows — no longer hold the index-wide lock.

  * Rebuild-in-progress flag.  Two search threads that both find a
    dirty shard would both kick off a rebuild against the same SQL,
    burning 2x the CPU and possibly producing different matrix objects
    (the later one wins, but it discards a result the first reader was
    about to use).  A `rebuilding` flag is set under the shard RLock
    before the SQL fetch; a second arrival waits on a per-shard
    `Condition` until the first finishes, then re-checks dirty.
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


@dataclass
class _IndexShard:
    """One per (item_kind, model) pair.

    Carries its own RLock + Condition so unrelated shards don't block
    each other.  RLock for the same-thread re-entry case (a single
    rebuild path can call into `_level_mask` etc. inside the lock).
    """

    matrix: FloatMatrix | None = None  # shape (n, d), float32
    ids: list[bytes] = field(default_factory=list)  # parallel; UUID bytes
    cold: npt.NDArray[np.bool_] | None = None  # shape (n,)
    levels: list[str] = field(default_factory=list)  # parallel; 'event'/'summary'/...
    dim: int = 0
    dirty: bool = True
    # `level_masks[level]` is a parallel boolean array True at rows
    # whose level matches.  Populated lazily on first level-filtered
    # search and invalidated alongside `dirty=True`.  Without this,
    # each search rebuilt the mask as a Python list comprehension over
    # `levels` — O(n) Python ops per call on a 100k-row shard.
    level_masks: dict[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    id_to_idx: dict[bytes, int] = field(default_factory=dict)
    # `rebuilding` is True while a thread is running the SQL fetch +
    # matrix materialization.  A second thread finding `rebuilding`
    # waits on the Condition until the first finishes, then re-checks
    # `dirty` (a write that arrived after the rebuild started should
    # not be lost, but in practice the next caller's dirty check picks
    # it up on the following search).
    rebuilding: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _cond: threading.Condition | None = None

    def __post_init__(self) -> None:
        # Condition needs to share the same underlying lock so notify
        # under the RLock wakes waiters.  RLock + Condition cooperate
        # because Condition uses the lock's acquire/release.
        self._cond = threading.Condition(self._lock)


class VectorIndex:
    """Per-storage in-memory cache for cosine top-k.

    Pure cache -- the source of truth stays in SQLite. The class never
    holds a connection; callers pass it in on `search`. Mutations to the
    underlying tables go through `mark_dirty(kind, model)`; the next
    `search` triggers a rebuild from SQL.
    """

    def __init__(self) -> None:
        self._shards: dict[tuple[str, str], _IndexShard] = {}
        # Index-wide lock guards the `_shards` dict itself (insert /
        # lookup).  Held only long enough to grab the shard reference;
        # the shard's own RLock is what guards the data inside.
        self._lock = threading.Lock()

    def _shard(self, kind: str, model: str) -> _IndexShard:
        """Get-or-create the shard for `(kind, model)`.

        Index-wide lock guards the dict only; the returned shard has
        its own lock for the data.
        """
        key = (kind, model)
        with self._lock:
            shard = self._shards.get(key)
            if shard is None:
                shard = _IndexShard()
                self._shards[key] = shard
            return shard

    def mark_dirty(self, kind: str | None = None, model: str | None = None) -> None:
        """Invalidate matching shards. Both args None -> mark every shard."""
        # Snapshot the matching shards under the index-wide lock so we
        # don't iterate while a concurrent `_shard()` adds a new key.
        with self._lock:
            targets = [
                shard
                for (k, m), shard in self._shards.items()
                if (kind is None or k == kind) and (model is None or m == model)
            ]
        for shard in targets:
            with shard._lock:
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
        shard = self._shard(kind, model)
        # Snapshot every attribute we'll touch under the shard lock so
        # a concurrent `_rebuild_shard` against the same shard (this
        # can only happen via `mark_dirty` flipping the flag mid-search,
        # then a different thread starting its own rebuild) cannot
        # replace `matrix` / `cold` / `levels` / `ids` underneath us
        # while we're mid-matmul.
        with shard._lock:
            self._ensure_built(shard, conn, rebuild_sql, model)
            matrix = shard.matrix
            cold = shard.cold
            levels_snapshot = shard.levels
            ids = shard.ids
            dim = shard.dim
            # Mask caches live on the shard but we resolve them inside
            # the lock so the rebuild that resets them can't race the
            # lookup that's about to consume them.
            level_mask = (
                _level_mask(shard, list(levels)) if levels is not None else None
            )
            id_mask = (
                _exclude_mask(shard, list(exclude_ids)) if exclude_ids else None
            )

        if matrix is None or matrix.shape[0] == 0:
            return []

        if len(query_vec) != dim:
            raise ValueError(
                f"query_vec dim {len(query_vec)} does not match shard dim {dim}"
            )
        q = np.asarray(query_vec, dtype=np.float32)
        scores = matrix @ q

        # Mask out filtered rows by setting their score to -inf.
        if not include_cold and cold is not None:
            scores = np.where(cold, -np.inf, scores)
        if level_mask is not None:
            scores = np.where(level_mask, scores, -np.inf)
        if id_mask is not None:
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
            out.append((UUID(bytes=ids[i]), int(i), score))
        return out

    def _ensure_built(
        self,
        shard: _IndexShard,
        conn: sqlite3.Connection,
        rebuild_sql: str,
        model: str,
    ) -> None:
        """Run a rebuild if needed, coordinating with concurrent rebuilders.

        Caller MUST hold `shard._lock`.  If another thread is already
        running a rebuild for this shard, this thread waits on the
        Condition until that rebuild completes (releasing the lock
        during the wait), then re-checks dirty.  The first thread to
        find dirty flips `rebuilding=True` and does the work; any
        thread blocked on the condition wakes up to a fresh cache.
        """
        # Re-entrant fast path: cache is fresh.
        if shard.matrix is not None and not shard.dirty:
            return
        assert shard._cond is not None
        # Wait out any in-progress rebuild before deciding to start
        # our own.  After the wait, re-check: the other thread's
        # rebuild may have already produced exactly what we wanted.
        while shard.rebuilding:
            shard._cond.wait()
            if shard.matrix is not None and not shard.dirty:
                return
        # No active rebuild and the shard is dirty (or never built).
        # Claim it and do the work; reset `rebuilding` + notify in a
        # try/finally so a SQL exception doesn't strand other waiters.
        shard.rebuilding = True
        try:
            _rebuild_shard(shard, conn, rebuild_sql, model)
        finally:
            shard.rebuilding = False
            shard._cond.notify_all()


def _rebuild_shard(
    shard: _IndexShard,
    conn: sqlite3.Connection,
    sql: str,
    model: str,
) -> None:
    """One full materialization. Caller holds `shard._lock`."""
    rows = conn.execute(sql, (model,)).fetchall()
    # Invalidate any cached masks regardless of whether we rebuild
    # from rows or end empty; the next search will rebuild on demand.
    shard.level_masks = {}
    shard.id_to_idx = {}
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


def _level_mask(shard: _IndexShard, levels: list[str]) -> npt.NDArray[np.bool_]:
    """Boolean mask over shard rows whose level is in `levels`.

    Per-level masks are memoized on the shard so a search with
    `levels=['summary']` doesn't rebuild the boolean from a Python list
    comprehension on every call.  For a 100k-row shard that previously
    cost 1-3ms per search; now it's a vectorized OR.
    """
    out = np.zeros(len(shard.levels), dtype=np.bool_)
    for level in levels:
        cached = shard.level_masks.get(level)
        if cached is None:
            cached = np.asarray(
                [lv == level for lv in shard.levels], dtype=np.bool_
            )
            shard.level_masks[level] = cached
        out |= cached
    return out


def _exclude_mask(
    shard: _IndexShard, exclude_ids: list[bytes]
) -> npt.NDArray[np.bool_]:
    """Boolean mask True at rows whose id is in `exclude_ids`."""
    if not shard.id_to_idx:
        shard.id_to_idx = {iid: i for i, iid in enumerate(shard.ids)}
    out = np.zeros(len(shard.ids), dtype=np.bool_)
    for iid in exclude_ids:
        idx = shard.id_to_idx.get(iid)
        if idx is not None:
            out[idx] = True
    return out
