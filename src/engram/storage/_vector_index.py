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

Locking model:

  * `VectorIndex._lock` is the index-wide lock — held ONLY to find-or-
    create a shard reference and to walk the shard dict on `mark_dirty`.
    Released immediately after.  A slow shard rebuild does NOT block
    searches against other shards.
  * `_IndexShard.lock` is the per-shard RLock — held during rebuild and
    during the snapshot that `search` reads.  Concurrent searches against
    DIFFERENT shards proceed in parallel; concurrent searches against
    the SAME shard serialize on rebuild but the matmul itself runs
    outside the lock against an immutable numpy view (the snapshot).

Without per-shard locks, a 100k-row, dim=768 rebuild (~250-500ms) would
block every concurrent searcher across every shard — `mark_dirty(kind=
"event")` would still freeze a `search(kind="memory_item")` for the
duration of the next event-shard rebuild that follows it.
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
    # `level_masks[level]` is a parallel boolean array True at rows
    # whose level matches.  Populated lazily on first level-filtered
    # search and invalidated alongside `dirty=True`.  Without this,
    # each search rebuilt the mask as a Python list comprehension over
    # `levels` — O(n) Python ops per call on a 100k-row shard.
    level_masks: dict[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    id_to_idx: dict[bytes, int] = field(default_factory=dict)
    # Per-shard RLock so rebuilds against ONE shard don't block searches
    # against OTHERS.  Reentrant so a nested call inside the rebuild
    # path doesn't deadlock.  Initialized lazily because the dataclass
    # default would share one lock across shards under the standard
    # `field(default=threading.RLock())` evaluation rules — use
    # `field(default_factory=threading.RLock)` to give each shard its
    # own.
    lock: threading.RLock = field(default_factory=threading.RLock)


class VectorIndex:
    """Per-storage in-memory cache for cosine top-k.

    Pure cache -- the source of truth stays in SQLite. The class never
    holds a connection; callers pass it in on `search`. Mutations to the
    underlying tables go through `mark_dirty(kind, model)`; the next
    `search` triggers a rebuild from SQL.
    """

    def __init__(self) -> None:
        self._shards: dict[tuple[str, str], _IndexShard] = {}
        # Index-wide lock guards only the `_shards` dict (find-or-create,
        # walk on mark_dirty).  Held briefly; the slow per-shard rebuild
        # runs under the SHARD's lock, not this one.
        self._lock = threading.Lock()

    def _shard(self, kind: str, model: str) -> _IndexShard:
        """Get-or-create the shard for `(kind, model)`. Caller MUST hold `_lock`."""
        key = (kind, model)
        shard = self._shards.get(key)
        if shard is None:
            shard = _IndexShard()
            self._shards[key] = shard
        return shard

    def mark_dirty(self, kind: str | None = None, model: str | None = None) -> None:
        """Invalidate matching shards. Both args None -> mark every shard.

        Sets the dirty flag UNDER each shard's lock so a concurrent
        search either (a) sees dirty=True and rebuilds, or (b) has
        already snapshotted matrix/ids/cold/levels and runs to
        completion against the old corpus — never a torn view.
        """
        with self._lock:
            # Snapshot the shard list under the index-wide lock so we
            # can release it before reaching for per-shard locks (which
            # might be held by a long-running rebuild).
            targets = [
                shard
                for (k, m), shard in self._shards.items()
                if (kind is None or k == kind) and (model is None or m == model)
            ]
        for shard in targets:
            with shard.lock:
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
        # Find-or-create the shard reference under the index-wide lock,
        # then release it immediately.  A concurrent search against a
        # DIFFERENT (kind, model) shard can now proceed without waiting
        # for THIS shard's rebuild.
        with self._lock:
            shard = self._shard(kind, model)

        # Snapshot the shard's authoritative state INSIDE the per-shard
        # lock so the matmul reads a consistent (matrix, ids, cold,
        # levels, dim) tuple.  Without the snapshot, a concurrent
        # _rebuild_shard could swap shard.matrix between the read on
        # this line and the read of `shard.ids` further down, yielding
        # an old-matrix paired with a new-ids list — silent corruption
        # (i.e. score index N referring to a different UUID than the
        # caller would expect).
        with shard.lock:
            if shard.dirty or shard.matrix is None:
                _rebuild_shard(shard, conn, rebuild_sql, model)
            matrix = shard.matrix
            ids = shard.ids
            cold = shard.cold
            levels_arr = shard.levels
            dim = shard.dim
            # Snapshot the cached level/exclude masks alongside so the
            # matmul-side helpers read a stable view; they're keyed off
            # `shard` itself so the lock continues to protect them
            # during the helper calls below.

        if matrix is None or matrix.shape[0] == 0:
            return []

        if len(query_vec) != dim:
            raise ValueError(f"query_vec dim {len(query_vec)} does not match shard dim {dim}")
        q = np.asarray(query_vec, dtype=np.float32)
        scores = matrix @ q

        # Mask out filtered rows by setting their score to -inf.
        if not include_cold and cold is not None:
            scores = np.where(cold, -np.inf, scores)
        if levels is not None:
            level_mask = _level_mask(shard, levels, levels_arr)
            scores = np.where(level_mask, scores, -np.inf)
        if exclude_ids:
            id_mask = _exclude_mask(shard, exclude_ids, ids)
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


def _rebuild_shard(
    shard: _IndexShard,
    conn: sqlite3.Connection,
    sql: str,
    model: str,
) -> None:
    """One full materialization.

    Caller MUST hold `shard.lock`; we re-assign every cached field at
    the end and a concurrent reader would otherwise see a torn shard.
    """
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


def _level_mask(
    shard: _IndexShard,
    levels: Sequence[str],
    snapshot_levels: list[str],
) -> npt.NDArray[np.bool_]:
    """Boolean mask over shard rows whose level is in `levels`.

    Per-level masks are memoized on the shard so a search with
    `levels=['summary']` doesn't rebuild the boolean from a Python list
    comprehension on every call.  For a 100k-row shard that previously
    cost 1-3ms per search; now it's a vectorized OR.

    `snapshot_levels` is the caller-snapshotted parallel level list (the
    same array referenced by `shard.levels` at lock-release time).  We
    pass it explicitly so a concurrent rebuild that swaps `shard.levels`
    underneath us still produces a mask of the right shape for the
    caller's snapshotted matrix.
    """
    with shard.lock:
        out = np.zeros(len(snapshot_levels), dtype=np.bool_)
        for level in levels:
            cached = shard.level_masks.get(level)
            # If the cache was invalidated mid-search (a rebuild between
            # our outer snapshot and now), `cached` length may not match
            # `snapshot_levels` — refresh against the snapshot.
            if cached is None or len(cached) != len(snapshot_levels):
                cached = np.asarray(
                    [lv == level for lv in snapshot_levels], dtype=np.bool_
                )
                # Only memoize when the cache matches the live shard so
                # we don't poison a fresh rebuild's mask cache with a
                # snapshot-sized array.
                if snapshot_levels is shard.levels:
                    shard.level_masks[level] = cached
            out |= cached
    return out


def _exclude_mask(
    shard: _IndexShard,
    exclude_ids: Sequence[bytes],
    snapshot_ids: list[bytes],
) -> npt.NDArray[np.bool_]:
    """Boolean mask True at rows whose id is in `exclude_ids`.

    `snapshot_ids` is the caller-snapshotted id list; same rationale as
    `_level_mask` above — keeps the mask shape aligned with the matmul.
    """
    with shard.lock:
        # Build / refresh the lookup table against the live ids if the
        # snapshot still matches; otherwise build a one-shot lookup
        # against the snapshot.  Either way the lookup is by id-bytes →
        # row-index against the snapshot list.
        if snapshot_ids is shard.ids:
            if not shard.id_to_idx:
                shard.id_to_idx = {iid: i for i, iid in enumerate(shard.ids)}
            lookup = shard.id_to_idx
        else:
            lookup = {iid: i for i, iid in enumerate(snapshot_ids)}
    out = np.zeros(len(snapshot_ids), dtype=np.bool_)
    for iid in exclude_ids:
        idx = lookup.get(iid)
        if idx is not None:
            out[idx] = True
    return out
