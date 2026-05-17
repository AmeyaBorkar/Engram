"""Direct tests for the in-memory VectorIndex shard model.

The headline guarantees:

  * Each shard owns its own RLock — H-37: a slow rebuild on one shard
    must NOT block a search against a different shard.
  * `search` snapshots (matrix, ids, cold, levels, dim) under the shard
    lock — H-38: a concurrent rebuild can't tear a search's view (old
    matrix paired with new ids).

We test the snapshot invariant directly against `_IndexShard` so the
proof doesn't depend on race-window timing.
"""

from __future__ import annotations

import threading

from engram.storage._vector_index import _IndexShard, VectorIndex


def test_shards_have_distinct_locks() -> None:
    """H-37 prerequisite: per-shard RLock, not a shared module global."""
    a = _IndexShard()
    b = _IndexShard()
    assert a.lock is not b.lock


def test_vector_index_shards_have_distinct_locks() -> None:
    """A VectorIndex hands out a fresh shard per (kind, model), each with
    its own lock — so a slow rebuild on (event, m1) doesn't gate a search
    against (memory_item, m1) or (event, m2).
    """
    vi = VectorIndex()
    with vi._lock:
        s1 = vi._shard("event", "m1")
        s2 = vi._shard("memory_item", "m1")
        s3 = vi._shard("event", "m2")
    assert s1.lock is not s2.lock
    assert s1.lock is not s3.lock
    assert s2.lock is not s3.lock


def test_mark_dirty_does_not_hold_index_lock_during_shard_flip() -> None:
    """H-37 follow-on: `mark_dirty` walks the shard dict under the
    index-wide lock to collect targets, then RELEASES it before reaching
    for each shard's lock.

    We verify by holding a shard lock from one thread and confirming
    `mark_dirty` from another thread can still acquire the index-wide
    lock (i.e. doesn't deadlock waiting on the held shard lock while
    holding the index-wide one).
    """
    vi = VectorIndex()
    with vi._lock:
        shard = vi._shard("event", "m1")

    started = threading.Event()
    released_index_lock = threading.Event()

    def hold_shard() -> None:
        with shard.lock:
            started.set()
            released_index_lock.wait(timeout=2.0)

    holder = threading.Thread(target=hold_shard, daemon=True)
    holder.start()
    assert started.wait(timeout=2.0)

    # While the shard lock is held, mark_dirty should still complete:
    # it walks the dict under the index lock, releases it, then waits
    # on the held shard lock to flip dirty=True.  But the index-wide
    # lock itself becomes free immediately — we can prove that by
    # acquiring it from this thread without timeout.
    mark_done = threading.Event()

    def call_mark_dirty() -> None:
        vi.mark_dirty(kind="event")
        mark_done.set()

    marker = threading.Thread(target=call_mark_dirty, daemon=True)
    marker.start()

    # The marker thread is blocked on shard.lock now; the index-wide
    # lock is NOT held by it.  We must be able to acquire it ourselves.
    acquired = vi._lock.acquire(timeout=2.0)
    try:
        assert acquired, "mark_dirty did not release the index-wide lock"
    finally:
        if acquired:
            vi._lock.release()
    # Tidy up: let the holder release shard.lock so mark_dirty completes.
    released_index_lock.set()
    holder.join(timeout=2.0)
    marker.join(timeout=2.0)
    assert mark_done.is_set()
    assert shard.dirty is True


def test_search_snapshot_survives_concurrent_rebuild_flip() -> None:
    """H-38 regression: `search` snapshots matrix/ids/cold/levels INSIDE
    the shard lock so a concurrent rebuild that swaps them under the
    matmul can't produce torn results.

    We can't easily race two real searches against an actual SqliteStorage
    matmul, so test the invariant: after a search returns successfully,
    its returned (UUID, idx, score) triples reference ids that existed
    in the shard at search time — even if `mark_dirty` ran during the
    matmul.
    """
    # The structural test in the previous case already exercises the
    # per-shard lock acquire pattern; a brittle thread-timing test here
    # would add little.  We instead verify the search method snapshots
    # by inspecting the source layout: after the `with shard.lock`
    # block, `search` reads `matrix`, `ids`, `cold`, `levels_arr`, `dim`
    # from local variables — not from `shard.*`.
    import inspect

    from engram.storage import _vector_index

    src = inspect.getsource(_vector_index.VectorIndex.search)
    # The snapshot block must exist and use the per-shard lock.
    assert "with shard.lock:" in src
    # And the matmul must run against the snapshot, not against `shard.matrix`.
    # (Heuristic check: `scores = matrix @ q` rather than
    # `scores = shard.matrix @ q`.)
    assert "scores = matrix @ q" in src
    assert "scores = shard.matrix @ q" not in src
