"""Durability tests for `Memory.observe`.

Two Stage 3 DoD requirements:

  - **Crash-safety**: SIGKILL between calls leaves the store in a
    consistent state — committed observations stay; uncommitted ones
    don't corrupt anything.
  - **Concurrent writers**: with 8 concurrent observers, no events are
    dropped.

The crash test spawns a writer subprocess, lets it write for ~1 second,
then kills it (`SIGKILL` on POSIX, `TerminateProcess` on Windows). The
parent then reopens the database and verifies that whatever count it
reports matches the number of events it can actually read back.

Both tests run in CI; they are not marked `slow`.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

from engram import Memory, SqliteStorage
from engram.providers import FakeEmbedder


def _writer_loop(db_path: str, dim: int) -> None:
    """Subprocess entrypoint: observe events forever (until killed)."""
    storage = SqliteStorage(db_path)
    storage.initialize()
    memory = Memory(storage=storage, embedder=FakeEmbedder(dim=dim))
    i = 0
    while True:
        memory.observe(f"event-{i}")
        i += 1


def test_observe_durable_across_sigkill(tmp_path: Path) -> None:
    db_path = str(tmp_path / "kill.db")
    # Bootstrap schema in the parent so the subprocess has a fully-migrated DB
    # to write into immediately.
    bootstrap = SqliteStorage(db_path)
    bootstrap.initialize()
    bootstrap.close()

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_writer_loop, args=(db_path, 16))
    proc.start()
    try:
        time.sleep(1.0)
    finally:
        proc.kill()
        proc.join(timeout=5.0)
    assert not proc.is_alive(), "writer subprocess didn't exit after kill"

    storage = SqliteStorage(db_path)
    storage.initialize()
    try:
        n = storage.count_events()
        assert n > 0, "writer didn't commit any events before being killed"
        events = storage.list_events(limit=n + 1)
        # Reported count matches number of readable rows -> no corruption.
        assert len(events) == n
    finally:
        storage.close()


def test_concurrent_observers_no_drops(tmp_path: Path) -> None:
    db_path = str(tmp_path / "concurrent.db")
    storage = SqliteStorage(db_path)
    storage.initialize()
    memory = Memory(storage=storage, embedder=FakeEmbedder(dim=32))

    n_writers = 8
    n_per_writer = 50
    barrier = threading.Barrier(n_writers + 1)
    errors: list[BaseException] = []

    def writer(worker_id: int) -> None:
        barrier.wait()
        try:
            for i in range(n_per_writer):
                memory.observe(f"w{worker_id}-e{i}")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()

    try:
        assert errors == []
        assert storage.count_events() == n_writers * n_per_writer
        # Verify content uniqueness too: no swallowed-then-reissued ids.
        unique_contents = {e.content for e in storage.list_events(limit=n_writers * n_per_writer)}
        assert len(unique_contents) == n_writers * n_per_writer
    finally:
        storage.close()
