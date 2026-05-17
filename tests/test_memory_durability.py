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


def _writer_loop(db_path: str, dim: int, progress_path: str) -> None:
    """Subprocess entrypoint: observe events forever (until killed).

    Writes the running commit count to `progress_path` after each
    observation so the parent can poll for liveness instead of guessing
    with a fixed sleep.
    """
    storage = SqliteStorage(db_path)
    storage.initialize()
    memory = Memory(storage=storage, embedder=FakeEmbedder(dim=dim))
    i = 0
    progress = Path(progress_path)
    while True:
        memory.observe(f"event-{i}")
        i += 1
        # Atomic-ish write: full content per tick is tiny.
        progress.write_text(str(i), encoding="utf-8")


def test_observe_durable_across_sigkill(tmp_path: Path) -> None:
    db_path = str(tmp_path / "kill.db")
    progress_path = tmp_path / "progress.txt"
    # Bootstrap schema in the parent so the subprocess has a fully-migrated DB
    # to write into immediately.
    bootstrap = SqliteStorage(db_path)
    bootstrap.initialize()
    bootstrap.close()

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_writer_loop, args=(db_path, 16, str(progress_path)))
    proc.start()
    try:
        # Poll for the subprocess to commit at least one event. Slow
        # runners (Windows spawn-import takes ~1s) get up to 30s of
        # leeway; a healthy runner exits the loop in <100ms.
        deadline = time.monotonic() + 30.0
        observed = 0
        while time.monotonic() < deadline:
            if progress_path.exists():
                try:
                    observed = int(progress_path.read_text(encoding="utf-8") or "0")
                except ValueError:
                    observed = 0
                if observed > 0:
                    break
            time.sleep(0.05)
        assert observed > 0, "subprocess never reported progress before deadline"
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
