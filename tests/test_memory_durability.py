"""Durability tests for `Memory.observe`.

Two Stage 3 DoD requirements:

  - **Crash-safety**: SIGKILL between calls leaves the store in a
    consistent state — committed observations stay; uncommitted ones
    don't corrupt anything.
  - **Concurrent writers**: with 8 concurrent observers, no events are
    dropped.

The crash test spawns a writer subprocess, polls a progress-file
heartbeat the subprocess updates after every commit, and tears the
subprocess down once the count clears a threshold (`SIGKILL` on POSIX,
`TerminateProcess` on Windows).  Polling for a count instead of
sleeping a fixed second decouples the test from machine-class latency
— slow CI runners still get the same coverage without inflating wall
clock.

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

    After every commit, rewrites `progress_path` with the running count
    so the parent can poll for liveness instead of sleeping on time.
    """
    storage = SqliteStorage(db_path)
    storage.initialize()
    memory = Memory(storage=storage, embedder=FakeEmbedder(dim=dim))
    i = 0
    progress = Path(progress_path)
    while True:
        memory.observe(f"event-{i}")
        i += 1
        # Atomic-ish via os.replace under a temp file.  A torn read
        # from the parent at worst returns a stale int — never a
        # corrupt one — and the parent's loop tolerates that.
        tmp = progress.with_suffix(progress.suffix + ".tmp")
        tmp.write_text(str(i), encoding="utf-8")
        tmp.replace(progress)


def test_observe_durable_across_sigkill(tmp_path: Path) -> None:
    db_path = str(tmp_path / "kill.db")
    progress_path = tmp_path / "progress.txt"
    # Bootstrap schema in the parent so the subprocess has a fully-migrated DB
    # to write into immediately.
    bootstrap = SqliteStorage(db_path)
    bootstrap.initialize()
    bootstrap.close()

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_writer_loop, args=(db_path, 16, str(progress_path))
    )
    proc.start()
    try:
        # Poll the progress file until the subprocess has committed at
        # least `min_events` events.  Replaces a fixed `time.sleep(1.0)`
        # — the wallclock target made the test flaky on slow runners
        # (subprocess didn't write anything in time on a loaded box).
        min_events = 10
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            if progress_path.exists():
                try:
                    if int(progress_path.read_text(encoding="utf-8")) >= min_events:
                        break
                except (ValueError, OSError):
                    # Torn read between the subprocess's write_text and
                    # replace; try again.
                    pass
            time.sleep(0.01)
        else:
            raise AssertionError(
                f"writer subprocess didn't reach {min_events} events within 30s"
            )
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
