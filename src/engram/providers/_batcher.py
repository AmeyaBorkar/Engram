"""Batcher — coalesces concurrent calls into a single fn(list) invocation.

Use case: many threads call `submit(item)` while a single underlying
function processes them as a batch. The first arrival starts a window;
subsequent submissions ride along until either `window_ms` elapses or
`max_batch` items accumulate. Each submit blocks until its batch is done.

The Stage 2 DoD requires that batching reduce steady-state provider-call
count by at least 5x; the `call_count` property is observable so tests
and benchmarks can assert that.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class _Pending(Generic[T, R]):
    __slots__ = ("completed", "error", "event", "item", "result")

    def __init__(self, item: T) -> None:
        self.item: T = item
        self.event: threading.Event = threading.Event()
        # `result` is the actual return value once `completed=True`.  We keep
        # them separate so that a legitimate result of `None` / `""` / `0`
        # is distinguishable from "the worker never ran a batch for me."
        self.result: R | None = None
        self.error: BaseException | None = None
        self.completed: bool = False


class Batcher(Generic[T, R]):
    """Coalesce concurrent `submit(item)` calls into batched `fn(list)` calls."""

    def __init__(
        self,
        fn: Callable[[list[T]], list[R]],
        *,
        window_ms: int = 20,
        max_batch: int = 64,
    ) -> None:
        if window_ms < 0:
            raise ValueError(f"window_ms must be >= 0, got {window_ms}")
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")

        self._fn = fn
        self._window = window_ms / 1000.0
        self._max_batch = max_batch
        self._cond = threading.Condition(threading.Lock())
        self._queue: list[_Pending[T, R]] = []
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of times the underlying `fn` has been invoked. Observable for tests."""
        return self._call_count

    def submit(self, item: T) -> R:
        """Submit `item`, blocking until its batch has been processed."""
        if self._stop.is_set():
            raise RuntimeError("Batcher is stopped")
        pending: _Pending[T, R] = _Pending(item)
        with self._cond:
            self._queue.append(pending)
            self._ensure_worker_locked()
            self._cond.notify_all()
        pending.event.wait()
        if pending.error is not None:
            raise pending.error
        if not pending.completed:  # pragma: no cover - defensive
            raise RuntimeError("batch event set without completing")
        # `result` may legitimately be a falsy value (None, 0, "", []); the
        # `completed` flag is the source of truth, not the value itself.
        return pending.result  # type: ignore[return-value]

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker thread. Pending submits will raise; in-flight
        submits wake up with `RuntimeError("Batcher stopped")` rather than
        blocking on a worker that will never serve them.
        """
        with self._cond:
            self._stop.set()
            # Drain any submits whose batch never ran.  Without this they
            # would block on `pending.event.wait()` forever after the worker
            # exits.
            drained = self._queue
            self._queue = []
            self._cond.notify_all()
        for pending in drained:
            if not pending.completed:
                pending.error = RuntimeError("Batcher stopped")
                pending.completed = True
                pending.event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=timeout)

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect_batch()
            if batch is None:
                continue
            self._dispatch(batch)

    def _collect_batch(self) -> list[_Pending[T, R]] | None:
        with self._cond:
            while not self._queue and not self._stop.is_set():
                self._cond.wait(timeout=1.0)
            if self._stop.is_set():
                return None

            deadline = time.monotonic() + self._window
            while True:
                if len(self._queue) >= self._max_batch:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

            batch = self._queue[: self._max_batch]
            self._queue = self._queue[self._max_batch :]
            return batch

    def _dispatch(self, batch: list[_Pending[T, R]]) -> None:
        self._call_count += 1
        items = [p.item for p in batch]
        try:
            results = self._fn(items)
        except Exception as exc:
            # Catch Exception, not BaseException — KeyboardInterrupt and
            # SystemExit should propagate up so a Ctrl+C actually shuts the
            # worker down instead of being silently redistributed to every
            # pending caller as a thread-local copy.
            for pending in batch:
                pending.error = exc
                pending.completed = True
                pending.event.set()
            return

        if len(results) != len(batch):
            err: Exception = RuntimeError(
                f"batched fn returned {len(results)} results for batch of {len(batch)}"
            )
            for pending in batch:
                pending.error = err
                pending.completed = True
                pending.event.set()
            return

        for pending, r in zip(batch, results, strict=True):
            pending.result = r
            pending.completed = True
            pending.event.set()
