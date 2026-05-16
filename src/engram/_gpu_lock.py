"""Process-wide bounded semaphore for GPU-bound model calls.

Local embedders and rerankers hold their weights in VRAM once, then
re-allocate *activation* memory on every forward pass. When the bench
fans out N parallel questions, each thread enqueues its own forward
pass; activations stack on the device and OOM happens at any N beyond
the card's headroom. Wrapping the actual `model.encode(...)` /
`model.predict(...)` calls in this semaphore decouples request-side
concurrency from device-side concurrency:

  * `--parallel 30` keeps fanning out chat / judge HTTP calls (good,
    those are I/O-bound and the LLM API is the wall-time bottleneck).
  * `--gpu-concurrency K` caps how many threads can be inside a CUDA
    forward pass at the same time -- the weights are loaded once, but
    only K thread can allocate activations concurrently.

`K=1` is the safe operating point on tight VRAM budgets (12 GB at fp32
with bge-large + bge-reranker-v2-m3 loaded). Raise to 2-4 if you have
headroom or are running fp16. The semaphore is process-wide because
all model instances share the same CUDA device; per-model semaphores
would not solve the problem.

Configured via:
  * `ENGRAM_GPU_CONCURRENCY` env var (read on first access).
  * `configure_gpu_concurrency(K)` for programmatic override.
  * The bench CLI's `--gpu-concurrency K` flag (sets the env var
    before any model is constructed).

Default: K=1.  The semaphore is *acquired* unconditionally inside the
context manager; on CPU-only fallbacks the cost is one un-contended
lock acquire (~100 ns), so leaving the wrap in place is cheap even on
hardware where it does nothing.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_DEFAULT = 1
_ENV_VAR = "ENGRAM_GPU_CONCURRENCY"

_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None
_size: int = _DEFAULT


def _read_env_size() -> int:
    raw = os.environ.get(_ENV_VAR, "")
    if not raw:
        return _DEFAULT
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT
    return max(1, n)


def _get() -> threading.BoundedSemaphore:
    """Return the singleton semaphore, lazy-initializing from env on first use."""
    global _semaphore, _size
    if _semaphore is None:
        with _lock:
            if _semaphore is None:
                _size = _read_env_size()
                _semaphore = threading.BoundedSemaphore(_size)
    return _semaphore


def configure_gpu_concurrency(n: int) -> None:
    """Reset the GPU-call semaphore to allow `n` concurrent forward passes.

    Intended for one-time setup at process start.  Must be called before
    any model.encode / model.predict goes through `gpu_section()`;
    permits already acquired by another thread are forgotten on reset
    (which would only matter if you reconfigure mid-flight, which you
    shouldn't).
    """
    global _semaphore, _size
    if n < 1:
        raise ValueError(f"gpu concurrency must be >= 1, got {n}")
    with _lock:
        _semaphore = threading.BoundedSemaphore(n)
        _size = n


@contextmanager
def gpu_section() -> Iterator[None]:
    """Acquire the global GPU semaphore for the duration of the block.

    Wrap any torch CUDA forward-pass call (`model.encode`,
    `model.predict`, ...) inside this context manager.  Holding the
    lock on CPU-only paths is harmless (~100 ns overhead); the
    embedder / reranker stay portable across devices without the call
    site having to branch on `.device`.
    """
    sem = _get()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def current_concurrency() -> int:
    """Currently-configured max-concurrent value (diagnostics)."""
    _get()
    return _size
