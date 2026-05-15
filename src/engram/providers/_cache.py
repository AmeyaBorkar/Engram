"""LRU cache primitive for provider responses.

Keyed by string (typically a content hash). Observable hit rate, fixed
max-size eviction.

Thread safety: an internal `threading.RLock` guards every mutation.
This matters because `LocalEmbedder.aembed` hops onto
`asyncio.to_thread`, so the cache can be accessed concurrently from
multiple worker threads when several `aembed`/`aembed_query` coroutines
fire at once. The lock is a no-op for the dominant single-thread path
(uncontended `RLock.acquire` is a CAS) and prevents `OrderedDict`
corruption when there is contention.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Generic, TypeVar

V = TypeVar("V")


def content_hash(*parts: str) -> str:
    """SHA-256 over `parts`, separated by a NUL byte to avoid collisions."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class Cache(Generic[V]):
    """Thread-safe LRU cache with observable hit/miss counters."""

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._data: OrderedDict[str, V] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> V | None:
        """Return the cached value for `key` if present, else `None`. Updates LRU order."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return None

    def set(self, key: str, value: V) -> None:
        """Insert `(key, value)`; evict oldest if over capacity."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Drop everything and reset counters."""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Hits divided by (hits + misses). Zero when no lookups have happened."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
