"""LRU cache primitive for provider responses.

Keyed by string (typically a content hash). Observable hit rate, fixed
max-size eviction. Not thread-safe by design — wrap with a lock if you
need cross-thread sharing; most provider call sites are single-threaded
or already locked at a higher level.
"""

from __future__ import annotations

import hashlib
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
    """LRU cache with observable hit/miss counters."""

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._data: OrderedDict[str, V] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> V | None:
        """Return the cached value for `key` if present, else `None`. Updates LRU order."""
        if key in self._data:
            self._data.move_to_end(key)
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def set(self, key: str, value: V) -> None:
        """Insert `(key, value)`; evict oldest if over capacity."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def clear(self) -> None:
        """Drop everything and reset counters."""
        self._data.clear()
        self._hits = 0
        self._misses = 0

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
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
