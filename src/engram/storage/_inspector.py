"""Read-only inspection helpers for storage backends.

Intended for tests, debugging, and benchmark manifests — not for hot paths.
"""

from __future__ import annotations

from typing import TypedDict

from engram.storage._protocol import Storage


class StorageStats(TypedDict):
    events: int
    memory_items: int
    embeddings: int
    provenance_links: int
    clusters: int
    by_level: dict[str, int]


def stats(storage: Storage) -> StorageStats:
    """Snapshot of row counts in the backend. Cheap; no secondary scans."""
    by_level_counts = storage.count_memory_items_by_level()
    by_level = {level.value: count for level, count in by_level_counts.items()}
    return {
        "events": storage.count_events(),
        "memory_items": storage.count_memory_items(),
        "embeddings": storage.count_embeddings(),
        "provenance_links": storage.count_provenance_links(),
        "clusters": storage.count_clusters(),
        "by_level": by_level,
    }


__all__ = ["StorageStats", "stats"]
