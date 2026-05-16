"""Engram storage backends.

The `Storage` protocol is the abstraction; backends implement it.
Currently ships `SqliteStorage`; alternate backends (Postgres / DuckDB /
sqlite-vec) are roadmap items, not on disk yet.  Implement the protocol
to add one.
"""

from engram.storage._inspector import StorageStats, stats
from engram.storage._protocol import Storage
from engram.storage.sqlite import SqliteStorage

__all__ = [
    "SqliteStorage",
    "Storage",
    "StorageStats",
    "stats",
]
