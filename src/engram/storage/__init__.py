"""Engram storage backends.

The `Storage` protocol is the abstraction; backends implement it. Stage 1
ships only `SqliteStorage`; Stage 9 adds `PostgresStorage` against the same
protocol.
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
