"""The `Storage` protocol.

A backend implements this surface. The SQLite backend in `sqlite.py` is the
only implementation in Stage 1; Stage 9 brings Postgres against the same
protocol.

The protocol is intentionally small and synchronous. Stage 9 layers an async
surface on top.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from engram.schemas import (
    Cluster,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
    ProvenanceLink,
)


@runtime_checkable
class Storage(Protocol):
    """Pluggable storage backend.

    All methods raise on integrity violations (duplicate ids, dangling
    foreign keys, CHECK failures). Backends MUST guarantee that a successful
    return means the row is durable on disk before returning.
    """

    def initialize(self) -> None:
        """Apply pending migrations. Idempotent."""

    def close(self) -> None:
        """Release backend resources. Safe to call multiple times."""

    def transaction(self) -> AbstractContextManager[None]:
        """Wrap a block of operations in an atomic transaction.

        If already inside a transaction, this is a no-op (re-entrant).
        """

    # --- events -------------------------------------------------------------

    def insert_event(self, event: Event) -> None: ...
    def insert_events(self, events: Iterable[Event]) -> int: ...
    def get_event(self, event_id: UUID) -> Event | None: ...
    def list_events(
        self,
        limit: int = 100,
        before: datetime | None = None,
        source: str | None = None,
    ) -> list[Event]: ...
    def count_events(self) -> int: ...

    # --- memory items -------------------------------------------------------

    def insert_memory_item(self, item: MemoryItem) -> None: ...
    def insert_memory_items(self, items: Iterable[MemoryItem]) -> int: ...
    def get_memory_item(self, item_id: UUID) -> MemoryItem | None: ...
    def list_memory_items(
        self,
        level: Level | None = None,
        cluster_id: UUID | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]: ...
    def update_memory_item_weight(self, item_id: UUID, weight: float) -> None: ...
    def count_memory_items(self) -> int: ...
    def count_memory_items_by_level(self) -> dict[Level, int]: ...

    # --- embeddings ---------------------------------------------------------

    def insert_embedding(self, embedding: Embedding) -> None: ...
    def get_embedding(self, item_id: UUID, item_kind: ItemKind, model: str) -> Embedding | None: ...
    def count_embeddings(self) -> int: ...

    # --- provenance ---------------------------------------------------------

    def link_provenance(
        self, memory_item_id: UUID, event_id: UUID, weight: float = 1.0
    ) -> ProvenanceLink: ...
    def get_supporting_events(self, memory_item_id: UUID) -> list[Event]: ...
    def get_supported_memory_items(self, event_id: UUID) -> list[MemoryItem]: ...
    def count_provenance_links(self) -> int: ...

    # --- clusters -----------------------------------------------------------

    def insert_cluster(self, cluster: Cluster) -> None: ...
    def get_cluster(self, cluster_id: UUID) -> Cluster | None: ...
    def count_clusters(self) -> int: ...

    # --- search -------------------------------------------------------------

    def search_event_embeddings(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
    ) -> list[tuple[UUID, str, float]]:
        """Top-k events by cosine similarity to `query_vec`.

        Returns `(event_id, content, score)` triples sorted by score desc.
        Both `query_vec` and stored embedding vectors are assumed unit-norm,
        so cosine similarity reduces to a dot product. Only embeddings
        matching `model` are considered.
        """
