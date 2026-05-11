"""The `Storage` protocol.

A backend implements this surface. The SQLite backend in `sqlite.py` is the
only implementation in Stage 1; Stage 9 brings Postgres against the same
protocol.

The protocol is intentionally small and synchronous. Stage 9 layers an async
surface on top.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from engram.schemas import (
    Cluster,
    Conflict,
    ConflictStatus,
    DecayState,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
    Outcome,
    Procedure,
    ProvenanceLink,
    Resolution,
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
    def update_memory_item_level(self, item_id: UUID, level: Level) -> None:
        """Move the item to a different hierarchy level.

        Used by the consolidation engine's promotion pass: a stable,
        frequently-corroborated `Level.SUMMARY` rises to `Level.ABSTRACTION`.
        Raises `KeyError` if the item is missing.
        """

    def iter_memory_items(
        self,
        *,
        level: Level | None = None,
        include_cold: bool = False,
        batch_size: int = 1000,
    ) -> Iterator[MemoryItem]:
        """Stream memory items, optionally filtered by `level`.

        Cold items are excluded by default (they are not part of the
        active surface). Pages via the backend's natural cursor; order
        is `(created_at ASC, id ASC)`.
        """

    def count_memory_items(self) -> int: ...
    def count_memory_items_by_level(self) -> dict[Level, int]: ...

    # --- temporal validity & invalidation (Stage 8) ------------------------

    def invalidate_memory_item(
        self,
        item_id: UUID,
        *,
        at: datetime,
        by: UUID | None = None,
    ) -> None:
        """Mark a memory item as invalidated at `at`.

        `by` is the id of the item that won the conflict that
        invalidated this one (NULL allowed for TTL-style invalidations
        with no replacement). Idempotent if already invalidated -- the
        existing `invalidated_at` is preserved (the *first* invalidation
        timestamp is what `as_of` queries care about). Raises `KeyError`
        if the item does not exist.
        """

    def set_validity_window(
        self,
        item_id: UUID,
        *,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> None:
        """Set the validity window for a memory item.

        Either argument may be `None` to leave that side unchanged --
        pass `valid_until=...` to add a TTL to an item that currently has
        no upper bound, for example. The storage layer enforces
        `valid_until >= valid_from`. Raises `KeyError` if the item does
        not exist.
        """

    def set_source_trust(self, item_id: UUID, trust: float | None) -> None:
        """Set the denormalized source trust on a memory item.

        `None` clears the value. Trust must be in `[0, 1]`. Raises
        `KeyError` if the item does not exist.
        """

    def search_memory_item_embeddings_as_of(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
        as_of: datetime | None = None,
        levels: Sequence[Level] | None = None,
        exclude_ids: Sequence[UUID] = (),
        include_cold: bool = False,
        candidate_multiplier: int = 4,
    ) -> list[tuple[UUID, str, float]]:
        """Temporal-aware variant of `search_memory_item_embeddings`.

        `as_of=None` is the "current state" mode: items with
        `invalidated_at IS NOT NULL` are excluded. `as_of=<datetime>` is
        the "as-of-then" mode: items are visible iff `valid_from <=
        as_of AND (valid_until IS NULL OR valid_until > as_of) AND
        (invalidated_at IS NULL OR invalidated_at > as_of)`.

        Returns at most `k` `(id, content, score)` triples sorted by
        score desc.

        Implementation note: the in-memory vector index is not
        validity-aware in this stage; the method over-fetches by
        `candidate_multiplier` and applies the temporal filter via a
        single SQL pass on the candidates. The multiplier protects
        against corpora where many items have been invalidated; raise it
        if your dataset is mostly historical.
        """

    # --- conflicts (Stage 8) -----------------------------------------------

    def record_conflict(self, conflict: Conflict) -> None:
        """Persist a new conflict row. Raises on duplicate id or
        duplicate (source_item_id, target_item_id) pair."""

    def get_conflict(self, conflict_id: UUID) -> Conflict | None:
        """Fetch a conflict by id, or None if missing."""

    def list_conflicts(
        self,
        *,
        status: ConflictStatus | None = None,
        memory_item_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Conflict]:
        """List conflicts, optionally filtered.

        `status` narrows to OPEN or RESOLVED. `memory_item_id` filters
        to rows where the item is either the source or the target --
        the conflicts graph is undirected from the caller's
        perspective, so both directions are walked. Ordered by
        `detected_at` desc (newest first).
        """

    def resolve_conflict(
        self,
        conflict_id: UUID,
        *,
        resolution: Resolution,
        resolved_winner_id: UUID | None,
        resolved_at: datetime,
    ) -> Conflict:
        """Atomically transition a conflict OPEN -> RESOLVED.

        The reconciler engine is the proper user of this method: it
        picks the winner per the chosen policy, then calls this to
        persist the decision. Raises `KeyError` if the conflict does
        not exist. Raises `RuntimeError` if the conflict is already
        resolved (callers should idempotently filter on status first).
        Validates that `resolved_winner_id` matches `source_item_id` or
        `target_item_id` (or is None for `Resolution.KEEP_BOTH`).
        Returns the refreshed `Conflict` row.
        """

    def count_conflicts(self) -> int: ...

    def count_conflicts_by_status(self) -> dict[ConflictStatus, int]: ...

    # --- procedures ---------------------------------------------------------

    def insert_procedure(self, procedure: Procedure) -> None:
        """Persist a new procedure. Raises on duplicate id."""

    def get_procedure(self, procedure_id: UUID) -> Procedure | None:
        """Fetch a procedure by id, or None if missing."""

    def list_procedures(
        self,
        *,
        outcome: Outcome | None = None,
        limit: int = 100,
    ) -> list[Procedure]:
        """List procedures, optionally filtered by `outcome`. Ordered by
        `created_at` desc (most recent first)."""

    def update_procedure_outcome(self, procedure_id: UUID, outcome: Outcome) -> None:
        """Set the outcome of a procedure + bump `updated_at`.

        Raises `KeyError` if the procedure does not exist. Idempotent if
        the outcome is already the requested value.
        """

    def count_procedures(self) -> int: ...
    def count_procedures_by_outcome(self) -> dict[Outcome, int]: ...

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
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        """Top-k events by cosine similarity to `query_vec`.

        Returns `(event_id, content, score)` triples sorted by score desc.
        Both `query_vec` and stored embedding vectors are assumed unit-norm,
        so cosine similarity reduces to a dot product. Only embeddings
        matching `model` are considered. Items pruned by the decay engine
        (`cold_at IS NOT NULL`) are excluded by default; pass
        `include_cold=True` to surface them anyway (audit / inspection
        flows).
        """

    def search_memory_item_embeddings(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
        levels: Sequence[Level] | None = None,
        exclude_ids: Sequence[UUID] = (),
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        """Top-k memory items by cosine similarity to `query_vec`.

        Returns `(item_id, content, score)` triples sorted by score desc.
        `levels` filters to a subset of `Level` (None means any). The
        contradiction detector passes `levels=[SUMMARY, ABSTRACTION]` -
        it has no interest in event-level memory items. `exclude_ids`
        is a small set of memory_item ids to skip (the engine excludes
        the just-inserted abstraction when checking against existing
        ones). Cold items are excluded by default.
        """

    def search_procedure_embeddings(
        self,
        query_vec: Sequence[float],
        *,
        k: int,
        model: str,
        outcomes: Sequence[Outcome] | None = None,
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        """Top-k procedures by cosine similarity of `query_vec` to the
        stored `situation` embedding.

        Returns `(procedure_id, situation, score)` triples sorted by
        score desc. `outcomes` filters to a subset of `Outcome` (None
        means any -- the agent benefits from seeing failures too).
        Cold procedures are excluded by default.
        """

    def score_events_by_ids(
        self,
        query_vec: Sequence[float],
        event_ids: Sequence[UUID],
        *,
        model: str,
        include_cold: bool = False,
    ) -> list[tuple[UUID, str, float]]:
        """Score the named events against `query_vec`, sorted by score desc.

        Returns `(event_id, content, score)` triples for every event in
        `event_ids` that has an embedding for `model`. Used by the Stage
        6 retriever to drill into a low-confidence abstraction's
        supporting events: it already has the event ids from the
        provenance links, and just needs them re-scored against the
        query. Cold events are excluded by default (matches
        `search_event_embeddings`).
        """

    # --- decay state --------------------------------------------------------

    def get_decay_state(self, item_id: UUID, kind: ItemKind) -> DecayState | None:
        """Return the per-item decay state, or `None` if the item is missing."""

    def iter_decay_states(
        self,
        kind: ItemKind,
        *,
        include_cold: bool = False,
        batch_size: int = 1000,
    ) -> Iterator[DecayState]:
        """Iterate every decay state for items of `kind`.

        Backends MUST stream rows in batches of `batch_size` so the engine
        can tick a multi-million-item store without holding everything in
        memory. Order is unspecified.
        """

    def update_decay_state(self, state: DecayState) -> None:
        """Persist the new decay state for the (item_id, kind) row.

        Raises `KeyError` if the item does not exist.
        """

    def mark_cold(self, item_id: UUID, kind: ItemKind, *, at: datetime) -> None:
        """Set `cold_at = at` on the item. Idempotent if already cold."""

    def unmark_cold(self, item_id: UUID, kind: ItemKind) -> None:
        """Clear `cold_at`. Used to restore an item to the hot pool (admin)."""

    def count_cold(self, kind: ItemKind) -> int:
        """Number of cold items of the given kind (`cold_at IS NOT NULL`)."""

    def delete_cold_items(self, kind: ItemKind) -> int:
        """Hard-delete every cold item of `kind`. Returns the number deleted.

        For the `delete` prune policy. Memory items cascade through
        provenance links; events that participate in provenance links cannot
        be deleted (the storage layer raises) - those callers should use the
        `cold` policy instead.
        """

    def decay_totals(self, kind: ItemKind) -> dict[str, int]:
        """Aggregate decay-state counters for items of `kind`.

        Returns a dict with keys `hot_items`, `cold_items`,
        `reinforcement_total`, `corroboration_total`,
        `contradiction_total`. The `*_total` figures sum over hot rows
        only (cold rows would conflate previously-pruned items with the
        active pool). Backends MUST compute this in the storage engine
        rather than streaming rows - the metrics surface is read on every
        scrape.
        """

    # --- consolidation helpers ---------------------------------------------

    def iter_unconsolidated_events_with_embeddings(
        self,
        *,
        model: str,
        limit: int | None = None,
        batch_size: int = 256,
    ) -> Iterator[tuple[Event, list[float]]]:
        """Stream `(event, vector)` pairs for events that have no
        provenance link yet (i.e. have not been consolidated into any
        memory item) and a stored embedding for `model`.

        Cold events are excluded - they have already been pruned from the
        active surface and consolidating them would resurrect them.

        Order: `created_at` ascending, then `id` ascending. Stable across
        replays so deterministic ordering of clusters and abstractions
        falls out automatically.
        """

    def insert_memory_item_with_provenance(
        self,
        item: MemoryItem,
        supporting_event_ids: Sequence[UUID],
        *,
        cluster: Cluster | None = None,
        embedding: Embedding | None = None,
        provenance_weights: Mapping[UUID, float] | None = None,
    ) -> list[ProvenanceLink]:
        """Atomically persist a consolidated memory item and its provenance.

        Inserts (in this order, in one transaction):
          * the optional `cluster` (so the item's `cluster_id` resolves)
          * the `item` itself
          * the optional `embedding` (typed as a `MEMORY_ITEM` embedding)
          * one provenance link per id in `supporting_event_ids`

        Raises:
          * `ValueError` if `item.level` is anything other than `EVENT`
            and `supporting_event_ids` is empty - non-event items must
            cite at least one supporting event (the README's invariant).
          * the usual storage errors on FK / CHECK violations.

        Returns the created provenance links in the same order as
        `supporting_event_ids`.
        """
