"""The `Memory` primitive.

Stages 3 + 4 surface:
  * `observe(content)` writes an event with its embedding
  * `retrieve(query, k)` returns the top-k events by cosine similarity,
    excluding pruned items by default
  * `reinforce` / `corroborate` / `contradict` apply decay-since-last
    plus a fresh signal and update the per-row weight
  * `tick(now=None)` runs the periodic decay sweep across the whole store

Later stages layer in:
  - consolidation (Stage 5): events cluster into abstractions
  - hierarchical retrieve (Stage 6): coarse-to-fine reads
  - procedural memory (Stage 7): situation -> action -> outcome
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from uuid import UUID

from engram.decay import DecayEngine, DecayMetrics, DecayParams, PrunePolicy, TickResult
from engram.decay._math import is_cold as _is_cold
from engram.providers._protocols import EmbeddingProvider
from engram.schemas import (
    DecayState,
    Embedding,
    Event,
    ItemKind,
    Level,
    RetrievalResult,
)
from engram.storage._protocol import Storage


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Memory:
    """Hierarchical memory with consolidation and principled decay.

    Stage 3 ships `observe` and `retrieve`. Stage 4 layers in `reinforce`
    / `corroborate` / `contradict` and the `tick` sweep, all driven by a
    `DecayEngine`. The decay engine is always present (with library
    defaults); callers who want pure-vector-store behavior can simply
    never call the signal methods - decay-only updates of untouched items
    only happen when the caller invokes `tick`.
    """

    def __init__(
        self,
        *,
        storage: Storage,
        embedder: EmbeddingProvider,
        decay_params: DecayParams | None = None,
        prune_policy: PrunePolicy = "cold",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._clock: Callable[[], datetime] = clock or _utcnow
        self._engine = DecayEngine(
            storage,
            params=decay_params,
            prune_policy=prune_policy,
            clock=self._clock,
        )

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder

    @property
    def decay(self) -> DecayEngine:
        return self._engine

    def observe(self, content: str | Event) -> Event:
        """Record an event and its embedding.

        Accepts a string (wrapped into an `Event` with default fields) or
        a fully-formed `Event`. Returns the persisted `Event`.

        Durability: the event and its embedding land in a single atomic
        transaction; on successful return both are on disk.
        """
        event = content if isinstance(content, Event) else Event(content=content)

        vector = self._embedder.embed([event.content])[0]
        normalized = _normalize(vector)
        embedding = Embedding(
            item_id=event.id,
            item_kind=ItemKind.EVENT,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(normalized),
        )

        with self._storage.transaction():
            self._storage.insert_event(event)
            self._storage.insert_embedding(embedding)

        return event

    def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        include_cold: bool = False,
    ) -> list[RetrievalResult]:
        """Return the top-k events most similar to `query` by cosine.

        Stage 3 is flat: every result is `level=EVENT`, `supported_by` is
        the singleton `(event_id,)`, and `confidence` equals the cosine
        score (already in `[0, 1]` for unit-norm vectors when both sides
        are aligned; clamped here just in case).

        Stage 4: items pruned by the decay engine are excluded by
        default. Pass `include_cold=True` for audit / inspection flows.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        query_vec = self._embedder.embed([query])[0]
        normalized = _normalize(query_vec)

        hits = self._storage.search_event_embeddings(
            normalized,
            k=k,
            model=self._embedder.model,
            include_cold=include_cold,
        )

        return [
            RetrievalResult(
                item_id=event_id,
                level=Level.EVENT,
                content=content,
                confidence=_clip01(score),
                score=score,
                supported_by=(event_id,),
            )
            for event_id, content, score in hits
        ]

    # --- Stage 4: decay surface --------------------------------------------

    def reinforce(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.EVENT,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        """Record a useful-recall reinforcement for an item.

        Returns the post-update `DecayState` so callers can react (e.g.
        log when an item just crossed the threshold).
        """
        return self._engine.reinforce(item_id, kind, count=count, now=now)

    def corroborate(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.MEMORY_ITEM,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        """Record a corroboration for an item (other evidence agrees)."""
        return self._engine.corroborate(item_id, kind, count=count, now=now)

    def contradict(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.MEMORY_ITEM,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        """Record a contradiction for an item."""
        return self._engine.contradict(item_id, kind, count=count, now=now)

    def tick(self, *, now: datetime | None = None) -> TickResult:
        """Run the periodic decay sweep over every hot item."""
        return self._engine.tick(now=now)

    async def tick_async(self, *, now: datetime | None = None) -> TickResult:
        """Async wrapper around `tick`."""
        return await self._engine.tick_async(now=now)

    def is_cold(self, item_id: UUID, kind: ItemKind = ItemKind.EVENT) -> bool:
        """True if the item is below the decay threshold (cold)."""
        state = self._storage.get_decay_state(item_id, kind)
        if state is None:
            return False
        if state.cold_at is not None:
            return True
        return _is_cold(state.weight, self._engine.params)

    def metrics(self) -> DecayMetrics:
        """Snapshot of the decay engine's observable counters."""
        return self._engine.metrics()


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
