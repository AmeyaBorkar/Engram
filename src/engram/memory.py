"""The `Memory` primitive.

Stages 3 + 4 + 5 + 6 surface:
  * `observe(content)` writes an event with its embedding
  * `retrieve(query, k, *, prefer=...)` returns the top-k items via
    coarse-to-fine retrieval over the hierarchy: abstractions first,
    drilling into supporting events when confidence is low
  * `reinforce` / `corroborate` / `contradict` apply decay-since-last
    plus a fresh signal and update the per-row weight
  * `tick(now=None)` runs the periodic decay sweep across the whole store
  * `consolidate(...)` clusters unconsolidated events, extracts a
    generalization per cluster via the chat provider, and links the
    resulting memory items into the hierarchy through provenance

Later stages layer in:
  - procedural memory (Stage 7): situation -> action -> outcome
  - contradiction & temporal reasoning (Stage 8)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from uuid import UUID

from engram.consolidation import (
    ConsolidationEngine,
    ConsolidationParams,
    ConsolidationResult,
    PromotionResult,
)
from engram.decay import DecayEngine, DecayMetrics, DecayParams, PrunePolicy, TickResult
from engram.decay._math import is_cold as _is_cold
from engram.providers._protocols import ChatProvider, EmbeddingProvider
from engram.retrieve import (
    HierarchicalRetriever,
    Reranker,
    RetrieveParams,
    RetrievePrefer,
)
from engram.schemas import (
    DecayState,
    Embedding,
    Event,
    ItemKind,
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
        chat: ChatProvider | None = None,
        decay_params: DecayParams | None = None,
        prune_policy: PrunePolicy = "cold",
        consolidation_params: ConsolidationParams | None = None,
        retrieve_params: RetrieveParams | None = None,
        reranker: Reranker | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._chat = chat
        self._clock: Callable[[], datetime] = clock or _utcnow
        self._engine = DecayEngine(
            storage,
            params=decay_params,
            prune_policy=prune_policy,
            clock=self._clock,
        )
        self._consolidation_params = (
            consolidation_params if consolidation_params is not None else ConsolidationParams()
        )
        if chat is not None:
            self._consolidator: ConsolidationEngine | None = ConsolidationEngine(
                storage,
                embedder=embedder,
                chat=chat,
                params=self._consolidation_params,
                clock=self._clock,
            )
        else:
            self._consolidator = None
        self._retrieve_params = retrieve_params if retrieve_params is not None else RetrieveParams()
        self._default_reranker = reranker
        self._retriever = HierarchicalRetriever(
            storage,
            embedder=embedder,
            params=self._retrieve_params,
            reinforce=self._engine.reinforce,
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
        k: int | None = None,
        *,
        prefer: RetrievePrefer | None = None,
        confidence_threshold: float | None = None,
        drill_k: int | None = None,
        include_cold: bool | None = None,
        reinforce: bool | None = None,
        reranker: Reranker | None = None,
    ) -> list[RetrievalResult]:
        """Return up to `k` items most relevant to `query`, coarse-to-fine.

        Stage 6 reads the consolidation hierarchy: top-k generalizations
        first, with optional drill-down into supporting events when the
        generalization's confidence is below `confidence_threshold` or
        the caller explicitly asked for `prefer="specific"`. The
        returned `RetrievalResult.level` reflects what was actually
        surfaced -- an abstraction, a summary, or a raw event.

        Backwards compatible with the Stage 3 surface: `retrieve(q)`
        and `retrieve(q, k=20)` work unchanged. The parameters override
        the per-Memory defaults set on the constructor; missing values
        fall back to those defaults.

        Args:
          query: free-text query.
          k: number of results to return.
          prefer: `"auto"` (default) / `"specific"` / `"general"`.
          confidence_threshold: in `auto`, abstractions at or above this
            score are returned as-is; below, the engine drills.
          drill_k: per low-confidence abstraction, how many supporting
            events to consider.
          include_cold: include items pruned by the decay engine.
          reinforce: fire reinforcement on every surfaced item. Off via
            `False` even if the Memory's default has it on (e.g. for
            an audit-style read that shouldn't influence weights).
          reranker: optional cross-encoder reranker. Defaults to the
            Memory-level reranker if one was passed to the constructor.
        """
        defaults = self._retrieve_params
        params = RetrieveParams(
            k=k if k is not None else defaults.k,
            prefer=prefer if prefer is not None else defaults.prefer,
            confidence_threshold=(
                confidence_threshold
                if confidence_threshold is not None
                else defaults.confidence_threshold
            ),
            drill_k=drill_k if drill_k is not None else defaults.drill_k,
            candidate_multiplier=defaults.candidate_multiplier,
            include_cold=include_cold if include_cold is not None else defaults.include_cold,
            reinforce_on_use=(reinforce if reinforce is not None else defaults.reinforce_on_use),
        )
        effective_reranker = reranker if reranker is not None else self._default_reranker
        return self._retriever.retrieve(query, params=params, reranker=effective_reranker)

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

    # --- Stage 5: consolidation surface ------------------------------------

    @property
    def consolidator(self) -> ConsolidationEngine:
        """Underlying `ConsolidationEngine`. Raises if no chat provider was given."""
        if self._consolidator is None:
            raise RuntimeError(
                "consolidation requires a chat provider; pass `chat=...` to Memory(...)"
            )
        return self._consolidator

    def consolidate(self, *, max_events: int | None = None) -> ConsolidationResult:
        """Run one consolidation pass.

        Pulls up to `max_events` unconsolidated events (or everything if
        None), clusters them, asks the chat provider for one
        generalization per cluster, and atomically lands a `MemoryItem`
        + provenance links per successful cluster.

        Raises `RuntimeError` if `Memory` was constructed without a chat
        provider.
        """
        return self.consolidator.consolidate(max_events=max_events)

    def promote(self, *, now: datetime | None = None) -> PromotionResult:
        """Promote stable summaries to abstractions.

        Iterates every `Level.SUMMARY` memory item and elevates those
        that meet the corroboration / contradiction / weight criteria
        configured on `ConsolidationParams.promotion_params`. Off by
        default (`enabled=False`); turn on once the corpus has had time
        to accumulate corroboration counts.
        """
        return self.consolidator.promote(now=now)


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]
