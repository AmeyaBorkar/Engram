"""The `Memory` primitive.

Stages 3 + 4 + 5 + 6 + 7 surface:
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
  * `record_procedure(situation, action, outcome)` writes a procedure
    with its situation embedding; `retrieve_procedures(situation, k)`
    returns analogous past procedures ranked by similarity x outcome
    x weight; `update_outcome(procedure_id, outcome)` flips the
    outcome and routes the change through the decay engine (success
    -> reinforce, failure -> contradict).

Later stages layer in:
  - contradiction & temporal reasoning (Stage 8)
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from uuid import UUID

from engram._otel import METRICS, span
from engram.consolidation import (
    ConsolidationEngine,
    ConsolidationParams,
    ConsolidationResult,
    PromotionResult,
)
from engram.decay import DecayEngine, DecayMetrics, DecayParams, PrunePolicy, TickResult
from engram.decay._math import is_cold as _is_cold
from engram.providers._protocols import ChatProvider, EmbeddingProvider
from engram.reconcile import Reconciler
from engram.retrieve import (
    HierarchicalRetriever,
    Reranker,
    RetrieveParams,
    RetrievePrefer,
)
from engram.retrieve._hyde import hyde_transform
from engram.retrieve._multi_query import expand_queries, reciprocal_rank_fusion
from engram.retrieve._react import react_judge
from engram.schemas import (
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
    ProcedureMatch,
    Resolution,
    RetrievalResult,
)
from engram.storage._protocol import Storage

# How each outcome maps to a decay-engine signal. SUCCESS / PARTIAL
# both reinforce (the procedure worked at least partly); FAILURE
# contradicts (it didn't work in this situation, weight it down);
# UNKNOWN is a no-op (no observation yet).
_OUTCOME_SIGNAL: dict[Outcome, str] = {
    Outcome.SUCCESS: "reinforce",
    Outcome.PARTIAL: "reinforce",
    Outcome.FAILURE: "contradict",
    Outcome.UNKNOWN: "noop",
}


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
        consolidate_chat: ChatProvider | None = None,
        decay_params: DecayParams | None = None,
        prune_policy: PrunePolicy = "cold",
        consolidation_params: ConsolidationParams | None = None,
        retrieve_params: RetrieveParams | None = None,
        reranker: Reranker | None = None,
        clock: Callable[[], datetime] | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._chat = chat
        # Use the stronger model for abstraction extraction +
        # contradiction MERGE when provided. Abstraction quality is
        # irreversible -- bad abstractions can't be recovered later.
        # The cheap retrieval / answering path can run on the
        # smaller `chat`.
        self._consolidate_chat = consolidate_chat if consolidate_chat is not None else chat
        self._tenant_id = tenant_id
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
        if self._consolidate_chat is not None:
            self._consolidator: ConsolidationEngine | None = ConsolidationEngine(
                storage,
                embedder=embedder,
                chat=self._consolidate_chat,
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
        # MERGE resolution synthesizes a new memory item via the chat
        # provider -- it's the same kind of abstraction work as
        # consolidation, so it uses the same (potentially stronger)
        # consolidate_chat.
        self._reconciler = Reconciler(
            storage,
            embedder=embedder,
            chat=self._consolidate_chat,
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

    @property
    def tenant_id(self) -> str | None:
        """Tenant scope this Memory was constructed with, if any.

        When non-None, every write (`observe`, `record_procedure`,
        consolidated abstractions) is tagged with this tenant. Stage 9a
        ships the surface; read-side enforcement (filtering retrieves
        by tenant) lands with the Postgres + RLS backend in v0.4.0.
        """
        return self._tenant_id

    def observe(self, content: str | Event) -> Event:
        """Record an event and its embedding.

        Accepts a string (wrapped into an `Event` with default fields) or
        a fully-formed `Event`. Returns the persisted `Event`.

        Durability: the event and its embedding land in a single atomic
        transaction; on successful return both are on disk.
        """
        with span("engram.memory.observe") as s:
            event = content if isinstance(content, Event) else Event(content=content)
            # Inject tenant scope if Memory was constructed with one and
            # the caller didn't pre-set it.
            if self._tenant_id is not None and event.tenant_id is None:
                event = event.model_copy(update={"tenant_id": self._tenant_id})

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

            if s is not None:
                s.set_attribute("engram.event_id", str(event.id))
                s.set_attribute("engram.embedder.model", self._embedder.model)
            METRICS.observe_call()
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
        as_of: datetime | None = None,
        hyde: bool | None = None,
        multi_query_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Return up to `k` items most relevant to `query`, coarse-to-fine.

        Stage 6 reads the consolidation hierarchy: top-k generalizations
        first, with optional drill-down into supporting events when the
        generalization's confidence is below `confidence_threshold` or
        the caller explicitly asked for `prefer="specific"`. The
        returned `RetrievalResult.level` reflects what was actually
        surfaced -- an abstraction, a summary, or a raw event.

        Stage 8 layers in temporal validity: items that have been
        invalidated by `Memory.reconcile` are excluded by default.
        `as_of=<datetime>` returns the state as of that timestamp --
        items whose validity window covers it AND whose invalidation
        (if any) happened after it.

        Backwards compatible with the Stage 3 surface: `retrieve(q)`
        and `retrieve(q, k=20)` work unchanged.

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
          as_of: temporal-as-of cutoff (Stage 8). When set, returns
            historically-correct state.
          hyde: if True and a chat provider is configured, transform
            the query into a hypothetical answer before retrieval
            (Tier 1 precision boost).
          multi_query_n: if >= 2 and a chat provider is configured,
            expand the query into N variants (original + paraphrases)
            and fuse the per-variant rankings via RRF. 1 = off.
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
            as_of=as_of if as_of is not None else defaults.as_of,
            hyde=hyde if hyde is not None else defaults.hyde,
            multi_query_n=(
                multi_query_n if multi_query_n is not None else defaults.multi_query_n
            ),
            rrf_k=defaults.rrf_k,
        )
        effective_reranker = reranker if reranker is not None else self._default_reranker
        # HyDE: transform the query into a hypothetical answer before
        # the retriever embeds it. Trades one chat call for retrieval
        # precision; provider-level caches deduplicate repeats.
        effective_query = query
        if params.hyde and self._chat is not None:
            effective_query = hyde_transform(query, self._chat)
        with span(
            "engram.memory.retrieve",
            k=params.k,
            prefer=params.prefer,
        ) as s:
            t0 = time.perf_counter()
            if params.multi_query_n >= 2 and self._chat is not None:
                results = self._multi_query_retrieve(
                    effective_query,
                    params=params,
                    reranker=effective_reranker,
                )
            else:
                results = self._retriever.retrieve(
                    effective_query, params=params, reranker=effective_reranker
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            METRICS.retrieve_call(k=params.k)
            METRICS.retrieve_latency(elapsed_ms, k=params.k)
            if s is not None:
                s.set_attribute("engram.retrieve.n_results", len(results))
                s.set_attribute("engram.retrieve.latency_ms", elapsed_ms)
                if params.as_of is not None:
                    s.set_attribute("engram.retrieve.as_of", params.as_of.isoformat())
            return results

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
        with span("engram.memory.consolidate", max_events=max_events) as s:
            result = self.consolidator.consolidate(max_events=max_events)
            METRICS.consolidate_call()
            if s is not None:
                s.set_attribute("engram.consolidate.events_processed", result.events_processed)
                s.set_attribute("engram.consolidate.clusters_formed", result.clusters_formed)
                s.set_attribute(
                    "engram.consolidate.abstractions_created", result.abstractions_created
                )
                s.set_attribute(
                    "engram.consolidate.conflicts_detected", result.conflicts_detected
                )
            return result

    def promote(self, *, now: datetime | None = None) -> PromotionResult:
        """Promote stable summaries to abstractions.

        Iterates every `Level.SUMMARY` memory item and elevates those
        that meet the corroboration / contradiction / weight criteria
        configured on `ConsolidationParams.promotion_params`. Off by
        default (`enabled=False`); turn on once the corpus has had time
        to accumulate corroboration counts.
        """
        return self.consolidator.promote(now=now)

    # --- Stage 7: procedural surface ---------------------------------------

    def record_procedure(
        self,
        situation: str,
        action: str,
        *,
        outcome: Outcome = Outcome.UNKNOWN,
        metadata: dict[str, object] | None = None,
    ) -> Procedure:
        """Record a procedure: "in this situation, this action had that outcome".

        Embeds `situation` and inserts the procedure plus its embedding
        atomically. If `outcome` is `SUCCESS`/`PARTIAL`, fires a
        reinforcement signal so the new procedure already carries the
        positive weight from the start; `FAILURE` fires a contradiction
        (so a known-bad pattern starts heavier on the cold side).
        `UNKNOWN` (the default) records with no signal -- typical when
        the agent will learn the outcome later and call `update_outcome`.

        Returns the persisted `Procedure` (with its assigned id and
        timestamps).
        """
        procedure = Procedure(
            situation=situation,
            action=action,
            outcome=outcome,
            metadata=dict(metadata) if metadata else {},
            tenant_id=self._tenant_id,
        )
        vector = self._embedder.embed([situation])[0]
        normalized = _normalize(vector)
        embedding = Embedding(
            item_id=procedure.id,
            item_kind=ItemKind.PROCEDURE,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(normalized),
        )
        with self._storage.transaction():
            self._storage.insert_procedure(procedure)
            self._storage.insert_embedding(embedding)
        self._fire_outcome_signal(procedure.id, outcome)
        return procedure

    def retrieve_procedures(
        self,
        situation: str,
        k: int = 5,
        *,
        outcomes: Sequence[Outcome] | None = None,
        include_cold: bool = False,
        reinforce: bool = True,
    ) -> list[ProcedureMatch]:
        """Find procedures whose situation matches the query, ranked.

        The ranking score is `similarity * weight * outcome_boost`:

          * `similarity` is the cosine of `situation` query vs stored.
          * `weight` is the procedure's decay-engine weight in [0, 1].
          * `outcome_boost`: SUCCESS=1.0, PARTIAL=0.8, FAILURE=0.6,
            UNKNOWN=0.7. Failures aren't suppressed -- the agent
            benefits from "this didn't work" lessons -- but successes
            outrank failures at equal similarity.

        Optional `outcomes` filter narrows the search at the index level
        (e.g. `outcomes=(Outcome.SUCCESS,)` to only return positive
        patterns).

        Reinforces every surfaced procedure if `reinforce=True` (the
        default): retrieval-as-use closes the loop between "the agent
        consulted this procedure" and "this procedure stays warm."
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query_vec = self._embedder.embed([situation])[0]
        normalized = _normalize(query_vec)
        hits = self._storage.search_procedure_embeddings(
            normalized,
            k=k,
            model=self._embedder.model,
            outcomes=outcomes,
            include_cold=include_cold,
        )
        matches: list[ProcedureMatch] = []
        for pid, _situation_text, similarity in hits:
            procedure = self._storage.get_procedure(pid)
            if procedure is None:  # pragma: no cover - raced delete
                continue
            score = _clip01(similarity) * procedure.weight * _outcome_boost(procedure.outcome)
            matches.append(
                ProcedureMatch(
                    procedure=procedure,
                    score=score,
                    similarity=similarity,
                )
            )
        matches.sort(key=lambda m: m.score, reverse=True)
        if reinforce:
            for m in matches:
                with contextlib.suppress(KeyError, RuntimeError, ValueError):
                    self._engine.reinforce(m.procedure.id, ItemKind.PROCEDURE)
        return matches

    def update_outcome(
        self,
        procedure_id: UUID,
        outcome: Outcome,
        *,
        now: datetime | None = None,
    ) -> Procedure:
        """Update a procedure's outcome and route the change through decay.

        Successful outcomes (SUCCESS / PARTIAL) call `reinforce`;
        FAILURE calls `contradict`; UNKNOWN is a no-op. Returns the
        updated `Procedure` (refetched from storage so the caller sees
        the bumped `updated_at`).

        Raises `KeyError` if the procedure id doesn't exist.
        """
        self._storage.update_procedure_outcome(procedure_id, outcome)
        self._fire_outcome_signal(procedure_id, outcome, now=now)
        result = self._storage.get_procedure(procedure_id)
        if result is None:  # pragma: no cover - raced delete
            raise KeyError(procedure_id)
        return result

    # --- E.9: iterative ReAct retrieval ------------------------------------

    def retrieve_iterative(
        self,
        query: str,
        k: int = 10,
        *,
        max_steps: int = 3,
        per_step_k: int | None = None,
        as_of: datetime | None = None,
        reinforce: bool = False,
    ) -> list[RetrievalResult]:
        """Multi-step retrieve with LLM-driven query refinement.

        Pipeline:
          1. Run base `retrieve(query)` to get an initial result set.
          2. Ask the chat provider whether the results are sufficient
             to answer the question. If yes -> return. If no -> the
             LLM emits a refined query.
          3. Run `retrieve(refined_query)` and merge new items.
          4. Repeat up to `max_steps` times.

        Returns the deduplicated union of all retrieved items, sorted
        by score descending, sliced to `k`.

        Requires a chat provider. Falls back to one-shot retrieve if
        chat is None.
        """
        if self._chat is None:
            return self.retrieve(query, k=k, as_of=as_of, reinforce=reinforce)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        leaf_k = per_step_k if per_step_k is not None else k
        seen_ids: set[UUID] = set()
        accumulated: list[RetrievalResult] = []
        current_query = query
        for _step in range(max_steps):
            step_results = self.retrieve(
                current_query,
                k=leaf_k,
                as_of=as_of,
                reinforce=False,
            )
            for r in step_results:
                if r.item_id not in seen_ids:
                    seen_ids.add(r.item_id)
                    accumulated.append(r)
            verdict = react_judge(query, accumulated, self._chat)
            if verdict.sufficient:
                break
            if not verdict.refined_query or verdict.refined_query == current_query:
                # No useful refinement -- stop iterating.
                break
            current_query = verdict.refined_query

        accumulated.sort(key=lambda r: r.score, reverse=True)
        sliced = accumulated[:k]
        if reinforce:
            for r in sliced:
                with contextlib.suppress(KeyError, RuntimeError, ValueError):
                    kind = ItemKind.EVENT if r.level is Level.EVENT else ItemKind.MEMORY_ITEM
                    self._engine.reinforce(r.item_id, kind)
        return sliced

    def _multi_query_retrieve(
        self,
        query: str,
        *,
        params: RetrieveParams,
        reranker: Reranker | None,
    ) -> list[RetrievalResult]:
        """Multi-query expansion + RRF fusion.

        Generates `params.multi_query_n - 1` paraphrases via the chat
        provider, retrieves each (with multi-query disabled to avoid
        recursion), and fuses the rankings via Reciprocal Rank Fusion.
        """
        if self._chat is None:  # pragma: no cover - upstream guard
            raise RuntimeError("multi-query retrieve requires a chat provider")
        queries = expand_queries(query, params.multi_query_n, self._chat)
        # Per-query params: turn multi-query off + the reranker off
        # (we rerank ONCE post-fusion if a reranker is given).
        leaf_params = RetrieveParams(
            k=params.k,
            prefer=params.prefer,
            confidence_threshold=params.confidence_threshold,
            drill_k=params.drill_k,
            candidate_multiplier=params.candidate_multiplier,
            include_cold=params.include_cold,
            reinforce_on_use=False,  # reinforce once post-fusion below
            as_of=params.as_of,
            hyde=False,  # already applied upstream if requested
            multi_query_n=1,
            rrf_k=params.rrf_k,
        )
        rankings: list[list[RetrievalResult]] = []
        for q in queries:
            rankings.append(
                self._retriever.retrieve(q, params=leaf_params, reranker=None)
            )
        fused = reciprocal_rank_fusion(rankings, k=params.rrf_k)
        # If the caller wants a reranker, apply it ONCE to the fused
        # top-(k*multiplier) for sharpness, then slice to k.
        if reranker is not None and fused:
            from engram.retrieve._reranker import RerankCandidate

            slice_size = min(len(fused), params.k * params.candidate_multiplier)
            cands = [
                RerankCandidate(result=r, prior_score=r.score)
                for r in fused[:slice_size]
            ]
            rerank_scores = reranker.rerank(query, cands)
            paired = sorted(
                zip(fused[:slice_size], rerank_scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            fused = [
                RetrievalResult(
                    item_id=r.item_id,
                    level=r.level,
                    content=r.content,
                    confidence=r.confidence,
                    score=score,
                    supported_by=r.supported_by,
                )
                for r, score in paired
            ]
        sliced = fused[: params.k]
        # Reinforcement-on-use closes the loop once at the surface level.
        if params.reinforce_on_use:
            for r in sliced:
                try:
                    kind = ItemKind.EVENT if r.level.value == "event" else ItemKind.MEMORY_ITEM
                    self._engine.reinforce(r.item_id, kind)
                except (KeyError, RuntimeError, ValueError):
                    pass
        return sliced

    # --- Stage 8: contradiction & temporal reasoning -----------------------

    def reconcile(
        self,
        conflict_id: UUID,
        *,
        resolution: Resolution,
        manual_winner_id: UUID | None = None,
        now: datetime | None = None,
    ) -> Conflict:
        """Resolve a detected `Conflict` and invalidate the loser.

        Picks a winner per `resolution`:

          * `PREFER_RECENT`: the later-created item wins.
          * `PREFER_TRUSTED`: the item with the higher `source_trust` wins
            (None treated as 0.0); ties fall back to PREFER_RECENT.
          * `PREFER_FREQUENT`: the item with the higher corroboration
            count wins (from decay state); ties fall back to
            PREFER_RECENT.
          * `KEEP_BOTH`: no winner; both items stay valid. The conflict
            is still marked RESOLVED so it stops surfacing on audits.
          * `MANUAL`: caller picks the winner via `manual_winner_id`
            (must be source or target of the conflict).

        The loser gets `invalidate_memory_item`d with the winner's id
        and the resolution timestamp; default `retrieve` no longer
        surfaces it, while `retrieve(..., as_of=t)` with t < invalidation
        time still does.

        Raises:
          KeyError: the conflict id does not exist.
          RuntimeError: the conflict is already resolved.
          ValueError: invalid `manual_winner_id` for MANUAL.
        """
        with span(
            "engram.memory.reconcile",
            resolution=resolution.value,
        ) as s:
            out = self._reconciler.reconcile(
                conflict_id,
                resolution=resolution,
                manual_winner_id=manual_winner_id,
                now=now,
            )
            METRICS.reconcile_call(resolution=resolution.value)
            if s is not None:
                s.set_attribute("engram.reconcile.conflict_id", str(conflict_id))
                if out.resolved_winner_id is not None:
                    s.set_attribute(
                        "engram.reconcile.winner_id", str(out.resolved_winner_id)
                    )
            return out

    def list_conflicts(
        self,
        *,
        status: ConflictStatus | None = None,
        memory_item_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Conflict]:
        """List conflicts, optionally filtered.

        `status` narrows to OPEN or RESOLVED. `memory_item_id` walks
        the conflict graph in both directions (source or target).
        Newest first.
        """
        return self._storage.list_conflicts(
            status=status,
            memory_item_id=memory_item_id,
            limit=limit,
        )

    # --- Stage 9: async surface --------------------------------------------
    #
    # Async parallel to the sync API for callers running inside an event
    # loop (web frameworks, agent platforms). The implementation routes
    # the sync body through `asyncio.to_thread` so SQLite's per-thread
    # connection model continues to apply -- no shared connections, no
    # surprise concurrency. Stage 10's Postgres backend can override
    # these on a subclass once an async-native connection pool lands.

    async def aobserve(self, content: str | Event) -> Event:
        """Async version of `observe`."""
        return await asyncio.to_thread(self.observe, content)

    async def aretrieve(
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
        as_of: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Async version of `retrieve`."""
        return await asyncio.to_thread(
            lambda: self.retrieve(
                query,
                k=k,
                prefer=prefer,
                confidence_threshold=confidence_threshold,
                drill_k=drill_k,
                include_cold=include_cold,
                reinforce=reinforce,
                reranker=reranker,
                as_of=as_of,
            )
        )

    async def areinforce(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.EVENT,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        return await asyncio.to_thread(
            lambda: self.reinforce(item_id, kind, count=count, now=now)
        )

    async def acorroborate(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.MEMORY_ITEM,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        return await asyncio.to_thread(
            lambda: self.corroborate(item_id, kind, count=count, now=now)
        )

    async def acontradict(
        self,
        item_id: UUID,
        kind: ItemKind = ItemKind.MEMORY_ITEM,
        *,
        count: int = 1,
        now: datetime | None = None,
    ) -> DecayState:
        return await asyncio.to_thread(
            lambda: self.contradict(item_id, kind, count=count, now=now)
        )

    async def aconsolidate(
        self, *, max_events: int | None = None
    ) -> ConsolidationResult:
        return await asyncio.to_thread(
            lambda: self.consolidate(max_events=max_events)
        )

    async def apromote(self, *, now: datetime | None = None) -> PromotionResult:
        return await asyncio.to_thread(lambda: self.promote(now=now))

    async def arecord_procedure(
        self,
        situation: str,
        action: str,
        *,
        outcome: Outcome = Outcome.UNKNOWN,
        metadata: dict[str, object] | None = None,
    ) -> Procedure:
        return await asyncio.to_thread(
            lambda: self.record_procedure(
                situation, action, outcome=outcome, metadata=metadata
            )
        )

    async def aretrieve_procedures(
        self,
        situation: str,
        k: int = 5,
        *,
        outcomes: Sequence[Outcome] | None = None,
        include_cold: bool = False,
        reinforce: bool = True,
    ) -> list[ProcedureMatch]:
        return await asyncio.to_thread(
            lambda: self.retrieve_procedures(
                situation,
                k,
                outcomes=outcomes,
                include_cold=include_cold,
                reinforce=reinforce,
            )
        )

    async def aupdate_outcome(
        self,
        procedure_id: UUID,
        outcome: Outcome,
        *,
        now: datetime | None = None,
    ) -> Procedure:
        return await asyncio.to_thread(
            lambda: self.update_outcome(procedure_id, outcome, now=now)
        )

    async def areconcile(
        self,
        conflict_id: UUID,
        *,
        resolution: Resolution,
        manual_winner_id: UUID | None = None,
        now: datetime | None = None,
    ) -> Conflict:
        return await asyncio.to_thread(
            lambda: self.reconcile(
                conflict_id,
                resolution=resolution,
                manual_winner_id=manual_winner_id,
                now=now,
            )
        )

    async def alist_conflicts(
        self,
        *,
        status: ConflictStatus | None = None,
        memory_item_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Conflict]:
        return await asyncio.to_thread(
            lambda: self.list_conflicts(
                status=status, memory_item_id=memory_item_id, limit=limit
            )
        )

    # --- E.7: aggregate user-state -----------------------------------------
    #
    # A single structured "what we know about this user" memory item.
    # Exactly one per tenant. Updated incrementally; surfaced
    # alongside any user-centric query.

    _USER_STATE_FLAG = "engram_user_state"

    def set_user_state(
        self,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
        supporting_event_ids: Sequence[UUID] = (),
    ) -> MemoryItem:
        """Upsert the per-tenant aggregate user-state memory item.

        If a Level.GLOBAL item with the `engram_user_state` metadata
        flag exists for this tenant, its content is replaced (and
        the embedding rebuilt). Otherwise a fresh one is created.

        `supporting_event_ids` lets the caller cite the events that
        justified the user-state. When empty AND no global item
        exists yet, a synthetic placeholder event is observed and
        used as the sole supporter (storage requires non-EVENT items
        to have at least one provenance link). When updating an
        existing user-state, the original provenance is preserved if
        no new event ids are supplied.

        Returns the persisted MemoryItem.
        """
        existing = self.get_user_state()
        # Resolve provenance.
        if supporting_event_ids:
            event_ids = list(supporting_event_ids)
        elif existing is None:
            placeholder = self.observe(f"user-state seed: {content[:200]}")
            event_ids = [placeholder.id]
        else:
            # Reuse existing provenance.
            event_ids = [e.id for e in self._storage.get_supporting_events(existing.id)]
            if not event_ids:  # pragma: no cover - defensive
                placeholder = self.observe(f"user-state seed: {content[:200]}")
                event_ids = [placeholder.id]

        merged_metadata: dict[str, object] = {self._USER_STATE_FLAG: True}
        if metadata:
            merged_metadata.update(metadata)

        # Drop the existing user-state and its embedding; we replace
        # rather than mutate so the content+embedding stay coherent.
        if existing is not None:
            self._delete_memory_item(existing.id)

        item = MemoryItem(
            level=Level.GLOBAL,
            content=content,
            metadata=merged_metadata,
            tenant_id=self._tenant_id,
        )
        vec = self._embedder.embed([content])[0]
        normalized = _normalize(vec)
        embedding = Embedding(
            item_id=item.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(normalized),
        )
        with self._storage.transaction():
            self._storage.insert_memory_item_with_provenance(
                item,
                event_ids,
                embedding=embedding,
            )
        return item

    def get_user_state(self) -> MemoryItem | None:
        """Return the per-tenant Level.GLOBAL user-state item, or None.

        Filters by tenant_id when this Memory was constructed with one.
        """
        for item in self._storage.iter_memory_items(level=Level.GLOBAL):
            if self._tenant_id is not None and item.tenant_id != self._tenant_id:
                continue
            if item.metadata.get(self._USER_STATE_FLAG):
                return item
        return None

    def _delete_memory_item(self, item_id: UUID) -> None:
        """Hard-delete a memory item + its embedding.

        Used only for the user-state replace path. Provenance links
        cascade through `ON DELETE CASCADE`. Cold sweep handles
        decay-pruned items via a separate path; this one is for the
        explicit "replace this item now" semantic.
        """
        conn = getattr(self._storage, "_connect", None)
        if conn is None:  # pragma: no cover - non-sqlite backend
            raise NotImplementedError(
                "user-state replace requires a backend exposing _connect()"
            )
        conn().execute("DELETE FROM memory_items WHERE id = ?", (item_id.bytes,))

    # --- E.8: topic layer --------------------------------------------------

    def record_topic(
        self,
        content: str,
        supporting_event_ids: Sequence[UUID],
        *,
        metadata: dict[str, object] | None = None,
        weight: float = 1.0,
    ) -> MemoryItem:
        """Record a Level.TOPIC memory item.

        Mid-grain abstraction between summary and abstraction; organized
        by topic cluster. Callers explicitly mint topic items when they
        know they're carving the corpus along topical lines (e.g. one
        topic per recurring conversation theme).

        Storage requires non-EVENT items to have at least one
        supporting event; pass the events that motivated this topic.
        Returns the persisted item.
        """
        if not supporting_event_ids:
            raise ValueError("record_topic requires at least one supporting event id")
        item = MemoryItem(
            level=Level.TOPIC,
            content=content,
            weight=weight,
            metadata=dict(metadata) if metadata else {},
            tenant_id=self._tenant_id,
        )
        vec = self._embedder.embed([content])[0]
        normalized = _normalize(vec)
        embedding = Embedding(
            item_id=item.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(normalized),
        )
        with self._storage.transaction():
            self._storage.insert_memory_item_with_provenance(
                item,
                list(supporting_event_ids),
                embedding=embedding,
            )
        return item

    # --- E.6: preference layer ---------------------------------------------

    def record_preference(
        self,
        content: str,
        *,
        source: str | None = None,
        weight: float = 1.0,
    ) -> tuple[Event, MemoryItem]:
        """Record a preference statement.

        Stores the raw `content` as an Event AND creates a
        `Level.PREFERENCE` MemoryItem with provenance to it. The
        preference layer outranks generic summaries when the caller
        retrieves with `Memory.retrieve_preferences`.

        Use when the caller knows the content is a preference. The
        `engram._preference.is_preference(text)` heuristic is exposed
        for users who want to auto-detect; this method assumes the
        decision has already been made.

        Returns `(event, preference_item)`.
        """
        event = Event(content=content, source=source, tenant_id=self._tenant_id)
        vector = self._embedder.embed([content])[0]
        normalized = _normalize(vector)
        event_emb = Embedding(
            item_id=event.id,
            item_kind=ItemKind.EVENT,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(normalized),
        )
        pref = MemoryItem(
            level=Level.PREFERENCE,
            content=content,
            weight=weight,
            tenant_id=self._tenant_id,
            metadata={"preference": {"source_event_id": str(event.id)}},
        )
        pref_emb = Embedding(
            item_id=pref.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(normalized),
        )
        with self._storage.transaction():
            self._storage.insert_event(event)
            self._storage.insert_embedding(event_emb)
            self._storage.insert_memory_item_with_provenance(
                pref,
                [event.id],
                embedding=pref_emb,
            )
        return event, pref

    def retrieve_preferences(
        self,
        query: str,
        k: int = 5,
        *,
        include_cold: bool = False,
        reinforce: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve top-k preference items only.

        Symmetric with `retrieve_procedures` (Stage 7). Filters to
        `Level.PREFERENCE` at the index level.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query_vec = self._embedder.embed([query])[0]
        normalized = _normalize(query_vec)
        hits = self._storage.search_memory_item_embeddings_as_of(
            normalized,
            k=k,
            model=self._embedder.model,
            levels=[Level.PREFERENCE],
            include_cold=include_cold,
        )
        results = [
            RetrievalResult(
                item_id=item_id,
                level=Level.PREFERENCE,
                content=content,
                confidence=_clip01(score),
                score=score,
                supported_by=(),
            )
            for item_id, content, score in hits
        ]
        if reinforce:
            for r in results:
                with contextlib.suppress(KeyError, RuntimeError, ValueError):
                    self._engine.reinforce(r.item_id, ItemKind.MEMORY_ITEM)
        return results

    def _fire_outcome_signal(
        self,
        procedure_id: UUID,
        outcome: Outcome,
        *,
        now: datetime | None = None,
    ) -> None:
        """Route an outcome change through the decay engine."""
        signal = _OUTCOME_SIGNAL[outcome]
        if signal == "reinforce":
            self._engine.reinforce(procedure_id, ItemKind.PROCEDURE, now=now)
        elif signal == "contradict":
            self._engine.contradict(procedure_id, ItemKind.PROCEDURE, now=now)
        # signal == "noop" for UNKNOWN -- intentional.


def _outcome_boost(outcome: Outcome) -> float:
    """Multiplier applied to procedure retrieval scores by outcome.

    Successes outrank failures at equal similarity, but failures stay
    surfaced (the agent can learn "this didn't work" too). The boost
    spread matters less than the relative ordering.
    """
    return {
        Outcome.SUCCESS: 1.0,
        Outcome.PARTIAL: 0.8,
        Outcome.UNKNOWN: 0.7,
        Outcome.FAILURE: 0.6,
    }[outcome]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]
