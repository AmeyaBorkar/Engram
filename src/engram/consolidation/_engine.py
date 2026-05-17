"""Storage-aware consolidation engine.

Pipeline (one `consolidate()` call):

  1. Stream unconsolidated events for the configured embedding model
     from storage, in deterministic `(created_at, id)` order.
  2. Cluster the embeddings (HDBSCAN or agglomerative; see
     `engram.consolidation._clustering`).
  3. For each cluster:
     a. Build the abstraction prompt from the cluster's events,
        ordered by `created_at`.
     b. Call the chat provider to produce a generalization, parse and
        validate strictly.
     c. Embed the abstraction text via the same embedding model.
     d. In one storage transaction: insert the cluster, the
        `MemoryItem`, the embedding, and one provenance link per
        supporting event. Provenance links the LLM marked as
        "load-bearing" via `AbstractionResult.supports` get weight
        1.0; the rest get a smaller weight (configurable).

The engine is opt-in (Stage 5 ships `Memory.consolidate` as the public
seam). No background scheduling here - Stage 9 ships a worker that
schedules consolidate alongside decay tick.

Determinism: stable event order + deterministic clustering + the
`AbstractionResult.supports` field flowing through to provenance
weights mean two consolidate runs over the same event/embedding state
with the same chat replies produce bit-identical memory_item rows.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from typing import Any

import numpy as np

from engram.consolidation._abstraction import (
    PROMPT_VERSION,
    AbstractionParseError,
    AbstractionRequest,
    AbstractionResult,
    aextract_abstraction,
    extract_abstraction,
)
from engram.consolidation._clustering import (
    ClusterAssignment,
    ClusterParams,
    FloatMatrix,
)
from engram.consolidation._clustering import (
    cluster as cluster_vectors,
)
from engram.consolidation._contradiction import (
    CandidateRow,
    ContradictionParams,
    DetectedConflict,
    conflicts_to_metadata,
    detect_contradictions,
)
from engram.providers._protocols import ChatProvider, EmbeddingProvider
from engram.schemas import (
    Cluster,
    Conflict,
    ConflictStatus,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
)
from engram.storage._protocol import Storage

_LOG = logging.getLogger("engram.consolidation")


from engram._time import utcnow as _utcnow  # noqa: E402


# Streaming-batch size for `iter_unconsolidated_events_with_embeddings`
# consumption. Audit H-55: the engine used to materialize the entire
# generator into a Python list (and an O(N) dense matrix) before doing
# any work; for a million-event backlog under `max_events=None` that
# alone is 4-8 GB resident. Consuming in 1024-row chunks bounds peak
# memory to O(batch * D) for the duration of one cluster pass.
_STREAM_BATCH_DEFAULT: int = 1024


# Audit M-59: per-engine LRU bound for tenant-id memoization.  64 keeps
# the cache small enough to stay in L2 for hot workloads while still
# letting a moderate per-pass candidate set ride hot through every
# cluster.
_TENANT_CACHE_MAX: int = 64

# Sentinel for `dict.get` misses where None is a valid stored value.
_CACHE_MISS: Any = object()


@dataclass(frozen=True, slots=True)
class PromotionParams:
    """Parameters of the promotion pass (summary -> abstraction).

    A summary becomes an abstraction when it has been corroborated at
    least `min_corroboration` times, has zero recorded contradictions
    (`max_contradiction == 0`), and its current weight is at or above
    `min_weight`. The promotion gate consults the persistent
    `Conflict` table (status=OPEN) — recorded conflicts that the
    reconciler has resolved (status=RESOLVED) no longer block
    promotion (audit H-57).

    Off by default; opt in when the corpus has had time to accumulate
    corroboration counts.
    """

    enabled: bool = False
    min_corroboration: int = 3
    max_contradiction: int = 0
    min_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.min_corroboration < 1:
            raise ValueError(f"min_corroboration must be >= 1, got {self.min_corroboration}")
        if self.max_contradiction < 0:
            raise ValueError(f"max_contradiction must be >= 0, got {self.max_contradiction}")
        if not 0.0 <= self.min_weight <= 1.0:
            raise ValueError(f"min_weight must be in [0, 1], got {self.min_weight!r}")


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Outcome of one `promote()` call."""

    started_at: datetime
    duration_ms: float
    candidates_examined: int
    promoted: int


@dataclass(frozen=True, slots=True)
class ConsolidationParams:
    """Parameters of one consolidate run.

    `cluster_params` controls how events are grouped. `support_weight`
    is the provenance weight for events the LLM did NOT mark as
    load-bearing in `AbstractionResult.supports` (those get 1.0).
    `level` is what the produced memory item lands at - in Stage 5
    everything is `Level.SUMMARY`; the promotion pass (later commit)
    elevates stable summaries to `Level.ABSTRACTION`.

    `contradiction_params` configures the contradiction detector. By
    default the detector is disabled - turning it on adds one LLM call
    per surviving candidate per consolidate, so callers opt in
    explicitly. `promotion_params` configures the summary -> abstraction
    promotion pass; also off by default.

    `stream_batch_size` caps how many `(event, vector)` pairs we
    materialize before running cluster + abstraction + write on them.
    A larger batch yields better clustering (more candidates to merge);
    smaller batches keep peak memory bounded. The default of 1024 is a
    pragmatic middle ground for ~768-dim embeddings (≈3 MB peak).
    """

    cluster_params: ClusterParams = field(default_factory=ClusterParams)
    support_weight: float = 0.5
    level: Level = Level.SUMMARY
    abstraction_max_retries: int = 1
    contradiction_params: ContradictionParams = field(default_factory=ContradictionParams)
    promotion_params: PromotionParams = field(default_factory=PromotionParams)
    stream_batch_size: int = _STREAM_BATCH_DEFAULT

    def __post_init__(self) -> None:
        if not 0.0 <= self.support_weight <= 1.0:
            raise ValueError(f"support_weight must be in [0, 1], got {self.support_weight!r}")
        if self.level is Level.EVENT:
            raise ValueError("consolidation produces summaries/abstractions, not raw events")
        if self.stream_batch_size < 1:
            raise ValueError(
                f"stream_batch_size must be >= 1, got {self.stream_batch_size}"
            )


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Outcome of one consolidate run.

    Counts: every event that flowed into the engine (`events_processed`),
    every cluster the algorithm formed (`clusters_formed`), every
    abstraction that successfully landed in storage
    (`abstractions_created`), and every abstraction that failed even
    after retries (`abstractions_failed`). `events_consolidated` is the
    number of events that ended up in some abstraction (i.e. now have a
    provenance link). `conflicts_detected` is the number of CONTRADICT
    verdicts the judge produced across all clusters (only non-zero
    when contradiction detection is enabled).
    """

    started_at: datetime
    duration_ms: float
    events_processed: int
    clusters_formed: int
    abstractions_created: int
    abstractions_failed: int
    events_consolidated: int
    conflicts_detected: int = 0


def _chunked(
    iterator: Iterator[tuple[Event, list[float]]],
    chunk_size: int,
) -> Iterator[list[tuple[Event, list[float]]]]:
    """Yield successive `chunk_size`-sized chunks from `iterator`.

    Re-implements `itertools.batched` over the protocol's iterator
    output, but yields list rather than tuple so callers can append
    without copying. Stops when the iterator is exhausted (no padding).
    """
    iterator = iter(iterator)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


class ConsolidationEngine:
    """Storage- and provider-aware consolidation engine."""

    def __init__(
        self,
        storage: Storage,
        *,
        embedder: EmbeddingProvider,
        chat: ChatProvider,
        params: ConsolidationParams | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._chat = chat
        self._params = params if params is not None else ConsolidationParams()
        self._clock: Callable[[], datetime] = clock or _utcnow
        # Audit M-59: per-engine tenant-id cache. A regular dict gives
        # us insertion-ordered eviction (we drop the oldest entry on
        # overflow) which is sufficient bounded-cache semantics here.
        # Insert-ordering is guaranteed for dict since Python 3.7.
        self._tenant_cache: dict[Any, str | None] = {}

    @property
    def params(self) -> ConsolidationParams:
        return self._params

    def consolidate(
        self,
        *,
        max_events: int | None = None,
        pass_deadline_s: float | None = None,
    ) -> ConsolidationResult:
        """Run one consolidation pass.

        `max_events` caps how many unconsolidated events are pulled
        from storage. None means "everything available". The engine
        does its work in a single transaction per cluster (the actual
        atomic insert is per-cluster), so partial progress survives
        crashes mid-pass.

        `pass_deadline_s` (audit H-58) sets a soft wall-clock budget
        in seconds for the entire pass. Once the deadline expires the
        engine breaks out of the cluster loop at the next iteration
        (in-flight LLM calls run to completion; we don't try to cancel
        them). A None deadline means "no budget", preserving the
        pre-fix behavior.
        """
        started_at = self._clock()
        wall = time.perf_counter()
        deadline_mono = self._deadline_or_none(pass_deadline_s)

        # 1. Stream unconsolidated events + embeddings in chunks rather
        # than materializing the whole generator (audit H-55). Each
        # chunk turns into one clustering pass + per-cluster
        # abstraction + write.
        stream = self._storage.iter_unconsolidated_events_with_embeddings(
            model=self._embedder.model,
            limit=max_events,
        )

        events_processed = 0
        clusters_formed = 0
        created = 0
        failed = 0
        events_consolidated = 0
        conflicts_detected = 0

        for chunk in _chunked(stream, self._params.stream_batch_size):
            if self._deadline_exceeded(deadline_mono):
                _LOG.warning(
                    "consolidation: pass deadline exceeded after processing "
                    "%d events; stopping pass early",
                    events_processed,
                )
                break
            chunk_events = [p[0] for p in chunk]
            chunk_vectors = np.asarray([p[1] for p in chunk], dtype=np.float32)
            events_processed += len(chunk_events)

            assignments = cluster_vectors(
                chunk_vectors, params=self._params.cluster_params
            )
            clusters_formed += len(assignments)

            for assignment in assignments:
                if self._deadline_exceeded(deadline_mono):
                    _LOG.warning(
                        "consolidation: pass deadline exceeded mid-chunk "
                        "(%d clusters skipped)",
                        len(assignments) - clusters_formed,
                    )
                    break
                outcome = self._consolidate_one_cluster(
                    chunk_events, chunk_vectors, assignment
                )
                if outcome is None:
                    failed += 1
                else:
                    created += 1
                    events_consolidated += len(assignment.members)
                    conflicts_detected += outcome
            else:
                # Inner loop exited normally; check outer deadline next.
                continue
            # Inner loop hit deadline; break the outer too.
            break

        return ConsolidationResult(
            started_at=started_at,
            duration_ms=(time.perf_counter() - wall) * 1000.0,
            events_processed=events_processed,
            clusters_formed=clusters_formed,
            abstractions_created=created,
            abstractions_failed=failed,
            events_consolidated=events_consolidated,
            conflicts_detected=conflicts_detected,
        )

    async def aconsolidate(
        self,
        *,
        max_events: int | None = None,
        max_concurrent_abstractions: int = 8,
        pass_deadline_s: float | None = None,
    ) -> ConsolidationResult:
        """Async sibling of `consolidate` with parallel LLM calls.

        The per-cluster abstraction extraction is the consolidation
        bottleneck: each cluster's `extract_abstraction` is a ~1-3 s
        chat round trip, and a 30-cluster haystack stacks those into
        60-90 s of wall time on the synchronous path. This method
        gathers them all via `asyncio.gather` with a semaphore-bounded
        concurrency, turning the wall time into roughly
        `max(per_call_latency)` plus the post-write fan-in.

        `max_concurrent_abstractions` caps in-flight LLM calls so we
        respect provider rate limits. 8 is a sensible default for
        Anthropic/OpenAI; lower it to 2-4 against slower or
        rate-limited providers.

        The storage writes stay serialized -- sqlite's per-thread
        connection model can't run concurrent writes from coroutines,
        and the inserts are fast enough that there's no benefit anyway.

        Audit H-61: after `asyncio.gather` produced all abstractions,
        the prior code embedded + wrote each one serially in the
        event loop (each embed is sync + blocking, defeating the
        purpose of the async path). The fix batches all abstraction
        embeds into one async `embedder.aembed` call BEFORE the write
        loop iterates, so the only post-gather wait is the synchronous
        sqlite write per cluster.

        `pass_deadline_s` (audit H-58) sets a soft wall-clock budget
        in seconds. We honor it on the streaming-chunk boundary and
        again before the write loop. In-flight LLM calls are not
        cancelled.
        """
        started_at = self._clock()
        wall = time.perf_counter()
        deadline_mono = self._deadline_or_none(pass_deadline_s)

        # Audit H-55: stream per chunk; audit H-58: deadline-check at
        # the chunk boundary so a frozen provider can't hang the loop.
        stream = self._storage.iter_unconsolidated_events_with_embeddings(
            model=self._embedder.model,
            limit=max_events,
        )

        events_processed = 0
        clusters_formed = 0
        created = 0
        failed = 0
        events_consolidated = 0
        conflicts_detected = 0

        for chunk in _chunked(stream, self._params.stream_batch_size):
            if self._deadline_exceeded(deadline_mono):
                _LOG.warning(
                    "consolidation(async): pass deadline exceeded after "
                    "processing %d events; stopping pass early",
                    events_processed,
                )
                break
            chunk_events = [p[0] for p in chunk]
            chunk_vectors = np.asarray([p[1] for p in chunk], dtype=np.float32)
            events_processed += len(chunk_events)
            assignments = cluster_vectors(
                chunk_vectors, params=self._params.cluster_params
            )
            clusters_formed += len(assignments)
            if not assignments:
                continue

            # Parallel LLM calls bounded by a semaphore. The result list is
            # aligned with `assignments`; failures appear as None.
            semaphore = asyncio.Semaphore(max(max_concurrent_abstractions, 1))

            async def _bounded_extract(
                assignment: ClusterAssignment,
                events: list[Event] = chunk_events,
            ) -> AbstractionResult | None:
                member_indices = list(assignment.members)
                request = AbstractionRequest(
                    observations=tuple(events[i].content for i in member_indices),
                    cohesion_hint=_clamp01(assignment.cohesion),
                )
                async with semaphore:
                    try:
                        return await aextract_abstraction(
                            request,
                            self._chat,
                            max_retries=self._params.abstraction_max_retries,
                        )
                    except AbstractionParseError:
                        _LOG.warning(
                            "consolidation: abstraction failed after retries "
                            "(cluster size=%d)",
                            len(member_indices),
                        )
                        return None

            results = await asyncio.gather(
                *(_bounded_extract(assignment) for assignment in assignments)
            )

            # Audit H-61: batch the abstraction embeds in one async
            # embedder call BEFORE the write loop, so the only
            # post-gather sync work is the sqlite write per cluster.
            ok_pairs = [
                (assignment, result)
                for assignment, result in zip(assignments, results, strict=True)
                if result is not None
            ]
            for assignment, result in zip(assignments, results, strict=True):
                if result is None:
                    failed += 1
            if not ok_pairs:
                continue
            ab_texts = [r.abstraction for _, r in ok_pairs]
            ab_vecs = await self._embedder.aembed(ab_texts)
            ab_units = [_normalize(v) for v in ab_vecs]
            for (assignment, result), ab_unit in zip(
                ok_pairs, ab_units, strict=True
            ):
                if self._deadline_exceeded(deadline_mono):
                    _LOG.warning(
                        "consolidation(async): pass deadline exceeded during "
                        "write loop"
                    )
                    break
                cluster_event_count = len(list(assignment.members))
                try:
                    cluster_conflicts = self._write_cluster_result(
                        chunk_events,
                        assignment,
                        result,
                        ab_unit=ab_unit,
                    )
                except (RuntimeError, ValueError) as exc:
                    _LOG.warning(
                        "consolidation: writing cluster failed (size=%d): %s",
                        cluster_event_count,
                        exc,
                    )
                    failed += 1
                    continue
                created += 1
                events_consolidated += cluster_event_count
                conflicts_detected += cluster_conflicts
            else:
                continue
            break

        return ConsolidationResult(
            started_at=started_at,
            duration_ms=(time.perf_counter() - wall) * 1000.0,
            events_processed=events_processed,
            clusters_formed=clusters_formed,
            abstractions_created=created,
            abstractions_failed=failed,
            events_consolidated=events_consolidated,
            conflicts_detected=conflicts_detected,
        )

    @staticmethod
    def _deadline_or_none(budget_s: float | None) -> float | None:
        """Convert a soft wall-clock budget into an absolute deadline.

        Returns `time.monotonic() + budget_s` so deadline checks are
        cheap monotonic comparisons. None means "no deadline" — pre-fix
        behavior.
        """
        if budget_s is None:
            return None
        if budget_s <= 0:
            raise ValueError(
                f"pass_deadline_s must be > 0 when set, got {budget_s!r}"
            )
        return time.monotonic() + budget_s

    @staticmethod
    def _deadline_exceeded(deadline_mono: float | None) -> bool:
        return deadline_mono is not None and time.monotonic() >= deadline_mono

    def _write_cluster_result(
        self,
        events: Sequence[Event],
        assignment: ClusterAssignment,
        result: AbstractionResult,
        *,
        ab_unit: Sequence[float] | None = None,
    ) -> int:
        """Embed + (optional) contradiction detect + write a single cluster.

        Refactored out of the sync path so the async path can call it
        without duplicating the body. Returns the number of detected
        contradictions (matching `_consolidate_one_cluster`'s return).

        Audit M-56: assert cluster-member uniqueness before flattening
        into provenance weights — duplicate ids would silently
        overwrite weights and skew downstream support scoring.

        Audit H-61: callers in the async path pre-embed every cluster's
        abstraction in one batched call and pass the normalized vector
        in via `ab_unit` so we don't re-embed inside this method.
        """
        member_indices = list(assignment.members)
        if len(set(member_indices)) != len(member_indices):
            raise ValueError(
                "ClusterAssignment.members contains duplicate indices: "
                f"{member_indices}"
            )
        cluster_events = [events[i] for i in member_indices]
        # Defensive: duplicate event-ids would silently overwrite
        # `provenance_weights` -- assert at the seam so a buggy
        # caller can't sneak a duplicate past the dict-build below.
        if len({e.id for e in cluster_events}) != len(cluster_events):
            raise ValueError(
                "cluster has duplicate event ids; refusing to write provenance"
            )
        request = AbstractionRequest(
            observations=tuple(e.content for e in cluster_events),
            cohesion_hint=_clamp01(assignment.cohesion),
        )

        if ab_unit is None:
            ab_vec = self._embedder.embed([result.abstraction])[0]
            # Audit M-66 (mirrors reconcile): dim sanity check.
            if len(ab_vec) != self._embedder.dim:
                raise RuntimeError(
                    f"embedder.embed returned vector of length {len(ab_vec)}, "
                    f"expected dim={self._embedder.dim}"
                )
            ab_unit = _normalize(ab_vec)
        conflicts = self._detect_conflicts(ab_unit, result.abstraction)

        cluster = Cluster(cohesion=_clamp01(assignment.cohesion))
        metadata = _build_metadata(result, assignment, request, conflicts)
        item = MemoryItem(
            level=self._params.level,
            content=result.abstraction,
            cluster_id=cluster.id,
            metadata=metadata,
            weight=_clamp01(result.confidence),
        )
        embedding = Embedding(
            item_id=item.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(ab_unit),
        )

        supporting_indices = set(result.supports)
        provenance_weights = {
            cluster_events[local_idx].id: (
                1.0 if local_idx in supporting_indices else self._params.support_weight
            )
            for local_idx in range(len(cluster_events))
        }

        with self._storage.transaction():
            self._storage.insert_memory_item_with_provenance(
                item,
                [e.id for e in cluster_events],
                cluster=cluster,
                embedding=embedding,
                provenance_weights=provenance_weights,
            )
            # Audit H-60: vector recall can legitimately surface the
            # same candidate twice (race, cold-restart, multiple tier
            # representations). Dedup by candidate id before recording
            # so the storage UNIQUE(source_item_id, target_item_id)
            # constraint can't crash mid-transaction.
            seen_candidates: set[Any] = set()
            for dc in conflicts:
                if dc.candidate_id in seen_candidates:
                    continue
                seen_candidates.add(dc.candidate_id)
                self._storage.record_conflict(
                    Conflict(
                        source_item_id=item.id,
                        target_item_id=dc.candidate_id,
                        similarity=dc.similarity,
                        verdict=dc.verdict,
                    )
                )
        return len(conflicts)

    def _consolidate_one_cluster(
        self,
        events: Sequence[Event],
        _vectors: FloatMatrix,
        assignment: ClusterAssignment,
    ) -> int | None:
        """Run abstraction + atomic write for one cluster.

        Returns the number of detected contradictions on success. Returns
        None on parse/extraction failure (already logged); the events
        stay unconsolidated for the next pass.
        """
        member_indices = list(assignment.members)
        cluster_events = [events[i] for i in member_indices]

        request = AbstractionRequest(
            observations=tuple(e.content for e in cluster_events),
            cohesion_hint=_clamp01(assignment.cohesion),
        )

        try:
            result = extract_abstraction(
                request,
                self._chat,
                max_retries=self._params.abstraction_max_retries,
            )
        except AbstractionParseError:
            _LOG.warning(
                "consolidation: abstraction failed after retries (cluster size=%d)",
                len(member_indices),
            )
            return None

        # Embed + write via the shared helper so sync and async paths
        # share the dedup / dim-check / provenance-build logic.
        try:
            return self._write_cluster_result(events, assignment, result)
        except (RuntimeError, ValueError) as exc:
            _LOG.warning(
                "consolidation: writing cluster failed (size=%d): %s",
                len(member_indices),
                exc,
            )
            return None

    def _detect_conflicts(
        self,
        new_vec: Sequence[float],
        new_text: str,
        *,
        new_tenant_id: str | None = None,
        new_item_id: Any = None,
    ) -> list[DetectedConflict]:
        cp = self._params.contradiction_params
        if not cp.enabled:
            return []
        # Vector recall: pull top-K candidates above threshold.
        # Recall across every consolidated tier so that contradictions
        # against a PREFERENCE / TOPIC / GLOBAL (Phase E levels) are
        # also surfaced -- otherwise "user loves Python" stored as a
        # PREFERENCE is invisible to a new ABSTRACTION saying
        # "user dislikes Python".
        #
        # Audit H-54: switched from the non-`_as_of` variant to
        # `search_memory_item_embeddings_as_of(..., as_of=None)`. The
        # as_of=None mode excludes items with `invalidated_at IS NOT
        # NULL`, so a new abstraction can't be flagged as
        # contradicting an already-invalidated item (which would have
        # spuriously re-opened a conflict on stale data).
        exclude_ids = (new_item_id,) if new_item_id is not None else ()
        hits = self._storage.search_memory_item_embeddings_as_of(
            new_vec,
            k=cp.max_candidates,
            model=self._embedder.model,
            as_of=None,
            levels=(
                Level.SUMMARY,
                Level.ABSTRACTION,
                Level.PREFERENCE,
                Level.TOPIC,
                Level.GLOBAL,
            ),
            exclude_ids=exclude_ids,
        )
        candidates = [
            CandidateRow(item_id=item_id, content=content, similarity=sim)
            for item_id, content, sim in hits
            if sim >= cp.similarity_threshold
        ]
        # Audit H-54 (cont'd): the storage signature doesn't carry a
        # tenant filter (Stage 9 schema gap), so we filter at the
        # application layer when the new item declares a tenant.
        # Cross-tenant contradictions are not meaningful — they leak
        # tenants into each other's conflict graph.
        if new_tenant_id is not None and candidates:
            candidates = [
                c
                for c in candidates
                if self._matches_tenant(c.item_id, new_tenant_id)
            ]
        if not candidates:
            return []
        return detect_contradictions(
            new_abstraction=new_text,
            candidates=candidates,
            chat=self._chat,
            params=cp,
        )

    def _matches_tenant(self, item_id: Any, tenant_id: str) -> bool:
        """Best-effort check that `item_id` shares `tenant_id` with the
        candidate row. Returns True if storage can't resolve the item
        (defensive: don't drop on lookup failures).

        Audit M-59: previously this re-fetched the same candidate via
        `get_memory_item` once per cluster that recalled it. A bounded
        per-engine cache of tenant id memoizes the lookup across
        clusters within the same consolidate pass.
        """
        cached = self._tenant_cache.get(item_id, _CACHE_MISS)
        if cached is _CACHE_MISS:
            item = self._storage.get_memory_item(item_id)
            if item is None:  # pragma: no cover - raced delete
                # Don't cache unresolved lookups; the row may land in
                # storage on a future iteration.
                return True
            cached = item.tenant_id
            # Bound the cache: drop the oldest entry when we fill up so
            # very long-running engines don't grow without bound. The
            # LRU semantics aren't strictly necessary at maxsize=64 but
            # keep behavior predictable.
            if len(self._tenant_cache) >= _TENANT_CACHE_MAX:
                self._tenant_cache.pop(next(iter(self._tenant_cache)))
            self._tenant_cache[item_id] = cached
        return cached == tenant_id

    # --- promotion ---------------------------------------------------------

    def promote(self, *, now: datetime | None = None) -> PromotionResult:
        """Promote stable, frequently-corroborated summaries.

        A summary clears the bar when:
          * `corroboration_count >= min_corroboration`
          * `contradiction_count <= max_contradiction` (default 0)
          * its weight is >= `min_weight`
          * the persistent `Conflict` table records no rows with
            status=OPEN that name this item — audit H-57. Resolved
            conflicts (status=RESOLVED) no longer block promotion;
            the reconciler's "flip OPEN -> RESOLVED" is the proper
            unblock signal.

        Promoted items move from `Level.SUMMARY` to
        `Level.ABSTRACTION`; their `cluster_id`, embedding, provenance,
        and decay state stay intact (only the level changes).

        Audit M-57: cold summaries (those the decay engine has marked
        cold via `mark_cold`) are silently skipped by
        `iter_memory_items` (its `include_cold=False` default). This
        is intentional — a cold summary is not part of the active
        surface, so promoting it would resurrect it. Callers that want
        cold items to compete for promotion should explicitly warm
        them first (reinforce → recompute decay state).

        Audit H-62: this is an N+1 storage walk (one stream for the
        candidates, one `get_decay_state` per candidate, one
        `list_conflicts` per candidate that survives the decay
        filters, and one `update_memory_item_level` per promotion).
        For a million-summary corpus this is hours. The structural
        fix is a bulk-promote SQL: SELECT summaries that satisfy the
        corroboration + weight + open-conflict-count predicates and
        UPDATE memory_item.level in one statement. That requires a
        new `Storage.bulk_promote_summaries(...)` method which
        belongs in a sibling cluster (storage edits are out of scope
        here). TODO once that lands.
        """
        started = now if now is not None else self._clock()
        wall = time.perf_counter()
        pp = self._params.promotion_params
        if not pp.enabled:
            return PromotionResult(
                started_at=started,
                duration_ms=(time.perf_counter() - wall) * 1000.0,
                candidates_examined=0,
                promoted=0,
            )

        candidates_examined = 0
        promoted = 0
        for item in self._storage.iter_memory_items(level=Level.SUMMARY):
            candidates_examined += 1
            state = self._storage.get_decay_state(item.id, ItemKind.MEMORY_ITEM)
            if state is None:
                continue
            if state.corroboration_count < pp.min_corroboration:
                continue
            if state.contradiction_count > pp.max_contradiction:
                continue
            if state.weight < pp.min_weight:
                continue
            # Audit H-57: use the persistent Conflict table (status=OPEN)
            # rather than metadata snapshot. Resolved conflicts (which
            # reconcile flipped to status=RESOLVED) no longer block
            # promotion -- the metadata blob never used to get cleared.
            if self._has_open_conflicts(item.id):
                continue
            self._storage.update_memory_item_level(item.id, Level.ABSTRACTION)
            promoted += 1

        return PromotionResult(
            started_at=started,
            duration_ms=(time.perf_counter() - wall) * 1000.0,
            candidates_examined=candidates_examined,
            promoted=promoted,
        )

    def _has_open_conflicts(self, item_id: Any) -> bool:
        """True if the conflicts table records any OPEN row naming
        `item_id` as source or target (audit H-57). The reconciler's
        status flip OPEN -> RESOLVED is the proper unblock signal;
        the old `metadata["consolidation"]["conflicts"]` snapshot
        never got cleared post-reconcile so promotion was blocked
        forever once a contradiction landed.
        """
        # `limit=1` is sufficient: we only care if any OPEN row exists.
        rows = self._storage.list_conflicts(
            memory_item_id=item_id,
            status=ConflictStatus.OPEN,
            limit=1,
        )
        return bool(rows)


def _build_metadata(
    result: AbstractionResult,
    assignment: ClusterAssignment,
    request: AbstractionRequest,
    conflicts: list[DetectedConflict],
) -> dict[str, Any]:
    """Provenance/audit fields stored on each consolidated memory item."""
    return {
        "consolidation": {
            "prompt_version": PROMPT_VERSION,
            "confidence": result.confidence,
            "cohesion": _clamp01(assignment.cohesion),
            "supports": list(result.supports),
            "n_observations": len(request.observations),
            "conflicts": conflicts_to_metadata(conflicts),
        }
    }


def _has_recorded_conflicts(item: MemoryItem) -> bool:
    """Deprecated: pre-audit gate that snapshotted contradictions in
    metadata. Retained as a small helper for back-compat callers that
    want to inspect the metadata blob without consulting storage; the
    promotion gate now consults `_has_open_conflicts` which reads the
    persistent `Conflict` table (audit H-57)."""
    consolidation = item.metadata.get("consolidation") if item.metadata else None
    if not isinstance(consolidation, dict):
        return False
    conflicts = consolidation.get("conflicts")
    return bool(conflicts)


from engram._vec_math import normalize as _normalize  # noqa: E402


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
