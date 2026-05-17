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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class PromotionParams:
    """Parameters of the promotion pass (summary -> abstraction).

    A summary becomes an abstraction when it has been corroborated at
    least `min_corroboration` times, has zero recorded contradictions
    (`max_contradiction == 0`), and its current weight is at or above
    `min_weight`. Recorded conflicts in
    `metadata['consolidation']['conflicts']` block promotion outright -
    a contradicted summary should be reconciled (Stage 8) before it
    rises in the hierarchy.

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

    `pass_deadline_s` (H-58): an aggregate wall-clock budget for the
    sync `consolidate()` pass.  When set, the engine stops dispatching
    new per-cluster LLM calls once `time.monotonic() -
    started_at_monotonic` exceeds the budget; clusters not yet
    processed are left for the next pass.  `None` disables the budget
    (the historical behavior).  The deadline is best-effort -- the
    current cluster's chat call is allowed to finish even if it
    overshoots, because the `ChatProvider` protocol does not yet
    expose a per-call timeout argument (changing that surface is
    Cluster A2's territory).
    """

    cluster_params: ClusterParams = field(default_factory=ClusterParams)
    support_weight: float = 0.5
    level: Level = Level.SUMMARY
    abstraction_max_retries: int = 1
    contradiction_params: ContradictionParams = field(default_factory=ContradictionParams)
    promotion_params: PromotionParams = field(default_factory=PromotionParams)
    pass_deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.support_weight <= 1.0:
            raise ValueError(f"support_weight must be in [0, 1], got {self.support_weight!r}")
        if self.level is Level.EVENT:
            raise ValueError("consolidation produces summaries/abstractions, not raw events")
        if self.pass_deadline_s is not None and self.pass_deadline_s <= 0.0:
            raise ValueError(
                f"pass_deadline_s must be > 0 or None, got {self.pass_deadline_s!r}"
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

    @property
    def params(self) -> ConsolidationParams:
        return self._params

    def consolidate(
        self,
        *,
        max_events: int | None = None,
    ) -> ConsolidationResult:
        """Run one consolidation pass.

        `max_events` caps how many unconsolidated events are pulled
        from storage. None means "everything available". The engine
        does its work in a single transaction per cluster (the actual
        atomic insert is per-cluster), so partial progress survives
        crashes mid-pass.

        Memory contract (H-55): the engine builds a dense `(N, dim)`
        matrix of all sampled embeddings and an `(N, N)` cosine matrix
        inside `cluster_vectors` for the agglomerative path.  For
        million-event backlogs the right pattern is to call
        `consolidate(max_events=K)` in a loop with `K` sized to fit in
        RAM (e.g. 5k-20k); clustering across a chunked subset still
        produces deterministic, useful summaries because clusters
        forming inside a single chunk are intra-temporal anyway.  The
        agglomerative similarity matrix is O(N^2 dim) bytes -- 10k
        events at fp32 is ~400 MiB.  The HDBSCAN path is roughly
        O(N log N) and tolerates much larger N.
        """
        started_at = self._clock()
        wall = time.perf_counter()

        # 1. Pull unconsolidated events + embeddings.
        #
        # The storage iterator streams in 256-row chunks; we still
        # materialize the union because clustering needs the full
        # matrix.  The materialization is bounded by `max_events` (the
        # caller's contract); for million-event backlogs, see the
        # docstring's loop-with-cap pattern.
        events: list[Event] = []
        vector_rows: list[list[float]] = []
        for event, vec in self._storage.iter_unconsolidated_events_with_embeddings(
            model=self._embedder.model,
            limit=max_events,
        ):
            events.append(event)
            vector_rows.append(vec)
        if not events:
            return ConsolidationResult(
                started_at=started_at,
                duration_ms=(time.perf_counter() - wall) * 1000.0,
                events_processed=0,
                clusters_formed=0,
                abstractions_created=0,
                abstractions_failed=0,
                events_consolidated=0,
            )

        vectors = np.asarray(vector_rows, dtype=np.float32)
        # Free the Python lists' duplicate references now that the
        # numpy matrix owns the floats; the storage rows already
        # released their pointer.
        del vector_rows

        # 2. Cluster.
        assignments = cluster_vectors(vectors, params=self._params.cluster_params)

        # 3. Per-cluster abstraction + atomic write.
        # H-58: aggregate deadline.  When `pass_deadline_s` is set the
        # loop stops dispatching new clusters once the wall budget is
        # spent.  The currently-running chat call is allowed to finish
        # (per-call cancellation requires a protocol-level
        # `chat(..., timeout=...)` knob which the ChatProvider seam
        # does not yet expose).
        deadline = (
            time.monotonic() + self._params.pass_deadline_s
            if self._params.pass_deadline_s is not None
            else None
        )
        created = 0
        failed = 0
        events_consolidated = 0
        conflicts_detected = 0
        for assignment in assignments:
            if deadline is not None and time.monotonic() >= deadline:
                _LOG.warning(
                    "consolidation: pass_deadline_s=%.3f exhausted; "
                    "%d cluster(s) deferred to the next pass",
                    self._params.pass_deadline_s,
                    len(assignments) - (created + failed),
                )
                break
            outcome = self._consolidate_one_cluster(events, vectors, assignment)
            if outcome is None:
                failed += 1
            else:
                created += 1
                events_consolidated += len(assignment.members)
                conflicts_detected += outcome

        return ConsolidationResult(
            started_at=started_at,
            duration_ms=(time.perf_counter() - wall) * 1000.0,
            events_processed=len(events),
            clusters_formed=len(assignments),
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
        """
        started_at = self._clock()
        wall = time.perf_counter()

        # Stream the storage iterator's chunks directly into the
        # working buffers; see `consolidate()` for the memory contract.
        events: list[Event] = []
        vector_rows: list[list[float]] = []
        for event, vec in self._storage.iter_unconsolidated_events_with_embeddings(
            model=self._embedder.model,
            limit=max_events,
        ):
            events.append(event)
            vector_rows.append(vec)
        if not events:
            return ConsolidationResult(
                started_at=started_at,
                duration_ms=(time.perf_counter() - wall) * 1000.0,
                events_processed=0,
                clusters_formed=0,
                abstractions_created=0,
                abstractions_failed=0,
                events_consolidated=0,
            )

        vectors = np.asarray(vector_rows, dtype=np.float32)
        del vector_rows
        assignments = cluster_vectors(vectors, params=self._params.cluster_params)

        # Parallel LLM calls bounded by a semaphore. The result list is
        # aligned with `assignments`; failures appear as None.
        semaphore = asyncio.Semaphore(max(max_concurrent_abstractions, 1))

        async def _bounded_extract(
            assignment: ClusterAssignment,
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
                        "consolidation: abstraction failed after retries (cluster size=%d)",
                        len(member_indices),
                    )
                    return None

        results = await asyncio.gather(
            *(_bounded_extract(assignment) for assignment in assignments)
        )

        # M-90 / H-61: batch every successful abstraction's embedding
        # into a single embed() call rather than embedding one text per
        # cluster inline in the write loop.  For a 30-cluster pass this
        # collapses 30 sequential RPC round-trips into one batched call.
        # The sync embedder is run on a worker thread via
        # `asyncio.to_thread` so it doesn't block the event loop -- the
        # original implementation called `self._embedder.embed` from
        # inside the post-gather write loop, which is on-loop work that
        # serialized everything we just parallelized.
        ok_assignments: list[ClusterAssignment] = []
        ok_results: list[AbstractionResult] = []
        for assignment, result in zip(assignments, results, strict=True):
            if result is not None:
                ok_assignments.append(assignment)
                ok_results.append(result)
        precomputed_vectors: list[list[float]] = []
        if ok_results:
            texts = [r.abstraction for r in ok_results]
            raw_vectors = await asyncio.to_thread(self._embedder.embed, texts)
            precomputed_vectors = [_normalize(v) for v in raw_vectors]

        created = 0
        failed = sum(1 for r in results if r is None)
        events_consolidated = 0
        conflicts_detected = 0
        ok_iter = iter(zip(ok_assignments, ok_results, precomputed_vectors, strict=True))
        # Walk the original ordering so the failure counter matches what
        # the sync path would produce.
        for assignment, result in zip(assignments, results, strict=True):
            if result is None:
                continue
            cluster_event_count = len(assignment.members)
            _, _, vec = next(ok_iter)
            try:
                cluster_conflicts = self._write_cluster_result(
                    events,
                    assignment,
                    result,
                    precomputed_vector=vec,
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

        return ConsolidationResult(
            started_at=started_at,
            duration_ms=(time.perf_counter() - wall) * 1000.0,
            events_processed=len(events),
            clusters_formed=len(assignments),
            abstractions_created=created,
            abstractions_failed=failed,
            events_consolidated=events_consolidated,
            conflicts_detected=conflicts_detected,
        )

    def _write_cluster_result(
        self,
        events: Sequence[Event],
        assignment: ClusterAssignment,
        result: AbstractionResult,
        *,
        precomputed_vector: Sequence[float] | None = None,
    ) -> int:
        """Embed + (optional) contradiction detect + write a single cluster.

        Refactored out of the sync path so the async path can call it
        without duplicating the body. Returns the number of detected
        contradictions (matching `_consolidate_one_cluster`'s return).

        `precomputed_vector`: M-90 / H-61 -- when the caller has already
        batched the abstraction embeddings (the async path does so to
        avoid one embed() RPC per cluster), pass the unit-normalized
        vector here to skip the per-call embed.
        """
        member_indices = _unique_members(assignment)
        cluster_events = [events[i] for i in member_indices]
        request = AbstractionRequest(
            observations=tuple(e.content for e in cluster_events),
            cohesion_hint=_clamp01(assignment.cohesion),
        )

        if precomputed_vector is None:
            ab_vec = self._embedder.embed([result.abstraction])[0]
            ab_unit = _normalize(ab_vec)
        else:
            ab_unit = list(precomputed_vector)
        conflicts = _dedupe_conflicts(self._detect_conflicts(ab_unit, result.abstraction))

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
            for dc in conflicts:
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
        member_indices = _unique_members(assignment)
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

        # Embed the abstraction text via the same embedding model.
        ab_vec = self._embedder.embed([result.abstraction])[0]
        ab_unit = _normalize(ab_vec)

        # Contradiction detection (vector recall + LLM judge). The recall
        # can surface the same candidate twice if it appears at multiple
        # levels (e.g. a SUMMARY and a TOPIC with identical text); the
        # dedupe prevents duplicate Conflict rows from a single pass.
        conflicts = _dedupe_conflicts(self._detect_conflicts(ab_unit, result.abstraction))

        # Build storage rows.
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

        # Provenance weights: events the LLM marked as load-bearing get 1.0;
        # the rest get the configurable `support_weight`.
        supporting_indices = set(result.supports)
        provenance_weights = {
            cluster_events[local_idx].id: (
                1.0 if local_idx in supporting_indices else self._params.support_weight
            )
            for local_idx in range(len(cluster_events))
        }

        # Stage 5 wrote conflicts into metadata only. Stage 8 promotes
        # them to first-class storage rows so the reconciler can manage
        # their lifecycle. The metadata blob stays for back-compat (the
        # promotion gate still reads it).
        with self._storage.transaction():
            self._storage.insert_memory_item_with_provenance(
                item,
                [e.id for e in cluster_events],
                cluster=cluster,
                embedding=embedding,
                provenance_weights=provenance_weights,
            )
            for dc in conflicts:
                self._storage.record_conflict(
                    Conflict(
                        source_item_id=item.id,
                        target_item_id=dc.candidate_id,
                        similarity=dc.similarity,
                        verdict=dc.verdict,
                    )
                )
        return len(conflicts)

    def _detect_conflicts(
        self,
        new_vec: Sequence[float],
        new_text: str,
    ) -> list[DetectedConflict]:
        cp = self._params.contradiction_params
        if not cp.enabled:
            return []
        # Vector recall: pull top-K candidates above threshold.
        #
        # H-54: use the `_as_of` variant with `as_of=None` so already-
        # invalidated items (a previous reconcile may have resolved them
        # into a successor) do NOT come back as fresh contradiction
        # candidates.  The non-`_as_of` variant returns rows regardless
        # of `invalidated_at`, so we'd burn judge calls comparing the
        # new abstraction against tombstones whose successors are
        # already in the active surface.
        #
        # Recall across every consolidated tier so that contradictions
        # against a PREFERENCE / TOPIC / GLOBAL (Phase E levels) are
        # also surfaced -- otherwise "user loves Python" stored as a
        # PREFERENCE is invisible to a new ABSTRACTION saying
        # "user dislikes Python".
        #
        # M-03 TODO: the judge prompt currently has no temporal-scope
        # hint, so "Alice lived in Paris" vs "Alice lives in Tokyo"
        # can be classified as CONTRADICT when the two statements are
        # actually a consistent timeline ("lived" = past, "lives" =
        # present).  The prompt fix lands in a follow-up touching
        # `prompts/judge_v1.txt`; until then callers running on
        # narrative corpora should expect occasional spurious
        # contradictions on tense-shifted facts.
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
        )
        candidates = [
            CandidateRow(item_id=item_id, content=content, similarity=sim)
            for item_id, content, sim in hits
            if sim >= cp.similarity_threshold
        ]
        if not candidates:
            return []
        return detect_contradictions(
            new_abstraction=new_text,
            candidates=candidates,
            chat=self._chat,
            params=cp,
        )

    # --- promotion ---------------------------------------------------------

    def promote(self, *, now: datetime | None = None) -> PromotionResult:
        """Promote stable, frequently-corroborated summaries.

        A summary clears the bar when:
          * `corroboration_count >= min_corroboration`
          * `contradiction_count <= max_contradiction` (default 0)
          * its weight is >= `min_weight`
          * its metadata records no recorded conflicts (Stage 5
            contradiction detection blocks promotion outright)

        Promoted items move from `Level.SUMMARY` to
        `Level.ABSTRACTION`; their `cluster_id`, embedding, provenance,
        and decay state stay intact (only the level changes).

        Scope (M-57): cold summaries are silently ignored.
        `iter_memory_items(level=Level.SUMMARY)` defaults to
        `include_cold=False`; an item the decay engine has already
        cooled is not promotion-eligible because the corroboration
        signal that earned the promotion is no longer active.  Pass
        `include_cold=True` at the storage layer if you specifically
        need to reanimate cold summaries (admin tooling).

        Cost (H-62): the implementation is one storage round-trip per
        candidate -- `get_decay_state` runs for every hot summary,
        producing an N+1 pattern.  At a million summaries this is the
        promotion pass's wall-time floor (~hours over a remote SQLite
        backend).  A bulk-promote SQL (`UPDATE memory_items SET level
        = 'abstraction' WHERE id IN (SELECT m.id FROM memory_items m
        JOIN decay_state d ...)`) would collapse it to O(1) but the
        storage API does not yet expose a `bulk_promote_summaries`
        seam.  When that ships, replace the loop here.

        Side effect (M-58): promoted summaries keep their `cluster_id`
        and embedding row, but the vector-index level denormalization
        is now stale (the row's stored `level` says ABSTRACTION while
        the index shard still keys it under SUMMARY).  The default
        retrieve path re-reads `level` from `memory_items` per row, so
        the staleness is cosmetic; downstream consumers that rely on
        the index's level shard (none in v0.4.0) would need a
        post-pass `storage.refresh_level_index()` call -- the seam
        doesn't exist yet, so this is documented for future-me.
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
            if self._has_open_conflicts(item):
                continue
            self._storage.update_memory_item_level(item.id, Level.ABSTRACTION)
            promoted += 1

        return PromotionResult(
            started_at=started,
            duration_ms=(time.perf_counter() - wall) * 1000.0,
            candidates_examined=candidates_examined,
            promoted=promoted,
        )

    def _has_open_conflicts(self, item: MemoryItem) -> bool:
        """Return True if `item` has any persistent Conflict row at
        status=OPEN.

        H-57: the previous implementation checked
        `metadata['consolidation']['conflicts']` -- a static snapshot
        taken when the contradiction detector fired.  No code path
        clears that metadata after reconcile resolves the conflict, so
        once an item was flagged it was permanently blocked from
        promotion even after the reconciler had picked it as the winner
        and removed the actual conflict.  Consulting the persistent
        `Conflict` table at status=OPEN means promotion correctly
        re-opens after reconcile.  We only need to know whether ANY
        open row exists, so `limit=1` keeps the storage cost bounded.
        """
        return bool(
            self._storage.list_conflicts(
                status=ConflictStatus.OPEN,
                memory_item_id=item.id,
                limit=1,
            )
        )


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


def _unique_members(assignment: ClusterAssignment) -> list[int]:
    """Return the cluster's members as a deterministic ordered list of
    unique indices.

    M-56: `ClusterAssignment.members` is typed as `tuple[int, ...]` but
    the schema does not enforce uniqueness, so a buggy clustering pass
    could surface duplicate indices. Each duplicate would map to the
    same supporting `event.id`, and the provenance-weights dict would
    silently overwrite the earlier weight -- a subtle determinism bug
    that produces different `load-bearing` weights depending on member
    order. Assert here so the failure is loud at the engine boundary
    instead of silently truncated downstream.
    """
    members = list(assignment.members)
    if len(set(members)) != len(members):
        raise ValueError(
            f"cluster {assignment!r} has duplicate member indices; "
            "clustering must return unique row positions"
        )
    return members


def _dedupe_conflicts(conflicts: list[DetectedConflict]) -> list[DetectedConflict]:
    """Drop duplicate conflicts pointing at the same candidate id.

    H-60: vector recall can surface the same `candidate_id` more than
    once when the storage layer has multiple rows representing the
    "same" idea at different levels (e.g. SUMMARY + TOPIC with
    identical text and identical embeddings). Recording two
    `Conflict` rows with the same `(source_item_id, target_item_id)`
    pair raises a storage IntegrityError on the second insert -- the
    schema has a uniqueness constraint on the pair. Dedupe up front
    so the contradiction-detection pass survives that case; keep the
    first occurrence (highest similarity since the recall returns in
    score-desc order).
    """
    seen: set[Any] = set()
    out: list[DetectedConflict] = []
    for dc in conflicts:
        if dc.candidate_id in seen:
            continue
        seen.add(dc.candidate_id)
        out.append(dc)
    return out


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
