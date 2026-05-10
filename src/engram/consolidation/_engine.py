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
from engram.providers._protocols import ChatProvider, EmbeddingProvider
from engram.schemas import (
    Cluster,
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
class ConsolidationParams:
    """Parameters of one consolidate run.

    `cluster_params` controls how events are grouped. `support_weight`
    is the provenance weight for events the LLM did NOT mark as
    load-bearing in `AbstractionResult.supports` (those get 1.0).
    `level` is what the produced memory item lands at - in Stage 5
    everything is `Level.SUMMARY`; the promotion pass (later commit)
    elevates stable summaries to `Level.ABSTRACTION`.
    """

    cluster_params: ClusterParams = field(default_factory=ClusterParams)
    support_weight: float = 0.5
    level: Level = Level.SUMMARY
    abstraction_max_retries: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.support_weight <= 1.0:
            raise ValueError(f"support_weight must be in [0, 1], got {self.support_weight!r}")
        if self.level is Level.EVENT:
            raise ValueError("consolidation produces summaries/abstractions, not raw events")


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Outcome of one consolidate run.

    Counts: every event that flowed into the engine (`events_processed`),
    every cluster the algorithm formed (`clusters_formed`), every
    abstraction that successfully landed in storage
    (`abstractions_created`), and every abstraction that failed even
    after retries (`abstractions_failed`). `events_consolidated` is the
    number of events that ended up in some abstraction (i.e. now have a
    provenance link).
    """

    started_at: datetime
    duration_ms: float
    events_processed: int
    clusters_formed: int
    abstractions_created: int
    abstractions_failed: int
    events_consolidated: int


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
        """
        started_at = self._clock()
        wall = time.perf_counter()

        # 1. Pull unconsolidated events + embeddings.
        pairs = list(
            self._storage.iter_unconsolidated_events_with_embeddings(
                model=self._embedder.model,
                limit=max_events,
            )
        )
        if not pairs:
            return ConsolidationResult(
                started_at=started_at,
                duration_ms=(time.perf_counter() - wall) * 1000.0,
                events_processed=0,
                clusters_formed=0,
                abstractions_created=0,
                abstractions_failed=0,
                events_consolidated=0,
            )

        events = [p[0] for p in pairs]
        vectors = np.asarray([p[1] for p in pairs], dtype=np.float32)

        # 2. Cluster.
        assignments = cluster_vectors(vectors, params=self._params.cluster_params)

        # 3. Per-cluster abstraction + atomic write.
        created = 0
        failed = 0
        events_consolidated = 0
        for assignment in assignments:
            ok = self._consolidate_one_cluster(events, vectors, assignment)
            if ok:
                created += 1
                events_consolidated += len(assignment.members)
            else:
                failed += 1

        return ConsolidationResult(
            started_at=started_at,
            duration_ms=(time.perf_counter() - wall) * 1000.0,
            events_processed=len(events),
            clusters_formed=len(assignments),
            abstractions_created=created,
            abstractions_failed=failed,
            events_consolidated=events_consolidated,
        )

    def _consolidate_one_cluster(
        self,
        events: Sequence[Event],
        _vectors: FloatMatrix,
        assignment: ClusterAssignment,
    ) -> bool:
        """Run abstraction + atomic write for one cluster.

        Returns True on success. False on parse/extraction failure (already
        logged); the events stay unconsolidated for the next pass.
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
            return False

        # Embed the abstraction text via the same embedding model.
        ab_vec = self._embedder.embed([result.abstraction])[0]
        ab_unit = _normalize(ab_vec)

        # Build storage rows.
        cluster = Cluster(cohesion=_clamp01(assignment.cohesion))
        metadata = _build_metadata(result, assignment, request)
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

        self._storage.insert_memory_item_with_provenance(
            item,
            [e.id for e in cluster_events],
            cluster=cluster,
            embedding=embedding,
            provenance_weights=provenance_weights,
        )
        return True


def _build_metadata(
    result: AbstractionResult,
    assignment: ClusterAssignment,
    request: AbstractionRequest,
) -> dict[str, Any]:
    """Provenance/audit fields stored on each consolidated memory item."""
    return {
        "consolidation": {
            "prompt_version": PROMPT_VERSION,
            "confidence": result.confidence,
            "cohesion": _clamp01(assignment.cohesion),
            "supports": list(result.supports),
            "n_observations": len(request.observations),
        }
    }


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
