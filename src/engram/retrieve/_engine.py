"""Hierarchical retriever: coarse-to-fine over the consolidation hierarchy.

Pipeline (one `retrieve(query, params)` call):

  1. Embed and normalize the query.
  2. If `prefer == "specific"`, skip straight to the event layer (the
     same flat retrieval Stage 3 shipped, plus optional rerank).
  3. Otherwise, pull `k * candidate_multiplier` candidates from the
     `{summary, abstraction}` layer.
       a. If that layer is empty AND `prefer == "auto"`, fall through to
          the event layer. Pure-vector-store callers (no consolidate
          ever called) still get useful results.
       b. If `prefer == "general"`, emit the abstractions as-is.
       c. If `prefer == "auto"`, the per-hit decision: confidence at or
          above the threshold means emit the abstraction; below the
          threshold means drill into its supporting events, score those
          fresh against the query, and emit the top `drill_k`.
  4. Optional cross-encoder reranker reorders the merged candidate set.
  5. Slice to `k`. Each emitted `RetrievalResult.level` faithfully
     reflects what the caller is reading.
  6. If `reinforce_on_use`, fire one reinforcement signal per surfaced
     item (closes the retrieval / decay loop the README pitches).

Determinism: identical embeddings + identical storage state + identical
params -> identical results. The query embedding is normalized
deterministically; ties in scores break by `(level, item_id)` so
re-runs are stable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from engram.providers._protocols import EmbeddingProvider
from engram.retrieve._params import RetrieveParams
from engram.retrieve._reranker import RerankCandidate, Reranker
from engram.schemas import DecayState, ItemKind, Level, RetrievalResult
from engram.storage._protocol import Storage

# `(item_id, kind, *, count, now)` -> `DecayState`. Matches `DecayEngine.reinforce`.
ReinforceFn = Callable[..., DecayState]

_LOG = logging.getLogger("engram.retrieve")

_GENERALIZATION_LEVELS: tuple[Level, ...] = (Level.SUMMARY, Level.ABSTRACTION)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One pre-rerank candidate. Stays internal to the engine."""

    item_id: UUID
    item_kind: ItemKind
    level: Level
    content: str
    score: float
    supported_by: tuple[UUID, ...]


class HierarchicalRetriever:
    """Stage 6 retriever. Reads abstractions; drills when warranted.

    The retriever is stateless beyond its parameters; every read goes
    through `Storage`. A `Memory` instance owns one of these and threads
    its `retrieve(...)` call through it.
    """

    def __init__(
        self,
        storage: Storage,
        *,
        embedder: EmbeddingProvider,
        params: RetrieveParams | None = None,
        reinforce: ReinforceFn | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._params = params if params is not None else RetrieveParams()
        # `reinforce` is the same callable shape as `DecayEngine.reinforce`.
        # The engine doesn't import the decay engine directly -- `Memory`
        # plumbs in `decay.reinforce` -- so we type the seam structurally.
        self._reinforce: ReinforceFn | None = reinforce

    @property
    def params(self) -> RetrieveParams:
        return self._params

    # --- public API --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        params: RetrieveParams | None = None,
        reranker: Reranker | None = None,
    ) -> list[RetrievalResult]:
        p = params if params is not None else self._params
        query_vec = self._embedder.embed([query])[0]
        normalized = _normalize(query_vec)

        if p.prefer == "specific":
            candidates = self._candidates_from_events(normalized, p)
        else:
            candidates = self._candidates_from_generalizations(normalized, p)
            if not candidates and p.prefer == "auto":
                candidates = self._candidates_from_events(normalized, p)

        if not candidates:
            return []

        results = self._finalize(query, candidates, p, reranker)
        if p.reinforce_on_use and self._reinforce is not None:
            self._fire_reinforcement(results)
        return results

    # --- candidate stages --------------------------------------------------

    def _candidates_from_generalizations(
        self,
        query_vec: Sequence[float],
        p: RetrieveParams,
    ) -> list[_Candidate]:
        """Top-k generalizations, with optional drill into supporting events."""
        candidate_count = max(p.k * p.candidate_multiplier, p.k)
        hits = self._storage.search_memory_item_embeddings(
            query_vec,
            k=candidate_count,
            model=self._embedder.model,
            levels=_GENERALIZATION_LEVELS,
            include_cold=p.include_cold,
        )
        if not hits:
            return []

        out: list[_Candidate] = []
        for item_id, content, score in hits:
            confidence = _clip01(score)
            keep_abstraction = p.prefer == "general" or confidence >= p.confidence_threshold
            level, supports = self._level_and_supports(item_id)
            if keep_abstraction or p.drill_k == 0:
                out.append(
                    _Candidate(
                        item_id=item_id,
                        item_kind=ItemKind.MEMORY_ITEM,
                        level=level,
                        content=content,
                        score=score,
                        supported_by=supports,
                    )
                )
                continue
            # Drill: score every supporting event against the query and
            # emit the top `drill_k`.
            drilled = self._drill_supporting_events(item_id=item_id, query_vec=query_vec, p=p)
            if not drilled:
                # No drillable events (memory item with no provenance,
                # which only happens for level=event by the storage
                # invariant). Surface the abstraction.
                out.append(
                    _Candidate(
                        item_id=item_id,
                        item_kind=ItemKind.MEMORY_ITEM,
                        level=level,
                        content=content,
                        score=score,
                        supported_by=supports,
                    )
                )
                continue
            out.extend(drilled)
        return out

    def _candidates_from_events(
        self,
        query_vec: Sequence[float],
        p: RetrieveParams,
    ) -> list[_Candidate]:
        """Stage 3 flat-retrieve candidate set."""
        candidate_count = max(p.k * p.candidate_multiplier, p.k)
        hits = self._storage.search_event_embeddings(
            query_vec,
            k=candidate_count,
            model=self._embedder.model,
            include_cold=p.include_cold,
        )
        return [
            _Candidate(
                item_id=event_id,
                item_kind=ItemKind.EVENT,
                level=Level.EVENT,
                content=content,
                score=score,
                supported_by=(event_id,),
            )
            for event_id, content, score in hits
        ]

    def _drill_supporting_events(
        self,
        *,
        item_id: UUID,
        query_vec: Sequence[float],
        p: RetrieveParams,
    ) -> list[_Candidate]:
        """Score the memory item's supporting events against the query."""
        if p.drill_k == 0:
            return []
        event_ids = [e.id for e in self._storage.get_supporting_events(item_id)]
        if not event_ids:
            return []
        scored = self._storage.score_events_by_ids(
            query_vec,
            event_ids,
            model=self._embedder.model,
            include_cold=p.include_cold,
        )
        return [
            _Candidate(
                item_id=event_id,
                item_kind=ItemKind.EVENT,
                level=Level.EVENT,
                content=content,
                score=score,
                supported_by=(event_id,),
            )
            for event_id, content, score in scored[: p.drill_k]
        ]

    def _level_and_supports(self, item_id: UUID) -> tuple[Level, tuple[UUID, ...]]:
        """Return the memory item's level + provenance event ids."""
        item = self._storage.get_memory_item(item_id)
        level = item.level if item is not None else Level.SUMMARY
        supports = tuple(e.id for e in self._storage.get_supporting_events(item_id))
        return level, supports

    # --- finalize -----------------------------------------------------------

    def _finalize(
        self,
        query: str,
        candidates: list[_Candidate],
        p: RetrieveParams,
        reranker: Reranker | None,
    ) -> list[RetrievalResult]:
        # Deduplicate on (item_kind, item_id): the drill could emit an
        # event that's also surfaced via another abstraction.
        seen: dict[tuple[ItemKind, UUID], _Candidate] = {}
        for cand in candidates:
            key = (cand.item_kind, cand.item_id)
            existing = seen.get(key)
            if existing is None or cand.score > existing.score:
                seen[key] = cand
        unique = list(seen.values())

        # Stable sort by (-score, level_priority, item_id).
        unique.sort(key=lambda c: (-c.score, _LEVEL_PRIORITY[c.level], c.item_id.bytes))

        if reranker is not None and unique:
            rerank_inputs = [
                RerankCandidate(
                    result=RetrievalResult(
                        item_id=c.item_id,
                        level=c.level,
                        content=c.content,
                        confidence=_clip01(c.score),
                        score=c.score,
                        supported_by=c.supported_by,
                    ),
                    prior_score=c.score,
                )
                for c in unique
            ]
            rerank_scores = reranker.rerank(query, rerank_inputs)
            if len(rerank_scores) != len(rerank_inputs):
                raise RuntimeError(
                    f"reranker {reranker.name!r} returned "
                    f"{len(rerank_scores)} scores for {len(rerank_inputs)} candidates"
                )
            zipped = sorted(
                zip(unique, rerank_scores, strict=True),
                key=lambda pair: (-pair[1], _LEVEL_PRIORITY[pair[0].level], pair[0].item_id.bytes),
            )
            unique = [c for c, _ in zipped]

        sliced = unique[: p.k]

        return [
            RetrievalResult(
                item_id=c.item_id,
                level=c.level,
                content=c.content,
                confidence=_clip01(c.score),
                score=c.score,
                supported_by=c.supported_by,
            )
            for c in sliced
        ]

    def _fire_reinforcement(self, results: Sequence[RetrievalResult]) -> None:
        """Reinforce every surfaced item.

        Errors from individual `reinforce` calls are logged and swallowed
        so a successful retrieval is never broken by a per-item issue:

          * `KeyError` -- raced deletion of the item between search and
            reinforce.
          * `RuntimeError` -- the item is cold (only reachable when the
            caller passed `include_cold=True`). Cold items should not be
            reinforced silently; the caller has to `unmark_cold` first.
        """
        if self._reinforce is None:
            return
        for r in results:
            kind = ItemKind.EVENT if r.level is Level.EVENT else ItemKind.MEMORY_ITEM
            try:
                self._reinforce(r.item_id, kind)
            except (KeyError, RuntimeError, ValueError):
                _LOG.debug(
                    "skipping reinforcement",
                    extra={"item_id": str(r.item_id), "kind": kind.value},
                )


# `event` first ensures specific-over-general at score ties; that's the
# behavior callers using `prefer=auto` expect when an abstraction's drill
# produced an event with the same cosine.
_LEVEL_PRIORITY: dict[Level, int] = {
    Level.EVENT: 0,
    Level.SUMMARY: 1,
    Level.ABSTRACTION: 2,
}


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
