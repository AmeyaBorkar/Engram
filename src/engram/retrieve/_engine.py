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
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from engram.providers._protocols import EmbeddingProvider
from engram.retrieve._bm25 import reciprocal_rank_fusion
from engram.retrieve._mmr import mmr_select
from engram.retrieve._params import RetrieveParams
from engram.retrieve._reranker import RerankCandidate, Reranker
from engram.schemas import DecayState, ItemKind, Level, RetrievalResult
from engram.storage._protocol import Storage

# `(item_id, kind, *, count, now)` -> `DecayState`. Matches `DecayEngine.reinforce`.
ReinforceFn = Callable[..., DecayState]

_LOG = logging.getLogger("engram.retrieve")

_GENERALIZATION_LEVELS: tuple[Level, ...] = (
    Level.SUMMARY,
    Level.TOPIC,
    Level.PREFERENCE,
    Level.ABSTRACTION,
    Level.GLOBAL,
)


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
        rerank_query: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k against `query`.

        `rerank_query`, when given, is fed to the cross-encoder reranker
        instead of `query`. The HyDE pipeline embeds against the
        hypothetical answer but reranks against the user's original
        question (per the HyDE paper). When `rerank_query` is None the
        reranker sees `query`.
        """
        p = params if params is not None else self._params
        # Prefer `embed_query` when the embedder advertises it -- that
        # applies asymmetric prompts (stella `s2p_query`, e5 "query: ")
        # at query time while keeping the document-side `embed()`
        # symmetric. Falls back to the symmetric path for embedders
        # without the method, so existing providers stay bit-identical.
        embed_query = getattr(self._embedder, "embed_query", None)
        if callable(embed_query):
            query_vec = embed_query(query)
        else:
            query_vec = self._embedder.embed([query])[0]
        normalized = _normalize(query_vec)

        if p.prefer == "specific":
            candidates = self._candidates_from_events(normalized, p)
        else:
            candidates = self._candidates_from_generalizations(normalized, p)
            if not candidates and p.prefer == "auto":
                candidates = self._candidates_from_events(normalized, p)

        # Hybrid: fuse the dense candidate ranking with extra candidate
        # streams over the event content (BM25 lexical, recent-window).
        # Recovers literal-token recall and recency that the embedder
        # smoothed away. Only operates on the event layer -- the
        # abstraction/summary layer is already paraphrased text the
        # embedder handles natively, and a lexical / recency stream
        # would mostly add noise there.
        if p.bm25_weight > 0 or p.recent_window_k > 0:
            candidates = self._fuse_hybrid_sources(
                query=query,
                dense=candidates,
                p=p,
            )

        # Lexical filter: drop candidates whose content does not match
        # the configured regex. Applied BEFORE rerank so the
        # cross-encoder never wastes a pass on items the caller has
        # already ruled out.
        if p.lexical_filter is not None and candidates:
            pattern = re.compile(p.lexical_filter, re.IGNORECASE)
            candidates = [c for c in candidates if pattern.search(c.content)]

        if not candidates:
            return []

        rerank_q = rerank_query if rerank_query is not None else query
        results = self._finalize(rerank_q, candidates, p, reranker)
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
        # Route through the validity-aware path (Stage 8). `as_of=None`
        # excludes invalidated items by default; `as_of=<datetime>`
        # returns items whose validity window covers that timestamp.
        hits = self._storage.search_memory_item_embeddings_as_of(
            query_vec,
            k=candidate_count,
            model=self._embedder.model,
            as_of=p.as_of,
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

    def _fuse_hybrid_sources(
        self,
        *,
        query: str,
        dense: list[_Candidate],
        p: RetrieveParams,
    ) -> list[_Candidate]:
        """Fuse the dense candidate ranking with optional lexical (BM25)
        and recent-window event streams via Reciprocal Rank Fusion.

        Per-stream behavior:

          * Dense: always present, contributes at weight 1.0.
          * BM25: when `p.bm25_weight > 0`, runs against the storage's
            BM25 index over event content. Contributes scaled by
            `bm25_weight` directly (the weight multiplies the RRF mass,
            so `0.5` is half a ranking, `1.5` is one-and-a-half -- no
            rounding, no integer collapse).
          * Recent-window: when `p.recent_window_k > 0`, pulls the
            top-N most-recent events by `created_at` desc and ranks
            them in that order (most recent = rank 1). Contributes
            at weight 1.0.

        BM25 returns event IDs; the dense stream may be at any level
        (EVENT, SUMMARY, ABSTRACTION, ...). To make RRF actually fuse
        rather than concatenate two disjoint id sets, BM25 event hits
        are remapped to their parent `MEMORY_ITEM` keys via
        `get_supported_memory_items`. An event with multiple parents
        contributes its rank to each; an event with no parents stays
        keyed by `(EVENT, eid)` so the event-only path still works.

        Storage backends without the optional methods are no-op
        fall-throughs for the stream that requires them; non-SQLite
        backends keep working.
        """
        rankings: list[list[tuple[tuple[ItemKind, UUID], float]]] = []
        weights: list[float] = []
        # Dense ranking is always present.
        dense_ranking: list[tuple[tuple[ItemKind, UUID], float]] = [
            ((c.item_kind, c.item_id), c.score) for c in dense
        ]
        rankings.append(dense_ranking)
        weights.append(1.0)
        dense_by_key = {(c.item_kind, c.item_id): c for c in dense}

        # BM25 stream.
        bm25_by_id: dict[UUID, tuple[str, float]] = {}
        if p.bm25_weight > 0:
            bm25_search = getattr(self._storage, "bm25_search_events", None)
            if callable(bm25_search):
                pool_size = max(p.k * p.candidate_multiplier, p.k)
                try:
                    bm25_hits: list[tuple[UUID, str, float]] = bm25_search(
                        query,
                        k=pool_size,
                        k1=p.bm25_k1,
                        b=p.bm25_b,
                        include_cold=p.include_cold,
                    )
                except (ValueError, RuntimeError):  # pragma: no cover - defensive
                    bm25_hits = []
                if bm25_hits:
                    bm25_by_id = {eid: (content, score) for eid, content, score in bm25_hits}
                    # Remap BM25 event hits to their parent memory-item
                    # keys so RRF actually overlaps with the dense
                    # stream (which is keyed by memory_item id when the
                    # dense path retrieved abstractions). Events with no
                    # parent stay keyed by (EVENT, eid).
                    bm25_ranking = self._bm25_remap_to_dense_keys(
                        bm25_hits, dense_by_key
                    )
                    rankings.append(bm25_ranking)
                    weights.append(float(p.bm25_weight))

        # Recent-window stream.
        recent_by_id: dict[UUID, str] = {}
        if p.recent_window_k > 0:
            recent_fn = getattr(self._storage, "list_recent_events", None)
            if callable(recent_fn):
                try:
                    recent_hits: list[tuple[UUID, str]] = recent_fn(
                        k=p.recent_window_k, include_cold=p.include_cold
                    )
                except (ValueError, RuntimeError):  # pragma: no cover - defensive
                    recent_hits = []
                if recent_hits:
                    # The score field is dead -- RRF uses rank, not the
                    # numeric score -- but we keep a placeholder for the
                    # tuple shape rather than rewriting the ranking type.
                    recent_ranking: list[tuple[tuple[ItemKind, UUID], float]] = [
                        ((ItemKind.EVENT, eid), 0.0)
                        for eid, _ in recent_hits
                    ]
                    recent_by_id = {eid: content for eid, content in recent_hits}
                    rankings.append(recent_ranking)
                    weights.append(1.0)

        if len(rankings) == 1:
            # Nothing to fuse with; return the dense ranking unchanged.
            return dense

        fused = reciprocal_rank_fusion(rankings, k=p.rrf_k, weights=weights)
        # Materialize back into `_Candidate` records, pulling content
        # from whichever stream owns the id.
        out: list[_Candidate] = []
        for key, fused_score in fused:
            existing = dense_by_key.get(key)
            if existing is not None:
                out.append(
                    _Candidate(
                        item_id=existing.item_id,
                        item_kind=existing.item_kind,
                        level=existing.level,
                        content=existing.content,
                        score=fused_score,
                        supported_by=existing.supported_by,
                    )
                )
                continue
            kind, item_id = key
            if kind is not ItemKind.EVENT:  # pragma: no cover - defensive
                continue
            # Look in BM25 first (it carries content + a numeric score),
            # then fall back to the recent window's content. Either way
            # we synthesize an event-level candidate with the fused
            # RRF score.
            content: str | None = None
            if item_id in bm25_by_id:
                content = bm25_by_id[item_id][0]
            elif item_id in recent_by_id:
                content = recent_by_id[item_id]
            if content is None:  # pragma: no cover - defensive
                continue
            out.append(
                _Candidate(
                    item_id=item_id,
                    item_kind=ItemKind.EVENT,
                    level=Level.EVENT,
                    content=content,
                    score=fused_score,
                    supported_by=(item_id,),
                )
            )
        return out

    def _bm25_remap_to_dense_keys(
        self,
        bm25_hits: Sequence[tuple[UUID, str, float]],
        dense_by_key: dict[tuple[ItemKind, UUID], _Candidate],
    ) -> list[tuple[tuple[ItemKind, UUID], float]]:
        """Remap BM25 event hits to keys that overlap with the dense
        stream so RRF actually fuses.

        Three cases per BM25 hit, in priority order:

          1. `(EVENT, eid)` is already in `dense_by_key` -- the dense
             path is event-keyed (`prefer == "specific"` or empty
             generalizations layer); keep the event key unchanged.
          2. The event has at least one parent memory_item that's in
             `dense_by_key` -- emit one (MEMORY_ITEM, mid) ranking entry
             per such parent so the BM25 mass reinforces the dense hit.
          3. Otherwise keep the (EVENT, eid) key and let RRF surface it
             as a standalone event-level candidate.

        Multi-parent events emit multiple entries at the same rank --
        that intentionally lets a literal-token-matching event boost
        every memory_item it supports.
        """
        # Fast-path: if no dense candidate is at the memory_item layer,
        # there's nothing to remap to -- skip the per-event lookup.
        dense_has_memory_item = any(
            kind is ItemKind.MEMORY_ITEM for kind, _ in dense_by_key
        )
        if not dense_has_memory_item:
            return [((ItemKind.EVENT, eid), score) for eid, _, score in bm25_hits]
        get_supported = getattr(self._storage, "get_supported_memory_items", None)
        out: list[tuple[tuple[ItemKind, UUID], float]] = []
        for eid, _content, score in bm25_hits:
            if (ItemKind.EVENT, eid) in dense_by_key:
                out.append(((ItemKind.EVENT, eid), score))
                continue
            parents: list[UUID] = []
            if callable(get_supported):
                try:
                    parents = [m.id for m in get_supported(eid)]
                except (ValueError, RuntimeError, KeyError):  # pragma: no cover
                    parents = []
            matched = [
                pid for pid in parents
                if (ItemKind.MEMORY_ITEM, pid) in dense_by_key
            ]
            if matched:
                # Emit one entry per matched parent at the same rank
                # position. RRF dedup (per-ranking `seen` set) only fires
                # on identical keys, so distinct parents each receive the
                # rank-equivalent mass.
                for pid in matched:
                    out.append(((ItemKind.MEMORY_ITEM, pid), score))
            else:
                out.append(((ItemKind.EVENT, eid), score))
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
            # Apply the optional time-decay boost BEFORE the sort so
            # recency reshapes the final ordering rather than just
            # tweaking the scores after the fact.
            if p.recency_lambda > 0:
                rerank_scores = self._apply_recency_boost(unique, rerank_scores, p)
            zipped = sorted(
                zip(unique, rerank_scores, strict=True),
                key=lambda pair: (-pair[1], _LEVEL_PRIORITY[pair[0].level], pair[0].item_id.bytes),
            )
            unique = [c for c, _ in zipped]
            rerank_scores_sorted = [score for _, score in zipped]
            # MMR diversity rerank, applied AFTER the cross-encoder so
            # it works on calibrated relevance scores. Fetches the
            # stored embeddings for the rerank pool so we can compute
            # pairwise cosine similarities cheaply.
            if p.mmr_lambda > 0 and len(unique) > 1:
                doc_vecs = self._fetch_candidate_vectors(unique)
                pool_size = (
                    p.mmr_pool_size
                    if p.mmr_pool_size > 0
                    else p.k * max(p.candidate_multiplier, 1)
                )
                unique = mmr_select(
                    unique,
                    rerank_scores_sorted,
                    doc_vecs,
                    k=min(len(unique), pool_size),
                    lambda_=p.mmr_lambda,
                )

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

    def _apply_recency_boost(
        self,
        candidates: Sequence[_Candidate],
        scores: Sequence[float],
        p: RetrieveParams,
    ) -> list[float]:
        """Additively boost rerank scores by recency.

        `bonus = recency_lambda * exp(-days_old / decay_days)` so a
        zero-day-old hit gets `+recency_lambda`, a `decay_days`-old hit
        gets `+0.37 * recency_lambda`, and very old hits get ~0.

        Additive (rather than multiplicative) because reranker logits
        can be negative; multiplying a negative score by `(1 + λ)`
        would push recent items DOWN the ranking instead of up. The
        additive form gives every recent item the same positive bump
        regardless of its raw sign -- a recent score of `+5` becomes
        `5 + λ·decay`, a recent score of `-2` becomes `-2 + λ·decay`,
        both moving up the same amount.

        Tuning: λ is in the same units as the reranker score. For
        BGE-reranker-v2-m3, typical positive logits are 2-8 and
        meaningful gaps are ~0.5-2. A `recency_lambda` of 0.1-0.3
        nudges close ties; 1.0+ is aggressive and will reshape the
        ordering substantially.

        Reference time: `p.as_of` if set, otherwise current UTC. Items
        whose `created_at` lookup fails fall through with no bonus.
        Uses a single batched lookup for all candidate timestamps.
        """
        if p.recency_lambda <= 0:
            return list(scores)
        ref = p.as_of if p.as_of is not None else datetime.now(tz=timezone.utc)
        # Batch-fetch every created_at in a single SQL round-trip per
        # ItemKind (was N round-trips in the original implementation).
        created_at_map = self._get_created_at_batch(candidates)
        decay_days = max(p.recency_decay_days, 1.0)
        out: list[float] = []
        for c, s in zip(candidates, scores, strict=True):
            created_at = created_at_map.get(c.item_id)
            if created_at is None:
                out.append(s)
                continue
            delta_sec = (ref - created_at).total_seconds()
            days_old = max(delta_sec / 86400.0, 0.0)
            bonus = p.recency_lambda * math.exp(-days_old / decay_days)
            out.append(s + bonus)
        return out

    def _get_created_at_batch(
        self, candidates: Sequence[_Candidate]
    ) -> dict[UUID, datetime]:
        batch = getattr(self._storage, "get_created_at_batch", None)
        if callable(batch):
            return batch([(c.item_id, c.item_kind) for c in candidates])
        # Fallback for storage backends without the batched accessor.
        out: dict[UUID, datetime] = {}
        for c in candidates:
            if c.item_kind is ItemKind.EVENT:
                event = self._storage.get_event(c.item_id)
                if event is not None:
                    out[c.item_id] = event.created_at
            else:
                item = self._storage.get_memory_item(c.item_id)
                if item is not None:
                    out[c.item_id] = item.created_at
        return out

    def _fetch_candidate_vectors(
        self, candidates: Sequence[_Candidate]
    ) -> list[Sequence[float] | None]:
        """Look up the stored dense embedding for every candidate.

        Returns a list aligned with `candidates`; entries are `None`
        when the embedding lookup failed (raced delete, model
        mismatch). MMR treats `None` as "no diversity pressure" so
        a single missing embedding doesn't poison the pool.

        Batches into one SQL round-trip per `ItemKind` when the
        storage backend exposes `get_embeddings_batch` (SqliteStorage
        does); falls back to per-candidate `get_embedding` calls
        otherwise.
        """
        model = self._embedder.model
        batch = getattr(self._storage, "get_embeddings_batch", None)
        if callable(batch):
            vectors_by_id = batch(
                [(c.item_id, c.item_kind) for c in candidates],
                model=model,
            )
            return [vectors_by_id.get(c.item_id) for c in candidates]
        out: list[Sequence[float] | None] = []
        for c in candidates:
            try:
                emb = self._storage.get_embedding(c.item_id, c.item_kind, model)
            except (KeyError, RuntimeError):  # pragma: no cover - defensive
                out.append(None)
                continue
            if emb is None:
                out.append(None)
            else:
                out.append(list(emb.vector))
        return out

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
    Level.TOPIC: 2,
    Level.PREFERENCE: 3,
    Level.ABSTRACTION: 4,
    Level.GLOBAL: 5,
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
