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
    """One pre-rerank candidate. Stays internal to the engine.

    `score` is used for ordering and can switch units across stages
    (raw cosine pre-fuse, RRF score post-fuse, cross-encoder logit
    post-rerank).  `dense_score` preserves the original dense cosine
    so the downstream `RetrievalResult.confidence` stays in [0, 1]
    even after RRF fusion or reranking (audit H-46).  When the
    candidate did not come through the dense path (BM25 / recent-only
    hit), `dense_score` is None.
    """

    item_id: UUID
    item_kind: ItemKind
    level: Level
    content: str
    score: float
    supported_by: tuple[UUID, ...]
    dense_score: float | None = None


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
                        dense_score=score,
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
                        dense_score=score,
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
            BM25 index over event content. Contributes with weight
            `p.bm25_weight` (fractional values supported — they scale
            the RRF contribution linearly via `reciprocal_rank_fusion`'s
            per-ranking `weights` arg).
          * Recent-window: when `p.recent_window_k > 0`, pulls the
            top-N most-recent events by `created_at` desc and ranks
            them in that order (most recent = rank 1). Contributes at
            weight 1.0.

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
        # Audit H-45: when the dense path ran over generalizations, the
        # dense keys are (MEMORY_ITEM, item_id) but BM25/recent emit
        # (EVENT, eid).  RRF treats those as disjoint and fuses
        # nothing.  Build the event→parent-memory_item index up front
        # so we can ALSO contribute each lexical/recent event hit to
        # its parent's rank, when that parent is in the dense pool.
        dense_memory_item_ids: set[UUID] = {
            c.item_id for c in dense if c.item_kind is ItemKind.MEMORY_ITEM
        }
        bm25_pool_size = max(p.k * p.candidate_multiplier, p.k)

        # BM25 stream.
        bm25_by_id: dict[UUID, tuple[str, float]] = {}
        bm25_hits: list[tuple[UUID, str, float]] = []
        if p.bm25_weight > 0:
            bm25_search = getattr(self._storage, "bm25_search_events", None)
            if callable(bm25_search):
                try:
                    bm25_hits = bm25_search(
                        query,
                        k=bm25_pool_size,
                        k1=p.bm25_k1,
                        b=p.bm25_b,
                        include_cold=p.include_cold,
                    )
                except (ValueError, RuntimeError):  # pragma: no cover - defensive
                    bm25_hits = []
                if bm25_hits:
                    bm25_ranking: list[tuple[tuple[ItemKind, UUID], float]] = [
                        ((ItemKind.EVENT, eid), score) for eid, _, score in bm25_hits
                    ]
                    bm25_by_id = {eid: (content, score) for eid, content, score in bm25_hits}
                    # Pass weight to RRF directly — fractional weights now
                    # scale the contribution linearly instead of being
                    # silently rounded to 1 by the previous list-replication
                    # approach (which made every value in (0, 1.5] behave
                    # identically).
                    rankings.append(bm25_ranking)
                    weights.append(float(p.bm25_weight))

        # Recent-window stream.
        recent_by_id: dict[UUID, str] = {}
        recent_hits: list[tuple[UUID, str]] = []
        if p.recent_window_k > 0:
            recent_fn = getattr(self._storage, "list_recent_events", None)
            if callable(recent_fn):
                try:
                    recent_hits = recent_fn(
                        k=p.recent_window_k, include_cold=p.include_cold
                    )
                except (ValueError, RuntimeError):  # pragma: no cover - defensive
                    recent_hits = []
                if recent_hits:
                    # `1.0 - i/n` is the raw recency score; RRF uses
                    # rank not score so the per-position value here is
                    # immaterial (audit M-29).  We keep the
                    # rank-ordered tuples (most recent first) and use
                    # a stable score so debug logs show the original
                    # position weight even though RRF discards it.
                    n = len(recent_hits)
                    recent_ranking: list[tuple[tuple[ItemKind, UUID], float]] = [
                        ((ItemKind.EVENT, eid), 1.0 - i / max(n, 1))
                        for i, (eid, _) in enumerate(recent_hits)
                    ]
                    recent_by_id = {eid: content for eid, content in recent_hits}
                    rankings.append(recent_ranking)
                    weights.append(1.0)

        # H-45 remap: if the dense path is over generalizations,
        # emit a SECOND ranking per non-dense stream that maps each
        # event to its parent memory_item ids (when the parent is in
        # the dense pool).  RRF then has overlapping keys to fuse and
        # the lexical signal actually contributes to the surface.  We
        # use a single batched lookup over the union of event ids in
        # BM25 + recent so the parent map costs one round trip even
        # for a large pool.
        if dense_memory_item_ids and (bm25_hits or recent_hits):
            event_ids_to_lookup = {eid for eid, *_ in bm25_hits} | {
                eid for eid, _ in recent_hits
            }
            event_to_parents = self._event_to_dense_parents(
                event_ids_to_lookup, dense_memory_item_ids
            )
            if event_to_parents:
                if bm25_hits:
                    remapped_bm25: list[tuple[tuple[ItemKind, UUID], float]] = []
                    for eid, _content, score in bm25_hits:
                        for parent_id in event_to_parents.get(eid, ()):
                            remapped_bm25.append(
                                ((ItemKind.MEMORY_ITEM, parent_id), score)
                            )
                    if remapped_bm25:
                        rankings.append(remapped_bm25)
                        weights.append(float(p.bm25_weight))
                if recent_hits:
                    remapped_recent: list[tuple[tuple[ItemKind, UUID], float]] = []
                    n_recent = len(recent_hits)
                    for i, (eid, _content) in enumerate(recent_hits):
                        for parent_id in event_to_parents.get(eid, ()):
                            remapped_recent.append(
                                (
                                    (ItemKind.MEMORY_ITEM, parent_id),
                                    1.0 - i / max(n_recent, 1),
                                )
                            )
                    if remapped_recent:
                        rankings.append(remapped_recent)
                        weights.append(1.0)

        if len(rankings) == 1:
            # Nothing to fuse with; return the dense ranking unchanged.
            return dense

        fused = reciprocal_rank_fusion(rankings, k=p.rrf_k, weights=weights)
        # Materialize back into `_Candidate` records, pulling content
        # from whichever stream owns the id.  We propagate the
        # original dense cosine on `dense_score` (audit H-46) so the
        # final `RetrievalResult.confidence` stays in [0, 1] instead
        # of collapsing to the ~1/61 RRF magnitude when fusion runs.
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
                        dense_score=(
                            existing.dense_score
                            if existing.dense_score is not None
                            else existing.score
                        ),
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
                    # No dense cosine available for a BM25/recent-only
                    # hit; downstream `confidence` falls back to score
                    # for these via `_finalize`.
                    dense_score=None,
                )
            )
        return out

    def _event_to_dense_parents(
        self,
        event_ids: set[UUID],
        dense_memory_item_ids: set[UUID],
    ) -> dict[UUID, list[UUID]]:
        """Map each event id -> its parent MemoryItem ids that appear
        in the dense candidate pool (audit H-45).

        The lookup is done one event at a time because Storage doesn't
        expose a batched `get_supported_memory_items` accessor.  We
        keep the cost bounded by the BM25/recent pool size, which is
        `k * candidate_multiplier` -- usually <= 30 -- so the per-id
        round-trip is acceptable.  Returns an empty dict when no event
        has a parent inside `dense_memory_item_ids`.
        """
        if not event_ids or not dense_memory_item_ids:
            return {}
        out: dict[UUID, list[UUID]] = {}
        for eid in event_ids:
            try:
                parents = self._storage.get_supported_memory_items(eid)
            except (KeyError, RuntimeError):  # pragma: no cover - defensive
                continue
            matched = [p.id for p in parents if p.id in dense_memory_item_ids]
            if matched:
                out[eid] = matched
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
                dense_score=score,
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
                dense_score=score,
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
                        confidence=_clip01(
                            c.dense_score if c.dense_score is not None else c.score
                        ),
                        score=c.score,
                        supported_by=c.supported_by,
                    ),
                    prior_score=c.score,
                )
                for c in unique
            ]
            try:
                rerank_scores_raw = reranker.rerank(query, rerank_inputs)
            except (RuntimeError, ValueError):
                # Audit H-51: a flaky cross-encoder must not kill the
                # retrieve.  Fall back to the pre-rerank order.  We log
                # at WARNING because the operator wants to know — the
                # surface keeps working, but the precision boost was
                # lost.
                _LOG.warning(
                    "reranker %r failed; falling back to pre-rerank order",
                    getattr(reranker, "name", reranker.__class__.__name__),
                    exc_info=True,
                )
                rerank_scores_raw = [ri.prior_score for ri in rerank_inputs]
            if len(rerank_scores_raw) != len(rerank_inputs):
                # Audit H-51 part 2: same hostile contract as above,
                # only this time the reranker returned the wrong COUNT
                # of scores instead of raising.  Pad / truncate with
                # the prior score so a single malformed cross-encoder
                # response doesn't break every concurrent retrieve.
                _LOG.warning(
                    "reranker %r returned %d scores for %d candidates; "
                    "padding with prior_score",
                    getattr(reranker, "name", reranker.__class__.__name__),
                    len(rerank_scores_raw),
                    len(rerank_inputs),
                )
                padded = list(rerank_scores_raw) + [
                    ri.prior_score for ri in rerank_inputs[len(rerank_scores_raw):]
                ]
                rerank_scores = padded[: len(rerank_inputs)]
            else:
                rerank_scores = list(rerank_scores_raw)
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
                # Guarantee MMR returns at least k items so the final
                # `unique[:p.k]` slice can fulfil the contract.  Otherwise
                # a caller setting `mmr_pool_size=3` with `k=10` would get
                # 3 results back — H-50.
                pool_size = max(pool_size, p.k)
                unique = mmr_select(
                    unique,
                    rerank_scores_sorted,
                    doc_vecs,
                    k=min(len(unique), pool_size),
                    lambda_=p.mmr_lambda,
                )
                # MMR re-orders `unique`; rebuild `rerank_scores_sorted`
                # in the post-MMR order so the recency boost (below)
                # acts on the right per-candidate score.  Pre-fix the
                # boost only ever applied PRE-MMR, so MMR's diversity
                # calc saw boosted scores and recency drove diversity
                # rather than relevance (audit H-49).
                score_by_id = {
                    id(c): s for c, s in zipped if id(c) in {id(u) for u in unique}
                }
                # `mmr_select` returns the same _Candidate objects, so
                # identity-based lookup is safe.  Items the MMR did NOT
                # pick fall out of `unique`; we still need the score
                # for each surviving candidate in its new position.
                # Worst-case (unusual MMR impl returning copies) we
                # fall back to a content+id lookup below.
                if all(id(c) in score_by_id for c in unique):
                    rerank_scores_sorted = [score_by_id[id(c)] for c in unique]
                else:  # pragma: no cover - defensive
                    by_key = {
                        (c.item_kind, c.item_id): s for c, s in zipped
                    }
                    rerank_scores_sorted = [
                        by_key.get((c.item_kind, c.item_id), c.score) for c in unique
                    ]
            # Audit H-49: recency boost applied AFTER MMR so MMR's
            # diversity term sees the calibrated rerank scores rather
            # than scores already inflated by recency.  The sort below
            # then folds the recency bonus into the final order.
            if p.recency_lambda > 0 and unique:
                rerank_scores_sorted = self._apply_recency_boost(
                    unique, rerank_scores_sorted, p
                )
                # Re-sort once the recency bonus has been folded in.
                paired = sorted(
                    zip(unique, rerank_scores_sorted, strict=True),
                    key=lambda pair: (
                        -pair[1],
                        _LEVEL_PRIORITY[pair[0].level],
                        pair[0].item_id.bytes,
                    ),
                )
                unique = [c for c, _ in paired]
                rerank_scores_sorted = [s for _, s in paired]

        if p.min_sessions_in_topk > 0:
            unique = self._enforce_session_diversity(
                unique, k=p.k, min_sessions=p.min_sessions_in_topk
            )
        if p.within_session_oversample:
            unique = self._enforce_within_session_oversample(unique, k=p.k)

        sliced = unique[: p.k]

        return [
            RetrievalResult(
                item_id=c.item_id,
                level=c.level,
                content=c.content,
                # Audit H-46: use the original dense cosine when
                # available so confidence stays in a meaningful [0, 1]
                # range across RRF / rerank / recency stages.  Fall
                # back to `score` when the candidate came in via a
                # lexical-only path with no dense cosine.
                confidence=_clip01(
                    c.dense_score if c.dense_score is not None else c.score
                ),
                score=c.score,
                supported_by=c.supported_by,
            )
            for c in sliced
        ]

    def _enforce_session_diversity(
        self,
        candidates: list[_Candidate],
        *,
        k: int,
        min_sessions: int,
    ) -> list[_Candidate]:
        """Reorder candidates so the top-k contains >= min_sessions distinct sessions.

        Reads `session_id` from each candidate event's metadata.  When a
        candidate isn't an EVENT (e.g., abstraction) or has no session_id
        in metadata, it's treated as belonging to a synthetic
        '__unknown__' bucket -- so abstraction-heavy retrieval is
        untouched.

        Algorithm: walk candidates in current ranking order; pick the
        highest-ranked candidate from each session (one per session)
        until we have min_sessions distinct sessions OR we run out of
        sessions; then fill the remaining top-k with the highest-ranked
        unpicked candidates (relevance order).  Items beyond top-k stay
        in relevance order so they're available to callers iterating
        past k.

        No-ops when the current top-k already has >= min_sessions
        distinct sessions, when candidates carry no session_id
        metadata, or when k >= len(candidates) (no rearrangement
        possible).
        """
        if not candidates or k <= 0 or min_sessions <= 1:
            return candidates
        top = candidates[:k]
        session_ids: list[str] = []
        for c in top:
            sid = self._session_id_for(c)
            session_ids.append(sid)
        distinct_in_top = len({s for s in session_ids if s != "__unknown__"})
        if distinct_in_top >= min_sessions:
            return candidates

        # Look at candidates BEYOND top-k for under-represented sessions.
        existing_sessions = set(session_ids)
        promotions: list[_Candidate] = []
        for c in candidates[k:]:
            sid = self._session_id_for(c)
            if sid == "__unknown__" or sid in existing_sessions:
                continue
            promotions.append(c)
            existing_sessions.add(sid)
            distinct_in_top += 1
            if distinct_in_top >= min_sessions:
                break

        if not promotions:
            # No under-represented sessions reachable. Leave order alone.
            return candidates

        # For each promotion, demote the lowest-ranked top-k candidate
        # from the MOST over-represented session (preferring duplicates
        # over uniques).  Stable preference: keep the first item per
        # session, demote later same-session items.
        from collections import Counter as _Counter

        counts = _Counter(session_ids)
        keepers: list[_Candidate] = []
        demoted: list[_Candidate] = []
        seen_per_session: _Counter = _Counter()
        for c in top:
            sid = self._session_id_for(c)
            seen_per_session[sid] += 1
            # Demote when this is a non-first item from a session that
            # has multiple representatives AND we still need to make
            # room for promotions.
            if (
                len(demoted) < len(promotions)
                and counts.get(sid, 0) > 1
                and seen_per_session[sid] > 1
            ):
                demoted.append(c)
            else:
                keepers.append(c)

        # If we couldn't demote enough duplicates (rare: every top-k
        # item is from a unique session but distinct < min_sessions),
        # drop the trailing keepers in relevance order.
        while len(keepers) + len(promotions) > k and keepers:
            demoted.append(keepers.pop())

        new_top = keepers + promotions
        new_top = new_top[:k]
        # Preserve overall relevance order: sort the new top-k by
        # original index in `candidates`.
        index_of = {id(c): i for i, c in enumerate(candidates)}
        new_top.sort(key=lambda c: index_of[id(c)])
        remainder = [c for c in candidates if c not in new_top and c not in demoted]
        return new_top + demoted + remainder

    def _session_id_for(self, cand: _Candidate) -> str:
        """Fetch session_id from cand's event metadata; '__unknown__' fallback."""
        if cand.item_kind is not ItemKind.EVENT:
            return "__unknown__"
        event = self._storage.get_event(cand.item_id)
        if event is None or not event.metadata:
            return "__unknown__"
        sid = event.metadata.get("session_id")
        return str(sid) if sid else "__unknown__"

    def _is_boundary_turn(self, cand: _Candidate) -> bool:
        """True iff cand is the first or last turn of its session.

        Reads `is_first_turn` / `is_last_turn` from event metadata
        (LongMemEval ingest writes these).  Returns False for
        non-events or events without metadata.
        """
        if cand.item_kind is not ItemKind.EVENT:
            return False
        event = self._storage.get_event(cand.item_id)
        if event is None or not event.metadata:
            return False
        return bool(
            event.metadata.get("is_first_turn") or event.metadata.get("is_last_turn")
        )

    def _enforce_within_session_oversample(
        self,
        candidates: list[_Candidate],
        *,
        k: int,
    ) -> list[_Candidate]:
        """For each session in top-k, ensure its boundary turns are also in top-k.

        Looks in the wider candidate pool (beyond k) for first-turn /
        last-turn events of each session represented in the current
        top-k.  Promotes any boundary turns found by swapping out the
        lowest-ranked NON-boundary items.  No-ops when no boundary
        turns are available in the candidate pool or when k <= 0.

        Designed to complement `min_sessions_in_topk`: diversity widens
        the SET of sessions in top-k, oversampling deepens our coverage
        of EACH session that's already there with its structural
        anchors.  Stacking is safe (this runs after diversity).
        """
        if not candidates or k <= 0 or len(candidates) <= k:
            return candidates

        top = candidates[:k]
        top_ids = {c.item_id for c in top}
        sessions_in_top: set[str] = set()
        for c in top:
            sid = self._session_id_for(c)
            if sid != "__unknown__":
                sessions_in_top.add(sid)
        if not sessions_in_top:
            return candidates

        # Look beyond top-k for boundary turns whose session is in
        # the current top-k.
        promotions: list[_Candidate] = []
        for c in candidates[k:]:
            if c.item_id in top_ids:
                continue
            sid = self._session_id_for(c)
            if sid not in sessions_in_top:
                continue
            if not self._is_boundary_turn(c):
                continue
            promotions.append(c)

        if not promotions:
            return candidates

        # Demote the lowest-ranked NON-boundary items from top-k to
        # make room.  Prefer to keep boundary turns already in top-k.
        keepers: list[_Candidate] = []
        demotable: list[_Candidate] = []
        for c in top:
            if self._is_boundary_turn(c):
                keepers.append(c)
            else:
                demotable.append(c)

        # Demote from the END of demotable (lowest-ranked first).
        n_demote = min(len(promotions), len(demotable))
        keep_demotables = demotable[: len(demotable) - n_demote]
        demoted = demotable[len(demotable) - n_demote :]

        new_top = keepers + keep_demotables + promotions
        # Preserve overall relevance order within the new top-k.
        index_of = {id(c): i for i, c in enumerate(candidates)}
        new_top.sort(key=lambda c: index_of[id(c)])
        new_top = new_top[:k]
        new_top_set = {id(c) for c in new_top}
        remainder = [
            c
            for c in candidates
            if id(c) not in new_top_set and c not in demoted
        ]
        return new_top + demoted + remainder

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

        Tuning: λ is in the same units as the reranker score.  Note
        that this means the SCALE of `recency_lambda` is reranker-
        specific: BGE-reranker-v2-m3 emits logits in roughly [-8, 8]
        with meaningful gaps of 0.5-2, so a `recency_lambda` of 0.1-0.3
        nudges close ties; 1.0+ is aggressive and reshapes the
        ordering substantially.  When swapping rerankers, recalibrate
        — a value tuned for BGE will misbehave on a cross-encoder
        with a different score range (Cohere reranker emits [0, 1]
        probabilities; a λ of 0.3 there is enormous).

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


from engram._vec_math import normalize as _normalize  # noqa: E402


from engram.decay._math import clamp01 as _clip01  # noqa: E402  # alias for legacy callers
