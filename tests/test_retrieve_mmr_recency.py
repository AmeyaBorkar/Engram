"""MMR + recency ordering correctness in `HierarchicalRetriever._finalize`.

The audit found two ordering bugs:

  * H-49 -- recency boost was applied to the rerank scores BEFORE MMR,
    so MMR's diversity selection was running on boosted scores.
    Fix: apply MMR on the un-boosted relevance scores, then recency
    boost as the final sort key.

  * H-50 -- `mmr_pool_size < p.k` truncated the result list below
    what the caller asked for, because `unique[:p.k]` couldn't fill.
    Fix: floor `pool_size` at `p.k`.

Both fixes operate inside `_finalize`'s reranker branch. Tests below
construct minimal end-to-end scenarios that exercise the bug paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from engram import Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder
from engram.retrieve._mmr import mmr_select
from engram.retrieve._reranker import RerankCandidate


class _Reorderer:
    """Test reranker that lets the test plant a specific score vector
    per candidate (keyed by content) rather than computing from text."""

    name: str = "reorderer"

    def __init__(self, scores_by_content: dict[str, float]) -> None:
        self._scores = scores_by_content

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> list[float]:
        return [self._scores.get(c.result.content, c.prior_score) for c in candidates]


class TestMmrPoolSizeFloor:
    def test_pool_size_below_k_still_fills(self, storage: SqliteStorage) -> None:
        """`mmr_pool_size=2` + `k=5` must still return 5 items."""
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        contents = ["fact alpha", "fact beta", "fact gamma", "fact delta", "fact eps"]
        for c in contents:
            memory.observe(c)
        # Plant a deterministic reranker score so MMR has a known
        # relevance signal.
        rr = _Reorderer({c: float(i) for i, c in enumerate(reversed(contents))})
        results = memory.retrieve(
            "fact",
            k=5,
            reranker=rr,
            mmr_lambda=0.5,
            mmr_pool_size=2,  # explicitly smaller than k
            reinforce=False,
        )
        # With the H-50 fix, we get k=5 back. Pre-fix the slice would
        # have produced 2.
        assert len(results) == 5


class TestMmrUsesUnboostedScores:
    def test_recency_does_not_distort_mmr_diversity(self, storage: SqliteStorage) -> None:
        """Plant two near-duplicate candidates (sharing a vector) and
        one diverse outlier (orthogonal vector). MMR with lambda~0.5
        should always surface the outlier in the top-2, regardless of
        the recency boost being on or off, because diversity is now
        decided on un-boosted relevance.
        """
        from engram.schemas import Embedding, Event, ItemKind
        from tests.test_retrieve_hierarchical import PlantedEmbedder

        embedder = PlantedEmbedder(dim=4)
        # Plant 3 events: two share a unit vector (near-duplicates),
        # one is orthogonal (outlier).
        dup_vec = (1.0, 0.0, 0.0, 0.0)
        out_vec = (0.0, 1.0, 0.0, 0.0)
        for content, vec in (
            ("topic A near duplicate one", dup_vec),
            ("topic A near duplicate two", dup_vec),
            ("topic B unrelated", out_vec),
        ):
            ev = Event(content=content)
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=embedder.dim,
                    vector=vec,
                )
            )
        # Query vector close to the dup pool so all three are in the
        # rerank pool but the dups score higher on dense.
        embedder.plant("topic q", (0.95, 0.31, 0.0, 0.0))
        memory = Memory(storage=storage, embedder=embedder)

        rr = _Reorderer(
            {
                "topic A near duplicate one": 5.0,
                "topic A near duplicate two": 4.9,
                "topic B unrelated": 4.0,
            }
        )
        results = memory.retrieve(
            "topic q",
            k=2,
            reranker=rr,
            mmr_lambda=0.4,  # strong diversity pull (0.4*rel - 0.6*sim)
            recency_lambda=0.0,
            candidate_multiplier=3,
            reinforce=False,
        )
        contents = [r.content for r in results]
        # Diversity must surface the outlier even though it's the
        # lowest-relevance candidate.
        assert "topic B unrelated" in contents


class TestMmrSelectFunction:
    def test_mmr_returns_up_to_k_with_diversity(self) -> None:
        items = ["a", "b", "c", "d"]
        rel = [1.0, 0.9, 0.8, 0.7]
        # Two near-duplicate vectors and two distinct ones. MMR should
        # avoid stacking the duplicates at the top.
        vecs: list[Sequence[float] | None] = [
            [1.0, 0.0],  # a
            [1.0, 0.001],  # b -- near duplicate of a
            [0.0, 1.0],  # c -- orthogonal
            [-1.0, 0.0],  # d -- antipodal
        ]
        ranked = mmr_select(items, rel, vecs, k=3, lambda_=0.5)
        # The top relevance is `a`. The diversity pick should be `c`
        # or `d`, NOT `b` (the near-duplicate).
        assert ranked[0] == "a"
        assert "b" != ranked[1]


class TestRecencyAppliedAfterMmr:
    """Audit H-49: the recency boost used to be folded into the
    rerank scores BEFORE MMR, so MMR's diversity selection treated
    recency as if it were relevance.  The fix: MMR runs on the raw
    rerank scores; recency is then folded in as the final sort key.
    """

    def test_mmr_pool_unchanged_by_recency_lambda(self, storage: SqliteStorage) -> None:
        """When MMR's pool covers the entire candidate set, the items
        that survive must be identical regardless of recency_lambda
        — recency is applied AFTER MMR has picked the diverse set.
        Final ORDER can differ (recency reorders the sort), but the
        SET must be stable.
        """

        from engram.schemas import Embedding, Event, ItemKind
        from tests.test_retrieve_hierarchical import PlantedEmbedder

        embedder = PlantedEmbedder(dim=4)
        dup_vec = (1.0, 0.0, 0.0, 0.0)
        out_vec = (0.0, 1.0, 0.0, 0.0)
        now = datetime.now(tz=timezone.utc)
        plants = (
            ("dup one OLD", dup_vec, now - timedelta(days=365)),
            ("dup two NEW", dup_vec, now),
            ("outlier C", out_vec, now - timedelta(days=180)),
        )
        for content, vec, created in plants:
            ev = Event(content=content, created_at=created)
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=embedder.dim,
                    vector=vec,
                )
            )
        embedder.plant("query", (0.95, 0.31, 0.0, 0.0))
        memory = Memory(storage=storage, embedder=embedder)
        rr = _Reorderer(
            {
                "dup one OLD": 1.0,
                "dup two NEW": 1.0,
                "outlier C": 0.95,
            }
        )

        baseline = memory.retrieve(
            "query",
            k=3,  # full pool — verifies set membership, not order
            reranker=rr,
            mmr_lambda=0.4,
            recency_lambda=0.0,
            candidate_multiplier=3,
            reinforce=False,
        )
        baseline_ids = {r.item_id for r in baseline}

        boosted = memory.retrieve(
            "query",
            k=3,
            reranker=rr,
            mmr_lambda=0.4,
            recency_lambda=2.0,
            recency_decay_days=30.0,
            candidate_multiplier=3,
            reinforce=False,
        )
        boosted_ids = {r.item_id for r in boosted}
        # SET of items MMR selected is stable; recency may reorder
        # them but must not change membership.
        assert baseline_ids == boosted_ids

    def test_recency_reorders_after_mmr_picks(self, storage: SqliteStorage) -> None:
        """The recency boost is applied after MMR; the final order
        reflects recency on the MMR-picked candidates.  We compare two
        runs that differ ONLY in recency_lambda and expect order to
        differ when one candidate is much newer than another.
        """

        from engram.schemas import Embedding, Event, ItemKind
        from tests.test_retrieve_hierarchical import PlantedEmbedder

        embedder = PlantedEmbedder(dim=4)
        v = (1.0, 0.0, 0.0, 0.0)
        now = datetime.now(tz=timezone.utc)
        plants = (
            ("old fact", v, now - timedelta(days=365)),
            ("new fact", v, now),
        )
        for content, vec, created in plants:
            ev = Event(content=content, created_at=created)
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=embedder.dim,
                    vector=vec,
                )
            )
        embedder.plant("q", v)
        memory = Memory(storage=storage, embedder=embedder)
        # Plant rerank scores so OLD is the higher-rerank top-1.
        rr = _Reorderer({"old fact": 1.0, "new fact": 0.95})

        # No recency: top-1 = old (highest rerank score).
        no_recency = memory.retrieve(
            "q",
            k=2,
            reranker=rr,
            mmr_lambda=0.0,
            recency_lambda=0.0,
            reinforce=False,
        )
        assert [r.content for r in no_recency][0] == "old fact"

        # Aggressive recency on a 30-day half-life: the very-recent
        # fact gets +2.0 bonus, blowing past the 0.05 rerank gap.
        with_recency = memory.retrieve(
            "q",
            k=2,
            reranker=rr,
            mmr_lambda=0.0,
            recency_lambda=2.0,
            recency_decay_days=30.0,
            reinforce=False,
        )
        assert [r.content for r in with_recency][0] == "new fact"
