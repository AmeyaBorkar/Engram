"""Tests for the BM25 index + RRF helper in `engram.retrieve._bm25`.

The RRF helper exposes per-ranking `weights` (vs the older trick of
appending the same ranking N times) and per-ranking dedupe (so a
doc_id appearing twice in the same ranking only counts its first
rank). Both are correctness-critical for hybrid retrieval.
"""

from __future__ import annotations

import pytest

from engram.retrieve._bm25 import BM25Index, reciprocal_rank_fusion


class TestRRFWeights:
    def test_no_weights_is_uniform(self) -> None:
        a = "alpha"
        b = "beta"
        c = "gamma"
        fused_unweighted = reciprocal_rank_fusion(
            [[(a, 1.0), (b, 0.5)], [(c, 1.0), (a, 0.5)]], k=60
        )
        fused_ones = reciprocal_rank_fusion(
            [[(a, 1.0), (b, 0.5)], [(c, 1.0), (a, 0.5)]],
            k=60,
            weights=[1.0, 1.0],
        )
        # Same scores, same order.
        assert [k for k, _ in fused_unweighted] == [k for k, _ in fused_ones]
        for (_k1, s1), (_k2, s2) in zip(fused_unweighted, fused_ones, strict=True):
            assert s1 == pytest.approx(s2, abs=1e-12)

    def test_weight_scales_ranking_mass(self) -> None:
        """A 2.0 weight doubles the ranking's RRF contribution."""
        a = "a"
        fused_one = reciprocal_rank_fusion([[(a, 1.0)]], k=60, weights=[1.0])
        fused_two = reciprocal_rank_fusion([[(a, 1.0)]], k=60, weights=[2.0])
        assert fused_two[0][1] == pytest.approx(fused_one[0][1] * 2.0, abs=1e-12)

    def test_fractional_weight_works_unlike_old_int_collapse(self) -> None:
        """A 0.5 weight halves the contribution -- this is what the
        earlier `for _ in range(round(w)): rankings.append(...)` trick
        broke (round(0.5) -> 0 with banker's rounding or 1, never 0.5)."""
        a = "a"
        fused_full = reciprocal_rank_fusion([[(a, 1.0)]], k=60, weights=[1.0])
        fused_half = reciprocal_rank_fusion([[(a, 1.0)]], k=60, weights=[0.5])
        assert fused_half[0][1] == pytest.approx(fused_full[0][1] * 0.5, abs=1e-12)

    def test_zero_weight_excludes_ranking(self) -> None:
        """A zero weight drops the ranking entirely (so doc 'b' is gone)."""
        a = "a"
        b = "b"
        fused = reciprocal_rank_fusion(
            [[(a, 1.0)], [(b, 1.0)]], k=60, weights=[1.0, 0.0]
        )
        keys = [k for k, _ in fused]
        assert keys == [a]

    def test_weights_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            reciprocal_rank_fusion(
                [[("a", 1.0)], [("b", 1.0)]], k=60, weights=[1.0]
            )

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weights must be >= 0"):
            reciprocal_rank_fusion([[("a", 1.0)]], k=60, weights=[-0.5])


class TestRRFDedupePerRanking:
    def test_duplicate_within_ranking_counts_once(self) -> None:
        """If a doc_id appears twice in the same ranking, only its
        first rank contributes -- otherwise a buggy upstream could
        double-count a document's mass without adding signal."""
        a = "a"
        # `a` at rank 1 should contribute 1/(60+1); the second
        # appearance at rank 2 must be dropped.
        fused = reciprocal_rank_fusion([[(a, 1.0), (a, 0.5)]], k=60)
        assert fused == [(a, pytest.approx(1.0 / 61.0, abs=1e-12))]

    def test_dedup_is_per_ranking_not_global(self) -> None:
        """A doc_id appearing in TWO different rankings still
        contributes from each -- that's the whole point of RRF."""
        a = "a"
        fused = reciprocal_rank_fusion([[(a, 1.0)], [(a, 0.7)]], k=60)
        # Two contributions at rank 1 = 2/(60+1).
        assert fused[0][1] == pytest.approx(2.0 / 61.0, abs=1e-12)


class TestBM25IndexBasics:
    def test_empty_index_returns_empty(self) -> None:
        idx: BM25Index[str] = BM25Index()
        assert idx.search("anything", k=5) == []

    def test_unknown_query_term_returns_empty(self) -> None:
        idx: BM25Index[str] = BM25Index()
        idx.add_doc("d1", "the cat sat on the mat")
        # No overlap -> all-zero score row -> no positives.
        assert idx.search("zebra unicorn", k=5) == []

    def test_basic_ranking_prefers_term_overlap(self) -> None:
        idx: BM25Index[str] = BM25Index()
        idx.add_doc("d1", "the cat sat on the mat")
        idx.add_doc("d2", "the dog ran across the road")
        hits = idx.search("cat mat", k=2)
        assert hits[0][0] == "d1"

    def test_frozen_after_search_blocks_add_doc(self) -> None:
        idx: BM25Index[str] = BM25Index()
        idx.add_doc("d1", "first")
        idx.search("first", k=1)  # freezes
        with pytest.raises(RuntimeError, match="frozen"):
            idx.add_doc("d2", "second")

    def test_invalid_k1_b_rejected(self) -> None:
        with pytest.raises(ValueError, match="k1 must be"):
            BM25Index(k1=-1.0)
        with pytest.raises(ValueError, match="b must be"):
            BM25Index(b=1.5)

    def test_invalid_k_rejected(self) -> None:
        idx: BM25Index[str] = BM25Index()
        idx.add_doc("d1", "anything")
        with pytest.raises(ValueError, match="k must be"):
            idx.search("anything", k=0)
