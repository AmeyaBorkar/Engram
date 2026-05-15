"""Tests for the standalone BM25 implementation."""

from __future__ import annotations

import pytest
from benchmarks.baselines._bm25 import BM25


def test_bm25_empty_returns_empty() -> None:
    b = BM25()
    assert b.topk("anything", k=10) == []


def test_bm25_single_doc() -> None:
    b = BM25()
    b.add("the quick brown fox")
    out = b.topk("fox", k=1)
    assert len(out) == 1
    idx, score = out[0]
    assert idx == 0
    assert score > 0


def test_bm25_ranks_more_relevant_higher() -> None:
    b = BM25()
    b.add("a totally unrelated document about the weather")
    b.add("the cat sat on the mat")
    b.add("nothing about cats here")
    out = b.topk("cat sat on mat", k=3)
    assert out[0][0] == 1


def test_bm25_term_only_in_some_docs() -> None:
    b = BM25()
    b.add("alpha beta gamma")
    b.add("delta epsilon zeta")
    out = b.topk("alpha", k=2)
    # Only doc 0 contains 'alpha'; zero-scored docs are dropped from
    # the result (matches the engine's BM25Index behaviour — pure
    # zero-score 'matches' add no information to RRF or downstream
    # consumers, and including them silently inflated baselines).
    assert len(out) == 1
    assert out[0][0] == 0
    assert out[0][1] > 0


def test_bm25_tokenizer_handles_punctuation_and_case() -> None:
    b = BM25()
    b.add("The quick, brown FOX!")
    b.add("nothing matching")
    # Query is lowercase punctuation-free but should still hit doc 0.
    out = b.topk("FOX brown", k=2)
    assert out[0][0] == 0


def test_bm25_rejects_invalid_k() -> None:
    b = BM25()
    b.add("hello")
    with pytest.raises(ValueError, match="k must be"):
        b.topk("hello", k=0)


def test_bm25_rejects_invalid_k1() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25(k1=0)


def test_bm25_rejects_invalid_b() -> None:
    with pytest.raises(ValueError, match="b must be"):
        BM25(b=2.0)


def test_bm25_avgdl_grows_with_corpus() -> None:
    b = BM25()
    b.add("one two")
    b.add("three four five")
    assert b.avgdl == pytest.approx(2.5)


def test_bm25_len_reflects_corpus() -> None:
    b = BM25()
    assert len(b) == 0
    b.add("x")
    b.add("y")
    assert len(b) == 2
