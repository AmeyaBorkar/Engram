"""Tests for the Chroma + BM25 hybrid baseline."""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from benchmarks.baselines.chroma_bm25 import ChromaBM25Retriever

from engram.bench import Hit, Retriever
from engram.providers import FakeEmbedder


def test_hybrid_satisfies_protocol() -> None:
    r = ChromaBM25Retriever(embedder=FakeEmbedder(dim=32))
    assert isinstance(r, Retriever)


def test_hybrid_add_returns_id_and_persists() -> None:
    r = ChromaBM25Retriever(embedder=FakeEmbedder(dim=32))
    given = r.add("hello", doc_id="custom-1")
    assert given == "custom-1"


def test_hybrid_empty_query_returns_empty() -> None:
    r = ChromaBM25Retriever(embedder=FakeEmbedder(dim=32))
    assert r.query("anything", k=5) == []


def test_hybrid_returns_hits() -> None:
    r = ChromaBM25Retriever(embedder=FakeEmbedder(dim=32))
    r.add("the cat sat on the mat")
    r.add("entirely unrelated topic")
    hits = r.query("the cat sat on the mat", k=2)
    assert all(isinstance(h, Hit) for h in hits)
    assert len(hits) <= 2
    # Top hit should be the matching document.
    assert hits[0].content == "the cat sat on the mat"


def test_hybrid_rrf_score_is_positive() -> None:
    r = ChromaBM25Retriever(embedder=FakeEmbedder(dim=32))
    r.add("hello world")
    hits = r.query("hello", k=1)
    assert hits[0].score > 0


def test_hybrid_rejects_invalid_k() -> None:
    r = ChromaBM25Retriever(embedder=FakeEmbedder(dim=32))
    with pytest.raises(ValueError, match="k must be"):
        r.query("x", k=0)


def test_hybrid_caps_results_at_k() -> None:
    r = ChromaBM25Retriever(embedder=FakeEmbedder(dim=32))
    for i in range(20):
        r.add(f"document {i}")
    hits = r.query("document", k=5)
    assert len(hits) == 5
