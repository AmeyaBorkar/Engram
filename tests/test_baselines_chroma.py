"""Tests for the Chroma baseline.

Skipped when `chromadb` isn't installed (i.e. dev install without the
`[bench]` extra). The CI smoke-benchmark job installs the extra, so
these run on every PR.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from benchmarks.baselines.chroma import ChromaRetriever

from engram.bench import Hit, Retriever
from engram.providers import FakeEmbedder


def test_chroma_retriever_satisfies_protocol() -> None:
    r = ChromaRetriever(embedder=FakeEmbedder(dim=32))
    assert isinstance(r, Retriever)


def test_chroma_retriever_add_returns_id() -> None:
    r = ChromaRetriever(embedder=FakeEmbedder(dim=32))
    out = r.add("hello")
    assert out  # non-empty string


def test_chroma_retriever_add_with_explicit_id() -> None:
    r = ChromaRetriever(embedder=FakeEmbedder(dim=32))
    assert r.add("hello", doc_id="custom-id-1") == "custom-id-1"


def test_chroma_retriever_query_returns_hits() -> None:
    r = ChromaRetriever(embedder=FakeEmbedder(dim=32))
    r.add("the cat sat on the mat")
    r.add("a totally different topic")
    hits = r.query("the cat sat on the mat", k=2)
    assert len(hits) == 2
    assert all(isinstance(h, Hit) for h in hits)
    assert hits[0].content == "the cat sat on the mat"


def test_chroma_retriever_score_is_similarity() -> None:
    r = ChromaRetriever(embedder=FakeEmbedder(dim=32))
    r.add("alpha")
    hits = r.query("alpha", k=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)


def test_chroma_retriever_rejects_invalid_k() -> None:
    r = ChromaRetriever(embedder=FakeEmbedder(dim=32))
    with pytest.raises(ValueError, match="k must be"):
        r.query("x", k=0)
