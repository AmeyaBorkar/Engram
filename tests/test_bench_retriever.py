"""Tests for the `Retriever` protocol and `EngramRetriever` adapter."""

from __future__ import annotations

import uuid

from engram import Memory, SqliteStorage
from engram.bench import EngramRetriever, Hit, Retriever
from engram.providers import FakeEmbedder


def _make_memory(tmp_path: object) -> Memory:
    storage = SqliteStorage(":memory:")
    storage.initialize()
    return Memory(storage=storage, embedder=FakeEmbedder(dim=32))


def test_engram_retriever_satisfies_protocol() -> None:
    r = EngramRetriever(_make_memory(None))
    assert isinstance(r, Retriever)
    assert r.name == "engram"


def test_engram_retriever_add_returns_uuid_string() -> None:
    r = EngramRetriever(_make_memory(None))
    doc_id = r.add("hello")
    # Round-trips through uuid.UUID.
    parsed = uuid.UUID(doc_id)
    assert parsed.version == 7


def test_engram_retriever_add_with_explicit_uuid_id() -> None:
    r = EngramRetriever(_make_memory(None))
    given = str(uuid.uuid4())
    out = r.add("with-id", doc_id=given)
    assert out == given


def test_engram_retriever_add_with_arbitrary_string_id_derives_uuid() -> None:
    r = EngramRetriever(_make_memory(None))
    a = r.add("one", doc_id="not-a-uuid")
    b = r.add("two", doc_id="not-a-uuid-2")
    # Both should be valid UUID strings, and deterministic per input.
    uuid.UUID(a)
    uuid.UUID(b)
    assert a != b


def test_engram_retriever_query_returns_hits() -> None:
    r = EngramRetriever(_make_memory(None))
    r.add("alpha")
    r.add("beta")
    hits = r.query("alpha", k=2)
    assert len(hits) == 2
    assert all(isinstance(h, Hit) for h in hits)
    assert hits[0].content == "alpha"


def test_engram_retriever_rejects_non_memory() -> None:
    import pytest

    r = EngramRetriever("not a memory")
    with pytest.raises(TypeError):
        r.add("x")
    with pytest.raises(TypeError):
        r.query("x", k=1)
