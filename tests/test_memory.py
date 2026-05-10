"""Tests for `Memory.observe` and `Memory.retrieve` (Stage 3)."""

from __future__ import annotations

import pytest

from engram import Event, Level, Memory, RetrievalResult
from engram.providers import FakeEmbedder
from engram.storage import SqliteStorage


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=64))


# --- observe --------------------------------------------------------------


def test_observe_string_creates_event(memory: Memory) -> None:
    event = memory.observe("hello world")
    assert isinstance(event, Event)
    assert event.content == "hello world"


def test_observe_persists_event(memory: Memory) -> None:
    event = memory.observe("hello")
    fetched = memory.storage.get_event(event.id)
    assert fetched is not None
    assert fetched.content == "hello"


def test_observe_persists_embedding(memory: Memory) -> None:
    event = memory.observe("hello")
    emb = memory.storage.get_embedding(
        event.id, item_kind=__import__("engram").ItemKind.EVENT, model=memory.embedder.model
    )
    assert emb is not None
    assert emb.dim == memory.embedder.dim


def test_observe_event_object_is_persisted_as_is(memory: Memory) -> None:
    event = Event(content="explicit", source="agent")
    returned = memory.observe(event)
    assert returned.id == event.id
    fetched = memory.storage.get_event(event.id)
    assert fetched is not None
    assert fetched.source == "agent"


def test_observe_atomicity_event_and_embedding_together(memory: Memory) -> None:
    """A failure mid-observe leaves nothing in storage; both rows or none."""
    storage = memory.storage
    n_before = storage.count_events()

    class BoomError(RuntimeError):
        pass

    class FailingEmbedder:
        name = "fail"
        model = "fail"
        dim = 64

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise BoomError("boom")

        async def aembed(self, texts: list[str]) -> list[list[float]]:
            raise BoomError("boom")

        def manifest_hash(self) -> str:
            return "fail"

    bad = Memory(storage=storage, embedder=FailingEmbedder())
    with pytest.raises(BoomError):
        bad.observe("never lands")

    assert storage.count_events() == n_before
    assert storage.count_embeddings() == 0


# --- retrieve -------------------------------------------------------------


def test_retrieve_empty_store_returns_empty(memory: Memory) -> None:
    assert memory.retrieve("anything") == []


def test_retrieve_returns_retrieval_results(memory: Memory) -> None:
    memory.observe("alpha")
    out = memory.retrieve("alpha", k=1)
    assert len(out) == 1
    assert isinstance(out[0], RetrievalResult)
    assert out[0].level == Level.EVENT


def test_retrieve_self_match_is_top(memory: Memory) -> None:
    memory.observe("the cat sat on the mat")
    memory.observe("a completely different topic about quantum physics")
    out = memory.retrieve("the cat sat on the mat", k=1)
    assert out[0].content == "the cat sat on the mat"
    assert out[0].score == pytest.approx(1.0, abs=1e-5)


def test_retrieve_supported_by_is_self_for_events(memory: Memory) -> None:
    event = memory.observe("hello")
    out = memory.retrieve("hello", k=1)
    assert out[0].supported_by == (event.id,)
    assert out[0].item_id == event.id


def test_retrieve_confidence_clamped_to_unit(memory: Memory) -> None:
    memory.observe("anything")
    out = memory.retrieve("anything")
    for r in out:
        assert 0.0 <= r.confidence <= 1.0


def test_retrieve_respects_k(memory: Memory) -> None:
    for i in range(20):
        memory.observe(f"event {i}")
    out = memory.retrieve("event", k=5)
    assert len(out) == 5


def test_retrieve_rejects_invalid_k(memory: Memory) -> None:
    with pytest.raises(ValueError, match="k must be"):
        memory.retrieve("x", k=0)


def test_retrieve_results_ordered_by_score_desc(memory: Memory) -> None:
    for i in range(5):
        memory.observe(f"event {i}")
    out = memory.retrieve("event 0", k=5)
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
