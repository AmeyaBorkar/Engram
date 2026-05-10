"""Tests for the public schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engram.ids import new_id
from engram.schemas import (
    Cluster,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
    ProvenanceLink,
)


def test_event_defaults() -> None:
    e = Event(content="hello")
    assert e.content == "hello"
    assert e.metadata == {}
    assert e.source is None
    assert e.id.version == 7
    assert e.created_at is not None


def test_event_is_frozen() -> None:
    e = Event(content="hello")
    with pytest.raises(ValidationError):
        e.content = "world"  # type: ignore[misc]


def test_memory_item_defaults() -> None:
    item = MemoryItem(level=Level.EVENT, content="x")
    assert item.weight == 1.0
    assert item.cluster_id is None
    assert item.metadata == {}


def test_memory_item_weight_bounds() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(level=Level.EVENT, content="x", weight=-0.1)
    with pytest.raises(ValidationError):
        MemoryItem(level=Level.EVENT, content="x", weight=1.1)


def test_memory_item_level_must_be_enum() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(level="not-a-level", content="x")  # type: ignore[arg-type]


def test_embedding_dim_must_match_vector_length() -> None:
    item_id = new_id()
    with pytest.raises(ValidationError, match="does not match"):
        Embedding(
            item_id=item_id,
            item_kind=ItemKind.EVENT,
            model="m",
            dim=4,
            vector=(0.1, 0.2),
        )


def test_embedding_accepts_correct_dim() -> None:
    e = Embedding(
        item_id=new_id(),
        item_kind=ItemKind.EVENT,
        model="m",
        dim=3,
        vector=(0.1, 0.2, 0.3),
    )
    assert e.dim == 3


def test_provenance_link_weight_bounds() -> None:
    with pytest.raises(ValidationError):
        ProvenanceLink(memory_item_id=new_id(), event_id=new_id(), weight=2.0)


def test_cluster_cohesion_bounds() -> None:
    with pytest.raises(ValidationError):
        Cluster(cohesion=-0.1)
    with pytest.raises(ValidationError):
        Cluster(cohesion=1.5)


def test_retrieval_result_basic() -> None:
    from engram.schemas import RetrievalResult

    item_id = new_id()
    r = RetrievalResult(
        item_id=item_id,
        level=Level.EVENT,
        content="hello",
        confidence=0.9,
        score=0.85,
        supported_by=(item_id,),
    )
    assert r.item_id == item_id
    assert r.level == Level.EVENT
    assert r.confidence == 0.9


def test_retrieval_result_confidence_bounds() -> None:
    from engram.schemas import RetrievalResult

    with pytest.raises(ValidationError):
        RetrievalResult(
            item_id=new_id(),
            level=Level.EVENT,
            content="x",
            confidence=1.1,
            score=0.5,
            supported_by=(new_id(),),
        )


def test_retrieval_result_is_frozen() -> None:
    from engram.schemas import RetrievalResult

    r = RetrievalResult(
        item_id=new_id(),
        level=Level.EVENT,
        content="x",
        confidence=0.5,
        score=0.5,
        supported_by=(new_id(),),
    )
    with pytest.raises(ValidationError):
        r.content = "y"  # type: ignore[misc]
