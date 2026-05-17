"""Tests for the public schemas."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from engram.ids import new_id
from engram.schemas import (
    SCHEMA_VERSION,
    Cluster,
    Conflict,
    DecayState,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
    Procedure,
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


# --- schema versioning + extra=forbid ---------------------------------------


def test_schema_version_is_string_one() -> None:
    """`SCHEMA_VERSION` is the canonical persisted-schema marker.

    Bumping this constant is the single signal that storage migrations
    must run; tests pin the value to make accidental bumps loud.
    """
    assert SCHEMA_VERSION == "1"
    assert isinstance(SCHEMA_VERSION, str)


class TestExtraForbid:
    """Persisted models reject unknown fields.

    A v2 field rename would otherwise silently drop the v1 name with no
    audit trail. Forbidding extras lets a renamed-field migration fail at
    parse time instead.
    """

    def test_event_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Event.model_validate({"content": "hi", "unknown_field": 1})

    def test_memory_item_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            MemoryItem.model_validate({"level": "event", "content": "x", "unknown_field": 1})

    def test_procedure_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Procedure.model_validate({"situation": "s", "action": "a", "unknown_field": 1})

    def test_embedding_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Embedding.model_validate(
                {
                    "item_id": str(new_id()),
                    "item_kind": "event",
                    "model": "m",
                    "dim": 1,
                    "vector": [0.0],
                    "unknown_field": 1,
                }
            )

    def test_conflict_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Conflict.model_validate(
                {
                    "source_item_id": str(new_id()),
                    "target_item_id": str(new_id()),
                    "similarity": 0.5,
                    "unknown_field": 1,
                }
            )

    def test_provenance_link_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ProvenanceLink.model_validate(
                {
                    "memory_item_id": str(new_id()),
                    "event_id": str(new_id()),
                    "unknown_field": 1,
                }
            )

    def test_decay_state_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            DecayState.model_validate(
                {
                    "item_id": str(new_id()),
                    "item_kind": "event",
                    "unknown_field": 1,
                }
            )


# --- MemoryItem temporal-default validator -----------------------------------


class TestMemoryItemValidFromBeforeValidator:
    """`_default_valid_from` is a `mode='before'` validator.

    Returns a new mapping rather than mutating `self`, so the model can be
    frozen in a future revision without breaking the default.
    """

    def test_valid_from_defaults_to_created_at(self) -> None:
        item = MemoryItem(level=Level.EVENT, content="x")
        assert item.valid_from == item.created_at

    def test_explicit_created_at_propagates_to_valid_from(self) -> None:
        when = datetime(2026, 1, 2, tzinfo=timezone.utc)
        item = MemoryItem(level=Level.EVENT, content="x", created_at=when)
        assert item.valid_from == when

    def test_caller_supplied_valid_from_wins(self) -> None:
        vf = datetime(2025, 6, 1, tzinfo=timezone.utc)
        item = MemoryItem(level=Level.EVENT, content="x", valid_from=vf)
        assert item.valid_from == vf

    def test_valid_until_before_valid_from_rejected(self) -> None:
        vf = datetime(2026, 1, 1, tzinfo=timezone.utc)
        vu = vf - timedelta(days=1)
        with pytest.raises(ValidationError, match="precedes"):
            MemoryItem(level=Level.EVENT, content="x", valid_from=vf, valid_until=vu)
