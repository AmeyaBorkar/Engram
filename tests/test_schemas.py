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


def test_event_content_accepts_128k_payload() -> None:
    """A 128 KiB content blob must ingest cleanly.

    Regression: LongMemEval-S question 852ce960 contains a pasted
    MediaWiki page exceeding the previous 64 KiB cap.  The benchmark
    counted that question as score=0 because of the cap, masking the
    underlying model performance.  Bumped to 1 MiB; verify the new
    headroom by ingesting a realistic-large blob below the new cap.
    """
    payload = "x" * (128 * 1024)
    e = Event(content=payload)
    assert len(e.content) == 128 * 1024


def test_event_content_accepts_1mib_payload() -> None:
    """Right at the 1 MiB cap is still accepted; 1 byte over is rejected."""
    at_cap = "y" * (1024 * 1024)
    Event(content=at_cap)  # exactly at cap
    over_cap = "z" * (1024 * 1024 + 1)
    with pytest.raises(ValidationError, match="String should have at most"):
        Event(content=over_cap)


def test_memory_item_content_accepts_128k_payload() -> None:
    """MemoryItem.content cap tracks Event.content; same 1 MiB headroom."""
    item = MemoryItem(level=Level.EVENT, content="z" * (128 * 1024))
    assert len(item.content) == 128 * 1024


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
    # Range is [-1, 1] to match mean pairwise cosine math; values
    # outside that band are still rejected.
    with pytest.raises(ValidationError):
        Cluster(cohesion=-1.1)
    with pytest.raises(ValidationError):
        Cluster(cohesion=1.5)
    Cluster(cohesion=-0.5)  # anti-correlated cluster is valid in storage
    Cluster(cohesion=1.0)
    Cluster(cohesion=-1.0)


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


# ---------------------------------------------------------------------------
# H-71: SCHEMA_VERSION and extra="forbid" on every persisted model
# ---------------------------------------------------------------------------


def test_schema_version_exported() -> None:
    assert isinstance(SCHEMA_VERSION, str)
    assert SCHEMA_VERSION  # non-empty


@pytest.mark.parametrize(
    ("kwargs", "model_cls"),
    [
        ({"content": "x", "unknown_field": 1}, Event),
        ({"level": Level.EVENT, "content": "x", "rogue": True}, MemoryItem),
        ({"situation": "x", "action": "y", "rogue": "v"}, Procedure),
        (
            {
                "item_id": new_id(),
                "item_kind": ItemKind.EVENT,
                "model": "m",
                "dim": 1,
                "vector": (0.1,),
                "rogue": 0,
            },
            Embedding,
        ),
        (
            {
                "memory_item_id": new_id(),
                "event_id": new_id(),
                "rogue": True,
            },
            ProvenanceLink,
        ),
        (
            {
                "item_id": new_id(),
                "item_kind": ItemKind.EVENT,
                "rogue": True,
            },
            DecayState,
        ),
    ],
)
def test_unknown_fields_rejected(kwargs: dict, model_cls: type) -> None:
    """`extra="forbid"` surfaces typos at the boundary rather than dropping data."""
    with pytest.raises(ValidationError):
        model_cls(**kwargs)


def test_conflict_unknown_fields_rejected() -> None:
    src, tgt = new_id(), new_id()
    with pytest.raises(ValidationError):
        Conflict(
            source_item_id=src,
            target_item_id=tgt,
            similarity=0.5,
            rogue=1,
        )


# ---------------------------------------------------------------------------
# H-71: frozen=True on every persisted model
# ---------------------------------------------------------------------------


def test_memory_item_is_frozen() -> None:
    item = MemoryItem(level=Level.EVENT, content="x")
    with pytest.raises(ValidationError):
        item.content = "y"  # type: ignore[misc]


def test_procedure_is_frozen() -> None:
    from engram.schemas import Outcome

    p = Procedure(situation="x", action="y")
    with pytest.raises(ValidationError):
        p.outcome = Outcome.SUCCESS  # type: ignore[misc]


def test_conflict_is_frozen() -> None:
    src, tgt = new_id(), new_id()
    c = Conflict(source_item_id=src, target_item_id=tgt, similarity=0.5)
    with pytest.raises(ValidationError):
        c.similarity = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# M-135: temporal-invariants validator does NOT mutate `self.valid_from`
# ---------------------------------------------------------------------------


def test_memory_item_valid_from_defaults_to_created_at_without_mutation() -> None:
    """A frozen-instance MemoryItem still receives `valid_from = created_at`
    when the caller omits both — but the default is applied in the
    `mode="before"` validator, so no post-validate mutation is needed.
    """
    item = MemoryItem(level=Level.EVENT, content="x")
    assert item.valid_from == item.created_at
    # `created_at` is preserved when the caller supplies it; the `mode="before"`
    # validator doesn't synthesize a fresh default in that path.
    when = datetime(2024, 6, 1, tzinfo=timezone.utc)
    item2 = MemoryItem(level=Level.EVENT, content="x", created_at=when)
    assert item2.valid_from == when


def test_memory_item_revalidate_dict_round_trip_preserves_defaults() -> None:
    """`model_validate(model.model_dump())` returns an equivalent instance.

    Re-validating round-trips through the `mode="before"` validator
    again; the dumped dict already has `valid_from` populated, so the
    default-injection branch is a no-op on a dump-load round trip.
    """
    item = MemoryItem(level=Level.EVENT, content="x")
    rt = MemoryItem.model_validate(item.model_dump())
    assert rt == item
    # And direct mutation still raises (frozen contract holds on the
    # re-loaded instance, not just the original).
    with pytest.raises(ValidationError):
        rt.content = "y"  # type: ignore[misc]


def test_memory_item_explicit_valid_from_preserved() -> None:
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    item = MemoryItem(level=Level.EVENT, content="x", valid_from=when)
    assert item.valid_from == when


def test_memory_item_valid_until_before_valid_from_rejected() -> None:
    earlier = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = earlier + timedelta(days=1)
    with pytest.raises(ValidationError):
        MemoryItem(level=Level.EVENT, content="x", valid_from=later, valid_until=earlier)


def test_memory_item_invalidated_by_requires_invalidated_at() -> None:
    with pytest.raises(ValidationError, match="invalidated_by"):
        MemoryItem(level=Level.EVENT, content="x", invalidated_by=new_id())
