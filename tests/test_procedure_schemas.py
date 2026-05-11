"""Stage 7 procedure schema tests.

Covers the surface of `Procedure`, `Outcome`, and `ProcedureMatch`:
defaults, validation, bounds, mutability rules. Storage and retrieval
land in separate tests; this file just locks the schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engram.schemas import ItemKind, Outcome, Procedure, ProcedureMatch


def test_outcome_enum_values() -> None:
    assert {o.value for o in Outcome} == {"success", "partial", "failure", "unknown"}


def test_item_kind_includes_procedure() -> None:
    assert ItemKind.PROCEDURE.value == "procedure"
    assert {k.value for k in ItemKind} == {"event", "memory_item", "procedure"}


def test_procedure_defaults() -> None:
    p = Procedure(situation="user asks for code review", action="list issues by severity")
    assert p.situation == "user asks for code review"
    assert p.action == "list issues by severity"
    assert p.outcome is Outcome.UNKNOWN
    assert p.weight == 1.0
    assert p.metadata == {}
    assert p.id.version == 7  # UUIDv7
    assert p.created_at is not None
    assert p.updated_at is not None


def test_procedure_explicit_outcome() -> None:
    p = Procedure(situation="x", action="y", outcome=Outcome.SUCCESS)
    assert p.outcome is Outcome.SUCCESS


def test_procedure_weight_bounds() -> None:
    Procedure(situation="x", action="y", weight=0.0)
    Procedure(situation="x", action="y", weight=1.0)
    with pytest.raises(ValidationError):
        Procedure(situation="x", action="y", weight=-0.01)
    with pytest.raises(ValidationError):
        Procedure(situation="x", action="y", weight=1.01)


def test_procedure_is_mutable_for_outcome_transitions() -> None:
    # frozen=False; outcome must be assignable after construction so the
    # outcome-feedback loop can flip UNKNOWN -> SUCCESS / FAILURE.
    p = Procedure(situation="x", action="y")
    p.outcome = Outcome.SUCCESS
    assert p.outcome is Outcome.SUCCESS


def test_procedure_metadata_preserves_arbitrary_keys() -> None:
    p = Procedure(
        situation="x",
        action="y",
        metadata={"trace_id": "abc-123", "tags": ["debug"]},
    )
    assert p.metadata["trace_id"] == "abc-123"
    assert p.metadata["tags"] == ["debug"]


def test_procedure_match_carries_procedure_and_scores() -> None:
    p = Procedure(situation="x", action="y", outcome=Outcome.SUCCESS)
    m = ProcedureMatch(procedure=p, score=0.85, similarity=0.92)
    assert m.procedure.id == p.id
    assert m.score == 0.85
    assert m.similarity == 0.92


def test_procedure_match_similarity_bounds() -> None:
    p = Procedure(situation="x", action="y")
    ProcedureMatch(procedure=p, score=0.5, similarity=-1.0)
    ProcedureMatch(procedure=p, score=0.5, similarity=1.0)
    with pytest.raises(ValidationError):
        ProcedureMatch(procedure=p, score=0.5, similarity=-1.5)
    with pytest.raises(ValidationError):
        ProcedureMatch(procedure=p, score=0.5, similarity=1.5)


def test_procedure_match_is_frozen() -> None:
    p = Procedure(situation="x", action="y")
    m = ProcedureMatch(procedure=p, score=0.5, similarity=0.5)
    with pytest.raises(ValidationError):
        m.score = 0.9  # type: ignore[misc]
