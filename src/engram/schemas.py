"""Engram core schemas.

These are the Pydantic v2 models that flow through every layer: storage,
consolidation, retrieval. They are deliberately small — most behavior lives
in the modules that operate on them, not on the models themselves.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engram.ids import new_id


class Level(str, Enum):
    """Hierarchy level a `MemoryItem` occupies."""

    EVENT = "event"
    SUMMARY = "summary"
    ABSTRACTION = "abstraction"


class ItemKind(str, Enum):
    """Kind of item an `Embedding` belongs to."""

    EVENT = "event"
    MEMORY_ITEM = "memory_item"
    PROCEDURE = "procedure"


class Outcome(str, Enum):
    """Outcome of a procedure attempt.

    Drives the reinforcement signal the decay engine applies:

      * `SUCCESS` -> reinforce (the procedure worked, surface it more).
      * `PARTIAL` -> reinforce (worked with caveats; still a positive
        lesson the agent should remember).
      * `FAILURE` -> contradict (the procedure didn't work; weight it
        down so the agent stops reaching for it in similar situations).
      * `UNKNOWN` -> no signal (recorded for completeness but the
        outcome hasn't been observed yet).
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Event(BaseModel):
    """A raw observation. Lands first; is never modified."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=new_id)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Cluster(BaseModel):
    """A grouping of related items, produced by consolidation."""

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=new_id)
    cohesion: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)


class MemoryItem(BaseModel):
    """An item somewhere in the hierarchy.

    `level == "event"` means a literal recall of an event;
    `level == "summary"` is a cluster summary;
    `level == "abstraction"` is a generalization promoted from summaries.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=new_id)
    level: Level
    content: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    cluster_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Procedure(BaseModel):
    """A remembered procedure: "in this situation, this action had that outcome."

    Stage 7 introduces procedures as a first-class memory item alongside
    `Event` and `MemoryItem`. Procedures are how the agent learns from
    doing: a successful pattern strengthens (`reinforce` on retrieval),
    a failed pattern weakens (`contradict` on outcome update).

    Retrieval is over `situation` -- callers describe the current
    context and get back analogous past procedures ranked by similarity.
    `action` and `outcome` ride along as payload. Embedding for
    retrieval targets the situation; storage keeps a single embedding
    per procedure regardless of how `action` evolves.

    The model is mutable (frozen=False) because `outcome` can transition
    -- a procedure may start as `UNKNOWN`, get observed as `SUCCESS`,
    and later be marked `FAILURE` after the user notices it stopped
    working. Decay state lives alongside in the storage row.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=new_id)
    situation: str
    action: str
    outcome: Outcome = Outcome.UNKNOWN
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ProcedureMatch(BaseModel):
    """One result returned from `Memory.retrieve_procedures`.

    Carries the full `Procedure` plus the retrieval score and the
    similarity to the query situation. The score factors in weight and
    outcome (failures don't get suppressed -- the agent benefits from
    "this didn't work" lessons too -- but successes outrank failures at
    equal similarity).
    """

    model_config = ConfigDict(frozen=True)

    procedure: Procedure
    score: float
    similarity: float = Field(ge=-1.0, le=1.0)


class Embedding(BaseModel):
    """A dense vector representation of an event or memory item."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=new_id)
    item_id: UUID
    item_kind: ItemKind
    model: str
    dim: int = Field(gt=0)
    vector: tuple[float, ...]
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _check_dim(self) -> Embedding:
        if len(self.vector) != self.dim:
            raise ValueError(
                f"vector length {len(self.vector)} does not match declared dim {self.dim}"
            )
        return self


class ProvenanceLink(BaseModel):
    """Links a `MemoryItem` to one of its supporting events.

    Provenance is load-bearing. Once a `MemoryItem` is consolidated (level
    other than `event`), it must always retain at least one link to an event;
    that invariant is enforced by storage CHECKs once Stage 5 ships.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=new_id)
    memory_item_id: UUID
    event_id: UUID
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)


class RetrievalResult(BaseModel):
    """One result returned from `Memory.retrieve`.

    Carries the same shape across hierarchy levels: at Stage 3 every result
    is `level=EVENT`, the `confidence` is the cosine similarity to the
    query, and `supported_by` is the singleton `[item_id]` (the event
    itself is its only support). Stage 6 generalizes - abstractions return
    the supporting event ids and a confidence derived from cluster cohesion.
    """

    model_config = ConfigDict(frozen=True)

    item_id: UUID
    level: Level
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    score: float
    supported_by: tuple[UUID, ...]


class DecayState(BaseModel):
    """Per-item mutable decay state.

    Lives alongside the immutable observation (`Event` or `MemoryItem`) and
    holds everything the decay engine needs to score, age, and prune the
    item: the running weight, the running counts of reinforcement /
    corroboration / contradiction signals, the timestamp of the last decay
    application, and an optional `cold_at` marking when the item dropped
    below the prune threshold.

    The model is frozen because storage owns the row and re-fetches a fresh
    `DecayState` after every mutation - a stale in-memory copy would silently
    diverge from the database.
    """

    model_config = ConfigDict(frozen=True)

    item_id: UUID
    item_kind: ItemKind
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    reinforcement_count: int = Field(default=0, ge=0)
    corroboration_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    last_decayed_at: datetime
    cold_at: datetime | None = None
