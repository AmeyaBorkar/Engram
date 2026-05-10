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
