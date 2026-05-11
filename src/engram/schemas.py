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
    """Hierarchy level a `MemoryItem` occupies.

    Stage 6 shipped `event` / `summary` / `abstraction`. Stage 9 layers
    in three more for retrieval routing:

      * `preference` (E.6) -- sentiment-laden / explicit-preference
        statements that should outrank generic summaries when the
        query is about user preferences.
      * `topic` (E.8) -- mid-grain abstraction between summary and
        abstraction, organized by topic cluster.
      * `global` (E.7) -- the aggregate user-state abstraction.
        Exactly one per tenant (per the storage convention). Surfaced
        alongside any user-centric query.
    """

    EVENT = "event"
    SUMMARY = "summary"
    ABSTRACTION = "abstraction"
    PREFERENCE = "preference"
    TOPIC = "topic"
    GLOBAL = "global"


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


class Verdict(str, Enum):
    """LLM judge's classification of how two memory items relate.

    Used by Stage 5's contradiction detector (`consolidation/_contradiction`)
    and Stage 8's first-class `Conflict` storage entity. AGREE and
    UNRELATED verdicts are not persisted as Conflicts; only CONTRADICT
    rows show up in the conflicts table.
    """

    AGREE = "agree"
    CONTRADICT = "contradict"
    UNRELATED = "unrelated"


class Resolution(str, Enum):
    """How `Memory.reconcile` picked the winner of a `Conflict`.

    The reconciler applies the policy at resolution time; the policy
    name is persisted on the conflict row so audits can replay the
    decision later.

      * `PREFER_RECENT` - the more recently-created item wins.
      * `PREFER_TRUSTED` - the item with the higher `source_trust` wins.
      * `PREFER_FREQUENT` - the item with the higher corroboration count
        (from decay state) wins.
      * `KEEP_BOTH` - no winner; both items stay valid. The conflict is
        still marked resolved so it stops surfacing on every audit pass.
      * `MANUAL` - the caller specified the winner explicitly.
      * `MERGE` - the reconciler calls the chat provider to synthesize a
        new memory item that captures both sides. Both originals are
        invalidated pointing to the merged item; the conflict carries
        `resolved_winner_id=None` (the new item is reachable via either
        original's `invalidated_by`).
    """

    PREFER_RECENT = "prefer_recent"
    PREFER_TRUSTED = "prefer_trusted"
    PREFER_FREQUENT = "prefer_frequent"
    KEEP_BOTH = "keep_both"
    MANUAL = "manual"
    MERGE = "merge"


class ConflictStatus(str, Enum):
    """Lifecycle stage of a `Conflict` row.

      * `OPEN` - detected by consolidation; awaiting reconciliation.
      * `RESOLVED` - the reconciler picked a winner (or chose
        `KEEP_BOTH`); `resolution`, `resolved_winner_id`, and
        `resolved_at` are filled in.
    """

    OPEN = "open"
    RESOLVED = "resolved"


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
    tenant_id: str | None = None


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

    Stage 8 layers temporal validity and explicit invalidation on top:

      * `valid_from` / `valid_until`: the window during which the item
        is "true." `valid_from` defaults to `created_at`; `valid_until`
        is `None` for facts that are still current. The temporal-aware
        retrieve path uses these to answer "as of when?" queries.
      * `invalidated_at` / `invalidated_by`: set when `Memory.reconcile`
        chooses the other side of a conflict; `invalidated_by` is the
        winner's id. Invalidated items are excluded from default
        retrieves but surface again with `as_of=` before invalidation.
      * `source_trust`: in `[0, 1]`; denormalized from the `Source` that
        introduced the item, used by `Resolution.PREFER_TRUSTED`.
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
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    invalidated_at: datetime | None = None
    invalidated_by: UUID | None = None
    source_trust: float | None = Field(default=None, ge=0.0, le=1.0)
    tenant_id: str | None = None

    @model_validator(mode="after")
    def _check_temporal_invariants(self) -> MemoryItem:
        # valid_from defaults to created_at when callers omit it.
        # frozen=False + default validate_assignment=False means direct
        # assignment is safe here (no re-validation loop).
        if self.valid_from is None:
            self.valid_from = self.created_at
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError(
                f"valid_until {self.valid_until.isoformat()} precedes "
                f"valid_from {self.valid_from.isoformat()}"
            )
        if self.invalidated_by is not None and self.invalidated_at is None:
            raise ValueError("invalidated_by set without invalidated_at")
        return self


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
    tenant_id: str | None = None


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


class Source(BaseModel):
    """A named provenance source with a trust weight.

    Stage 8 uses `source.trust` to break ties during conflict resolution
    when the caller picks `Resolution.PREFER_TRUSTED`. The trust value
    is denormalized onto `MemoryItem.source_trust` at write time so the
    reconciler and the temporal retrieve path do not need to consult a
    Source registry to do their work - the float on the row is the
    authoritative copy.

    The `name` is free-form (it could be a username, a tool name, a
    URL) and matches the `Event.source` string when an event is the
    proximate origin of a memory item. Sources are a *policy* concept,
    not a storage entity in this stage; a future stage may introduce a
    `sources` table once the use cases warrant it.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    trust: float = Field(ge=0.0, le=1.0)


class Conflict(BaseModel):
    """A detected contradiction between two `MemoryItem` rows.

    Stage 5's contradiction detector creates one of these (status=OPEN)
    when the LLM judge classifies a candidate pair as
    `Verdict.CONTRADICT`. Stage 8's reconciler resolves them: it picks
    a winner per the chosen `Resolution` policy, invalidates the loser
    (sets `invalidated_at` + `invalidated_by` on the losing memory
    item), and flips the conflict to status=RESOLVED.

    Mutability: `frozen=False` because status, resolution,
    resolved_winner_id, and resolved_at flip during reconciliation. The
    storage row is the source of truth; in-memory copies are snapshots.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=new_id)
    source_item_id: UUID
    target_item_id: UUID
    similarity: float = Field(ge=-1.0, le=1.0)
    verdict: Verdict = Verdict.CONTRADICT
    status: ConflictStatus = ConflictStatus.OPEN
    resolution: Resolution | None = None
    resolved_winner_id: UUID | None = None
    resolved_at: datetime | None = None
    detected_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _check_status_invariants(self) -> Conflict:
        if self.source_item_id == self.target_item_id:
            raise ValueError("source_item_id and target_item_id must differ")
        if self.status is ConflictStatus.RESOLVED:
            if self.resolution is None:
                raise ValueError("resolved conflict requires a resolution")
            if self.resolved_at is None:
                raise ValueError("resolved conflict requires resolved_at")
            # KEEP_BOTH and MERGE both legitimately have no winner: KEEP_BOTH
            # because both stay valid, MERGE because the merged-into id is
            # tracked via the `invalidated_by` field on both originals.
            if (
                self.resolution
                not in (Resolution.KEEP_BOTH, Resolution.MERGE)
                and self.resolved_winner_id is None
            ):
                raise ValueError(
                    f"resolution={self.resolution.value} requires resolved_winner_id"
                )
            if self.resolved_winner_id is not None and self.resolved_winner_id not in (
                self.source_item_id,
                self.target_item_id,
            ):
                raise ValueError(
                    "resolved_winner_id must equal source_item_id or target_item_id"
                )
        else:  # OPEN
            if self.resolution is not None:
                raise ValueError("open conflict must have no resolution")
            if self.resolved_at is not None:
                raise ValueError("open conflict must have no resolved_at")
            if self.resolved_winner_id is not None:
                raise ValueError("open conflict must have no resolved_winner_id")
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
