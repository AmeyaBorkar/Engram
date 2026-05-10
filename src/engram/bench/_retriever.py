"""Common retriever protocol for benchmark baselines.

Every system Engram is benchmarked against (Chroma, Chroma+BM25, mem0,
Letta, …) implements this small surface so the benchmark harness can
drive them all the same way.

The protocol is intentionally narrower than `Memory` itself: just `add`
and `query`. Baselines that don't have a notion of provenance, decay,
or hierarchy don't need to fake them — that's where Engram earns its
keep, and the comparison should reflect what each system actually does.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    """One result returned from a retriever's `query`."""

    id: str
    content: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    """Common surface for benchmark baselines and Engram itself."""

    name: str

    def add(self, content: str, doc_id: str | None = None) -> str:
        """Index `content`. Returns the document id (generated if not given)."""

    def query(self, query: str, k: int) -> Sequence[Hit]:
        """Return up to `k` hits, ordered by score desc."""


class EngramRetriever:
    """Adapt `engram.Memory` to the `Retriever` protocol."""

    name: str = "engram"

    def __init__(self, memory: object) -> None:
        # `memory` is typed as object to avoid an import cycle with
        # `engram.memory`; consumers pass an `engram.Memory` instance.
        self._memory = memory

    def add(self, content: str, doc_id: str | None = None) -> str:
        from engram import Event, Memory  # local import avoids cycle

        memory = self._memory
        if not isinstance(memory, Memory):
            raise TypeError(f"expected engram.Memory, got {type(memory).__name__}")
        event_in = (
            Event(id=_uuid_from_doc_id(doc_id), content=content) if doc_id is not None else content
        )
        event = memory.observe(event_in)
        return str(event.id)

    def query(self, query: str, k: int) -> list[Hit]:
        from engram import Memory

        memory = self._memory
        if not isinstance(memory, Memory):
            raise TypeError(f"expected engram.Memory, got {type(memory).__name__}")
        results = memory.retrieve(query, k=k)
        return [Hit(id=str(r.item_id), content=r.content, score=r.score) for r in results]


def _uuid_from_doc_id(doc_id: str) -> uuid.UUID:
    """Coerce a doc_id string into a UUID — accept either a UUID string or
    derive a deterministic UUID5 from arbitrary text."""
    try:
        return uuid.UUID(doc_id)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_OID, doc_id)
