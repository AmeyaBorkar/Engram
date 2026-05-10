"""The `Memory` primitive.

Stage 3 surface: `observe(content)` writes an event with its embedding;
`retrieve(query, k)` returns the top-k events by cosine similarity. The
hierarchy is still flat — every `RetrievalResult` is `level=EVENT` —
but the pipeline is real and the public API is settled.

Later stages layer in:
  - decay (Stage 4): weights evolve with time / reinforcement / corroboration
  - consolidation (Stage 5): events cluster into abstractions
  - hierarchical retrieve (Stage 6): coarse-to-fine reads
  - procedural memory (Stage 7): situation -> action -> outcome
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from engram.providers._protocols import EmbeddingProvider
from engram.schemas import (
    Embedding,
    Event,
    ItemKind,
    Level,
    RetrievalResult,
)
from engram.storage._protocol import Storage


class Memory:
    """Hierarchical memory with consolidation and principled decay.

    Stage 3 ships `observe` and `retrieve`. The class accepts a storage
    backend and an embedding provider; users compose them explicitly.
    """

    def __init__(self, *, storage: Storage, embedder: EmbeddingProvider) -> None:
        self._storage = storage
        self._embedder = embedder

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder

    def observe(self, content: str | Event) -> Event:
        """Record an event and its embedding.

        Accepts a string (wrapped into an `Event` with default fields) or
        a fully-formed `Event`. Returns the persisted `Event`.

        Durability: the event and its embedding land in a single atomic
        transaction; on successful return both are on disk.
        """
        event = content if isinstance(content, Event) else Event(content=content)

        vector = self._embedder.embed([event.content])[0]
        normalized = _normalize(vector)
        embedding = Embedding(
            item_id=event.id,
            item_kind=ItemKind.EVENT,
            model=self._embedder.model,
            dim=self._embedder.dim,
            vector=tuple(normalized),
        )

        with self._storage.transaction():
            self._storage.insert_event(event)
            self._storage.insert_embedding(embedding)

        return event

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return the top-k events most similar to `query` by cosine.

        Stage 3 is flat: every result is `level=EVENT`, `supported_by` is
        the singleton `(event_id,)`, and `confidence` equals the cosine
        score (already in `[0, 1]` for unit-norm vectors when both sides
        are aligned; clamped here just in case).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        query_vec = self._embedder.embed([query])[0]
        normalized = _normalize(query_vec)

        hits = self._storage.search_event_embeddings(normalized, k=k, model=self._embedder.model)

        return [
            RetrievalResult(
                item_id=event_id,
                level=Level.EVENT,
                content=content,
                confidence=_clip01(score),
                score=score,
                supported_by=(event_id,),
            )
            for event_id, content, score in hits
        ]


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
