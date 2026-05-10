"""Stage 6 DoD: hierarchical retrieve beats flat by >= 10 % recall@k.

The test plants a synthetic dataset where the surface-form events
do not embed directly onto the topic centroid, but the consolidation
layer's summary does. This is the README's pitch: an LLM-generated
generalization captures *what the events mean together*, even when
the events themselves wouldn't surface against a conceptual query.

Setup per topic (5 topics, 4 events each):

  * Topic centroid `c_i` is a one-hot vector in dim 0..4.
  * Per topic, plant 4 events whose vectors are one-hot in dims
    5..8 -- orthogonal to the centroid, so flat retrieve over the
    events alone gets cosine 0 against the centroid query.
  * Plant a `Level.SUMMARY` at exactly the centroid, linked to the
    4 events via provenance.
  * 5 queries, one per centroid.

Hierarchical (`prefer="auto"`) sees the summary at cosine 1.0 and
surfaces it -- recall@1 = 1.0 per query. Flat (`prefer="specific"`)
sees events at cosine 0 to the query and surfaces noise -- recall@1
near 0. Lift >= 100 percentage points, far above the 10-point bar.

The test pins the ratio rather than absolute numbers because
FakeEmbedder noise in the rest of the dim space contributes a small
random hit rate; we don't want to flake on its variance.
"""

from __future__ import annotations

import pytest

from engram import (
    Embedding,
    Event,
    ItemKind,
    Level,
    Memory,
    MemoryItem,
    SqliteStorage,
)
from engram.providers._fake import FakeEmbedder
from tests.test_retrieve_hierarchical import PlantedEmbedder

DIM = 16
N_TOPICS = 5
EVENTS_PER_TOPIC = 4


def _one_hot(idx: int, dim: int = DIM) -> tuple[float, ...]:
    v = [0.0] * dim
    v[idx] = 1.0
    return tuple(v)


def _build_corpus(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
) -> list[tuple[set[bytes], tuple[float, ...]]]:
    """Plant the synthetic split. Returns per-topic (relevant_ids, query_vec)."""
    relevant: list[tuple[set[bytes], tuple[float, ...]]] = []
    for t in range(N_TOPICS):
        centroid_vec = _one_hot(t)
        relevant_ids: set[bytes] = set()

        # Events orthogonal to centroid (dims 5..8 cycled across topics).
        events: list[Event] = []
        for j in range(EVENTS_PER_TOPIC):
            event_dim = (N_TOPICS + j + (t * EVENTS_PER_TOPIC)) % DIM
            if event_dim < N_TOPICS:
                # Avoid colliding with another topic's centroid dim.
                event_dim = (event_dim + N_TOPICS) % DIM
            ev = Event(content=f"topic-{t}-event-{j}")
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=embedder.dim,
                    vector=_one_hot(event_dim),
                )
            )
            events.append(ev)
            relevant_ids.add(ev.id.bytes)

        # Summary at the centroid, linked to events via provenance.
        summary = MemoryItem(level=Level.SUMMARY, content=f"topic-{t}-summary")
        storage.insert_memory_item_with_provenance(
            summary,
            [e.id for e in events],
            embedding=Embedding(
                item_id=summary.id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=embedder.model,
                dim=embedder.dim,
                vector=centroid_vec,
            ),
        )
        relevant_ids.add(summary.id.bytes)
        relevant.append((relevant_ids, centroid_vec))
    return relevant


def _recall_at_k(
    memory: Memory,
    *,
    relevant_per_query: list[tuple[set[bytes], tuple[float, ...]]],
    embedder: PlantedEmbedder,
    k: int,
    prefer: str,
) -> float:
    hits = 0
    for q_idx, (relevant_ids, query_vec) in enumerate(relevant_per_query):
        query_text = f"q_{q_idx}_{prefer}"
        embedder.plant(query_text, query_vec)
        results = memory.retrieve(
            query_text,
            k=k,
            prefer=prefer,
            reinforce=False,  # type: ignore[arg-type]
        )
        if any(r.item_id.bytes in relevant_ids for r in results):
            hits += 1
    return hits / len(relevant_per_query)


@pytest.mark.parametrize("k", [1, 3])
def test_hierarchical_beats_flat_by_at_least_ten_points(storage: SqliteStorage, k: int) -> None:
    embedder = PlantedEmbedder(dim=DIM)
    memory = Memory(storage=storage, embedder=embedder)
    relevant_per_query = _build_corpus(storage, embedder)

    flat_recall = _recall_at_k(
        memory,
        relevant_per_query=relevant_per_query,
        embedder=embedder,
        k=k,
        prefer="specific",
    )
    hierarchical_recall = _recall_at_k(
        memory,
        relevant_per_query=relevant_per_query,
        embedder=embedder,
        k=k,
        prefer="auto",
    )

    lift = hierarchical_recall - flat_recall
    assert lift >= 0.10, (
        f"hierarchical recall@{k} = {hierarchical_recall:.2f}, "
        f"flat recall@{k} = {flat_recall:.2f}, "
        f"lift = {lift:.2f} (target >= 0.10)"
    )
    # Absolute hierarchical recall should be at least 80 % -- the
    # summary is at exact cosine 1.0 to the query.
    assert hierarchical_recall >= 0.80, (
        f"hierarchical recall@{k} = {hierarchical_recall:.2f} "
        f"(expected >= 0.80 with planted centroid summaries)"
    )


def test_general_prefer_returns_only_summaries(storage: SqliteStorage) -> None:
    embedder = PlantedEmbedder(dim=DIM)
    memory = Memory(storage=storage, embedder=embedder)
    relevant_per_query = _build_corpus(storage, embedder)

    embedder.plant("q_general", relevant_per_query[0][1])
    results = memory.retrieve("q_general", k=10, prefer="general", reinforce=False)
    assert results, "expected at least one summary result"
    assert all(r.level is Level.SUMMARY for r in results), (
        f"prefer=general surfaced non-summary levels: {[r.level for r in results]}"
    )
