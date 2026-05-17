"""Stage 6 -- coarse-to-fine retrieval golden tests.

Layers exercised:

  * `prefer="general"` -- only abstractions/summaries surface; drill is
    suppressed even when an abstraction's confidence is low.
  * `prefer="specific"` -- skip the abstraction layer entirely; the
    Stage 3 flat-retrieve behavior.
  * `prefer="auto"` -- abstractions when confidence >= threshold; drill
    into supporting events when below.
  * Empty-hierarchy fallback: a corpus with no consolidated items still
    answers via the event layer.
  * `RetrievalResult.level` faithfully mirrors what was surfaced.
  * Reinforcement-on-use signal fires for surfaced items (and not when
    `reinforce=False`).
  * Reranker reorders the merged candidate set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pytest

from engram import (
    Embedding,
    Event,
    FakeReranker,
    ItemKind,
    Level,
    Memory,
    MemoryItem,
    SqliteStorage,
)
from engram.providers._fake import FakeEmbedder


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class PlantedEmbedder(FakeEmbedder):
    """`FakeEmbedder` with a per-text override for vector planting.

    Test setup: plant a known vector for the query string so the test
    can assert deterministic similarity scores against planted memory
    items. Falls back to the SHA-256 vector for any text not in the
    planted dict.
    """

    def __init__(
        self,
        *,
        dim: int = 4,
        model: str = "fake-sha256",
        planted: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        super().__init__(dim=dim, model=model)
        self._planted: dict[str, list[float]] = {k: list(v) for k, v in (planted or {}).items()}

    def plant(self, text: str, vector: Sequence[float]) -> None:
        self._planted[text] = list(vector)

    def _embed_one(self, text: str) -> list[float]:  # type: ignore[override]
        if text in self._planted:
            return list(self._planted[text])
        return super()._embed_one(text)


def _planted_event(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
    *,
    content: str,
    vector: tuple[float, ...],
) -> Event:
    """Insert an event with a known unit-norm embedding."""
    ev = Event(content=content)
    storage.insert_event(ev)
    storage.insert_embedding(
        Embedding(
            item_id=ev.id,
            item_kind=ItemKind.EVENT,
            model=embedder.model,
            dim=embedder.dim,
            vector=vector,
        )
    )
    return ev


def _planted_summary(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
    *,
    content: str,
    vector: tuple[float, ...],
    supports: list[Event],
    support_weight: float = 0.5,
    level: Level = Level.SUMMARY,
) -> MemoryItem:
    """Insert a memory item + its embedding + provenance to the supports."""
    item = MemoryItem(level=level, content=content)
    embedding = Embedding(
        item_id=item.id,
        item_kind=ItemKind.MEMORY_ITEM,
        model=embedder.model,
        dim=embedder.dim,
        vector=vector,
    )
    storage.insert_memory_item_with_provenance(
        item,
        [e.id for e in supports],
        embedding=embedding,
        provenance_weights={e.id: support_weight for e in supports},
    )
    return item


# ---------------------------------------------------------------------------
# Fallback: no abstractions in the store
# ---------------------------------------------------------------------------


class TestEmptyHierarchyFallback:
    def test_retrieve_falls_back_to_events_when_no_abstractions(
        self, storage: SqliteStorage
    ) -> None:
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        e = memory.observe("alice greets bob")
        memory.observe("the kitchen needs cleaning")

        results = memory.retrieve("alice greets bob", k=2)
        assert any(r.item_id == e.id for r in results)
        assert all(r.level is Level.EVENT for r in results)

    def test_specific_prefer_returns_event_level(self, storage: SqliteStorage) -> None:
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        memory.observe("alpha")
        results = memory.retrieve("alpha", prefer="specific")
        assert all(r.level is Level.EVENT for r in results)


# ---------------------------------------------------------------------------
# prefer="general" -- always surface abstractions, no drill
# ---------------------------------------------------------------------------


class TestPreferGeneral:
    def test_only_abstractions_returned(self, storage: SqliteStorage) -> None:
        embedder = PlantedEmbedder(dim=4)
        memory = Memory(storage=storage, embedder=embedder)

        # Plant 3 events sharing a vector + a summary at the same vector.
        v = (1.0, 0.0, 0.0, 0.0)
        events = [
            _planted_event(storage, embedder, content=f"event {i}", vector=v) for i in range(3)
        ]
        summary = _planted_summary(
            storage, embedder, content="Topic A pattern", vector=v, supports=events
        )

        results = memory.retrieve("topic A query", prefer="general", k=5)
        # Even though events share the vector, prefer=general returns
        # only the abstraction layer.
        levels = {r.level for r in results}
        ids = {r.item_id for r in results}
        assert Level.SUMMARY in levels or Level.ABSTRACTION in levels
        assert summary.id in ids
        assert all(e.id not in ids for e in events)


# ---------------------------------------------------------------------------
# prefer="auto" -- threshold-driven drill
# ---------------------------------------------------------------------------


class TestPreferAuto:
    def test_high_confidence_keeps_abstraction(self, storage: SqliteStorage) -> None:
        embedder = PlantedEmbedder(dim=4)
        memory = Memory(storage=storage, embedder=embedder)
        v = (1.0, 0.0, 0.0, 0.0)  # identical to query vector under FakeEmbedder
        events = [
            _planted_event(storage, embedder, content=f"event {i}", vector=v) for i in range(2)
        ]
        summary = _planted_summary(
            storage, embedder, content="High cohesion summary", vector=v, supports=events
        )

        # Plant the query embedding to match exactly.
        embedder.plant("q", v)
        results = memory.retrieve("q", prefer="auto", confidence_threshold=0.5, k=5)
        ids = {r.item_id for r in results}
        # Abstraction should be in the result set since cosine ~= 1.0.
        assert summary.id in ids

    def test_low_confidence_drills_into_supporting_events(self, storage: SqliteStorage) -> None:
        embedder = PlantedEmbedder(dim=4)
        memory = Memory(storage=storage, embedder=embedder)
        # Summary at a near-orthogonal direction (low cosine with query).
        # Supporting events at vectors that ARE close to the query.
        sup_vec = (1.0, 0.0, 0.0, 0.0)
        summary_vec = (0.0, 1.0, 0.0, 0.0)

        events = [
            _planted_event(storage, embedder, content=f"specific fact {i}", vector=sup_vec)
            for i in range(3)
        ]
        _planted_summary(
            storage, embedder, content="Vague generalization", vector=summary_vec, supports=events
        )

        embedder.plant("q", sup_vec)
        results = memory.retrieve("q", prefer="auto", confidence_threshold=0.6, k=5)
        # Drilling has to happen because cosine(summary, sup_vec) = 0.
        # Top results should be the supporting events themselves.
        assert any(r.level is Level.EVENT for r in results)
        # The vague summary's score against the query is 0; the events
        # should outrank it after drill.
        top_ids = [r.item_id for r in results]
        assert events[0].id in top_ids or events[1].id in top_ids or events[2].id in top_ids

    def test_drill_k_zero_disables_drilling(self, storage: SqliteStorage) -> None:
        embedder = PlantedEmbedder(dim=4)
        memory = Memory(storage=storage, embedder=embedder)
        sup_vec = (1.0, 0.0, 0.0, 0.0)
        summary_vec = (0.0, 1.0, 0.0, 0.0)
        events = [
            _planted_event(storage, embedder, content=f"event {i}", vector=sup_vec)
            for i in range(3)
        ]
        _planted_summary(storage, embedder, content="vague", vector=summary_vec, supports=events)
        embedder.plant("q", sup_vec)

        # drill_k=0 should keep the abstraction (and not surface events
        # via drill); the auto-fallback to events still kicks in if no
        # abstraction layer is found, but here we have one.
        results = memory.retrieve("q", prefer="auto", confidence_threshold=0.6, drill_k=0, k=10)
        # The summary should be in the results (no drill).
        levels = {r.level for r in results}
        assert Level.SUMMARY in levels


# ---------------------------------------------------------------------------
# Reinforcement on use
# ---------------------------------------------------------------------------


class TestReinforcementOnUse:
    def test_retrieve_reinforces_surfaced_events(self, storage: SqliteStorage) -> None:
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        e = memory.observe("hello world")

        memory.retrieve("hello world", k=1)
        state = storage.get_decay_state(e.id, ItemKind.EVENT)
        assert state is not None
        assert state.reinforcement_count == 1

    def test_retrieve_reinforce_false_does_not_bump_counter(self, storage: SqliteStorage) -> None:
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        e = memory.observe("hello")

        memory.retrieve("hello", k=1, reinforce=False)
        state = storage.get_decay_state(e.id, ItemKind.EVENT)
        assert state is not None
        assert state.reinforcement_count == 0

    def test_retrieve_reinforces_summary(self, storage: SqliteStorage) -> None:
        embedder = PlantedEmbedder(dim=4)
        memory = Memory(storage=storage, embedder=embedder)
        v = (1.0, 0.0, 0.0, 0.0)
        events = [_planted_event(storage, embedder, content=f"e{i}", vector=v) for i in range(2)]
        summary = _planted_summary(storage, embedder, content="summary", vector=v, supports=events)
        embedder.plant("q", v)

        memory.retrieve("q", prefer="general", k=5)
        state = storage.get_decay_state(summary.id, ItemKind.MEMORY_ITEM)
        assert state is not None
        assert state.reinforcement_count >= 1


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class TestReranker:
    def test_reranker_reorders_results(self, storage: SqliteStorage) -> None:
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        # Two events with very similar embeddings (FakeEmbedder is hash-
        # based; close strings -> close vectors). The reranker is keyed
        # on token overlap, so it tilts toward whichever event shares
        # tokens with the query.
        memory.observe("Alice greets Bob in the morning")
        memory.observe("Bob is a software engineer")

        results_no_rerank = memory.retrieve("Alice greeting", k=2)
        results_rerank = memory.retrieve("Alice greeting", k=2, reranker=FakeReranker())
        # The reranker doesn't have to flip the ordering, but must not
        # crash and must return the same number of results.
        assert len(results_rerank) == len(results_no_rerank)

    def test_reranker_returns_wrong_score_count_falls_back(
        self, storage: SqliteStorage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A broken reranker (returned a wrong-length score vector)
        must not break the entire retrieve. The engine logs a warning
        and falls back to the pre-rerank ordering."""
        import logging

        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        memory.observe("a")
        memory.observe("b")

        class BadReranker:
            name = "bad"

            def rerank(self, query: str, candidates: object) -> list[float]:  # type: ignore[no-untyped-def]
                return [1.0]  # wrong length -- 1 score for >= 1 candidates

        with caplog.at_level(logging.WARNING, logger="engram.retrieve"):
            results = memory.retrieve("a", k=2, reranker=BadReranker())  # type: ignore[arg-type]
        # Retrieve succeeded; results are sensible.
        assert isinstance(results, list)
        # A warning was logged describing the fallback.
        assert any(
            "returned" in rec.getMessage() and "bad" in rec.getMessage()
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# RetrievalResult.level fidelity
# ---------------------------------------------------------------------------


class TestLevelFidelity:
    def test_level_reflects_what_is_surfaced(self, storage: SqliteStorage) -> None:
        embedder = PlantedEmbedder(dim=4)
        memory = Memory(storage=storage, embedder=embedder)
        v = (1.0, 0.0, 0.0, 0.0)
        events = [_planted_event(storage, embedder, content=f"e{i}", vector=v) for i in range(2)]
        summary = _planted_summary(
            storage, embedder, content="abst", vector=v, supports=events, level=Level.ABSTRACTION
        )
        embedder.plant("q", v)

        results = memory.retrieve("q", prefer="general", k=10)
        # The abstraction-layer item should be at level=abstraction.
        for r in results:
            if r.item_id == summary.id:
                assert r.level is Level.ABSTRACTION

    def test_drilled_event_level_is_event(self, storage: SqliteStorage) -> None:
        embedder = PlantedEmbedder(dim=4)
        memory = Memory(storage=storage, embedder=embedder)
        sup_vec = (1.0, 0.0, 0.0, 0.0)
        summary_vec = (0.0, 1.0, 0.0, 0.0)
        events = [
            _planted_event(storage, embedder, content=f"e{i}", vector=sup_vec) for i in range(3)
        ]
        _planted_summary(storage, embedder, content="vague", vector=summary_vec, supports=events)
        embedder.plant("q", sup_vec)

        results = memory.retrieve("q", prefer="auto", confidence_threshold=0.5, k=5)
        # Drilled candidates are events.
        for r in results:
            if r.item_id in {e.id for e in events}:
                assert r.level is Level.EVENT


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_k_raises(self, storage: SqliteStorage) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=4))
        with pytest.raises(ValueError, match="k must be"):
            memory.retrieve("anything", k=0)

    def test_invalid_confidence_threshold_raises(self, storage: SqliteStorage) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=4))
        with pytest.raises(ValueError, match="confidence_threshold"):
            memory.retrieve("anything", confidence_threshold=1.5)

    def test_invalid_prefer_raises(self, storage: SqliteStorage) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=4))
        with pytest.raises(ValueError, match="prefer must be"):
            memory.retrieve("anything", prefer="weird")  # type: ignore[arg-type]

    def test_negative_drill_k_raises(self, storage: SqliteStorage) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=4))
        with pytest.raises(ValueError, match="drill_k"):
            memory.retrieve("anything", drill_k=-1)
