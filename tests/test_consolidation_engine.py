"""End-to-end tests for `ConsolidationEngine` + `Memory.consolidate`."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram import Memory, SqliteStorage
from engram.consolidation import (
    AbstractionResult,
    ClusterParams,
    ConsolidationEngine,
    ConsolidationParams,
    ConsolidationResult,
)
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.schemas import Event, ItemKind, Level


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ab_response(text: str, *, supports: list[int] | None = None, confidence: float = 0.7) -> str:
    return json.dumps({"abstraction": text, "confidence": confidence, "supports": supports or []})


# ---------------------------------------------------------------------------
# Construction / parameter validation
# ---------------------------------------------------------------------------


class TestConsolidationParams:
    def test_defaults(self) -> None:
        p = ConsolidationParams()
        assert p.support_weight == 0.5
        assert p.level is Level.SUMMARY
        assert p.abstraction_max_retries == 1

    def test_support_weight_bounds(self) -> None:
        with pytest.raises(ValueError, match="support_weight"):
            ConsolidationParams(support_weight=-0.01)
        with pytest.raises(ValueError, match="support_weight"):
            ConsolidationParams(support_weight=1.01)

    def test_event_level_rejected(self) -> None:
        with pytest.raises(ValueError, match="raw events"):
            ConsolidationParams(level=Level.EVENT)


# ---------------------------------------------------------------------------
# Engine pipeline (golden trace)
# ---------------------------------------------------------------------------


class TestConsolidateGoldenTrace:
    def test_clusters_groups_events_and_creates_abstraction(self, tmp_path: Path) -> None:
        # Six events arranged into two tight clusters by content. Use the
        # FakeEmbedder, which produces deterministic vectors per text.
        # Because hash-based embeddings are pseudo-orthogonal, we craft
        # textual similarity by reusing tokens: the two groups share most
        # text so their embeddings are NOT necessarily nearby.
        # For a deterministic golden trace we instead seed the embedder
        # so distinct contents land in distinct directions, and use the
        # storage layer to plant the embeddings ourselves with controlled
        # vectors.
        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()

        embedder = FakeEmbedder(dim=8)

        # Manually craft two clusters by inserting events + embeddings
        # with known unit vectors.
        from engram.schemas import Embedding

        cluster_a_vec = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cluster_b_vec = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cluster_a_events = []
        cluster_b_events = []
        for i in range(3):
            ev = Event(content=f"alpha-{i}", created_at=_now() + timedelta(seconds=i))
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=cluster_a_vec,
                )
            )
            cluster_a_events.append(ev)
        for i in range(3):
            ev = Event(content=f"beta-{i}", created_at=_now() + timedelta(seconds=10 + i))
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=cluster_b_vec,
                )
            )
            cluster_b_events.append(ev)

        # Build the rendered prompt for each cluster and script the FakeChat.
        from engram.consolidation import AbstractionRequest, render_prompt

        req_a = AbstractionRequest(
            observations=tuple(e.content for e in cluster_a_events),
            cohesion_hint=1.0,
        )
        req_b = AbstractionRequest(
            observations=tuple(e.content for e in cluster_b_events),
            cohesion_hint=1.0,
        )
        scripts = {
            content_hash(render_prompt(req_a)): _ab_response(
                "alpha events form a generalization", supports=[0, 1, 2]
            ),
            content_hash(render_prompt(req_b)): _ab_response(
                "beta events form a generalization", supports=[0]
            ),
        }
        chat = FakeChat(scripts=scripts)

        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(
                    method="agglomerative",
                    cohesion_threshold=0.95,
                    min_cluster_size=2,
                ),
            ),
        )
        result = memory.consolidate()
        assert isinstance(result, ConsolidationResult)
        assert result.events_processed == 6
        assert result.clusters_formed == 2
        assert result.abstractions_created == 2
        assert result.abstractions_failed == 0
        assert result.events_consolidated == 6

        # Two memory items landed at level=summary with provenance to the
        # right events.
        items = storage.list_memory_items(level=Level.SUMMARY, limit=10)
        assert len(items) == 2
        contents = {i.content for i in items}
        assert contents == {
            "alpha events form a generalization",
            "beta events form a generalization",
        }
        # Provenance integrity.
        for item in items:
            supports = storage.get_supporting_events(item.id)
            assert len(supports) == 3
            assert all(e is not None for e in supports)
            # Metadata records prompt version + confidence.
            assert "consolidation" in item.metadata
            assert item.metadata["consolidation"]["prompt_version"] == "v1"

        # The events that were consolidated now have provenance and won't
        # be re-fetched on the next pass.
        leftover = list(storage.iter_unconsolidated_events_with_embeddings(model=embedder.model))
        assert leftover == []

        storage.close()

    def test_idempotent_second_call_does_nothing(self, tmp_path: Path) -> None:
        # After everything is consolidated, a second consolidate() finds
        # no work and reports zeros.
        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        from engram.consolidation import AbstractionRequest, render_prompt
        from engram.schemas import Embedding

        events = []
        same_vec = (1.0,) + (0.0,) * 7
        for i in range(3):
            ev = Event(content=f"e-{i}", created_at=_now() + timedelta(seconds=i))
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=same_vec,
                )
            )
            events.append(ev)

        req = AbstractionRequest(observations=tuple(e.content for e in events), cohesion_hint=1.0)
        chat = FakeChat(scripts={content_hash(render_prompt(req)): _ab_response("X", supports=[0])})
        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2)
            ),
        )

        first = memory.consolidate()
        assert first.abstractions_created == 1

        second = memory.consolidate()
        assert second.events_processed == 0
        assert second.abstractions_created == 0

        storage.close()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_parse_failure_increments_failed_counter(self, tmp_path: Path) -> None:
        # Chat returns invalid JSON; the engine should mark this cluster
        # as failed without crashing the whole pass.
        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        from engram.schemas import Embedding

        same = (1.0,) + (0.0,) * 7
        for i in range(2):
            ev = Event(content=f"e-{i}")
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=same,
                )
            )

        chat = FakeChat(default="not json")
        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2),
                abstraction_max_retries=0,
            ),
        )
        result = memory.consolidate()
        assert result.clusters_formed == 1
        assert result.abstractions_failed == 1
        assert result.abstractions_created == 0
        # No memory_item landed.
        assert storage.list_memory_items(level=Level.SUMMARY) == []
        storage.close()

    def test_consolidate_without_chat_raises(self, tmp_path: Path) -> None:
        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=8))
        with pytest.raises(RuntimeError, match="chat provider"):
            memory.consolidate()
        storage.close()


# ---------------------------------------------------------------------------
# Provenance weights
# ---------------------------------------------------------------------------


class TestProvenanceWeights:
    def test_supports_get_full_weight_others_get_support_weight(self, tmp_path: Path) -> None:
        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        from engram.consolidation import AbstractionRequest, render_prompt
        from engram.schemas import Embedding

        same = (1.0,) + (0.0,) * 7
        events = []
        for i in range(3):
            ev = Event(content=f"e-{i}", created_at=_now() + timedelta(seconds=i))
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=same,
                )
            )
            events.append(ev)
        req = AbstractionRequest(observations=tuple(e.content for e in events), cohesion_hint=1.0)
        # LLM marks index 0 as load-bearing.
        chat = FakeChat(scripts={content_hash(render_prompt(req)): _ab_response("Y", supports=[0])})
        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2),
                support_weight=0.3,
            ),
        )
        memory.consolidate()
        item = storage.list_memory_items(level=Level.SUMMARY, limit=1)[0]

        weights_by_event = {}
        for ev in events:
            for link in [ln for ln in storage.get_supporting_events(item.id) if ln.id == ev.id]:
                # Get the link weight via direct lookup on provenance_links.
                row = (
                    storage._connect()
                    .execute(
                        "SELECT weight FROM provenance_links "
                        "WHERE memory_item_id = ? AND event_id = ?",
                        (item.id.bytes, ev.id.bytes),
                    )
                    .fetchone()
                )
                weights_by_event[ev.id] = row["weight"]
                _ = link

        # Index 0 -> events[0] gets 1.0; rest get 0.3.
        assert weights_by_event[events[0].id] == pytest.approx(1.0)
        assert weights_by_event[events[1].id] == pytest.approx(0.3)
        assert weights_by_event[events[2].id] == pytest.approx(0.3)
        storage.close()


# ---------------------------------------------------------------------------
# Result schema sanity
# ---------------------------------------------------------------------------


class TestResultObject:
    def test_empty_store_returns_zeros(self, tmp_path: Path) -> None:
        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=8), chat=FakeChat(default="{}"))
        result = memory.consolidate()
        assert result.events_processed == 0
        assert result.clusters_formed == 0
        assert result.abstractions_created == 0
        assert result.abstractions_failed == 0
        assert result.events_consolidated == 0
        assert result.duration_ms >= 0
        storage.close()


# ---------------------------------------------------------------------------
# Direct engine instantiation
# ---------------------------------------------------------------------------


class TestEngineDirect:
    def test_engine_can_be_used_outside_memory(self, tmp_path: Path) -> None:
        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)
        chat = FakeChat(default=_ab_response("X", supports=[]))
        engine = ConsolidationEngine(
            storage,
            embedder=embedder,
            chat=chat,
            params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2)
            ),
        )
        result = engine.consolidate(max_events=10)
        assert isinstance(result, ConsolidationResult)
        storage.close()


# ---------------------------------------------------------------------------
# Result is a dataclass with the documented fields
# ---------------------------------------------------------------------------


def test_consolidation_result_is_immutable() -> None:
    import dataclasses

    result = ConsolidationResult(
        started_at=_now(),
        duration_ms=1.0,
        events_processed=0,
        clusters_formed=0,
        abstractions_created=0,
        abstractions_failed=0,
        events_consolidated=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.events_processed = 99  # type: ignore[misc]


def test_abstraction_result_field_named_supports() -> None:
    # Defensive: a typo in the field name would silently slip through
    # consolidation; pin the schema.
    fields = AbstractionResult.model_fields
    assert "supports" in fields
    assert "abstraction" in fields
    assert "confidence" in fields


# ---------------------------------------------------------------------------
# Internal helpers (defensive guards)
# ---------------------------------------------------------------------------


def test_unique_members_rejects_duplicates() -> None:
    """M-56: a cluster with a duplicate member index is a clustering
    bug; the engine asserts up front so the failure is loud at the
    boundary rather than silently truncating the provenance-weights
    dict downstream."""
    from engram.consolidation import ClusterAssignment
    from engram.consolidation._engine import _unique_members

    # Sanity: unique input passes through.
    asg = ClusterAssignment(members=(0, 2, 3), cohesion=0.9)
    assert _unique_members(asg) == [0, 2, 3]

    dup = ClusterAssignment(members=(0, 1, 0), cohesion=0.8)
    with pytest.raises(ValueError, match="duplicate member"):
        _unique_members(dup)


def test_pass_deadline_validates_positive() -> None:
    """H-58: `pass_deadline_s` must be > 0 if set."""
    with pytest.raises(ValueError, match="pass_deadline_s"):
        ConsolidationParams(pass_deadline_s=0.0)
    with pytest.raises(ValueError, match="pass_deadline_s"):
        ConsolidationParams(pass_deadline_s=-1.0)
    # Defaults + None + positive all accept.
    ConsolidationParams()
    ConsolidationParams(pass_deadline_s=None)
    ConsolidationParams(pass_deadline_s=0.001)


def test_pass_deadline_stops_dispatch(tmp_path: Path) -> None:
    """H-58: when the aggregate deadline fires the engine stops
    dispatching new clusters and returns partial progress.  We use a
    chat provider that sleeps to make the budget bite without making
    the test slow."""
    import time as _time

    from engram.providers._fake import FakeChat, FakeEmbedder
    from engram.schemas import Embedding

    storage = SqliteStorage(tmp_path / "x.db")
    storage.initialize()
    embedder = FakeEmbedder(dim=8)

    # Plant two well-separated 3-event clusters so the engine produces
    # two cluster assignments; the deadline should bite after the
    # first cluster's chat call.
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cluster_a_vec = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cluster_b_vec = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(3):
        ev = Event(content=f"a-{i}", created_at=base + timedelta(seconds=i))
        storage.insert_event(ev)
        storage.insert_embedding(
            Embedding(
                item_id=ev.id,
                item_kind=ItemKind.MEMORY_ITEM if False else ItemKind.EVENT,
                model=embedder.model,
                dim=8,
                vector=cluster_a_vec,
            )
        )
    for i in range(3):
        ev = Event(content=f"b-{i}", created_at=base + timedelta(seconds=10 + i))
        storage.insert_event(ev)
        storage.insert_embedding(
            Embedding(
                item_id=ev.id,
                item_kind=ItemKind.EVENT,
                model=embedder.model,
                dim=8,
                vector=cluster_b_vec,
            )
        )

    class SleepyChat(FakeChat):
        def __init__(self) -> None:
            super().__init__(default=_ab_response("abs", confidence=0.7))
            self.calls = 0

        def chat(self, messages: list) -> str:  # type: ignore[override]
            self.calls += 1
            _time.sleep(0.2)
            return super().chat(messages)

    chat = SleepyChat()
    # Deadline shorter than two chat calls (0.4s) but longer than the
    # cluster-loop overhead (~0s).  After the first chat call (0.2s)
    # the deadline check (0.05s) trips and the second cluster is
    # deferred.
    memory = Memory(
        storage=storage,
        embedder=embedder,
        chat=chat,
        consolidation_params=ConsolidationParams(
            cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2),
            pass_deadline_s=0.05,
        ),
    )
    result = memory.consolidate()
    assert result.clusters_formed == 2
    # Exactly one chat call should have happened; the second cluster
    # was deferred by the deadline.
    assert chat.calls == 1
    assert result.abstractions_created == 1
    storage.close()


def test_dedupe_conflicts_keeps_first_by_candidate_id() -> None:
    """H-60: vector recall can surface the same candidate id twice when
    multiple levels share text + embedding. Recording two `Conflict`
    rows for the same (source, target) pair raises a uniqueness error;
    dedupe up front so the contradiction-detection pass survives."""
    from uuid import uuid4

    from engram.consolidation import DetectedConflict, Verdict
    from engram.consolidation._engine import _dedupe_conflicts

    a_id = uuid4()
    b_id = uuid4()
    d1 = DetectedConflict(candidate_id=a_id, similarity=0.95, verdict=Verdict.CONTRADICT)
    d2 = DetectedConflict(candidate_id=b_id, similarity=0.91, verdict=Verdict.CONTRADICT)
    d3 = DetectedConflict(candidate_id=a_id, similarity=0.88, verdict=Verdict.CONTRADICT)

    out = _dedupe_conflicts([d1, d2, d3])
    # First-seen wins; the d3 duplicate of a_id is dropped.
    assert [c.candidate_id for c in out] == [a_id, b_id]
    assert out[0].similarity == pytest.approx(0.95)
