"""Throughput benchmark + provenance integrity property tests.

Stage 5 DoD targets:
  * Throughput >= 100 events / s on the fake provider.
  * Provenance integrity invariant survives Hypothesis fuzzing.

The throughput test is `slow`-marked - run with `pytest -m slow`. It
seeds N events with planted embeddings (so clustering is fast and
deterministic), scripts FakeChat to return one valid abstraction per
cluster, and asserts the per-event wallclock budget holds.

The Hypothesis test runs many consolidate-style scenarios with
arbitrary cluster shapes and verifies the post-consolidation invariant:
*every* non-event memory item has at least one provenance link.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engram import Memory, SqliteStorage
from engram.consolidation import (
    AbstractionRequest,
    ClusterParams,
    ConsolidationParams,
    render_prompt,
)
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.schemas import Embedding, Event, ItemKind, Level


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _seed_clusters(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
    *,
    n_clusters: int,
    events_per_cluster: int,
) -> dict[int, list[Event]]:
    """Plant `n_clusters` non-overlapping clusters each of
    `events_per_cluster` events. Returns the cluster_idx -> events map."""
    out: dict[int, list[Event]] = {}
    for cidx in range(n_clusters):
        # Distinct one-hot direction per cluster so they don't bleed.
        vec = [0.0] * embedder.dim
        vec[cidx % embedder.dim] = 1.0
        cluster_vec = tuple(vec)
        events = []
        for i in range(events_per_cluster):
            ev = Event(
                content=f"cluster-{cidx}-event-{i}",
                created_at=_now() + timedelta(seconds=cidx * 100 + i),
            )
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=embedder.dim,
                    vector=cluster_vec,
                )
            )
            events.append(ev)
        out[cidx] = events
    return out


def _script_chat_for_clusters(
    cluster_events: dict[int, list[Event]],
) -> FakeChat:
    """Build a FakeChat whose scripts cover every cluster prompt the
    engine will issue."""
    scripts: dict[str, str] = {}
    for cidx, events in cluster_events.items():
        req = AbstractionRequest(observations=tuple(e.content for e in events), cohesion_hint=1.0)
        scripts[content_hash(render_prompt(req))] = json.dumps(
            {
                "abstraction": f"cluster {cidx} pattern",
                "confidence": 0.7,
                "supports": list(range(len(events))),
            }
        )
    return FakeChat(scripts=scripts)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_consolidate_throughput_at_least_100_events_per_second(tmp_path: Path) -> None:
    """SCOREBOARD target: >= 100 events/s with FakeProvider."""
    storage = SqliteStorage(tmp_path / "x.db")
    storage.initialize()
    # Pick dim >= n_clusters so the one-hot cluster vectors stay disjoint.
    embedder = FakeEmbedder(dim=64)

    n_clusters = 30
    events_per_cluster = 10
    total_events = n_clusters * events_per_cluster

    cluster_events = _seed_clusters(
        storage, embedder, n_clusters=n_clusters, events_per_cluster=events_per_cluster
    )
    chat = _script_chat_for_clusters(cluster_events)

    memory = Memory(
        storage=storage,
        embedder=embedder,
        chat=chat,
        consolidation_params=ConsolidationParams(
            cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2),
        ),
    )

    start = time.perf_counter()
    result = memory.consolidate()
    elapsed = time.perf_counter() - start

    assert result.events_processed == total_events
    assert result.abstractions_created == n_clusters
    rate = total_events / elapsed
    assert rate >= 100.0, (
        f"throughput = {rate:.1f} events/s (target >= 100); elapsed={elapsed:.3f}s "
        f"for {total_events} events"
    )
    storage.close()


# ---------------------------------------------------------------------------
# Hypothesis: provenance integrity
# ---------------------------------------------------------------------------


@st.composite
def _consolidation_scenario(
    draw: st.DrawFn,
) -> tuple[int, int]:
    """Generate `(n_clusters, events_per_cluster)` keeping totals modest."""
    n_clusters = draw(st.integers(min_value=1, max_value=4))
    events_per_cluster = draw(st.integers(min_value=2, max_value=5))
    return n_clusters, events_per_cluster


@given(scenario=_consolidation_scenario())
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_provenance_integrity_invariant(tmp_path: Path, scenario: tuple[int, int]) -> None:
    """Across arbitrary consolidation scenarios, every non-event memory
    item has at least one provenance link."""
    n_clusters, events_per_cluster = scenario
    storage = SqliteStorage(tmp_path / "provenance.db")
    storage.initialize()
    try:
        embedder = FakeEmbedder(dim=8)
        cluster_events = _seed_clusters(
            storage,
            embedder,
            n_clusters=n_clusters,
            events_per_cluster=events_per_cluster,
        )
        chat = _script_chat_for_clusters(cluster_events)

        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2),
            ),
        )
        memory.consolidate()

        # Invariant: every memory item that is NOT level=event has at
        # least one provenance link.
        for item in storage.iter_memory_items(include_cold=True):
            if item.level is Level.EVENT:
                continue
            supports = storage.get_supporting_events(item.id)
            assert supports, f"non-event item {item.id} has no provenance"
    finally:
        storage.close()


@given(scenario=_consolidation_scenario())
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_consolidated_events_no_longer_unconsolidated(
    tmp_path: Path, scenario: tuple[int, int]
) -> None:
    """Events that landed in some abstraction must not appear in the
    next iter_unconsolidated_events_with_embeddings call."""
    n_clusters, events_per_cluster = scenario
    storage = SqliteStorage(tmp_path / "leftover.db")
    storage.initialize()
    try:
        embedder = FakeEmbedder(dim=8)
        cluster_events = _seed_clusters(
            storage,
            embedder,
            n_clusters=n_clusters,
            events_per_cluster=events_per_cluster,
        )
        chat = _script_chat_for_clusters(cluster_events)

        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2),
            ),
        )
        result = memory.consolidate()

        # Every consolidated event has a provenance link, so it is no
        # longer "unconsolidated".
        leftover = list(storage.iter_unconsolidated_events_with_embeddings(model=embedder.model))
        assert len(leftover) == n_clusters * events_per_cluster - result.events_consolidated
    finally:
        storage.close()
