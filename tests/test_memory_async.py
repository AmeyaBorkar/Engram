"""Stage 9 async surface tests.

Every public sync method on `Memory` has an async parallel
(`aobserve`, `aretrieve`, `aconsolidate`, `areconcile`, etc.). These
tests verify the wiring -- they don't test asyncio concurrency proper
(which the underlying sync code can't exploit on SQLite's per-thread
connection model).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Iterator
from pathlib import Path
from typing import Any, TypeVar

import pytest

from engram import (
    Conflict,
    ConflictStatus,
    ItemKind,
    Level,
    Memory,
    MemoryItem,
    Outcome,
    Resolution,
    SqliteStorage,
)
from engram.consolidation import (
    AbstractionRequest,
    ClusterParams,
    ConsolidationParams,
    render_prompt,
)
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.schemas import Embedding, Event


# Audit M-134: kept local rather than promoted to conftest.
# Async tests need cross-thread SQLite access (asyncio.to_thread runs
# the sync body on a worker thread). `:memory:` databases are
# per-connection in SQLite -- two threads see two separate databases.
# A tempfile sidesteps the issue. The shared conftest `storage` fixture
# uses `:memory:` and is unsuitable here; `disk_storage` is conftest's
# equivalent but is named for the storage-on-disk semantics, not for
# cross-thread access. Renaming would churn many tests; keeping a local
# fixture under its meaning-specific name documents the intent.
@pytest.fixture
def file_storage(tmp_path: Path) -> Iterator[SqliteStorage]:
    backend = SqliteStorage(tmp_path / "async.db")
    backend.initialize()
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def memory(file_storage: SqliteStorage) -> Memory:
    return Memory(storage=file_storage, embedder=FakeEmbedder(dim=8))


_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


class TestAsyncBasics:
    def test_aobserve_returns_event(self, memory: Memory) -> None:
        event = _run(memory.aobserve("hello"))
        assert isinstance(event, Event)
        assert event.content == "hello"

    def test_aretrieve_returns_results(self, memory: Memory) -> None:
        _run(memory.aobserve("the cat sat on the mat"))
        results = _run(memory.aretrieve("cat", k=5))
        assert any("cat" in r.content for r in results)

    def test_aretrieve_accepts_as_of(self, memory: Memory) -> None:
        from datetime import datetime, timezone

        _run(memory.aobserve("only fact"))
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        # No items at as_of=2099 would only show items valid then.
        # Since events default valid_from=created_at (now), they ARE valid
        # at 2099. But invalidated items would be excluded; we just check
        # the parameter threads through.
        results = _run(memory.aretrieve("only", k=5, as_of=future))
        # Stage 6 retrieve over events falls through (no abstractions
        # exist); the events surface. Just check no crash.
        assert isinstance(results, list)


class TestAsyncDecay:
    def test_areinforce(self, memory: Memory) -> None:
        event = _run(memory.aobserve("x"))
        state = _run(memory.areinforce(event.id, ItemKind.EVENT))
        assert state.reinforcement_count == 1

    def test_acorroborate(self, memory: Memory, file_storage: SqliteStorage) -> None:
        item = MemoryItem(level=Level.SUMMARY, content="x")
        file_storage.insert_memory_item(item)
        state = _run(memory.acorroborate(item.id, ItemKind.MEMORY_ITEM))
        assert state.corroboration_count == 1

    def test_acontradict(self, memory: Memory, file_storage: SqliteStorage) -> None:
        item = MemoryItem(level=Level.SUMMARY, content="x")
        file_storage.insert_memory_item(item)
        state = _run(memory.acontradict(item.id, ItemKind.MEMORY_ITEM))
        assert state.contradiction_count == 1


class TestAsyncProcedure:
    def test_arecord_then_aretrieve(self, memory: Memory) -> None:
        proc = _run(
            memory.arecord_procedure(
                "flaky test",
                "rerun with --no-cov",
                outcome=Outcome.SUCCESS,
            )
        )
        matches = _run(memory.aretrieve_procedures("flaky test", k=1, reinforce=False))
        assert len(matches) == 1
        assert matches[0].procedure.id == proc.id

    def test_aupdate_outcome(self, memory: Memory) -> None:
        proc = _run(memory.arecord_procedure("s", "a"))
        updated = _run(memory.aupdate_outcome(proc.id, Outcome.SUCCESS))
        assert updated.outcome is Outcome.SUCCESS


class TestAsyncConsolidation:
    def test_aconsolidate_runs(self, file_storage: SqliteStorage) -> None:
        embedder = FakeEmbedder(dim=8)
        # Two clustering-friendly events.
        same_vec = (1.0,) + (0.0,) * 7
        events = [Event(content=f"event-{i}") for i in range(2)]
        for e in events:
            file_storage.insert_event(e)
            file_storage.insert_embedding(
                Embedding(
                    item_id=e.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=same_vec,
                )
            )
        req = AbstractionRequest(observations=tuple(e.content for e in events), cohesion_hint=1.0)
        chat = FakeChat(
            scripts={
                content_hash(render_prompt(req)): json.dumps(
                    {"abstraction": "summary", "confidence": 0.7, "supports": [0, 1]}
                )
            }
        )
        memory = Memory(
            storage=file_storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2)
            ),
        )
        result = _run(memory.aconsolidate())
        assert result.abstractions_created == 1

    def test_apromote_runs(self, file_storage: SqliteStorage) -> None:
        # promote with no eligible candidates is a no-op but must not crash.
        memory = Memory(
            storage=file_storage,
            embedder=FakeEmbedder(dim=8),
            chat=FakeChat(default="{}"),
        )
        result = _run(memory.apromote())
        assert result.promoted == 0


class TestAsyncReconcile:
    def test_areconcile_and_alist_conflicts(
        self, memory: Memory, file_storage: SqliteStorage
    ) -> None:
        from datetime import datetime, timezone

        older = MemoryItem(
            level=Level.SUMMARY,
            content="older",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        newer = MemoryItem(
            level=Level.SUMMARY,
            content="newer",
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            valid_from=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        file_storage.insert_memory_item(older)
        file_storage.insert_memory_item(newer)
        c = Conflict(source_item_id=newer.id, target_item_id=older.id, similarity=0.9)
        file_storage.record_conflict(c)

        # alist_conflicts surfaces it as OPEN.
        rows = _run(memory.alist_conflicts(status=ConflictStatus.OPEN))
        assert {x.id for x in rows} == {c.id}

        # areconcile resolves it.
        resolved = _run(
            memory.areconcile(
                c.id,
                resolution=Resolution.PREFER_RECENT,
                now=datetime(2026, 5, 1, tzinfo=timezone.utc),
            )
        )
        assert resolved.resolved_winner_id == newer.id

        # alist_conflicts shows nothing OPEN now.
        rows = _run(memory.alist_conflicts(status=ConflictStatus.OPEN))
        assert rows == []
