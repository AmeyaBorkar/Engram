"""Contradiction-aware retrieval (E.14) tests.

When two memory items have an open conflict, `Memory.retrieve(
surface_conflicts=True)` includes BOTH sides in the result list so
the agent sees the disagreement rather than picking silently.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engram import (
    Conflict,
    Memory,
    Resolution,
    SqliteStorage,
)
from engram.providers._fake import FakeEmbedder
from engram.schemas import Embedding, ItemKind, Level, MemoryItem


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


def _seed_pair(
    memory: Memory,
    *,
    a_text: str,
    b_text: str,
    same_vector: bool = True,
) -> tuple[MemoryItem, MemoryItem]:
    """Seed two memory items that share the same embedding (so both
    surface from the same query). Record an open Conflict between them."""
    a = MemoryItem(
        level=Level.SUMMARY,
        content=a_text,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    b = MemoryItem(
        level=Level.SUMMARY,
        content=b_text,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        valid_from=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    memory.storage.insert_memory_item(a)
    memory.storage.insert_memory_item(b)
    vec = tuple(memory.embedder.embed([a_text])[0])
    other_vec = vec if same_vector else tuple(memory.embedder.embed([b_text])[0])
    memory.storage.insert_embedding(
        Embedding(
            item_id=a.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=memory.embedder.model,
            dim=memory.embedder.dim,
            vector=vec,
        )
    )
    memory.storage.insert_embedding(
        Embedding(
            item_id=b.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=memory.embedder.model,
            dim=memory.embedder.dim,
            vector=other_vec,
        )
    )
    conflict = Conflict(source_item_id=b.id, target_item_id=a.id, similarity=0.95)
    memory.storage.record_conflict(conflict)
    return a, b


class TestSurfaceConflicts:
    def test_off_default(self, memory: Memory) -> None:
        a, b = _seed_pair(memory, a_text="X is true", b_text="X is false")
        results = memory.retrieve(
            "X is true",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
        )
        # Both already share embeddings so both retrieve normally;
        # but the point: surface_conflicts=False doesn't ADD anything.
        ids = {r.item_id for r in results}
        # Both surface since they share embedding (default retrieve
        # gives both). The behavior we're checking: no special
        # conflict-driven expansion happens.
        assert len(results) == 2
        assert {a.id, b.id} == ids

    def test_on_adds_conflict_partner_if_not_already_present(
        self, memory: Memory
    ) -> None:
        """The classic case: query embedding matches only ONE side
        strongly. Without `surface_conflicts`, the other side doesn't
        appear. With `surface_conflicts=True`, it does."""
        a, b = _seed_pair(
            memory,
            a_text="X is in state alpha",
            b_text="X is in state beta",
            same_vector=False,
        )
        # Query matches A strongly (its embedding); B doesn't surface
        # naturally because its embedding is different.
        without = memory.retrieve(
            "X is in state alpha",
            k=1,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
        )
        # Only A surfaces at k=1 without the flag.
        without_ids = {r.item_id for r in without}
        assert a.id in without_ids
        # With the flag: the conflict partner B is appended.
        with_conflicts = memory.retrieve(
            "X is in state alpha",
            k=1,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            surface_conflicts=True,
        )
        with_ids = {r.item_id for r in with_conflicts}
        assert a.id in with_ids
        assert b.id in with_ids

    def test_resolved_conflicts_dont_surface(self, memory: Memory) -> None:
        """After reconcile (which invalidates the loser), the loser
        doesn't appear via surface_conflicts -- it's been superseded.
        """
        a, b = _seed_pair(
            memory,
            a_text="X is true",
            b_text="X is false",
            same_vector=False,
        )
        # Resolve the conflict so a is invalidated by b.
        for c in memory.list_conflicts():
            memory.reconcile(c.id, resolution=Resolution.PREFER_RECENT)
        results = memory.retrieve(
            "X is false",
            k=1,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            surface_conflicts=True,
        )
        ids = {r.item_id for r in results}
        # B (the winner) surfaces. A (the invalidated loser) does NOT
        # because surface_conflicts only includes ACTIVE conflict
        # partners (open conflict + non-invalidated sibling), and
        # the conflict is now resolved.
        assert b.id in ids
        assert a.id not in ids

    def test_dedup_doesnt_double_add(self, memory: Memory) -> None:
        """If both sides ALREADY surface naturally, surface_conflicts
        doesn't double-add."""
        a, b = _seed_pair(
            memory, a_text="X is true", b_text="X is false", same_vector=True
        )
        results = memory.retrieve(
            "X is true",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            surface_conflicts=True,
        )
        ids = [r.item_id for r in results]
        # Both surface but no duplicates.
        assert len(ids) == len(set(ids)) == 2
        assert {a.id, b.id} == set(ids)
