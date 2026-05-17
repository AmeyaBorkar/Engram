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

    def test_final_order_is_score_sorted(self, memory: Memory) -> None:
        """Audit M-34: siblings used to be appended at the tail
        regardless of score, leaving the final list out of rank order.
        After the fix, the merged list is re-sorted so a high-scoring
        sibling sits adjacent to its parent in true rank order."""
        a, b = _seed_pair(
            memory,
            a_text="X is in state alpha",
            b_text="X is in state beta",
            same_vector=False,
        )
        # Seed a low-similarity decoy that scores BELOW the conflict
        # sibling so we can detect the re-sort.
        decoy = MemoryItem(
            level=Level.SUMMARY,
            content="unrelated content far away",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        memory.storage.insert_memory_item(decoy)
        # Plant the decoy with a vector orthogonal to the query vector.
        embedder = memory.embedder
        decoy_vec = tuple(embedder.embed([decoy.content])[0])
        memory.storage.insert_embedding(
            Embedding(
                item_id=decoy.id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=embedder.model,
                dim=embedder.dim,
                vector=decoy_vec,
            )
        )
        results = memory.retrieve(
            "X is in state alpha",
            k=10,
            prefer="general",
            confidence_threshold=0.0,
            reinforce=False,
            surface_conflicts=True,
        )
        # Scores must be monotonically non-increasing in the surface
        # output.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), scores
        # Sibling `b` should sit AT or just below `a` (its parent),
        # not at the bottom of the list past unrelated decoys.
        ids = [r.item_id for r in results]
        a_idx = ids.index(a.id)
        b_idx = ids.index(b.id)
        assert b_idx == a_idx + 1, (a_idx, b_idx, ids)

    def test_shared_sibling_fetched_once(self, memory: Memory) -> None:
        """Audit M-33: two results that share the SAME conflict
        partner used to each trigger a get_memory_item lookup for
        that partner.  After the dedup fix, the partner is fetched
        once regardless of how many parents reference it."""
        a, b = _seed_pair(
            memory,
            a_text="X is in state alpha",
            b_text="X is in state beta",
            same_vector=True,
        )
        # Plant a third item `c` that's also in conflict with `b`
        # (so two of the K results -> same sibling).
        c = MemoryItem(
            level=Level.SUMMARY,
            content="X is in state alpha (variant)",
            created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            valid_from=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        memory.storage.insert_memory_item(c)
        vec_a = tuple(memory.embedder.embed(["X is in state alpha"])[0])
        memory.storage.insert_embedding(
            Embedding(
                item_id=c.id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=memory.embedder.model,
                dim=memory.embedder.dim,
                vector=vec_a,
            )
        )
        memory.storage.record_conflict(
            Conflict(source_item_id=b.id, target_item_id=c.id, similarity=0.9)
        )
        # Count get_memory_item calls.
        real = memory.storage.get_memory_item
        calls: list[object] = []

        def _spy(item_id: object) -> object:
            calls.append(item_id)
            return real(item_id)  # type: ignore[arg-type]

        memory.storage.get_memory_item = _spy  # type: ignore[method-assign,attr-defined]
        try:
            memory.retrieve(
                "X is in state alpha",
                k=10,
                prefer="general",
                confidence_threshold=0.0,
                reinforce=False,
                surface_conflicts=True,
            )
        finally:
            memory.storage.get_memory_item = real  # type: ignore[method-assign,attr-defined]
        # Sibling `b` participates as conflict partner for BOTH `a`
        # and `c`.  The dedup'd lookup must fetch it exactly once.
        b_lookups = [cid for cid in calls if cid == b.id]
        assert len(b_lookups) == 1, b_lookups
