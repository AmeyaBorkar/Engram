"""Tests for the `consolidate_chat` parameter.

Memory(chat=cheap, consolidate_chat=strong) routes:

  * consolidation (abstraction extraction) through `consolidate_chat`
  * reconcile-MERGE (LLM synthesis of merged content) through
    `consolidate_chat`
  * HyDE / multi-query expansion / EngramAgent through `chat`

This lets users invest the strong model on the irreversible
abstraction step and use the cheap model everywhere else.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from engram import (
    Conflict,
    Memory,
    MemoryItem,
    Resolution,
    SqliteStorage,
)
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder


def _seed_with_provenance_simple(
    storage: SqliteStorage,
    *,
    content: str,
    created_at: datetime,
) -> MemoryItem:
    from engram.schemas import Embedding, Event, ItemKind, Level

    embedder = FakeEmbedder(dim=8)
    ev = Event(content=f"evidence: {content}", created_at=created_at)
    storage.insert_event(ev)
    storage.insert_embedding(
        Embedding(
            item_id=ev.id,
            item_kind=ItemKind.EVENT,
            model=embedder.model,
            dim=embedder.dim,
            vector=tuple(embedder.embed([ev.content])[0]),
        )
    )
    item = MemoryItem(
        level=Level.SUMMARY,
        content=content,
        created_at=created_at,
        valid_from=created_at,
    )
    storage.insert_memory_item(item)
    storage.insert_embedding(
        Embedding(
            item_id=item.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=embedder.model,
            dim=embedder.dim,
            vector=tuple(embedder.embed([content])[0]),
        )
    )
    storage.link_provenance(item.id, ev.id, weight=1.0)
    return item


class TestConsolidateChat:
    def test_default_falls_back_to_chat(self, storage: SqliteStorage) -> None:
        cheap = FakeChat(default="cheap reply")
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=8), chat=cheap)
        # The reconciler uses consolidate_chat; defaults to chat.
        # Smoke: construction works; no separate strong provider.
        assert memory._consolidate_chat is cheap

    def test_explicit_consolidate_chat(self, storage: SqliteStorage) -> None:
        cheap = FakeChat(default="cheap")
        strong = FakeChat(default="strong")
        memory = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            chat=cheap,
            consolidate_chat=strong,
        )
        assert memory._chat is cheap
        assert memory._consolidate_chat is strong

    def test_consolidate_only_provider(self, storage: SqliteStorage) -> None:
        """Allowed: consolidate_chat without chat. consolidation +
        reconcile-MERGE work; chat-dependent features (HyDE etc) are
        disabled."""
        strong = FakeChat(default="strong")
        memory = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            chat=None,
            consolidate_chat=strong,
        )
        assert memory._chat is None
        assert memory._consolidate_chat is strong


class TestMergeUsesConsolidateChat:
    def test_merge_calls_strong_provider(self, storage: SqliteStorage) -> None:
        """MERGE -> the merged content comes from consolidate_chat, not
        from the cheap chat."""
        from engram.reconcile._merge import render_merge_prompt

        embedder = FakeEmbedder(dim=8)
        older = _seed_with_provenance_simple(
            storage,
            content="A",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        newer = _seed_with_provenance_simple(
            storage,
            content="B",
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        conflict = Conflict(source_item_id=newer.id, target_item_id=older.id, similarity=0.9)
        storage.record_conflict(conflict)

        cheap = FakeChat(default="DEFINITELY_NOT_THE_MERGE")
        merge_prompt = render_merge_prompt(a="A", b="B")
        strong = FakeChat(
            scripts={content_hash(merge_prompt): json.dumps({"merged": "STRONG_MERGE"})}
        )
        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=cheap,
            consolidate_chat=strong,
        )
        memory.reconcile(
            conflict.id,
            resolution=Resolution.MERGE,
            now=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
        older_fresh = storage.get_memory_item(older.id)
        assert older_fresh is not None
        assert older_fresh.invalidated_by is not None
        merged = storage.get_memory_item(older_fresh.invalidated_by)
        assert merged is not None
        assert merged.content == "STRONG_MERGE"


@pytest.fixture
def storage() -> Iterator[SqliteStorage]:
    backend = SqliteStorage(":memory:")
    backend.initialize()
    try:
        yield backend
    finally:
        backend.close()
