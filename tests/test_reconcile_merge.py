"""Stage 8 v0.3.1 MERGE resolution tests.

Exercises the `Resolution.MERGE` policy where the reconciler asks the
chat provider to synthesize a new memory item that captures both
sides of a conflict. The originals are invalidated pointing to the
new merged item.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

import pytest

from engram import (
    Conflict,
    ConflictStatus,
    Embedding,
    Event,
    ItemKind,
    Level,
    Memory,
    MemoryItem,
    Resolution,
    SqliteStorage,
    new_id,
)
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.reconcile import Reconciler
from engram.reconcile._merge import (
    parse_merge_response,
    render_merge_prompt,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _seed_with_provenance(
    storage: SqliteStorage,
    embedder: FakeEmbedder,
    *,
    content: str,
    created_at: datetime,
) -> MemoryItem:
    """Plant a memory item plus one supporting event + provenance link."""
    event = Event(content=f"evidence: {content}", created_at=created_at)
    storage.insert_event(event)
    storage.insert_embedding(
        Embedding(
            item_id=event.id,
            item_kind=ItemKind.EVENT,
            model=embedder.model,
            dim=embedder.dim,
            vector=tuple(embedder.embed([event.content])[0]),
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
    storage.link_provenance(item.id, event.id, weight=1.0)
    return item


# ---------------------------------------------------------------------------
# Prompt + parser
# ---------------------------------------------------------------------------


class TestMergePromptAndParser:
    def test_render_inlines_payloads(self) -> None:
        prompt = render_merge_prompt(a="multi\nline", b="single")
        # Newlines inside the payload are escaped, so they can't
        # accidentally introduce new prompt sections.
        statements = prompt[prompt.find("STATEMENTS"):]
        assert "multi\\nline" in statements

    def test_critical_rules_precede_statements(self) -> None:
        prompt = render_merge_prompt(a="x", b="y")
        assert prompt.index("CRITICAL RULES") < prompt.index("STATEMENTS")

    def test_parse_round_trip(self) -> None:
        out = parse_merge_response(json.dumps({"merged": "synthesized truth"}))
        assert out.merged == "synthesized truth"

    def test_parse_strips_code_fence(self) -> None:
        out = parse_merge_response(
            "```json\n" + json.dumps({"merged": "x"}) + "\n```"
        )
        assert out.merged == "x"

    def test_parse_rejects_missing_field(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError, match="schema mismatch"):
            parse_merge_response(json.dumps({"output": "x"}))

    def test_parse_rejects_invalid_json(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError, match="invalid merge JSON"):
            parse_merge_response("not json at all")

    def test_parse_rejects_empty(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError, match="empty"):
            parse_merge_response("")


# ---------------------------------------------------------------------------
# Reconciler MERGE path
# ---------------------------------------------------------------------------


class TestReconcilerMergePath:
    def test_merge_creates_new_item_and_invalidates_both(
        self, storage: SqliteStorage
    ) -> None:
        embedder = FakeEmbedder(dim=8)
        older = _seed_with_provenance(
            storage, embedder, content="A is true", created_at=_utc(2026, 1, 1)
        )
        newer = _seed_with_provenance(
            storage, embedder, content="A is false", created_at=_utc(2026, 4, 1)
        )
        conflict = Conflict(
            source_item_id=newer.id, target_item_id=older.id, similarity=0.95
        )
        storage.record_conflict(conflict)
        # Script FakeChat with the merge response. The Reconciler orders
        # the (a, b) pair so b is the newer item -- that's our `newer`.
        prompt = render_merge_prompt(a=older.content, b=newer.content)
        chat = FakeChat(
            scripts={
                content_hash(prompt): json.dumps(
                    {"merged": "A's state changed over time."}
                )
            }
        )
        reconciler = Reconciler(storage, embedder=embedder, chat=chat)
        out = reconciler.reconcile(
            conflict.id,
            resolution=Resolution.MERGE,
            now=_utc(2026, 5, 1),
        )
        assert out.status is ConflictStatus.RESOLVED
        assert out.resolution is Resolution.MERGE
        assert out.resolved_winner_id is None

        # Both originals are invalidated pointing to the merged item.
        older_fresh = storage.get_memory_item(older.id)
        newer_fresh = storage.get_memory_item(newer.id)
        assert older_fresh is not None
        assert newer_fresh is not None
        assert older_fresh.invalidated_at == _utc(2026, 5, 1)
        assert newer_fresh.invalidated_at == _utc(2026, 5, 1)
        assert older_fresh.invalidated_by == newer_fresh.invalidated_by
        merged_id = older_fresh.invalidated_by
        assert merged_id is not None

        # The merged item exists with the LLM-produced content.
        merged = storage.get_memory_item(merged_id)
        assert merged is not None
        assert merged.content == "A's state changed over time."
        assert merged.level is Level.SUMMARY
        assert merged.invalidated_at is None
        # Provenance is the union of both parents' supporting events.
        events = storage.get_supporting_events(merged_id)
        assert len(events) == 2

    def test_merge_falls_back_to_b_on_parse_failure(
        self, storage: SqliteStorage
    ) -> None:
        """If the chat provider returns garbage, the merged content
        defaults to statement b (the newer item) verbatim."""
        embedder = FakeEmbedder(dim=8)
        older = _seed_with_provenance(
            storage, embedder, content="X was here", created_at=_utc(2026, 1, 1)
        )
        newer = _seed_with_provenance(
            storage, embedder, content="X moved on", created_at=_utc(2026, 4, 1)
        )
        conflict = Conflict(
            source_item_id=newer.id, target_item_id=older.id, similarity=0.95
        )
        storage.record_conflict(conflict)
        chat = FakeChat(default="not json")
        reconciler = Reconciler(storage, embedder=embedder, chat=chat)
        out = reconciler.reconcile(
            conflict.id,
            resolution=Resolution.MERGE,
            now=_utc(2026, 5, 1),
        )
        # Find the merged item via the invalidated_by pointer.
        older_fresh = storage.get_memory_item(older.id)
        assert older_fresh is not None
        assert older_fresh.invalidated_by is not None
        merged = storage.get_memory_item(older_fresh.invalidated_by)
        assert merged is not None
        assert merged.content == newer.content
        # Conflict still resolves cleanly.
        assert out.status is ConflictStatus.RESOLVED

    def test_merge_requires_chat(self, storage: SqliteStorage) -> None:
        embedder = FakeEmbedder(dim=8)
        older = _seed_with_provenance(
            storage, embedder, content="x", created_at=_utc(2026, 1, 1)
        )
        newer = _seed_with_provenance(
            storage, embedder, content="y", created_at=_utc(2026, 4, 1)
        )
        conflict = Conflict(
            source_item_id=newer.id, target_item_id=older.id, similarity=0.95
        )
        storage.record_conflict(conflict)
        reconciler = Reconciler(storage, embedder=embedder, chat=None)
        with pytest.raises(ValueError, match="chat provider"):
            reconciler.reconcile(
                conflict.id,
                resolution=Resolution.MERGE,
                now=_utc(2026, 5, 1),
            )

    def test_merge_requires_embedder(self, storage: SqliteStorage) -> None:
        embedder = FakeEmbedder(dim=8)
        older = _seed_with_provenance(
            storage, embedder, content="x", created_at=_utc(2026, 1, 1)
        )
        newer = _seed_with_provenance(
            storage, embedder, content="y", created_at=_utc(2026, 4, 1)
        )
        conflict = Conflict(
            source_item_id=newer.id, target_item_id=older.id, similarity=0.95
        )
        storage.record_conflict(conflict)
        reconciler = Reconciler(
            storage, embedder=None, chat=FakeChat(default="{}")
        )
        with pytest.raises(ValueError, match="embedder"):
            reconciler.reconcile(
                conflict.id,
                resolution=Resolution.MERGE,
                now=_utc(2026, 5, 1),
            )

    def test_merge_requires_some_provenance(
        self, storage: SqliteStorage
    ) -> None:
        """Both parents without supporting events can't merge -- the new
        memory item is a non-EVENT level and storage enforces non-empty
        supporting_event_ids at that level."""
        embedder = FakeEmbedder(dim=8)
        older = MemoryItem(
            level=Level.SUMMARY,
            content="x",
            created_at=_utc(2026, 1, 1),
            valid_from=_utc(2026, 1, 1),
        )
        newer = MemoryItem(
            level=Level.SUMMARY,
            content="y",
            created_at=_utc(2026, 4, 1),
            valid_from=_utc(2026, 4, 1),
        )
        storage.insert_memory_item(older)
        storage.insert_memory_item(newer)
        conflict = Conflict(
            source_item_id=newer.id, target_item_id=older.id, similarity=0.95
        )
        storage.record_conflict(conflict)
        reconciler = Reconciler(
            storage,
            embedder=embedder,
            chat=FakeChat(default=json.dumps({"merged": "z"})),
        )
        with pytest.raises(ValueError, match="provenance"):
            reconciler.reconcile(
                conflict.id,
                resolution=Resolution.MERGE,
                now=_utc(2026, 5, 1),
            )


# ---------------------------------------------------------------------------
# Memory.reconcile delegates MERGE correctly
# ---------------------------------------------------------------------------


class TestMemoryReconcileMerge:
    def test_memory_threads_chat_and_embedder(
        self, storage: SqliteStorage
    ) -> None:
        embedder = FakeEmbedder(dim=8)
        older = _seed_with_provenance(
            storage, embedder, content="a", created_at=_utc(2026, 1, 1)
        )
        newer = _seed_with_provenance(
            storage, embedder, content="b", created_at=_utc(2026, 4, 1)
        )
        conflict = Conflict(
            source_item_id=newer.id, target_item_id=older.id, similarity=0.95
        )
        storage.record_conflict(conflict)
        prompt = render_merge_prompt(a="a", b="b")
        chat = FakeChat(scripts={content_hash(prompt): json.dumps({"merged": "ab"})})
        memory = Memory(storage=storage, embedder=embedder, chat=chat)
        out = memory.reconcile(
            conflict.id,
            resolution=Resolution.MERGE,
            now=_utc(2026, 5, 1),
        )
        assert out.status is ConflictStatus.RESOLVED
        assert out.resolution is Resolution.MERGE
        # Merged item retrievable from the storage with the LLM content.
        older_fresh = storage.get_memory_item(older.id)
        assert older_fresh is not None
        merged_id = older_fresh.invalidated_by
        assert merged_id is not None
        merged = storage.get_memory_item(merged_id)
        assert merged is not None
        assert merged.content == "ab"


# ---------------------------------------------------------------------------
# Conflict schema accepts MERGE without winner
# ---------------------------------------------------------------------------


def test_conflict_schema_allows_merge_without_winner() -> None:
    a, b = new_id(), new_id()
    c = Conflict(
        source_item_id=a,
        target_item_id=b,
        similarity=0.9,
        status=ConflictStatus.RESOLVED,
        resolution=Resolution.MERGE,
        resolved_winner_id=None,
        resolved_at=_utc(2026, 5, 1),
    )
    assert c.resolved_winner_id is None
    assert c.resolution is Resolution.MERGE


def test_merge_orders_a_b_by_recency(storage: SqliteStorage) -> None:
    """Reconciler always passes the older side as `a` and the newer as
    `b` to the merge prompt, regardless of which side of the Conflict
    was source vs target. Pins the uncovered branch."""
    embedder = FakeEmbedder(dim=8)
    # Older = source (reverse of the usual fixture pattern).
    older = _seed_with_provenance(
        storage, embedder, content="old", created_at=_utc(2026, 1, 1)
    )
    newer = _seed_with_provenance(
        storage, embedder, content="new", created_at=_utc(2026, 4, 1)
    )
    # Source is the OLDER side here.
    conflict = Conflict(
        source_item_id=older.id, target_item_id=newer.id, similarity=0.95
    )
    storage.record_conflict(conflict)
    # The prompt must receive (a=old, b=new) -- the script with that
    # ordering must match. If the reconciler instead passed (a=new, b=old),
    # the FakeChat would fall through to default and the test would still
    # pass (with default merged content), so we use the scripted-and-
    # check shape to pin ordering.
    prompt = render_merge_prompt(a="old", b="new")
    chat = FakeChat(
        scripts={content_hash(prompt): json.dumps({"merged": "ordered-correctly"})}
    )
    reconciler = Reconciler(storage, embedder=embedder, chat=chat)
    reconciler.reconcile(
        conflict.id, resolution=Resolution.MERGE, now=_utc(2026, 5, 1)
    )
    older_fresh = storage.get_memory_item(older.id)
    assert older_fresh is not None
    assert older_fresh.invalidated_by is not None
    merged = storage.get_memory_item(older_fresh.invalidated_by)
    assert merged is not None
    assert merged.content == "ordered-correctly"


def test_uuid_round_trip(storage: SqliteStorage) -> None:
    """Smoke: the merged item's id round-trips through storage."""
    embedder = FakeEmbedder(dim=8)
    a = _seed_with_provenance(
        storage, embedder, content="a", created_at=_utc(2026, 1, 1)
    )
    b = _seed_with_provenance(
        storage, embedder, content="b", created_at=_utc(2026, 4, 1)
    )
    conflict = Conflict(source_item_id=b.id, target_item_id=a.id, similarity=0.9)
    storage.record_conflict(conflict)
    prompt = render_merge_prompt(a="a", b="b")
    chat = FakeChat(scripts={content_hash(prompt): json.dumps({"merged": "merged"})})
    reconciler = Reconciler(storage, embedder=embedder, chat=chat)
    reconciler.reconcile(
        conflict.id, resolution=Resolution.MERGE, now=_utc(2026, 5, 1)
    )
    a_fresh = storage.get_memory_item(a.id)
    assert a_fresh is not None
    assert isinstance(a_fresh.invalidated_by, UUID)
