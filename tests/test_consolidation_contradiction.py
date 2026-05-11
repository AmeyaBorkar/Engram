"""Tests for the contradiction detection layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram import Memory, SqliteStorage
from engram.consolidation import (
    CandidateRow,
    ContradictionParams,
    DetectedConflict,
    JudgeResponse,
    Verdict,
    conflicts_to_metadata,
    detect_contradictions,
    judge,
    parse_judge_response,
    render_judge_prompt,
)
from engram.consolidation._contradiction import load_judge_prompt
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.schemas import ItemKind, Level, MemoryItem, new_id

# ---------------------------------------------------------------------------
# Schema + parsing
# ---------------------------------------------------------------------------


class TestJudgeResponse:
    def test_verdict_enum(self) -> None:
        r = JudgeResponse(verdict=Verdict.AGREE)
        assert r.verdict is Verdict.AGREE

    def test_invalid_verdict(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JudgeResponse.model_validate({"verdict": "maybe"})


class TestParseJudgeResponse:
    def test_clean(self) -> None:
        text = json.dumps({"verdict": "contradict"})
        assert parse_judge_response(text).verdict is Verdict.CONTRADICT

    def test_with_fence(self) -> None:
        text = "```json\n" + json.dumps({"verdict": "agree"}) + "\n```"
        assert parse_judge_response(text).verdict is Verdict.AGREE

    def test_empty(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError, match="empty"):
            parse_judge_response("")

    def test_invalid(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError, match="invalid judge JSON"):
            parse_judge_response("{not json}")

    def test_schema_mismatch(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError, match="schema mismatch"):
            parse_judge_response(json.dumps({"answer": "yes"}))


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_template_loads(self) -> None:
        text = load_judge_prompt()
        assert "{a}" in text
        assert "{b}" in text
        assert "JSON" in text

    def test_render_substitutes(self) -> None:
        text = render_judge_prompt(a="It rains.", b="It does not rain.")
        assert "It rains." in text
        assert "It does not rain." in text
        assert "{a}" not in text
        assert "{b}" not in text

    def test_render_inlines_newlines(self) -> None:
        text = render_judge_prompt(a="line1\nline2", b="x")
        assert "line1\\nline2" in text


# ---------------------------------------------------------------------------
# judge() against FakeChat
# ---------------------------------------------------------------------------


class TestJudge:
    def _scripted(self, *, a: str, b: str, response: str) -> FakeChat:
        prompt = render_judge_prompt(a=a, b=b)
        return FakeChat(scripts={content_hash(prompt): response})

    def test_returns_contradict(self) -> None:
        chat = self._scripted(a="A", b="B", response=json.dumps({"verdict": "contradict"}))
        assert judge(a="A", b="B", chat=chat) is Verdict.CONTRADICT

    def test_returns_agree(self) -> None:
        chat = self._scripted(a="A", b="B", response=json.dumps({"verdict": "agree"}))
        assert judge(a="A", b="B", chat=chat) is Verdict.AGREE

    def test_returns_unrelated_on_garbage(self) -> None:
        chat = FakeChat(default="not json")
        assert judge(a="A", b="B", chat=chat, max_retries=0) is Verdict.UNRELATED

    def test_retries_on_garbage(self) -> None:
        class FlakyChat:
            name = "x"
            model = "x"

            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return "garbage"
                return json.dumps({"verdict": "agree"})

            async def achat(self, messages):
                return self.chat(messages)

            def manifest_hash(self) -> str:
                return "x"

        chat = FlakyChat()
        verdict = judge(a="A", b="B", chat=chat, max_retries=1)
        assert verdict is Verdict.AGREE
        assert chat.calls == 2


# ---------------------------------------------------------------------------
# detect_contradictions
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    def test_disabled_returns_empty(self) -> None:
        chat = FakeChat(default=json.dumps({"verdict": "contradict"}))
        candidates = [CandidateRow(item_id=new_id(), content="other", similarity=0.95)]
        out = detect_contradictions(
            new_abstraction="X",
            candidates=candidates,
            chat=chat,
            params=ContradictionParams(enabled=False),
        )
        assert out == []

    def test_returns_only_contradictions(self) -> None:
        from engram.consolidation import render_prompt as _unused

        c1 = CandidateRow(item_id=new_id(), content="agreer", similarity=0.91)
        c2 = CandidateRow(item_id=new_id(), content="conflicter", similarity=0.88)
        c3 = CandidateRow(item_id=new_id(), content="unrelated", similarity=0.85)

        scripts = {
            content_hash(render_judge_prompt(a="X", b="agreer")): json.dumps({"verdict": "agree"}),
            content_hash(render_judge_prompt(a="X", b="conflicter")): json.dumps(
                {"verdict": "contradict"}
            ),
            content_hash(render_judge_prompt(a="X", b="unrelated")): json.dumps(
                {"verdict": "unrelated"}
            ),
        }
        chat = FakeChat(scripts=scripts)
        out = detect_contradictions(
            new_abstraction="X",
            candidates=[c1, c2, c3],
            chat=chat,
            params=ContradictionParams(enabled=True, max_candidates=10),
        )
        assert len(out) == 1
        assert out[0].candidate_id == c2.item_id
        assert out[0].verdict is Verdict.CONTRADICT
        _ = _unused

    def test_caps_at_max_candidates(self) -> None:
        candidates = [
            CandidateRow(item_id=new_id(), content=f"c{i}", similarity=0.9 - 0.01 * i)
            for i in range(10)
        ]
        chat = FakeChat(default=json.dumps({"verdict": "contradict"}))
        out = detect_contradictions(
            new_abstraction="X",
            candidates=candidates,
            chat=chat,
            params=ContradictionParams(enabled=True, max_candidates=2),
        )
        assert len(out) == 2


# ---------------------------------------------------------------------------
# ContradictionParams validation
# ---------------------------------------------------------------------------


class TestContradictionParams:
    def test_similarity_bounds(self) -> None:
        with pytest.raises(ValueError, match="similarity_threshold"):
            ContradictionParams(similarity_threshold=-0.1)
        with pytest.raises(ValueError, match="similarity_threshold"):
            ContradictionParams(similarity_threshold=1.1)

    def test_max_candidates_non_negative(self) -> None:
        with pytest.raises(ValueError, match="max_candidates"):
            ContradictionParams(max_candidates=-1)

    def test_max_retries_non_negative(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            ContradictionParams(max_retries=-1)


# ---------------------------------------------------------------------------
# conflicts_to_metadata
# ---------------------------------------------------------------------------


class TestConflictsToMetadata:
    def test_serializable_shape(self) -> None:
        c = DetectedConflict(candidate_id=new_id(), similarity=0.91, verdict=Verdict.CONTRADICT)
        out = conflicts_to_metadata([c])
        assert isinstance(out, list)
        assert isinstance(out[0]["candidate_id"], str)
        assert out[0]["verdict"] == "contradict"
        assert out[0]["similarity"] == 0.91
        # Round-trips through JSON.
        json.dumps(out)


# ---------------------------------------------------------------------------
# Engine integration: contradiction is recorded on metadata
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_disabled_default_records_nothing(self, tmp_path: Path) -> None:
        # By default contradiction detection is off.
        from engram.consolidation import ClusterParams, ConsolidationParams
        from engram.schemas import Embedding, Event

        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        # Plant an existing similar abstraction.
        existing_vec = (1.0,) + (0.0,) * 7
        existing = MemoryItem(level=Level.SUMMARY, content="Alice prefers tea")
        storage.insert_memory_item(existing)
        storage.insert_embedding(
            Embedding(
                item_id=existing.id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=embedder.model,
                dim=8,
                vector=existing_vec,
            )
        )

        # Two events that will cluster and produce a new abstraction.
        events = []
        from datetime import datetime, timedelta, timezone

        for i in range(2):
            ev = Event(
                content=f"e-{i}",
                created_at=datetime.now(tz=timezone.utc) + timedelta(seconds=i),
            )
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=existing_vec,
                )
            )
            events.append(ev)

        from engram.consolidation import AbstractionRequest, render_prompt

        req = AbstractionRequest(observations=tuple(e.content for e in events), cohesion_hint=1.0)
        chat = FakeChat(
            scripts={
                content_hash(render_prompt(req)): json.dumps(
                    {"abstraction": "Alice prefers coffee", "confidence": 0.8, "supports": []}
                )
            }
        )

        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2)
            ),
        )
        result = memory.consolidate()
        assert result.conflicts_detected == 0
        items = storage.list_memory_items(level=Level.SUMMARY, limit=10)
        new_item = next(i for i in items if i.content == "Alice prefers coffee")
        assert new_item.metadata["consolidation"]["conflicts"] == []
        storage.close()

    def test_enabled_records_contradiction(self, tmp_path: Path) -> None:
        from engram.consolidation import (
            AbstractionRequest,
            ClusterParams,
            ConsolidationParams,
            render_prompt,
        )
        from engram.schemas import Embedding, Event

        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        # The new abstraction text the LLM will emit.
        new_abstraction = "Alice prefers coffee"
        # Embed both `new_abstraction` and the existing candidate via the same
        # FakeEmbedder, then plant the existing one with the SAME vector as
        # `new_abstraction` so the cosine similarity is 1.0 and the candidate
        # clears the vector-recall threshold.
        new_vec = tuple(embedder.embed([new_abstraction])[0])
        existing = MemoryItem(level=Level.SUMMARY, content="Alice prefers tea")
        storage.insert_memory_item(existing)
        storage.insert_embedding(
            Embedding(
                item_id=existing.id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=embedder.model,
                dim=8,
                vector=new_vec,
            )
        )

        events = []
        from datetime import datetime, timedelta, timezone

        # The events themselves can have any embedding - they only matter for
        # clustering. Pick a fixed unit vector so they cluster together.
        cluster_vec = (1.0,) + (0.0,) * 7
        for i in range(2):
            ev = Event(
                content=f"e-{i}",
                created_at=datetime.now(tz=timezone.utc) + timedelta(seconds=i),
            )
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=8,
                    vector=cluster_vec,
                )
            )
            events.append(ev)

        req = AbstractionRequest(observations=tuple(e.content for e in events), cohesion_hint=1.0)
        scripts = {
            content_hash(render_prompt(req)): json.dumps(
                {"abstraction": new_abstraction, "confidence": 0.8, "supports": []}
            ),
            content_hash(render_judge_prompt(a=new_abstraction, b=existing.content)): json.dumps(
                {"verdict": "contradict"}
            ),
        }
        chat = FakeChat(scripts=scripts)

        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
            consolidation_params=ConsolidationParams(
                cluster_params=ClusterParams(method="agglomerative", min_cluster_size=2),
                contradiction_params=ContradictionParams(enabled=True, similarity_threshold=0.9),
            ),
        )
        result = memory.consolidate()
        assert result.conflicts_detected == 1
        items = storage.list_memory_items(level=Level.SUMMARY, limit=10)
        new_item = next(i for i in items if i.content == new_abstraction)
        conflicts = new_item.metadata["consolidation"]["conflicts"]
        assert len(conflicts) == 1
        assert conflicts[0]["candidate_id"] == str(existing.id)
        assert conflicts[0]["verdict"] == "contradict"
        # Stage 8: the conflict is also a first-class storage row.
        from engram import ConflictStatus

        rows = storage.list_conflicts(memory_item_id=new_item.id)
        assert len(rows) == 1
        row = rows[0]
        assert row.source_item_id == new_item.id
        assert row.target_item_id == existing.id
        assert row.status is ConflictStatus.OPEN
        assert row.verdict.value == "contradict"
        # Walking the graph from the existing item finds the same conflict.
        rows_from_existing = storage.list_conflicts(memory_item_id=existing.id)
        assert len(rows_from_existing) == 1
        assert rows_from_existing[0].id == row.id
        storage.close()


# ---------------------------------------------------------------------------
# search_memory_item_embeddings (the storage seam)
# ---------------------------------------------------------------------------


class TestSearchMemoryItemEmbeddings:
    def test_filters_by_level(self, tmp_path: Path) -> None:
        from engram.schemas import Embedding

        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        same = (1.0,) + (0.0,) * 7
        for level in (Level.SUMMARY, Level.ABSTRACTION):
            item = MemoryItem(level=level, content=f"item at {level.value}")
            storage.insert_memory_item(item)
            storage.insert_embedding(
                Embedding(
                    item_id=item.id,
                    item_kind=ItemKind.MEMORY_ITEM,
                    model=embedder.model,
                    dim=8,
                    vector=same,
                )
            )

        all_levels = storage.search_memory_item_embeddings(same, k=10, model=embedder.model)
        assert len(all_levels) == 2

        only_summary = storage.search_memory_item_embeddings(
            same, k=10, model=embedder.model, levels=[Level.SUMMARY]
        )
        assert len(only_summary) == 1
        assert "summary" in only_summary[0][1]

        storage.close()

    def test_excludes_ids(self, tmp_path: Path) -> None:
        from engram.schemas import Embedding

        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        same = (1.0,) + (0.0,) * 7
        a = MemoryItem(level=Level.SUMMARY, content="a")
        b = MemoryItem(level=Level.SUMMARY, content="b")
        for item in (a, b):
            storage.insert_memory_item(item)
            storage.insert_embedding(
                Embedding(
                    item_id=item.id,
                    item_kind=ItemKind.MEMORY_ITEM,
                    model=embedder.model,
                    dim=8,
                    vector=same,
                )
            )
        results = storage.search_memory_item_embeddings(
            same, k=10, model=embedder.model, exclude_ids=[a.id]
        )
        ids = {r[0] for r in results}
        assert ids == {b.id}
        storage.close()

    def test_excludes_cold_by_default(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        from engram.schemas import Embedding

        storage = SqliteStorage(tmp_path / "x.db")
        storage.initialize()
        embedder = FakeEmbedder(dim=8)

        same = (1.0,) + (0.0,) * 7
        cold = MemoryItem(level=Level.SUMMARY, content="cold")
        hot = MemoryItem(level=Level.SUMMARY, content="hot")
        for item in (cold, hot):
            storage.insert_memory_item(item)
            storage.insert_embedding(
                Embedding(
                    item_id=item.id,
                    item_kind=ItemKind.MEMORY_ITEM,
                    model=embedder.model,
                    dim=8,
                    vector=same,
                )
            )
        storage.mark_cold(cold.id, ItemKind.MEMORY_ITEM, at=datetime.now(tz=timezone.utc))
        results = storage.search_memory_item_embeddings(same, k=10, model=embedder.model)
        ids = {r[0] for r in results}
        assert ids == {hot.id}
        storage.close()
