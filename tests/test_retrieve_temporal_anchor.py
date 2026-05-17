"""Tier 4 temporal-anchor extraction tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from engram import Memory, SqliteStorage
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.retrieve._temporal import (
    compute_temporal_anchor,
    is_temporal_query,
    parse_temporal_response,
    render_temporal_prompt,
)

# ---------------------------------------------------------------------------
# Heuristic gate
# ---------------------------------------------------------------------------


class TestIsTemporalQuery:
    @pytest.mark.parametrize(
        "text",
        [
            "what was true yesterday?",
            "where was I last week?",
            "two weeks ago I started a project",
            "since March 2024",
            "on 2024-03-15 what happened?",
            "did I work on Friday?",
            "in 3 months",
            "as of June 2026",
            "by next Tuesday",
        ],
    )
    def test_catches_temporal(self, text: str) -> None:
        assert is_temporal_query(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "what is the user's favorite color?",
            "where do they live?",
            "describe their work",
            "",
            "facts unrelated to time",
            # Non-temporal prepositional phrases that the audit
            # called out as false-positives in the old regex
            # (`since|before|after|until|by` matched indiscriminately).
            "I did it by hand",
            "stand by",
            "look before you leap",
            "they walked after the meeting concluded with a hug",
            "you have to fight until victory",
            "since then I have been busy",
        ],
    )
    def test_doesnt_false_positive(self, text: str) -> None:
        assert not is_temporal_query(text), text

    @pytest.mark.parametrize(
        "text",
        [
            # Prepositional cues SHOULD still fire on date-shaped or
            # temporal-noun followers.
            "since March 2024",
            "before 2023",
            "after yesterday",
            "until next week",
            "by next Tuesday",
            "by friday",
            "before the morning",
        ],
    )
    def test_prepositional_temporal_still_caught(self, text: str) -> None:
        assert is_temporal_query(text), text


# ---------------------------------------------------------------------------
# Prompt + parser
# ---------------------------------------------------------------------------


class TestPromptRender:
    def test_substitution(self) -> None:
        now = datetime(2026, 5, 11, 10, 0, tzinfo=timezone.utc)
        prompt = render_temporal_prompt("when did X happen?", now)
        assert "when did X happen?" in prompt
        assert "2026-05-11" in prompt

    def test_critical_rules_precede_question(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        prompt = render_temporal_prompt("q", now)
        assert prompt.index("CRITICAL RULES") < prompt.index("QUESTION")


class TestParser:
    def test_parses_anchor(self) -> None:
        out = parse_temporal_response(
            json.dumps({"anchor": "2024-03-15T00:00:00+00:00", "reasoning": "x"})
        )
        assert out.anchor == "2024-03-15T00:00:00+00:00"
        assert out.reasoning == "x"

    def test_parses_null_anchor(self) -> None:
        out = parse_temporal_response(
            json.dumps({"anchor": None, "reasoning": "non-temporal"})
        )
        assert out.anchor is None

    def test_rejects_non_json(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError):
            parse_temporal_response("not json")


# ---------------------------------------------------------------------------
# compute_temporal_anchor
# ---------------------------------------------------------------------------


class TestComputeAnchor:
    def test_non_temporal_query_skips_chat(self) -> None:
        called = False

        class _Spy:
            name: str = "spy"
            model: str = "spy"

            def chat(self, messages: object) -> str:
                nonlocal called
                called = True
                return ""

            async def achat(self, messages: object) -> str:
                return ""

            def manifest_hash(self) -> str:
                return "spy"

        anchor = compute_temporal_anchor(
            "what does the user prefer",
            _Spy(),  # type: ignore[arg-type,unused-ignore]
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor is None
        assert not called

    def test_temporal_query_calls_chat_and_parses_anchor(self) -> None:
        chat = FakeChat(
            default=json.dumps(
                {"anchor": "2024-03-15T00:00:00+00:00", "reasoning": "..."}
            )
        )
        anchor = compute_temporal_anchor(
            "where was I in March 2024?",
            chat,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor == datetime(2024, 3, 15, tzinfo=timezone.utc)

    def test_null_anchor_returns_none(self) -> None:
        chat = FakeChat(
            default=json.dumps({"anchor": None, "reasoning": "non-temporal"})
        )
        # The heuristic gate must allow the call -- pick a query with a
        # cue but the LLM decides null.
        anchor = compute_temporal_anchor(
            "what happened yesterday in general",
            chat,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor is None

    def test_unparseable_anchor_returns_none(self) -> None:
        chat = FakeChat(
            default=json.dumps({"anchor": "not-a-date", "reasoning": "..."})
        )
        anchor = compute_temporal_anchor(
            "where was I yesterday",
            chat,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor is None

    def test_chat_error_returns_none(self) -> None:
        class _Err:
            name: str = "err"
            model: str = "err"

            def chat(self, messages: object) -> str:
                raise RuntimeError("boom")

            async def achat(self, messages: object) -> str:
                raise RuntimeError("boom")

            def manifest_hash(self) -> str:
                return "err"

        anchor = compute_temporal_anchor(
            "yesterday",
            _Err(),  # type: ignore[arg-type,unused-ignore]
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor is None

    def test_transient_chat_error_retries_until_budget(self) -> None:
        """A chat exception is now retryable: the loop continues to
        the next attempt rather than bailing immediately. A successful
        retry produces an anchor (M-41 fix)."""

        call_count = 0

        class _Flaky:
            name: str = "flaky"
            model: str = "flaky"

            def chat(self, messages: object) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("transient")
                return json.dumps(
                    {"anchor": "2024-03-15T00:00:00+00:00", "reasoning": ""}
                )

            async def achat(self, messages: object) -> str:
                return ""

            def manifest_hash(self) -> str:
                return "flaky"

        anchor = compute_temporal_anchor(
            "where was I in March 2024?",
            _Flaky(),  # type: ignore[arg-type,unused-ignore]
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
            max_retries=1,
        )
        assert call_count == 2
        assert anchor == datetime(2024, 3, 15, tzinfo=timezone.utc)

    def test_persistent_chat_error_exhausts_budget_then_returns_none(self) -> None:
        call_count = 0

        class _AlwaysFails:
            name: str = "fail"
            model: str = "fail"

            def chat(self, messages: object) -> str:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("persistent")

            async def achat(self, messages: object) -> str:
                raise RuntimeError("persistent")

            def manifest_hash(self) -> str:
                return "fail"

        anchor = compute_temporal_anchor(
            "yesterday",
            _AlwaysFails(),  # type: ignore[arg-type,unused-ignore]
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
            max_retries=2,
        )
        assert anchor is None
        # max_retries=2 means up to 3 attempts.
        assert call_count == 3

    def test_non_utc_anchor_normalized_to_utc(self) -> None:
        """An anchor with a non-UTC offset is converted to UTC so
        downstream comparisons (validity windows, as_of) all live in
        a single timezone (M-40 fix)."""
        chat = FakeChat(
            default=json.dumps(
                {"anchor": "2024-03-15T10:00:00+05:30", "reasoning": ""}
            )
        )
        anchor = compute_temporal_anchor(
            "where was I in March 2024?",
            chat,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor is not None
        assert anchor.tzinfo == timezone.utc
        # +05:30 of 10:00 == 04:30 UTC.
        assert anchor == datetime(2024, 3, 15, 4, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Memory.retrieve(temporal=True) wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_with_chat(storage: SqliteStorage) -> Memory:
    return Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=8),
        chat=FakeChat(default=""),
    )


class TestMemoryRetrieveWithTemporal:
    def test_temporal_off_by_default(self, storage: SqliteStorage) -> None:
        called = False

        class _Spy:
            name: str = "spy"
            model: str = "spy"

            def chat(self, messages: object) -> str:
                nonlocal called
                called = True
                return ""

            async def achat(self, messages: object) -> str:
                return ""

            def manifest_hash(self) -> str:
                return "spy"

        memory = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            chat=_Spy(),  # type: ignore[arg-type,unused-ignore]
        )
        memory.observe("anything")
        memory.retrieve("when did X happen", k=1, reinforce=False)
        # Default temporal=False -> no chat call.
        assert not called

    def test_explicit_as_of_overrides_temporal(self, storage: SqliteStorage) -> None:
        """When the caller passes as_of explicitly, the temporal codegen
        path is bypassed (no chat call)."""
        called = False

        class _Spy:
            name: str = "spy"
            model: str = "spy"

            def chat(self, messages: object) -> str:
                nonlocal called
                called = True
                return ""

            async def achat(self, messages: object) -> str:
                return ""

            def manifest_hash(self) -> str:
                return "spy"

        memory = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            chat=_Spy(),  # type: ignore[arg-type,unused-ignore]
        )
        memory.observe("anything")
        memory.retrieve(
            "yesterday",
            k=1,
            reinforce=False,
            temporal=True,
            as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert not called

    def test_temporal_true_runs_codegen_for_temporal_query(
        self, memory_with_chat: Memory
    ) -> None:
        from engram.providers._cache import content_hash

        chat = memory_with_chat._chat
        assert chat is not None
        memory_with_chat.observe("anything")
        # Build the expected prompt for our query + clock.
        now = memory_with_chat._clock()
        prompt = render_temporal_prompt("yesterday", now)
        chat.scripts = {  # type: ignore[attr-defined]
            content_hash(prompt): json.dumps(
                {"anchor": "2026-05-10T00:00:00+00:00", "reasoning": ""}
            )
        }
        # Just verify the path doesn't crash -- the actual `as_of`
        # being applied is tested by the temporal tests in Stage 8.
        memory_with_chat.retrieve(
            "yesterday", k=1, reinforce=False, temporal=True
        )
