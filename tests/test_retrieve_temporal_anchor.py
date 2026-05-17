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
        ],
    )
    def test_doesnt_false_positive(self, text: str) -> None:
        assert not is_temporal_query(text), text


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
        out = parse_temporal_response(json.dumps({"anchor": None, "reasoning": "non-temporal"}))
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
            default=json.dumps({"anchor": "2024-03-15T00:00:00+00:00", "reasoning": "..."})
        )
        anchor = compute_temporal_anchor(
            "where was I in March 2024?",
            chat,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor == datetime(2024, 3, 15, tzinfo=timezone.utc)

    def test_null_anchor_returns_none(self) -> None:
        chat = FakeChat(default=json.dumps({"anchor": None, "reasoning": "non-temporal"}))
        # The heuristic gate must allow the call -- pick a query with a
        # cue but the LLM decides null.
        anchor = compute_temporal_anchor(
            "what happened yesterday in general",
            chat,
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
        )
        assert anchor is None

    def test_unparseable_anchor_returns_none(self) -> None:
        chat = FakeChat(default=json.dumps({"anchor": "not-a-date", "reasoning": "..."}))
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

    def test_temporal_true_runs_codegen_for_temporal_query(self, memory_with_chat: Memory) -> None:
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
        memory_with_chat.retrieve("yesterday", k=1, reinforce=False, temporal=True)


# ---------------------------------------------------------------------------
# Audit regression tests
# ---------------------------------------------------------------------------


class TestTemporalRegexTightening:
    """Audit M-39: bare prepositions used to over-trigger the temporal
    codegen call.  After the tightening, a date-shaped follower is
    required for `since/before/after/until/by` to count as temporal.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "fixed it by hand",
            "before lunch we talked",
            "after the meeting",
            "until further notice",
            "since launch we have grown",
        ],
    )
    def test_bare_preposition_no_longer_temporal(self, text: str) -> None:
        assert not is_temporal_query(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "since 2023 the market shifted",
            "since yesterday",
            "by next Tuesday",
            "before January",
            "until last Monday",
            "after 5pm",
            "before the 15th",
        ],
    )
    def test_preposition_with_date_token_still_temporal(self, text: str) -> None:
        assert is_temporal_query(text), text


class TestTemporalRetryExhaustion:
    """Audit M-41: transient chat errors used to short-circuit the
    retry loop on first failure.  Now they consume the retry budget
    and only the final attempt's failure returns None."""

    def test_transient_error_then_success_returns_anchor(self) -> None:
        attempt = {"n": 0}

        class _FlakyChat:
            name: str = "flaky"
            model: str = "flaky"

            def chat(self, messages: object) -> str:
                attempt["n"] += 1
                if attempt["n"] < 2:
                    raise RuntimeError("transient")
                return json.dumps({"anchor": "2024-03-15T00:00:00+00:00", "reasoning": "ok"})

            async def achat(self, messages: object) -> str:
                return ""

            def manifest_hash(self) -> str:
                return "flaky"

        anchor = compute_temporal_anchor(
            "yesterday",
            _FlakyChat(),  # type: ignore[arg-type,unused-ignore]
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
            max_retries=2,
        )
        assert anchor == datetime(2024, 3, 15, tzinfo=timezone.utc)
        # Two attempts: first raised, second succeeded.
        assert attempt["n"] == 2

    def test_all_attempts_fail_returns_none(self) -> None:
        attempt = {"n": 0}

        class _AlwaysErr:
            name: str = "err"
            model: str = "err"

            def chat(self, messages: object) -> str:
                attempt["n"] += 1
                raise RuntimeError("never")

            async def achat(self, messages: object) -> str:
                raise RuntimeError("never")

            def manifest_hash(self) -> str:
                return "err"

        anchor = compute_temporal_anchor(
            "yesterday",
            _AlwaysErr(),  # type: ignore[arg-type,unused-ignore]
            now=datetime(2026, 5, 11, tzinfo=timezone.utc),
            max_retries=2,
        )
        assert anchor is None
        # All three attempts were used (max_retries=2 means 3 total).
        assert attempt["n"] == 3


class TestTemporalNaiveDatetimeWarn:
    """Audit M-40: naive datetimes coerce to UTC but emit a WARNING
    so an operator can spot the prompt drift."""

    def test_naive_datetime_warns_and_coerces(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging as _logging

        chat = FakeChat(default=json.dumps({"anchor": "2024-03-15T12:00:00", "reasoning": "naive"}))
        with caplog.at_level(_logging.WARNING, logger="engram.retrieve"):
            anchor = compute_temporal_anchor(
                "yesterday",
                chat,
                now=datetime(2026, 5, 11, tzinfo=timezone.utc),
            )
        assert anchor == datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
        assert any("naive datetime" in r.getMessage() for r in caplog.records)

    def test_tz_aware_datetime_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging as _logging

        chat = FakeChat(
            default=json.dumps({"anchor": "2024-03-15T12:00:00+00:00", "reasoning": "ok"})
        )
        with caplog.at_level(_logging.WARNING, logger="engram.retrieve"):
            anchor = compute_temporal_anchor(
                "yesterday",
                chat,
                now=datetime(2026, 5, 11, tzinfo=timezone.utc),
            )
        assert anchor == datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
        assert not any("naive datetime" in r.getMessage() for r in caplog.records)
