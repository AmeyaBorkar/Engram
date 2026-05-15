"""Verification-pass tests for EngramAgent."""

from __future__ import annotations

import json

import pytest

from engram import Memory, SqliteStorage
from engram.integrations import EngramAgent
from engram.integrations._verify import (
    VerifyVerdict,
    parse_verify_response,
    render_verify_prompt,
    verify_answer,
)
from engram.providers._fake import FakeChat, FakeEmbedder


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


class TestRender:
    def test_substitution(self) -> None:
        prompt = render_verify_prompt(question="q", context="ctx", answer="a")
        assert "q" in prompt
        assert "ctx" in prompt
        assert "a" in prompt

    def test_empty_context_shows_placeholder(self) -> None:
        prompt = render_verify_prompt(question="q", context="", answer="a")
        assert "(none)" in prompt

    def test_scrub_tags_handles_attribute_and_self_closing_variants(self) -> None:
        # Attribute, self-closing, whitespace-padded tag bypasses that
        # the narrower previous regex `</?(question|context|answer)>`
        # let through.  All three should be neutralized to entities.
        adversarial = (
            'A <context attr="x">leak</context> '
            "B <answer/>more "
            "C < context >wide"
        )
        prompt = render_verify_prompt(question="q", context=adversarial, answer="a")
        # Neither `<context` nor `<answer` should appear as a live tag
        # anywhere in the rendered prompt.  Entity-encoded forms (`&lt;`)
        # are fine.
        for needle in ('<context attr', "<answer/>", "< context >"):
            assert needle not in prompt, f"unscrubbed: {needle!r} in prompt"


class TestParser:
    def test_parses_supported(self) -> None:
        v = parse_verify_response(json.dumps({"supported": True, "reason": "ok"}))
        assert v.supported
        assert v.reason == "ok"

    def test_rejects_missing_supported(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError):
            parse_verify_response(json.dumps({"reason": "x"}))

    def test_rejects_non_json(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError):
            parse_verify_response("not json")


class TestFallback:
    def test_garbage_returns_supported_true(self) -> None:
        chat = FakeChat(default="not parseable")
        v = verify_answer(
            question="q", context="ctx", answer="a", chat=chat, max_retries=1
        )
        assert v.supported  # bias to "stop retrying"


class TestEngramAgentVerify:
    def test_default_off(self, memory: Memory) -> None:
        chat = FakeChat(default="answer")
        agent = EngramAgent(memory, chat)
        turn = agent.chat("q")
        assert turn.reply == "answer"

    def test_verify_pass_no_retry_when_supported(self, memory: Memory) -> None:
        chat = FakeChat(default="answer")
        call_count = 0

        def chat_fn(messages: list) -> str:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            # First call is the agent's chat; second is the verifier.
            if call_count == 1:
                return "the answer"
            return json.dumps({"supported": True, "reason": "ok"})

        chat.chat = chat_fn  # type: ignore[method-assign,assignment]
        agent = EngramAgent(
            memory, chat, verify=True, auto_observe=False
        )
        turn = agent.chat("q")
        assert turn.reply == "the answer"
        # 1 chat + 1 verify; no retry.
        assert call_count == 2

    def test_verify_retries_on_unsupported(self, memory: Memory) -> None:
        chat = FakeChat(default="")
        call_count = 0

        def chat_fn(messages: list) -> str:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            # 1: initial answer.
            # 2: verifier says unsupported.
            # 3: retry chat -> new answer.
            # 4: verifier says supported.
            if call_count == 1:
                return "hallucinated"
            if call_count == 2:
                return json.dumps({"supported": False, "reason": "no ctx"})
            if call_count == 3:
                return "grounded"
            return json.dumps({"supported": True, "reason": "ok"})

        chat.chat = chat_fn  # type: ignore[method-assign,assignment]
        agent = EngramAgent(
            memory, chat, verify=True, verify_max_retries=1, auto_observe=False
        )
        turn = agent.chat("q")
        assert turn.reply == "grounded"
        assert call_count == 4

    def test_verify_caps_at_max_retries(self, memory: Memory) -> None:
        chat = FakeChat(default="")
        call_count = 0

        def chat_fn(messages: list) -> str:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            # Always answer "x"; verifier always says unsupported.
            if call_count % 2 == 1:
                return "x"
            return json.dumps({"supported": False, "reason": "no"})

        chat.chat = chat_fn  # type: ignore[method-assign,assignment]
        agent = EngramAgent(
            memory, chat, verify=True, verify_max_retries=3, auto_observe=False
        )
        turn = agent.chat("q")
        # 1 initial answer + 3 (verify+retry-answer) pairs = 7 calls.
        # The final verify is at call_count=8.
        assert call_count == 8
        assert turn.reply == "x"

    def test_invalid_verify_max_retries(self, memory: Memory) -> None:
        chat = FakeChat(default="")
        with pytest.raises(ValueError, match="verify_max_retries"):
            EngramAgent(memory, chat, verify_max_retries=-1)


class TestVerdictModel:
    def test_default_reason(self) -> None:
        v = VerifyVerdict(supported=True)
        assert v.reason == ""

    def test_frozen(self) -> None:
        v = VerifyVerdict(supported=True)
        with pytest.raises(ValueError, match=r"frozen|immutable"):
            v.supported = False
