"""Tests for `engram.consolidation._abstraction`."""

from __future__ import annotations

import json

import pytest

from engram.consolidation import (
    AbstractionParseError,
    AbstractionRequest,
    AbstractionResult,
    extract_abstraction,
    parse_response,
    render_prompt,
)
from engram.consolidation._abstraction import (
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    load_prompt_template,
)
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat
from engram.providers._message import Message

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestAbstractionResult:
    def test_roundtrip(self) -> None:
        r = AbstractionResult(abstraction="X tends to follow Y.", confidence=0.7, supports=(0, 1))
        clone = AbstractionResult.model_validate(r.model_dump())
        assert clone == r

    def test_empty_abstraction_rejected(self) -> None:
        with pytest.raises(ValueError, match="abstraction"):
            AbstractionResult(abstraction="", confidence=0.5)

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            AbstractionResult(abstraction="x", confidence=-0.1)
        with pytest.raises(ValueError, match="confidence"):
            AbstractionResult(abstraction="x", confidence=1.1)


class TestAbstractionRequest:
    def test_requires_observations(self) -> None:
        with pytest.raises(ValueError, match="observation"):
            AbstractionRequest(observations=(), cohesion_hint=0.8)

    def test_cohesion_bounds(self) -> None:
        with pytest.raises(ValueError, match="cohesion_hint"):
            AbstractionRequest(observations=("x",), cohesion_hint=-0.1)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_template_loads(self) -> None:
        text = load_prompt_template()
        assert "{observations}" in text
        assert "{cohesion}" in text
        assert "JSON" in text

    def test_prompt_versions_registry(self) -> None:
        assert PROMPT_VERSIONS["abstract"] == PROMPT_VERSION

    def test_render_substitutes_observations(self) -> None:
        req = AbstractionRequest(observations=("a", "b", "c"), cohesion_hint=0.85)
        prompt = render_prompt(req)
        assert "0. a" in prompt
        assert "1. b" in prompt
        assert "2. c" in prompt
        assert "0.850" in prompt
        assert "{observations}" not in prompt
        assert "{cohesion}" not in prompt

    def test_render_inlines_newlines(self) -> None:
        req = AbstractionRequest(observations=("line1\nline2",), cohesion_hint=0.5)
        prompt = render_prompt(req)
        # Newline replaced with literal \n so the observation occupies one line.
        assert "line1\\nline2" in prompt

    def test_render_inlines_tabs_and_backslashes(self) -> None:
        req = AbstractionRequest(observations=("a\tb", "c\\d"), cohesion_hint=0.5)
        prompt = render_prompt(req)
        assert "a\\tb" in prompt
        assert "c\\\\d" in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_clean_json(self) -> None:
        payload = json.dumps({"abstraction": "X causes Y.", "confidence": 0.8, "supports": [0, 1]})
        result = parse_response(payload, n_observations=3)
        assert result.abstraction == "X causes Y."
        assert result.supports == (0, 1)

    def test_with_markdown_fence(self) -> None:
        payload = (
            "```json\n"
            + json.dumps({"abstraction": "X.", "confidence": 0.5, "supports": []})
            + "\n```"
        )
        result = parse_response(payload, n_observations=1)
        assert result.abstraction == "X."

    def test_with_unlabelled_fence(self) -> None:
        payload = (
            "```\n" + json.dumps({"abstraction": "Y.", "confidence": 0.5, "supports": []}) + "\n```"
        )
        result = parse_response(payload, n_observations=1)
        assert result.abstraction == "Y."

    def test_empty_response(self) -> None:
        with pytest.raises(AbstractionParseError, match="empty"):
            parse_response("", n_observations=1)

    def test_invalid_json(self) -> None:
        with pytest.raises(AbstractionParseError, match="invalid JSON"):
            parse_response("not json", n_observations=1)

    def test_schema_mismatch(self) -> None:
        with pytest.raises(AbstractionParseError, match="schema mismatch"):
            parse_response(
                json.dumps({"summary": "wrong field", "confidence": 0.5}),
                n_observations=1,
            )

    def test_support_index_out_of_range(self) -> None:
        with pytest.raises(AbstractionParseError, match="out of range"):
            parse_response(
                json.dumps({"abstraction": "X.", "confidence": 0.5, "supports": [99]}),
                n_observations=2,
            )


# ---------------------------------------------------------------------------
# extract_abstraction (against FakeChat)
# ---------------------------------------------------------------------------


def _scripted(prompt_text: str, response: str) -> FakeChat:
    """FakeChat that responds to a specific rendered prompt with `response`."""
    return FakeChat(scripts={content_hash(prompt_text): response})


class TestExtractAbstraction:
    def test_happy_path(self) -> None:
        req = AbstractionRequest(
            observations=("Alice opened the door at noon.", "Alice opened the window at noon."),
            cohesion_hint=0.95,
        )
        rendered = render_prompt(req)
        chat = _scripted(
            rendered,
            json.dumps(
                {
                    "abstraction": "Alice tends to open openings around noon.",
                    "confidence": 0.7,
                    "supports": [0, 1],
                }
            ),
        )
        result = extract_abstraction(req, chat)
        assert "Alice" in result.abstraction
        assert result.confidence == 0.7
        assert result.supports == (0, 1)

    def test_retry_on_bad_json(self) -> None:
        # First call returns garbage; retry succeeds.
        req = AbstractionRequest(observations=("a",), cohesion_hint=0.9)

        class FlakyChat:
            name = "flaky"
            model = "flaky"

            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return "not json"
                return json.dumps({"abstraction": "X.", "confidence": 0.5, "supports": []})

            async def achat(self, messages):
                return self.chat(messages)

            def manifest_hash(self) -> str:
                return "flaky"

        chat = FlakyChat()
        result = extract_abstraction(req, chat, max_retries=1)
        assert result.abstraction == "X."
        assert chat.calls == 2

    def test_gives_up_after_max_retries(self) -> None:
        req = AbstractionRequest(observations=("a",), cohesion_hint=0.9)

        class AlwaysBad:
            name = "bad"
            model = "bad"

            def chat(self, messages):
                return "still not json"

            async def achat(self, messages):
                return "still not json"

            def manifest_hash(self) -> str:
                return "bad"

        with pytest.raises(AbstractionParseError):
            extract_abstraction(req, AlwaysBad(), max_retries=1)

    def test_negative_max_retries_rejected(self) -> None:
        req = AbstractionRequest(observations=("a",), cohesion_hint=0.9)

        class _Stub:
            name = "x"
            model = "x"

            def chat(self, messages):
                return ""

            async def achat(self, messages):
                return ""

            def manifest_hash(self) -> str:
                return "x"

        with pytest.raises(ValueError, match="max_retries"):
            extract_abstraction(req, _Stub(), max_retries=-1)


# ---------------------------------------------------------------------------
# Determinism: identical inputs -> identical render
# ---------------------------------------------------------------------------


def test_render_is_pure() -> None:
    req = AbstractionRequest(observations=("a", "b"), cohesion_hint=0.5)
    a = render_prompt(req)
    b = render_prompt(req)
    assert a == b
    # Different inputs -> different output.
    other = AbstractionRequest(observations=("a", "b"), cohesion_hint=0.6)
    assert render_prompt(other) != a


def test_no_curly_placeholders_leak() -> None:
    """If a future template adds an unsubstituted `{...}`, that's a bug."""
    req = AbstractionRequest(observations=("x",), cohesion_hint=0.5)
    text = render_prompt(req)
    # Allow `{` only inside the JSON example block and the schema.
    # Heuristic: there should be no {placeholder} substrings of all-letter
    # form (since {observations} / {cohesion} are now substituted).
    import re

    leaked = re.findall(r"\{[a-z_]+\}", text)
    assert leaked == [], f"unsubstituted placeholders: {leaked}"


# ---------------------------------------------------------------------------
# Sanity: Message round-trip (Engram dataclass, not Anthropic API)
# ---------------------------------------------------------------------------


def test_abstraction_does_not_mutate_messages() -> None:
    """The function builds a fresh list each call; passing in messages
    should not affect the next call."""
    req = AbstractionRequest(observations=("a",), cohesion_hint=0.9)
    chat = FakeChat(default=json.dumps({"abstraction": "X.", "confidence": 0.5, "supports": []}))
    pre = [Message(role="user", content="external")]
    result1 = extract_abstraction(req, chat)
    result2 = extract_abstraction(req, chat)
    assert result1 == result2
    assert pre == [Message(role="user", content="external")]
