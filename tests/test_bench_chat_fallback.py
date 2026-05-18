"""Tests for the content-filter fallback chat wrapper.

JOURNEY §27 documented `06f04340` (a benign LongMemEval dinner-recipe
question) consistently failing under Kimi K2.6 with HTTP 400 +
`content_filter` -- a provider false-positive. The fallback wrapper
catches that specific class of error and reroutes to a secondary chat
provider so the question completes and gets judged on merit.

These tests pin:
  1. The content-filter detection predicate (markers + HTTP-4xx gate)
  2. The wrapper passes through primary success without invoking fallback
  3. The wrapper falls back on content-filter errors
  4. The wrapper re-raises non-filter errors (network / 5xx / timeout)
  5. `manifest_hash()` composes primary + fallback
  6. The CLI `--chat-fallback NAME[:MODEL]` parser
  7. `build_chat` + `build_provider` honor the fallback args
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch


class _FakePrimaryChat:
    """A scriptable primary chat for testing the fallback wrapper.

    `responses` is a list of either str (return) or Exception (raise).
    `acalls` / `calls` track invocation counts to confirm the wrapper
    didn't double-invoke or skip the primary.
    """

    name = "fake-primary"
    model = "primary-model"

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.acalls = 0

    def chat(self, _messages) -> str:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def achat(self, _messages) -> str:
        self.acalls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def manifest_hash(self) -> str:
        return "fake-primary-hash"


class _FakeFallbackChat:
    name = "fake-fallback"
    model = "fallback-model"

    def __init__(self, response: str = "fallback-answer") -> None:
        self.response = response
        self.calls = 0
        self.acalls = 0

    def chat(self, _messages) -> str:
        self.calls += 1
        return self.response

    async def achat(self, _messages) -> str:
        self.acalls += 1
        return self.response

    def manifest_hash(self) -> str:
        return "fake-fallback-hash"


# ---------------------------------------------------------------------------
# Predicate: _is_content_filter_error
# ---------------------------------------------------------------------------


def test_predicate_detects_kimi_content_filter_pattern() -> None:
    """The exact wire format observed for LongMemEval `06f04340`."""
    from engram.bench._real_provider import _is_content_filter_error

    # Verbatim error string from the n=500 v3a manifest's `error` field.
    err = Exception(
        "BadRequestError: Error code: 400 - {'error': {'message': "
        "'Error from provider: Provider returned error', 'code': 400, "
        '\'metadata\': {\'raw\': \'{"error":{"code":400,"message":'
        '"The request was rejected because it was considered high risk",'
        '"type":"content_filter"}}\'}}}'
    )
    assert _is_content_filter_error(err) is True


def test_predicate_detects_azure_responsible_ai() -> None:
    """Azure OpenAI emits ResponsibleAIPolicyViolation; the marker must hit."""
    from engram.bench._real_provider import _is_content_filter_error

    err = Exception("HTTP 400: ResponsibleAIPolicyViolation: prompt flagged")
    assert _is_content_filter_error(err) is True


def test_predicate_rejects_5xx_outage() -> None:
    """A transient 502/503 from any provider must NOT be reclassified
    as content filter -- the right behaviour for an outage is to fail
    the question loudly, not silently reroute to a fallback that the
    user is paying separately for."""
    from engram.bench._real_provider import _is_content_filter_error

    err = Exception("APIError: 503 Service Unavailable - upstream timeout")
    assert _is_content_filter_error(err) is False


def test_predicate_rejects_bare_400_no_marker() -> None:
    """A 400 without a content-filter marker is a malformed request, not
    a filter rejection. Falling back would mask the bug."""
    from engram.bench._real_provider import _is_content_filter_error

    err = Exception("BadRequestError: 400 - {'error': 'invalid model name'}")
    assert _is_content_filter_error(err) is False


def test_predicate_rejects_marker_without_http_4xx() -> None:
    """The marker alone is insufficient -- some unrelated message could
    mention 'content filter' as prose. Require the 4xx signal too."""
    from engram.bench._real_provider import _is_content_filter_error

    err = Exception("logging: skipping content filter step (not configured)")
    assert _is_content_filter_error(err) is False


def test_predicate_rejects_timeout() -> None:
    from engram.bench._real_provider import _is_content_filter_error

    err = Exception("APITimeoutError: request exceeded 180.0s")
    assert _is_content_filter_error(err) is False


# ---------------------------------------------------------------------------
# Wrapper: _ContentFilterFallbackChat
# ---------------------------------------------------------------------------


def test_wrapper_passes_through_primary_success() -> None:
    """On a normal primary success, the wrapper does NOT invoke the
    fallback. The fallback's `calls` counter must stay at zero."""
    from engram.bench._real_provider import _ContentFilterFallbackChat

    primary = _FakePrimaryChat(["primary-answer"])
    fallback = _FakeFallbackChat()
    wrapped = _ContentFilterFallbackChat(primary=primary, fallback=fallback)
    result = wrapped.chat([])
    assert result == "primary-answer"
    assert primary.calls == 1
    assert fallback.calls == 0


def test_wrapper_falls_back_on_content_filter() -> None:
    """When primary raises a content-filter error, wrapper invokes
    fallback and returns its result -- the question gets answered."""
    from engram.bench._real_provider import _ContentFilterFallbackChat

    filter_err = Exception(
        "BadRequestError: 400 - request rejected because it was "
        "considered high risk: content_filter"
    )
    primary = _FakePrimaryChat([filter_err])
    fallback = _FakeFallbackChat("rescued-by-fallback")
    wrapped = _ContentFilterFallbackChat(primary=primary, fallback=fallback)
    result = wrapped.chat([])
    assert result == "rescued-by-fallback"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_wrapper_reraises_non_filter_errors() -> None:
    """Network errors, 5xx, timeouts must propagate -- the fallback is
    only for the narrow content-filter case."""
    from engram.bench._real_provider import _ContentFilterFallbackChat

    network_err = ConnectionError("network unreachable")
    primary = _FakePrimaryChat([network_err])
    fallback = _FakeFallbackChat()
    wrapped = _ContentFilterFallbackChat(primary=primary, fallback=fallback)

    import pytest

    with pytest.raises(ConnectionError, match="network unreachable"):
        wrapped.chat([])
    assert primary.calls == 1
    assert fallback.calls == 0


def test_wrapper_async_path_passes_through() -> None:
    from engram.bench._real_provider import _ContentFilterFallbackChat

    primary = _FakePrimaryChat(["async-primary"])
    fallback = _FakeFallbackChat()
    wrapped = _ContentFilterFallbackChat(primary=primary, fallback=fallback)
    result = asyncio.run(wrapped.achat([]))
    assert result == "async-primary"
    assert primary.acalls == 1
    assert fallback.acalls == 0


def test_wrapper_async_path_falls_back_on_content_filter() -> None:
    from engram.bench._real_provider import _ContentFilterFallbackChat

    err = Exception("BadRequestError 400 high risk content_filter")
    primary = _FakePrimaryChat([err])
    fallback = _FakeFallbackChat("async-rescued")
    wrapped = _ContentFilterFallbackChat(primary=primary, fallback=fallback)
    result = asyncio.run(wrapped.achat([]))
    assert result == "async-rescued"
    assert fallback.acalls == 1


def test_wrapper_manifest_hash_composes_primary_and_fallback() -> None:
    """The wrapper's manifest_hash must distinguish runs that used a
    fallback from runs that didn't, so reproducibility metadata captures
    the actual chat pipeline that produced each answer."""
    from engram.bench._real_provider import _ContentFilterFallbackChat

    primary = _FakePrimaryChat([])
    fallback = _FakeFallbackChat()
    wrapped = _ContentFilterFallbackChat(primary=primary, fallback=fallback)
    h = wrapped.manifest_hash()
    assert "fake-primary-hash" in h
    assert "fake-fallback-hash" in h
    assert "fallback=" in h


def test_wrapper_exposes_primary_model_name() -> None:
    """Wrapper.model should reflect the PRIMARY model (what the run
    'mostly' used) so manifest reporting isn't dominated by the
    fallback. Fallback identity is preserved in manifest_hash."""
    from engram.bench._real_provider import _ContentFilterFallbackChat

    primary = _FakePrimaryChat([])
    fallback = _FakeFallbackChat()
    wrapped = _ContentFilterFallbackChat(primary=primary, fallback=fallback)
    assert wrapped.model == "primary-model"


# ---------------------------------------------------------------------------
# CLI: _parse_chat_fallback
# ---------------------------------------------------------------------------


def test_cli_parse_chat_fallback_none() -> None:
    from engram.bench._cli import _parse_chat_fallback

    assert _parse_chat_fallback(None) == (None, None)
    assert _parse_chat_fallback("") == (None, None)


def test_cli_parse_chat_fallback_name_only() -> None:
    from engram.bench._cli import _parse_chat_fallback

    assert _parse_chat_fallback("openrouter") == ("openrouter", None)


def test_cli_parse_chat_fallback_name_and_model() -> None:
    from engram.bench._cli import _parse_chat_fallback

    assert _parse_chat_fallback("openrouter:openai/gpt-4o-mini") == (
        "openrouter",
        "openai/gpt-4o-mini",
    )


def test_cli_parse_chat_fallback_handles_models_with_colons() -> None:
    """Model names containing further colons (some Anthropic version IDs)
    should round-trip through the parser unchanged.  We split on the
    FIRST colon only."""
    from engram.bench._cli import _parse_chat_fallback

    assert _parse_chat_fallback("anthropic:claude-haiku-4-5:20251001") == (
        "anthropic",
        "claude-haiku-4-5:20251001",
    )


def test_cli_parser_accepts_chat_fallback() -> None:
    from engram.bench._cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "noop",
            "--chat",
            "opencode-go",
            "--embedder",
            "fake",
            "--chat-fallback",
            "openrouter:openai/gpt-4o-mini",
        ]
    )
    assert args.chat_fallback == "openrouter:openai/gpt-4o-mini"


def test_cli_parser_chat_fallback_defaults_to_none() -> None:
    from engram.bench._cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "noop", "--chat", "fake", "--embedder", "fake"])
    assert args.chat_fallback is None


# ---------------------------------------------------------------------------
# build_chat + build_provider wiring
# ---------------------------------------------------------------------------


def test_build_chat_accepts_fallback_kwarg() -> None:
    """`build_chat(name, model, fallback=...)` wraps the result in
    `_ContentFilterFallbackChat` when fallback is provided."""
    from engram.bench._real_provider import (
        _ContentFilterFallbackChat,
        build_chat,
    )

    fallback = _FakeFallbackChat()
    chat = build_chat("fake", model=None, fallback=fallback)  # type: ignore[arg-type]
    assert isinstance(chat, _ContentFilterFallbackChat)


def test_build_chat_no_fallback_returns_bare_chat() -> None:
    """When fallback is omitted, no wrapper -- preserves existing
    behaviour for callers that don't opt in."""
    from engram.bench._real_provider import (
        _ContentFilterFallbackChat,
        build_chat,
    )

    chat = build_chat("fake", model=None)
    assert not isinstance(chat, _ContentFilterFallbackChat)


def _stub_openai_sdk():
    """Mirror tests/test_bench_chat_max_tokens.py — stub the openai SDK
    so chat builders can construct without a real API key."""
    from unittest.mock import patch as _patch

    from engram.providers import openai as _openai_adapter

    fake_client = type("FakeClient", (), {"chat": None, "embeddings": None})
    return _patch.multiple(
        _openai_adapter._openai_module,
        OpenAI=lambda **_kw: fake_client(),
        AsyncOpenAI=lambda **_kw: fake_client(),
    )


def test_build_provider_wires_chat_fallback() -> None:
    """End-to-end: a real provider name + a real fallback name should
    produce a _MixedProvider whose chat is the wrapper."""
    from engram.bench._real_provider import (
        _ContentFilterFallbackChat,
        build_provider,
    )

    with (
        _stub_openai_sdk(),
        patch.dict(
            os.environ,
            {"OPENCODE_API_KEY": "sk-test", "OPENROUTER_API_KEY": "sk-test"},
            clear=False,
        ),
    ):
        provider = build_provider(
            embedder_name="fake",
            chat_name="opencode-go",
            chat_model="kimi-k2.6",
            chat_fallback_name="openrouter",
            chat_fallback_model="openai/gpt-4o-mini",
        )
    assert isinstance(provider.chat, _ContentFilterFallbackChat)
    # Manifest hash must capture both legs
    h = provider.chat.manifest_hash()
    assert "fallback=" in h


def test_build_provider_rejects_unknown_fallback_name() -> None:
    import pytest

    from engram.bench._real_provider import build_provider

    with (
        _stub_openai_sdk(),
        patch.dict(
            os.environ,
            {"OPENCODE_API_KEY": "sk-test"},
            clear=False,
        ),
    ):
        with pytest.raises(ValueError, match="unknown chat fallback"):
            build_provider(
                embedder_name="fake",
                chat_name="opencode-go",
                chat_model="kimi-k2.6",
                chat_fallback_name="nonsense",
            )
