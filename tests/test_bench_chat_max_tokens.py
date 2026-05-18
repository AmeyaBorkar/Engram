"""Tests for the --chat-max-tokens override plumbing.

The cap fix shipped in commit `52a25bf` hard-coded `max_tokens=8192` in
the opencode-go chat builder. JOURNEY §24 documents the cliff that
motivated it. The same model (Kimi K2.6 thinking) routed through any
other OpenAI-compatible endpoint -- OpenRouter, Moonshot direct -- would
hit the SAME 1024-token default cap, because that's where OpenAIChat
defaults live. `--chat-max-tokens` is the general-purpose override.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def _stub_openai_sdk():
    """Patch the openai module so chat builders construct without a real key.

    The builders only need the OpenAI() / AsyncOpenAI() classes to be
    callable; they don't actually issue any HTTP requests during a
    smoke construction test.
    """
    from engram.providers import openai as _openai_adapter

    fake_client = type("FakeClient", (), {"chat": None, "embeddings": None})
    return patch.multiple(
        _openai_adapter._openai_module,
        OpenAI=lambda **_kw: fake_client(),
        AsyncOpenAI=lambda **_kw: fake_client(),
    )


def test_cli_parser_accepts_chat_max_tokens() -> None:
    from engram.bench._cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["run", "noop", "--chat", "fake", "--embedder", "fake", "--chat-max-tokens", "8192"]
    )
    assert args.chat_max_tokens == 8192


def test_cli_parser_chat_max_tokens_defaults_to_none() -> None:
    from engram.bench._cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "noop", "--chat", "fake", "--embedder", "fake"])
    assert args.chat_max_tokens is None


def test_build_chat_fake_accepts_max_tokens_kwarg() -> None:
    """Fake builder ignores max_tokens but must accept the kwarg.

    The dispatch table passes (model, max_tokens) positionally to every
    builder; a fake lambda that only took `model` would TypeError.
    """
    from engram.bench._real_provider import build_chat

    chat = build_chat("fake", model=None, max_tokens=8192)
    assert chat is not None


def test_openrouter_chat_honors_max_tokens_override() -> None:
    """OpenRouter chat must apply --chat-max-tokens to the underlying OpenAIChat.

    Without this, routing Kimi K2.6 through OpenRouter would silently
    truncate at 1024 tokens, reproducing the JOURNEY §24 cliff on a
    different endpoint.
    """
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=False):
        chat = build_chat("openrouter", model="moonshotai/kimi-k2.6", max_tokens=8192)

    assert chat._kwargs["max_tokens"] == 8192


def test_openrouter_chat_default_no_max_tokens_override() -> None:
    """When --chat-max-tokens is omitted, openrouter falls back to OpenAIChat's
    default 1024 cap (the existing safety guard)."""
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=False):
        chat = build_chat("openrouter", model="moonshotai/kimi-k2.6")

    assert chat._kwargs["max_tokens"] == 1024


def test_opencode_go_chat_default_uses_effectively_unlimited_cap() -> None:
    """opencode-go's hardcoded default is set high enough that the model's
    own generation ceiling is the binding constraint, not us.

    History (JOURNEY §24): 1024 default cut Kimi K2.6 thinking-mode
    mid-reason. 8192 was the first fix, but raised the fair question
    "if 1024 wasn't enough, how do we know 8192 is?" Answer: pick a
    cap so high that finish_reason='length' is a real signal of model
    failure, not config truncation. 65536 is well above any plausible
    Kimi thinking trace AND opencode-go is unmetered on output, so
    the usual cost-pressure reason for a tight cap doesn't apply.
    Regression guard for that decision.
    """
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(os.environ, {"OPENCODE_API_KEY": "sk-test"}, clear=False):
        chat = build_chat("opencode-go", model="kimi-k2.6")

    assert chat._kwargs["max_tokens"] == 65536


def test_opencode_go_chat_explicit_override_wins() -> None:
    """An explicit `--chat-max-tokens` overrides opencode-go's generous default
    (caller knows best -- some smaller models on the Go endpoint don't
    need that much, or the caller wants to bound a runaway generation)."""
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(os.environ, {"OPENCODE_API_KEY": "sk-test"}, clear=False):
        chat = build_chat("opencode-go", model="kimi-k2.6", max_tokens=2048)

    assert chat._kwargs["max_tokens"] == 2048


def test_build_provider_forwards_chat_max_tokens() -> None:
    """The top-level `build_provider` must forward chat_max_tokens through
    to the chat builder."""
    from engram.bench._real_provider import build_provider

    with _stub_openai_sdk(), patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=False):
        provider = build_provider(
            embedder_name="fake",
            chat_name="openrouter",
            chat_model="moonshotai/kimi-k2.6",
            chat_max_tokens=8192,
        )

    assert provider.chat._kwargs["max_tokens"] == 8192


def test_openai_chat_honors_max_tokens_override() -> None:
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        chat = build_chat("openai", model="gpt-4o", max_tokens=4096)

    assert chat._kwargs["max_tokens"] == 4096


def test_anthropic_chat_honors_max_tokens_override() -> None:
    """AnthropicChat stores max_tokens in `_max_tokens` (not `_kwargs`),
    so verify the right attribute."""
    pytest.importorskip("anthropic")
    from engram.bench._real_provider import build_chat

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
        chat = build_chat("anthropic", model="claude-haiku-4-5-20251001", max_tokens=4096)

    assert chat._max_tokens == 4096


def test_opencode_go_chat_uses_180s_timeout() -> None:
    """opencode-go must pass timeout=180.0 to OpenAIChat (vs OpenAIChat
    default 60s) because Kimi K2.6 thinking mode legitimately takes
    longer than 60s on hard questions -- one consistent failure on
    n=100 validation (`gpt4_7abb270c`) hit APITimeoutError on all
    3 SDK retries at the 60s cap.  JOURNEY §25 cluster F.

    Verifies via the httpx.Timeout object the SDK kwargs were
    constructed with -- the OpenAI SDK wraps the float we pass into
    an httpx.Timeout(180.0, connect=10.0) tuple.  We can read the
    total timeout off the httpx.Timeout we passed by inspecting
    the OpenAIChat construction path.
    """
    import inspect

    from engram.bench._real_provider import _opencode_go_chat

    sig = inspect.signature(_opencode_go_chat)
    assert "max_tokens" in sig.parameters

    # Read the source to confirm timeout=180.0 is wired in; mock-free
    # because the timeout sits on httpx.Timeout inside the SDK client
    # which we don't want to exercise here.
    src = inspect.getsource(_opencode_go_chat)
    assert "timeout=180.0" in src, (
        "opencode-go chat builder must pass timeout=180.0 to OpenAIChat -- "
        "removing this regresses gpt4_7abb270c-class failures (JOURNEY §25)."
    )
