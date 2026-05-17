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

    with _stub_openai_sdk(), patch.dict(
        os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=False
    ):
        chat = build_chat("openrouter", model="moonshotai/kimi-k2.6", max_tokens=8192)

    assert chat._kwargs["max_tokens"] == 8192


def test_openrouter_chat_default_no_max_tokens_override() -> None:
    """When --chat-max-tokens is omitted, openrouter falls back to OpenAIChat's
    default 1024 cap (the existing safety guard)."""
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(
        os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=False
    ):
        chat = build_chat("openrouter", model="moonshotai/kimi-k2.6")

    assert chat._kwargs["max_tokens"] == 1024


def test_opencode_go_chat_default_uses_8192() -> None:
    """opencode-go's hardcoded 8192 default survives when --chat-max-tokens is unset.

    Regression guard for the commit `52a25bf` fix: anyone running
    `--chat opencode-go --chat-model kimi-k2.6` without specifying
    a cap should still get 8192 (not the 1024 default that bit us).
    """
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(
        os.environ, {"OPENCODE_API_KEY": "sk-test"}, clear=False
    ):
        chat = build_chat("opencode-go", model="kimi-k2.6")

    assert chat._kwargs["max_tokens"] == 8192


def test_opencode_go_chat_explicit_override_wins() -> None:
    """An explicit `--chat-max-tokens` overrides opencode-go's 8192 default
    (caller knows best -- some smaller models on the Go endpoint don't
    need 8K)."""
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(
        os.environ, {"OPENCODE_API_KEY": "sk-test"}, clear=False
    ):
        chat = build_chat("opencode-go", model="kimi-k2.6", max_tokens=2048)

    assert chat._kwargs["max_tokens"] == 2048


def test_build_provider_forwards_chat_max_tokens() -> None:
    """The top-level `build_provider` must forward chat_max_tokens through
    to the chat builder."""
    from engram.bench._real_provider import build_provider

    with _stub_openai_sdk(), patch.dict(
        os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=False
    ):
        provider = build_provider(
            embedder_name="fake",
            chat_name="openrouter",
            chat_model="moonshotai/kimi-k2.6",
            chat_max_tokens=8192,
        )

    assert provider.chat._kwargs["max_tokens"] == 8192


def test_openai_chat_honors_max_tokens_override() -> None:
    from engram.bench._real_provider import build_chat

    with _stub_openai_sdk(), patch.dict(
        os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
    ):
        chat = build_chat("openai", model="gpt-4o", max_tokens=4096)

    assert chat._kwargs["max_tokens"] == 4096


def test_anthropic_chat_honors_max_tokens_override() -> None:
    """AnthropicChat stores max_tokens in `_max_tokens` (not `_kwargs`),
    so verify the right attribute."""
    pytest.importorskip("anthropic")
    from engram.bench._real_provider import build_chat

    with patch.dict(
        os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False
    ):
        chat = build_chat("anthropic", model="claude-haiku-4-5-20251001", max_tokens=4096)

    assert chat._max_tokens == 4096
