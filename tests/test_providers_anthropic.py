"""Tests for the Anthropic adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.providers import ChatProvider, Message
from engram.providers.anthropic import AnthropicChat


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_block() -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name="some_tool", input={})


def _make_response(*blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks))


def test_anthropic_chat_satisfies_protocol() -> None:
    c = AnthropicChat(client=MagicMock(), async_client=AsyncMock())
    assert isinstance(c, ChatProvider)


def test_anthropic_chat_concatenates_text_blocks() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(
        _text_block("hello "), _text_block("world")
    )
    c = AnthropicChat(client=client, async_client=AsyncMock())
    assert c.chat([Message(role="user", content="hi")]) == "hello world"


def test_anthropic_chat_drops_non_text_blocks() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(
        _text_block("answer:"), _tool_block(), _text_block(" 42")
    )
    c = AnthropicChat(client=client, async_client=AsyncMock())
    assert c.chat([Message(role="user", content="x")]) == "answer: 42"


def test_anthropic_chat_separates_system_message() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(client=client, async_client=AsyncMock())
    msgs = [
        Message(role="system", content="be terse"),
        Message(role="user", content="hi"),
    ]
    c.chat(msgs)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "be terse"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_chat_concatenates_multiple_system_messages() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(client=client, async_client=AsyncMock())
    msgs = [
        Message(role="system", content="first system"),
        Message(role="system", content="second system"),
        Message(role="user", content="x"),
    ]
    c.chat(msgs)
    assert client.messages.create.call_args.kwargs["system"] == "first system\n\nsecond system"


def test_anthropic_chat_omits_system_when_absent() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(client=client, async_client=AsyncMock())
    c.chat([Message(role="user", content="x")])
    assert "system" not in client.messages.create.call_args.kwargs


def test_anthropic_chat_passes_max_tokens() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(client=client, async_client=AsyncMock(), max_tokens=512)
    c.chat([Message(role="user", content="x")])
    assert client.messages.create.call_args.kwargs["max_tokens"] == 512


def test_anthropic_chat_passes_completion_kwargs() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(
        client=client,
        async_client=AsyncMock(),
        completion_kwargs={"temperature": 0.0, "top_p": 0.95},
    )
    c.chat([Message(role="user", content="x")])
    kw = client.messages.create.call_args.kwargs
    assert kw["temperature"] == 0.0
    assert kw["top_p"] == 0.95


def test_anthropic_chat_async() -> None:
    aclient = AsyncMock()
    aclient.messages.create.return_value = _make_response(_text_block("async-ok"))
    c = AnthropicChat(client=MagicMock(), async_client=aclient)
    assert asyncio.run(c.achat([Message(role="user", content="x")])) == "async-ok"


def test_anthropic_chat_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicChat(max_tokens=0, client=MagicMock(), async_client=AsyncMock())


def test_anthropic_chat_manifest_hash_pins_model_max_tokens_kwargs() -> None:
    base_kwargs = {"client": MagicMock(), "async_client": AsyncMock()}
    a = AnthropicChat(model="claude-haiku-4-5-20251001", max_tokens=1024, **base_kwargs)
    b = AnthropicChat(model="claude-haiku-4-5-20251001", max_tokens=512, **base_kwargs)
    c = AnthropicChat(model="claude-sonnet-4-6", max_tokens=1024, **base_kwargs)
    d = AnthropicChat(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        completion_kwargs={"temperature": 1.0},
        **base_kwargs,
    )
    assert a.manifest_hash() != b.manifest_hash()
    assert a.manifest_hash() != c.manifest_hash()
    assert a.manifest_hash() != d.manifest_hash()


# --- per-call kwargs ------------------------------------------------------


def test_anthropic_chat_per_call_kwargs_override_constructor() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(
        client=client,
        async_client=AsyncMock(),
        completion_kwargs={"temperature": 0.0, "top_p": 0.9},
    )
    c.chat([Message(role="user", content="x")], extra={"temperature": 0.7})
    kw = client.messages.create.call_args.kwargs
    assert kw["temperature"] == 0.7  # overridden
    assert kw["top_p"] == 0.9  # passed through


def test_anthropic_chat_per_call_kwargs_async() -> None:
    aclient = AsyncMock()
    aclient.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(client=MagicMock(), async_client=aclient)
    asyncio.run(
        c.achat([Message(role="user", content="x")], extra={"temperature": 0.5})
    )
    kw = aclient.messages.create.call_args.kwargs
    assert kw["temperature"] == 0.5


def test_anthropic_chat_per_call_kwargs_does_not_mutate_constructor_state() -> None:
    client = MagicMock()
    client.messages.create.return_value = _make_response(_text_block("ok"))
    c = AnthropicChat(
        client=client,
        async_client=AsyncMock(),
        completion_kwargs={"temperature": 0.0},
    )
    c.chat([Message(role="user", content="x")], extra={"temperature": 1.0})
    c.chat([Message(role="user", content="x")])
    last_kw = client.messages.create.call_args.kwargs
    assert last_kw["temperature"] == 0.0


# --- close / context manager ---------------------------------------------


def test_anthropic_chat_close_idempotent() -> None:
    client = MagicMock()
    c = AnthropicChat(client=client, async_client=AsyncMock())
    c.close()
    c.close()


def test_anthropic_chat_context_manager_closes() -> None:
    client = MagicMock()
    aclient = AsyncMock()
    with AnthropicChat(client=client, async_client=aclient) as c:
        assert isinstance(c, AnthropicChat)
    assert client.close.called
