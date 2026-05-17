"""Tests for the OpenAI adapter.

Uses `unittest.mock` to stand in for the real OpenAI client so tests do
not call out to the network and do not require an API key. The adapter
itself is the unit under test; the SDK's behavior is its own problem.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.providers import ChatProvider, EmbeddingProvider, Message
from engram.providers.openai import OpenAIChat, OpenAIEmbedder


def _make_embed_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


def _make_chat_response(
    content: str | None, *, finish_reason: str | None = None
) -> SimpleNamespace:
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


# --- OpenAIEmbedder -------------------------------------------------------


def test_openai_embedder_satisfies_protocol() -> None:
    client = MagicMock()
    aclient = AsyncMock()
    e = OpenAIEmbedder(client=client, async_client=aclient)
    assert isinstance(e, EmbeddingProvider)


def test_openai_embedder_calls_client_with_expected_args() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = _make_embed_response([[0.1, 0.2]])
    e = OpenAIEmbedder(
        model="text-embedding-3-small", dim=1536, client=client, async_client=AsyncMock()
    )
    out = e.embed(["hello"])
    assert out == [[0.1, 0.2]]
    client.embeddings.create.assert_called_once()
    args = client.embeddings.create.call_args
    assert args.kwargs["model"] == "text-embedding-3-small"
    assert args.kwargs["input"] == ["hello"]
    # native dim → no dimensions arg
    assert "dimensions" not in args.kwargs


def test_openai_embedder_passes_dimensions_when_non_native() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = _make_embed_response([[0.1] * 512])
    e = OpenAIEmbedder(
        model="text-embedding-3-small", dim=512, client=client, async_client=AsyncMock()
    )
    e.embed(["hello"])
    assert client.embeddings.create.call_args.kwargs["dimensions"] == 512


def test_openai_embedder_async() -> None:
    aclient = AsyncMock()
    aclient.embeddings.create.return_value = _make_embed_response([[1.0, 2.0]])
    e = OpenAIEmbedder(client=MagicMock(), async_client=aclient)
    out = asyncio.run(e.aembed(["hi"]))
    assert out == [[1.0, 2.0]]


def test_openai_embedder_rejects_invalid_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        OpenAIEmbedder(dim=0, client=MagicMock(), async_client=AsyncMock())


def test_openai_embedder_manifest_hash_pins_model_and_dim() -> None:
    e1 = OpenAIEmbedder(
        model="text-embedding-3-small", dim=1536, client=MagicMock(), async_client=AsyncMock()
    )
    e2 = OpenAIEmbedder(
        model="text-embedding-3-small", dim=512, client=MagicMock(), async_client=AsyncMock()
    )
    e3 = OpenAIEmbedder(
        model="text-embedding-3-large", dim=1536, client=MagicMock(), async_client=AsyncMock()
    )
    assert e1.manifest_hash() != e2.manifest_hash()
    assert e1.manifest_hash() != e3.manifest_hash()


# --- OpenAIChat -----------------------------------------------------------


def test_openai_chat_satisfies_protocol() -> None:
    c = OpenAIChat(client=MagicMock(), async_client=AsyncMock())
    assert isinstance(c, ChatProvider)


def test_openai_chat_calls_client_with_messages_in_role_form() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response("hello back")
    c = OpenAIChat(model="gpt-4o-mini", client=client, async_client=AsyncMock())
    msgs = [
        Message(role="system", content="be brief"),
        Message(role="user", content="hi"),
    ]
    assert c.chat(msgs) == "hello back"
    args = client.chat.completions.create.call_args
    assert args.kwargs["model"] == "gpt-4o-mini"
    assert args.kwargs["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_openai_chat_passes_completion_kwargs() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response("ok")
    c = OpenAIChat(
        client=client,
        async_client=AsyncMock(),
        completion_kwargs={"temperature": 0.0, "max_tokens": 100},
    )
    c.chat([Message(role="user", content="x")])
    kw = client.chat.completions.create.call_args.kwargs
    assert kw["temperature"] == 0.0
    assert kw["max_tokens"] == 100


def test_openai_chat_returns_empty_string_on_none_content() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response(None)
    c = OpenAIChat(client=client, async_client=AsyncMock())
    assert c.chat([Message(role="user", content="x")]) == ""


def test_openai_chat_warns_on_length_truncation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`finish_reason == "length"` means the model hit max_tokens and was
    cut off; the adapter must log a WARNING so the caller can tell a
    truncated response apart from a complete short one. See JOURNEY §24.
    """
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response(
        "cut off mid", finish_reason="length"
    )
    c = OpenAIChat(model="kimi-k2.6", client=client, async_client=AsyncMock())
    with caplog.at_level("WARNING", logger="engram.providers.openai"):
        out = c.chat([Message(role="user", content="x")])
    assert out == "cut off mid"  # content still returned, not dropped
    assert any(
        "max_tokens" in rec.message and "kimi-k2.6" in rec.message
        for rec in caplog.records
    )


def test_openai_chat_warns_on_length_truncation_with_null_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The empty-response failure mode from JOURNEY §24: model used the
    full max_tokens budget for reasoning and emitted nothing to the
    answer channel. `finish_reason` is the only signal that this isn't
    a clean "model had nothing to say" empty response.
    """
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response(
        None, finish_reason="length"
    )
    c = OpenAIChat(model="kimi-k2.6", client=client, async_client=AsyncMock())
    with caplog.at_level("WARNING", logger="engram.providers.openai"):
        out = c.chat([Message(role="user", content="x")])
    assert out == ""
    assert any("0 chars" in rec.message for rec in caplog.records)


def test_openai_chat_warns_on_content_filter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response(
        "blocked", finish_reason="content_filter"
    )
    c = OpenAIChat(client=client, async_client=AsyncMock())
    with caplog.at_level("WARNING", logger="engram.providers.openai"):
        c.chat([Message(role="user", content="x")])
    assert any("content_filter" in rec.message for rec in caplog.records)


def test_openai_chat_does_not_warn_on_clean_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sanity: a normal "stop" finish_reason must NOT emit a warning,
    otherwise every benchmark run is noise.
    """
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response(
        "all done", finish_reason="stop"
    )
    c = OpenAIChat(client=client, async_client=AsyncMock())
    with caplog.at_level("WARNING", logger="engram.providers.openai"):
        c.chat([Message(role="user", content="x")])
    assert not caplog.records


def test_openai_chat_async() -> None:
    aclient = AsyncMock()
    aclient.chat.completions.create.return_value = _make_chat_response("async-ok")
    c = OpenAIChat(client=MagicMock(), async_client=aclient)
    assert asyncio.run(c.achat([Message(role="user", content="x")])) == "async-ok"


def test_openai_chat_manifest_hash_changes_with_kwargs() -> None:
    c1 = OpenAIChat(
        client=MagicMock(), async_client=AsyncMock(), completion_kwargs={"temperature": 0}
    )
    c2 = OpenAIChat(
        client=MagicMock(), async_client=AsyncMock(), completion_kwargs={"temperature": 1}
    )
    assert c1.manifest_hash() != c2.manifest_hash()


def test_openai_chat_redacts_api_key_from_error_message() -> None:
    """An exception raised by the SDK with a key in its message must be redacted."""
    client = MagicMock()
    leaked = "sk-abcdefghijklmnopqrstuvwxyz12345678"
    client.chat.completions.create.side_effect = RuntimeError(
        f"Bad request — sent token={leaked}"
    )
    c = OpenAIChat(client=client, async_client=AsyncMock())
    with pytest.raises(RuntimeError) as excinfo:
        c.chat([Message(role="user", content="hi")])
    assert leaked not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)
    # Original error chained as __cause__ so debugging context survives.
    assert excinfo.value.__cause__ is not None


def test_openai_embedder_redacts_api_key_from_error_message() -> None:
    client = MagicMock()
    leaked = "sk-ant-api01-secretkey1234567890abcdef"
    client.embeddings.create.side_effect = RuntimeError(
        f"auth failed token={leaked}"
    )
    e = OpenAIEmbedder(client=client, async_client=AsyncMock())
    with pytest.raises(RuntimeError) as excinfo:
        e.embed(["x"])
    assert leaked not in str(excinfo.value)
