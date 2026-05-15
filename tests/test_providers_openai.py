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


def _make_chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


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


# --- chunking (M-93) -------------------------------------------------------


def test_openai_embedder_chunks_large_input_list() -> None:
    """Inputs larger than `chunk_size` are split into multiple SDK calls."""
    client = MagicMock()
    # Each chunked call returns N vectors; we want call count to match
    # the number of chunks, and the final output to preserve order.
    chunk_count = {"n": 0}

    def fake_create(**kw: object) -> object:
        chunk_count["n"] += 1
        n = len(kw["input"])  # type: ignore[arg-type]
        return _make_embed_response([[float(chunk_count["n"])] * 2 for _ in range(n)])

    client.embeddings.create.side_effect = fake_create
    e = OpenAIEmbedder(
        client=client, async_client=AsyncMock(), chunk_size=4
    )
    out = e.embed([f"t{i}" for i in range(10)])
    assert len(out) == 10
    # 10 / 4 -> 3 chunks (4, 4, 2)
    assert chunk_count["n"] == 3


def test_openai_embedder_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        OpenAIEmbedder(chunk_size=0, client=MagicMock(), async_client=AsyncMock())


def test_openai_embedder_empty_input_makes_no_calls() -> None:
    client = MagicMock()
    e = OpenAIEmbedder(client=client, async_client=AsyncMock())
    out = e.embed([])
    assert out == []
    client.embeddings.create.assert_not_called()


def test_openai_embedder_async_chunks() -> None:
    aclient = AsyncMock()
    chunk_count = {"n": 0}

    async def fake_create(**kw: object) -> object:
        chunk_count["n"] += 1
        n = len(kw["input"])  # type: ignore[arg-type]
        return _make_embed_response([[float(chunk_count["n"])] for _ in range(n)])

    aclient.embeddings.create.side_effect = fake_create
    e = OpenAIEmbedder(client=MagicMock(), async_client=aclient, chunk_size=3)
    out = asyncio.run(e.aembed([f"t{i}" for i in range(7)]))
    assert len(out) == 7
    assert chunk_count["n"] == 3  # 3, 3, 1


# --- per-call kwargs (M-173) -----------------------------------------------


def test_openai_chat_per_call_kwargs_override_constructor() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response("ok")
    c = OpenAIChat(
        client=client,
        async_client=AsyncMock(),
        completion_kwargs={"temperature": 0.0, "max_tokens": 100},
    )
    c.chat([Message(role="user", content="x")], extra={"temperature": 0.7})
    kw = client.chat.completions.create.call_args.kwargs
    assert kw["temperature"] == 0.7  # overridden
    assert kw["max_tokens"] == 100  # passed through


def test_openai_chat_per_call_kwargs_async() -> None:
    aclient = AsyncMock()
    aclient.chat.completions.create.return_value = _make_chat_response("ok")
    c = OpenAIChat(client=MagicMock(), async_client=aclient)
    asyncio.run(
        c.achat([Message(role="user", content="x")], extra={"temperature": 0.5})
    )
    kw = aclient.chat.completions.create.call_args.kwargs
    assert kw["temperature"] == 0.5


def test_openai_chat_per_call_kwargs_does_not_mutate_constructor_state() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_chat_response("ok")
    c = OpenAIChat(
        client=client,
        async_client=AsyncMock(),
        completion_kwargs={"temperature": 0.0},
    )
    c.chat([Message(role="user", content="x")], extra={"temperature": 1.0})
    c.chat([Message(role="user", content="x")])
    last_kw = client.chat.completions.create.call_args.kwargs
    # Second call must have reverted to the constructor default.
    assert last_kw["temperature"] == 0.0


# --- close / context manager (M-94) ---------------------------------------


def test_openai_embedder_close_idempotent() -> None:
    client = MagicMock()
    e = OpenAIEmbedder(client=client, async_client=AsyncMock())
    e.close()
    e.close()  # idempotent; should not raise


def test_openai_chat_close_idempotent() -> None:
    client = MagicMock()
    c = OpenAIChat(client=client, async_client=AsyncMock())
    c.close()
    c.close()


def test_openai_embedder_context_manager_closes() -> None:
    client = MagicMock()
    aclient = AsyncMock()
    with OpenAIEmbedder(client=client, async_client=aclient) as e:
        assert isinstance(e, OpenAIEmbedder)
    # client.close was invoked on context exit; SDK's `close` exists as
    # a real method on the MagicMock so it should have been called.
    assert client.close.called


def test_openai_chat_context_manager_closes() -> None:
    client = MagicMock()
    aclient = AsyncMock()
    with OpenAIChat(client=client, async_client=aclient) as c:
        assert isinstance(c, OpenAIChat)
    assert client.close.called
