"""Tests for `FakeEmbedder`, `FakeChat`, and the provider protocols."""

from __future__ import annotations

import asyncio
import math

import pytest

from engram.providers import (
    ChatProvider,
    EmbeddingProvider,
    FakeChat,
    FakeEmbedder,
    Message,
    content_hash,
)

# --- FakeEmbedder ---------------------------------------------------------


def test_fake_embedder_satisfies_protocol() -> None:
    e = FakeEmbedder()
    assert isinstance(e, EmbeddingProvider)


def test_fake_embedder_deterministic() -> None:
    a = FakeEmbedder(dim=64).embed(["hello"])[0]
    b = FakeEmbedder(dim=64).embed(["hello"])[0]
    assert a == b


def test_fake_embedder_different_text_different_vector() -> None:
    e = FakeEmbedder(dim=64)
    a, b = e.embed(["alpha", "beta"])
    assert a != b


def test_fake_embedder_unit_norm() -> None:
    vec = FakeEmbedder(dim=128).embed(["whatever"])[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)


def test_fake_embedder_dim_respected() -> None:
    for dim in (8, 64, 128, 384, 1024):
        vec = FakeEmbedder(dim=dim).embed(["hello"])[0]
        assert len(vec) == dim


def test_fake_embedder_rejects_invalid_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        FakeEmbedder(dim=0)


def test_fake_embedder_batch_matches_singles() -> None:
    e = FakeEmbedder(dim=32)
    batch = e.embed(["a", "b", "c"])
    singles = [e.embed([t])[0] for t in ["a", "b", "c"]]
    assert batch == singles


def test_fake_embedder_async_matches_sync() -> None:
    e = FakeEmbedder(dim=32)
    sync = e.embed(["x", "y"])
    async_ = asyncio.run(e.aembed(["x", "y"]))
    assert sync == async_


def test_fake_embedder_manifest_hash_stable() -> None:
    a = FakeEmbedder(dim=128).manifest_hash()
    b = FakeEmbedder(dim=128).manifest_hash()
    assert a == b


def test_fake_embedder_manifest_hash_changes_with_dim() -> None:
    a = FakeEmbedder(dim=64).manifest_hash()
    b = FakeEmbedder(dim=128).manifest_hash()
    assert a != b


# --- FakeChat -------------------------------------------------------------


def test_fake_chat_satisfies_protocol() -> None:
    c = FakeChat()
    assert isinstance(c, ChatProvider)


def test_fake_chat_default_no_user_message() -> None:
    c = FakeChat(default="empty-conv")
    reply = c.chat([Message(role="system", content="be helpful")])
    assert reply == "empty-conv"


def test_fake_chat_scripted_response_by_content_hash() -> None:
    user_text = "what's the weather?"
    key = content_hash(user_text)
    c = FakeChat(scripts={key: "sunny"})
    reply = c.chat([Message(role="user", content=user_text)])
    assert reply == "sunny"


def test_fake_chat_falls_back_to_default_when_no_script() -> None:
    c = FakeChat(default="<default>")
    reply = c.chat([Message(role="user", content="anything")])
    assert reply == "<default>"


def test_fake_chat_default_fallback_includes_input_hash_when_no_default_set() -> None:
    c = FakeChat()
    reply = c.chat([Message(role="user", content="hi")])
    expected_prefix = content_hash("hi")[:12]
    assert expected_prefix in reply


def test_fake_chat_uses_last_user_message() -> None:
    c = FakeChat(scripts={content_hash("second"): "got-it"})
    reply = c.chat(
        [
            Message(role="user", content="first"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="second"),
        ]
    )
    assert reply == "got-it"


def test_fake_chat_async_matches_sync() -> None:
    c = FakeChat(default="<d>")
    msgs = [Message(role="user", content="x")]
    sync = c.chat(msgs)
    async_ = asyncio.run(c.achat(msgs))
    assert sync == async_


def test_fake_chat_manifest_hash_stable_for_same_config() -> None:
    a = FakeChat(scripts={"k": "v"}, default="d").manifest_hash()
    b = FakeChat(scripts={"k": "v"}, default="d").manifest_hash()
    assert a == b


def test_fake_chat_manifest_hash_changes_with_scripts() -> None:
    a = FakeChat(scripts={"k": "v1"}).manifest_hash()
    b = FakeChat(scripts={"k": "v2"}).manifest_hash()
    assert a != b


def test_fake_chat_manifest_hash_changes_with_default() -> None:
    a = FakeChat(default="A").manifest_hash()
    b = FakeChat(default="B").manifest_hash()
    assert a != b
