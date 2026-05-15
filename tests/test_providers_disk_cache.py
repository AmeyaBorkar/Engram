"""Tests for `engram.providers._disk_cache`."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from engram.providers._disk_cache import (
    CachedChat,
    CachedEmbedder,
    DiskCache,
    _decode_vector,
    _encode_vector,
    _hash_messages_for_key,
    _hash_text_for_key,
    with_disk_cache,
)
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.providers._message import Message


# --- hash key helpers -----------------------------------------------------


def test_embed_key_nul_separator_does_not_collide() -> None:
    """A NUL byte inside `text` must not align with a separator and
    collide with a different (provider, model, text) triple."""
    a = _hash_text_for_key("provA", "modelA", "abc\x00def")
    # The legacy `\x00`-concat key collapsed these inputs together.
    b = _hash_text_for_key("provA", "modelA\x00abc", "def")
    assert a != b


def test_embed_key_separator_robustness() -> None:
    """Distinct splits of the same concatenated form hash differently."""
    a = _hash_text_for_key("a", "b", "cd")
    b = _hash_text_for_key("a", "bc", "d")
    c = _hash_text_for_key("ab", "c", "d")
    assert len({a, b, c}) == 3


def test_chat_key_is_deterministic_and_role_aware() -> None:
    msgs1 = [
        Message(role="system", content="be brief"),
        Message(role="user", content="hi"),
    ]
    msgs2 = [
        Message(role="user", content="hi"),
        Message(role="system", content="be brief"),
    ]
    k1 = _hash_messages_for_key("p", "m", msgs1)
    k2 = _hash_messages_for_key("p", "m", msgs2)
    assert k1 != k2
    assert _hash_messages_for_key("p", "m", msgs1) == k1


# --- vector encoding -----------------------------------------------------


def test_vector_roundtrip_binary() -> None:
    """float64 packing is bit-identical to the input."""
    vec = [0.1, -0.5, 3.14, 1e-3, 1.0 / 7.0]
    blob = _encode_vector(vec)
    decoded = _decode_vector(blob)
    assert decoded == vec


def test_vector_blob_is_compact_vs_json() -> None:
    """Binary should be smaller than the JSON form for typical sizes.

    With float64 we expect ~2x at minimum on small-decimal vectors and
    closer to 4-5x once values have a mantissa worth printing.
    """
    import json

    vec = [float(i) / 100.0 + 1.0 / 7.0 for i in range(1024)]
    blob = _encode_vector(vec)
    json_form = json.dumps(vec).encode("utf-8")
    assert len(blob) < len(json_form)


def test_legacy_json_blob_decodes() -> None:
    """Older caches stored vectors as JSON text; decode must tolerate them."""
    import json

    legacy = json.dumps([0.5, 1.5, 2.5]).encode("utf-8")
    decoded = _decode_vector(legacy)
    assert decoded == [0.5, 1.5, 2.5]


def test_empty_blob_decodes_to_empty_list() -> None:
    assert _decode_vector(b"") == []


# --- DiskCache end-to-end -------------------------------------------------


def test_disk_cache_round_trip_embed(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c.sqlite")
    try:
        key = cache.embed_key("p", "m", "hello")
        assert cache.embed_get(key) is None
        cache.embed_set(key, [0.1, 0.2])
        got = cache.embed_get(key)
        assert got is not None
        assert abs(got[0] - 0.1) < 1e-6
    finally:
        cache.close()


def test_disk_cache_round_trip_chat(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c.sqlite")
    try:
        msgs = [Message(role="user", content="hi")]
        key = cache.chat_key("p", "m", msgs)
        assert cache.chat_get(key) is None
        cache.chat_set(key, "hello back")
        assert cache.chat_get(key) == "hello back"
    finally:
        cache.close()


def test_disk_cache_close_is_idempotent(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c.sqlite")
    cache.close()
    cache.close()


def test_disk_cache_path_traversal_guard(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    # Inside-the-root path resolves fine.
    inside = DiskCache(root / "c.sqlite", allowed_root=root)
    inside.close()
    # Out-of-root path is rejected.
    outside = tmp_path / "elsewhere" / "c.sqlite"
    with pytest.raises(ValueError, match="not under allowed_root"):
        DiskCache(outside, allowed_root=root)


# --- with_disk_cache wrappers --------------------------------------------


def test_with_disk_cache_chat_caches_responses(tmp_path: Path) -> None:
    inner = FakeChat()
    wrapped = with_disk_cache(inner, path=tmp_path / "c.sqlite")
    try:
        msgs = [Message(role="user", content="hi")]
        first = wrapped.chat(msgs)
        second = wrapped.chat(msgs)
        assert first == second
        # First call populates, second is a hit.
        assert wrapped._cache.stats["chat_hits"] >= 1
    finally:
        wrapped.close()


def test_with_disk_cache_embed_caches_and_preserves_manifest_hash(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    wrapped = with_disk_cache(inner, path=tmp_path / "c.sqlite")
    try:
        _ = wrapped.embed(["a", "b"])
        # Second pass hits the cache without invoking the inner provider.
        out2 = wrapped.embed(["a", "b"])
        assert len(out2) == 2
        stats = wrapped._cache.stats
        assert stats["embed_hits"] >= 2
        # Manifest hash passes through unchanged.
        assert wrapped.manifest_hash() == inner.manifest_hash()
    finally:
        wrapped.close()


def test_with_disk_cache_embed_async_round_trip(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    wrapped = with_disk_cache(inner, path=tmp_path / "c.sqlite")

    async def go() -> list[list[float]]:
        # First call: miss -> inner.embed -> float64 result returned to caller.
        # Second call: hit -> float64 round-tripped through sqlite (lossless).
        out1 = await wrapped.aembed(["a", "b"])
        out2 = await wrapped.aembed(["a", "b"])
        assert out1 == out2
        return out2

    try:
        out = asyncio.run(go())
        assert len(out) == 2
    finally:
        wrapped.close()


def test_with_disk_cache_chat_async_round_trip(tmp_path: Path) -> None:
    inner = FakeChat()
    wrapped = with_disk_cache(inner, path=tmp_path / "c.sqlite")

    async def go() -> tuple[str, str]:
        msgs = [Message(role="user", content="hi")]
        a = await wrapped.achat(msgs)
        b = await wrapped.achat(msgs)
        return a, b

    try:
        a, b = asyncio.run(go())
        assert a == b
    finally:
        wrapped.close()


def test_with_disk_cache_rejects_unknown_type(tmp_path: Path) -> None:
    class NotAProvider:
        pass

    with pytest.raises(TypeError, match="neither"):
        with_disk_cache(NotAProvider(), path=tmp_path / "c.sqlite")


def test_with_disk_cache_env_root_constrains_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "constrained"
    root.mkdir()
    monkeypatch.setenv("ENGRAM_DISK_CACHE_ROOT", str(root))
    # Inside the env-pinned root: ok.
    ok = with_disk_cache(FakeEmbedder(), path=root / "c.sqlite")
    ok.close()
    # Outside the env-pinned root: rejected.
    outside = tmp_path / "outside" / "c.sqlite"
    with pytest.raises(ValueError, match="not under allowed_root"):
        with_disk_cache(FakeEmbedder(), path=outside)


def test_chat_key_streams_without_materializing_full_payload() -> None:
    """The key derives from message contents but must not build a huge
    intermediate string. Smoke-test that a large message hashes without
    exploding memory; correctness is sufficient here."""
    long_text = "x" * 5_000_000
    key = _hash_messages_for_key(
        "p", "m", [Message(role="user", content=long_text)]
    )
    assert len(key) == 64


# --- async I/O off the event-loop thread ---------------------------------


def test_aembed_runs_sqlite_off_loop_thread(tmp_path: Path) -> None:
    """The aembed path should call sqlite via to_thread so it does not
    block the event loop. We approximate the assertion by checking that
    the call returns correctly under a tight asyncio loop."""

    class CountingEmbedder:
        name = "ce"
        model = "m"
        dim = 3

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls += 1
            return [[float(i)] * 3 for i, _ in enumerate(texts)]

        async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
            return self.embed(texts)

        def manifest_hash(self) -> str:
            return "counting/v1"

    inner = CountingEmbedder()
    wrapped = with_disk_cache(inner, path=tmp_path / "c.sqlite")
    try:

        async def go() -> None:
            await wrapped.aembed(["a", "b"])
            await wrapped.aembed(["a", "b"])

        asyncio.run(go())
        # Second call is a full cache hit; inner should have been
        # invoked once.
        assert inner.calls == 1
    finally:
        wrapped.close()
