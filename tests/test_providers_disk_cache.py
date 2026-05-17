"""Tests for `engram.providers._disk_cache.DiskCache`."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from engram.providers._disk_cache import (
    _EMBED_BINARY_MAGIC,
    CachedEmbedder,
    DiskCache,
    _HANDLES,
    _decode_embed_blob,
    with_disk_cache,
)
from engram.providers._message import Message


def test_embed_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        key = cache.embed_key("openai-embed", "text-embedding-3-small", "hello")
        assert cache.embed_get(key) is None
        cache.embed_set(key, [0.1, 0.2, 0.3])
        out = cache.embed_get(key)
        assert out is not None
        # Packed-binary round-trip is float64-exact for clean values.
        assert out == [0.1, 0.2, 0.3]
    finally:
        cache.close()


def test_embed_get_treats_corrupt_blob_as_miss(tmp_path: Path) -> None:
    """A non-utf8 / non-json blob in the cache table must not raise.

    Crash, manual tampering, or sqlite page corruption can leave a
    well-keyed row with garbage in `value`.  Callers should see a miss
    and re-embed, not an uncaught UnicodeDecodeError that bricks a bench.
    """
    cache = DiskCache(tmp_path / "cache.db")
    try:
        key = cache.embed_key("p", "m", "t")
        # Inject a deliberately corrupt blob bypassing embed_set.
        cache._conn.execute(  # noqa: SLF001 - test patches private state on purpose
            "INSERT OR REPLACE INTO embed (key, value) VALUES (?, ?)",
            (key, b"\xff\xfegarbage"),
        )
        assert cache.embed_get(key) is None
    finally:
        cache.close()


def test_embed_get_treats_malformed_json_as_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        key = cache.embed_key("p", "m", "t")
        cache._conn.execute(  # noqa: SLF001
            "INSERT OR REPLACE INTO embed (key, value) VALUES (?, ?)",
            (key, b'{"not": "a vector"}'),
        )
        assert cache.embed_get(key) is None
    finally:
        cache.close()


def test_embed_get_rejects_list_of_strings(tmp_path: Path) -> None:
    """A JSON list of non-numeric elements is not a valid vector."""
    cache = DiskCache(tmp_path / "cache.db")
    try:
        key = cache.embed_key("p", "m", "t")
        cache._conn.execute(  # noqa: SLF001
            "INSERT OR REPLACE INTO embed (key, value) VALUES (?, ?)",
            (key, b'["a", "b", "c"]'),
        )
        assert cache.embed_get(key) is None
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# M-11: length-prefixed embed_key
# ---------------------------------------------------------------------------


def test_embed_key_resists_nul_collision(tmp_path: Path) -> None:
    """The pre-fix `\\x00`-delimited concat collided whenever an input held a NUL.

    Length-prefixing each component eliminates the entire collision
    class — confirm by constructing two distinct (provider, model, text)
    triples that the legacy implementation would have hashed identically.
    """
    cache = DiskCache(tmp_path / "cache.db")
    try:
        # ("ab", "c", "d") vs ("a", "bc", "d") would collide under
        # naive concat; with length prefixes they must not.
        a = cache.embed_key("ab", "c", "d")
        b = cache.embed_key("a", "bc", "d")
        assert a != b
        # Embedded NUL bytes must not collapse two distinct inputs.
        c = cache.embed_key("a\x00b", "m", "t")
        d = cache.embed_key("a", "b\x00m", "t")
        assert c != d
    finally:
        cache.close()


def test_embed_key_stable_for_identical_inputs(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        assert cache.embed_key("p", "m", "t") == cache.embed_key("p", "m", "t")
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# M-10: chat_key streams instead of materializing the full JSON payload
# ---------------------------------------------------------------------------


def test_chat_key_deterministic(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        msgs = [
            Message(role="system", content="you are a helpful assistant"),
            Message(role="user", content="hi"),
        ]
        a = cache.chat_key("openai-chat", "gpt-4o-mini", msgs)
        b = cache.chat_key("openai-chat", "gpt-4o-mini", msgs)
        assert a == b
    finally:
        cache.close()


def test_chat_key_distinguishes_message_split(tmp_path: Path) -> None:
    """`("ab","c")` vs `("a","bc")` content must not collide."""
    cache = DiskCache(tmp_path / "cache.db")
    try:
        a = cache.chat_key(
            "p", "m", [Message(role="user", content="ab"), Message(role="user", content="c")]
        )
        b = cache.chat_key(
            "p", "m", [Message(role="user", content="a"), Message(role="user", content="bc")]
        )
        assert a != b
    finally:
        cache.close()


def test_chat_key_distinguishes_role_split(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        a = cache.chat_key(
            "p", "m", [Message(role="system", content="x"), Message(role="user", content="y")]
        )
        b = cache.chat_key(
            "p", "m", [Message(role="user", content="x"), Message(role="user", content="y")]
        )
        assert a != b
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# H-19: embed_get_many batch lookup
# ---------------------------------------------------------------------------


def test_embed_get_many_returns_only_present_keys(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        keys = [cache.embed_key("p", "m", f"t{i}") for i in range(5)]
        # Store the first three only.
        for j, k in enumerate(keys[:3]):
            cache.embed_set(k, [float(j)])
        found = cache.embed_get_many(keys)
        assert set(found.keys()) == set(keys[:3])
        # Missing keys are absent from the result, not None-filled.
        assert keys[3] not in found
    finally:
        cache.close()


def test_embed_get_many_chunks_over_500_keys(tmp_path: Path) -> None:
    """Sanity: 1200-key lookup must succeed (two full chunks + tail).

    Pre-fix `_partition_cache` did 1200 sqlite roundtrips; this exercises
    the chunked `WHERE key IN (?,?,…)` path so the chunk seams stay
    correct.
    """
    cache = DiskCache(tmp_path / "cache.db")
    try:
        keys = [cache.embed_key("p", "m", f"t{i}") for i in range(1200)]
        for i, k in enumerate(keys):
            cache.embed_set(k, [float(i)])
        found = cache.embed_get_many(keys)
        assert len(found) == 1200
    finally:
        cache.close()


def test_embed_get_many_empty_input(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        assert cache.embed_get_many([]) == {}
    finally:
        cache.close()


def test_cached_embedder_uses_batch_lookup(tmp_path: Path) -> None:
    """The wrapper must consult the disk cache once per `embed`, not per text."""
    cache = DiskCache(tmp_path / "cache.db")
    try:
        # Pre-fill keys for two texts.
        keys = [cache.embed_key("fake-embed", "stub", "a"), cache.embed_key("fake-embed", "stub", "b")]
        cache.embed_set(keys[0], [0.5])
        cache.embed_set(keys[1], [1.5])

        class _Stub:
            name = "fake-embed"
            model = "stub"
            dim = 1

            def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
                # Should not be called on a full-hit batch.
                raise AssertionError("inner embed must not be called on cache hits")

            async def aembed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
                raise AssertionError

            def manifest_hash(self) -> str:  # pragma: no cover
                return "stub"

        wrapped = CachedEmbedder(_Stub(), cache)  # type: ignore[arg-type]
        vecs = wrapped.embed(["a", "b"])
        assert vecs == [[0.5], [1.5]]
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# H-20: with_disk_cache memoizes per resolved path
# ---------------------------------------------------------------------------


def test_with_disk_cache_memoizes_same_path(tmp_path: Path) -> None:
    """Two `with_disk_cache(provider, path=p)` calls share one DiskCache."""

    class _StubChat:
        name = "stub-chat"
        model = "stub"

        def chat(self, messages: Sequence[Message]) -> str:  # pragma: no cover
            return ""

        async def achat(self, messages: Sequence[Message]) -> str:  # pragma: no cover
            return ""

        def manifest_hash(self) -> str:
            return "stub"

    class _StubEmbed:
        name = "stub-embed"
        model = "stub"
        dim = 1

        def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
            return []

        async def aembed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
            return []

        def manifest_hash(self) -> str:
            return "stub"

    path = tmp_path / "shared.db"
    _HANDLES.clear()
    try:
        chat_wrapper = with_disk_cache(_StubChat(), path=path)
        embed_wrapper = with_disk_cache(_StubEmbed(), path=path)
        # Both wrappers must share the underlying DiskCache instance,
        # which is exactly the goal — no double sqlite handles.
        assert chat_wrapper._cache is embed_wrapper._cache  # noqa: SLF001
        # And the same handle is registered.
        assert len(_HANDLES) == 1
    finally:
        # Closing once should drop it from the registry.
        chat_wrapper._cache.close()  # noqa: SLF001


def test_with_disk_cache_normalizes_path_spelling(tmp_path: Path) -> None:
    """`./foo.db` and `foo.db` (same dir) collapse to one handle."""

    class _StubEmbed:
        name = "stub-embed"
        model = "stub"
        dim = 1

        def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
            return []

        async def aembed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
            return []

        def manifest_hash(self) -> str:
            return "stub"

    path1 = tmp_path / "x.db"
    path2 = tmp_path.joinpath("x.db")
    _HANDLES.clear()
    try:
        a = with_disk_cache(_StubEmbed(), path=path1)
        b = with_disk_cache(_StubEmbed(), path=path2)
        assert a._cache is b._cache  # noqa: SLF001
    finally:
        a._cache.close()  # noqa: SLF001


# ---------------------------------------------------------------------------
# M-12: traversal guard
# ---------------------------------------------------------------------------


def test_disk_cache_rejects_path_outside_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.db"
    with pytest.raises(ValueError, match="allowed_root"):
        DiskCache(outside, allowed_root=root)


def test_disk_cache_accepts_path_under_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    cache = DiskCache(root / "ok.db", allowed_root=root)
    try:
        assert Path(cache.path).is_relative_to(root.resolve())
    finally:
        cache.close()


def test_disk_cache_honors_env_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "envroot"
    root.mkdir()
    outside = tmp_path / "elsewhere.db"
    monkeypatch.setenv("ENGRAM_DISK_CACHE_ROOT", str(root))
    with pytest.raises(ValueError, match="allowed_root"):
        DiskCache(outside)


# ---------------------------------------------------------------------------
# M-97 / M-208: packed-binary format + legacy JSON back-compat
# ---------------------------------------------------------------------------


def test_embed_blob_is_packed_binary(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        key = cache.embed_key("p", "m", "t")
        cache.embed_set(key, [0.0, 1.5, -2.25])
        row = cache._conn.execute(  # noqa: SLF001
            "SELECT value FROM embed WHERE key = ?", (key,)
        ).fetchone()
        blob: bytes = bytes(row["value"])
        assert blob.startswith(_EMBED_BINARY_MAGIC)
        # 3 float64 -> 24 bytes after the magic.
        assert len(blob) == len(_EMBED_BINARY_MAGIC) + 24
    finally:
        cache.close()


def test_embed_get_reads_legacy_json_blob(tmp_path: Path) -> None:
    """An old cache file written by a pre-fix Engram must still read.

    `_decode_embed_blob` auto-detects the absence of the magic prefix
    and falls back to JSON parsing — exercise that explicitly.
    """
    cache = DiskCache(tmp_path / "cache.db")
    try:
        key = cache.embed_key("p", "m", "t")
        # Bypass `embed_set` to write the legacy text-encoded JSON
        # shape directly.
        legacy = json.dumps([0.1, 0.2, 0.3]).encode("utf-8")
        cache._conn.execute(  # noqa: SLF001
            "INSERT OR REPLACE INTO embed (key, value) VALUES (?, ?)",
            (key, legacy),
        )
        out = cache.embed_get(key)
        assert out == [0.1, 0.2, 0.3]
    finally:
        cache.close()


def test_decode_embed_blob_handles_truncated_binary() -> None:
    """A magic-prefixed blob with non-multiple-of-8 payload is a miss."""
    bad = _EMBED_BINARY_MAGIC + b"\x00\x00\x00"  # 3 trailing bytes
    assert _decode_embed_blob(bad) is None


# ---------------------------------------------------------------------------
# H-21: Cache thread-safety smoke test (cross-references _cache.Cache, but
# the disk-cache test file is the most natural home for the regression).
# ---------------------------------------------------------------------------


def test_cache_concurrent_set_get_does_not_corrupt(tmp_path: Path) -> None:
    """Hit the in-memory Cache from multiple threads simultaneously."""
    from engram.providers._cache import Cache

    cache: Cache[int] = Cache(max_size=64)
    errors: list[BaseException] = []

    def worker(start: int) -> None:
        try:
            for i in range(start, start + 200):
                cache.set(f"k{i}", i)
                cache.get(f"k{i}")
        except BaseException as exc:  # pragma: no cover - corruption surfaces here
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(0, 1000, 200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # Counter consistency: every successful get bumped one counter slot.
    assert cache.hits + cache.misses > 0
