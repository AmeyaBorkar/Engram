"""Tests for `engram.providers._disk_cache.DiskCache`."""

from __future__ import annotations

from pathlib import Path

from engram.providers._disk_cache import DiskCache


def test_embed_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache.db")
    try:
        key = cache.embed_key("openai-embed", "text-embedding-3-small", "hello")
        assert cache.embed_get(key) is None
        cache.embed_set(key, [0.1, 0.2, 0.3])
        out = cache.embed_get(key)
        assert out is not None
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
