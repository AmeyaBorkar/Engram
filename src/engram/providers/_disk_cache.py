"""Persistent disk cache for chat + embed provider responses.

In-memory `Cache` (see `_cache.py`) only survives the process; cross-run
ablations re-pay for the same LLM and embedder calls every time. The
disk cache stores `(model, prompt_hash) -> response` in a sqlite file
so a second run over the same haystack reads from disk instead of the
network / GPU.

Schema is intentionally minimal: two tables (`chat`, `embed`), no
versioning, no migrations. Bump the path when the cache shape changes
or just delete the file. Cache entries are not deduplicated across
providers -- the key includes `provider/model` so swapping chat
providers does not contaminate.

Concurrency: WAL mode, foreign keys off, single shared connection
behind a lock. Acceptable for the bench's single-process usage. For
multi-process callers, point each process at a distinct path or wrap
your own coordination.

Vector storage: vectors are stored as a 1-byte tag + packed float64
(little-endian). JSON-text storage cost ~3-4x the binary form and
quintuple-digit ser/de cycles per run. float64 (not float32) so the
round-trip is bit-identical to the Python float the caller produced
-- a downstream cosine score should not move just because the value
came from cache. The tag lets future formats slot in without
invalidating today's caches. Tag 0x02 = float64 LE; 0x01 reserved
for an explicit float32 format if a future caller wants the size
savings and is willing to absorb the precision loss.

Path safety: `with_disk_cache(path=...)` accepts any path by default.
Pass `allowed_root=<Path>` to constrain the resolved path to live
under that root -- recommended for multi-tenant or untrusted-input
deployments. Out-of-root paths raise `ValueError`.

Wire it in via `with_disk_cache(provider, path=...)`; the returned
wrapper proxies the provider's surface and consults / fills the cache
on every call. Both wrappers expose `close()` so the underlying
sqlite connection releases on shutdown.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import struct
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.providers._message import Message

if TYPE_CHECKING:
    from engram.providers._protocols import ChatProvider, EmbeddingProvider


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embed (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
"""

# Vector blob format tags. Position 0 is a 1-byte tag; the remainder is
# the payload in that format. Old JSON-encoded entries are detected by
# the first byte being `[` (0x5B) -- they are migrated on read.
_TAG_FLOAT32 = 0x01  # reserved for future opt-in size-over-precision callers
_TAG_FLOAT64 = 0x02  # default; lossless round-trip with Python float


def _encode_vector(vec: Sequence[float]) -> bytes:
    """Pack a float vector as tag + float64 LE.

    float64 keeps the round-trip bit-identical to the caller's input,
    so a cache hit and a cache miss return the same vector. With 8
    bytes per element vs. ~17 in JSON text, this is still a ~2x size
    win in the worst case and ~3-4x for typical small floats.
    """
    arr = bytearray(1 + 8 * len(vec))
    arr[0] = _TAG_FLOAT64
    struct.pack_into(f"<{len(vec)}d", arr, 1, *vec)
    return bytes(arr)


def _decode_vector(blob: bytes) -> list[float]:
    """Unpack a vector blob; tolerate float32 and the legacy JSON encoding."""
    if not blob:
        return []
    tag = blob[0]
    if tag == _TAG_FLOAT64:
        n = (len(blob) - 1) // 8
        return list(struct.unpack_from(f"<{n}d", blob, 1))
    if tag == _TAG_FLOAT32:
        n = (len(blob) - 1) // 4
        return list(struct.unpack_from(f"<{n}f", blob, 1))
    # Legacy: pre-binary JSON encoding (text starting with `[`).
    return list(json.loads(blob.decode("utf-8")))


def _hash_messages_for_key(provider: str, model: str, messages: Sequence[Message]) -> str:
    """Stream a SHA-256 over the canonical (provider, model, messages) form.

    Length-prefixed parts -- never a single concatenated string -- so
    no two distinct inputs hash to the same key just because one
    happens to contain the separator. Streams in chunks so the full
    payload never materializes in memory before hashing (M-10).
    """
    h = hashlib.sha256()

    def _feed(s: str) -> None:
        b = s.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)

    _feed(provider)
    _feed(model)
    h.update(len(messages).to_bytes(8, "big"))
    for m in messages:
        _feed(m.role)
        _feed(m.content)
    return h.hexdigest()


def _hash_text_for_key(provider: str, model: str, text: str) -> str:
    """Length-prefixed hash; NUL bytes in `text` cannot collide with the separator."""
    h = hashlib.sha256()
    for part in (provider, model, text):
        b = part.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def _validate_path(path: str | Path, allowed_root: Path | None) -> Path:
    """Resolve `path` and, if `allowed_root` is given, refuse traversal escapes.

    Resolves both arguments before comparing so symlinks and `..`
    segments can't smuggle the cache out of the configured root.
    """
    resolved = Path(path).resolve()
    if allowed_root is None:
        return resolved
    root_resolved = allowed_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"disk cache path {resolved!s} is not under allowed_root {root_resolved!s}"
        ) from exc
    return resolved


class DiskCache:
    """SQLite-backed (key -> value) cache for provider responses.

    Two columns: `key` (SHA-256 hex) and `value` (text for chat, packed
    float32 for embed vectors). Entries are never evicted -- the bench
    corpus is bounded, and the user can rm the file when they want to
    start fresh.
    """

    def __init__(self, path: str | Path, *, allowed_root: Path | None = None) -> None:
        resolved = _validate_path(path, allowed_root)
        # Make sure the parent directory exists; sqlite would otherwise fail
        # with an opaque error inside the connect() call.
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(resolved)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA temp_store = MEMORY")
        self._conn.executescript(_SCHEMA_SQL)
        self._chat_hits = 0
        self._chat_misses = 0
        self._embed_hits = 0
        self._embed_misses = 0
        self._closed = False

    # --- chat ---------------------------------------------------------------

    def chat_key(self, provider: str, model: str, messages: Sequence[Message]) -> str:
        """Hash (provider, model, messages) into a 64-char hex key.

        Streams length-prefixed parts into SHA-256 rather than building
        the full JSON payload first. Two motivations: (1) the payload
        may contain sensitive content the redactor would scrub if it
        ever entered a log path, so reducing its dwell time in memory
        is harm-reduction; (2) zero-allocation hashing is measurably
        faster on hot retrieval paths.
        """
        return _hash_messages_for_key(provider, model, messages)

    def chat_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM chat WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self._chat_misses += 1
            return None
        self._chat_hits += 1
        return str(row["value"])

    def chat_set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO chat (key, value) VALUES (?, ?)",
                (key, value),
            )

    # --- embed --------------------------------------------------------------

    def embed_key(self, provider: str, model: str, text: str) -> str:
        """Length-prefixed key; NUL bytes in `text` cannot collide."""
        return _hash_text_for_key(provider, model, text)

    def embed_get(self, key: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM embed WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self._embed_misses += 1
            return None
        self._embed_hits += 1
        return _decode_vector(bytes(row["value"]))

    def embed_set(self, key: str, vector: Sequence[float]) -> None:
        blob = _encode_vector(vector)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embed (key, value) VALUES (?, ?)",
                (key, blob),
            )

    # --- observability ------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "chat_hits": self._chat_hits,
            "chat_misses": self._chat_misses,
            "embed_hits": self._embed_hits,
            "embed_misses": self._embed_misses,
        }

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True


class CachedChat:
    """Wrap a `ChatProvider` to consult / fill a `DiskCache` on calls."""

    def __init__(self, inner: ChatProvider, cache: DiskCache) -> None:
        self._inner = inner
        self._cache = cache
        self.name = getattr(inner, "name", "cached-chat")
        self.model = inner.model

    def chat(self, messages: Sequence[Message]) -> str:
        key = self._cache.chat_key(self.name, self.model, messages)
        cached = self._cache.chat_get(key)
        if cached is not None:
            return cached
        result = self._inner.chat(messages)
        self._cache.chat_set(key, result)
        return result

    async def achat(self, messages: Sequence[Message]) -> str:
        key = self._cache.chat_key(self.name, self.model, messages)
        cached = await asyncio.to_thread(self._cache.chat_get, key)
        if cached is not None:
            return cached
        result = await self._inner.achat(messages)
        await asyncio.to_thread(self._cache.chat_set, key, result)
        return result

    def manifest_hash(self) -> str:
        # Pass through the underlying provider's manifest hash unmodified.
        # The disk-cache is a transparent overlay: two runs that differ
        # only in cache-on/cache-off should report the same manifest row.
        return self._inner.manifest_hash()

    def close(self) -> None:
        """Close the underlying cache + delegate close() if available."""
        self._cache.close()
        inner_close = getattr(self._inner, "close", None)
        if callable(inner_close):
            inner_close()


class CachedEmbedder:
    """Wrap an `EmbeddingProvider` to consult / fill a `DiskCache`."""

    def __init__(self, inner: EmbeddingProvider, cache: DiskCache) -> None:
        self._inner = inner
        self._cache = cache
        self.name = getattr(inner, "name", "cached-embed")
        self.model = inner.model
        self.dim = inner.dim

    def _partition_cache(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float] | None], list[int], list[str]]:
        results: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_text: list[str] = []
        for i, t in enumerate(texts):
            key = self._cache.embed_key(self.name, self.model, t)
            cached = self._cache.embed_get(key)
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(t)
        return results, miss_idx, miss_text

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        results, miss_idx, miss_text = self._partition_cache(texts)
        if miss_text:
            new_vecs = self._inner.embed(miss_text)
            for idx, vec in zip(miss_idx, new_vecs, strict=True):
                results[idx] = vec
                key = self._cache.embed_key(self.name, self.model, texts[idx])
                self._cache.embed_set(key, vec)
        return [r for r in results if r is not None]

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # Move the sqlite I/O off the event-loop thread so concurrent
        # async callers don't serialize on the GIL during a tight
        # cache-lookup loop.
        results, miss_idx, miss_text = await asyncio.to_thread(
            self._partition_cache, list(texts)
        )
        if miss_text:
            new_vecs = await self._inner.aembed(miss_text)

            def _fill_misses() -> None:
                for idx, vec in zip(miss_idx, new_vecs, strict=True):
                    results[idx] = vec
                    key = self._cache.embed_key(self.name, self.model, texts[idx])
                    self._cache.embed_set(key, vec)

            await asyncio.to_thread(_fill_misses)
        return [r for r in results if r is not None]

    def embed_query(self, query: str) -> list[float]:
        """Forward asymmetric-query embedding through the cache.

        For models that expose `embed_query`, route through the
        underlying provider's `embed_query`. The cache key is namespaced
        with a `:query` suffix on the provider name so asymmetric
        models do not share slots with their document-side encoding of
        the same text. Symmetric models (no `embed_query` method) fall
        back to the document-side `embed` and reuse the same cache.
        """
        embed_query = getattr(self._inner, "embed_query", None)
        if not callable(embed_query):
            return self.embed([query])[0]
        key = self._cache.embed_key(f"{self.name}:query", self.model, query)
        cached = self._cache.embed_get(key)
        if cached is not None:
            return cached
        vec: list[float] = embed_query(query)
        self._cache.embed_set(key, vec)
        return vec

    def manifest_hash(self) -> str:
        # Transparent overlay -- preserve the underlying provider hash.
        return self._inner.manifest_hash()

    def close(self) -> None:
        """Close the underlying cache + delegate close() if available."""
        self._cache.close()
        inner_close = getattr(self._inner, "close", None)
        if callable(inner_close):
            inner_close()


def with_disk_cache(
    provider: Any,
    *,
    path: str | Path,
    allowed_root: str | Path | None = None,
) -> Any:
    """Wrap `provider` with a disk-cache appropriate to its surface.

    Detects whether the input is a chat or embedding provider by
    duck-typing on its public methods. Returns the wrapped provider;
    pass it anywhere the original provider was accepted.

    Args:
      provider: a chat or embedding provider.
      path: sqlite file path for the cache.
      allowed_root: if given, `path` must resolve to a location under
        this directory. Raises `ValueError` otherwise. Use this in
        multi-tenant or untrusted-config deployments to prevent path
        traversal smuggling a sqlite file outside the intended dir.
        Default `None` keeps the legacy "anywhere on disk" behavior.
        Set `ENGRAM_DISK_CACHE_ROOT` in the environment to apply a
        default root without changing call sites.
    """
    if allowed_root is None:
        env_root = os.environ.get("ENGRAM_DISK_CACHE_ROOT")
        if env_root:
            allowed_root = env_root
    root = Path(allowed_root) if allowed_root is not None else None
    cache = DiskCache(path, allowed_root=root)
    if hasattr(provider, "chat") and hasattr(provider, "achat"):
        return CachedChat(provider, cache)
    if hasattr(provider, "embed") and hasattr(provider, "aembed"):
        return CachedEmbedder(provider, cache)
    raise TypeError(
        f"with_disk_cache: {type(provider).__name__} is neither a chat nor an embedding provider"
    )
