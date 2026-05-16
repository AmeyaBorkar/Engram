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

Wire it in via `with_disk_cache(provider, path=...)`; the returned
wrapper proxies the provider's surface and consults / fills the cache
on every call.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
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


class DiskCache:
    """SQLite-backed (key -> value) cache for provider responses.

    Two columns: `key` (SHA-256 hex) and `value` (text for chat,
    json blob for embed vectors). Entries are never evicted -- the
    bench corpus is bounded, and the user can rm the file when they
    want to start fresh.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
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

    # --- chat ---------------------------------------------------------------

    def chat_key(self, provider: str, model: str, messages: Sequence[Message]) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "messages": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chat_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM chat WHERE key = ?", (key,)
            ).fetchone()
            # Counter updates under the same lock that protects the
            # sqlite operation so two concurrent chat_get calls don't
            # race on the integer increment (CPython int += is not
            # atomic across threads under contention).
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
        payload = f"{provider}\x00{model}\x00{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def embed_get(self, key: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM embed WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self._embed_misses += 1
                return None
            # Treat any deserialization failure as a cache miss.  A
            # truncated / corrupt blob (post-crash, manual file tampering,
            # sqlite page corruption) used to propagate UnicodeDecodeError
            # / JSONDecodeError uncaught and brick the whole bench;
            # instead we drop the row, log nothing (caller will re-embed),
            # and bump the miss counter.
            try:
                decoded = json.loads(row["value"].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._embed_misses += 1
                return None
            if not isinstance(decoded, list) or not all(
                isinstance(x, (int, float)) for x in decoded
            ):
                self._embed_misses += 1
                return None
            self._embed_hits += 1
        return [float(x) for x in decoded]

    def embed_set(self, key: str, vector: Sequence[float]) -> None:
        blob = json.dumps(list(vector)).encode("utf-8")
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
            self._conn.close()


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
        cached = self._cache.chat_get(key)
        if cached is not None:
            return cached
        result = await self._inner.achat(messages)
        self._cache.chat_set(key, result)
        return result

    def manifest_hash(self) -> str:
        return f"{self._inner.manifest_hash()}/disk-cached"


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
        results, miss_idx, miss_text = self._partition_cache(texts)
        if miss_text:
            new_vecs = await self._inner.aembed(miss_text)
            for idx, vec in zip(miss_idx, new_vecs, strict=True):
                results[idx] = vec
                key = self._cache.embed_key(self.name, self.model, texts[idx])
                self._cache.embed_set(key, vec)
        return [r for r in results if r is not None]

    def embed_query(self, query: str) -> list[float]:
        """Forward asymmetric-query embedding through the cache.

        Symmetric models (no asymmetric prompt) fall back to `embed`.
        Asymmetric models go through the underlying provider's
        `embed_query` and store the result keyed with a "query"
        prefix so it does not collide with the document-side cache
        entry for the same text.
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
        return f"{self._inner.manifest_hash()}/disk-cached"


def with_disk_cache(
    provider: Any,
    *,
    path: str | Path,
) -> Any:
    """Wrap `provider` with a disk-cache appropriate to its surface.

    Detects whether the input is a chat or embedding provider by
    duck-typing on its public methods. Returns the wrapped provider;
    pass it anywhere the original provider was accepted.
    """
    cache = DiskCache(path)
    if hasattr(provider, "chat") and hasattr(provider, "achat"):
        return CachedChat(provider, cache)
    if hasattr(provider, "embed") and hasattr(provider, "aembed"):
        return CachedEmbedder(provider, cache)
    raise TypeError(
        f"with_disk_cache: {type(provider).__name__} is neither a chat nor an embedding provider"
    )
