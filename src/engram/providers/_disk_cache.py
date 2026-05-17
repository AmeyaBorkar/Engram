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
import os
import sqlite3
import struct
import threading
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from engram.providers._message import Message

if TYPE_CHECKING:
    from engram.providers._protocols import ChatProvider, EmbeddingProvider


_SCHEMA_VERSION = 1
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

# Magic prefix that distinguishes the packed-binary embed format from the
# legacy JSON-text format.  Eight bytes so the chance of a legit
# UTF-8-encoded JSON list starting with this prefix is negligible —
# `b"EGRMVEC1"` is not valid JSON, and an embed_get() that sees this
# prefix decodes via numpy `frombuffer` instead of `json.loads`.
_EMBED_BINARY_MAGIC: bytes = b"EGRMVEC1"
# Chunk size for `WHERE key IN (?,?,...)` lookups.  SQLite's default
# parameter limit is 999 / 32766 depending on build; 500 stays
# comfortably under either while amortizing the round-trip cost.
_EMBED_GET_MANY_CHUNK: int = 500


# Module-level handle registry — `with_disk_cache(provider, path=...)` is
# called once per chat provider and once per embedder, and both calls
# point at the same path.  Without de-duplication the bench opens two
# sqlite connections + two WAL files against the same file, races on
# schema creation, and counts hits twice in `.stats`.  The registry
# memoizes by the resolved absolute path so any path-spelling
# difference (`./cache.db` vs `cache.db`) collapses to a single handle.
_HANDLE_LOCK = threading.Lock()
_HANDLES: dict[str, DiskCache] = {}


def _resolve_cache_path(path: str | Path) -> Path:
    """Normalize a user path to an absolute, resolved `Path`.

    Centralized so the registry key, the traversal guard, and the
    sqlite open all use the exact same string.  `Path.resolve(strict=False)`
    on Windows handles drive letter case folding and `~` expansion via
    `Path.expanduser` first.
    """
    return Path(path).expanduser().resolve()


class DiskCache:
    """SQLite-backed (key -> value) cache for provider responses.

    Two columns: `key` (SHA-256 hex) and `value` (text for chat,
    json blob for embed vectors). Entries are never evicted -- the
    bench corpus is bounded, and the user can rm the file when they
    want to start fresh.

    `allowed_root` optionally pins the cache file's resolved location
    under a specific directory tree.  When set, any attempt to open a
    cache outside that subtree raises `ValueError` — protects bench
    harnesses that accept a cache path from user input from being
    coerced into writing somewhere unexpected.  The env override
    `ENGRAM_DISK_CACHE_ROOT` applies the same constraint globally
    (caller-provided `allowed_root=` wins when both are present).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        allowed_root: str | Path | None = None,
    ) -> None:
        resolved = _resolve_cache_path(path)
        env_root = os.environ.get("ENGRAM_DISK_CACHE_ROOT")
        root: Path | None
        if allowed_root is not None:
            root = _resolve_cache_path(allowed_root)
        elif env_root:
            root = _resolve_cache_path(env_root)
        else:
            root = None
        if root is not None:
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"DiskCache path {str(resolved)!r} is not under "
                    f"allowed_root {str(root)!r}"
                ) from exc
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
        # Record the on-disk schema version so a future schema bump can
        # detect old caches and migrate / blow them away.  Until a
        # schema-aware migration arrives, a mismatch is just a warning.
        existing_ver = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if existing_ver == 0:
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif existing_ver != _SCHEMA_VERSION:
            warnings.warn(
                f"DiskCache at {self._path!r} has user_version="
                f"{existing_ver}, expected {_SCHEMA_VERSION}; results "
                "may be stale.  Delete the file to recreate.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._chat_hits = 0
        self._chat_misses = 0
        self._embed_hits = 0
        self._embed_misses = 0

    @property
    def path(self) -> str:
        return self._path

    # --- chat ---------------------------------------------------------------

    def chat_key(self, provider: str, model: str, messages: Sequence[Message]) -> str:
        """SHA-256 over (provider, model, messages) without materializing JSON.

        Streams each message through `hashlib.update(...)` so a long
        haystack message doesn't allocate a full JSON blob in memory
        before hashing.  Components are length-prefixed so a `provider`
        value that ends with the same bytes a `model` value starts with
        cannot collide.
        """
        h = hashlib.sha256()
        _hash_lp(h, provider.encode("utf-8"))
        _hash_lp(h, model.encode("utf-8"))
        # Count of messages as 8-byte big-endian so an empty trailing
        # message can't be confused with the absence of a message.
        h.update(struct.pack(">Q", len(messages)))
        for m in messages:
            _hash_lp(h, m.role.encode("utf-8"))
            _hash_lp(h, m.content.encode("utf-8"))
        return h.hexdigest()

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
        """SHA-256 over (provider, model, text), length-prefixed per component.

        The previous `\\x00`-delimited concatenation collided whenever
        any input contained a literal NUL byte (rare for OpenAI prompts,
        possible for local pipelines that pre-tokenize through binary
        wire formats).  Length-prefixing each component eliminates the
        collision class entirely — `("ab", "c", "d")` cannot hash
        to the same value as `("a", "bc", "d")`.
        """
        h = hashlib.sha256()
        _hash_lp(h, provider.encode("utf-8"))
        _hash_lp(h, model.encode("utf-8"))
        _hash_lp(h, text.encode("utf-8"))
        return h.hexdigest()

    def embed_get(self, key: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM embed WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self._embed_misses += 1
                return None
            decoded = _decode_embed_blob(row["value"])
            if decoded is None:
                # Treat any deserialization failure as a cache miss.  A
                # truncated / corrupt blob (post-crash, manual file
                # tampering, sqlite page corruption) used to propagate
                # UnicodeDecodeError / JSONDecodeError uncaught and brick
                # the whole bench; instead we drop the row, log nothing
                # (caller will re-embed), and bump the miss counter.
                self._embed_misses += 1
                return None
            self._embed_hits += 1
        return decoded

    def embed_get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        """Batch variant of `embed_get`.

        Returns only the keys actually present in the cache.  Splits
        large key lists into chunks of 500 to stay under SQLite's
        per-statement parameter cap.  Hits bump the embed-hit counter
        once per matched key; misses are *not* counted here because
        the caller is partitioning into hit/miss sets and only the
        eventual miss-then-fill code path should count a true miss.
        """
        if not keys:
            return {}
        out: dict[str, list[float]] = {}
        seen: set[str] = set()
        with self._lock:
            for i in range(0, len(keys), _EMBED_GET_MANY_CHUNK):
                chunk = list(keys[i : i + _EMBED_GET_MANY_CHUNK])
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT key, value FROM embed WHERE key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    k = str(row["key"])
                    if k in seen:
                        continue
                    decoded = _decode_embed_blob(row["value"])
                    if decoded is None:
                        continue
                    out[k] = decoded
                    seen.add(k)
            self._embed_hits += len(out)
        return out

    def embed_set(self, key: str, vector: Sequence[float]) -> None:
        blob = _encode_embed_blob(vector)
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
        # Drop from the registry so a fresh open after close returns a
        # new connection instead of handing back the closed one.
        with _HANDLE_LOCK:
            for k, v in list(_HANDLES.items()):
                if v is self:
                    del _HANDLES[k]


def _hash_lp(h: "hashlib._Hash", data: bytes) -> None:
    """Update `h` with a length-prefixed byte string.

    Eight-byte big-endian length followed by the bytes.  Centralized so
    `chat_key` and `embed_key` agree on the format.
    """
    h.update(struct.pack(">Q", len(data)))
    h.update(data)


def _encode_embed_blob(vector: Sequence[float]) -> bytes:
    """Pack a vector as `magic | little-endian float64[*]`.

    Roughly 4x smaller than the JSON-text format (each float8 is 8
    bytes vs ~17 characters of `"-0.123456789,"`) and 5-10x faster to
    serialize / deserialize at corpus scale.  `embed_get` falls back
    to the legacy JSON parser when the magic prefix is absent so
    pre-existing cache files keep working.
    """
    arr = np.asarray(vector, dtype="<f8")
    return _EMBED_BINARY_MAGIC + arr.tobytes()


def _decode_embed_blob(value: Any) -> list[float] | None:
    """Decode a stored embed blob; return None on any deserialization issue.

    Handles three shapes:
      * `magic | float64 bytes` — the current packed-binary format
        produced by `_encode_embed_blob`.
      * raw JSON text (legacy) — caches written by an older Engram
        version.  Auto-detected by the absence of the magic prefix.
      * anything else — returns None (caller treats as a cache miss).
    """
    if value is None:
        return None
    blob: bytes
    if isinstance(value, (bytes, bytearray, memoryview)):
        blob = bytes(value)
    elif isinstance(value, str):
        # Legacy text-encoded JSON path; SQLite stored the JSON
        # directly when the column was TEXT.
        try:
            decoded = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return _coerce_vector_list(decoded)
    else:
        return None

    if blob.startswith(_EMBED_BINARY_MAGIC):
        payload = blob[len(_EMBED_BINARY_MAGIC) :]
        if len(payload) % 8 != 0:
            return None
        try:
            arr = np.frombuffer(payload, dtype="<f8")
        except ValueError:
            return None
        return [float(x) for x in arr]
    # Legacy JSON-text-as-bytes path (the pre-packed-binary format
    # written by Engram <= 0.2.1).  Decode and validate.
    try:
        decoded = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return _coerce_vector_list(decoded)


def _coerce_vector_list(decoded: Any) -> list[float] | None:
    if not isinstance(decoded, list):
        return None
    out: list[float] = []
    for x in decoded:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        out.append(float(x))
    return out


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
        # One `WHERE key IN (?, ?, …)` batch lookup instead of N
        # per-text round trips.  At chunk_size=4096 this turns ~4096
        # sqlite calls into ~9.  Misses are partitioned the same way
        # they used to be so the rest of `embed` / `aembed` is
        # unchanged.
        results: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_text: list[str] = []
        keys: list[str] = [
            self._cache.embed_key(self.name, self.model, t) for t in texts
        ]
        found = self._cache.embed_get_many(keys)
        for i, key in enumerate(keys):
            vec = found.get(key)
            if vec is not None:
                results[i] = vec
            else:
                miss_idx.append(i)
                miss_text.append(texts[i])
        # Account for the misses now that we know the partition; the
        # legacy per-row `embed_get` flow counted them inline.
        self._cache._embed_misses += len(miss_idx)  # noqa: SLF001 - sibling helper
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
    allowed_root: str | Path | None = None,
) -> Any:
    """Wrap `provider` with a disk-cache appropriate to its surface.

    Detects whether the input is a chat or embedding provider by
    duck-typing on its public methods. Returns the wrapped provider;
    pass it anywhere the original provider was accepted.

    Repeated calls with the same path return wrappers backed by the
    same `DiskCache` connection — no duplicate sqlite handles, no
    racing WAL writers, no double-counted hits.
    """
    cache = _get_or_create_cache(path, allowed_root=allowed_root)
    if hasattr(provider, "chat") and hasattr(provider, "achat"):
        return CachedChat(provider, cache)
    if hasattr(provider, "embed") and hasattr(provider, "aembed"):
        return CachedEmbedder(provider, cache)
    raise TypeError(
        f"with_disk_cache: {type(provider).__name__} is neither a chat nor an embedding provider"
    )


def _get_or_create_cache(
    path: str | Path,
    *,
    allowed_root: str | Path | None,
) -> DiskCache:
    """Return a memoized `DiskCache` for `path` (resolved absolute).

    Two concurrent `with_disk_cache` calls on the same path return the
    same handle.  Construction is serialized under a module-level lock
    so a race between two threads ends with exactly one DiskCache
    open against the file.
    """
    resolved = str(_resolve_cache_path(path))
    with _HANDLE_LOCK:
        existing = _HANDLES.get(resolved)
        if existing is not None:
            return existing
        cache = DiskCache(path, allowed_root=allowed_root)
        _HANDLES[resolved] = cache
        return cache
