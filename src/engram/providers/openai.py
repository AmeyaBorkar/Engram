"""OpenAI provider adapters.

Public submodule (`from engram.providers.openai import OpenAIEmbedder, OpenAIChat`).
Behind the `[openai]` extra. Importing this module without the `openai`
package installed raises a clear, actionable `ImportError`.

Both adapters accept either an explicit `client` / `async_client` (handy
for tests using `unittest.mock`) or construct their own from `api_key`.

OpenAI-compatible endpoints (Moonshot/Kimi, OpenRouter, Together, vLLM,
LM Studio, ...) work via the `base_url` parameter -- the SDK speaks the
same wire protocol, and any provider that accepts an OpenAI-shaped
request fits behind these adapters. Set `base_url` and `api_key`
appropriately and the rest of Engram doesn't need to know which
endpoint is on the other side. Manifest hashes include `base_url` so
two runs at different endpoints stay distinguishable.

Both adapters expose `close()` (and async `aclose()`) so the underlying
HTTP pools release on shutdown; they are also usable as context
managers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from engram.providers._message import Message

try:
    import openai as _openai_module
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "openai is not installed. Install with: pip install 'engram[openai]'"
    ) from _exc

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


# Default per-batch chunk size when the input list exceeds the
# configured chunk budget. 2048 is conservative for the OpenAI
# embeddings API (current published limit is 2048 items per request).
_DEFAULT_EMBED_CHUNK_SIZE = 2048


@dataclass(frozen=True)
class _EmbedModelInfo:
    """Single source of truth for per-model embedding behavior.

    `native_dim`: the model's native output dimensionality.
    `supports_dimensions`: whether the backend accepts the
      `dimensions=` truncation arg on the embeddings call.
    """

    native_dim: int
    supports_dimensions: bool


# Per-model native dim + `dimensions=`-arg support. Consolidates what
# used to live in two parallel structures (`_OPENAI_NATIVE_EMBED_DIMS`
# + `_SUPPORTS_DIMENSIONS`) -- the previous shape made it possible to
# add a model in one table and forget the other, leaving the runtime
# silently wrong on an OpenRouter route. Adding an entry here updates
# both invariants in one place.
_EMBED_MODELS: dict[str, _EmbedModelInfo] = {
    # OpenAI direct -- supports `dimensions=`.
    "text-embedding-3-small": _EmbedModelInfo(1536, True),
    "text-embedding-3-large": _EmbedModelInfo(3072, True),
    "text-embedding-ada-002": _EmbedModelInfo(1536, True),
    # OpenRouter -- `openai/*` routes to OpenAI and inherits the support.
    "openai/text-embedding-3-large": _EmbedModelInfo(3072, True),
    "openai/text-embedding-3-small": _EmbedModelInfo(1536, True),
    # OpenRouter -- third-party models (DO NOT support dimensions=).
    "qwen/qwen3-embedding-8b": _EmbedModelInfo(4096, False),
    "qwen/qwen3-embedding-4b": _EmbedModelInfo(2560, False),
    "qwen/qwen3-embedding-0.6b": _EmbedModelInfo(1024, False),
    "baai/bge-m3": _EmbedModelInfo(1024, False),
    "google/gemini-embedding-001": _EmbedModelInfo(3072, False),
}


def _native_dim(model: str) -> int:
    """Native embedding dim per model; -1 if unknown."""
    info = _EMBED_MODELS.get(model)
    return info.native_dim if info is not None else -1


def _supports_dimensions(model: str) -> bool:
    """True iff the backend accepts the `dimensions=` truncation arg."""
    info = _EMBED_MODELS.get(model)
    return info.supports_dimensions if info is not None else False


def _to_openai_messages(messages: Sequence[Message]) -> Any:
    # Returned as `Any` so the OpenAI SDK's strict `ChatCompletionMessageParam`
    # union (which can't be expressed cleanly from outside the SDK) is satisfied.
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAIEmbedder:
    """OpenAI embeddings adapter (`text-embedding-3-small` by default).

    Pass `base_url` to point at any OpenAI-compatible embeddings endpoint
    (OpenRouter, Together, vLLM, ...); leaving it None uses OpenAI's
    official endpoint. `default_headers` flows into the SDK client and
    is the right place for OpenRouter's optional ranking headers
    (`HTTP-Referer`, `X-Title`).

    `send_dimensions` controls whether the `dimensions=` arg is sent on
    the embeddings call. OpenAI native models (text-embedding-3-*)
    support truncation via this arg; many third-party models (Qwen,
    Gemini, BGE) reject it. The default auto-detects via the
    `_EMBED_MODELS` table: known models with the support flag set send
    `dimensions=`; everything else does not.

    `chunk_size` caps the items-per-request the SDK call sends. The
    OpenAI embeddings API has a 2048-item cap; large input lists are
    sliced and re-stitched here so callers can hand in arbitrary
    batches.

    `close()` releases the SDK's HTTP pool. The class is also usable as
    a context manager (`with OpenAIEmbedder(...) as e: ...`); `async with`
    is supported too.
    """

    name: str = "openai-embed"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        default_headers: dict[str, str] | None = None,
        send_dimensions: bool | None = None,
        chunk_size: int = _DEFAULT_EMBED_CHUNK_SIZE,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.model = model
        self.dim = dim
        self._base_url = base_url
        self._default_headers = dict(default_headers) if default_headers else None
        self._chunk_size = chunk_size
        # If the caller didn't decide, send `dimensions` only for models
        # whose backend accepts it. OpenAI native and OR's `openai/*`
        # routes do; Qwen / Gemini / BGE typically reject the kwarg.
        if send_dimensions is None:
            send_dimensions = _supports_dimensions(model)
        self._send_dimensions = send_dimensions
        sdk_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            sdk_kwargs["base_url"] = base_url
        if self._default_headers is not None:
            sdk_kwargs["default_headers"] = self._default_headers
        self._client: OpenAI = client if client is not None else _openai_module.OpenAI(**sdk_kwargs)
        self._aclient: AsyncOpenAI = (
            async_client if async_client is not None else _openai_module.AsyncOpenAI(**sdk_kwargs)
        )
        self._closed = False

    def _build_kwargs(self, chunk: list[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "input": chunk}
        if self._send_dimensions and self.dim != _native_dim(self.model):
            kwargs["dimensions"] = self.dim
        return kwargs

    def _chunks(self, texts: Sequence[str]) -> list[list[str]]:
        text_list = list(texts)
        if len(text_list) <= self._chunk_size:
            return [text_list] if text_list else []
        return [
            text_list[i : i + self._chunk_size]
            for i in range(0, len(text_list), self._chunk_size)
        ]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for chunk in self._chunks(texts):
            resp = self._client.embeddings.create(**self._build_kwargs(chunk))
            out.extend(list(item.embedding) for item in resp.data)
        return out

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for chunk in self._chunks(texts):
            resp = await self._aclient.embeddings.create(**self._build_kwargs(chunk))
            out.extend(list(item.embedding) for item in resp.data)
        return out

    def manifest_hash(self) -> str:
        suffix = f"/base={self._base_url}" if self._base_url else ""
        return f"openai-embed/{self.model}/dim={self.dim}{suffix}/v1"

    def close(self) -> None:
        """Release the underlying SDK HTTP pools."""
        if self._closed:
            return
        self._closed = True
        _close_sdk_client(self._client)
        # The async client's `.close()` returns a coroutine; if we are
        # not in an event loop here, fire-and-forget via a fresh loop
        # so we still tear down sockets. The common case is that the
        # caller will use `aclose()` or `async with`; this is a safety
        # net for callers who only know `close()`.
        _close_sdk_async_client_sync(self._aclient)

    async def aclose(self) -> None:
        """Async variant of `close()`."""
        if self._closed:
            return
        self._closed = True
        _close_sdk_client(self._client)
        await _close_sdk_async_client(self._aclient)

    def __enter__(self) -> OpenAIEmbedder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> OpenAIEmbedder:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class OpenAIChat:
    """OpenAI chat-completions adapter (`gpt-4o-mini` by default).

    Pass `base_url` to point at any OpenAI-compatible chat endpoint
    (Moonshot/Kimi at `https://api.moonshot.ai/v1`, OpenRouter at
    `https://openrouter.ai/api/v1`, Together, vLLM, LM Studio, ...).

    `completion_kwargs` are merged into every `.chat()` / `.achat()`
    call. Pass per-call overrides via the `extra` arg on either method
    -- they win over the constructor defaults when keys collide.

    `close()` / `aclose()` release the underlying HTTP pool.
    """

    name: str = "openai-chat"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self._kwargs: dict[str, Any] = dict(completion_kwargs or {})
        self._base_url = base_url
        self._default_headers = dict(default_headers) if default_headers else None
        sdk_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            sdk_kwargs["base_url"] = base_url
        if self._default_headers is not None:
            sdk_kwargs["default_headers"] = self._default_headers
        self._client: OpenAI = client if client is not None else _openai_module.OpenAI(**sdk_kwargs)
        self._aclient: AsyncOpenAI = (
            async_client if async_client is not None else _openai_module.AsyncOpenAI(**sdk_kwargs)
        )
        self._closed = False

    def _merge_kwargs(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        if not extra:
            return self._kwargs
        merged = dict(self._kwargs)
        merged.update(extra)
        return merged

    def chat(
        self,
        messages: Sequence[Message],
        *,
        extra: dict[str, Any] | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),
            **self._merge_kwargs(extra),
        )
        content = resp.choices[0].message.content
        return content if content is not None else ""

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        extra: dict[str, Any] | None = None,
    ) -> str:
        resp = await self._aclient.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),
            **self._merge_kwargs(extra),
        )
        content = resp.choices[0].message.content
        return content if content is not None else ""

    def manifest_hash(self) -> str:
        kwargs_blob = json.dumps(self._kwargs, sort_keys=True, default=str)
        h = hashlib.sha256(kwargs_blob.encode("utf-8")).hexdigest()[:16]
        suffix = f"/base={self._base_url}" if self._base_url else ""
        return f"openai-chat/{self.model}{suffix}/{h}"

    def close(self) -> None:
        """Release the underlying SDK HTTP pools."""
        if self._closed:
            return
        self._closed = True
        _close_sdk_client(self._client)
        _close_sdk_async_client_sync(self._aclient)

    async def aclose(self) -> None:
        """Async variant of `close()`."""
        if self._closed:
            return
        self._closed = True
        _close_sdk_client(self._client)
        await _close_sdk_async_client(self._aclient)

    def __enter__(self) -> OpenAIChat:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> OpenAIChat:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _close_sdk_client(client: Any) -> None:
    """Best-effort sync close of an OpenAI SDK client."""
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # noqa: BLE001 - defensive: SDK shape may evolve
            pass


async def _close_sdk_async_client(client: Any) -> None:
    """Best-effort async close of an OpenAI async SDK client."""
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            result = closer()
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - defensive
            pass


def _close_sdk_async_client_sync(client: Any) -> None:
    """Sync close for an async SDK client when not in an event loop.

    Tries asyncio.run() in a fresh loop -- if the caller already has
    a running loop we silently skip; that path should use `aclose()`.
    """
    closer = getattr(client, "close", None)
    if not callable(closer):
        return
    try:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: safe to spin one up to drive the coroutine.
            try:
                asyncio.run(_close_sdk_async_client(client))
            except Exception:  # noqa: BLE001 - defensive
                pass
    except Exception:  # noqa: BLE001 - defensive
        pass
