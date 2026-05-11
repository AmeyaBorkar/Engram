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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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
    Gemini, BGE) reject it. The default auto-detects: known third-party
    models with a recognized native dim skip `dimensions`; everything
    else sends it whenever `dim != native_dim`.
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
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.model = model
        self.dim = dim
        self._base_url = base_url
        self._default_headers = dict(default_headers) if default_headers else None
        # If the caller didn't decide, send `dimensions` only for models
        # whose backend accepts it. OpenAI native and OR's `openai/*`
        # routes do; Qwen / Gemini / BGE typically reject the kwarg.
        if send_dimensions is None:
            send_dimensions = model in _SUPPORTS_DIMENSIONS
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

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self._send_dimensions and self.dim != _native_dim(self.model):
            kwargs["dimensions"] = self.dim
        resp = self._client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in resp.data]

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self._send_dimensions and self.dim != _native_dim(self.model):
            kwargs["dimensions"] = self.dim
        resp = await self._aclient.embeddings.create(**kwargs)
        return [list(item.embedding) for item in resp.data]

    def manifest_hash(self) -> str:
        suffix = f"/base={self._base_url}" if self._base_url else ""
        return f"openai-embed/{self.model}/dim={self.dim}{suffix}/v1"


class OpenAIChat:
    """OpenAI chat-completions adapter (`gpt-4o-mini` by default).

    Pass `base_url` to point at any OpenAI-compatible chat endpoint
    (Moonshot/Kimi at `https://api.moonshot.ai/v1`, OpenRouter at
    `https://openrouter.ai/api/v1`, Together, vLLM, LM Studio, ...).
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

    def chat(self, messages: Sequence[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),
            **self._kwargs,
        )
        content = resp.choices[0].message.content
        return content if content is not None else ""

    async def achat(self, messages: Sequence[Message]) -> str:
        resp = await self._aclient.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),
            **self._kwargs,
        )
        content = resp.choices[0].message.content
        return content if content is not None else ""

    def manifest_hash(self) -> str:
        kwargs_blob = json.dumps(self._kwargs, sort_keys=True, default=str)
        h = hashlib.sha256(kwargs_blob.encode("utf-8")).hexdigest()[:16]
        suffix = f"/base={self._base_url}" if self._base_url else ""
        return f"openai-chat/{self.model}{suffix}/{h}"


# Native output dim per embedding model. -1 means "unknown"; the
# OpenAIEmbedder falls back to the caller-supplied dim and skips the
# `dimensions=` kwarg.
_OPENAI_NATIVE_EMBED_DIMS: dict[str, int] = {
    # OpenAI direct.
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # OpenRouter -- `openai/*` routes to OpenAI (supports dimensions=).
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-3-small": 1536,
    # OpenRouter -- third-party models (DO NOT support dimensions=).
    "qwen/qwen3-embedding-8b": 4096,
    "qwen/qwen3-embedding-4b": 2560,
    "qwen/qwen3-embedding-0.6b": 1024,
    "baai/bge-m3": 1024,
    "google/gemini-embedding-001": 3072,
}

# Models whose backend accepts the `dimensions=` truncation arg.
# Membership drives the default `send_dimensions` choice in
# OpenAIEmbedder so an OpenRouter caller doesn't send an arg Qwen /
# Gemini / BGE will reject.
_SUPPORTS_DIMENSIONS: set[str] = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
}


def _native_dim(model: str) -> int:
    """Native embedding dim per model; -1 if unknown."""
    return _OPENAI_NATIVE_EMBED_DIMS.get(model, -1)


def _to_openai_messages(messages: Sequence[Message]) -> Any:
    # Returned as `Any` so the OpenAI SDK's strict `ChatCompletionMessageParam`
    # union (which can't be expressed cleanly from outside the SDK) is satisfied.
    return [{"role": m.role, "content": m.content} for m in messages]
