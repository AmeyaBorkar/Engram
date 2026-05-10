"""OpenAI provider adapters.

Public submodule (`from engram.providers.openai import OpenAIEmbedder, OpenAIChat`).
Behind the `[openai]` extra. Importing this module without the `openai`
package installed raises a clear, actionable `ImportError`.

Both adapters accept either an explicit `client` / `async_client` (handy
for tests using `unittest.mock`) or construct their own from `api_key`.
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
    """OpenAI embeddings adapter (`text-embedding-3-small` by default)."""

    name: str = "openai-embed"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        *,
        api_key: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.model = model
        self.dim = dim
        self._client: OpenAI = (
            client if client is not None else _openai_module.OpenAI(api_key=api_key)
        )
        self._aclient: AsyncOpenAI = (
            async_client
            if async_client is not None
            else _openai_module.AsyncOpenAI(api_key=api_key)
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dim != _native_dim(self.model):
            kwargs["dimensions"] = self.dim
        resp = self._client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in resp.data]

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dim != _native_dim(self.model):
            kwargs["dimensions"] = self.dim
        resp = await self._aclient.embeddings.create(**kwargs)
        return [list(item.embedding) for item in resp.data]

    def manifest_hash(self) -> str:
        return f"openai-embed/{self.model}/dim={self.dim}/v1"


class OpenAIChat:
    """OpenAI chat-completions adapter (`gpt-4o-mini` by default)."""

    name: str = "openai-chat"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        completion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._kwargs: dict[str, Any] = dict(completion_kwargs or {})
        self._client: OpenAI = (
            client if client is not None else _openai_module.OpenAI(api_key=api_key)
        )
        self._aclient: AsyncOpenAI = (
            async_client
            if async_client is not None
            else _openai_module.AsyncOpenAI(api_key=api_key)
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
        return f"openai-chat/{self.model}/{h}"


def _native_dim(model: str) -> int:
    """Native embedding dim per OpenAI model. Used to skip the `dimensions` arg
    when the user wants the natural size (some endpoints reject explicit native dim)."""
    return {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }.get(model, -1)


def _to_openai_messages(messages: Sequence[Message]) -> Any:
    # Returned as `Any` so the OpenAI SDK's strict `ChatCompletionMessageParam`
    # union (which can't be expressed cleanly from outside the SDK) is satisfied.
    return [{"role": m.role, "content": m.content} for m in messages]
