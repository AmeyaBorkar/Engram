"""Anthropic provider adapter.

Public submodule (`from engram.providers.anthropic import AnthropicChat`).
Behind the `[anthropic]` extra. Importing without the `anthropic` package
installed raises a clear, actionable `ImportError`.

Anthropic's Messages API differs from OpenAI's chat-completions in two
ways we paper over:

  - System messages are a top-level `system` argument rather than a role
    in the messages list. We concatenate any `role="system"` messages
    (separated by blank lines) and pass them through as `system`.
  - Responses are a list of typed content blocks. We concatenate the
    text blocks and ignore the rest (tool use etc. land in later stages).

`close()` / `aclose()` release the underlying SDK HTTP pools; sync and
async context-manager forms are supported.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from engram.providers._message import Message

try:
    import anthropic as _anthropic_module
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "anthropic is not installed. Install with: pip install 'engram[anthropic]'"
    ) from _exc

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic


_DEFAULT_MAX_TOKENS = 1024


class AnthropicChat:
    """Anthropic Messages API adapter (`claude-haiku-4-5-20251001` by default).

    `completion_kwargs` are merged into every `.chat()` / `.achat()`
    call. Pass per-call overrides via the `extra` arg on either method
    -- they win over the constructor defaults when keys collide.

    `close()` / `aclose()` release the underlying HTTP pool.
    """

    name: str = "anthropic-chat"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        api_key: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: Anthropic | None = None,
        async_client: AsyncAnthropic | None = None,
        completion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.model = model
        self._max_tokens = max_tokens
        self._kwargs: dict[str, Any] = dict(completion_kwargs or {})
        self._client: Anthropic = (
            client if client is not None else _anthropic_module.Anthropic(api_key=api_key)
        )
        self._aclient: AsyncAnthropic = (
            async_client
            if async_client is not None
            else _anthropic_module.AsyncAnthropic(api_key=api_key)
        )
        self._closed = False

    def chat(
        self,
        messages: Sequence[Message],
        *,
        extra: dict[str, Any] | None = None,
    ) -> str:
        kwargs = self._build_kwargs(messages, extra=extra)
        resp = self._client.messages.create(**kwargs)
        return _join_text_blocks(resp.content)

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        extra: dict[str, Any] | None = None,
    ) -> str:
        kwargs = self._build_kwargs(messages, extra=extra)
        resp = await self._aclient.messages.create(**kwargs)
        return _join_text_blocks(resp.content)

    def manifest_hash(self) -> str:
        kwargs_blob = json.dumps(self._kwargs, sort_keys=True, default=str)
        h = hashlib.sha256(kwargs_blob.encode("utf-8")).hexdigest()[:16]
        return f"anthropic-chat/{self.model}/max_tokens={self._max_tokens}/{h}"

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

    def __enter__(self) -> AnthropicChat:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> AnthropicChat:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in non_system],
            **self._kwargs,
        }
        if extra:
            kwargs.update(extra)
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        return kwargs


def _join_text_blocks(content: Any) -> str:
    """Collapse Anthropic's typed content blocks into plain text.

    Tool-use and other non-text blocks are dropped at this layer; later
    stages that need them will use a different surface.
    """
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _close_sdk_client(client: Any) -> None:
    """Best-effort sync close of an Anthropic SDK client."""
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # noqa: BLE001 - defensive: SDK shape may evolve
            pass


async def _close_sdk_async_client(client: Any) -> None:
    """Best-effort async close of an Anthropic async SDK client."""
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            result = closer()
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - defensive
            pass


def _close_sdk_async_client_sync(client: Any) -> None:
    """Sync close for an async SDK client when not in an event loop."""
    closer = getattr(client, "close", None)
    if not callable(closer):
        return
    try:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(_close_sdk_async_client(client))
            except Exception:  # noqa: BLE001 - defensive
                pass
    except Exception:  # noqa: BLE001 - defensive
        pass
