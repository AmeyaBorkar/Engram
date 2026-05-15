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
# Anthropic SDK's default request timeout is 600s — far longer than
# anything an interactive workload should tolerate. A stuck endpoint
# should bubble up promptly instead of hanging the caller for ten minutes.
_DEFAULT_TIMEOUT_SECONDS: float = 60.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 10.0


def _build_sdk_kwargs(api_key: str | None, timeout: float | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if timeout is not None:
        try:
            import httpx  # noqa: PLC0415

            kwargs["timeout"] = httpx.Timeout(
                timeout, connect=_DEFAULT_CONNECT_TIMEOUT_SECONDS
            )
        except ImportError:  # pragma: no cover - httpx ships with anthropic SDK
            kwargs["timeout"] = timeout
    return kwargs


class AnthropicChat:
    """Anthropic Messages API adapter (`claude-haiku-4-5-20251001` by default)."""

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
        timeout: float | None = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.model = model
        self._max_tokens = max_tokens
        self._kwargs: dict[str, Any] = dict(completion_kwargs or {})
        sdk_kwargs = _build_sdk_kwargs(api_key, timeout)
        self._client: Anthropic = (
            client if client is not None else _anthropic_module.Anthropic(**sdk_kwargs)
        )
        self._aclient: AsyncAnthropic = (
            async_client
            if async_client is not None
            else _anthropic_module.AsyncAnthropic(**sdk_kwargs)
        )

    def chat(self, messages: Sequence[Message]) -> str:
        kwargs = self._build_kwargs(messages)
        resp = self._client.messages.create(**kwargs)
        return _join_text_blocks(resp.content, self.model)

    async def achat(self, messages: Sequence[Message]) -> str:
        kwargs = self._build_kwargs(messages)
        resp = await self._aclient.messages.create(**kwargs)
        return _join_text_blocks(resp.content, self.model)

    def manifest_hash(self) -> str:
        kwargs_blob = json.dumps(self._kwargs, sort_keys=True, default=str)
        h = hashlib.sha256(kwargs_blob.encode("utf-8")).hexdigest()[:16]
        return f"anthropic-chat/{self.model}/max_tokens={self._max_tokens}/{h}"

    def _build_kwargs(self, messages: Sequence[Message]) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in non_system],
            **self._kwargs,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        return kwargs


def _join_text_blocks(content: Any, model: str) -> str:
    """Collapse Anthropic's typed content blocks into plain text.

    Tool-use and other non-text blocks are dropped at this layer; later
    stages that need them will use a different surface.

    Raises RuntimeError if `content` is empty (the upstream model returned
    a degenerate payload — content-filter block, server error, or an
    Anthropic-compatible endpoint quirk) so the caller doesn't get a
    silently empty string and treat it as a successful empty response.
    """
    if not content:
        raise RuntimeError(
            f"Anthropic chat returned empty content for model {model!r}; "
            "the response may have been content-filter blocked or the "
            "endpoint is misbehaving."
        )
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)
