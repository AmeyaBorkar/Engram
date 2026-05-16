"""LlamaIndex integration.

`EngramLlamaIndexMemory` wraps an Engram `Memory` and exposes it as a
LlamaIndex `BaseMemory`-shaped surface. LlamaIndex memory plugs into
their chat engines / agents and gets called per chat turn.

The class deliberately doesn't subclass LlamaIndex types -- it
duck-types the interface. That keeps `import engram.integrations.
llamaindex` from pulling in LlamaIndex unless the caller has it
installed (LlamaIndex's pydantic-v2 hierarchy is heavy).

API methods (mirroring LlamaIndex's BaseMemory shape):

  * `put(message)` -- observe the message content as an event.
  * `get(input=None)` -- retrieve top-k memories formatted as
    pseudo-messages for the chat engine to inject.
  * `reset()` -- not supported (Engram is durable storage; reset is
    not a memory-layer concept). Raises NotImplementedError; the
    caller should manage the storage backend if they need a reset.

For real LlamaIndex integration, users wrap with
`SimpleComposableMemory`, `ChatMemoryBuffer`, etc -- this adapter is
the bridge to those.
"""

from __future__ import annotations

import logging

_LOG_LI = logging.getLogger(__name__)

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from engram.integrations._context import format_context
from engram.memory import Memory


@dataclass
class _PseudoChatMessage:
    """Mimic `llama_index.core.base.llms.types.ChatMessage`.

    LlamaIndex's ChatMessage has `role` and `content`. We use a local
    dataclass so this module doesn't import llama_index at module load
    time. Callers that want a real `ChatMessage` can convert (see
    `as_chat_message`).
    """

    role: str
    content: str
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def as_chat_message(self) -> Any:
        """Convert to a real `llama_index.core.base.llms.types.ChatMessage`.

        `MessageRole(self.role)` raises ValueError for any role string
        outside LlamaIndex's enum (system / user / assistant / function /
        tool / chatbot / model / developer).  When the caller built this
        pseudo-message with a custom or typo'd role, we surface a clearer
        error pointing at the offending value instead of a deep-stack
        pydantic / enum error from inside LlamaIndex.
        """
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        try:
            role = MessageRole(self.role)
        except ValueError as exc:
            valid = ", ".join(repr(m.value) for m in MessageRole)
            raise ValueError(
                f"role {self.role!r} is not a LlamaIndex MessageRole; "
                f"valid values are: {valid}"
            ) from exc
        return ChatMessage(
            role=role,
            content=self.content,
            additional_kwargs=self.additional_kwargs,
        )


class EngramLlamaIndexMemory:
    """LlamaIndex BaseMemory-shaped adapter for Engram.

    Args:
      memory: an `engram.Memory` instance.
      k: top-k retrieval count per `get(...)` call.
      system_role: role attached to the returned context message.
        LlamaIndex chat engines typically expect `"system"`.
      include_level: tag each surfaced memory with its level.
    """

    def __init__(
        self,
        memory: Memory,
        *,
        k: int = 5,
        system_role: str = "system",
        include_level: bool = True,
    ) -> None:
        self._memory = memory
        self._k = k
        self._system_role = system_role
        self._include_level = include_level

    @property
    def memory(self) -> Memory:
        return self._memory

    def put(self, message: _PseudoChatMessage | Any) -> None:
        """Observe a chat message as an Engram event.

        Accepts either the local `_PseudoChatMessage` or a real
        LlamaIndex `ChatMessage` (duck-typed on `.content`).
        """
        content = getattr(message, "content", None)
        if content:
            self._memory.observe(str(content))

    def get(
        self, input: str | None = None, **kwargs: Any
    ) -> Sequence[_PseudoChatMessage]:
        """Return one pseudo-message that carries the retrieved memory
        context.

        LlamaIndex's chat engine expects `get(input=question)` to
        return a list of messages to prepend to the chat. We pack the
        context into a single system-role message.

        Honors `token_limit` and `chat_history_limit` kwargs as best-effort
        caps on the retrieved-memory budget; other LlamaIndex-supplied
        kwargs are accepted (so newer chat-engine versions don't
        TypeError here) but emit a debug-level log so a caller can spot
        when the adapter is dropping options the rest of the framework
        might honor.
        """
        if not input:
            return []
        # Best-effort respect for LlamaIndex's standard kwargs.  Both are
        # caps; we use whichever is more restrictive against the
        # constructor's `k`.
        effective_k = self._k
        token_limit = kwargs.pop("token_limit", None)
        chat_history_limit = kwargs.pop("chat_history_limit", None)
        if isinstance(chat_history_limit, int) and chat_history_limit > 0:
            effective_k = min(effective_k, chat_history_limit)
        if kwargs:
            _LOG_LI.debug(
                "EngramLlamaIndexMemory.get ignoring unknown kwargs: %s",
                sorted(kwargs.keys()),
            )
        results = self._memory.retrieve(input, k=effective_k, reinforce=False)
        if not results:
            return []
        context = format_context(results, include_level=self._include_level)
        # token_limit is a soft cap: we don't tokenize here (LlamaIndex's
        # tokenizer isn't always importable), so we approximate via char
        # count at the standard 4-chars-per-token rule.  Caller still
        # gets the most-relevant top-k; only the rendered context is
        # truncated.
        if isinstance(token_limit, int) and token_limit > 0:
            cap_chars = token_limit * 4
            if len(context) > cap_chars:
                context = context[:cap_chars]
        return [
            _PseudoChatMessage(
                role=self._system_role,
                content=f"Relevant memories:\n{context}",
            )
        ]

    def get_all(self) -> Sequence[_PseudoChatMessage]:
        """Not supported -- Engram doesn't store messages as chat turns;
        it stores events. Returns an empty list."""
        return []

    def reset(self) -> None:
        """Reset is not a memory-layer concept; managing storage is the
        caller's responsibility."""
        raise NotImplementedError(
            "EngramLlamaIndexMemory does not support reset; manage the "
            "underlying SqliteStorage / Postgres backend directly if you "
            "need a clean slate."
        )


__all__ = ["EngramLlamaIndexMemory"]
