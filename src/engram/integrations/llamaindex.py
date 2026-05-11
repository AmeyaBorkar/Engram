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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engram.integrations._context import format_context
from engram.memory import Memory

if TYPE_CHECKING:  # pragma: no cover - import-only for typing
    pass


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
        """Convert to a real `llama_index.core.base.llms.types.ChatMessage`."""
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        return ChatMessage(
            role=MessageRole(self.role),
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
        self, input: str | None = None, **_: Any
    ) -> Sequence[_PseudoChatMessage]:
        """Return one pseudo-message that carries the retrieved memory
        context.

        LlamaIndex's chat engine expects `get(input=question)` to
        return a list of messages to prepend to the chat. We pack the
        context into a single system-role message.
        """
        if not input:
            return []
        results = self._memory.retrieve(input, k=self._k, reinforce=False)
        if not results:
            return []
        context = format_context(results, include_level=self._include_level)
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
