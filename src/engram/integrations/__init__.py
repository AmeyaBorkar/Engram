"""Framework integrations for Engram.

Each submodule is independently importable -- importing
`engram.integrations` itself does NOT pull in framework dependencies.
Users opt into the heavy deps via the matching extras:

  * `engrampy[langgraph]` for LangGraph nodes / state helpers
  * `engrampy[llamaindex]` for the LlamaIndex BaseMemory adapter
  * No extras needed for the framework-agnostic `EngramAgent`.

Direct OpenAI / Anthropic agent loops use the agent wrapper plus the
existing `engram.providers.openai` / `engram.providers.anthropic`
adapters; there's no separate "engram.integrations.openai" -- the
agent wrapper is the integration.
"""

from engram.integrations._agent import EngramAgent
from engram.integrations._context import format_context

__all__ = [
    "EngramAgent",
    "format_context",
]
