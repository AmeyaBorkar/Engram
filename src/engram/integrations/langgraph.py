"""LangGraph integration.

Two helpers for plugging Engram memory into a LangGraph state graph:

  * `EngramRetrieveNode` -- a callable graph node that reads the
    user query from the state, retrieves relevant memories, and
    writes a formatted context string into the state under the
    configured `context_key` (default `"engram_context"`).
  * `EngramObserveNode` -- a callable graph node that observes the
    latest user message + assistant reply for future retrieval.

The classes are deliberately thin: they hold the Memory handle, the
state-key conventions, and the retrieve params. The actual graph
topology is the caller's call.

Import is lazy: `import engram.integrations.langgraph` does not load
LangGraph -- the LangGraph imports happen inside method bodies. This
keeps `import engram` light for users who don't need the framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from engram.integrations._context import format_context
from engram.memory import Memory


@dataclass(frozen=True)
class EngramRetrieveNode:
    """LangGraph node that injects memory context into the state.

    Usage:

        from langgraph.graph import StateGraph
        from engram import Memory
        from engram.integrations.langgraph import EngramRetrieveNode

        memory = Memory(storage=..., embedder=...)
        graph = StateGraph(MyState)
        graph.add_node("retrieve_memory", EngramRetrieveNode(memory))
        graph.add_node("agent", my_agent_fn)
        graph.add_edge("retrieve_memory", "agent")

    Reads `state[query_key]` (default `"query"`), retrieves top-k
    memories, formats them, and writes the result to
    `state[context_key]` (default `"engram_context"`). The state must
    be a mapping (dict or pydantic.BaseModel-with-dict-coercion).
    """

    memory: Memory
    query_key: str = "query"
    context_key: str = "engram_context"
    k: int = 5
    include_level: bool = True

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        query = state.get(self.query_key, "")
        if not query:
            return {self.context_key: ""}
        results = self.memory.retrieve(query, k=self.k, reinforce=False)
        context = format_context(results, include_level=self.include_level)
        return {self.context_key: context}


@dataclass(frozen=True)
class EngramObserveNode:
    """LangGraph node that observes user + assistant messages.

    Usage at the tail of an agent turn:

        graph.add_node("record", EngramObserveNode(memory))
        graph.add_edge("agent", "record")

    Reads `state[user_key]` and `state[reply_key]` and observes both
    as events. Returns the unchanged state (LangGraph state updates
    are merged; an empty return = no-op merge).
    """

    memory: Memory
    user_key: str = "query"
    reply_key: str = "reply"

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        user = state.get(self.user_key)
        reply = state.get(self.reply_key)
        if user:
            self.memory.observe(str(user))
        if reply:
            self.memory.observe(str(reply))
        return {}


__all__ = [
    "EngramObserveNode",
    "EngramRetrieveNode",
]
