"""Tests for the LangGraph integration helpers.

The nodes (`EngramRetrieveNode`, `EngramObserveNode`) are callable
state-graph functions; LangGraph itself isn't required to test them
because they take and return mappings. We do verify they compose
correctly into a `langgraph.graph.StateGraph` though, since that's
the documented usage.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from engram import Memory, SqliteStorage
from engram.integrations.langgraph import EngramObserveNode, EngramRetrieveNode
from engram.providers._fake import FakeChat, FakeEmbedder


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


class GraphState(TypedDict, total=False):
    query: str
    engram_context: str
    reply: str


class TestEngramRetrieveNode:
    def test_empty_query_returns_empty_context(self, memory: Memory) -> None:
        node = EngramRetrieveNode(memory)
        out = node({"query": ""})
        assert out == {"engram_context": ""}

    def test_with_relevant_memory(self, memory: Memory) -> None:
        memory.observe("user has a dog")
        node = EngramRetrieveNode(memory, k=5)
        out = node({"query": "user has a dog"})
        assert "engram_context" in out
        assert "user has a dog" in out["engram_context"]

    def test_custom_state_keys(self, memory: Memory) -> None:
        memory.observe("hi")
        node = EngramRetrieveNode(memory, query_key="q", context_key="ctx")
        out = node({"q": "hi"})
        assert "ctx" in out
        assert "engram_context" not in out

    def test_composes_into_state_graph(self, memory: Memory) -> None:
        """End-to-end: build a tiny StateGraph with the retrieve node
        and verify it produces the context state slot."""
        memory.observe("the cat sat on the mat")

        def agent(state: GraphState) -> dict[str, str]:
            return {"reply": f"Got context: {state.get('engram_context', '')}"}

        builder = StateGraph(GraphState)
        builder.add_node("retrieve", EngramRetrieveNode(memory))
        builder.add_node("agent", agent)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "agent")
        builder.add_edge("agent", END)
        graph = builder.compile()
        result = graph.invoke({"query": "the cat sat on the mat"})
        assert "Got context" in result["reply"]
        assert "the cat" in result["engram_context"]


class TestEngramObserveNode:
    def test_observes_user_and_reply(self, memory: Memory) -> None:
        node = EngramObserveNode(memory)
        before = memory.storage.count_events()
        result = node({"query": "user says hi", "reply": "bot says hi"})
        assert result == {}
        after = memory.storage.count_events()
        assert after - before == 2

    def test_no_user_no_reply_is_noop(self, memory: Memory) -> None:
        node = EngramObserveNode(memory)
        before = memory.storage.count_events()
        node({})
        assert memory.storage.count_events() == before

    def test_full_loop_state_graph(self, memory: Memory) -> None:
        chat = FakeChat(default="bot reply")

        def agent(state: GraphState) -> dict[str, str]:
            return {"reply": chat.chat([])}

        builder = StateGraph(GraphState)
        builder.add_node("retrieve", EngramRetrieveNode(memory))
        builder.add_node("agent", agent)
        builder.add_node("record", EngramObserveNode(memory))
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "agent")
        builder.add_edge("agent", "record")
        builder.add_edge("record", END)
        graph = builder.compile()
        result = graph.invoke({"query": "hello"})
        assert result["reply"] == "bot reply"
        # Both query and reply got observed.
        assert memory.storage.count_events() == 2
