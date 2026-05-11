"""Tests for the LlamaIndex memory adapter."""

from __future__ import annotations

import pytest

from engram import Memory, SqliteStorage
from engram.integrations.llamaindex import (
    EngramLlamaIndexMemory,
    _PseudoChatMessage,
)
from engram.providers._fake import FakeEmbedder


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


class TestEngramLlamaIndexMemoryBasics:
    def test_put_observes_message(self, memory: Memory) -> None:
        adapter = EngramLlamaIndexMemory(memory)
        before = memory.storage.count_events()
        adapter.put(_PseudoChatMessage(role="user", content="hello"))
        assert memory.storage.count_events() == before + 1

    def test_put_with_empty_content_noop(self, memory: Memory) -> None:
        adapter = EngramLlamaIndexMemory(memory)
        before = memory.storage.count_events()
        adapter.put(_PseudoChatMessage(role="user", content=""))
        assert memory.storage.count_events() == before

    def test_get_returns_context_message(self, memory: Memory) -> None:
        memory.observe("user prefers dark mode")
        adapter = EngramLlamaIndexMemory(memory, k=5)
        messages = list(adapter.get(input="user prefers dark mode"))
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert "Relevant memories:" in messages[0].content
        assert "user prefers dark mode" in messages[0].content

    def test_get_empty_input_returns_empty(self, memory: Memory) -> None:
        adapter = EngramLlamaIndexMemory(memory)
        assert list(adapter.get(input=None)) == []
        assert list(adapter.get(input="")) == []

    def test_get_empty_corpus_returns_empty(self, memory: Memory) -> None:
        adapter = EngramLlamaIndexMemory(memory)
        # Nothing observed; retrieve finds nothing; the adapter returns []
        # rather than a bare "Relevant memories:\n" pseudo-message.
        assert list(adapter.get(input="anything")) == []

    def test_get_all_returns_empty(self, memory: Memory) -> None:
        adapter = EngramLlamaIndexMemory(memory)
        assert list(adapter.get_all()) == []

    def test_reset_raises(self, memory: Memory) -> None:
        adapter = EngramLlamaIndexMemory(memory)
        with pytest.raises(NotImplementedError, match="reset"):
            adapter.reset()


class TestEngramLlamaIndexMemoryDuckTyping:
    """Verify the adapter duck-types LlamaIndex's BaseMemory shape
    enough that callers can wrap it in a real `SimpleComposableMemory`
    or chat engine without changes."""

    def test_put_accepts_anything_with_content(self, memory: Memory) -> None:
        from dataclasses import dataclass

        @dataclass
        class FakeLlamaIndexMessage:
            role: str
            content: str

        adapter = EngramLlamaIndexMemory(memory)
        before = memory.storage.count_events()
        adapter.put(FakeLlamaIndexMessage(role="user", content="hello"))
        assert memory.storage.count_events() == before + 1

    def test_pseudo_message_converts_to_real(self, memory: Memory) -> None:
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        adapter = EngramLlamaIndexMemory(memory)
        memory.observe("x")
        messages = adapter.get(input="x")
        assert len(messages) == 1
        # The pseudo-message converts to a real LlamaIndex ChatMessage.
        real: ChatMessage = messages[0].as_chat_message()
        assert isinstance(real, ChatMessage)
        assert real.role == MessageRole.SYSTEM
