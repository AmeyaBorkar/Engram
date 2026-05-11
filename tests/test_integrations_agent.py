"""Tests for the framework-agnostic `EngramAgent` + `format_context`."""

from __future__ import annotations

import pytest

from engram import Memory, Outcome, SqliteStorage, new_id
from engram.integrations import EngramAgent, format_context
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.schemas import Level, MemoryItem, Procedure, ProcedureMatch, RetrievalResult


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_input_empty_output(self) -> None:
        assert format_context([]) == ""

    def test_retrieval_results_formatted_with_level(self) -> None:
        item_id = new_id()
        r = RetrievalResult(
            item_id=item_id,
            level=Level.SUMMARY,
            content="user has a dog",
            confidence=0.9,
            score=0.9,
            supported_by=(),
        )
        out = format_context([r])
        assert "[summary]" in out
        assert "user has a dog" in out
        assert out.startswith("- ")

    def test_include_score(self) -> None:
        r = RetrievalResult(
            item_id=new_id(),
            level=Level.EVENT,
            content="x",
            confidence=0.5,
            score=0.5,
            supported_by=(),
        )
        out = format_context([r], include_score=True)
        assert "score=0.50" in out

    def test_procedure_match_formatted_with_arrow(self) -> None:
        p = Procedure(
            situation="flaky test",
            action="rerun with --no-cov",
            outcome=Outcome.SUCCESS,
        )
        m = ProcedureMatch(procedure=p, score=0.85, similarity=0.9)
        out = format_context([m])
        assert "[procedure]" in out
        assert "flaky test -> rerun with --no-cov" in out

    def test_max_items_caps(self) -> None:
        results = [
            RetrievalResult(
                item_id=new_id(),
                level=Level.EVENT,
                content=f"hit-{i}",
                confidence=1.0,
                score=1.0,
                supported_by=(),
            )
            for i in range(5)
        ]
        out = format_context(results, max_items=2)
        assert out.count("\n") == 1  # 2 lines = one newline between them
        assert "hit-0" in out
        assert "hit-1" in out
        assert "hit-3" not in out

    def test_custom_bullet(self) -> None:
        r = RetrievalResult(
            item_id=new_id(),
            level=Level.SUMMARY,
            content="x",
            confidence=0.5,
            score=0.5,
            supported_by=(),
        )
        out = format_context([r], bullet="* ")
        assert out.startswith("* ")


# ---------------------------------------------------------------------------
# EngramAgent
# ---------------------------------------------------------------------------


class TestEngramAgentBasics:
    def test_chat_returns_turn_with_reply(self, memory: Memory) -> None:
        memory.observe("user prefers Python over Java")
        chat = FakeChat(default="Python is great for this.")
        agent = EngramAgent(memory, chat)
        turn = agent.chat("What language should I use?")
        assert turn.reply == "Python is great for this."
        assert "user prefers" in turn.retrieved_context
        # The user message + reply both got observed by default.
        assert len(turn.observed_event_ids) == 2

    def test_auto_observe_off(self, memory: Memory) -> None:
        before = memory.storage.count_events()
        chat = FakeChat(default="Reply.")
        agent = EngramAgent(memory, chat, auto_observe=False)
        turn = agent.chat("What's up?")
        assert turn.observed_event_ids == ()
        assert memory.storage.count_events() == before

    def test_no_relevant_context_uses_bare_system(self, memory: Memory) -> None:
        # Empty memory -> no retrieved context.
        chat = FakeChat(default="Reply.")
        agent = EngramAgent(memory, chat)
        turn = agent.chat("Hi.")
        assert turn.retrieved_context == ""
        assert "Relevant memories:" not in turn.system_prompt

    def test_custom_system_prompt(self, memory: Memory) -> None:
        chat = FakeChat(default="Reply.")
        agent = EngramAgent(
            memory, chat, system_prompt="You are pirate-bot."
        )
        turn = agent.chat("hi")
        assert turn.system_prompt.startswith("You are pirate-bot.")

    def test_record_procedure_outcome(self, memory: Memory) -> None:
        chat = FakeChat(default="Reply.")
        agent = EngramAgent(memory, chat)
        agent.record_procedure_outcome(
            "user reports flaky test",
            "rerun with --no-cov",
            Outcome.SUCCESS,
        )
        procs = memory.storage.list_procedures()
        assert len(procs) == 1
        assert procs[0].outcome is Outcome.SUCCESS


class TestEngramAgentWithMemoryContext:
    def test_context_appears_in_system_message(self, memory: Memory) -> None:
        # Plant a relevant memory item.
        memory.storage.insert_memory_item(
            MemoryItem(level=Level.SUMMARY, content="user prefers dark mode")
        )
        # Embed it so retrieve can find it.
        from engram.schemas import Embedding, ItemKind

        memory.storage.insert_embedding(
            Embedding(
                item_id=memory.storage.list_memory_items(limit=1)[0].id,
                item_kind=ItemKind.MEMORY_ITEM,
                model=memory.embedder.model,
                dim=memory.embedder.dim,
                vector=tuple(memory.embedder.embed(["user prefers dark mode"])[0]),
            )
        )

        chat = FakeChat(default="Ok.")
        agent = EngramAgent(memory, chat, auto_observe=False)
        turn = agent.chat("user prefers dark mode")
        assert "Relevant memories:" in turn.system_prompt
        assert "user prefers dark mode" in turn.retrieved_context
