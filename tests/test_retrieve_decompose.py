"""Query decomposition tests."""

from __future__ import annotations

import pytest

from engram import Memory, SqliteStorage
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.retrieve._decompose import (
    decompose_query,
    render_decompose_prompt,
)


class TestPromptRender:
    def test_query_inlined(self) -> None:
        prompt = render_decompose_prompt("multi\nline question")
        # Newlines escaped so QUESTION block stays single-line.
        assert "multi\\nline question" in prompt

    def test_critical_rules_precede_question(self) -> None:
        prompt = render_decompose_prompt("x")
        assert prompt.index("CRITICAL RULES") < prompt.index("QUESTION")


class TestDecomposeQuery:
    def test_returns_original_plus_sub_queries(self) -> None:
        chat = FakeChat(default="sub-q-1\nsub-q-2\nsub-q-3")
        out = decompose_query("complex question?", chat)
        assert out == ["complex question?", "sub-q-1", "sub-q-2", "sub-q-3"]

    def test_strips_numbered_markers(self) -> None:
        chat = FakeChat(default="1. first\n2) second\n- third")
        out = decompose_query("orig?", chat)
        assert out == ["orig?", "first", "second", "third"]

    def test_drops_duplicate_of_original(self) -> None:
        chat = FakeChat(default="orig?\nactual sub-question")
        out = decompose_query("orig?", chat)
        assert out.count("orig?") == 1
        assert "actual sub-question" in out

    def test_fallback_on_chat_error(self) -> None:
        class _Err:
            name: str = "err"
            model: str = "err"

            def chat(self, messages: object) -> str:
                raise RuntimeError("boom")

            async def achat(self, messages: object) -> str:
                raise RuntimeError("boom")

            def manifest_hash(self) -> str:
                return "err"

        out = decompose_query("query", _Err())  # type: ignore[arg-type,unused-ignore]
        assert out == ["query"]


@pytest.fixture
def memory_with_chat(storage: SqliteStorage) -> Memory:
    return Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=8),
        chat=FakeChat(default=""),
    )


class TestMemoryRetrieveWithDecompose:
    def test_decompose_off_default(self, storage: SqliteStorage) -> None:
        """`decompose=False` default -> chat is NOT called."""
        embedder = FakeEmbedder(dim=8)
        call_count = 0

        class _Spy:
            name: str = "spy"
            model: str = "spy"

            def chat(self, messages: object) -> str:
                nonlocal call_count
                call_count += 1
                return ""

            async def achat(self, messages: object) -> str:
                return ""

            def manifest_hash(self) -> str:
                return "spy"

        memory = Memory(storage=storage, embedder=embedder, chat=_Spy())  # type: ignore[arg-type,unused-ignore]
        memory.observe("a fact")
        memory.retrieve("q", k=1, reinforce=False)
        assert call_count == 0

    def test_decompose_uses_chat_and_fuses(
        self, storage: SqliteStorage
    ) -> None:
        embedder = FakeEmbedder(dim=8)
        # Plant 3 events.
        memory_no_chat = Memory(storage=storage, embedder=embedder)
        for content in (
            "user lives in Vermont",
            "user works at Acme Corp",
            "user has a dog Rex",
        ):
            memory_no_chat.observe(content)

        decompose_prompt = render_decompose_prompt("where + work + pet?")
        chat = FakeChat(
            scripts={
                content_hash(decompose_prompt): (
                    "user lives in Vermont\nuser works at Acme Corp\nuser has a dog Rex"
                )
            }
        )
        memory = Memory(storage=storage, embedder=embedder, chat=chat)
        results = memory.retrieve(
            "where + work + pet?",
            k=10,
            decompose=True,
            reinforce=False,
        )
        contents = {r.content for r in results}
        # All three planted memories surface because each was hit by
        # its own sub-query.
        assert "user lives in Vermont" in contents
        assert "user works at Acme Corp" in contents
        assert "user has a dog Rex" in contents

    def test_decompose_no_op_without_chat(
        self, storage: SqliteStorage
    ) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=8))
        memory.observe("x")
        results = memory.retrieve("query", k=1, decompose=True, reinforce=False)
        # Without chat, decompose=True is a no-op; one-shot retrieve.
        assert isinstance(results, list)
