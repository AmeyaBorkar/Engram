"""HyDE (Hypothetical Document Embeddings) retrieval tests."""

from __future__ import annotations

import pytest

from engram import Memory, SqliteStorage
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.retrieve._hyde import (
    hyde_transform,
    render_hyde_prompt,
)
from engram.schemas import Embedding, Event, ItemKind


@pytest.fixture
def memory_with_chat(storage: SqliteStorage) -> Memory:
    return Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=8),
        chat=FakeChat(default="hypothetical answer placeholder."),
    )


class TestRenderPrompt:
    def test_query_inlined(self) -> None:
        prompt = render_hyde_prompt("multi\nline query")
        # Newline escaped so the QUESTION section stays single-line.
        assert "multi\\nline query" in prompt
        assert "CRITICAL RULES" in prompt

    def test_critical_rules_precede_question(self) -> None:
        prompt = render_hyde_prompt("x")
        assert prompt.index("CRITICAL RULES") < prompt.index("QUESTION")


class TestHydeTransform:
    def test_returns_hypothetical(self) -> None:
        chat = FakeChat(default="The user has a dog named Rex.")
        out = hyde_transform("what pet does the user have?", chat)
        assert out == "The user has a dog named Rex."

    def test_falls_back_to_query_on_empty_response(self) -> None:
        chat = FakeChat(default="")
        out = hyde_transform("question?", chat)
        assert out == "question?"

    def test_falls_back_when_hypothetical_too_short(self) -> None:
        """If the chat returned filler (shorter than the query), bail."""
        # query is long, response is filler.
        chat = FakeChat(default="ok")
        long_query = "A reasonably long question about the user's preferences."
        out = hyde_transform(long_query, chat)
        assert out == long_query

    def test_strips_whitespace(self) -> None:
        chat = FakeChat(default="\n  The user has a dog named Rex.  \n")
        out = hyde_transform("what pet?", chat)
        assert out == "The user has a dog named Rex."


class TestMemoryRetrieveWithHyde:
    def test_hyde_routes_through_chat(
        self, memory_with_chat: Memory, storage: SqliteStorage
    ) -> None:
        """Plant a memory phrased like an answer; verify that HyDE
        finds it when the query is phrased like a question."""
        embedder = memory_with_chat.embedder
        # Plant the memory using ITS embedding so similarity is exact
        # when HyDE produces the matching text.
        memory_phrase = "The user has a golden retriever named Rex."
        event = Event(content=memory_phrase)
        storage.insert_event(event)
        storage.insert_embedding(
            Embedding(
                item_id=event.id,
                item_kind=ItemKind.EVENT,
                model=embedder.model,
                dim=embedder.dim,
                vector=tuple(embedder.embed([memory_phrase])[0]),
            )
        )

        # Question form -- the embedder won't bridge "question" and
        # "answer" perfectly. Without HyDE, the cosine similarity is
        # whatever the FakeEmbedder (hash-based) produces.
        question = "what pet does the user have?"

        # Script FakeChat to return the matching phrase exactly when
        # given the HyDE prompt for our question.
        hyde_prompt = render_hyde_prompt(question)
        chat = FakeChat(scripts={content_hash(hyde_prompt): memory_phrase})
        memory = Memory(
            storage=storage,
            embedder=embedder,
            chat=chat,
        )

        # Without HyDE: hash-based similarity, no guarantee.
        # With HyDE: the chat returns the exact memory text; embedding
        # similarity to the stored memory is then perfect.
        with_hyde = memory.retrieve(question, k=1, reinforce=False, hyde=True)
        assert len(with_hyde) >= 1
        assert with_hyde[0].item_id == event.id
        # The HyDE-mapped query should produce a higher score than the
        # raw query (perfect similarity vs hash collision).
        without_hyde = memory.retrieve(question, k=1, reinforce=False, hyde=False)
        assert with_hyde[0].score >= without_hyde[0].score

    def test_hyde_no_op_without_chat(self, storage: SqliteStorage) -> None:
        """If Memory was constructed without a chat provider, hyde=True
        is a no-op (the original query is used)."""
        memory = Memory(
            storage=storage,
            embedder=FakeEmbedder(dim=8),
            chat=None,
        )
        memory.observe("anything")
        # Doesn't crash and returns sensible results.
        results = memory.retrieve("hi", k=1, hyde=True, reinforce=False)
        assert isinstance(results, list)

    def test_default_hyde_is_off(self, memory_with_chat: Memory) -> None:
        """Existing callers that don't pass hyde= get the legacy
        behavior (no chat call, raw query used)."""
        memory_with_chat.observe("a known fact")

        call_count = 0
        original_chat = memory_with_chat._chat
        assert original_chat is not None

        def _spy(messages: object) -> str:
            nonlocal call_count
            call_count += 1
            return original_chat.chat(messages)  # type: ignore[arg-type]

        original_chat.chat = _spy  # type: ignore[method-assign]
        # No hyde argument -> retriever does NOT call the chat.
        memory_with_chat.retrieve("query", k=1, reinforce=False)
        assert call_count == 0
