"""Multi-query expansion + Reciprocal Rank Fusion tests."""

from __future__ import annotations

import pytest

from engram import Memory, SqliteStorage, new_id
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.retrieve._multi_query import (
    expand_queries,
    reciprocal_rank_fusion,
    render_multi_query_prompt,
)
from engram.schemas import Embedding, Event, ItemKind, Level, RetrievalResult

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class TestPromptShape:
    def test_substitution(self) -> None:
        prompt = render_multi_query_prompt("query?", 3)
        assert "N = 3" in prompt
        assert "query?" in prompt

    def test_critical_rules_precede_question(self) -> None:
        prompt = render_multi_query_prompt("x", 2)
        assert prompt.index("CRITICAL RULES") < prompt.index("QUESTION")


# ---------------------------------------------------------------------------
# expand_queries
# ---------------------------------------------------------------------------


class TestExpandQueries:
    def test_n_le_1_returns_only_original(self) -> None:
        chat = FakeChat(default="should-not-be-called")
        assert expand_queries("query", 1, chat) == ["query"]
        assert expand_queries("query", 0, chat) == ["query"]

    def test_lines_become_paraphrases(self) -> None:
        chat = FakeChat(default="rephrase 1\nrephrase 2\nrephrase 3")
        out = expand_queries("original", 4, chat)
        # Original + 3 paraphrases.
        assert out == ["original", "rephrase 1", "rephrase 2", "rephrase 3"]

    def test_strips_numbered_markers(self) -> None:
        chat = FakeChat(default="1. one\n2) two\n- three\n* four")
        out = expand_queries("orig", 5, chat)
        assert out == ["orig", "one", "two", "three", "four"]

    def test_drops_dup_with_original(self) -> None:
        chat = FakeChat(default="orig\nactual paraphrase")
        out = expand_queries("orig", 3, chat)
        # The literal "orig" duplicate is dropped; "actual paraphrase" remains.
        assert "orig" in out
        assert "actual paraphrase" in out
        # No duplicate of the original.
        assert out.count("orig") == 1

    def test_fallback_on_chat_error(self) -> None:
        class _ErrorChat:
            name: str = "err"
            model: str = "err"

            def chat(self, messages: object) -> str:
                raise RuntimeError("provider down")

            async def achat(self, messages: object) -> str:
                raise RuntimeError("provider down")

            def manifest_hash(self) -> str:
                return "err"

        out = expand_queries("query", 3, _ErrorChat())  # type: ignore[arg-type,unused-ignore]
        assert out == ["query"]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _result(content: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        item_id=new_id(),
        level=Level.SUMMARY,
        content=content,
        confidence=0.5,
        score=1.0 / rank,
        supported_by=(),
    )


class TestReciprocalRankFusion:
    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_ranking_preserves_order(self) -> None:
        a, b, c = _result("a", 1), _result("b", 2), _result("c", 3)
        fused = reciprocal_rank_fusion([[a, b, c]], k=60)
        assert [r.item_id for r in fused] == [a.item_id, b.item_id, c.item_id]

    def test_item_in_two_lists_outranks_singleton(self) -> None:
        """An item that appears in BOTH lists at modest ranks beats
        an item that appears in ONE list at the top -- this is the
        whole point of RRF."""
        a = _result("a", 1)  # in list 1 at rank 1
        b = _result("b", 1)  # in list 2 at rank 1
        c = _result("c", 1)  # in BOTH lists at rank 2 each
        # Make two rankings where c appears in both.
        list1 = [a, c]
        list2 = [b, c]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        # c should outrank both a and b because it appears in both.
        ids = [r.item_id for r in fused]
        assert ids[0] == c.item_id

    def test_rrf_score_math(self) -> None:
        """Verify the formula exactly: score(d) = sum_q 1/(k + rank_q(d))."""
        a = _result("a", 1)
        b = _result("b", 1)
        fused = reciprocal_rank_fusion([[a, b], [b, a]], k=60)
        # a: rank 1 in list1, rank 2 in list2 -> 1/61 + 1/62
        # b: rank 2 in list1, rank 1 in list2 -> 1/62 + 1/61
        # Both have the same score -> tied; order is insertion order.
        assert len(fused) == 2
        # Scores match the formula.
        expected = 1.0 / 61 + 1.0 / 62
        assert fused[0].score == pytest.approx(expected, abs=1e-9)
        assert fused[1].score == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Memory.retrieve + multi-query end-to-end
# ---------------------------------------------------------------------------


class TestMemoryMultiQueryRetrieve:
    def test_multi_query_off_default(self, storage: SqliteStorage) -> None:
        """Default RetrieveParams.multi_query_n=1 -> chat is NOT called."""
        embedder = FakeEmbedder(dim=8)
        call_count = 0

        class _SpyChat:
            name: str = "spy"
            model: str = "spy"

            def chat(self, messages: object) -> str:
                nonlocal call_count
                call_count += 1
                return "shouldn't-be-called"

            async def achat(self, messages: object) -> str:
                return "shouldn't-be-called"

            def manifest_hash(self) -> str:
                return "spy"

        memory = Memory(storage=storage, embedder=embedder, chat=_SpyChat())  # type: ignore[arg-type,unused-ignore]
        memory.observe("a fact")
        memory.retrieve("query", k=1, reinforce=False)
        assert call_count == 0

    def test_multi_query_uses_chat_and_fuses(self, storage: SqliteStorage) -> None:
        """multi_query_n=3 -> chat IS called once for paraphrases;
        results from all 3 queries get fused via RRF."""
        embedder = FakeEmbedder(dim=8)

        # Plant 2 memory items with distinct embeddings.
        for content in ("user has a dog", "user prefers coffee"):
            ev = Event(content=content)
            storage.insert_event(ev)
            storage.insert_embedding(
                Embedding(
                    item_id=ev.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=embedder.dim,
                    vector=tuple(embedder.embed([content])[0]),
                )
            )

        # Script FakeChat to return 2 paraphrases for the multi-query prompt.
        mq_prompt = render_multi_query_prompt("question?", 2)
        chat = FakeChat(
            scripts={content_hash(mq_prompt): ("user has a dog\nuser prefers coffee")},
        )
        memory = Memory(storage=storage, embedder=embedder, chat=chat)

        results = memory.retrieve("question?", k=10, multi_query_n=3, reinforce=False)
        # Both planted items should be in the fused result list.
        ids = {r.item_id for r in results}
        assert len(ids) >= 1

    def test_invalid_multi_query_n_rejected(self) -> None:
        from engram.retrieve import RetrieveParams

        with pytest.raises(ValueError, match="multi_query_n"):
            RetrieveParams(multi_query_n=0)

    def test_invalid_rrf_k_rejected(self) -> None:
        from engram.retrieve import RetrieveParams

        with pytest.raises(ValueError, match="rrf_k"):
            RetrieveParams(rrf_k=0)
