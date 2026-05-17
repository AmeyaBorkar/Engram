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


class TestLeafParamPassThrough:
    """Audit H-47: caller-tuned Phase E/F knobs (bm25_weight,
    mmr_lambda, recency_lambda, lexical_filter, …) used to be dropped
    when `_decomposed_retrieve` and `_multi_query_retrieve` built
    fresh `RetrieveParams` for each leaf.  The fix uses
    `params.replace(...)` so every knob the caller already set
    propagates to the leaves automatically."""

    def test_decompose_leaf_inherits_bm25_weight(
        self, storage: SqliteStorage
    ) -> None:
        embedder = FakeEmbedder(dim=8)
        # Plant 2 events so BM25 has documents to score.
        memory_no_chat = Memory(storage=storage, embedder=embedder)
        memory_no_chat.observe("apple banana cherry")
        memory_no_chat.observe("dragon elderberry fig")

        captured: list[float] = []
        real = memory_no_chat._retriever.retrieve

        def _spy(query: str, **kw: object) -> object:
            captured.append(kw["params"].bm25_weight)  # type: ignore[attr-defined,union-attr]
            return real(query, **kw)  # type: ignore[arg-type]

        decompose_prompt = render_decompose_prompt("apple cherry?")
        chat = FakeChat(
            scripts={content_hash(decompose_prompt): "apple\ncherry"}
        )
        memory = Memory(storage=storage, embedder=embedder, chat=chat)
        memory._retriever.retrieve = _spy  # type: ignore[method-assign]
        try:
            memory.retrieve(
                "apple cherry?",
                k=2,
                decompose=True,
                bm25_weight=1.5,
                reinforce=False,
            )
        finally:
            memory._retriever.retrieve = real  # type: ignore[method-assign]
        # Every leaf must inherit the caller's bm25_weight.
        assert captured, "leaf retrieves never ran"
        assert all(w == 1.5 for w in captured), captured

    def test_multi_query_leaf_inherits_recency_lambda(
        self, storage: SqliteStorage
    ) -> None:
        from engram.retrieve._multi_query import render_multi_query_prompt

        embedder = FakeEmbedder(dim=8)
        memory_no_chat = Memory(storage=storage, embedder=embedder)
        memory_no_chat.observe("alpha")
        memory_no_chat.observe("beta")

        captured: list[float] = []
        real = memory_no_chat._retriever.retrieve

        def _spy(query: str, **kw: object) -> object:
            captured.append(kw["params"].recency_lambda)  # type: ignore[attr-defined,union-attr]
            return real(query, **kw)  # type: ignore[arg-type]

        prompt = render_multi_query_prompt("orig query", 1)
        chat = FakeChat(scripts={content_hash(prompt): "paraphrase one"})
        memory = Memory(storage=storage, embedder=embedder, chat=chat)
        memory._retriever.retrieve = _spy  # type: ignore[method-assign]
        try:
            memory.retrieve(
                "orig query",
                k=2,
                multi_query_n=2,
                recency_lambda=0.5,
                recency_decay_days=30.0,
                reinforce=False,
            )
        finally:
            memory._retriever.retrieve = real  # type: ignore[method-assign]
        assert captured, "leaf retrieves never ran"
        assert all(rl == 0.5 for rl in captured), captured


class TestMultiQueryParallelDispatch:
    """Audit H-48: multi-query used to dispatch leaves serially while
    decompose used a thread pool.  Now both go through
    `_parallel_leaf_retrieves` so the fan-out cost matches between
    the two surfaces."""

    def test_multi_query_uses_parallel_leaf_retrieves(
        self, storage: SqliteStorage
    ) -> None:
        from engram.retrieve._multi_query import render_multi_query_prompt

        embedder = FakeEmbedder(dim=8)
        memory_no_chat = Memory(storage=storage, embedder=embedder)
        memory_no_chat.observe("hello world")

        prompt = render_multi_query_prompt("q", 1)
        chat = FakeChat(scripts={content_hash(prompt): "paraphrase"})
        memory = Memory(storage=storage, embedder=embedder, chat=chat)

        call_count = {"n": 0}
        real = memory._parallel_leaf_retrieves

        def _spy(queries: object, leaf_params: object) -> object:
            call_count["n"] += 1
            return real(queries, leaf_params)  # type: ignore[arg-type]

        memory._parallel_leaf_retrieves = _spy  # type: ignore[method-assign]
        try:
            memory.retrieve(
                "q",
                k=2,
                multi_query_n=2,
                reinforce=False,
            )
        finally:
            memory._parallel_leaf_retrieves = real  # type: ignore[method-assign]
        assert call_count["n"] == 1, (
            "_multi_query_retrieve should route leaves through "
            "_parallel_leaf_retrieves (audit H-48)"
        )


class TestDecomposeRerankerUsesSubQuery:
    """Audit M-31: the reranker used to be called with the ORIGINAL
    query for every sub-query's results.  The fix reranks each leaf
    against its OWN sub-query, then RRF-fuses the reranked rankings.
    """

    def test_reranker_sees_each_sub_query(self, storage: SqliteStorage) -> None:
        from engram.retrieve._reranker import RerankCandidate

        embedder = FakeEmbedder(dim=8)
        for content in ("alpha", "beta", "gamma"):
            Memory(storage=storage, embedder=embedder).observe(content)

        seen_queries: list[str] = []

        class _RecordingReranker:
            name = "recording"

            def rerank(
                self, query: str, candidates: list[RerankCandidate]
            ) -> list[float]:
                seen_queries.append(query)
                return [c.prior_score for c in candidates]

        decompose_prompt = render_decompose_prompt("alpha + beta?")
        chat = FakeChat(
            scripts={content_hash(decompose_prompt): "alpha question\nbeta question"}
        )
        memory = Memory(storage=storage, embedder=embedder, chat=chat)
        memory.retrieve(
            "alpha + beta?",
            k=5,
            decompose=True,
            reranker=_RecordingReranker(),  # type: ignore[arg-type]
            reinforce=False,
        )
        # The reranker should have been called with the SUB-queries,
        # not just the original.  We expect at least the two sub-
        # queries to appear (the original is also in the queries list,
        # so all 3 should show up).
        assert "alpha question" in seen_queries, seen_queries
        assert "beta question" in seen_queries, seen_queries
