"""Iterative ReAct retrieval tests."""

from __future__ import annotations

import json

import pytest

from engram import Memory, SqliteStorage
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat, FakeEmbedder
from engram.retrieve._react import (
    ReactVerdict,
    parse_react_response,
    react_judge,
    render_react_prompt,
)

# ---------------------------------------------------------------------------
# Prompt + parser
# ---------------------------------------------------------------------------


class TestPromptRender:
    def test_substitution(self) -> None:
        prompt = render_react_prompt("my question?", [])
        assert "my question?" in prompt
        assert "(none yet)" in prompt

    def test_critical_rules_precede_question(self) -> None:
        prompt = render_react_prompt("x", [])
        assert prompt.index("CRITICAL RULES") < prompt.index("ORIGINAL QUESTION")

    def test_memories_inline_with_level_tags(self) -> None:
        from uuid import uuid4

        from engram.schemas import Level, RetrievalResult

        r = RetrievalResult(
            item_id=uuid4(),
            level=Level.SUMMARY,
            content="user has a dog",
            confidence=0.9,
            score=0.9,
            supported_by=(),
        )
        prompt = render_react_prompt("question?", [r])
        assert "[summary] user has a dog" in prompt


class TestParser:
    def test_parses_sufficient_true(self) -> None:
        out = parse_react_response(json.dumps({"sufficient": True}))
        assert out.sufficient
        assert out.refined_query == ""

    def test_parses_with_refined_query(self) -> None:
        out = parse_react_response(
            json.dumps({"sufficient": False, "refined_query": "dog name"})
        )
        assert not out.sufficient
        assert out.refined_query == "dog name"

    def test_rejects_missing_sufficient(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError):
            parse_react_response(json.dumps({"refined_query": "x"}))

    def test_rejects_invalid_json(self) -> None:
        from engram.consolidation import AbstractionParseError

        with pytest.raises(AbstractionParseError):
            parse_react_response("not json")

    def test_strips_code_fence(self) -> None:
        out = parse_react_response(
            "```json\n" + json.dumps({"sufficient": True}) + "\n```"
        )
        assert out.sufficient


# ---------------------------------------------------------------------------
# react_judge fall-back semantics
# ---------------------------------------------------------------------------


class TestJudgeFallback:
    def test_garbage_response_returns_sufficient_true(self) -> None:
        """Defensive fallback: stop iterating on a broken judge."""
        chat = FakeChat(default="not parseable")
        verdict = react_judge("q", [], chat, max_retries=1)
        assert verdict.sufficient
        assert verdict.refined_query == ""

    def test_well_formed_response_used_as_is(self) -> None:
        prompt = render_react_prompt("question?", [])
        chat = FakeChat(
            scripts={
                content_hash(prompt): json.dumps(
                    {"sufficient": False, "refined_query": "refined"}
                )
            }
        )
        verdict = react_judge("question?", [], chat, max_retries=0)
        assert not verdict.sufficient
        assert verdict.refined_query == "refined"


# ---------------------------------------------------------------------------
# Memory.retrieve_iterative
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_with_chat(storage: SqliteStorage) -> Memory:
    return Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=8),
        chat=FakeChat(default=json.dumps({"sufficient": True})),
    )


class TestRetrieveIterativeBasics:
    def test_falls_back_to_one_shot_without_chat(
        self, storage: SqliteStorage
    ) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=8))
        memory.observe("a fact")
        # No chat -> one-shot retrieve; doesn't crash.
        results = memory.retrieve_iterative("query", k=1)
        assert isinstance(results, list)

    def test_stops_when_judge_says_sufficient(
        self, memory_with_chat: Memory
    ) -> None:
        memory_with_chat.observe("user has a dog")
        # Judge always returns sufficient=True -> exactly one retrieve step.
        results = memory_with_chat.retrieve_iterative(
            "user has a dog", k=5, max_steps=10
        )
        assert results
        assert any("dog" in r.content for r in results)

    def test_refines_query_when_insufficient(
        self, storage: SqliteStorage
    ) -> None:
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder, chat=None)
        # Plant two events: one matches "step 1", one matches "step 2".
        memory.observe("step 1 result")
        memory.observe("step 2 hidden answer")

        # Configure chat: first call says insufficient with refined
        # query "step 2"; second call says sufficient.
        step1_prompt_initial = render_react_prompt("question?", [])
        chat = FakeChat(
            scripts={
                # We don't pre-script all intermediate prompts (the
                # accumulated-results block changes between calls); the
                # default reply handles them.
            },
            default=json.dumps({"sufficient": True}),
        )
        # Use a more flexible spy chat: count calls; return refined for first,
        # sufficient for second.
        call_count = 0

        def chat_fn(messages: list) -> str:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(
                    {"sufficient": False, "refined_query": "step 2"}
                )
            return json.dumps({"sufficient": True})

        chat.chat = chat_fn  # type: ignore[method-assign,assignment]
        memory._chat = chat
        # Wire reconciler back since we've now set chat.
        results = memory.retrieve_iterative("question?", k=5, max_steps=3)
        assert call_count == 2
        contents = {r.content for r in results}
        assert "step 2 hidden answer" in contents
        # Pin shape: step1_prompt_initial was just used for typing
        _ = step1_prompt_initial

    def test_max_steps_bound(self, storage: SqliteStorage) -> None:
        """Even if the judge always says insufficient, we stop at max_steps."""
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder, chat=None)
        memory.observe("fact")
        call_count = 0

        def chat_fn(messages: list) -> str:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            # Each step returns a different refined query so we don't
            # short-circuit on the "no change" guard.
            return json.dumps(
                {"sufficient": False, "refined_query": f"q-{call_count}"}
            )

        chat = FakeChat(default="")
        chat.chat = chat_fn  # type: ignore[method-assign,assignment]
        memory._chat = chat
        memory.retrieve_iterative("query", k=5, max_steps=4)
        # The judge is skipped on the final iteration (its verdict can't
        # change the outcome — the loop exits anyway), so max_steps=4
        # produces max_steps-1 chat calls.
        assert call_count == 3

    def test_stops_on_no_change_refinement(
        self, storage: SqliteStorage
    ) -> None:
        """If the judge returns the same refined_query as the current
        query, we stop (defensive)."""
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder, chat=None)
        memory.observe("fact")
        call_count = 0

        def chat_fn(messages: list) -> str:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            # Always returns the same query "query" -> retrieve_iterative
            # should bail after the first such response.
            return json.dumps(
                {"sufficient": False, "refined_query": "query"}
            )

        chat = FakeChat(default="")
        chat.chat = chat_fn  # type: ignore[method-assign,assignment]
        memory._chat = chat
        memory.retrieve_iterative("query", k=5, max_steps=10)
        # One step's retrieve, then judge with no change -> stop.
        assert call_count == 1

    def test_invalid_args_rejected(self, memory_with_chat: Memory) -> None:
        with pytest.raises(ValueError, match="k must be"):
            memory_with_chat.retrieve_iterative("q", k=0)
        with pytest.raises(ValueError, match="max_steps must be"):
            memory_with_chat.retrieve_iterative("q", k=1, max_steps=0)

    def test_no_chat_early_return_matches_multi_step_reinforcement(
        self, storage: SqliteStorage
    ) -> None:
        """Audit M-37: the no-chat early-return path used to forward
        `reinforce=reinforce` straight into `retrieve`, which called
        `_engine.reinforce` per leaf result before dedup/slice.  The
        multi-step path calls `retrieve(reinforce=False)` for every
        leaf and reinforces ONCE at the surface (after dedup + slice).
        After the fix the two paths produce the same reinforce-count
        per surfaced item.
        """
        embedder = FakeEmbedder(dim=8)
        memory = Memory(storage=storage, embedder=embedder)
        memory.observe("a fact")
        # Capture reinforcement calls.
        seen: list[tuple[object, object]] = []
        real = memory._engine.reinforce

        def _spy(item_id: object, kind: object, **kw: object) -> object:
            seen.append((item_id, kind))
            return real(item_id, kind, **kw)  # type: ignore[arg-type]

        memory._engine.reinforce = _spy  # type: ignore[method-assign,assignment]
        try:
            results = memory.retrieve_iterative("a fact", k=1, reinforce=True)
        finally:
            memory._engine.reinforce = real  # type: ignore[method-assign,assignment]
        assert results
        # Each surfaced item must be reinforced exactly once (not
        # twice — pre-fix the inner retrieve's `reinforce=True` would
        # have driven a second call).
        for r in results:
            count = sum(1 for it, _ in seen if it == r.item_id)
            assert count == 1, (r.item_id, count)


class TestVerdictModel:
    def test_default_refined_query(self) -> None:
        v = ReactVerdict(sufficient=True)
        assert v.refined_query == ""

    def test_frozen(self) -> None:
        v = ReactVerdict(sufficient=True)
        with pytest.raises(ValueError, match=r"frozen|immutable"):
            v.sufficient = False
