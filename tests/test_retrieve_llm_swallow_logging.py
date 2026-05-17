"""Verify the four LLM-touching retrieve modules log a WARNING when
they swallow a chat exception and fall back to a safe default.

Modules:
  * `_hyde.hyde_transform`
  * `_multi_query.expand_queries`
  * `_decompose.decompose_query`
  * `_temporal.compute_temporal_anchor`
  * `_react.react_judge`

The audit (H-52, H-53) said bare `except Exception: return <default>`
was too quiet -- an operator investigating a recall regression had
no signal that the LLM seam was silently inert. The fix logs the
exception type + message to the `engram.retrieve` logger before
falling back.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from engram.retrieve._decompose import decompose_query
from engram.retrieve._hyde import hyde_transform
from engram.retrieve._multi_query import expand_queries
from engram.retrieve._react import react_judge
from engram.retrieve._temporal import compute_temporal_anchor


class _ErrChat:
    name: str = "err"
    model: str = "err"

    def chat(self, messages: object) -> str:
        raise RuntimeError("boom")

    async def achat(self, messages: object) -> str:
        raise RuntimeError("boom")

    def manifest_hash(self) -> str:
        return "err"


class TestSwallowLogsWarning:
    def test_hyde_logs_on_chat_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="engram.retrieve"):
            out = hyde_transform("query", _ErrChat())  # type: ignore[arg-type]
        assert out == "query"
        assert any(
            "hyde" in r.getMessage().lower() and "boom" in r.getMessage()
            for r in caplog.records
        )

    def test_multi_query_logs_on_chat_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="engram.retrieve"):
            out = expand_queries("query", 3, _ErrChat())  # type: ignore[arg-type]
        assert out == ["query"]
        assert any(
            "expand_queries" in r.getMessage() and "boom" in r.getMessage()
            for r in caplog.records
        )

    def test_decompose_logs_on_chat_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="engram.retrieve"):
            out = decompose_query("query", _ErrChat())  # type: ignore[arg-type]
        assert out == ["query"]
        assert any(
            "decompose" in r.getMessage() and "boom" in r.getMessage()
            for r in caplog.records
        )

    def test_temporal_logs_on_chat_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="engram.retrieve"):
            anchor = compute_temporal_anchor(
                "yesterday",
                _ErrChat(),  # type: ignore[arg-type]
                now=datetime(2026, 5, 11, tzinfo=timezone.utc),
                max_retries=0,
            )
        assert anchor is None
        assert any(
            "temporal" in r.getMessage().lower()
            and "RuntimeError" in r.getMessage()
            for r in caplog.records
        )

    def test_react_judge_catches_and_logs_chat_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """H-53 fix: react_judge previously did not catch chat
        exceptions, so a provider blip broke iterative retrieve
        entirely. The judge now returns `sufficient=True` to stop
        iteration and logs a warning."""
        with caplog.at_level(logging.WARNING, logger="engram.retrieve"):
            verdict = react_judge(
                "q",
                [],
                _ErrChat(),  # type: ignore[arg-type]
                max_retries=0,
            )
        assert verdict.sufficient
        assert verdict.refined_query == ""
        # The message identifies the source and the exception class.
        assert any(
            "react_judge" in r.getMessage() and "RuntimeError" in r.getMessage()
            for r in caplog.records
        )


class TestDecomposeCap:
    """`decompose_query` caps the sub-query list at 5 -- matching the
    prompt's documented contract ("emit at most 5 sub-queries"). A
    non-compliant LLM that emits 100 lines should not produce 100
    leaf retrieves."""

    def test_cap_at_five(self) -> None:
        from engram.providers._fake import FakeChat

        # Eight lines -> 5 sub-queries (cap) + 1 (original) = 6 total.
        many = "\n".join(f"sub query {i}" for i in range(8))
        chat = FakeChat(default=many)
        out = decompose_query("q", chat)
        assert out[0] == "q"
        # Total length is 1 (original) + min(5, emitted) = 6.
        assert len(out) == 6


class TestMultiQuerySoftUpperBound:
    """`expand_queries` documents `n` as a soft upper bound. When the
    LLM under-fills, we don't pad with duplicates."""

    def test_under_fill_is_allowed(self) -> None:
        from engram.providers._fake import FakeChat

        # Asked for 5, LLM only emits 2.
        chat = FakeChat(default="p1\np2")
        out = expand_queries("q", 5, chat)
        # Original + 2 paraphrases = 3, even though we asked for 5.
        assert out == ["q", "p1", "p2"]
