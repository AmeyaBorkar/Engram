"""Smoke tests for benchmarks/re_judge.py.

Pin the strict-fair rubric footer text and the script's argument
contract so future ruff/refactor passes don't silently drop the
clarifications that recover the JOURNEY §26 judge-strict FNs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RE_JUDGE_PATH = _REPO_ROOT / "benchmarks" / "re_judge.py"


def _load_re_judge():
    if "_test_re_judge_module" in sys.modules:
        return sys.modules["_test_re_judge_module"]
    sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_test_re_judge_module", _RE_JUDGE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_re_judge_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_re_judge_module_loads() -> None:
    mod = _load_re_judge()
    assert hasattr(mod, "main")
    assert hasattr(mod, "_STRICT_FAIR_FOOTER")
    assert hasattr(mod, "_re_judge_one")
    assert hasattr(mod, "_ReJudgeResult")


def test_strict_fair_footer_carries_three_clarifications() -> None:
    """The strict-fair rubric must address the three judge-strict FN
    patterns identified in the n=500 v3a manifest forensic:
      (a) embedded gold value in verbose response
      (b) equivalent abstain paraphrases
      (c) pronoun drift
    """
    mod = _load_re_judge()
    footer = mod._STRICT_FAIR_FOOTER
    # Pin presence of each clarification
    assert "verbatim" in footer or "embedded" in footer or "embedded" in footer.lower(), (
        "footer must address case (a): embedded gold value"
    )
    assert "abstain" in footer.lower(), "footer must address case (b)"
    assert "pronoun" in footer.lower(), "footer must address case (c)"


def test_cli_parser_requires_manifest() -> None:
    mod = _load_re_judge()
    with pytest.raises(SystemExit):
        mod.main([])  # no --manifest


def test_cli_parser_defaults() -> None:
    """Defaults must point at a dated snapshot, not the floating alias."""
    mod = _load_re_judge()
    # We can't easily inspect argparse defaults without running -- but we
    # can check the helptext mentions the snapshot pin.
    src = _RE_JUDGE_PATH.read_text(encoding="utf-8")
    assert "openai/gpt-4o-2024-08-06" in src, (
        "default judge model must be a dated snapshot, not 'openai/gpt-4o'"
    )
    # And that the help mentions snapshot pinning
    assert "drift" in src.lower() or "pinned" in src.lower()


def test_re_judge_one_skips_errored_records() -> None:
    """Records with an `error` field or empty response must short-circuit
    -- they can't flip and we shouldn't waste a judge call on them."""
    mod = _load_re_judge()

    # Stub suite module attributes the function reaches into
    class FakeSuite:
        _JUDGE_INSTRUCTIONS = {"multi-session": "fake instructions"}

        @staticmethod
        def _read_prompt(name: str) -> str:
            return "fake prompt {instructions} {question} {gold} {response}"

        @staticmethod
        def _parse_judge_verdict(raw: str) -> bool:
            return False

        class Message:  # type: ignore[no-untyped-def]
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

    class FakeChat:
        def chat(self, _messages: list) -> str:
            raise AssertionError("chat must not be called for errored records")

    record = {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "Q?",
        "gold": "G",
        "response": "",
        "score": 0.0,
        "error": "BadRequestError",
    }
    result = mod._re_judge_one(
        suite=FakeSuite,
        chat=FakeChat(),
        record=record,
        use_strict_fair=False,
    )
    assert result.new_score == result.old_score == 0.0
    assert "skipped" in result.new_raw.lower()


def test_re_judge_one_calls_chat_for_normal_records() -> None:
    """Verify the normal path actually invokes the judge."""
    mod = _load_re_judge()
    calls: list[str] = []

    class FakeSuite:
        _JUDGE_INSTRUCTIONS = {"multi-session": "answer yes or no"}

        @staticmethod
        def _read_prompt(name: str) -> str:
            return "{instructions}\n{question}\n{gold}\n{response}"

        @staticmethod
        def _parse_judge_verdict(raw: str) -> bool:
            return raw.strip().lower().startswith("yes")

        class Message:  # type: ignore[no-untyped-def]
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

    class FakeChat:
        def chat(self, messages: list) -> str:
            calls.append(messages[0].content)
            return "yes"  # flip wrong -> right

    record = {
        "question_id": "q2",
        "question_type": "multi-session",
        "question": "How many?",
        "gold": "3",
        "response": "...3 total.",
        "score": 0.0,
    }
    result = mod._re_judge_one(
        suite=FakeSuite,
        chat=FakeChat(),
        record=record,
        use_strict_fair=False,
    )
    assert len(calls) == 1
    assert result.old_score == 0.0
    assert result.new_score == 1.0


def test_re_judge_strict_fair_appends_footer_to_prompt() -> None:
    """The strict-fair flag must append clarifications to the judge prompt
    so we can actually USE the recovery footer end-to-end."""
    mod = _load_re_judge()
    captured: dict[str, str] = {}

    class FakeSuite:
        _JUDGE_INSTRUCTIONS = {"knowledge-update": "base rubric"}

        @staticmethod
        def _read_prompt(name: str) -> str:
            return "{instructions}|||{question}|||{gold}|||{response}"

        @staticmethod
        def _parse_judge_verdict(raw: str) -> bool:
            return True

        class Message:  # type: ignore[no-untyped-def]
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

    class CaptureChat:
        def chat(self, messages: list) -> str:
            captured["prompt"] = messages[0].content
            return "yes"

    record = {
        "question_id": "q3",
        "question_type": "knowledge-update",
        "question": "?",
        "gold": "G",
        "response": "R",
        "score": 0.0,
    }
    mod._re_judge_one(
        suite=FakeSuite,
        chat=CaptureChat(),
        record=record,
        use_strict_fair=True,
    )
    assert "Clarifications:" in captured["prompt"]
    assert "verbatim" in captured["prompt"] or "embedded" in captured["prompt"].lower()
