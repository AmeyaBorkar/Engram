"""Tests for the longmemeval answer-prompt variant infrastructure.

JOURNEY §24 shipped the cap fix (1024 → 65536 max_tokens) which lifted
the n=100 validation from 69% → 80%. Forensic on the 20 remaining
failures pointed at two cheap-to-fix patterns -- _abs questions
expecting "you didn't mention X but mentioned Y" and sss-preference
questions expecting a tailored synthesis. v3 attacks both:
  - base prompt: explanatory abstain instruction
  - qtype hint: sss-preference synthesis directive

These tests pin the v3 contract so a future ruff-format / prompt-edit
cleanup doesn't silently drop the hint or the explanatory-abstain line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# longmemeval lives in benchmarks/suites/, not on sys.path -- load it
# the same way the bench loader does.
_ROOT = Path(__file__).parent.parent
_SUITE_PATH = _ROOT / "benchmarks" / "suites" / "longmemeval.py"


def _load_suite_module():
    if "_longmemeval_test_module" in sys.modules:
        return sys.modules["_longmemeval_test_module"]
    # Ensure benchmarks/ is importable so any `from .X` imports inside
    # the suite resolve.  The suite is loaded as a path-based module
    # (no package context), so it should only use absolute imports.
    sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location("_longmemeval_test_module", _SUITE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_longmemeval_test_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_v3_registered_in_prompt_variants() -> None:
    """v3 must appear in _PROMPT_VARIANTS with the v3 template + hints map."""
    lme = _load_suite_module()
    assert "v3" in lme._PROMPT_VARIANTS
    template_version, hints = lme._PROMPT_VARIANTS["v3"]
    assert template_version == "v3"
    assert isinstance(hints, dict)


def test_v3_template_file_exists_and_has_required_slots() -> None:
    """v3 prompt file must exist and reference {memory}, {question},
    {question_date}, {qtype_hint} so .format() doesn't KeyError at runtime."""
    template = (
        _ROOT / "benchmarks" / "suites" / "prompts" / "longmemeval_answer_v3.txt"
    ).read_text(encoding="utf-8")
    for slot in ("{memory}", "{question}", "{question_date}", "{qtype_hint}"):
        assert slot in template, f"v3 template missing slot {slot!r}"


def test_v3_template_carries_explanatory_abstain_directive() -> None:
    """The whole point of v3 is to make abstain answers contextual.
    Pin the key phrasing so a prompt edit can't silently drop it."""
    template = (
        _ROOT / "benchmarks" / "suites" / "prompts" / "longmemeval_answer_v3.txt"
    ).read_text(encoding="utf-8")
    # Don't pin the exact wording (gives copywriting room), but pin that
    # the contextual-decline pattern is described.
    assert "RELATED information IS in memory" in template
    assert "did not mention X" in template
    # And that the bare "I don't know" pattern is explicitly discouraged
    assert 'NOT just say "I don\'t know"' in template


def test_v3_sss_preference_hint_directs_synthesis() -> None:
    """sss-preference hint must direct the model toward a synthesis
    starting with 'The user would prefer' rather than declining."""
    lme = _load_suite_module()
    hint = lme._V3_QTYPE_HINTS["single-session-preference"]
    assert "The user would prefer" in hint
    assert "Do NOT say 'I don't know'" in hint


def test_v3_does_not_carry_hints_for_lifted_qtypes() -> None:
    """The cap fix already moved multi-session / temporal-reasoning /
    knowledge-update +17 to +23 pp on n=100.  Adding qtype hints there
    risks overfitting the n=100 manifest.  Pin the explicit decision:
    only sss-preference gets a hint in v3."""
    lme = _load_suite_module()
    assert set(lme._V3_QTYPE_HINTS.keys()) == {"single-session-preference"}, (
        "v3 hint scope changed -- update this test AND JOURNEY §24 follow-up"
    )


def test_cli_parser_accepts_v3() -> None:
    """`--prompt-version v3` must be in the argparse choices."""
    from engram.bench._cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["run", "longmemeval", "--chat", "fake", "--embedder", "fake", "--prompt-version", "v3"]
    )
    assert args.prompt_version == "v3"


def test_cli_parser_rejects_unknown_prompt_version() -> None:
    """Regression guard: only the documented variants are accepted."""
    from engram.bench._cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "longmemeval",
                "--chat",
                "fake",
                "--embedder",
                "fake",
                "--prompt-version",
                "v99",
            ]
        )


def test_v3_template_renders_with_realistic_inputs() -> None:
    """Smoke: load the v3 template and .format() it with realistic
    inputs (including the sss-preference qtype hint).  Catches both
    template typos and a missing format slot."""
    lme = _load_suite_module()
    template_version, hints = lme._PROMPT_VARIANTS["v3"]
    raw = (
        _ROOT / "benchmarks" / "suites" / "prompts" / f"longmemeval_answer_{template_version}.txt"
    ).read_text(encoding="utf-8")
    rendered = raw.format(
        memory="[1] User said pizza is bad",
        question="What does the user prefer for dinner?",
        question_date="2024-01-15",
        qtype_hint=hints.get("single-session-preference", ""),
    )
    assert "User said pizza is bad" in rendered
    assert "What does the user prefer for dinner?" in rendered
    assert "The user would prefer" in rendered  # from the hint


def test_v3a_registered_in_prompt_variants() -> None:
    """v3a must appear in _PROMPT_VARIANTS with the v3a template + hints."""
    lme = _load_suite_module()
    assert "v3a" in lme._PROMPT_VARIANTS
    template_version, hints = lme._PROMPT_VARIANTS["v3a"]
    assert template_version == "v3a"
    assert isinstance(hints, dict)


def test_v3a_template_file_exists_and_has_required_slots() -> None:
    """v3a prompt file must exist with all format slots intact."""
    template = (
        _ROOT / "benchmarks" / "suites" / "prompts" / "longmemeval_answer_v3a.txt"
    ).read_text(encoding="utf-8")
    for slot in ("{memory}", "{question}", "{question_date}", "{qtype_hint}"):
        assert slot in template, f"v3a template missing slot {slot!r}"


def test_v3a_template_keeps_explanatory_abstain_directive() -> None:
    """v3a must inherit v3's explanatory-abstain pattern -- the abstain
    fix is the load-bearing change that lifted sss-user from 78.6%
    to 92.9% on n=100 (manifest e8682d19)."""
    template = (
        _ROOT / "benchmarks" / "suites" / "prompts" / "longmemeval_answer_v3a.txt"
    ).read_text(encoding="utf-8")
    assert "RELATED information IS in memory" in template
    assert "did not mention X" in template


def test_v3a_template_requires_counting_enumeration() -> None:
    """The whole point of v3a is to force enumeration on counting
    questions.  Pin the directive so a future prompt edit doesn't
    silently drop it."""
    template = (
        _ROOT / "benchmarks" / "suites" / "prompts" / "longmemeval_answer_v3a.txt"
    ).read_text(encoding="utf-8")
    # Don't pin the exact wording (gives copywriting room), but pin
    # that the count-enumerate pattern is described.
    lower = template.lower()
    assert "counting" in lower or "count" in lower
    assert "enumerate" in lower
    assert "before stating" in lower or "before the final" in lower or "before adding" in lower


def test_v3a_hints_extend_v3_with_multi_session() -> None:
    """v3a hints must include EVERYTHING in v3 (so we keep the
    sss-preference recovery) PLUS multi-session enumeration."""
    lme = _load_suite_module()
    v3_hints = lme._V3_QTYPE_HINTS
    v3a_hints = lme._V3A_QTYPE_HINTS
    # v3a is a superset of v3
    for qt, hint in v3_hints.items():
        assert qt in v3a_hints, f"v3a dropped v3 hint for {qt!r}"
        assert v3a_hints[qt] == hint, f"v3a changed v3 hint for {qt!r}"
    # And adds multi-session
    assert "multi-session" in v3a_hints
    ms_hint = v3a_hints["multi-session"]
    assert "enumerate" in ms_hint.lower()
    assert "off-by-one" in ms_hint.lower() or "off by one" in ms_hint.lower()


def test_cli_parser_accepts_v3a() -> None:
    """`--prompt-version v3a` must be in the argparse choices."""
    from engram.bench._cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["run", "longmemeval", "--chat", "fake", "--embedder", "fake", "--prompt-version", "v3a"]
    )
    assert args.prompt_version == "v3a"


def test_v3a_template_renders_with_multi_session_hint() -> None:
    """Smoke: render v3a with the multi-session hint and confirm
    enumeration guidance lands in the final prompt."""
    lme = _load_suite_module()
    template_version, hints = lme._PROMPT_VARIANTS["v3a"]
    raw = (
        _ROOT / "benchmarks" / "suites" / "prompts" / f"longmemeval_answer_{template_version}.txt"
    ).read_text(encoding="utf-8")
    rendered = raw.format(
        memory="[1] Bought apple. [2] Bought banana. [3] Bought cherry.",
        question="How many fruits did I buy?",
        question_date="2024-01-15",
        qtype_hint=hints.get("multi-session", ""),
    )
    assert "How many fruits did I buy?" in rendered
    assert "enumerate" in rendered.lower()
