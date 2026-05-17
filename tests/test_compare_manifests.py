"""Tests for benchmarks/compare_manifests.py.

These exercise the pure-function pieces (classification, length stats,
diffing, report-building) without touching the filesystem layer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_SCRIPT_PATH = _ROOT / "benchmarks" / "compare_manifests.py"

_spec = importlib.util.spec_from_file_location("compare_manifests", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def test_classify_response_empty() -> None:
    assert cm._classify_response("") == "empty"
    assert cm._classify_response("   \n  ") == "empty"
    assert cm._classify_response(None) == "non_str"


def test_classify_response_refusal() -> None:
    assert cm._classify_response("I don't know.") == "refusal"
    assert cm._classify_response("I do not know.") == "refusal"
    # Long refusal-prefix is no longer refusal (might contain answer)
    assert cm._classify_response("I don't know but " + "x" * 200) != "refusal"


def test_classify_response_cot_preamble() -> None:
    assert cm._classify_response("The user is asking about...") == "cot_preamble"
    assert cm._classify_response("Let me look at memory [1]") == "cot_preamble"
    assert cm._classify_response("Memory [1]: User said...") == "cot_preamble"


def test_classify_response_concrete() -> None:
    assert cm._classify_response("Business Administration.") == "concrete"
    assert cm._classify_response("7") == "concrete"


def test_classify_response_verbose_other() -> None:
    long = "Something happened. " * 50  # >400 chars, no CoT preamble
    assert cm._classify_response(long) == "verbose_other"


def test_overall_prefers_accuracy_correct() -> None:
    agg = {"accuracy": 0.68, "accuracy_correct": 0.685}
    assert cm._overall(agg) == 0.685


def test_overall_falls_back_to_accuracy() -> None:
    agg = {"accuracy": 0.714}
    assert cm._overall(agg) == 0.714


def test_qtype_accuracies_prefers_correct() -> None:
    agg = {
        "accuracy": 0.5,
        "accuracy_correct": 0.5,
        "accuracy_temporal-reasoning": 0.6,
        "accuracy_correct_temporal-reasoning": 0.61,
        "accuracy_multi-session": 0.7,
    }
    qt = cm._qtype_accuracies(agg)
    assert qt["temporal-reasoning"] == 0.61  # correct preferred
    assert qt["multi-session"] == 0.7  # fallback when no _correct variant
    assert "correct" not in qt  # the bare overall is skipped


def test_diff_questions_basic() -> None:
    base = [
        {
            "question_id": "q1",
            "question_type": "ms",
            "score": 1.0,
            "question": "A?",
            "gold": "yes",
            "response": "yes",
        },
        {
            "question_id": "q2",
            "question_type": "ms",
            "score": 0.0,
            "question": "B?",
            "gold": "yes",
            "response": "I don't know.",
        },
        {
            "question_id": "q3",
            "question_type": "ku",
            "score": 1.0,
            "question": "C?",
            "gold": "no",
            "response": "no",
        },
    ]
    new = [
        {
            "question_id": "q1",
            "question_type": "ms",
            "score": 1.0,
            "question": "A?",
            "gold": "yes",
            "response": "yes",
        },
        {
            "question_id": "q2",
            "question_type": "ms",
            "score": 1.0,
            "question": "B?",
            "gold": "yes",
            "response": "yes",
        },
        {
            "question_id": "q3",
            "question_type": "ku",
            "score": 0.0,
            "question": "C?",
            "gold": "no",
            "response": "",
        },
    ]
    d = cm._diff_questions(base, new)
    assert len(d["PP"]) == 1
    assert len(d["FP"]) == 1
    assert len(d["PF"]) == 1
    assert d["FP"][0]["qid"] == "q2"
    assert d["PF"][0]["qid"] == "q3"


def test_diff_questions_handles_disjoint_sets() -> None:
    base = [
        {
            "question_id": "x",
            "question_type": "ms",
            "score": 1.0,
            "question": "?",
            "gold": "g",
            "response": "r",
        }
    ]
    new = [
        {
            "question_id": "y",
            "question_type": "ms",
            "score": 0.0,
            "question": "?",
            "gold": "g",
            "response": "",
        }
    ]
    d = cm._diff_questions(base, new)
    assert len(d["PP"]) == 0
    assert len(d["only_in_base"]) == 1
    assert len(d["only_in_new"]) == 1


def test_shape_summary_counts_cliff() -> None:
    pq = [
        {"response": "x" * 4500},  # in cliff range
        {"response": "x" * 100},  # not cliff
        {"response": ""},  # empty
        {"response": "x" * 3700},  # cliff
    ]
    s = cm._shape_summary(pq)
    assert s["cliff_count"] == 2
    assert s["classes"]["empty"] == 1


def test_build_report_end_to_end(tmp_path: Path) -> None:
    """Synthetic two-manifest comparison renders without error and
    surfaces the headline delta + flip counts."""
    base_manifest = {
        "aggregate_metrics": {"accuracy": 0.5, "accuracy_multi-session": 0.5},
        "engram_config": {"k": 10},
        "git_commit": "aaaaaaa",
        "per_question": [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "score": 1.0,
                "question": "A?",
                "gold": "a",
                "response": "a",
            },
            {
                "question_id": "q2",
                "question_type": "multi-session",
                "score": 0.0,
                "question": "B?",
                "gold": "b",
                "response": "",
            },
        ],
    }
    new_manifest = {
        "aggregate_metrics": {"accuracy": 1.0, "accuracy_multi-session": 1.0},
        "engram_config": {"k": 20, "max_tokens": 8192},
        "git_commit": "bbbbbbb",
        "per_question": [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "score": 1.0,
                "question": "A?",
                "gold": "a",
                "response": "a",
            },
            {
                "question_id": "q2",
                "question_type": "multi-session",
                "score": 1.0,
                "question": "B?",
                "gold": "b",
                "response": "b",
            },
        ],
    }
    base_p = tmp_path / "base.json"
    new_p = tmp_path / "new.json"
    base_p.write_text(json.dumps(base_manifest))
    new_p.write_text(json.dumps(new_manifest))

    report = cm.build_report(base_p, new_p, base_manifest, new_manifest)
    assert "50.0% -> 100.0%" in report
    assert "+50.0pp" in report
    assert "FP (recovered failures, base failed but new passed): **1**" in report
    assert "max_tokens" in report  # config delta surfaces


def test_main_writes_to_outfile(tmp_path: Path) -> None:
    base = {"aggregate_metrics": {"accuracy": 0.5}, "per_question": []}
    new = {"aggregate_metrics": {"accuracy": 0.6}, "per_question": []}
    bp, np_ = tmp_path / "b.json", tmp_path / "n.json"
    out = tmp_path / "report.md"
    bp.write_text(json.dumps(base))
    np_.write_text(json.dumps(new))
    rc = cm.main([str(bp), str(np_), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "50.0% -> 60.0%" in text
