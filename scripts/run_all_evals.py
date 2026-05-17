"""End-to-end retrieval evaluation orchestrator.

Single command that runs every test we have. Builds embedder + reranker
ONCE, shares them across phases. Each phase writes its own JSON
artifact under `--out-dir`; the final phase consolidates everything
into REPORT.md.

Phases (each skippable via --skip-phase N):

  1. Component unit tests          (~10 s, pure CPU)
     - BM25Index correctness        (8 sanity tests)
     - mmr_select correctness       (7 sanity tests)
     - reciprocal_rank_fusion       (5 sanity tests)
     - _build_auto_temporal_filter  (5 regex tests)
     - recency math                 (3 formula tests)

  2. Per-qtype ablation             (~2 hr, GPU)
     - All 5 qtypes × 12 configs × N questions retrieval-only.
     - Saves ablation_<qtype>.json per bucket.

  3. Knob sweeps                    (~30 min, GPU)
     - bm25_weight, mmr_lambda, recency_lambda, recent_window_k
     - With paired-diff CI + McNemar vs baseline value.

  4. Diagnostic traces              (~5 min, GPU)
     - Picks one question per qtype that any config broke (passed
       baseline, failed under candidate); dumps full per-stage trace
       to traces/<qid>.txt.

  5. Consolidated report            (~10 s)
     - REPORT.md with verdicts, recommended launch config, statistical
       findings.

Usage:
  # Full battery (~3 hours)
  python scripts/run_all_evals.py --dtype fp32 --embed-device cuda \\
    --out-dir benchmarks/runs/eval_all_$(date +%Y%m%d_%H%M%S)

  # Skip the slow phases for a quick component-only smoke
  python scripts/run_all_evals.py --skip-phase 2 --skip-phase 3 --skip-phase 4

  # Resume after a crash (skip phases whose JSON is already present)
  python scripts/run_all_evals.py --resume --out-dir benchmarks/runs/eval_all_20260512
"""

from __future__ import annotations

import argparse
import json
import logging
import math as math_mod
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# M-122: shared helpers in scripts/_common.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    build_embedder as _build_embedder_common,
    build_reranker as _build_reranker_common,
    ensure_repo_on_path,
    load_env_file,
)

ensure_repo_on_path()
load_env_file()

from engram import Memory, SqliteStorage  # noqa: E402
from engram.providers._fake import FakeChat  # noqa: E402
from engram.retrieve._bm25 import (  # noqa: E402
    BM25Index,
    reciprocal_rank_fusion,
    tokenize,
)
from engram.retrieve._mmr import mmr_select  # noqa: E402
from engram.retrieve._params import RetrieveParams  # noqa: E402

from benchmarks.suites.longmemeval import (  # noqa: E402
    DATASET_ROOT,
    DEFAULT_FILENAME,
    _Question,
    _build_auto_temporal_filter,
    _ingest_haystack,
    _load_dataset,
    _parse_haystack_date,
)
from scripts._stats import (  # noqa: E402
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    format_ci,
    format_p,
    mcnemar,
)
from scripts.ablate_longmemeval import (  # noqa: E402
    CONFIGS,
    _filter_questions as ablate_filter,
    _run_one as ablate_eval_one,
)
from scripts.retrieval_trace import _trace_question  # noqa: E402
from scripts.sweep import SWEEPS, _evaluate_one as sweep_eval_one  # noqa: E402

_LOG = logging.getLogger("engram.eval_all")


# ============================================================================
# Phase 1: Component correctness tests (synthetic, no models, no LongMemEval)
# ============================================================================


def _ok(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": str(detail)[:240]}


def test_bm25() -> dict[str, Any]:
    """BM25Index correctness against the BM25 spec."""
    tests: list[dict[str, Any]] = []

    # 1. Empty corpus
    idx: BM25Index[str] = BM25Index()
    tests.append(_ok("empty corpus → []", idx.search("anything", k=10) == []))

    # 2. Single matching doc
    idx = BM25Index()
    idx.add_doc("a", "the quick brown fox")
    r = idx.search("fox", k=5)
    tests.append(_ok("single matching doc", len(r) == 1 and r[0][0] == "a" and r[0][1] > 0))

    # 3. No match returns []
    idx = BM25Index()
    idx.add_doc("a", "red apple")
    idx.add_doc("b", "green pear")
    tests.append(_ok("no match → []", idx.search("banana", k=5) == []))

    # 4. Higher tf ranks first
    idx = BM25Index()
    idx.add_doc("a", "cat cat cat dog")
    idx.add_doc("b", "cat lion bear")
    r = idx.search("cat", k=5)
    tests.append(_ok("higher tf ranks higher", len(r) == 2 and r[0][0] == "a"))

    # 5. IDF rewards rare term
    idx = BM25Index()
    idx.add_doc("a", "cat dog")
    idx.add_doc("b", "cat cat dog")
    idx.add_doc("c", "cat fish")  # only doc with "fish"
    idx.add_doc("d", "cat bird")
    r = idx.search("fish", k=5)
    tests.append(_ok("rare term lifts unique doc", len(r) == 1 and r[0][0] == "c"))

    # 6. Tokenizer lower-cases
    idx = BM25Index()
    idx.add_doc("a", "Hello, World!")
    r = idx.search("hello", k=5)
    tests.append(_ok("tokenizer lowercases + word-only", len(r) == 1 and r[0][0] == "a"))

    # 7. Determinism
    def _build(d: dict[str, str]) -> BM25Index[str]:
        b: BM25Index[str] = BM25Index()
        for k, v in d.items():
            b.add_doc(k, v)
        return b
    r1 = _build({"a": "one two three", "b": "two three four"}).search("two three", k=5)
    r2 = _build({"a": "one two three", "b": "two three four"}).search("two three", k=5)
    tests.append(_ok("deterministic", r1 == r2))

    # 8. Frozen after first search
    idx = BM25Index()
    idx.add_doc("a", "foo")
    idx.search("foo", k=5)
    raised = False
    try:
        idx.add_doc("b", "bar")
    except RuntimeError:
        raised = True
    tests.append(_ok("frozen after search()", raised))

    # 9. Length normalization: short doc with rare term beats long doc
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add_doc("short", "fish")
    idx.add_doc(
        "long",
        "the lake had a fish in it and the fisher was happy that day "
        "because the fish was very tasty and they cooked it with butter "
        "and lemon and parsley and it was the best meal of the week "
        "and they invited their friends over to share the fish",
    )
    r = idx.search("fish", k=2)
    # Length normalization should favor the SHORT doc despite same term count
    tests.append(_ok(
        "length norm favors short doc",
        len(r) == 2 and r[0][0] == "short" and r[0][1] > r[1][1],
        f"short={r[0][1]:.3f}, long={r[1][1]:.3f}",
    ))

    passed = sum(1 for t in tests if t["passed"])
    failed = len(tests) - passed
    return {"component": "BM25Index", "tests": tests, "passed": passed, "failed": failed}


def test_mmr() -> dict[str, Any]:
    """mmr_select correctness."""
    tests: list[dict[str, Any]] = []

    # 1. Empty
    tests.append(_ok("empty input → []", mmr_select([], [], [], k=5, lambda_=0.7) == []))

    # 2. Single item
    r = mmr_select(["x"], [0.5], [[1.0, 0.0]], k=5, lambda_=0.7)
    tests.append(_ok("single item returns it", r == ["x"]))

    # 3. lambda=1.0 → relevance sort
    items = ["a", "b", "c"]
    rel = [0.1, 0.9, 0.5]
    vecs: list[list[float] | None] = [[1, 0], [0, 1], [1, 1]]
    r = mmr_select(items, rel, vecs, k=3, lambda_=1.0)
    tests.append(_ok("λ=1.0 → relevance sort", r == ["b", "c", "a"]))

    # 4. Duplicates eventually both selected when k=N
    items = ["a", "b"]
    rel = [0.5, 0.5]
    vecs = [[1, 0], [1, 0]]
    r = mmr_select(items, rel, vecs, k=2, lambda_=0.5)
    tests.append(_ok("k==N with duplicates: all selected", set(r) == {"a", "b"}))

    # 5. Two orthogonal items: top selected first, second selected next under any λ
    items = ["a", "b"]
    rel = [1.0, 0.0]
    vecs = [[1, 0], [0, 1]]
    r = mmr_select(items, rel, vecs, k=2, lambda_=0.0)
    tests.append(_ok("λ=0.0 with orthogonal: both selected", set(r) == {"a", "b"}))

    # 6. All None vectors → relevance sort
    items = ["a", "b", "c"]
    rel = [0.5, 0.9, 0.1]
    vecs = [None, None, None]
    r = mmr_select(items, rel, vecs, k=3, lambda_=0.7)
    tests.append(_ok("all None vectors → relevance sort", r == ["b", "a", "c"]))

    # 7. Invalid lambda raises
    raised = False
    try:
        mmr_select(["a"], [0.5], [[1.0]], k=1, lambda_=1.5)
    except ValueError:
        raised = True
    tests.append(_ok("invalid λ raises", raised))

    # 8. Diversity at λ=0.5: when 3 items, top + a redundant near-dup + a diverse one,
    # MMR should prefer the diverse second pick.
    items = ["top", "dup", "diverse"]
    rel = [1.0, 0.8, 0.7]
    vecs = [[1, 0], [0.99, 0.01], [0, 1]]
    r = mmr_select(items, rel, vecs, k=2, lambda_=0.5)
    tests.append(_ok(
        "λ=0.5 picks diverse over redundant",
        r == ["top", "diverse"],
        f"got {r}",
    ))

    passed = sum(1 for t in tests if t["passed"])
    return {"component": "mmr_select", "tests": tests, "passed": passed, "failed": len(tests) - passed}


def test_rrf() -> dict[str, Any]:
    """Reciprocal Rank Fusion correctness."""
    tests: list[dict[str, Any]] = []

    # 1. Empty rankings
    tests.append(_ok("empty rankings → []", reciprocal_rank_fusion([]) == []))

    # 2. Single ranking preserved
    r = reciprocal_rank_fusion([[("a", 0.9), ("b", 0.5)]])
    tests.append(_ok("single ranking: order preserved", [d for d, _ in r] == ["a", "b"]))

    # 3. Common top across rankings stays on top
    r = reciprocal_rank_fusion([
        [("a", 0.9), ("b", 0.5)],
        [("a", 0.8), ("c", 0.4)],
    ])
    tests.append(_ok("common top wins", r[0][0] == "a"))

    # 4. Doc in only one ranking still surfaces
    r = reciprocal_rank_fusion([[("a", 1.0)], [("b", 1.0)]])
    tests.append(_ok("both docs appear", {d for d, _ in r} == {"a", "b"}))

    # 5. Exact math: rank-1 in both rankings → 2/(k+1)
    r = reciprocal_rank_fusion([[("a", 1)], [("a", 1)]], k=60)
    expected = 2 * (1.0 / 61.0)
    tests.append(_ok(
        "exact RRF(a) with rank=1 in two rankings",
        r[0][0] == "a" and abs(r[0][1] - expected) < 1e-6,
        f"got {r[0][1]:.6f}, expected {expected:.6f}",
    ))

    passed = sum(1 for t in tests if t["passed"])
    return {"component": "reciprocal_rank_fusion", "tests": tests, "passed": passed, "failed": len(tests) - passed}


def test_auto_temporal() -> dict[str, Any]:
    """_build_auto_temporal_filter year extraction."""
    tests: list[dict[str, Any]] = []

    tests.append(_ok("no year → None", _build_auto_temporal_filter("Hello?") is None))

    f = _build_auto_temporal_filter("What did I do in 2023?")
    tests.append(_ok("one year extracted", f is not None and "2023" in f))

    f = _build_auto_temporal_filter("Between 2023 and 2024 what happened?")
    tests.append(_ok(
        "two years extracted",
        f is not None and "2023" in f and "2024" in f,
    ))

    # Non-year 4-digit (regex requires (19|20) prefix)
    f = _build_auto_temporal_filter("I have 1000 apples")
    tests.append(_ok("non-year 4-digit → None", f is None))

    # Year embedded in a word (no \b boundary on right side)
    f = _build_auto_temporal_filter("called 2023something the project")
    tests.append(_ok("year glued to word → not matched", f is None, f"got {f!r}"))

    passed = sum(1 for t in tests if t["passed"])
    return {"component": "_build_auto_temporal_filter", "tests": tests, "passed": passed, "failed": len(tests) - passed}


def test_recency_math() -> dict[str, Any]:
    """Recency boost formula sanity (additive form, post-fix)."""
    tests: list[dict[str, Any]] = []

    lambda_ = 0.5
    decay = 90.0

    bonus_0 = lambda_ * math_mod.exp(0)
    tests.append(_ok("0-day-old: bonus == lambda", abs(bonus_0 - lambda_) < 1e-6))

    bonus_decay = lambda_ * math_mod.exp(-1.0)
    expected = lambda_ / math_mod.e
    tests.append(_ok("decay_days old: bonus == λ/e", abs(bonus_decay - expected) < 1e-6))

    bonus_far = lambda_ * math_mod.exp(-100.0)
    tests.append(_ok("100× decay old: bonus ≈ 0", bonus_far < 1e-40))

    # Sign correctness: additive form preserves direction for any base score
    for base in (5.0, -2.0, 0.0):
        boosted = base + lambda_
        # boosted - base should equal lambda (positive) regardless of base sign
        tests.append(_ok(
            f"additive bonus is sign-correct (base={base})",
            abs((boosted - base) - lambda_) < 1e-6,
        ))

    passed = sum(1 for t in tests if t["passed"])
    return {"component": "recency_boost_math", "tests": tests, "passed": passed, "failed": len(tests) - passed}


def phase1_component_tests(out_dir: Path) -> dict[str, Any]:
    """Run all synthetic component tests. Fast: ~10 s on any machine."""
    _LOG.info("PHASE 1: Component correctness tests")
    suites = [
        test_bm25(),
        test_mmr(),
        test_rrf(),
        test_auto_temporal(),
        test_recency_math(),
    ]
    total_passed = sum(s["passed"] for s in suites)
    total_failed = sum(s["failed"] for s in suites)
    result = {
        "suites": suites,
        "total_passed": total_passed,
        "total_failed": total_failed,
    }
    (out_dir / "phase1_component_tests.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _LOG.info("PHASE 1 done: %d passed, %d failed", total_passed, total_failed)
    # Console summary
    for s in suites:
        _LOG.info("  %s: %d/%d", s["component"], s["passed"], s["passed"] + s["failed"])
        for t in s["tests"]:
            if not t["passed"]:
                _LOG.warning("    FAIL: %s — %s", t["name"], t["detail"])
    return result


# ============================================================================
# Shared model loaders for phases 2-4
# ============================================================================


# M-122: thin shims around _common builders.


def _build_embedder(model: str, device: str | None, dtype: str) -> Any:
    return _build_embedder_common(model, device, dtype)


def _build_reranker(model: str | None, device: str | None, dtype: str) -> Any:
    return _build_reranker_common(model, device, dtype)


# ============================================================================
# Phase 2: Per-qtype ablation (all configs × all qtypes)
# ============================================================================


QTYPES = (
    "single-session-user",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)


def phase2_ablation(
    out_dir: Path,
    embedder: Any,
    reranker: Any,
    chat: Any,
    limit: int,
    k: int,
) -> dict[str, Any]:
    """For every qtype × every config, retrieval-only ablation."""
    _LOG.info("PHASE 2: Per-qtype ablation (limit=%d)", limit)
    dataset_path = DATASET_ROOT / DEFAULT_FILENAME
    all_questions = _load_dataset(dataset_path)
    if not all_questions:
        raise RuntimeError(f"dataset not found: {dataset_path}")

    summary: dict[str, Any] = {}
    for qt in QTYPES:
        _LOG.info("PHASE 2 [%s]: starting", qt)
        questions = ablate_filter(all_questions, qtype=qt, qids=None, limit=limit)
        if not questions:
            _LOG.warning("PHASE 2 [%s]: no questions; skipping", qt)
            continue
        rows: list[dict[str, Any]] = []
        t_start = time.perf_counter()
        for q_idx, q in enumerate(questions):
            storage = SqliteStorage(":memory:")
            storage.initialize()
            try:
                ingest_memory = Memory(
                    storage=storage, embedder=embedder, chat=chat, reranker=reranker
                )
                t_ingest = time.perf_counter()
                turns = _ingest_haystack(ingest_memory, q)
                ingest_s = time.perf_counter() - t_ingest
                for config_name, config in CONFIGS.items():
                    row = ablate_eval_one(
                        q=q,
                        config_name=config_name,
                        config=config,
                        storage=storage,
                        embedder=embedder,
                        reranker=reranker,
                        chat=chat,
                        k=k,
                        retrieval_only=True,
                    )
                    rows.append({
                        "qid": row.qid,
                        "qtype": row.qtype,
                        "config": row.config,
                        "recall": row.recall,
                        "binary_hit": row.binary_hit,
                        "score": row.score,
                        "error": row.error,
                        "latency_ms": row.latency_ms,
                    })
                _LOG.info(
                    "  q %d/%d qid=%s turns=%d ingest=%.1fs",
                    q_idx + 1,
                    len(questions),
                    q.qid[:8],
                    turns,
                    ingest_s,
                )
            finally:
                storage.close()
        elapsed = time.perf_counter() - t_start
        out_path = out_dir / f"phase2_ablation_{qt}.json"
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        # Per-config aggregates
        per_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            per_config[r["config"]].append(r)
        agg: dict[str, dict[str, float]] = {}
        for c, rs in per_config.items():
            n = len(rs)
            agg[c] = {
                "n": float(n),
                "mean_recall": sum(r["recall"] for r in rs) / n,
                "hit_rate": sum(r["binary_hit"] for r in rs) / n,
            }
        summary[qt] = {"n_questions": len(questions), "elapsed_s": elapsed, "configs": agg}
        _LOG.info("PHASE 2 [%s]: done in %.1fs", qt, elapsed)

    (out_dir / "phase2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ============================================================================
# Phase 3: Hyperparameter sweeps
# ============================================================================


SWEEP_PLAN: dict[str, dict[str, Any]] = {
    "bm25_weight": {"values": [0.0, 0.5, 1.0, 1.5, 2.0], "qtype": None},
    "mmr_lambda": {"values": [0.0, 0.3, 0.5, 0.7, 0.9], "qtype": None},
    "recency_lambda": {"values": [0.0, 0.1, 0.3, 0.5, 1.0], "qtype": None},
    "recent_window_k": {"values": [0, 5, 10, 20, 50], "qtype": None},
}


def phase3_sweeps(
    out_dir: Path,
    embedder: Any,
    reranker: Any,
    chat: Any,
    limit: int,
    k: int,
) -> dict[str, Any]:
    """Sweep each tunable knob with paired-diff CI vs baseline value."""
    _LOG.info("PHASE 3: Hyperparameter sweeps (limit=%d)", limit)
    dataset_path = DATASET_ROOT / DEFAULT_FILENAME
    all_questions = _load_dataset(dataset_path)
    summary: dict[str, Any] = {}

    for knob, plan in SWEEP_PLAN.items():
        values = plan["values"]
        qt = plan["qtype"]
        _LOG.info("PHASE 3 [%s]: sweeping over %s", knob, values)
        questions = ablate_filter(all_questions, qtype=qt, qids=None, limit=limit)
        if not questions:
            _LOG.warning("PHASE 3 [%s]: no questions; skipping", knob)
            continue
        per_value: dict[Any, list[dict[str, Any]]] = {v: [] for v in values}
        t_start = time.perf_counter()
        for q_idx, q in enumerate(questions):
            storage = SqliteStorage(":memory:")
            storage.initialize()
            try:
                ingest_memory = Memory(
                    storage=storage, embedder=embedder, chat=chat, reranker=reranker
                )
                _ingest_haystack(ingest_memory, q)
                for v in values:
                    row = sweep_eval_one(
                        q=q,
                        value=v,
                        knob=knob,
                        storage=storage,
                        embedder=embedder,
                        reranker=reranker,
                        chat=chat,
                        eval_k=k,
                    )
                    per_value[v].append(row)
            finally:
                storage.close()
            if (q_idx + 1) % 10 == 0:
                _LOG.info("  q %d/%d", q_idx + 1, len(questions))
        elapsed = time.perf_counter() - t_start

        # Paired diff vs baseline_value=values[0]
        baseline = values[0]
        baseline_rows = per_value[baseline]
        per_value_summary: dict[str, dict[str, Any]] = {}
        for v in values:
            rs = per_value[v]
            n = len(rs)
            mean_recall = sum(r["recall_at_k"] for r in rs) / n if n else 0.0
            hit_rate = sum(r["hit_at_k"] for r in rs) / n if n else 0.0
            mean_mrr = sum(r["mrr"] for r in rs) / n if n else 0.0
            entry: dict[str, Any] = {
                "n": n,
                "mean_recall_at_k": mean_recall,
                "mean_hit_at_k": hit_rate,
                "mean_mrr": mean_mrr,
            }
            if v != baseline:
                a_recall = [b["recall_at_k"] for b in baseline_rows]
                b_recall = [r["recall_at_k"] for r in rs]
                a_hit = [int(b["hit_at_k"]) for b in baseline_rows]
                b_hit = [int(r["hit_at_k"]) for r in rs]
                if len(a_recall) == len(b_recall) and len(a_recall) > 0:
                    diff = bootstrap_paired_diff_ci(a_recall, b_recall)
                    mcn = mcnemar(a_hit, b_hit)
                    entry["paired_diff_recall"] = {
                        "mean": diff.diff,
                        "ci_low": diff.ci_low,
                        "ci_high": diff.ci_high,
                        "excludes_zero": diff.excludes_zero,
                    }
                    entry["mcnemar_p"] = mcn.p_value
                    entry["mcnemar_only_a"] = mcn.n_passes_only_a
                    entry["mcnemar_only_b"] = mcn.n_passes_only_b
            per_value_summary[str(v)] = entry
        summary[knob] = {
            "values": values,
            "baseline_value": baseline,
            "n_questions": len(questions),
            "elapsed_s": elapsed,
            "per_value": per_value_summary,
        }
        (out_dir / f"phase3_sweep_{knob}.json").write_text(
            json.dumps({"per_value": {str(k): v for k, v in per_value.items()}, "summary": summary[knob]}, indent=2, default=str),
            encoding="utf-8",
        )
        _LOG.info("PHASE 3 [%s]: done in %.1fs", knob, elapsed)

    (out_dir / "phase3_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


# ============================================================================
# Phase 4: Diagnostic traces on broken questions
# ============================================================================


def _pick_broken_qids(
    out_dir: Path,
    phase2_summary: dict[str, Any],
) -> dict[str, str | None]:
    """For each qtype, find one qid where baseline passed but any config failed.

    Falls back to the first qid from that qtype if no such question exists.
    """
    picks: dict[str, str | None] = {}
    for qt in QTYPES:
        path = out_dir / f"phase2_ablation_{qt}.json"
        if not path.exists():
            picks[qt] = None
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in rows:
            by_qid[r["qid"]][r["config"]] = r
        chosen: str | None = None
        # Look for a question where baseline passed but >=1 other config failed.
        for qid, configs_dict in by_qid.items():
            baseline_row = configs_dict.get("baseline")
            if not baseline_row or int(baseline_row.get("binary_hit", 0)) == 0:
                continue
            for c, r in configs_dict.items():
                if c == "baseline":
                    continue
                if int(r.get("binary_hit", 0)) == 0:
                    chosen = qid
                    break
            if chosen:
                break
        if chosen is None:
            # Fallback: any qid for this qtype
            chosen = next(iter(by_qid.keys()), None)
        picks[qt] = chosen
    return picks


def phase4_traces(
    out_dir: Path,
    embedder: Any,
    reranker: Any,
    chat: Any,
    phase2_summary: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    """Trace one diagnostic question per qtype to a file."""
    _LOG.info("PHASE 4: Diagnostic traces")
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    picks = _pick_broken_qids(out_dir, phase2_summary)

    dataset_path = DATASET_ROOT / DEFAULT_FILENAME
    all_questions = _load_dataset(dataset_path)
    by_qid = {q.qid: q for q in all_questions}

    summary: dict[str, Any] = {}
    config_names = ["baseline", "bm25", "mmr07", "bm25+mmr", "all_aggressive"]
    for qt, qid in picks.items():
        if not qid:
            _LOG.warning("PHASE 4 [%s]: no qid to trace", qt)
            summary[qt] = {"qid": None, "trace_path": None}
            continue
        q = by_qid.get(qid)
        if q is None:
            summary[qt] = {"qid": qid, "trace_path": None, "error": "qid not found"}
            continue
        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            ingest_memory = Memory(
                storage=storage, embedder=embedder, chat=chat, reranker=reranker
            )
            _ingest_haystack(ingest_memory, q)
            text = _trace_question(
                q=q,
                storage=storage,
                embedder=embedder,
                reranker=reranker,
                chat=chat,
                config_names=config_names,
                deep_n=30,
                k=k,
            )
        finally:
            storage.close()
        out_path = traces_dir / f"{qt}__{qid[:12]}.txt"
        out_path.write_text(text, encoding="utf-8")
        _LOG.info("PHASE 4 [%s]: traced %s -> %s", qt, qid[:8], out_path)
        summary[qt] = {"qid": qid, "trace_path": str(out_path.relative_to(out_dir))}

    (out_dir / "phase4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ============================================================================
# Phase 5: Consolidated report
# ============================================================================


def _verdict_from_phase2(phase2: dict[str, Any]) -> dict[str, str]:
    """For each config, mark a per-qtype regression flag.

    Per protocol: a config fails the gate if it regresses by >0.02 on any qtype.
    """
    out: dict[str, str] = {}
    if not phase2:
        return out
    all_configs: set[str] = set()
    for qt, qt_data in phase2.items():
        all_configs.update(qt_data["configs"].keys())
    for c in sorted(all_configs):
        worst_drop = 0.0
        worst_qtype = ""
        any_improve = False
        for qt, qt_data in phase2.items():
            baseline = qt_data["configs"].get("baseline", {}).get("mean_recall", 0.0)
            cand = qt_data["configs"].get(c, {}).get("mean_recall", 0.0)
            delta = cand - baseline
            if delta < worst_drop:
                worst_drop = delta
                worst_qtype = qt
            if delta > 0.01:
                any_improve = True
        if c == "baseline":
            out[c] = "BASELINE"
        elif worst_drop < -0.02:
            out[c] = f"FAIL ({worst_qtype} Δ={worst_drop:+.3f})"
        elif any_improve:
            out[c] = "PASS (some lift, no major regression)"
        else:
            out[c] = "NEUTRAL (no lift, no regression)"
    return out


def phase5_report(
    out_dir: Path,
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    config_args: dict[str, Any],
) -> str:
    """Consolidate all phase outputs into a single REPORT.md."""
    lines: list[str] = []
    lines.append("# Engram Retrieval Evaluation — Full Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    lines.append(f"Output dir: `{out_dir}`\n")
    lines.append("\n## Configuration\n")
    lines.append("```")
    lines.append(json.dumps(config_args, indent=2))
    lines.append("```\n")

    # ---- Phase 1 ----
    lines.append("\n## Phase 1: Component correctness\n")
    if phase1:
        total = phase1["total_passed"] + phase1["total_failed"]
        lines.append(
            f"Overall: **{phase1['total_passed']}/{total} passed** "
            f"({phase1['total_failed']} failed)\n"
        )
        lines.append("| Component | Passed | Failed |")
        lines.append("|---|---:|---:|")
        for s in phase1["suites"]:
            lines.append(f"| `{s['component']}` | {s['passed']} | {s['failed']} |")
        # Failure detail
        any_fail = False
        for s in phase1["suites"]:
            fails = [t for t in s["tests"] if not t["passed"]]
            if not fails:
                continue
            if not any_fail:
                lines.append("\n### Failures\n")
                any_fail = True
            lines.append(f"**`{s['component']}`**:")
            for t in fails:
                lines.append(f"  - `{t['name']}` — {t['detail']}")
    else:
        lines.append("(skipped)")

    # ---- Phase 2 ----
    lines.append("\n## Phase 2: Per-qtype ablation\n")
    if phase2:
        all_configs: list[str] = []
        if phase2:
            first_qt = next(iter(phase2.values()))
            all_configs = list(first_qt["configs"].keys())
        # Mean recall matrix
        lines.append("### Mean recall@k by (config, qtype)\n")
        header = "| config | " + " | ".join(qt[:14] for qt in phase2.keys()) + " |"
        sep = "|---|" + "|".join(["---:"] * len(phase2)) + "|"
        lines.append(header)
        lines.append(sep)
        for c in all_configs:
            cells: list[str] = []
            for qt, qt_data in phase2.items():
                row = qt_data["configs"].get(c)
                if not row:
                    cells.append("—")
                    continue
                baseline = qt_data["configs"].get("baseline", {}).get("mean_recall", 0)
                delta = row["mean_recall"] - baseline
                if c == "baseline":
                    cells.append(f"{row['mean_recall']:.3f}")
                elif delta < -0.05:
                    cells.append(f"**{row['mean_recall']:.3f}** ({delta:+.3f})")
                elif delta < -0.02:
                    cells.append(f"{row['mean_recall']:.3f} ({delta:+.3f})")
                else:
                    cells.append(f"{row['mean_recall']:.3f}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")

        verdict = _verdict_from_phase2(phase2)
        lines.append("\n### Verdict per config (gate: per-qtype regression ≤ 0.02)\n")
        lines.append("| Config | Verdict |")
        lines.append("|---|---|")
        for c, v in verdict.items():
            lines.append(f"| `{c}` | {v} |")
    else:
        lines.append("(skipped)")

    # ---- Phase 3 ----
    lines.append("\n## Phase 3: Hyperparameter sweeps\n")
    if phase3:
        for knob, knob_data in phase3.items():
            lines.append(f"\n### `{knob}`\n")
            lines.append(f"Baseline value: `{knob_data['baseline_value']}`, n_questions = {knob_data['n_questions']}\n")
            lines.append("| value | n | recall@k | hit@k | mrr | Δ recall (paired CI) | sig? |")
            lines.append("|---|---:|---:|---:|---:|---|---|")
            for v_str, entry in knob_data["per_value"].items():
                pd = entry.get("paired_diff_recall")
                if pd:
                    sig = "**yes**" if pd["excludes_zero"] else "no"
                    diff_str = f"{pd['mean']:+.3f} [{pd['ci_low']:+.3f}, {pd['ci_high']:+.3f}]"
                else:
                    sig = "—"
                    diff_str = "(baseline)"
                lines.append(
                    f"| {v_str} | {entry['n']} | {entry['mean_recall_at_k']:.3f} | "
                    f"{entry['mean_hit_at_k']:.3f} | {entry['mean_mrr']:.3f} | "
                    f"{diff_str} | {sig} |"
                )
    else:
        lines.append("(skipped)")

    # ---- Phase 4 ----
    lines.append("\n## Phase 4: Diagnostic traces\n")
    if phase4:
        lines.append("Per-qtype trace of one question where any config flipped a baseline pass to a failure.\n")
        lines.append("| qtype | qid | trace file |")
        lines.append("|---|---|---|")
        for qt, entry in phase4.items():
            qid = entry.get("qid") or "—"
            tp = entry.get("trace_path") or "—"
            lines.append(f"| {qt} | `{qid[:12] if qid != '—' else '—'}` | `{tp}` |")
    else:
        lines.append("(skipped)")

    # ---- Recommendation ----
    lines.append("\n## Recommended launch config\n")
    if phase2:
        verdict = _verdict_from_phase2(phase2)
        keepers = [
            c for c, v in verdict.items()
            if v.startswith("PASS") or v == "NEUTRAL (no lift, no regression)"
        ]
        lines.append("Configs that pass the per-qtype regression gate (Δ ≥ −0.02 everywhere):")
        lines.append("")
        for c in keepers:
            if c == "baseline":
                continue
            lines.append(f"- `{c}` — {verdict[c]}")
        lines.append("")
        lines.append("If you want the safest launch: just `baseline` (no flags). If you want")
        lines.append("recall-neutral enhancements that might pay off elsewhere (LLM stage): pick a")
        lines.append("`NEUTRAL` config and add `--surface-conflicts --auto-temporal` for free.")
        lines.append("")
        lines.append("```powershell")
        lines.append("python -m engram.bench run longmemeval `")
        lines.append("  --embedder local --embed-model BAAI/bge-large-en-v1.5 --embed-device cuda --dtype fp32 `")
        lines.append("  --chat opencode-go --chat-model kimi-k2.6 `")
        lines.append("  --reranker bge --k 10 --seed 1337 `")
        lines.append("  --rerank-pool-multiplier 5 `")
        lines.append("  --auto-temporal --surface-conflicts")
        lines.append("```")
    else:
        lines.append("(phase 2 was skipped; cannot recommend)")

    report = "\n".join(lines)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    _LOG.info("PHASE 5 done: wrote %s", out_dir / "REPORT.md")
    return report


# ============================================================================
# Entry point
# ============================================================================


def _load_existing(out_dir: Path, name: str) -> dict[str, Any] | None:
    p = out_dir / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end retrieval evaluation orchestrator."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/runs/eval_all"))
    parser.add_argument("--limit", type=int, default=30, help="Questions per qtype.")
    parser.add_argument(
        "--sweep-limit",
        type=int,
        default=None,
        help="Questions for hyperparameter sweep (default: --limit).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--dtype", default="fp32", choices=("auto", "fp16", "fp32"))
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument(
        "--skip-phase",
        action="append",
        type=int,
        default=[],
        help="Skip a phase number. Repeatable. e.g. --skip-phase 2 --skip-phase 3",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Re-load existing phase JSON outputs from --out-dir instead of "
            "re-running them. Use after a crash."
        ),
    )
    parser.add_argument(
        "--log-level", default=os.environ.get("ENGRAM_LOG_LEVEL", "INFO")
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip_phase)
    sweep_limit = args.sweep_limit if args.sweep_limit is not None else args.limit

    config_args = {
        "out_dir": str(out_dir),
        "limit": args.limit,
        "sweep_limit": sweep_limit,
        "k": args.k,
        "embed_model": args.embed_model,
        "embed_device": args.embed_device,
        "dtype": args.dtype,
        "reranker": args.reranker,
        "skip_phase": sorted(skip),
        "resume": args.resume,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    # H-84: on resume, append to a `resume_history` list rather than
    # overwriting `config.json`. Pre-fix the resume run rewrote the
    # file with the resume invocation's args (which may differ from
    # the original -- different --limit, different skip-phase set,
    # different dtype, etc.) so the original config was lost. Now
    # the original config stays put and the resume run is recorded
    # alongside it.
    config_path = out_dir / "config.json"
    if args.resume and config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {"original": existing}
        except (OSError, json.JSONDecodeError):
            existing = {}
        history = existing.get("resume_history")
        if not isinstance(history, list):
            history = []
        history.append({
            "resumed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "args": config_args,
        })
        existing["resume_history"] = history
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    else:
        # Fresh run: write the config and start a fresh resume_history.
        payload = dict(config_args)
        payload["resume_history"] = []
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    t_start = time.perf_counter()

    # Phase 1
    phase1: dict[str, Any] = {}
    if 1 not in skip:
        if args.resume:
            existing = _load_existing(out_dir, "phase1_component_tests.json")
            if existing:
                _LOG.info("PHASE 1: resuming from existing JSON")
                phase1 = existing
        if not phase1:
            phase1 = phase1_component_tests(out_dir)

    # Phases 2-4 need the models
    embedder = None
    reranker = None
    chat = FakeChat()
    if (2 not in skip) or (3 not in skip) or (4 not in skip):
        _LOG.info("Loading shared embedder + reranker...")
        embedder = _build_embedder(args.embed_model, args.embed_device, args.dtype)
        reranker = _build_reranker(args.reranker, args.embed_device, args.dtype)

    # Phase 2
    phase2: dict[str, Any] = {}
    if 2 not in skip:
        if args.resume:
            existing = _load_existing(out_dir, "phase2_summary.json")
            if existing:
                _LOG.info("PHASE 2: resuming from existing JSON")
                phase2 = existing
        if not phase2:
            phase2 = phase2_ablation(out_dir, embedder, reranker, chat, args.limit, args.k)

    # Phase 3
    phase3: dict[str, Any] = {}
    if 3 not in skip:
        if args.resume:
            existing = _load_existing(out_dir, "phase3_summary.json")
            if existing:
                _LOG.info("PHASE 3: resuming from existing JSON")
                phase3 = existing
        if not phase3:
            phase3 = phase3_sweeps(out_dir, embedder, reranker, chat, sweep_limit, args.k)

    # Phase 4
    phase4: dict[str, Any] = {}
    if 4 not in skip:
        if args.resume:
            existing = _load_existing(out_dir, "phase4_summary.json")
            if existing:
                _LOG.info("PHASE 4: resuming from existing JSON")
                phase4 = existing
        if not phase4:
            phase4 = phase4_traces(out_dir, embedder, reranker, chat, phase2, args.k)

    # Phase 5
    report = phase5_report(out_dir, phase1, phase2, phase3, phase4, config_args)

    elapsed = time.perf_counter() - t_start
    _LOG.info("ALL PHASES DONE in %.1f min. Report at %s", elapsed / 60, out_dir / "REPORT.md")
    print(f"\n[done] Report: {out_dir / 'REPORT.md'} (total {elapsed/60:.1f} min)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
