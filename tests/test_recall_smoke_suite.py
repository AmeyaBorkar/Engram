"""End-to-end test for the recall-smoke benchmark suite."""

from __future__ import annotations

import json
from pathlib import Path

from engram.bench import FakeProvider, load_suite, run


def test_recall_smoke_suite_loads() -> None:
    suite = load_suite("recall-smoke")
    assert suite.name == "recall-smoke"
    assert suite.dataset_checksum


def test_recall_smoke_suite_runs_engram_at_recall_1(tmp_path: Path) -> None:
    """Engram with deterministic FakeEmbedder must hit recall@10 = 1.0 on
    exact-text queries. This is the floor every retriever should clear;
    the suite exists to verify the wiring, not embedding quality."""
    manifest_path = run("recall-smoke", provider=FakeProvider(), runs_dir=tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "recall-smoke"
    metrics = payload["aggregate_metrics"]
    # engram is always in the metrics; chroma/chroma+bm25 are present iff
    # chromadb is installed locally.
    assert "engram_recall@10" in metrics
    assert metrics["engram_recall@10"] == 1.0


def test_recall_smoke_per_question_payload_well_formed(tmp_path: Path) -> None:
    manifest_path = run("recall-smoke", provider=FakeProvider(), runs_dir=tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_q = payload["per_question"]
    assert per_q
    sample = per_q[0]
    for required in ("system", "query_idx", "query", "recall@k", "k"):
        assert required in sample
