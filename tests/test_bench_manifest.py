"""Tests for the manifest writer."""

from __future__ import annotations

import json
from pathlib import Path

from engram.bench import Manifest, gather_environment, manifest_from_run


def test_gather_environment_populates_required_fields() -> None:
    env = gather_environment()
    assert "python_version" in env
    assert "os" in env
    assert "cpu" in env
    assert "ram_gb" in env
    assert "git_commit" in env
    assert "git_dirty" in env


def test_manifest_writes_well_formed_json(tmp_path: Path) -> None:
    m: Manifest = manifest_from_run(
        suite_name="noop",
        provider_name="fake",
        provider_hash="abc",
        dataset_version="v1",
        dataset_checksum="d" * 64,
        aggregate_metrics={"score": 0.9},
        confidence_intervals={"score": (0.85, 0.95)},
        per_question=[{"q": 1, "ok": True}],
        latency_ms={"observe": [1.0, 2.0]},
        engram_config={"k": 10},
    )
    path = m.write(tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["suite"] == "noop"
    assert loaded["provider"] == "fake"
    assert loaded["aggregate_metrics"]["score"] == 0.9
    assert loaded["confidence_intervals"]["score"] == [0.85, 0.95]
    assert loaded["latency_ms"]["observe"] == [1.0, 2.0]
    assert loaded["engram_config"]["k"] == 10


def test_manifest_filename_has_suite_and_sha(tmp_path: Path) -> None:
    m = manifest_from_run(
        suite_name="noop",
        provider_name="fake",
        provider_hash="x",
        dataset_version="v1",
        dataset_checksum="d" * 64,
        aggregate_metrics={},
        confidence_intervals={},
        per_question=[],
        latency_ms={},
    )
    path = m.write(tmp_path)
    name = path.name
    assert name.endswith("-noop.json")
