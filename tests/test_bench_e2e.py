"""End-to-end harness test against the noop suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.bench import FakeProvider, load_suite, main, run


def test_load_noop_suite() -> None:
    suite = load_suite("noop")
    assert suite.name == "noop"
    assert suite.dataset_version


def test_load_unknown_suite_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        load_suite("definitely-not-a-real-suite")


def test_run_noop_writes_manifest(tmp_path: Path) -> None:
    manifest_path = run("noop", provider=FakeProvider(), runs_dir=tmp_path)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "noop"
    assert payload["provider"] == "fake"
    assert payload["aggregate_metrics"] == {"noop": 1.0}
    assert payload["confidence_intervals"] == {"noop": [1.0, 1.0]}


def test_cli_run_noop(tmp_path: Path) -> None:
    rc = main(["run", "noop", "--provider", "fake", "--runs-dir", str(tmp_path)])
    assert rc == 0
    manifests = list(tmp_path.glob("*-noop.json"))
    assert len(manifests) == 1


def test_cli_unknown_suite_returns_2(tmp_path: Path) -> None:
    rc = main(["run", "no-such-suite", "--provider", "fake", "--runs-dir", str(tmp_path)])
    assert rc == 2
