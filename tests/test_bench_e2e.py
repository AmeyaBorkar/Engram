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


def test_fake_provider_exposes_embedder_and_chat() -> None:
    from engram.providers import ChatProvider, EmbeddingProvider

    p = FakeProvider()
    assert isinstance(p.embedder, EmbeddingProvider)
    assert isinstance(p.chat, ChatProvider)


def test_fake_provider_manifest_hash_changes_with_dim() -> None:
    a = FakeProvider(dim=64).manifest_hash()
    b = FakeProvider(dim=128).manifest_hash()
    assert a != b
    assert a.startswith("fake/")


def test_load_unknown_suite_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        load_suite("definitely-not-a-real-suite")


def test_run_noop_writes_manifest(tmp_path: Path) -> None:
    manifest_path = run("noop", provider=FakeProvider(), runs_dir=tmp_path)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "noop"
    assert payload["provider"] == "fake"
    assert payload["aggregate_metrics"]["noop"] == pytest.approx(1.0)
    ci = payload["confidence_intervals"]["noop"]
    assert ci[0] == pytest.approx(1.0)
    assert ci[1] == pytest.approx(1.0)


def test_cli_run_noop(tmp_path: Path) -> None:
    rc = main(["run", "noop", "--provider", "fake", "--runs-dir", str(tmp_path)])
    assert rc == 0
    manifests = list(tmp_path.glob("*-noop.json"))
    assert len(manifests) == 1


def test_cli_unknown_suite_returns_2(tmp_path: Path) -> None:
    rc = main(["run", "no-such-suite", "--provider", "fake", "--runs-dir", str(tmp_path)])
    assert rc == 2
