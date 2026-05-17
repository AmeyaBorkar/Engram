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


def test_cli_refuses_fake_embedder_with_real_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-81 regression: pairing a real chat (Anthropic, Moonshot,
    OpenCode) with the fake embedder produces nonsense retrievals --
    refuse the run and tell the operator to add `--embedder openai`.

    Mock the API key so the chat-builder gets that far; the embedder
    check has to fire BEFORE the chat builder so missing-key errors
    don't mask the bad pairing.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    rc = main(
        [
            "run",
            "noop",
            "--embedder",
            "fake",
            "--chat",
            "anthropic",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    # Returns 2 (the CLI's "config error" exit) -- not 0, not raising.
    assert rc == 2


def test_cli_resolves_provider_returns_pre_wrap_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-73 regression: the provider_hash captured for the manifest
    must come from the UNWRAPPED provider so cached + uncached runs
    of the same config produce identical hashes.
    """
    from argparse import Namespace
    from engram.bench._cli import _resolve_provider

    # Build a minimal Namespace that satisfies _resolve_provider.
    args = Namespace(
        provider="fake",
        embedder=None,
        chat=None,
        embed_model=None,
        embed_dim=None,
        embed_device=None,
        chat_model=None,
        dtype="auto",
        disk_cache=None,
    )
    provider1, hash1 = _resolve_provider(args)
    provider2, hash2 = _resolve_provider(args)
    # Same config -> same hash both times.
    assert hash1 == hash2
    # The hash always corresponds to the pre-wrap provider's hash.
    assert hash1 == provider1.manifest_hash()
    assert hash2 == provider2.manifest_hash()
