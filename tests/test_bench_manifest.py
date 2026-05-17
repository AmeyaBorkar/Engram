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


def test_manifest_same_second_collision_does_not_overwrite(tmp_path: Path) -> None:
    """H-75 regression: two writes that hash to the same timestamp
    must NOT clobber each other.

    The pre-fix code computed a second-precision filename and used
    ``path.write_text(...)`` directly, so two suite runs landing in
    the same second silently overwrote each other's evidence. The
    fix appends a 4-hex salt suffix on collision and writes via
    ``tempfile.mkstemp + os.replace`` so the swap is atomic.
    """
    base_kwargs: dict[str, object] = {
        "suite_name": "noop",
        "provider_name": "fake",
        "provider_hash": "x",
        "dataset_version": "v1",
        "dataset_checksum": "d" * 64,
        "aggregate_metrics": {},
        "confidence_intervals": {},
        "per_question": [],
        "latency_ms": {},
    }
    # Pin the same timestamp on both manifests to force a collision
    # without waiting on a sub-second clock.
    m1 = manifest_from_run(**base_kwargs)
    m2 = manifest_from_run(**base_kwargs)
    m2.timestamp = m1.timestamp
    m2.git_commit = m1.git_commit
    m2.git_dirty = m1.git_dirty
    # Distinguish the payloads so the test can verify both survived.
    m1.aggregate_metrics = {"score": 0.1}
    m2.aggregate_metrics = {"score": 0.2}

    p1 = m1.write(tmp_path)
    p2 = m2.write(tmp_path)
    assert p1 != p2, "collision was not detected; one manifest overwrote the other"
    assert p1.exists()
    assert p2.exists()
    loaded1 = json.loads(p1.read_text(encoding="utf-8"))
    loaded2 = json.loads(p2.read_text(encoding="utf-8"))
    assert loaded1["aggregate_metrics"]["score"] == 0.1
    assert loaded2["aggregate_metrics"]["score"] == 0.2


def test_manifest_write_is_atomic_no_partial_file(tmp_path: Path) -> None:
    """H-75 regression: a successful write leaves no .tmp file behind.

    The atomic write strategy is ``tempfile.mkstemp + os.replace``;
    on success the temp file is renamed onto the destination, so
    listing ``tmp_path`` afterwards must show only the final
    manifest filename.
    """
    m = manifest_from_run(
        suite_name="noop",
        provider_name="fake",
        provider_hash="x",
        dataset_version="v1",
        dataset_checksum="d" * 64,
        aggregate_metrics={"score": 1.0},
        confidence_intervals={"score": (0.9, 1.0)},
        per_question=[],
        latency_ms={},
    )
    path = m.write(tmp_path)
    leftovers = [
        p for p in tmp_path.iterdir() if p.suffix == ".tmp" or ".tmp" in p.suffixes
    ]
    assert leftovers == [], f"atomic write left .tmp files behind: {leftovers}"
    assert path.exists()
