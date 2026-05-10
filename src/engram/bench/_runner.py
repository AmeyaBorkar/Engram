"""Suite loader and runner."""

from __future__ import annotations

import importlib
from pathlib import Path

from engram.bench._manifest import Manifest, manifest_from_run
from engram.bench._provider import Provider
from engram.bench._suite import Suite


def load_suite(name: str) -> Suite:
    """Look up a suite by name.

    Search order:
      1. `engram.bench.suites.<name>`  - built-ins (e.g. noop).
      2. `benchmarks.suites.<name>`     - project-local suites.

    The module must expose a `SUITE` attribute conforming to the `Suite`
    protocol.
    """
    last_error: Exception | None = None
    for module_path in (f"engram.bench.suites.{name}", f"benchmarks.suites.{name}"):
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            last_error = exc
            continue
        suite_obj = getattr(module, "SUITE", None)
        if suite_obj is None:
            raise ValueError(f"module {module_path!r} has no `SUITE` attribute")
        if not isinstance(suite_obj, Suite):
            raise TypeError(f"module {module_path!r}.SUITE does not satisfy the Suite protocol")
        return suite_obj

    raise ValueError(
        f"suite {name!r} not found in engram.bench.suites or benchmarks.suites"
    ) from last_error


def run(suite_name: str, *, provider: Provider, runs_dir: Path) -> Path:
    """Run `suite_name` against `provider`, writing a manifest to `runs_dir`.

    Returns the manifest path.
    """
    suite = load_suite(suite_name)

    suite.setup(provider)
    try:
        result = suite.run()
    finally:
        suite.teardown()

    manifest: Manifest = manifest_from_run(
        suite_name=result.name,
        provider_name=provider.name,
        provider_hash=provider.manifest_hash(),
        dataset_version=suite.dataset_version,
        dataset_checksum=suite.dataset_checksum,
        aggregate_metrics=result.aggregate_metrics,
        confidence_intervals=result.confidence_intervals,
        per_question=result.per_question,
        latency_ms=result.latency_ms,
    )
    return manifest.write(runs_dir)
