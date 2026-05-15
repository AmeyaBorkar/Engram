"""Suite loader and runner."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from engram.bench._manifest import Manifest, manifest_from_run
from engram.bench._provider import Provider
from engram.bench._suite import Suite


def load_suite(name: str) -> Suite:
    """Look up a suite by name.

    The lookup name is CLI-friendly (hyphens allowed); it is mapped to a
    Python module name by replacing `-` with `_`.

    Search order:
      1. `engram.bench.suites.<name>`  - built-ins (e.g. noop).
      2. `benchmarks.suites.<name>`     - project-local suites.

    The module must expose a `SUITE` attribute conforming to the `Suite`
    protocol.
    """
    module_name = name.replace("-", "_")
    last_error: Exception | None = None
    for module_path in (
        f"engram.bench.suites.{module_name}",
        f"benchmarks.suites.{module_name}",
    ):
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


def run(
    suite_name: str,
    *,
    provider: Provider,
    runs_dir: Path,
    suite_config: dict[str, Any] | None = None,
    provider_hash: str | None = None,
    engram_config: dict[str, Any] | None = None,
) -> Path:
    """Run `suite_name` against `provider`, writing a manifest to `runs_dir`.

    `suite_config`, when given, is forwarded to the suite's optional
    `configure(**cfg)` method before `setup`. Suites that don't expose
    `configure` silently ignore it; suites that do (e.g. longmemeval)
    pick up the Phase E knobs without the loader knowing about them.

    `provider_hash` defaults to `provider.manifest_hash()`; pass an
    override when the caller wants to record the *unwrapped* provider
    identity (e.g. when a transparent disk cache has been wrapped
    around the inner embedder/chat -- see `_cli._resolve_provider`).

    `engram_config`, when given, is recorded verbatim on the manifest
    so reproducers can read the active retrieval knobs without having
    to scrape `git log` or SCOREBOARD.md.

    Returns the manifest path.
    """
    suite = load_suite(suite_name)

    if suite_config:
        configure = getattr(suite, "configure", None)
        if configure is not None:
            configure(**suite_config)

    suite.setup(provider)
    try:
        result = suite.run()
    finally:
        suite.teardown()

    manifest: Manifest = manifest_from_run(
        suite_name=result.name,
        provider_name=provider.name,
        provider_hash=provider_hash if provider_hash is not None else provider.manifest_hash(),
        dataset_version=suite.dataset_version,
        dataset_checksum=suite.dataset_checksum,
        aggregate_metrics=result.aggregate_metrics,
        confidence_intervals=result.confidence_intervals,
        per_question=result.per_question,
        latency_ms=result.latency_ms,
        engram_config=engram_config,
    )
    return manifest.write(runs_dir)
