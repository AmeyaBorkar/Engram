"""Suite loader and runner."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from engram.bench._manifest import Manifest, manifest_from_run
from engram.bench._provider import Provider
from engram.bench._suite import Suite


def _serialize_for_manifest(value: Any) -> Any:
    """Convert arbitrary config values into JSON-safe descriptors.

    Audit H-76: `engram_config` is the reproducibility ledger for a
    bench run; ~30 knobs flow through `suite_config` (BM25 / MMR /
    recency, drill / pool, reranker, consolidate / judge providers,
    Phase E agent flags). Provider instances and `Reranker` objects
    aren't JSON-serializable, so we lower them to a stable descriptor
    (`name` / `model` / `manifest_hash()` when available) instead of
    dropping them. The goal is a JSON blob a reviewer can read to
    reproduce a run, not the live objects themselves.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_for_manifest(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_for_manifest(v) for k, v in value.items()}
    desc: dict[str, Any] = {"type": type(value).__name__}
    for attr in ("name", "model", "dim"):
        val = getattr(value, attr, None)
        if isinstance(val, (str, int, float, bool)):
            desc[attr] = val
    mh = getattr(value, "manifest_hash", None)
    if callable(mh):
        try:
            hashed = mh()
        except Exception:  # pragma: no cover - manifest hash is best-effort
            hashed = None
        if isinstance(hashed, str):
            desc["manifest_hash"] = hashed
    return desc


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
) -> Path:
    """Run `suite_name` against `provider`, writing a manifest to `runs_dir`.

    `suite_config`, when given, is forwarded to the suite's optional
    `configure(**cfg)` method before `setup`. Suites that don't expose
    `configure` silently ignore it; suites that do (e.g. longmemeval)
    pick up the Phase E knobs without the loader knowing about them.

    Returns the manifest path.
    """
    suite = load_suite(suite_name)

    if suite_config:
        configure = getattr(suite, "configure", None)
        if configure is not None:
            # Filter suite_config to only the kwargs configure() accepts.
            # Lets the caller stash extra reproducibility-ledger entries
            # (e.g. `chat`, `embedder` descriptors that the suite doesn't
            # need at runtime because the provider already carries them)
            # in suite_config without forcing every suite's configure()
            # signature to grow a `**_extra` catch-all. The full
            # suite_config is still serialized into engram_config below.
            sig_params = inspect.signature(configure).parameters
            accepts_var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
            )
            if accepts_var_kw:
                accepted = suite_config
            else:
                accepted = {
                    k: v for k, v in suite_config.items() if k in sig_params
                }
            configure(**accepted)

    suite.setup(provider)
    try:
        result = suite.run()
    finally:
        suite.teardown()

    engram_config = (
        {str(k): _serialize_for_manifest(v) for k, v in suite_config.items()}
        if suite_config
        else {}
    )
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
        engram_config=engram_config,
    )
    return manifest.write(runs_dir)
