"""Reproduce every Engram benchmark suite end-to-end.

Runs each suite against the FakeProvider so it's fully reproducible
without an API key. Real-LLM runs use the same harness via the CLI
(`python -m engram.bench run <suite> --provider <provider>`); this
script is the CI-friendly fake-provider sweep.

Outputs one manifest per suite under `benchmarks/runs/ci/<timestamp>/`.

Usage:

    python scripts/reproduce_benchmarks.py            # all suites
    python scripts/reproduce_benchmarks.py --only recall-smoke procedural-transfer

Stage 10 prep: callers (CI on tagged release) drive this script to
re-emit manifests after every release; the receipts in
`benchmarks/runs/release/` cite the matching commit.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the repo root importable so `benchmarks.suites.*` resolves when
# this script is run directly (`python scripts/reproduce_benchmarks.py`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engram.bench import FakeProvider, manifest_from_run  # noqa: E402

# Suites shipped today. Each entry: (module_path, name, attr).
# The `attr` is the module-level singleton (typically `SUITE`).
ALL_SUITES: tuple[tuple[str, str, str], ...] = (
    ("benchmarks.suites.recall_smoke", "recall-smoke", "SUITE"),
    ("benchmarks.suites.latency_at_scale", "latency-at-scale", "SUITE"),
    ("benchmarks.suites.procedural_transfer", "procedural-transfer", "SUITE"),
    (
        "benchmarks.suites.contradiction_temporal",
        "contradiction-temporal",
        "SUITE",
    ),
    # locomo + longmemeval are heavy + require real-LLM runs; skip in
    # fake-provider mode. CI users invoke them separately via the CLI.
)


def _load_suite(module_path: str, attr: str) -> Any:
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _run_one(
    suite: Any,
    provider: FakeProvider,
    out_dir: Path,
) -> Path:
    suite.setup(provider)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        result = suite.run()
    finally:
        suite.teardown()
    duration_s = time.perf_counter() - t0

    manifest = manifest_from_run(
        suite_name=suite.name,
        provider_name=provider.name,
        provider_hash=provider.manifest_hash(),
        dataset_version=suite.dataset_version,
        dataset_checksum=suite.dataset_checksum,
        aggregate_metrics=result.aggregate_metrics,
        confidence_intervals=result.confidence_intervals,
        per_question=result.per_question,
        latency_ms=result.latency_ms,
        engram_config={
            "started_at": started.isoformat(),
            "duration_s": duration_s,
        },
    )
    return manifest.write(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        help="Run only the named suites (default: all).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("benchmarks/runs/ci"),
        help="Output directory. A timestamped subdir is created beneath it.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="FakeEmbedder dimension.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = args.runs_dir / timestamp

    provider = FakeProvider(dim=args.dim)

    selected = ALL_SUITES
    if args.only:
        names = set(args.only)
        selected = tuple(s for s in ALL_SUITES if s[1] in names)
        missing = names - {s[1] for s in selected}
        if missing:
            print(f"unknown suite(s): {sorted(missing)}", file=sys.stderr)
            return 2

    failures: list[str] = []
    for module_path, name, attr in selected:
        print(f"==> {name}")
        try:
            suite = _load_suite(module_path, attr)
            path = _run_one(suite, provider, out_dir)
            print(f"    manifest: {path}")
        except Exception as exc:
            failures.append(name)
            print(f"    FAILED: {exc!r}", file=sys.stderr)

    print(f"\nManifests in: {out_dir}")
    if failures:
        print(f"\n{len(failures)} suite(s) failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
