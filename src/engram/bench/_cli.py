"""CLI entry point for the benchmark harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engram.bench._provider import FakeProvider, Provider
from engram.bench._runner import run as run_suite

_PROVIDERS: dict[str, type[Provider]] = {
    "fake": FakeProvider,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engram-bench",
        description="Engram benchmark harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark suite.")
    run.add_argument("suite", help="Suite name (e.g. 'noop').")
    run.add_argument(
        "--provider",
        default="fake",
        choices=sorted(_PROVIDERS.keys()),
        help="Provider to use (default: fake).",
    )
    run.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("benchmarks/runs"),
        help="Directory in which to write the manifest (default: benchmarks/runs).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        provider_cls = _PROVIDERS[args.provider]
        provider: Provider = provider_cls()
        try:
            manifest_path = run_suite(args.suite, provider=provider, runs_dir=args.runs_dir)
        except (ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"manifest: {manifest_path}")
        return 0

    parser.print_help()
    return 1
