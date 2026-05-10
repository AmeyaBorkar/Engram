"""CLI entry point for the benchmark harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engram.bench._provider import FakeProvider, Provider
from engram.bench._real_provider import build_provider
from engram.bench._runner import run as run_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engram-bench",
        description="Engram benchmark harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark suite.")
    run.add_argument("suite", help="Suite name (e.g. 'noop').")
    # Provider selection: `--provider fake` keeps backwards compat for
    # CI smoke runs. For real-provider runs, `--embedder` and `--chat`
    # name the two halves independently (Anthropic / Moonshot have no
    # embedding model, so the OpenAI embedder is the standard pair).
    run.add_argument(
        "--provider",
        default=None,
        choices=("fake",),
        help="Shortcut for --embedder fake --chat fake. Default: fake.",
    )
    run.add_argument(
        "--embedder",
        default=None,
        choices=("fake", "openai"),
        help="Embedding provider (default: fake).",
    )
    run.add_argument(
        "--chat",
        default=None,
        choices=("fake", "openai", "anthropic", "moonshot"),
        help="Chat provider (default: fake).",
    )
    run.add_argument(
        "--embed-model",
        default=None,
        help="Override the embedder model name (e.g. text-embedding-3-large).",
    )
    run.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Override the embedding dimensionality.",
    )
    run.add_argument(
        "--chat-model",
        default=None,
        help=("Override the chat model name (e.g. gpt-4o, claude-haiku-4-5-20251001, kimi-k2.6)."),
    )
    run.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("benchmarks/runs"),
        help="Directory in which to write the manifest (default: benchmarks/runs).",
    )

    return parser


def _resolve_provider(args: argparse.Namespace) -> Provider:
    """Build a Provider from CLI flags. `--provider fake` is a shortcut."""
    if args.provider == "fake":
        if args.embedder or args.chat:
            print(
                "warning: --provider fake overrides --embedder/--chat",
                file=sys.stderr,
            )
        return FakeProvider()

    embedder = args.embedder or "fake"
    chat = args.chat or "fake"
    if embedder == "fake" and chat == "fake":
        return FakeProvider()
    return build_provider(
        embedder_name=embedder,
        chat_name=chat,
        embed_model=args.embed_model,
        embed_dim=args.embed_dim,
        chat_model=args.chat_model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            provider = _resolve_provider(args)
            manifest_path = run_suite(args.suite, provider=provider, runs_dir=args.runs_dir)
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"manifest: {manifest_path}")
        return 0

    parser.print_help()
    return 1
