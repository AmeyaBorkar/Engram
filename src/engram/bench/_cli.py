"""CLI entry point for the benchmark harness."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from engram.bench._provider import FakeProvider, Provider
from engram.bench._real_provider import build_chat, build_provider
from engram.bench._runner import run as run_suite


def _configure_logging(level_name: str) -> None:
    """Attach a stderr handler so suite progress shows up by default.

    The root logger is unconfigured in library code so import-time logs
    don't surprise embedders. The CLI is the right place to wire it in.
    `ENGRAM_LOG_LEVEL` overrides the default.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
    else:
        logging.getLogger().setLevel(level)


def _load_env_file(path: Path) -> bool:
    """Best-effort `.env` loading. Existing env vars take precedence.

    Returns True if `.env` was loaded, False otherwise. We use
    `python-dotenv` when available (the `[bench]` extra installs it);
    fall through to a tiny built-in parser otherwise so the CLI still
    works without the dep.
    """
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return True
    load_dotenv(path, override=False)
    return True


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
        choices=("fake", "openai", "local", "openrouter"),
        help=(
            "Embedding provider (default: fake). `local` runs "
            "sentence-transformers on your GPU (CPU fallback) and "
            "needs no API key; default model is BAAI/bge-large-en-v1.5. "
            "`openrouter` defaults to qwen/qwen3-embedding-8b (MTEB ~70.6, "
            "$0.01/M tokens) and reuses OPENROUTER_API_KEY for chat + embed."
        ),
    )
    run.add_argument(
        "--chat",
        default=None,
        choices=(
            "fake",
            "openai",
            "anthropic",
            "moonshot",
            "opencode-zen",
            "opencode-go",
            "openrouter",
        ),
        help=(
            "Chat provider (default: fake). `openrouter` exposes Claude / "
            "GPT / Kimi / DeepSeek / Gemini behind one API key; default "
            "model is anthropic/claude-haiku-4-5."
        ),
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
        "--embed-device",
        default=None,
        help=(
            "Force the device for `--embedder local`: cpu, cuda, mps, or "
            "cuda:N. Defaults to auto-detect (CUDA > MPS > CPU). Useful when "
            "CUDA is broken (driver/arch mismatch) and you want CPU explicitly."
        ),
    )
    run.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "fp16", "fp32"),
        help=(
            "Numeric precision for local embedder + BGE reranker. `auto` "
            "(default) uses fp16 on CUDA, fp32 elsewhere. fp16 halves "
            "VRAM (essential for 12 GB cards running stella + reranker "
            "concurrently). `fp32` forces full precision if you need to "
            "reproduce a baseline bit-exactly."
        ),
    )
    run.add_argument(
        "--chat-model",
        default=None,
        help=(
            "Override the chat model name. Examples: gpt-4o, "
            "claude-haiku-4-5-20251001 (Anthropic direct), kimi-k2.6 (Moonshot), "
            "claude-haiku-4-5 / gpt-5.5-mini / kimi-k2.6 (OpenCode Zen)."
        ),
    )
    run.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("benchmarks/runs"),
        help="Directory in which to write the manifest (default: benchmarks/runs).",
    )
    run.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help=(
            "Path to a .env file to load before resolving the provider "
            "(default: .env in cwd). Existing environment variables "
            "always take precedence over .env values."
        ),
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the number of items the suite processes. Useful for "
            "smoke runs. Overrides any suite-specific cap env var "
            "(LONGMEMEVAL_MAX_QUESTIONS, etc)."
        ),
    )
    run.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override retrieval top-k for suites that respect it (e.g. longmemeval).",
    )
    run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (suites that respect it).",
    )

    # --- Phase E retrieve flags (per-question on suites that wire them).
    run.add_argument("--hyde", action="store_true", help="Enable HyDE query transform.")
    run.add_argument(
        "--multi-query-n",
        type=int,
        default=1,
        help="Expand the query into N variants and fuse via RRF (1=off).",
    )
    run.add_argument(
        "--decompose",
        action="store_true",
        help="Decompose multi-hop queries into sub-questions and fuse via RRF.",
    )
    run.add_argument(
        "--temporal",
        action="store_true",
        help="Auto-anchor as_of= from the question via the chat provider.",
    )
    run.add_argument(
        "--surface-conflicts",
        action="store_true",
        help="Append the OTHER side of any open conflicts to the retrieved set.",
    )
    run.add_argument(
        "--reranker",
        default=None,
        choices=("bge", "none"),
        help="Cross-encoder reranker. 'bge' uses BAAI/bge-reranker-v2-m3.",
    )
    run.add_argument(
        "--reranker-model",
        default=None,
        help="Override the reranker model id (e.g. BAAI/bge-reranker-v2-m3).",
    )

    # --- Phase E agent flags (engage EngramAgent when any is set).
    run.add_argument(
        "--cot",
        action="store_true",
        help="Chain-of-thought system instruction for the answer step.",
    )
    run.add_argument(
        "--self-consistency-n",
        type=int,
        default=1,
        help="Draw N samples and vote (1=off). Requires a stochastic chat provider.",
    )
    run.add_argument(
        "--verify",
        action="store_true",
        help="Run a verification pass on the answer; retry on unsupported.",
    )
    run.add_argument(
        "--verify-max-retries",
        type=int,
        default=1,
        help="Bound on the verify-driven retry loop (default 1).",
    )

    # --- Secondary chat slots (consolidate, judge).
    run.add_argument(
        "--consolidate",
        action="store_true",
        help=(
            "Run memory.consolidate() between haystack ingest and retrieve. "
            "Clusters the events and abstracts each cluster into a "
            "Level.SUMMARY MemoryItem. This is Engram's core novelty "
            "(hierarchical memory) -- without this flag the bench treats "
            "Engram as a flat retriever."
        ),
    )
    run.add_argument(
        "--consolidate-chat",
        default=None,
        choices=(
            "fake",
            "openai",
            "anthropic",
            "moonshot",
            "opencode-zen",
            "opencode-go",
            "openrouter",
        ),
        help=(
            "Separate chat provider for the irreversible consolidation step "
            "(abstraction + reconciliation). Falls back to --chat when omitted."
        ),
    )
    run.add_argument(
        "--consolidate-chat-model",
        default=None,
        help="Model name for --consolidate-chat.",
    )
    run.add_argument(
        "--judge-chat",
        default=None,
        choices=(
            "fake",
            "openai",
            "anthropic",
            "moonshot",
            "opencode-zen",
            "opencode-go",
            "openrouter",
        ),
        help=(
            "Separate chat provider for the LongMemEval judge. Falls back to "
            "--chat when omitted. Use an independent model to avoid "
            "self-preference bias."
        ),
    )
    run.add_argument(
        "--judge-chat-model",
        default=None,
        help="Model name for --judge-chat.",
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
    # CLI uses "fp16"/"fp32" as shorthand; the embedder accepts the
    # full names so map here.
    dtype_map = {"auto": "auto", "fp16": "float16", "fp32": "float32"}
    return build_provider(
        embedder_name=embedder,
        chat_name=chat,
        embed_model=args.embed_model,
        embed_dim=args.embed_dim,
        embed_device=args.embed_device,
        embed_dtype=dtype_map[args.dtype],
        chat_model=args.chat_model,
    )


def _resolve_suite_config(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a suite-config dict.

    The runner forwards this to the suite's `configure(**cfg)` method
    when one exists. Suites that don't implement `configure` ignore it;
    suites that do (longmemeval at minimum) pick up the Phase E knobs.

    Secondary chat providers (`consolidate_chat`, `judge_chat`) are
    pre-built here so the suite never has to know about the provider
    catalog -- it just receives ChatProvider instances.
    """
    cfg: dict[str, Any] = {}
    if args.k is not None:
        cfg["k"] = args.k
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.hyde:
        cfg["hyde"] = True
    if args.multi_query_n and args.multi_query_n > 1:
        cfg["multi_query_n"] = args.multi_query_n
    if args.decompose:
        cfg["decompose"] = True
    if args.temporal:
        cfg["temporal"] = True
    if args.surface_conflicts:
        cfg["surface_conflicts"] = True
    if args.reranker and args.reranker != "none":
        from engram.retrieve._bge_reranker import BGEReranker

        dtype_map = {"auto": "auto", "fp16": "float16", "fp32": "float32"}
        if args.reranker == "bge":
            cfg["reranker"] = BGEReranker(
                model=args.reranker_model or "BAAI/bge-reranker-v2-m3",
                device=args.embed_device,
                dtype=dtype_map[args.dtype],  # type: ignore[arg-type]
            )
        else:  # pragma: no cover - argparse choices already filter
            raise ValueError(f"unknown reranker: {args.reranker!r}")
    if args.cot:
        cfg["cot"] = True
    if args.self_consistency_n and args.self_consistency_n > 1:
        cfg["self_consistency_n"] = args.self_consistency_n
    if args.verify:
        cfg["verify"] = True
        cfg["verify_max_retries"] = args.verify_max_retries
    if args.consolidate:
        cfg["consolidate"] = True
    if args.consolidate_chat:
        cfg["consolidate_chat"] = build_chat(
            args.consolidate_chat, args.consolidate_chat_model
        )
    if args.judge_chat:
        cfg["judge_chat"] = build_chat(args.judge_chat, args.judge_chat_model)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if _load_env_file(args.env_file):
            print(f"loaded {args.env_file}", file=sys.stderr)
        # `.env` may carry `ENGRAM_LOG_LEVEL=DEBUG`; load it before
        # configuring logging so user overrides take effect.
        _configure_logging(os.environ.get("ENGRAM_LOG_LEVEL", "INFO"))
        # `--limit` overrides the suite-specific cap env vars. Set them
        # BEFORE importing the suite -- the suite reads them at module
        # import time in its `SUITE = ...()` line.
        if args.limit is not None:
            for var in ("LONGMEMEVAL_MAX_QUESTIONS", "LOCOMO_MAX_QUESTIONS"):
                os.environ[var] = str(args.limit)
            print(f"--limit {args.limit} applied", file=sys.stderr)
        try:
            provider = _resolve_provider(args)
            suite_config = _resolve_suite_config(args)
            manifest_path = run_suite(
                args.suite,
                provider=provider,
                runs_dir=args.runs_dir,
                suite_config=suite_config,
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"manifest: {manifest_path}")
        return 0

    parser.print_help()
    return 1
