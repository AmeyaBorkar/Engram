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
        "--chat-max-tokens",
        type=int,
        default=None,
        help=(
            "Override the primary chat provider's max_tokens cap. Required "
            "when routing a thinking-mode model (e.g. moonshotai/kimi-k2.6) "
            "through a generic OpenAI-compatible endpoint like OpenRouter "
            "that otherwise inherits OpenAIChat's 1024-token safety guard "
            "and truncates mid-reasoning. JOURNEY §24 documents the cliff. "
            "Suggested values: 8192 for Kimi K2.6 thinking, 4096 for most "
            "non-thinking models, leave unset for short-answer suites."
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
        "--sample",
        type=int,
        default=None,
        help=(
            "Stratified sample N questions across the dataset's question "
            "types (deterministic with --seed). Unlike --limit, which "
            "takes the leading N rows in dataset order, --sample preserves "
            "the qtype distribution -- use this for fast discovery runs "
            "(e.g. --sample 100) before committing to a 500-question full "
            "eval. Mutually exclusive with --limit."
        ),
    )
    run.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Process N questions concurrently via a thread pool. Default 1 "
            "(serial; bit-identical to prior code path). 20-30 typically "
            "cuts wall-time 10-30x on LLM-bound runs without exceeding the "
            "provider's concurrent-request limit. Embedder calls inside "
            "ThreadPoolExecutor are still serialized by the underlying "
            "model lock (sentence-transformers, BGE reranker) -- the win "
            "is on chat / judge HTTP calls and downstream ops."
        ),
    )
    run.add_argument(
        "--gpu-concurrency",
        type=int,
        default=1,
        help=(
            "Cap concurrent CUDA forward passes (embedder + reranker share "
            "this semaphore). Default 1 -- safe on 12 GB cards at fp32. "
            "Raise to 2-4 with headroom (24 GB cards, or fp16). Decoupled "
            "from --parallel: chat / judge HTTP fan-out at high parallel "
            "while GPU work serializes here, preventing OOM. Set via the "
            "ENGRAM_GPU_CONCURRENCY env var if you bypass this flag."
        ),
    )
    run.add_argument(
        "--prompt-version",
        default="v1",
        choices=("v1", "v2", "v2a", "v2b", "v2c"),
        help=(
            "Answer-prompt template version (longmemeval only). "
            "v1 = original. "
            "v2 = bundled abstain + per-qtype + scratchpad (n=500 regression). "
            "v2a = abstain anchoring only, softened (no qtype hints, no CoT). "
            "v2b = per-qtype format hints only (no abstain, no CoT). "
            "v2c = v2a + v2b combined. "
            "Default v1. See JOURNEY section 23 and the v2 follow-up for the "
            "design rationale per variant."
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
    run.add_argument(
        "--bm25-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the BM25 lexical ranking in the dense+BM25 RRF fusion. "
            "0 (default) disables BM25 entirely; 1.0 = equal weight; recovers "
            "literal-token recall (years, codes, names)."
        ),
    )
    run.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.0,
        help=(
            "Maximal Marginal Relevance diversity weight, applied AFTER the "
            "cross-encoder rerank. 0 (default) is off; 0.7 balances relevance "
            "with diversity (suppresses near-duplicates in the top-k)."
        ),
    )
    run.add_argument(
        "--recency-lambda",
        type=float,
        default=0.0,
        help=(
            "Time-decay rerank boost. 0 (default) is off. Values around 0.05 - "
            "0.15 give recent events a small multiplicative bump on the "
            "rerank score -- useful for knowledge-update + temporal-reasoning."
        ),
    )
    run.add_argument(
        "--drill-k",
        type=int,
        default=None,
        help=(
            "RetrieveParams.drill_k -- how many supporting events to consider "
            "per low-confidence abstraction. Default 3."
        ),
    )
    run.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "RetrieveParams.confidence_threshold -- abstractions above this "
            "score are surfaced as-is; below it the engine drills. Default 0.7."
        ),
    )
    run.add_argument(
        "--rerank-pool-multiplier",
        type=int,
        default=None,
        help=(
            "RetrieveParams.candidate_multiplier -- size of the pre-rerank pool "
            "is k * this. Default 3; raise to 5 - 10 to give the cross-encoder "
            "more candidates to choose from (at the cost of rerank wall time)."
        ),
    )
    run.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="BM25 k1 hyperparameter. Default 1.5 (Lucene default).",
    )
    run.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 b hyperparameter. Default 0.75 (Lucene default).",
    )
    run.add_argument(
        "--recency-decay-days",
        type=float,
        default=90.0,
        help="Half-life-shape parameter for --recency-lambda. Default 90 days.",
    )
    run.add_argument(
        "--mmr-pool-size",
        type=int,
        default=0,
        help=("Override the MMR candidate pool size. 0 (default) uses k * rerank-pool-multiplier."),
    )
    run.add_argument(
        "--recent-window-k",
        type=int,
        default=0,
        help=(
            "Recent-window hybrid: include the top-N most-recent events "
            "in the RRF fusion alongside dense + BM25. 0 (default) is off."
        ),
    )
    run.add_argument(
        "--enable-tools",
        action="store_true",
        help=(
            "Computational tool layer (longmemeval): the answer prompt "
            "advertises a catalog of <tool>OP(args)</tool> tags (SUM, "
            "COUNT, AVG, MIN, MAX, DAYS_BETWEEN, WEEKS_BETWEEN, "
            "MONTHS_BETWEEN, YEARS_BETWEEN). The LLM emits tags; we "
            "substitute them with deterministic Python computations. "
            "Targets the 21 temporal-reasoning + 24 multi-session "
            "hard-wall failures where Kimi K2.6 can't reliably do "
            "arithmetic or date math. Monotonic: tool failures leave "
            "the tag in place. Stacks with --answer-form structured "
            "(tools execute on the extracted final_answer)."
        ),
    )
    run.add_argument(
        "--answer-form",
        default="freeform",
        choices=("freeform", "structured"),
        help=(
            "How the answer is extracted from the chat response. "
            "'freeform' (default) returns the raw response as-is. "
            "'structured' appends a JSON-output instruction to the "
            "answer prompt and extracts just the `final_answer` field; "
            "blocks CoT-leakage at the output layer. Monotonic: falls "
            "back to raw response if JSON parse fails."
        ),
    )
    run.add_argument(
        "--context-format",
        default="flat",
        choices=("flat", "grouped"),
        help=(
            "How retrieved memory is rendered into the answer prompt. "
            "'flat' (default) is the original bulleted-rank list with "
            "score/level annotations. 'grouped' groups retrieved events "
            "by session_id with explicit boundary markers, speaker "
            "labels, and turn indices, dropping the score noise. "
            "Targets multi-session and _abs questions where structural "
            "context matters more than rank ordering."
        ),
    )
    run.add_argument(
        "--within-session-oversample",
        action="store_true",
        help=(
            "For each session present in top-k, promote that session's "
            "first-turn and last-turn events from the wider candidate "
            "pool into top-k (substituting lowest-ranked non-boundary "
            "items). First turns often establish topic; last turns "
            "often carry resolution. Boundary turns are read from "
            "`is_first_turn` / `is_last_turn` event metadata (LongMemEval "
            "ingest writes these). Complements --min-sessions-in-topk."
        ),
    )
    run.add_argument(
        "--min-sessions-in-topk",
        type=int,
        default=0,
        help=(
            "Enforce >= N distinct session_ids in the final top-k. "
            "Reorders the post-rerank candidate set to swap in events "
            "from underrepresented sessions when the top-k is "
            "dominated by one session. Targets within-session-rank "
            "failures (JOURNEY sections 16-17): currently the cross-"
            "encoder can fill top-k with 5-9 similar turns from one "
            "wrong session, drowning out the gold turns from the "
            "answer session(s). Default 0 (off). Try 3-5."
        ),
    )
    run.add_argument(
        "--auto-temporal",
        action="store_true",
        help=(
            "Per-question year extraction: scan the question for "
            "\\b(19|20)\\d{2}\\b tokens and pass them as a lexical_filter "
            "regex (OR'd) so the rerank pool only contains events "
            "matching those years. Falls back to unfiltered retrieve "
            "when the filter empties the pool. LongMemEval temporal-"
            "reasoning queries hit ~9-out-of-10 of the time when they "
            "name a year; this surgically protects recall on the rest."
        ),
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
        "--distill-chat",
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
            "Two-Pass Answer Distillation (TPAD): after the primary "
            "answer, run a separate chat to extract just the final "
            "answer from a verbose response. Useful when the primary "
            "LLM produces CoT preambles or hint-echoes that the judge "
            "penalizes. Typically a cheaper model (Haiku, GPT-4o-mini). "
            "Monotonic: falls back to the primary response on failure."
        ),
    )
    run.add_argument(
        "--distill-chat-model",
        default=None,
        help="Model name for --distill-chat (e.g. anthropic/claude-haiku-4-5).",
    )
    run.add_argument(
        "--aconsolidate-concurrency",
        type=int,
        default=8,
        help=(
            "Max concurrent abstraction LLM calls inside aconsolidate. "
            "Default 8 (rate-limit safe). Raise to 30-50 against Haiku / "
            "GPT-4o-mini for ~5-10x faster consolidation."
        ),
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

    run.add_argument(
        "--disk-cache",
        type=Path,
        default=None,
        help=(
            "Path to a sqlite file used as a persistent (chat, embed) "
            "response cache. Re-runs over the same prompts hit disk "
            "instead of the network / GPU. Zero quality lift, huge "
            "cost/wall-time saving on ablation sweeps."
        ),
    )

    return parser


def _resolve_provider(args: argparse.Namespace) -> tuple[Provider, str]:
    """Build a Provider from CLI flags. `--provider fake` is a shortcut.

    Returns ``(provider, provider_hash)``. The hash is computed off the
    UNWRAPPED provider so two runs with the same configured embedder +
    chat -- one cached, one not -- share an identical
    ``provider_hash``. Capturing it post-wrap (H-73) would let the
    DiskCache wrapper drift the hash and break manifest-level
    comparability between cached + uncached sweeps.

    When `--disk-cache PATH` is set, the resulting chat + embed
    providers are wrapped with `with_disk_cache(path=PATH)` so every
    response is cached on disk. The wrapper proxies the original
    surface, so downstream code never sees a difference.
    """
    if args.provider == "fake":
        if args.embedder or args.chat:
            print(
                "warning: --provider fake overrides --embedder/--chat",
                file=sys.stderr,
            )
        provider = FakeProvider()
        return provider, provider.manifest_hash()

    embedder = args.embedder or "fake"
    chat = args.chat or "fake"
    if embedder == "fake" and chat == "fake":
        provider = FakeProvider()
        return provider, provider.manifest_hash()
    # Refuse the silent-meaningless-retrieval trap: a real chat (Anthropic,
    # Moonshot, OpenCode) paired with the fake embedder produces hashed-
    # text fingerprint vectors that bear no relation to the query
    # semantics. The retrieve step then surfaces arbitrary haystack rows
    # and the chat earnestly answers nonsense (H-81). The fix the user
    # almost always wants is `--embedder openai`, since none of those
    # providers ship an embedding endpoint of their own; surface that
    # hint in the error rather than silently degrading.
    if chat in ("anthropic", "moonshot", "opencode-zen", "opencode-go") and embedder == "fake":
        raise ValueError(
            f"--chat {chat!r} has no embedder of its own; pairing with "
            "--embedder fake produces meaningless retrievals. "
            "Pass --embedder openai (or --embedder local for an offline run)."
        )
    # CLI uses "fp16"/"fp32" as shorthand; the embedder accepts the
    # full names so map here.
    dtype_map = {"auto": "auto", "fp16": "float16", "fp32": "float32"}
    real_provider = build_provider(
        embedder_name=embedder,
        chat_name=chat,
        embed_model=args.embed_model,
        embed_dim=args.embed_dim,
        embed_device=args.embed_device,
        embed_dtype=dtype_map[args.dtype],
        chat_model=args.chat_model,
        chat_max_tokens=args.chat_max_tokens,
    )
    # Snapshot the manifest hash BEFORE wrapping -- H-73. The disk-cache
    # wrapper doesn't proxy `manifest_hash`, so a post-wrap capture
    # would drift between cached and uncached runs of the same config.
    pre_wrap_hash = real_provider.manifest_hash()
    if args.disk_cache is not None:
        from engram.providers._disk_cache import with_disk_cache

        cache_path = Path(args.disk_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        real_provider.embedder = with_disk_cache(real_provider.embedder, path=str(cache_path))
        real_provider.chat = with_disk_cache(real_provider.chat, path=str(cache_path))
        print(f"disk cache enabled: {cache_path}", file=sys.stderr)
    return real_provider, pre_wrap_hash


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
    if args.bm25_weight and args.bm25_weight > 0:
        cfg["bm25_weight"] = args.bm25_weight
    if args.mmr_lambda and args.mmr_lambda > 0:
        cfg["mmr_lambda"] = args.mmr_lambda
    if args.recency_lambda and args.recency_lambda > 0:
        cfg["recency_lambda"] = args.recency_lambda
    if args.drill_k is not None:
        cfg["drill_k"] = args.drill_k
    if args.confidence_threshold is not None:
        cfg["confidence_threshold"] = args.confidence_threshold
    if args.rerank_pool_multiplier is not None:
        cfg["candidate_multiplier"] = args.rerank_pool_multiplier
    if args.bm25_k1 != 1.5:
        cfg["bm25_k1"] = args.bm25_k1
    if args.bm25_b != 0.75:
        cfg["bm25_b"] = args.bm25_b
    if args.recency_decay_days != 90.0:
        cfg["recency_decay_days"] = args.recency_decay_days
    if args.mmr_pool_size > 0:
        cfg["mmr_pool_size"] = args.mmr_pool_size
    if args.recent_window_k > 0:
        cfg["recent_window_k"] = args.recent_window_k
    if args.auto_temporal:
        cfg["auto_temporal"] = True
    if args.min_sessions_in_topk > 0:
        cfg["min_sessions_in_topk"] = args.min_sessions_in_topk
    if args.within_session_oversample:
        cfg["within_session_oversample"] = True
    if args.context_format != "flat":
        cfg["context_format"] = args.context_format
    if args.answer_form != "freeform":
        cfg["answer_form"] = args.answer_form
    if args.enable_tools:
        cfg["enable_tools"] = True
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
        cfg["consolidate_chat"] = build_chat(args.consolidate_chat, args.consolidate_chat_model)
    if args.aconsolidate_concurrency != 8:
        cfg["aconsolidate_concurrency"] = args.aconsolidate_concurrency
    if args.judge_chat:
        cfg["judge_chat"] = build_chat(args.judge_chat, args.judge_chat_model)
    if args.distill_chat:
        cfg["distill_chat"] = build_chat(args.distill_chat, args.distill_chat_model)
    if args.sample is not None:
        cfg["sample_n"] = args.sample
    if args.parallel != 1:
        cfg["parallel"] = args.parallel
    if args.gpu_concurrency != 1:
        cfg["gpu_concurrency"] = args.gpu_concurrency
    if args.prompt_version != "v1":
        cfg["prompt_version"] = args.prompt_version
    return cfg


def _engram_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Snapshot retrieval-affecting CLI args into a manifest-friendly dict.

    Skips file paths, log levels, and the deferred-to-suite items
    (`--env-file`, `--runs-dir`, `--disk-cache`, `--log-level`). Skips
    reranker model objects too -- the model name lives in
    `args.reranker_model`, which we keep. Captures the resolved
    primary chat + embedder so a sweep with different `--chat-model`
    values shows up explicitly in the manifest rather than collapsing
    behind the coarse `--chat` choice (M-151 hint).
    """
    skip = {
        "command",
        "env_file",
        "runs_dir",
        "disk_cache",
        "log_level",
        "provider",
    }
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if k in skip:
            continue
        # Path / object values would force `default=str` in JSON; the
        # manifest writer already does that. Stringify Path values
        # eagerly here so a JSON-walking consumer doesn't have to.
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if _load_env_file(args.env_file):
            print(f"loaded {args.env_file}", file=sys.stderr)
        # `.env` may carry `ENGRAM_LOG_LEVEL=DEBUG`; load it before
        # configuring logging so user overrides take effect.
        _configure_logging(os.environ.get("ENGRAM_LOG_LEVEL", "INFO"))
        # `--gpu-concurrency` sets the engram._gpu_lock semaphore size
        # via env var.  Must happen BEFORE provider construction so
        # the LocalEmbedder / BGEReranker pick up the cap on their
        # first forward pass (the semaphore is lazily initialized
        # from the env var).
        if args.gpu_concurrency != 1:
            os.environ["ENGRAM_GPU_CONCURRENCY"] = str(args.gpu_concurrency)
            print(
                f"--gpu-concurrency {args.gpu_concurrency} applied",
                file=sys.stderr,
            )
        # `--limit` is plumbed via `suite_config` (audit H-74). Suites
        # respect it in their `configure(**)` method; the old behaviour
        # of mutating `os.environ` left the variable set across
        # in-process suite runs and only worked at all for LongMemEval
        # (LoCoMo never read `LOCOMO_MAX_QUESTIONS`).
        if args.limit is not None and args.sample is None:
            print(f"--limit {args.limit} applied", file=sys.stderr)
        elif args.sample is not None:
            print(
                f"--sample {args.sample} applied (stratified across qtypes)",
                file=sys.stderr,
            )
        try:
            provider, provider_hash = _resolve_provider(args)
            suite_config = _resolve_suite_config(args)
            # Capture the primary chat + embedder in engram_config so
            # the manifest's reproducibility ledger includes them as
            # structured descriptors (name / model / manifest_hash),
            # not just the composite provider_hash. Previously only
            # judge_chat / consolidate_chat / distill_chat appeared
            # here, which made it impossible to tell from a manifest
            # which answer-generation backend was in use — see JOURNEY
            # §24 (had to grovel through provider_hash to identify
            # the opencode-go thinking-mode regime shift).
            if args.chat and "chat" not in suite_config:
                suite_config["chat"] = provider.chat
            if args.embedder and "embedder" not in suite_config:
                suite_config["embedder"] = provider.embedder
            # H-74: plumb --limit via suite_config instead of os.environ.
            if args.limit is not None:
                suite_config["max_questions"] = args.limit
            engram_config = _engram_config_from_args(args)
            manifest_path = run_suite(
                args.suite,
                provider=provider,
                runs_dir=args.runs_dir,
                suite_config=suite_config,
                provider_hash=provider_hash,
                engram_config=engram_config,
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"manifest: {manifest_path}")
        return 0

    parser.print_help()
    return 1
