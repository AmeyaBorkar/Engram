"""Per-question retrieval trace tool.

For a single LongMemEval question (specified by qid or sampled from a
qtype), dump every retrieval stage's output side-by-side:

  1. The question + gold answer + answer_session_ids.
  2. Dense top-N (raw cosine, no rerank).
  3. BM25 top-N (lexical, raw scores).
  4. Recent-window top-N (most-recent events).
  5. Final top-k under each of M configs.

Every retrieved event is annotated:
  - `[GOLD]` if its session is in `answer_session_ids`
  - `[....]` otherwise

plus rank, score, session_id (last 4 chars), and a 60-char content
preview. This lets you see exactly which stage surfaces the right
evidence (or doesn't) -- the answer to "is this component design or
implementation" almost always lives in this trace.

Usage:
  # Trace a specific qid
  python scripts/retrieval_trace.py --qid 6d550036 --dtype fp32 --embed-device cuda

  # Trace the first 3 multi-session questions
  python scripts/retrieval_trace.py --qtype multi-session --limit 3 --dtype fp32 --embed-device cuda

  # Trace a question that bm25+mmr broke (from the ablation output)
  python scripts/retrieval_trace.py --qid 6d550036 --configs baseline,bm25,mmr07,bm25+mmr --dtype fp32 --embed-device cuda
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
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


_load_env()

from engram import Memory, SqliteStorage  # noqa: E402
from engram.providers._fake import FakeChat  # noqa: E402
from engram.retrieve._params import RetrieveParams  # noqa: E402

from benchmarks.suites.longmemeval import (  # noqa: E402
    DATASET_ROOT,
    DEFAULT_FILENAME,
    _Question,
    _build_auto_temporal_filter,
    _ingest_haystack,
    _load_dataset,
    _parse_haystack_date,
)
from scripts.ablate_longmemeval import CONFIGS  # noqa: E402

_LOG = logging.getLogger("engram.trace")


def _build_embedder(model: str, device: str | None, dtype: str) -> Any:
    from engram.providers.local import LocalEmbedder

    dtype_map: dict[str, str] = {"auto": "auto", "fp16": "float16", "fp32": "float32"}
    return LocalEmbedder(
        model=model,
        device=device,
        dtype=dtype_map.get(dtype, "auto"),  # type: ignore[arg-type]
    )


def _build_reranker(model: str | None, device: str | None, dtype: str) -> Any:
    if model is None or model.lower() == "none":
        return None
    from engram.retrieve._bge_reranker import BGEReranker

    dtype_map: dict[str, str] = {"auto": "auto", "fp16": "float16", "fp32": "float32"}
    return BGEReranker(
        model=model,
        device=device,
        dtype=dtype_map.get(dtype, "auto"),  # type: ignore[arg-type]
    )


def _normalize(vec: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return list(vec)
    return [x / n for x in vec]


def _preview(content: str, width: int = 60) -> str:
    """Single-line preview of an event's content, escaping newlines."""
    flat = content.replace("\n", " ").replace("\r", " ")
    if len(flat) <= width:
        return flat
    return flat[: width - 1] + "…"


def _session_tag(session_id: str | None, answer_sessions: set[str]) -> str:
    """`[GOLD]` if the session is in the gold set, `[....]` otherwise.

    Trailing 4 chars of the session_id for orientation (full ids are
    too long for the trace columns).
    """
    if session_id is None:
        return "[NONE] none"
    suffix = session_id[-4:]
    marker = "[GOLD]" if session_id in answer_sessions else "[....]"
    return f"{marker} {suffix}"


def _trace_dense(
    *,
    storage: SqliteStorage,
    embedder: Any,
    question: str,
    k: int,
    answer_sessions: set[str],
) -> list[str]:
    """Run dense vector retrieve directly via storage (no rerank, no fusion)."""
    query_vec = embedder.embed([question])[0]
    normalized = _normalize(query_vec)
    hits = storage.search_event_embeddings(
        normalized,
        k=k,
        model=embedder.model,
    )
    lines = [f"\n--- Dense top-{k} (raw cosine, no rerank) ---"]
    lines.append(f"{'rank':>4} | {'session':<12} | {'score':>6} | content")
    lines.append(f"{'-'*4}-+-{'-'*12}-+-{'-'*6}-+-{'-'*60}")
    for i, (eid, content, score) in enumerate(hits, start=1):
        ev = storage.get_event(eid)
        sid = ev.metadata.get("session_id") if ev else None
        tag = _session_tag(sid if isinstance(sid, str) else None, answer_sessions)
        lines.append(f"{i:>4} | {tag:<12} | {score:>6.3f} | {_preview(content)}")
    return lines


def _trace_bm25(
    *,
    storage: SqliteStorage,
    question: str,
    k: int,
    answer_sessions: set[str],
) -> list[str]:
    """Run BM25 directly via storage helper."""
    try:
        hits = storage.bm25_search_events(question, k=k)
    except (ValueError, RuntimeError) as exc:
        return [f"\n--- BM25 top-{k} (ERROR: {exc}) ---"]
    lines = [f"\n--- BM25 top-{k} (lexical, raw scores) ---"]
    lines.append(f"{'rank':>4} | {'session':<12} | {'score':>6} | content")
    lines.append(f"{'-'*4}-+-{'-'*12}-+-{'-'*6}-+-{'-'*60}")
    if not hits:
        lines.append("(no BM25 hits for this query)")
        return lines
    for i, (eid, content, score) in enumerate(hits, start=1):
        ev = storage.get_event(eid)
        sid = ev.metadata.get("session_id") if ev else None
        tag = _session_tag(sid if isinstance(sid, str) else None, answer_sessions)
        lines.append(f"{i:>4} | {tag:<12} | {score:>6.2f} | {_preview(content)}")
    return lines


def _trace_recent(
    *,
    storage: SqliteStorage,
    k: int,
    answer_sessions: set[str],
) -> list[str]:
    """Run the recent-window list directly via storage helper."""
    hits = storage.list_recent_events(k=k)
    lines = [f"\n--- Recent-window top-{k} (created_at DESC) ---"]
    lines.append(f"{'rank':>4} | {'session':<12} | content")
    lines.append(f"{'-'*4}-+-{'-'*12}-+-{'-'*60}")
    for i, (eid, content) in enumerate(hits, start=1):
        ev = storage.get_event(eid)
        sid = ev.metadata.get("session_id") if ev else None
        tag = _session_tag(sid if isinstance(sid, str) else None, answer_sessions)
        lines.append(f"{i:>4} | {tag:<12} | {_preview(content)}")
    return lines


def _trace_config(
    *,
    config_name: str,
    config: dict[str, Any],
    storage: SqliteStorage,
    embedder: Any,
    reranker: Any,
    chat: Any,
    q: _Question,
    k: int,
    answer_sessions: set[str],
) -> list[str]:
    """Run the full retrieve pipeline with one config and dump top-k."""
    auto_temporal = bool(config.get("_auto_temporal", False))
    param_overrides = {kk: vv for kk, vv in config.items() if not kk.startswith("_")}
    base_params = RetrieveParams(k=k, **param_overrides)
    memory = Memory(
        storage=storage,
        embedder=embedder,
        chat=chat,
        retrieve_params=base_params,
        reranker=reranker,
    )
    retrieve_kwargs: dict[str, Any] = {"k": k, "reinforce": False}
    if config.get("recency_lambda", 0) and config["recency_lambda"] > 0:
        question_dt = _parse_haystack_date(q.question_date)
        if question_dt is not None:
            retrieve_kwargs["as_of"] = question_dt
    used_filter: str | None = None
    if auto_temporal:
        filt = _build_auto_temporal_filter(q.question)
        if filt:
            retrieve_kwargs["lexical_filter"] = filt
            used_filter = filt
    t0 = time.perf_counter()
    try:
        results = memory.retrieve(q.question, **retrieve_kwargs)
    except Exception as exc:
        return [f"\n--- Config: {config_name} (ERROR: {exc}) ---"]
    if auto_temporal and not results and "lexical_filter" in retrieve_kwargs:
        retrieve_kwargs.pop("lexical_filter", None)
        results = memory.retrieve(q.question, **retrieve_kwargs)
        used_filter = f"{used_filter} (fell back, no hits)"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    retrieved_sessions: list[str | None] = []
    for r in results:
        try:
            ev = memory.storage.get_event(r.item_id)
        except (KeyError, RuntimeError):
            ev = None
        if ev is not None:
            sid = ev.metadata.get("session_id")
            retrieved_sessions.append(sid if isinstance(sid, str) else None)
        else:
            try:
                supports = memory.storage.get_supporting_events(r.item_id)
            except (KeyError, RuntimeError):
                supports = []
            sid_via_provenance: str | None = None
            for ev in supports:
                sid = ev.metadata.get("session_id")
                if isinstance(sid, str):
                    sid_via_provenance = sid
                    break
            retrieved_sessions.append(sid_via_provenance)

    n_gold = sum(1 for s in retrieved_sessions if s in answer_sessions)
    n_unique_gold = len({s for s in retrieved_sessions if s in answer_sessions})
    recall = n_unique_gold / max(len(answer_sessions), 1)
    lines = [
        f"\n--- Config: {config_name} (top-{k}, {elapsed_ms:.0f}ms) ---"
    ]
    if used_filter:
        lines.append(f"    [auto-temporal filter: {used_filter}]")
    lines.append(
        f"    recall@{k}={recall:.2f}  ({n_unique_gold}/{len(answer_sessions)} answer sessions; "
        f"{n_gold} of {len(results)} retrieved items are from a gold session)"
    )
    lines.append(f"{'rank':>4} | {'session':<12} | {'level':<10} | {'score':>6} | content")
    lines.append(f"{'-'*4}-+-{'-'*12}-+-{'-'*10}-+-{'-'*6}-+-{'-'*60}")
    for i, (r, sid) in enumerate(zip(results, retrieved_sessions, strict=True), start=1):
        tag = _session_tag(sid, answer_sessions)
        lines.append(
            f"{i:>4} | {tag:<12} | {r.level.value:<10} | {r.score:>6.3f} | {_preview(r.content)}"
        )
    return lines


def _trace_question(
    *,
    q: _Question,
    storage: SqliteStorage,
    embedder: Any,
    reranker: Any,
    chat: Any,
    config_names: list[str],
    deep_n: int,
    k: int,
) -> str:
    """Trace one full question through every stage and config."""
    answer_sessions = set(q.answer_session_ids)
    n_haystack_events = sum(len(s) for s in q.haystack_sessions)
    header = [
        "=" * 80,
        f"qid={q.qid}  [{q.qtype}]",
        f"question_date={q.question_date}",
        f"Q: {q.question}",
        f"GOLD: {q.gold}",
        f"answer_session_ids ({len(answer_sessions)}): "
        + ", ".join(s[-4:] for s in sorted(answer_sessions)),
        f"haystack: {len(q.haystack_sessions)} sessions, {n_haystack_events} events",
        "=" * 80,
    ]

    parts = [*header]
    parts.extend(_trace_dense(
        storage=storage, embedder=embedder, question=q.question,
        k=deep_n, answer_sessions=answer_sessions,
    ))
    parts.extend(_trace_bm25(
        storage=storage, question=q.question, k=deep_n, answer_sessions=answer_sessions,
    ))
    parts.extend(_trace_recent(
        storage=storage, k=min(deep_n, 20), answer_sessions=answer_sessions,
    ))

    parts.append("\n" + "-" * 80)
    parts.append("FINAL TOP-K PER CONFIG")
    parts.append("-" * 80)
    for c in config_names:
        if c not in CONFIGS:
            parts.append(f"\n  (unknown config: {c}; skipping)")
            continue
        parts.extend(_trace_config(
            config_name=c,
            config=CONFIGS[c],
            storage=storage,
            embedder=embedder,
            reranker=reranker,
            chat=chat,
            q=q,
            k=k,
            answer_sessions=answer_sessions,
        ))
    return "\n".join(parts)


def _filter_questions(
    questions: list[_Question],
    *,
    qid: str | None,
    qtype: str | None,
    limit: int | None,
) -> list[_Question]:
    if qid:
        # Allow partial-prefix matches for convenience (qids are long).
        return [q for q in questions if q.qid.startswith(qid)][:1]
    if qtype:
        questions = [q for q in questions if q.qtype == qtype]
    if limit is not None:
        questions = questions[:limit]
    return questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trace every retrieval stage on a single LongMemEval question."
    )
    parser.add_argument("--qid", default=None, help="qid (prefix allowed).")
    parser.add_argument("--qtype", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--configs",
        default="baseline,bm25,mmr07,bm25+mmr,all_aggressive",
        help=f"Configs to trace. Available: {','.join(CONFIGS.keys())}",
    )
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--dtype", default="fp32", choices=("auto", "fp16", "fp32"))
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--k", type=int, default=10, help="Top-k for per-config trace.")
    parser.add_argument(
        "--deep-n",
        type=int,
        default=30,
        help="Top-N for per-stage raw dumps (dense / BM25 / recent).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write trace to file too.")
    parser.add_argument("--log-level", default=os.environ.get("ENGRAM_LOG_LEVEL", "WARNING"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    dataset_path = DATASET_ROOT / DEFAULT_FILENAME
    all_questions = _load_dataset(dataset_path)
    if not all_questions:
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    questions = _filter_questions(
        all_questions, qid=args.qid, qtype=args.qtype, limit=args.limit
    )
    if not questions:
        print("no questions match the filter", file=sys.stderr)
        return 2

    embedder = _build_embedder(args.embed_model, args.embed_device, args.dtype)
    reranker = _build_reranker(args.reranker, args.embed_device, args.dtype)
    chat = FakeChat()

    out_chunks: list[str] = []
    for q in questions:
        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            memory = Memory(storage=storage, embedder=embedder, chat=chat, reranker=reranker)
            t0 = time.perf_counter()
            turns = _ingest_haystack(memory, q)
            _LOG.warning(
                "[ingest] qid=%s turns=%d in %.1fs",
                q.qid, turns, time.perf_counter() - t0,
            )
            chunk = _trace_question(
                q=q,
                storage=storage,
                embedder=embedder,
                reranker=reranker,
                chat=chat,
                config_names=config_names,
                deep_n=args.deep_n,
                k=args.k,
            )
            out_chunks.append(chunk)
        finally:
            storage.close()
    final = "\n\n".join(out_chunks)
    print(final)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(final, encoding="utf-8")
        print(f"\n[trace written to {args.output}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
