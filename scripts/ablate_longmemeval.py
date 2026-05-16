"""Per-question ablation harness for LongMemEval retrieval flags.

Build the haystack once per question, then replay it through N retrieve
configs side-by-side. Two metrics per (question, config):

  * retrieval recall: did the correct haystack session(s) appear in
    top-k? Computed from `Event.metadata.session_id` vs
    `q.answer_session_ids` -- no LLM call required.
  * (optional) answer accuracy: feed the retrieved set to the chat
    provider, judge the response. Costs N LLM calls per question per
    config; on by default but skip with --retrieval-only for fast
    sweeps.

Output:
  * JSON manifest with per-(question, config) rows.
  * Markdown summary printed to stdout with per-config aggregates +
    per-question pass/fail matrix so you can eyeball which configs
    hurt which questions.

Examples:

  # Fast: just retrieval recall on 30 multi-session questions.
  python scripts/ablate_longmemeval.py --qtype multi-session --limit 30 \
    --retrieval-only \
    --configs baseline,bm25,mmr07,recent,recency,autotemp,all_aggressive

  # Full: retrieval recall + answer + judge on a small subset (slow).
  python scripts/ablate_longmemeval.py --qtype multi-session --limit 10 \
    --configs baseline,bm25,all_aggressive --chat opencode-go \
    --chat-model kimi-k2.6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add repo root to sys.path so `from engram import ...` and
# `from benchmarks.suites.longmemeval import ...` work when running this
# script via `python scripts/ablate_longmemeval.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_env(path: Path = Path(".env")) -> None:
    """Best-effort `.env` loader. Existing env vars take precedence."""
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
from engram.providers._message import Message  # noqa: E402
from engram.providers._protocols import ChatProvider  # noqa: E402
from engram.retrieve._params import RetrieveParams  # noqa: E402

from benchmarks.suites.longmemeval import (  # noqa: E402
    DATASET_ROOT,
    DEFAULT_FILENAME,
    _Question,
    _build_auto_temporal_filter,
    _format_memory,
    _ingest_haystack,
    _judge,
    _load_dataset,
    _parse_haystack_date,
    _read_prompt,
)

_LOG = logging.getLogger("engram.ablate")


# Each entry is a dict of RetrieveParams overrides + bench-level flags.
# `_auto_temporal: True` is a bench-level flag (per-question lexical
# filter computed from year tokens, with empty-pool fallback); all
# other keys map directly to RetrieveParams fields.
CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "bm25": {"bm25_weight": 1.0},
    "mmr07": {"mmr_lambda": 0.7},
    "mmr03": {"mmr_lambda": 0.3},
    "recent": {"recent_window_k": 10},
    "recency": {"recency_lambda": 0.1},
    "autotemp": {"_auto_temporal": True},
    "bm25+aut": {"bm25_weight": 1.0, "_auto_temporal": True},
    "bm25+mmr": {"bm25_weight": 1.0, "mmr_lambda": 0.7},
    "bm25+rec": {"bm25_weight": 1.0, "recent_window_k": 10},
    "all_aggressive": {
        "bm25_weight": 1.0,
        "mmr_lambda": 0.7,
        "recency_lambda": 0.1,
        "recent_window_k": 10,
        "_auto_temporal": True,
    },
    "conservative": {
        "bm25_weight": 1.0,
        "_auto_temporal": True,
    },
}


@dataclass(frozen=True, slots=True)
class _RunResult:
    """One row of the ablation matrix."""

    qid: str
    qtype: str
    config: str
    recall: float
    binary_hit: float
    top_k_session_ids: tuple[str, ...]
    answer_session_ids: tuple[str, ...]
    score: float | None  # None when --retrieval-only
    response: str | None
    error: str | None
    latency_ms: float


def _build_chat(chat_name: str, chat_model: str | None) -> ChatProvider:
    """Construct a chat provider via the bench's standard catalog."""
    from engram.bench._real_provider import build_chat

    return build_chat(chat_name, chat_model)


def _build_embedder(model: str, device: str | None, dtype: str) -> Any:
    """Construct a LocalEmbedder once and share across the whole run."""
    from engram.providers.local import LocalEmbedder

    dtype_map: dict[str, str] = {"auto": "auto", "fp16": "float16", "fp32": "float32"}
    return LocalEmbedder(
        model=model,
        device=device,
        dtype=dtype_map.get(dtype, "auto"),  # type: ignore[arg-type]
    )


def _build_reranker(model: str, device: str | None, dtype: str) -> Any:
    from engram.retrieve._bge_reranker import BGEReranker

    dtype_map: dict[str, str] = {"auto": "auto", "fp16": "float16", "fp32": "float32"}
    return BGEReranker(
        model=model,
        device=device,
        dtype=dtype_map.get(dtype, "auto"),  # type: ignore[arg-type]
    )


def _retrieved_session_ids(
    memory: Memory, results: Sequence[Any]
) -> set[str]:
    """Pull session_id from each result via the Event metadata blob.

    Memory items (consolidated abstractions) don't carry a session_id
    directly, but every event does. For Stage 6 consolidation, abstract
    items have provenance pointing to the events they were derived
    from; we union those sessions too so consolidation-mode retrieval
    still scores correctly.
    """
    out: set[str] = set()
    for r in results:
        try:
            event = memory.storage.get_event(r.item_id)
        except (KeyError, RuntimeError):  # pragma: no cover - defensive
            event = None
        if event is not None:
            sid = event.metadata.get("session_id")
            if isinstance(sid, str):
                out.add(sid)
            continue
        # Memory item: walk provenance.
        try:
            supports = memory.storage.get_supporting_events(r.item_id)
        except (KeyError, RuntimeError):  # pragma: no cover - defensive
            supports = []
        for ev in supports:
            sid = ev.metadata.get("session_id")
            if isinstance(sid, str):
                out.add(sid)
    return out


def _run_one(
    *,
    q: _Question,
    config_name: str,
    config: dict[str, Any],
    storage: SqliteStorage,
    embedder: Any,
    reranker: Any,
    chat: Any,
    k: int,
    retrieval_only: bool,
) -> _RunResult:
    """Run one (question, config) pair against an already-ingested storage."""
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

    error: str | None = None
    response: str | None = None
    score: float | None = None
    t_start = time.perf_counter()
    try:
        retrieve_kwargs: dict[str, Any] = {"k": k, "reinforce": False}
        if config.get("recency_lambda", 0) and config["recency_lambda"] > 0:
            question_dt = _parse_haystack_date(q.question_date)
            if question_dt is not None:
                retrieve_kwargs["as_of"] = question_dt
        if auto_temporal:
            filt = _build_auto_temporal_filter(q.question)
            if filt:
                retrieve_kwargs["lexical_filter"] = filt
        results = memory.retrieve(q.question, **retrieve_kwargs)
        if (
            auto_temporal
            and not results
            and retrieve_kwargs.get("lexical_filter")
        ):
            retrieve_kwargs.pop("lexical_filter", None)
            results = memory.retrieve(q.question, **retrieve_kwargs)

        retrieved = _retrieved_session_ids(memory, results)
        answer_sessions = set(q.answer_session_ids)
        if answer_sessions:
            recall = len(retrieved & answer_sessions) / len(answer_sessions)
            binary = 1.0 if (retrieved & answer_sessions) else 0.0
        else:
            recall = 0.0
            binary = 0.0

        if not retrieval_only:
            memory_text = _format_memory(results)
            prompt = _read_prompt("answer").format(
                memory=memory_text,
                question=q.question,
                question_date=q.question_date or "(date unknown)",
            )
            response = chat.chat([Message(role="user", content=prompt)])
            correct = _judge(
                chat,
                qtype=q.qtype,
                question=q.question,
                gold=q.gold,
                response=response,
            )
            score = 1.0 if correct else 0.0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        recall = 0.0
        binary = 0.0
        retrieved = set()

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    return _RunResult(
        qid=q.qid,
        qtype=q.qtype,
        config=config_name,
        recall=recall,
        binary_hit=binary,
        top_k_session_ids=tuple(sorted(retrieved)),
        answer_session_ids=tuple(q.answer_session_ids),
        score=score,
        response=response,
        error=error,
        latency_ms=latency_ms,
    )


def _filter_questions(
    questions: list[_Question],
    *,
    qtype: str | None,
    qids: list[str] | None,
    limit: int | None,
) -> list[_Question]:
    out = questions
    if qtype:
        out = [q for q in out if q.qtype == qtype]
    if qids:
        wanted = set(qids)
        out = [q for q in out if q.qid in wanted]
    if limit is not None:
        out = out[:limit]
    return out


def _markdown_summary(results: list[_RunResult], configs: list[str]) -> str:
    """Per-config aggregate + per-question pass/fail matrix."""
    by_config: dict[str, list[_RunResult]] = {c: [] for c in configs}
    for r in results:
        by_config[r.config].append(r)

    lines: list[str] = []
    lines.append("## Aggregate per config\n")
    lines.append(
        "| Config | N | Mean recall | Top-k hit rate | Mean LLM score | Mean latency (ms) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in configs:
        rows = by_config[c]
        if not rows:
            continue
        n = len(rows)
        mean_recall = sum(r.recall for r in rows) / n
        hit_rate = sum(r.binary_hit for r in rows) / n
        scored = [r for r in rows if r.score is not None]
        mean_score = (
            sum(r.score for r in scored) / len(scored)  # type: ignore[misc]
            if scored
            else None
        )
        mean_lat = sum(r.latency_ms for r in rows) / n
        score_str = f"{mean_score:.3f}" if mean_score is not None else "n/a"
        lines.append(
            f"| {c} | {n} | {mean_recall:.3f} | {hit_rate:.3f} | {score_str} | {mean_lat:.0f} |"
        )

    # Per-question matrix: rows = qid, columns = config.
    qids = sorted({r.qid for r in results})
    lines.append("\n## Per-question matrix (binary hit = correct session in top-k)\n")
    header = "| qid | qtype | " + " | ".join(configs) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(configs)) + "|"
    lines.append(header)
    lines.append(sep)
    by_qid: dict[str, dict[str, _RunResult]] = {}
    qtype_by_qid: dict[str, str] = {}
    for r in results:
        by_qid.setdefault(r.qid, {})[r.config] = r
        qtype_by_qid[r.qid] = r.qtype
    for qid in qids:
        row_results = by_qid[qid]
        cells: list[str] = []
        for c in configs:
            r = row_results.get(c)
            if r is None:
                cells.append("·")
            elif r.error:
                cells.append("ERR")
            else:
                cells.append(f"{r.binary_hit:.0f}")
        lines.append(f"| {qid[:8]} | {qtype_by_qid[qid]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-question retrieve-config ablation for LongMemEval."
    )
    parser.add_argument(
        "--qtype",
        default=None,
        help="Filter to this question_type (e.g. multi-session, single-session-user).",
    )
    parser.add_argument(
        "--qid",
        default=None,
        help="Comma-separated question_id list. Overrides --qtype if both set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of questions after filtering. Default: all matching.",
    )
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS.keys()),
        help=f"Comma-separated configs to run. Available: {','.join(CONFIGS.keys())}",
    )
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-large-en-v1.5",
    )
    parser.add_argument("--embed-device", default=None)
    parser.add_argument(
        "--dtype", default="fp32", choices=("auto", "fp16", "fp32")
    )
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip the answer+judge LLM step; only compute retrieval recall.",
    )
    parser.add_argument("--chat", default="opencode-go")
    parser.add_argument("--chat-model", default="kimi-k2.6")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/runs/ablation.json"),
        help="JSON manifest output path.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("ENGRAM_LOG_LEVEL", "INFO"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Resolve config list.
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in config_names if c not in CONFIGS]
    if unknown:
        print(f"unknown configs: {unknown}", file=sys.stderr)
        print(f"available: {list(CONFIGS.keys())}", file=sys.stderr)
        return 2

    # Load dataset.
    dataset_path = DATASET_ROOT / DEFAULT_FILENAME
    all_questions = _load_dataset(dataset_path)
    if not all_questions:
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    qids = [q.strip() for q in args.qid.split(",") if q.strip()] if args.qid else None
    questions = _filter_questions(
        all_questions, qtype=args.qtype, qids=qids, limit=args.limit
    )
    if not questions:
        print("no questions match the filter", file=sys.stderr)
        return 2
    _LOG.info(
        "ablation: %d questions × %d configs = %d runs (retrieval-only=%s)",
        len(questions),
        len(config_names),
        len(questions) * len(config_names),
        args.retrieval_only,
    )

    # Build providers once.
    embedder = _build_embedder(args.embed_model, args.embed_device, args.dtype)
    reranker = _build_reranker(args.reranker, args.embed_device, args.dtype)
    chat = _build_chat(args.chat, args.chat_model)

    results: list[_RunResult] = []
    for q_idx, q in enumerate(questions):
        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            ingest_memory = Memory(
                storage=storage,
                embedder=embedder,
                chat=chat,
                reranker=reranker,
            )
            t_ingest = time.perf_counter()
            turns = _ingest_haystack(ingest_memory, q)
            ingest_ms = (time.perf_counter() - t_ingest) * 1000.0
            _LOG.info(
                "q %d/%d [%s] qid=%s ingested %d turns in %.1fs",
                q_idx + 1,
                len(questions),
                q.qtype,
                q.qid[:8],
                turns,
                ingest_ms / 1000.0,
            )
            for config_name in config_names:
                config = CONFIGS[config_name]
                row = _run_one(
                    q=q,
                    config_name=config_name,
                    config=config,
                    storage=storage,
                    embedder=embedder,
                    reranker=reranker,
                    chat=chat,
                    k=args.k,
                    retrieval_only=args.retrieval_only,
                )
                results.append(row)
                _LOG.info(
                    "  %s -> recall=%.2f hit=%d score=%s lat=%.0fms %s",
                    config_name.ljust(15),
                    row.recall,
                    int(row.binary_hit),
                    f"{row.score:.0f}" if row.score is not None else "·",
                    row.latency_ms,
                    f"ERR={row.error}" if row.error else "",
                )
        finally:
            storage.close()

    # Persist + print summary.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_args": vars(args) | {"output": str(args.output)},
        "results": [
            {
                "qid": r.qid,
                "qtype": r.qtype,
                "config": r.config,
                "recall": r.recall,
                "binary_hit": r.binary_hit,
                "top_k_session_ids": list(r.top_k_session_ids),
                "answer_session_ids": list(r.answer_session_ids),
                "score": r.score,
                "response": r.response,
                "error": r.error,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _LOG.info("wrote %d rows to %s", len(results), args.output)

    print(_markdown_summary(results, config_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
