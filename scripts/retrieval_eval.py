"""Retrieval-only evaluator for LongMemEval.

Validates the WHOLE retrieve pipeline against ground truth without
spending a single LLM call. For each question:

  1. Ingest the haystack (events tagged with `session_id` metadata).
  2. For each config, run `Memory.retrieve(query, k=50)`.
  3. Score retrieval against `question.answer_session_ids`:
        - recall@k = |retrieved_sessions ∩ answer_sessions| / |answer|
        - hit@k    = 1 if any answer session in retrieved, else 0
        - multi_recall@k = 1 if ALL answer sessions retrieved, else 0
        - mrr      = 1 / rank of first correct event (None -> 0)
        - first_correct_rank = rank of first correct event (1-indexed) or None
        - precision@k = (correct events in top-k) / k

Three k cutoffs (10 / 20 / 50) tell you where the failure mode is:

  - High recall@50, low recall@10  -> dense ranks too low; rerank/MMR
                                       is the problem
  - Low recall@50 across the board -> dense retrieval is broken;
                                       embedder or BM25 or hybrid logic
                                       is wrong
  - Multi-recall << hit -> retrieve finds ONE answer session but
                            misses the others; multi-session failure

Cost: pure CPU + GPU (no API). ~30-45 min for 500 questions × N configs
on a 12GB CUDA card with BGE-large + reranker. Disk: ~5 MB JSON output.

Usage:
  # Compare conservative vs aggressive on multi-session (fast)
  python scripts/retrieval_eval.py --qtype multi-session --limit 60 \\
    --configs baseline,conservative,all_aggressive --dtype fp32

  # Full 500q eval, all configs (slow)
  python scripts/retrieval_eval.py --configs baseline,conservative,bm25,mmr07,recent,all_aggressive \\
    --dtype fp32 --output benchmarks/runs/retrieval_eval_full.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, asdict
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

_LOG = logging.getLogger("engram.retrieval_eval")


# Each config is a dict of RetrieveParams overrides + bench-level flags.
# `_auto_temporal: True` triggers per-question year extraction (handled
# in the bench loop, not RetrieveParams).
CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "bm25": {"bm25_weight": 1.0},
    "mmr07": {"mmr_lambda": 0.7},
    "mmr03": {"mmr_lambda": 0.3},
    "recent": {"recent_window_k": 10},
    "recency": {"recency_lambda": 0.1},
    "autotemp": {"_auto_temporal": True},
    "conservative": {"bm25_weight": 1.0, "_auto_temporal": True},
    "bm25+mmr": {"bm25_weight": 1.0, "mmr_lambda": 0.7},
    "bm25+rec": {"bm25_weight": 1.0, "recent_window_k": 10},
    "all_aggressive": {
        "bm25_weight": 1.0,
        "mmr_lambda": 0.7,
        "recency_lambda": 0.1,
        "recent_window_k": 10,
        "_auto_temporal": True,
    },
}

K_CUTOFFS: tuple[int, ...] = (10, 20, 50)


@dataclass(frozen=True, slots=True)
class _Metrics:
    recall_at: dict[int, float]
    hit_at: dict[int, float]
    multi_recall_at: dict[int, float]
    precision_at: dict[int, float]
    mrr: float
    first_correct_rank: int | None
    n_retrieved: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class _RunRow:
    qid: str
    qtype: str
    config: str
    metrics: _Metrics


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


def _result_to_event_session_pair(
    memory: Memory, result: Any
) -> tuple[str | None, str | None]:
    """Return (event_id_hex, session_id) for one retrieval result.

    For event-level results, session_id comes straight from event
    metadata. For memory-item-level results (consolidated abstractions),
    we walk the provenance and take the FIRST supporting event's
    session_id -- the abstraction is "from" that session for the
    purposes of recall scoring. Returns (None, None) if neither lookup
    succeeds.
    """
    try:
        event = memory.storage.get_event(result.item_id)
    except (KeyError, RuntimeError):
        event = None
    if event is not None:
        sid = event.metadata.get("session_id")
        return (event.id.hex, sid if isinstance(sid, str) else None)
    try:
        supports = memory.storage.get_supporting_events(result.item_id)
    except (KeyError, RuntimeError):
        supports = []
    for ev in supports:
        sid = ev.metadata.get("session_id")
        if isinstance(sid, str):
            return (ev.id.hex, sid)
    return (None, None)


def _compute_metrics(
    results: Sequence[Any],
    memory: Memory,
    answer_session_ids: Sequence[str],
    k_cutoffs: Sequence[int],
    latency_ms: float,
) -> _Metrics:
    """Compute retrieval metrics against the ground truth answer sessions.

    Empty answer_session_ids is treated as "no ground truth" -> all
    metrics 0 (a defensive case; LongMemEval always has at least one).
    """
    answer_set = set(answer_session_ids)
    if not answer_set:
        zero = {k: 0.0 for k in k_cutoffs}
        return _Metrics(
            recall_at=zero,
            hit_at=zero,
            multi_recall_at=zero,
            precision_at=zero,
            mrr=0.0,
            first_correct_rank=None,
            n_retrieved=len(results),
            latency_ms=latency_ms,
        )

    # Map each result to its session_id, in rank order.
    result_sessions: list[str | None] = []
    for r in results:
        _, sid = _result_to_event_session_pair(memory, r)
        result_sessions.append(sid)

    # First correct rank (1-indexed).
    first_correct: int | None = None
    for idx, sid in enumerate(result_sessions, start=1):
        if sid in answer_set:
            first_correct = idx
            break
    mrr = (1.0 / first_correct) if first_correct is not None else 0.0

    recall_at: dict[int, float] = {}
    hit_at: dict[int, float] = {}
    multi_recall_at: dict[int, float] = {}
    precision_at: dict[int, float] = {}
    n_answer = len(answer_set)
    for k in k_cutoffs:
        top_k_sessions = [s for s in result_sessions[:k] if s is not None]
        unique_top = set(top_k_sessions)
        intersect = unique_top & answer_set
        recall_at[k] = len(intersect) / n_answer
        hit_at[k] = 1.0 if intersect else 0.0
        multi_recall_at[k] = 1.0 if intersect == answer_set else 0.0
        # precision: how many of the top-k items are from answer sessions
        n_correct_in_topk = sum(1 for s in top_k_sessions if s in answer_set)
        precision_at[k] = (n_correct_in_topk / k) if k > 0 else 0.0

    return _Metrics(
        recall_at=recall_at,
        hit_at=hit_at,
        multi_recall_at=multi_recall_at,
        precision_at=precision_at,
        mrr=mrr,
        first_correct_rank=first_correct,
        n_retrieved=len(results),
        latency_ms=latency_ms,
    )


def _evaluate_one(
    *,
    q: _Question,
    config_name: str,
    config: dict[str, Any],
    storage: SqliteStorage,
    embedder: Any,
    reranker: Any,
    chat: Any,
    eval_k: int,
    k_cutoffs: Sequence[int],
) -> _RunRow:
    auto_temporal = bool(config.get("_auto_temporal", False))
    param_overrides = {kk: vv for kk, vv in config.items() if not kk.startswith("_")}
    base_params = RetrieveParams(k=eval_k, **param_overrides)
    memory = Memory(
        storage=storage,
        embedder=embedder,
        chat=chat,
        retrieve_params=base_params,
        reranker=reranker,
    )

    t0 = time.perf_counter()
    try:
        retrieve_kwargs: dict[str, Any] = {"k": eval_k, "reinforce": False}
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
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _LOG.warning("retrieve failed for qid=%s config=%s: %s", q.qid, config_name, exc)
        results = []
    latency_ms = (time.perf_counter() - t0) * 1000.0

    metrics = _compute_metrics(
        results=results,
        memory=memory,
        answer_session_ids=q.answer_session_ids,
        k_cutoffs=k_cutoffs,
        latency_ms=latency_ms,
    )
    return _RunRow(qid=q.qid, qtype=q.qtype, config=config_name, metrics=metrics)


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


def _aggregate(
    rows: list[_RunRow], k_cutoffs: Sequence[int]
) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate metrics per (qtype, config), plus an 'all' qtype bucket.

    Returns dict keyed on (qtype_or_all, config) -> metric_name -> value.
    """
    buckets: dict[tuple[str, str], list[_RunRow]] = {}
    for r in rows:
        buckets.setdefault((r.qtype, r.config), []).append(r)
        buckets.setdefault(("__all__", r.config), []).append(r)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, rs in buckets.items():
        n = len(rs)
        if n == 0:
            continue
        agg: dict[str, float] = {"n": float(n)}
        for k in k_cutoffs:
            agg[f"recall@{k}"] = sum(r.metrics.recall_at[k] for r in rs) / n
            agg[f"hit@{k}"] = sum(r.metrics.hit_at[k] for r in rs) / n
            agg[f"multi_recall@{k}"] = sum(r.metrics.multi_recall_at[k] for r in rs) / n
            agg[f"precision@{k}"] = sum(r.metrics.precision_at[k] for r in rs) / n
        agg["mrr"] = sum(r.metrics.mrr for r in rs) / n
        first_ranks = [
            r.metrics.first_correct_rank
            for r in rs
            if r.metrics.first_correct_rank is not None
        ]
        agg["median_first_correct_rank"] = (
            float(statistics.median(first_ranks)) if first_ranks else float("nan")
        )
        agg["pct_with_a_hit"] = (
            sum(1 for r in rs if r.metrics.first_correct_rank is not None) / n
        )
        agg["latency_ms"] = sum(r.metrics.latency_ms for r in rs) / n
        out[key] = agg
    return out


def _markdown(
    agg: dict[tuple[str, str], dict[str, float]],
    configs: list[str],
    k_cutoffs: Sequence[int],
) -> str:
    qtypes = sorted({qt for (qt, _c) in agg if qt != "__all__"})
    qtypes.append("__all__")

    lines: list[str] = []
    lines.append("# Retrieval evaluation\n")

    primary_k = k_cutoffs[0]
    lines.append(f"## Headline (k={primary_k})\n")
    lines.append(
        f"| qtype | n | config | recall@{primary_k} | hit@{primary_k} | "
        f"multi-recall@{primary_k} | mrr | median 1st rank |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for qt in qtypes:
        for c in configs:
            row = agg.get((qt, c))
            if row is None:
                continue
            n = int(row["n"])
            mrank = row["median_first_correct_rank"]
            mrank_str = "—" if mrank != mrank else f"{mrank:.0f}"
            lines.append(
                f"| {qt} | {n} | {c} | {row[f'recall@{primary_k}']:.3f} | "
                f"{row[f'hit@{primary_k}']:.3f} | "
                f"{row[f'multi_recall@{primary_k}']:.3f} | "
                f"{row['mrr']:.3f} | {mrank_str} |"
            )
        lines.append("")

    lines.append("## Recall@k by config (overall)\n")
    header = "| config | " + " | ".join(f"recall@{k}" for k in k_cutoffs) + " | mrr |"
    sep = "|---|" + "|".join(["---:"] * (len(k_cutoffs) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in configs:
        row = agg.get(("__all__", c))
        if row is None:
            continue
        cells = [f"{row[f'recall@{k}']:.3f}" for k in k_cutoffs]
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {row['mrr']:.3f} |")
    lines.append("")

    lines.append("## Per-qtype recall@10 by config\n")
    header = "| qtype | " + " | ".join(configs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(configs)) + "|"
    lines.append(header)
    lines.append(sep)
    for qt in qtypes[:-1]:  # skip __all__ here
        cells: list[str] = []
        for c in configs:
            row = agg.get((qt, c))
            cells.append(f"{row[f'recall@{primary_k}']:.3f}" if row else "—")
        lines.append(f"| {qt} | " + " | ".join(cells) + " |")
    overall_cells: list[str] = []
    for c in configs:
        row = agg.get(("__all__", c))
        overall_cells.append(f"**{row[f'recall@{primary_k}']:.3f}**" if row else "—")
    lines.append("| **overall** | " + " | ".join(overall_cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval-only evaluator for LongMemEval (no LLM calls)."
    )
    parser.add_argument("--qtype", default=None)
    parser.add_argument("--qid", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--configs",
        default="baseline,conservative,all_aggressive",
        help=f"Comma-separated configs. Available: {','.join(CONFIGS.keys())}",
    )
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--dtype", default="fp32", choices=("auto", "fp16", "fp32"))
    parser.add_argument(
        "--reranker",
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model id, or 'none' to skip rerank (raw dense ordering).",
    )
    parser.add_argument(
        "--eval-k",
        type=int,
        default=50,
        help="Top-k to fetch per question. Metrics computed at k=10/20/50 cutoffs by default.",
    )
    parser.add_argument(
        "--k-cutoffs",
        default="10,20,50",
        help="Comma-separated k cutoffs to compute metrics at (must each be <= --eval-k).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/runs/retrieval_eval.json"),
    )
    parser.add_argument("--log-level", default=os.environ.get("ENGRAM_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in config_names if c not in CONFIGS]
    if unknown:
        print(f"unknown configs: {unknown}", file=sys.stderr)
        print(f"available: {list(CONFIGS.keys())}", file=sys.stderr)
        return 2
    k_cutoffs = tuple(int(s.strip()) for s in args.k_cutoffs.split(",") if s.strip())
    if any(k > args.eval_k for k in k_cutoffs):
        print(
            f"all --k-cutoffs must be <= --eval-k={args.eval_k}",
            file=sys.stderr,
        )
        return 2

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
        "retrieval_eval: %d questions × %d configs = %d runs",
        len(questions),
        len(config_names),
        len(questions) * len(config_names),
    )

    embedder = _build_embedder(args.embed_model, args.embed_device, args.dtype)
    reranker = _build_reranker(args.reranker, args.embed_device, args.dtype)
    # Memory.retrieve doesn't make chat calls for our configs; we still
    # need a ChatProvider to satisfy the Memory constructor. The fake
    # chat is enough -- multi_query / decompose / hyde / temporal are
    # not exercised here.
    chat = FakeChat()

    rows: list[_RunRow] = []
    t_start = time.perf_counter()
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
            for config_name in config_names:
                row = _evaluate_one(
                    q=q,
                    config_name=config_name,
                    config=CONFIGS[config_name],
                    storage=storage,
                    embedder=embedder,
                    reranker=reranker,
                    chat=chat,
                    eval_k=args.eval_k,
                    k_cutoffs=k_cutoffs,
                )
                rows.append(row)
            primary_k = k_cutoffs[0]
            recalls = " ".join(
                f"{c[:8]}={[r for r in rows if r.qid == q.qid and r.config == c][-1].metrics.recall_at[primary_k]:.2f}"
                for c in config_names
            )
            _LOG.info(
                "q %d/%d [%s] qid=%s turns=%d ingest=%.1fs %s",
                q_idx + 1,
                len(questions),
                q.qtype,
                q.qid[:8],
                turns,
                ingest_ms / 1000.0,
                recalls,
            )
        finally:
            storage.close()
    total_s = time.perf_counter() - t_start
    _LOG.info("retrieval_eval: %d rows in %.1fs", len(rows), total_s)

    agg = _aggregate(rows, k_cutoffs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_args": vars(args) | {"output": str(args.output)},
        "k_cutoffs": list(k_cutoffs),
        "configs": config_names,
        "questions_n": len(questions),
        "rows": [
            {
                "qid": r.qid,
                "qtype": r.qtype,
                "config": r.config,
                "recall_at": r.metrics.recall_at,
                "hit_at": r.metrics.hit_at,
                "multi_recall_at": r.metrics.multi_recall_at,
                "precision_at": r.metrics.precision_at,
                "mrr": r.metrics.mrr,
                "first_correct_rank": r.metrics.first_correct_rank,
                "n_retrieved": r.metrics.n_retrieved,
                "latency_ms": r.metrics.latency_ms,
            }
            for r in rows
        ],
        "aggregate": {
            f"{qt}|{c}": metrics for (qt, c), metrics in agg.items()
        },
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _LOG.info("wrote %s", args.output)

    print(_markdown(agg, config_names, k_cutoffs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
