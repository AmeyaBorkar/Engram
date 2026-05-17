"""Deep retrieval inspector with event-level ground truth.

LongMemEval-S includes a per-turn `has_answer` label (True/False/None)
we had been silently discarding. This script:

  1. Dumps the question + gold answer + every answer-session turn with
     its has_answer flag, so a human can read what the answer
     *actually* depends on.
  2. Runs retrieve under a chosen config.
  3. Annotates every retrieved item with:
        [GOLD-EVT]  — this turn is has_answer=True (the actual answer)
        [SAME-SESS] — has_answer=False but in an answer session
        [....]      — not in any answer session at all
  4. Computes both event-level and session-level recall@k so you can
     see exactly how much our session-level proxy is overestimating
     retrieval quality.

When the event-level recall is much lower than the session-level
recall, the LLM is being shown the right session but the wrong
*turn* — and the rest of the gap from retrieval to end-to-end
accuracy lives there.

Usage:
  # Inspect one specific question
  python scripts/inspect_retrieval.py --qid 0a995998 --dtype fp32 --embed-device cuda

  # Sample 5 multi-session questions
  python scripts/inspect_retrieval.py --qtype multi-session --limit 5 --dtype fp32 --embed-device cuda

  # Compute event-level recall stats across many questions (no per-question dump)
  python scripts/inspect_retrieval.py --qtype multi-session --limit 30 --stats-only --dtype fp32 --embed-device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# M-122: shared helpers in scripts/_common.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    build_embedder as _build_embedder_common,
    build_reranker as _build_reranker_common,
    ensure_repo_on_path,
    load_env_file,
)

ensure_repo_on_path()
load_env_file()

from engram import Memory, SqliteStorage  # noqa: E402
from engram.providers._fake import FakeChat  # noqa: E402
from engram.retrieve._params import RetrieveParams  # noqa: E402

from benchmarks.suites.longmemeval import (  # noqa: E402
    DATASET_ROOT,
    DEFAULT_FILENAME,
    _Question,
    _ingest_haystack,
    _load_dataset,
    _parse_haystack_date,
)
from scripts.ablate_longmemeval import CONFIGS  # noqa: E402

_LOG = logging.getLogger("engram.inspect")


# M-122: thin shims around _common builders.


def _build_embedder(model: str, device: str | None, dtype: str) -> Any:
    return _build_embedder_common(model, device, dtype)


def _build_reranker(model: str | None, device: str | None, dtype: str) -> Any:
    return _build_reranker_common(model, device, dtype)


def _preview(content: str, width: int) -> str:
    flat = content.replace("\n", " ").replace("\r", " ")
    if len(flat) <= width:
        return flat
    return flat[: width - 1] + "…"


def _annotate_event(
    sid: str | None,
    has_answer: bool | None,
    answer_sessions: set[str],
) -> str:
    """Build the visual marker for one event in the trace.

    Three categories:
      [GOLD-EVT]  — the turn IS the answer (has_answer=True)
      [SAME-SESS] — in an answer session but not the answer turn
                    (has_answer=False, OR has_answer=None but session matches)
      [........] — not in any answer session
    """
    in_answer_session = sid in answer_sessions
    if has_answer is True:
        return "[GOLD-EVT]"
    if in_answer_session:
        return "[SAME-SESS]"
    return "[........]"


def _dump_answer_sessions(q: _Question, width: int) -> list[str]:
    """Print every turn of every answer session with has_answer flags."""
    lines: list[str] = []
    answer_set = set(q.answer_session_ids)
    sessions_in_order: list[tuple[int, str]] = [
        (idx, sid) for idx, sid in enumerate(q.haystack_session_ids) if sid in answer_set
    ]
    lines.append(
        f"\n## Answer sessions ({len(sessions_in_order)} of {len(q.haystack_session_ids)} haystack sessions)\n"
    )
    for idx, sid in sessions_in_order:
        sess = q.haystack_sessions[idx]
        date = q.haystack_dates[idx] if idx < len(q.haystack_dates) else "?"
        n_gold = sum(1 for t in sess if t.get("has_answer") is True)
        n_false = sum(1 for t in sess if t.get("has_answer") is False)
        lines.append(f"\n### session `{sid}` (haystack idx {idx}, date {date})")
        lines.append(
            f"   {len(sess)} turns; {n_gold} has_answer=True; {n_false} has_answer=False\n"
        )
        for j, turn in enumerate(sess):
            ha = turn.get("has_answer")
            if ha is True:
                marker = "✦GOLD✦"
            elif ha is False:
                marker = "  -   "
            else:
                marker = " (na) "
            role = turn.get("role", "?").ljust(9)
            content = _preview(turn.get("content", ""), width)
            lines.append(f"  {marker} [{j:2d}] [{role}] {content}")
    return lines


def _result_to_event_meta(memory: Memory, result: Any) -> dict[str, Any]:
    """Pull session_id + has_answer + content + level for one retrieved item.

    M-171: for memory-item (consolidated abstraction) results, walk
    ALL supporting events and union their session_ids / has_answer
    flags rather than returning the FIRST hit. Multi-session
    abstractions used to be tagged by whichever event happened to
    sort first under whatever ordering ``get_supporting_events``
    returned, which biased the "session-level recall" computation
    toward whichever single source we sampled. The unioned form
    accurately reflects "this abstraction is from sessions {X, Y}".
    """
    try:
        event = memory.storage.get_event(result.item_id)
    except (KeyError, RuntimeError):
        event = None
    if event is not None:
        return {
            "kind": "event",
            "session_id": event.metadata.get("session_id"),
            "session_ids": [event.metadata.get("session_id")]
            if event.metadata.get("session_id")
            else [],
            "has_answer": event.metadata.get("has_answer"),
            "content": event.content,
        }
    # Memory item: walk ALL provenance, not just the first hit.
    try:
        supports = memory.storage.get_supporting_events(result.item_id)
    except (KeyError, RuntimeError):
        supports = []
    sids: list[str] = []
    has_answer_any = False
    has_answer_seen = False
    for ev in supports:
        sid = ev.metadata.get("session_id")
        if isinstance(sid, str):
            sids.append(sid)
        ha = ev.metadata.get("has_answer")
        if ha is not None:
            has_answer_seen = True
            if ha is True:
                has_answer_any = True
    if sids:
        # Surface the FIRST hit on the legacy `session_id` key for
        # backwards compat with the visual marker code; the unioned
        # list of all hits is on `session_ids`.
        return {
            "kind": "memory_item",
            "session_id": sids[0],
            "session_ids": sids,
            "has_answer": has_answer_any if has_answer_seen else None,
            "content": result.content,
        }
    return {
        "kind": "unknown",
        "session_id": None,
        "session_ids": [],
        "has_answer": None,
        "content": result.content,
    }


def _inspect_question(
    *,
    q: _Question,
    storage: SqliteStorage,
    memory: Memory,
    answer_session_ids: set[str],
    config_name: str,
    config: dict[str, Any],
    k: int,
    width: int,
    stats_only: bool,
) -> dict[str, Any]:
    """Run retrieve, compute both recall flavors, and (unless stats_only)
    print the per-question dump.
    """
    # M-172: count gold events post-filter. The ingest path
    # (`_ingest_haystack`) silently drops turns with empty content,
    # but pre-fix the counter scanned the raw haystack and reported
    # a denominator that included those filtered turns. The event-
    # level recall thus understated the actual reach of the retrieve
    # path. Match the ingest filter exactly so the metric is fair.
    gold_event_count = 0
    for idx, sid in enumerate(q.haystack_session_ids):
        if sid not in answer_session_ids:
            continue
        for t in q.haystack_sessions[idx]:
            content = t.get("content")
            if not content:
                # Same filter `_ingest_haystack` applies: empty
                # content -> not ingested -> not eligible to count
                # as a gold event for recall purposes.
                continue
            if t.get("has_answer") is True:
                gold_event_count += 1

    retrieve_kwargs: dict[str, Any] = {"k": k, "reinforce": False}
    # H-86: always pass as_of when the question has a parseable date.
    question_dt = _parse_haystack_date(q.question_date)
    if question_dt is not None:
        retrieve_kwargs["as_of"] = question_dt
    results = memory.retrieve(q.question, **retrieve_kwargs)

    annotated: list[dict[str, Any]] = []
    for r in results:
        meta = _result_to_event_meta(memory, r)
        annotated.append(
            {
                "rank": len(annotated) + 1,
                "score": float(r.score),
                "session_id": meta["session_id"],
                # M-171: keep the unioned list of all supporting session_ids
                # so multi-session abstractions can be scored fairly.
                "session_ids": meta.get("session_ids", []),
                "has_answer": meta["has_answer"],
                "content": meta["content"],
            }
        )

    # Metrics
    # M-171: build the retrieved-sessions set from the union of every
    # supporting event's session_id, not just the first one.
    retrieved_sessions: set[str] = set()
    for a in annotated:
        for sid in a.get("session_ids") or []:
            if isinstance(sid, str):
                retrieved_sessions.add(sid)
        if a["session_id"] is not None and a["session_id"] not in retrieved_sessions:
            retrieved_sessions.add(a["session_id"])
    session_hit = bool(retrieved_sessions & answer_session_ids)
    session_recall = (
        len(retrieved_sessions & answer_session_ids) / len(answer_session_ids)
        if answer_session_ids
        else 0.0
    )
    n_gold_retrieved = sum(1 for a in annotated if a["has_answer"] is True)
    event_hit = n_gold_retrieved > 0
    event_recall = (n_gold_retrieved / gold_event_count) if gold_event_count > 0 else 0.0

    if not stats_only:
        print("=" * 88)
        print(f"qid={q.qid}  [{q.qtype}]  config={config_name}  k={k}")
        print(f"question: {q.question}")
        print(f"gold answer: {q.gold}")
        print(f"answer sessions ({len(answer_session_ids)}): {sorted(answer_session_ids)}")
        print(
            f"haystack: {len(q.haystack_session_ids)} sessions, "
            f"{sum(len(s) for s in q.haystack_sessions)} turns, "
            f"{gold_event_count} has_answer=True turns"
        )
        print("=" * 88)
        for line in _dump_answer_sessions(q, width):
            print(line)
        print(f"\n## Retrieved top-{k} (config={config_name})")
        print(f"{'rank':>4} | {'marker':<11} | {'session':<10} | {'score':>7} | content")
        print(f"{'-' * 4}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 7}-+-" + "-" * width)
        for a in annotated:
            marker = _annotate_event(a["session_id"], a["has_answer"], answer_session_ids)
            sid_disp = (a["session_id"] or "?")[-10:]
            print(
                f"{a['rank']:>4} | {marker:<11} | {sid_disp:<10} | {a['score']:>7.3f} | "
                f"{_preview(a['content'], width)}"
            )
        print("\n## Recall comparison")
        print(
            f"  session-level recall@{k}: {session_recall:.3f} "
            f"(found {len(retrieved_sessions & answer_session_ids)}/{len(answer_session_ids)} answer sessions)"
        )
        print(
            f"  event-level recall@{k}:   {event_recall:.3f} "
            f"(found {n_gold_retrieved}/{gold_event_count} has_answer=True events)"
        )
        if event_recall < session_recall - 0.05 and event_recall < 1.0:
            print(f"  ⚠ EVENT-LEVEL GAP: we hit the right session but missed the gold turn")

    return {
        "qid": q.qid,
        "qtype": q.qtype,
        "config": config_name,
        "session_recall": session_recall,
        "session_hit": float(session_hit),
        "event_recall": event_recall,
        "event_hit": float(event_hit),
        "n_gold_events": gold_event_count,
        "n_gold_retrieved": n_gold_retrieved,
    }


def _filter_questions(
    questions: list[_Question],
    *,
    qid: str | None,
    qtype: str | None,
    limit: int | None,
) -> list[_Question]:
    if qid:
        return [q for q in questions if q.qid.startswith(qid)][:1]
    if qtype:
        questions = [q for q in questions if q.qtype == qtype]
    if limit is not None:
        questions = questions[:limit]
    return questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval with event-level (has_answer) ground truth."
    )
    parser.add_argument("--qid", default=None)
    parser.add_argument("--qtype", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap on questions after filtering. None (default) = all "
            "matching. The previous default of 1 made --stats-only "
            "useless by accident."
        ),
    )
    parser.add_argument("--config", default="baseline", choices=sorted(CONFIGS.keys()))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--width", type=int, default=100, help="Content preview width.")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--dtype", default="fp32", choices=("auto", "fp16", "fp32"))
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Don't print per-question dumps; just aggregate metrics.",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON with all rows.")
    parser.add_argument("--log-level", default=os.environ.get("ENGRAM_LOG_LEVEL", "WARNING"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    dataset_path = DATASET_ROOT / DEFAULT_FILENAME
    all_questions = _load_dataset(dataset_path)
    if not all_questions:
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    questions = _filter_questions(all_questions, qid=args.qid, qtype=args.qtype, limit=args.limit)
    if not questions:
        print("no questions match the filter", file=sys.stderr)
        return 2

    embedder = _build_embedder(args.embed_model, args.embed_device, args.dtype)
    reranker = _build_reranker(args.reranker, args.embed_device, args.dtype)
    chat = FakeChat()

    config = CONFIGS[args.config]
    param_overrides = {kk: vv for kk, vv in config.items() if not kk.startswith("_")}
    base_params = RetrieveParams(k=args.k, **param_overrides)

    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for q_idx, q in enumerate(questions):
        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            memory = Memory(
                storage=storage,
                embedder=embedder,
                chat=chat,
                retrieve_params=base_params,
                reranker=reranker,
            )
            _ingest_haystack(memory, q)
            row = _inspect_question(
                q=q,
                storage=storage,
                memory=memory,
                answer_session_ids=set(q.answer_session_ids),
                config_name=args.config,
                config=config,
                k=args.k,
                width=args.width,
                stats_only=args.stats_only,
            )
            rows.append(row)
            if args.stats_only and (q_idx + 1) % 10 == 0:
                _LOG.warning("  q %d/%d done", q_idx + 1, len(questions))
        finally:
            storage.close()
    elapsed = time.perf_counter() - t_start

    # Aggregate
    if rows:
        n = len(rows)
        print("\n" + "=" * 88)
        print(f"AGGREGATE over {n} questions (config={args.config}, k={args.k}, {elapsed:.1f}s)")
        print("=" * 88)
        per_qtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            per_qtype[r["qtype"]].append(r)
        print(
            f"{'qtype':<28} | {'n':>3} | {'sess hit':>9} | {'sess R@k':>9} | "
            f"{'evt hit':>8} | {'evt R@k':>8} | {'gap':>7}"
        )
        print(f"{'-' * 28}-+-{'-' * 3}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 7}")
        for qt, rs in sorted(per_qtype.items()):
            n_qt = len(rs)
            sh = sum(r["session_hit"] for r in rs) / n_qt
            sr = sum(r["session_recall"] for r in rs) / n_qt
            eh = sum(r["event_hit"] for r in rs) / n_qt
            er = sum(r["event_recall"] for r in rs) / n_qt
            gap = sr - er
            print(
                f"{qt:<28} | {n_qt:>3} | {sh:>9.3f} | {sr:>9.3f} | "
                f"{eh:>8.3f} | {er:>8.3f} | {gap:>+7.3f}"
            )
        print(f"{'-' * 28}-+-{'-' * 3}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 7}")
        sh = sum(r["session_hit"] for r in rows) / n
        sr = sum(r["session_recall"] for r in rows) / n
        eh = sum(r["event_hit"] for r in rows) / n
        er = sum(r["event_recall"] for r in rows) / n
        gap = sr - er
        print(
            f"{'(overall)':<28} | {n:>3} | {sh:>9.3f} | {sr:>9.3f} | "
            f"{eh:>8.3f} | {er:>8.3f} | {gap:>+7.3f}"
        )
        print("\nKey:")
        print("  sess hit  = at least one event from any answer session in top-k")
        print("  sess R@k  = fraction of answer sessions retrieved (overestimates correctness)")
        print("  evt hit   = at least one has_answer=True event in top-k")
        print("  evt R@k   = fraction of has_answer=True events retrieved (true recall)")
        print("  gap       = (sess R@k - evt R@k); high gap == retrieval looks fine but isn't")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n[wrote {args.output}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
