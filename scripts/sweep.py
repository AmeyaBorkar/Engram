"""Hyperparameter sweep over a single retrieval knob.

For each value in the sweep grid, build the haystack once, retrieve
once per (question, value), score against ground truth, aggregate with
bootstrap CIs. Output: Pareto-style table that shows the knob's effect
on overall + per-qtype recall@10 with statistical bounds.

The sweep is single-knob by design: we want to attribute the effect to
ONE component before mixing. Combined sweeps go through the retrieval
evaluator's CONFIGS dict instead.

Predefined sweep grids match `docs/EVAL_PROTOCOL.md`. Override with
`--values v1,v2,...`.

Usage:
  # BM25 weight sweep (5 values, full LongMemEval)
  python scripts/sweep.py --knob bm25_weight --dtype fp32 \
    --output benchmarks/runs/sweep_bm25_weight.json

  # MMR lambda sweep on multi-session (fast)
  python scripts/sweep.py --knob mmr_lambda --qtype multi-session \
    --limit 60 --dtype fp32 \
    --output benchmarks/runs/sweep_mmr_lambda_multi.json
"""

from __future__ import annotations

import argparse
import json
import logging
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
from scripts._stats import (  # noqa: E402
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    format_ci,
    format_p,
    mcnemar,
)

_LOG = logging.getLogger("engram.sweep")


# Sweep grids match the EVAL_PROTOCOL.md standard grids. Each entry is
# (kwarg-name-on-RetrieveParams-or-special, [values...]).
SWEEPS: dict[str, list[Any]] = {
    "bm25_weight": [0.0, 0.5, 1.0, 1.5, 2.0],
    "bm25_k1": [1.0, 1.5, 2.0],
    "bm25_b": [0.5, 0.75, 0.9],
    "mmr_lambda": [0.0, 0.3, 0.5, 0.7, 0.9],
    "mmr_pool_size": [0, 30, 50, 100],
    "recency_lambda": [0.0, 0.05, 0.1, 0.2],
    "recency_decay_days": [30.0, 60.0, 90.0, 180.0],
    "recent_window_k": [0, 5, 10, 20, 50],
    "candidate_multiplier": [3, 5, 10],
    "k": [5, 10, 20],
    "confidence_threshold": [0.5, 0.6, 0.7, 0.8],
    "drill_k": [0, 3, 5, 10],
}


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


def _retrieved_session_ids(memory: Memory, results: Sequence[Any]) -> set[str]:
    """Pull session_id from each result via Event metadata + provenance."""
    out: set[str] = set()
    for r in results:
        try:
            event = memory.storage.get_event(r.item_id)
        except (KeyError, RuntimeError):
            event = None
        if event is not None:
            sid = event.metadata.get("session_id")
            if isinstance(sid, str):
                out.add(sid)
            continue
        try:
            supports = memory.storage.get_supporting_events(r.item_id)
        except (KeyError, RuntimeError):
            supports = []
        for ev in supports:
            sid = ev.metadata.get("session_id")
            if isinstance(sid, str):
                out.add(sid)
    return out


def _evaluate_one(
    *,
    q: _Question,
    value: Any,
    knob: str,
    storage: SqliteStorage,
    embedder: Any,
    reranker: Any,
    chat: Any,
    eval_k: int,
) -> dict[str, Any]:
    """Run one (question, value) and return per-question metrics."""
    # Build RetrieveParams with the knob set.
    rp_kwargs: dict[str, Any] = {"k": eval_k}
    if knob == "k":
        rp_kwargs["k"] = int(value)
    else:
        rp_kwargs[knob] = value
    base_params = RetrieveParams(**rp_kwargs)
    memory = Memory(
        storage=storage,
        embedder=embedder,
        chat=chat,
        retrieve_params=base_params,
        reranker=reranker,
    )
    retrieve_kwargs: dict[str, Any] = {"k": rp_kwargs["k"], "reinforce": False}
    if knob == "recency_lambda" and value and value > 0:
        question_dt = _parse_haystack_date(q.question_date)
        if question_dt is not None:
            retrieve_kwargs["as_of"] = question_dt
    t0 = time.perf_counter()
    try:
        results = memory.retrieve(q.question, **retrieve_kwargs)
    except Exception as exc:
        _LOG.warning(
            "retrieve failed for qid=%s knob=%s val=%s: %s", q.qid, knob, value, exc
        )
        results = []
    latency_ms = (time.perf_counter() - t0) * 1000.0
    retrieved = _retrieved_session_ids(memory, results)
    answer = set(q.answer_session_ids)
    if not answer:
        recall = 0.0
        hit = 0.0
        multi = 0.0
    else:
        intersect = retrieved & answer
        recall = len(intersect) / len(answer)
        hit = 1.0 if intersect else 0.0
        multi = 1.0 if intersect == answer else 0.0
    # First-correct rank @ 1-indexed
    first_rank: int | None = None
    for idx, r in enumerate(results, start=1):
        _, sid = _result_to_session(memory, r)
        if sid in answer:
            first_rank = idx
            break
    mrr = (1.0 / first_rank) if first_rank else 0.0
    return {
        "qid": q.qid,
        "qtype": q.qtype,
        "value": value,
        "recall_at_k": recall,
        "hit_at_k": hit,
        "multi_recall_at_k": multi,
        "mrr": mrr,
        "first_correct_rank": first_rank,
        "n_retrieved": len(results),
        "latency_ms": latency_ms,
    }


def _result_to_session(memory: Memory, result: Any) -> tuple[str | None, str | None]:
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


def _markdown_sweep_report(
    knob: str,
    values: list[Any],
    per_value: dict[Any, list[dict[str, Any]]],
    baseline_value: Any,
) -> str:
    """Render the headline sweep report with bootstrap CIs + McNemar."""
    lines: list[str] = []
    lines.append(f"# Sweep: `{knob}` over {values}\n")

    # Per-value aggregates with bootstrap CIs (overall, then per-qtype).
    lines.append("## Overall (bootstrap 95% CI)\n")
    lines.append(
        "| value | n | recall@k | hit@k | multi-recall@k | mrr |"
    )
    lines.append("|---|---:|---|---|---|---|")
    baseline_rows = per_value[baseline_value]
    baseline_recall = [r["recall_at_k"] for r in baseline_rows]
    baseline_hit = [int(r["hit_at_k"]) for r in baseline_rows]
    baseline_multi = [r["multi_recall_at_k"] for r in baseline_rows]
    baseline_mrr = [r["mrr"] for r in baseline_rows]
    for v in values:
        rows = per_value.get(v, [])
        if not rows:
            continue
        n = len(rows)
        recalls = [r["recall_at_k"] for r in rows]
        hits = [int(r["hit_at_k"]) for r in rows]
        multis = [r["multi_recall_at_k"] for r in rows]
        mrrs = [r["mrr"] for r in rows]
        r_ci = bootstrap_mean_ci(recalls)
        h_ci = bootstrap_mean_ci(hits)
        m_ci = bootstrap_mean_ci(multis)
        mrr_ci = bootstrap_mean_ci(mrrs)
        lines.append(
            f"| {v} | {n} | {format_ci(r_ci.mean, r_ci.ci_low, r_ci.ci_high)} | "
            f"{format_ci(h_ci.mean, h_ci.ci_low, h_ci.ci_high)} | "
            f"{format_ci(m_ci.mean, m_ci.ci_low, m_ci.ci_high)} | "
            f"{format_ci(mrr_ci.mean, mrr_ci.ci_low, mrr_ci.ci_high)} |"
        )

    # Paired lift vs baseline.
    lines.append(f"\n## Lift vs baseline (`{knob}={baseline_value}`)\n")
    lines.append(
        "| value | Δ recall (paired CI) | Δ multi-recall (paired CI) | Δ mrr (paired CI) | McNemar hit@k |"
    )
    lines.append("|---|---|---|---|---|")
    for v in values:
        if v == baseline_value:
            lines.append(f"| {v} | (baseline) | — | — | — |")
            continue
        rows = per_value.get(v, [])
        if not rows:
            continue
        cand_recall = [r["recall_at_k"] for r in rows]
        cand_hit = [int(r["hit_at_k"]) for r in rows]
        cand_multi = [r["multi_recall_at_k"] for r in rows]
        cand_mrr = [r["mrr"] for r in rows]
        # Align rows by qid (paired diff).
        bid_to_row: dict[str, dict[str, Any]] = {r["qid"]: r for r in baseline_rows}
        a_recall: list[float] = []
        b_recall: list[float] = []
        a_hit: list[int] = []
        b_hit: list[int] = []
        a_multi: list[float] = []
        b_multi: list[float] = []
        a_mrr: list[float] = []
        b_mrr: list[float] = []
        for r in rows:
            b_row = bid_to_row.get(r["qid"])
            if b_row is None:
                continue
            a_recall.append(b_row["recall_at_k"])
            b_recall.append(r["recall_at_k"])
            a_hit.append(int(b_row["hit_at_k"]))
            b_hit.append(int(r["hit_at_k"]))
            a_multi.append(b_row["multi_recall_at_k"])
            b_multi.append(r["multi_recall_at_k"])
            a_mrr.append(b_row["mrr"])
            b_mrr.append(r["mrr"])
        d_recall = bootstrap_paired_diff_ci(a_recall, b_recall)
        d_multi = bootstrap_paired_diff_ci(a_multi, b_multi)
        d_mrr = bootstrap_paired_diff_ci(a_mrr, b_mrr)
        mcn = mcnemar(a_hit, b_hit)
        sig_marker = "**" if d_recall.excludes_zero else ""
        lines.append(
            f"| {v} | {sig_marker}{d_recall.diff:+.3f} [{d_recall.ci_low:+.3f}, "
            f"{d_recall.ci_high:+.3f}]{sig_marker} | "
            f"{d_multi.diff:+.3f} [{d_multi.ci_low:+.3f}, {d_multi.ci_high:+.3f}] | "
            f"{d_mrr.diff:+.3f} [{d_mrr.ci_low:+.3f}, {d_mrr.ci_high:+.3f}] | "
            f"{format_p(mcn.p_value)} (only_A={mcn.n_passes_only_a}, only_B={mcn.n_passes_only_b}) |"
        )

    # Per-qtype overall recall.
    qtypes = sorted({r["qtype"] for rs in per_value.values() for r in rs})
    lines.append("\n## Per-qtype recall@k (mean)\n")
    header = "| qtype | n | " + " | ".join(str(v) for v in values) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "|".join(["---:"] * len(values)) + "|")
    for qt in qtypes:
        rep_n = sum(1 for r in per_value[values[0]] if r["qtype"] == qt)
        cells: list[str] = []
        for v in values:
            rs = [r for r in per_value[v] if r["qtype"] == qt]
            if not rs:
                cells.append("—")
            else:
                m = sum(r["recall_at_k"] for r in rs) / len(rs)
                cells.append(f"{m:.3f}")
        lines.append(f"| {qt} | {rep_n} | " + " | ".join(cells) + " |")

    # Failure mode: which qids broke (passed at baseline, fail at best).
    lines.append("\n## Failure mode vs baseline (hit@k flipped to FAIL)\n")
    for v in values:
        if v == baseline_value:
            continue
        rows = per_value.get(v, [])
        if not rows:
            continue
        bid_to = {r["qid"]: r for r in baseline_rows}
        broken: list[tuple[str, str]] = []
        for r in rows:
            b = bid_to.get(r["qid"])
            if b is None:
                continue
            if int(b["hit_at_k"]) == 1 and int(r["hit_at_k"]) == 0:
                broken.append((r["qid"], r["qtype"]))
        if broken:
            lines.append(f"\n### `{knob}={v}` broke {len(broken)} questions:")
            for qid, qt in broken[:30]:  # cap list at 30
                lines.append(f"  - `{qid[:12]}` ({qt})")
            if len(broken) > 30:
                lines.append(f"  - ... and {len(broken) - 30} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep over a single retrieval knob."
    )
    parser.add_argument(
        "--knob",
        required=True,
        choices=sorted(SWEEPS.keys()),
        help="Knob to sweep.",
    )
    parser.add_argument(
        "--values",
        default=None,
        help="Comma-separated value override (otherwise uses the standard grid).",
    )
    parser.add_argument(
        "--baseline-value",
        default=None,
        help="Value to use as baseline for lift comparisons. Default: first in grid.",
    )
    parser.add_argument("--qtype", default=None)
    parser.add_argument("--qid", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--dtype", default="fp32", choices=("auto", "fp16", "fp32"))
    parser.add_argument(
        "--reranker",
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model id, or 'none' to skip rerank.",
    )
    parser.add_argument("--eval-k", type=int, default=10, help="Top-k for the eval.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/runs/sweep.json"),
    )
    parser.add_argument("--log-level", default=os.environ.get("ENGRAM_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Coerce values to the right type. The standard grids are typed
    # correctly; CLI overrides come in as strings.
    if args.values is not None:
        raw_values = [s.strip() for s in args.values.split(",") if s.strip()]
        # Coerce to int/float based on the knob's standard grid.
        standard = SWEEPS[args.knob]
        sample = standard[0]
        if isinstance(sample, int):
            values: list[Any] = [int(v) for v in raw_values]
        elif isinstance(sample, float):
            values = [float(v) for v in raw_values]
        else:
            values = list(raw_values)
    else:
        values = list(SWEEPS[args.knob])

    baseline_value: Any
    if args.baseline_value is not None:
        sample = values[0]
        if isinstance(sample, int):
            baseline_value = int(args.baseline_value)
        elif isinstance(sample, float):
            baseline_value = float(args.baseline_value)
        else:
            baseline_value = args.baseline_value
    else:
        baseline_value = values[0]
    if baseline_value not in values:
        values.insert(0, baseline_value)

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
        "sweep: knob=%s values=%s questions=%d total_runs=%d",
        args.knob,
        values,
        len(questions),
        len(questions) * len(values),
    )

    # Build providers once.
    embedder = _build_embedder(args.embed_model, args.embed_device, args.dtype)
    reranker = _build_reranker(args.reranker, args.embed_device, args.dtype)
    chat = FakeChat()

    per_value: dict[Any, list[dict[str, Any]]] = {v: [] for v in values}
    t_start = time.perf_counter()
    for q_idx, q in enumerate(questions):
        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            ingest_memory = Memory(
                storage=storage, embedder=embedder, chat=chat, reranker=reranker
            )
            t_ingest = time.perf_counter()
            turns = _ingest_haystack(ingest_memory, q)
            ingest_ms = (time.perf_counter() - t_ingest) * 1000.0
            for v in values:
                row = _evaluate_one(
                    q=q,
                    value=v,
                    knob=args.knob,
                    storage=storage,
                    embedder=embedder,
                    reranker=reranker,
                    chat=chat,
                    eval_k=args.eval_k,
                )
                per_value[v].append(row)
            if (q_idx + 1) % 10 == 0 or q_idx == len(questions) - 1:
                _LOG.info(
                    "q %d/%d [%s] qid=%s turns=%d ingest=%.1fs",
                    q_idx + 1,
                    len(questions),
                    q.qtype,
                    q.qid[:8],
                    turns,
                    ingest_ms / 1000.0,
                )
        finally:
            storage.close()
    _LOG.info("sweep done in %.1fs", time.perf_counter() - t_start)

    # Persist.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_args": vars(args) | {"output": str(args.output)},
        "knob": args.knob,
        "values": values,
        "baseline_value": baseline_value,
        "per_value": {str(v): rows for v, rows in per_value.items()},
        "questions_n": len(questions),
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    _LOG.info("wrote %s", args.output)

    print(_markdown_sweep_report(args.knob, values, per_value, baseline_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
