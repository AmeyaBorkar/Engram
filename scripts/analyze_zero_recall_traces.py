"""Analyze the zero-event-recall trace dumps for failure-mode patterns.

After `scripts/batch_trace.py` produces per-qid traces for every
question where event_recall@10 = 0 (despite has_answer=True turns
existing), this script scans the trace files and classifies the
failure mechanism per question:

  - "session_miss": dense retrieval missed every answer session
  - "wrong_turn_in_session": session hit but the actual gold turn(s)
        rank below the cutoff (k=10) — proxy-lie cases
  - "gold_at_deep_dense_rank": where in the dense top-50 the gold
        sessions first appear; if it's ≥ 30 the rerank is unlikely
        to pull them up
  - "competing_session": the same off-topic session dominates top-10
        (sign of one strong distractor)

Outputs a markdown summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# M-170: prefer the structured `<trace-meta>{...}</trace-meta>` tag
# the trace writer emits at the top of every dump. The regexes
# below are kept as fallbacks for legacy traces that pre-date the
# tag, but anything fresh should parse cleanly via the JSON path.
_META_RE = re.compile(r"<trace-meta>(.+?)</trace-meta>", re.DOTALL)
_HEADER_RE = re.compile(r"^qid=(\S+)\s+\[(\S+)\]\s*$")
_GOLD_RE = re.compile(r"^GOLD:\s*(.+)$")
_QUESTION_RE = re.compile(r"^Q:\s*(.+)$")
_ANSWER_SESSIONS_RE = re.compile(r"^answer_session_ids\s*\((\d+)\):\s*(.+)$")
_DENSE_ROW_RE = re.compile(r"^\s*(\d+)\s*\|\s*\[(GOLD|\.+)\]\s*(\S+)\s*\|\s*([0-9.]+)\s*\|")
# Final top-k rows have an extra `level` column between session and score:
#   "   1 | [....] d8_2  | event      |  0.661 | content"
_FINAL_ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*\[(GOLD|\.+)\]\s*(\S+)\s*\|\s*\S+\s*\|\s*([0-9.]+)\s*\|"
)
_FINAL_RECALL_RE = re.compile(r"recall@\d+=([0-9.]+)\s+\((\d+)/(\d+)")


def _parse_trace(text: str) -> dict:
    out: dict = {
        "qid": None,
        "qtype": None,
        "question": None,
        "gold": None,
        "answer_sessions": [],
        "dense_gold_ranks": [],
        "dense_top1_session": None,
        "dense_top10_top_session_count": 0,
        "final_session_recall": None,
        "final_n_gold_in_topk": 0,
    }
    # M-170: try the structured `<trace-meta>` tag first. If it's
    # there, we get everything in one JSON parse and skip the
    # regex-fragile header scan.
    tag_match = _META_RE.search(text)
    if tag_match:
        try:
            meta = json.loads(tag_match.group(1))
        except json.JSONDecodeError:
            meta = None
        if isinstance(meta, dict):
            out["qid"] = meta.get("qid") or out["qid"]
            out["qtype"] = meta.get("qtype") or out["qtype"]
            out["question"] = meta.get("question") or out["question"]
            out["gold"] = meta.get("gold") or out["gold"]
            answer_ids = meta.get("answer_session_ids")
            if isinstance(answer_ids, list):
                out["answer_sessions"] = list(answer_ids)
    lines = text.splitlines()
    in_dense_section = False
    dense_rows_seen = 0
    top10_session_count: Counter[str] = Counter()
    for line in lines:
        # The legacy regex path remains as a fallback so older
        # traces (pre-M-170) still parse cleanly.
        m = _HEADER_RE.match(line)
        if m and out["qid"] is None:
            out["qid"] = m.group(1)
            out["qtype"] = m.group(2)
            continue
        m = _QUESTION_RE.match(line)
        if m and out["question"] is None:
            out["question"] = m.group(1)
            continue
        m = _GOLD_RE.match(line)
        if m and out["gold"] is None:
            out["gold"] = m.group(1)
            continue
        m = _ANSWER_SESSIONS_RE.match(line)
        if m and not out["answer_sessions"]:
            out["answer_sessions"] = [s.strip() for s in m.group(2).split(",")]
            continue
        if line.startswith("--- Dense top-"):
            in_dense_section = True
            dense_rows_seen = 0
            continue
        if line.startswith("--- ") and not line.startswith("--- Dense top-"):
            in_dense_section = False
            continue
        if in_dense_section:
            rm = _DENSE_ROW_RE.match(line)
            if rm:
                dense_rows_seen += 1
                rank = int(rm.group(1))
                is_gold = rm.group(2) == "GOLD"
                session = rm.group(3)
                if dense_rows_seen == 1:
                    out["dense_top1_session"] = session
                if is_gold:
                    out["dense_gold_ranks"].append(rank)
        if "recall@10=" in line and (
            "Config: baseline" in (lines[lines.index(line) - 1] if lines.index(line) > 0 else "")
            or out["final_session_recall"] is None
        ):
            # Take first recall@10 line we find under FINAL TOP-K
            pass
    # Second pass for the FINAL recall + top-10 session distribution
    final_section = False
    rerank_rows_seen = 0
    for line in lines:
        if line.startswith("FINAL TOP-K PER CONFIG"):
            final_section = True
            continue
        if not final_section:
            continue
        m = _FINAL_RECALL_RE.search(line)
        if m and out["final_session_recall"] is None:
            out["final_session_recall"] = float(m.group(1))
            continue
        rm = _FINAL_ROW_RE.match(line)
        if rm and rerank_rows_seen < 10:
            rerank_rows_seen += 1
            sess = rm.group(3)
            top10_session_count[sess] += 1
    out["dense_top10_top_session_count"] = (
        top10_session_count.most_common(1)[0][1] if top10_session_count else 0
    )
    out["dense_top10_top_session"] = (
        top10_session_count.most_common(1)[0][0] if top10_session_count else None
    )
    return out


def _classify(parsed: dict) -> str:
    """Bucket the failure mechanism per question."""
    if not parsed["dense_gold_ranks"]:
        return "session_miss"  # no gold appears anywhere in top-50 dense
    min_rank = min(parsed["dense_gold_ranks"])
    if parsed["dense_top10_top_session_count"] >= 5:
        return "competing_session"  # one off-topic session dominates top-10
    if min_rank > 20:
        return "gold_at_deep_dense_rank"
    if min_rank > 10:
        return "wrong_turn_in_session"  # gold reached top-50 but not top-10
    return "rerank_pushed_gold_out"  # gold was top-10 in dense but rerank dropped it


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("benchmarks/runs/traces_zero_recall"),
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for trace_path in sorted(args.trace_dir.glob("trace_*.txt")):
        text = trace_path.read_text(encoding="utf-8")
        parsed = _parse_trace(text)
        parsed["failure_class"] = _classify(parsed)
        parsed["min_gold_rank"] = (
            min(parsed["dense_gold_ranks"]) if parsed["dense_gold_ranks"] else None
        )
        parsed["n_gold_in_top50"] = len(parsed["dense_gold_ranks"])
        rows.append(parsed)

    if not rows:
        print(f"No trace files in {args.trace_dir}", file=__import__("sys").stderr)
        return 2

    # Markdown table
    print(f"# Zero-event-recall failure analysis ({len(rows)} questions)\n")
    print(
        f"| qid | qtype | failure class | min gold rank (top-50) | n gold in top-50 | final sess R@10 | top-10 dominated by |"
    )
    print(f"|---|---|---|---:|---:|---:|---|")
    for r in rows:
        mr = r["min_gold_rank"]
        mr_disp = str(mr) if mr is not None else "—"
        dom = (
            f"`{r['dense_top10_top_session']}` ({r['dense_top10_top_session_count']}/10)"
            if r["dense_top10_top_session_count"] >= 3
            else "—"
        )
        sr = r["final_session_recall"]
        sr_disp = f"{sr:.2f}" if sr is not None else "?"
        print(
            f"| `{r['qid'][:12]}` | {r['qtype']} | **{r['failure_class']}** | {mr_disp} | {r['n_gold_in_top50']} | {sr_disp} | {dom} |"
        )

    print(f"\n## Failure class distribution\n")
    by_class: Counter[str] = Counter(r["failure_class"] for r in rows)
    print("| class | count |")
    print("|---|---:|")
    for c, n in by_class.most_common():
        print(f"| {c} | {n} |")

    print(f"\n## Failure class × qtype\n")
    matrix: dict[tuple[str, str], int] = defaultdict(int)
    qtypes = sorted({r["qtype"] for r in rows})
    classes = sorted({r["failure_class"] for r in rows})
    for r in rows:
        matrix[(r["failure_class"], r["qtype"])] += 1
    print("| failure class | " + " | ".join(qtypes) + " |")
    print("|---|" + "|".join(["---:"] * len(qtypes)) + "|")
    for c in classes:
        cells = [str(matrix.get((c, qt), 0)) for qt in qtypes]
        print(f"| {c} | " + " | ".join(cells) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
