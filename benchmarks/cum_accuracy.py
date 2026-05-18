"""Print per-question cumulative accuracy from a longmemeval manifest.

For each question in the manifest's `per_question` list, prints a row
with the running cumulative correct count and cumulative accuracy.
Useful for spotting WHERE accuracy degrades: a plateau then a slump
points at a clustered failure mode (e.g., a qtype block that's
struggling), while a steady downward slope points at systematic drift.

Order: the manifest's per_question is written in dataset / question_id
order (deterministic across runs at the same seed), NOT in completion
order. So cumulative accuracy here is "accuracy after the first N
questions in dataset order", which is reproducible — if you re-run
with the same seed, the same Nth row should give the same cum_acc.

Usage:
    python benchmarks/cum_accuracy.py <manifest.json>
    python benchmarks/cum_accuracy.py <manifest.json> --by-qtype
    python benchmarks/cum_accuracy.py <manifest.json> --csv > cum.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _emit_per_question(per_question: list[dict], *, csv: bool) -> None:
    """One row per question, with running cumulative accuracy."""
    if csv:
        print("q,qid,qtype,score,cum_correct,cum_acc")
    else:
        print(f"{'q':>4}  {'qid':<32}  {'qtype':<22}  {'score':>5}  {'cum_cor':>7}  {'cum_acc':>7}")
        print(f"{'-' * 4}  {'-' * 32}  {'-' * 22}  {'-' * 5}  {'-' * 7}  {'-' * 7}")
    cum_correct = 0.0
    for i, q in enumerate(per_question, start=1):
        score = float(q.get("score", 0.0))
        cum_correct += score
        cum_acc = cum_correct / i
        qid = str(q.get("question_id", ""))[:32]
        qtype = str(q.get("question_type", ""))[:22]
        if csv:
            print(f"{i},{qid},{qtype},{score:.1f},{cum_correct:.1f},{cum_acc:.4f}")
        else:
            marker = ""
            if i % 10 == 0:
                marker = "  <-- checkpoint"
            print(
                f"{i:>4}  {qid:<32}  {qtype:<22}  {score:>5.1f}  "
                f"{cum_correct:>7.1f}  {cum_acc:>7.4f}{marker}"
            )


def _emit_per_qtype(per_question: list[dict], *, csv: bool) -> None:
    """For each qtype, running cumulative accuracy WITHIN that qtype.

    Reveals which qtype is dragging the overall number down, and
    whether it's a few catastrophic failures vs steady weakness.
    """
    seen: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "correct": 0.0})
    if csv:
        print("q,qtype,n_in_qtype,correct_in_qtype,cum_acc_in_qtype")
    else:
        print(f"{'q':>4}  {'qtype':<22}  {'n_qt':>4}  {'cor':>5}  {'cum_acc_qt':>10}")
        print(f"{'-' * 4}  {'-' * 22}  {'-' * 4}  {'-' * 5}  {'-' * 10}")
    for i, q in enumerate(per_question, start=1):
        qt = str(q.get("question_type", "unknown"))
        score = float(q.get("score", 0.0))
        s = seen[qt]
        s["n"] += 1
        s["correct"] += score
        cum_acc_qt = s["correct"] / s["n"]
        if csv:
            print(f"{i},{qt},{int(s['n'])},{s['correct']:.1f},{cum_acc_qt:.4f}")
        else:
            print(f"{i:>4}  {qt:<22}  {int(s['n']):>4}  {s['correct']:>5.1f}  {cum_acc_qt:>10.4f}")


def _emit_summary(per_question: list[dict]) -> None:
    """Final per-qtype summary table, prints AFTER the main rows."""
    by_qt: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "correct": 0.0})
    for q in per_question:
        qt = str(q.get("question_type", "unknown"))
        by_qt[qt]["n"] += 1
        by_qt[qt]["correct"] += float(q.get("score", 0.0))
    total_n = sum(s["n"] for s in by_qt.values())
    total_cor = sum(s["correct"] for s in by_qt.values())
    print()
    print("=== final per-qtype summary ===")
    print(f"{'qtype':<22}  {'n':>4}  {'cor':>5}  {'acc':>7}")
    print(f"{'-' * 22}  {'-' * 4}  {'-' * 5}  {'-' * 7}")
    for qt in sorted(by_qt):
        s = by_qt[qt]
        print(f"{qt:<22}  {int(s['n']):>4}  {s['correct']:>5.1f}  {s['correct'] / s['n']:>7.4f}")
    print(f"{'-' * 22}  {'-' * 4}  {'-' * 5}  {'-' * 7}")
    print(f"{'TOTAL':<22}  {int(total_n):>4}  {total_cor:>5.1f}  {total_cor / total_n:>7.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to a longmemeval manifest JSON")
    parser.add_argument(
        "--by-qtype",
        action="store_true",
        help="Per-row, show cumulative accuracy WITHIN each qtype instead of overall.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV instead of a fixed-width table.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the final per-qtype summary block.",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pq = manifest.get("per_question") or []
    if not pq:
        print(f"error: manifest has no per_question entries", file=sys.stderr)
        return 1

    if args.by_qtype:
        _emit_per_qtype(pq, csv=args.csv)
    else:
        _emit_per_question(pq, csv=args.csv)

    if not args.no_summary and not args.csv:
        _emit_summary(pq)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
