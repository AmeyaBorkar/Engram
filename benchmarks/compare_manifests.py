"""Compare two LongMemEval manifests and emit a per-question delta report.

Usage:
    python benchmarks/compare_manifests.py BASELINE.json NEW.json [--out report.md]

Produces a markdown report covering:
- Headline: overall and per-qtype accuracy delta
- Flip analysis: which questions went correct->wrong vs wrong->correct
- Response-shape evolution: length distribution, empty / cap-cliff / CoT-preamble counts
- Failure-mode reclassification: empty / cot_preamble / refusal / wrong_concrete buckets
- Samples of flipped questions (lost wins and recovered failures)

The script reads both manifests as the project's JSON shape:
    {
        "aggregate_metrics": {"accuracy": ..., "accuracy_<qtype>": ...},
        "engram_config": {...},
        "per_question": [
            {"question_id", "question", "question_type", "gold", "response", "score", ...},
            ...
        ],
        ...
    }

Exit codes:
    0 - success
    2 - file or argument error

Designed for forensic comparison after a code change (e.g., the
JOURNEY 24 max_tokens=8192 cap fix): point it at the pre-fix manifest
and the post-fix manifest to see exactly what moved and what didn't.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Response-classification regexes -- same buckets as JOURNEY 24's analysis.
_COT_STARTERS = re.compile(
    r"^\s*(the user|let me|looking at|i need to|first,|to answer|"
    r"based on the memory|to determine|i'll|i will|step \d|memory \[|"
    r"the question|let's)",
    re.I,
)
_REFUSAL = re.compile(
    r"^\s*(i don't know|i do not know|i don't have|i cannot|"
    r"i can't determine|sorry|unfortunately)",
    re.I,
)

# Cap-cliff bounds in characters (1024 tokens ~= 4096 chars at ~4 char/token;
# the empirical cliff in our manifests sits in this window).
_CLIFF_LOW = 3500
_CLIFF_HIGH = 5000


def _classify_response(response: Any) -> str:
    """Bucket a response string into a single failure-mode label."""
    if not isinstance(response, str):
        return "non_str"
    stripped = response.strip()
    if not stripped:
        return "empty"
    if _REFUSAL.match(stripped) and len(stripped) < 100:
        return "refusal"
    if _COT_STARTERS.match(stripped):
        return "cot_preamble"
    if len(stripped) > 400:
        return "verbose_other"
    return "concrete"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"error: manifest not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _qtype_accuracies(agg: dict[str, Any]) -> dict[str, float]:
    """Return per-qtype accuracy as a flat dict, preferring accuracy_correct."""
    out: dict[str, float] = {}
    for key, val in agg.items():
        if isinstance(val, (int, float)) and (
            key.startswith("accuracy_correct_") or
            (key.startswith("accuracy_") and not key.startswith("accuracy_correct"))
        ):
            qtype = key.removeprefix("accuracy_correct_").removeprefix("accuracy_")
            if qtype in {"correct"}:  # the bare "accuracy_correct" overall
                continue
            # Prefer accuracy_correct_<qtype> when both exist.
            if key.startswith("accuracy_correct_") or qtype not in out:
                out[qtype] = float(val)
    return out


def _overall(agg: dict[str, Any]) -> float:
    """Prefer accuracy_correct (excludes errored items) over accuracy."""
    if "accuracy_correct" in agg:
        return float(agg["accuracy_correct"])
    return float(agg.get("accuracy", 0.0))


def _length_stats(responses: list[str]) -> dict[str, int]:
    if not responses:
        return {"p50": 0, "p90": 0, "p99": 0, "max": 0, "mean": 0}
    sorted_lens = sorted(len(r) if isinstance(r, str) else 0 for r in responses)
    n = len(sorted_lens)
    return {
        "p50": sorted_lens[n // 2],
        "p90": sorted_lens[min(n - 1, int(n * 0.90))],
        "p99": sorted_lens[min(n - 1, int(n * 0.99))],
        "max": sorted_lens[-1],
        "mean": int(statistics.mean(sorted_lens)),
    }


def _shape_summary(pq: list[dict]) -> dict[str, Any]:
    """Aggregate response-shape stats for a manifest's per_question list."""
    responses = [q.get("response", "") for q in pq]
    classifications = Counter(_classify_response(r) for r in responses)
    cliff = sum(
        1 for r in responses
        if isinstance(r, str) and _CLIFF_LOW <= len(r) <= _CLIFF_HIGH
    )
    return {
        "n": len(pq),
        "length": _length_stats(responses),
        "classes": dict(classifications),
        "cliff_count": cliff,
    }


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _fmt_pp(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{100 * delta:.1f}pp"


def _diff_questions(
    base_pq: list[dict], new_pq: list[dict]
) -> dict[str, list[dict]]:
    """Return question lists keyed by transition: PP, FF, PF, FP, only_in_base, only_in_new."""
    base_by_id = {q.get("question_id"): q for q in base_pq if q.get("question_id")}
    new_by_id = {q.get("question_id"): q for q in new_pq if q.get("question_id")}
    common = base_by_id.keys() & new_by_id.keys()
    pp, ff, pf, fp = [], [], [], []
    for qid in common:
        b, n = base_by_id[qid], new_by_id[qid]
        bp, np_ = b.get("score", 0) >= 0.5, n.get("score", 0) >= 0.5
        rec = {
            "qid": qid,
            "qtype": b.get("question_type") or n.get("question_type"),
            "question": b.get("question") or n.get("question"),
            "gold": b.get("gold") or n.get("gold"),
            "base_response": b.get("response", ""),
            "new_response": n.get("response", ""),
            "base_score": b.get("score", 0),
            "new_score": n.get("score", 0),
        }
        if bp and np_:
            pp.append(rec)
        elif (not bp) and (not np_):
            ff.append(rec)
        elif bp and not np_:
            pf.append(rec)
        else:
            fp.append(rec)
    return {
        "PP": pp, "FF": ff, "PF": pf, "FP": fp,
        "only_in_base": [
            {"qid": qid, **base_by_id[qid]}
            for qid in (base_by_id.keys() - new_by_id.keys())
        ],
        "only_in_new": [
            {"qid": qid, **new_by_id[qid]}
            for qid in (new_by_id.keys() - base_by_id.keys())
        ],
    }


def _truncate(s: Any, n: int = 100) -> str:
    if s is None:
        return "_(none)_"
    s = str(s)
    if not s.strip():
        return "_(empty)_"
    return s if len(s) <= n else s[: n - 1] + "..."


def _sample_table(records: list[dict], limit: int = 8) -> str:
    if not records:
        return "_(none)_\n"
    by_qtype: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_qtype[r["qtype"] or "?"].append(r)
    lines = ["| qtype | question | gold | base resp | new resp |", "|---|---|---|---|---|"]
    shown = 0
    for qtype in sorted(by_qtype.keys(), key=lambda k: -len(by_qtype[k])):
        for r in by_qtype[qtype][: max(1, limit // max(1, len(by_qtype)))]:
            if shown >= limit:
                break
            lines.append(
                f"| {qtype} | {_truncate(r['question'], 80)} | "
                f"{_truncate(r['gold'], 60)} | "
                f"{_truncate(r['base_response'], 80)} | "
                f"{_truncate(r['new_response'], 80)} |"
            )
            shown += 1
    return "\n".join(lines) + "\n"


def _qtype_flip_table(diff: dict[str, list[dict]]) -> str:
    """Per-qtype flip counts: how many PP / FF / PF / FP per qtype."""
    by_qtype: dict[str, Counter] = defaultdict(Counter)
    for label in ("PP", "FF", "PF", "FP"):
        for r in diff[label]:
            by_qtype[r["qtype"] or "?"][label] += 1
    lines = ["| qtype | PP | FF | **PF (lost)** | **FP (gained)** | net |", "|---|---:|---:|---:|---:|---:|"]
    grand = Counter()
    for qtype in sorted(by_qtype.keys()):
        c = by_qtype[qtype]
        net = c["FP"] - c["PF"]
        sign = "+" if net >= 0 else ""
        lines.append(
            f"| {qtype} | {c['PP']} | {c['FF']} | **{c['PF']}** | **{c['FP']}** | {sign}{net} |"
        )
        grand.update(c)
    net_total = grand["FP"] - grand["PF"]
    sign = "+" if net_total >= 0 else ""
    lines.append(
        f"| **total** | {grand['PP']} | {grand['FF']} | **{grand['PF']}** | **{grand['FP']}** | **{sign}{net_total}** |"
    )
    return "\n".join(lines) + "\n"


def build_report(
    base_path: Path, new_path: Path, base: dict, new: dict
) -> str:
    base_pq = base.get("per_question", [])
    new_pq = new.get("per_question", [])

    base_overall = _overall(base["aggregate_metrics"])
    new_overall = _overall(new["aggregate_metrics"])
    overall_delta = new_overall - base_overall

    base_qt = _qtype_accuracies(base["aggregate_metrics"])
    new_qt = _qtype_accuracies(new["aggregate_metrics"])
    all_qtypes = sorted(set(base_qt) | set(new_qt))

    base_shape = _shape_summary(base_pq)
    new_shape = _shape_summary(new_pq)

    diff = _diff_questions(base_pq, new_pq)

    out: list[str] = []
    out.append("# Manifest comparison report\n")
    out.append(f"- **Baseline**: `{base_path.name}`")
    out.append(f"- **New**: `{new_path.name}`")
    out.append(f"- **Baseline commit**: `{base.get('git_commit', '?')[:7]}`")
    out.append(f"- **New commit**: `{new.get('git_commit', '?')[:7]}`")
    out.append("")

    # --- Headline ---
    out.append("## Headline\n")
    out.append(
        f"- **Overall accuracy: {_fmt_pct(base_overall)} -> "
        f"{_fmt_pct(new_overall)}** ({_fmt_pp(overall_delta)})"
    )
    out.append(f"- Questions: base {base_shape['n']} / new {new_shape['n']}")
    out.append("")

    # --- Per-qtype ---
    out.append("## Per-qtype accuracy\n")
    out.append("| qtype | base | new | Δ |")
    out.append("|---|---:|---:|---:|")
    for qt in all_qtypes:
        b = base_qt.get(qt)
        n = new_qt.get(qt)
        if b is None or n is None:
            continue
        out.append(f"| {qt} | {_fmt_pct(b)} | {_fmt_pct(n)} | {_fmt_pp(n - b)} |")
    out.append("")

    # --- Flip analysis ---
    out.append("## Question-level flip analysis\n")
    out.append(
        f"- PP (still pass): {len(diff['PP'])}\n"
        f"- FF (still fail): {len(diff['FF'])}\n"
        f"- PF (lost wins, base passed but new failed): **{len(diff['PF'])}**\n"
        f"- FP (recovered failures, base failed but new passed): **{len(diff['FP'])}**\n"
        f"- Net flips: **{len(diff['FP']) - len(diff['PF']):+d}**\n"
        f"- Only in base: {len(diff['only_in_base'])}\n"
        f"- Only in new: {len(diff['only_in_new'])}\n"
    )
    out.append("### Per-qtype flips\n")
    out.append(_qtype_flip_table(diff))

    # --- Response shape ---
    out.append("## Response-shape distribution\n")
    out.append("| metric | base | new | Δ |")
    out.append("|---|---:|---:|---:|")
    for k in ("p50", "p90", "p99", "max", "mean"):
        b = base_shape["length"][k]
        n = new_shape["length"][k]
        out.append(f"| {k} length | {b} | {n} | {n - b:+d} |")
    out.append(
        f"| cliff hits (3500-5000ch) | {base_shape['cliff_count']} | "
        f"{new_shape['cliff_count']} | {new_shape['cliff_count'] - base_shape['cliff_count']:+d} |"
    )
    out.append("")

    out.append("### Failure-mode classification\n")
    out.append("| class | base | new | Δ |")
    out.append("|---|---:|---:|---:|")
    all_classes = sorted(set(base_shape["classes"]) | set(new_shape["classes"]))
    for cls in all_classes:
        b = base_shape["classes"].get(cls, 0)
        n = new_shape["classes"].get(cls, 0)
        out.append(f"| {cls} | {b} | {n} | {n - b:+d} |")
    out.append("")

    # --- Samples ---
    out.append("## Sample flipped questions\n")
    out.append("### Lost wins (PF): base passed, new failed\n")
    out.append(_sample_table(diff["PF"], limit=10))
    out.append("\n### Recovered (FP): base failed, new passed\n")
    out.append(_sample_table(diff["FP"], limit=10))

    # --- Config diff (best-effort) ---
    base_cfg = base.get("engram_config", {})
    new_cfg = new.get("engram_config", {})
    if base_cfg or new_cfg:
        out.append("\n## engram_config delta\n")
        all_keys = sorted(set(base_cfg) | set(new_cfg))
        out.append("| key | base | new |")
        out.append("|---|---|---|")
        for k in all_keys:
            b = base_cfg.get(k, "_unset_")
            n = new_cfg.get(k, "_unset_")
            if b != n:
                bs = _truncate(json.dumps(b, default=str), 80) if b != "_unset_" else "_unset_"
                ns = _truncate(json.dumps(n, default=str), 80) if n != "_unset_" else "_unset_"
                out.append(f"| {k} | {bs} | {ns} |")
        out.append("")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path, help="baseline manifest .json")
    parser.add_argument("new", type=Path, help="new manifest .json")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write markdown report to this path (default: stdout)",
    )
    args = parser.parse_args(argv)

    base = _load(args.baseline)
    new = _load(args.new)
    report = build_report(args.baseline, args.new, base, new)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
