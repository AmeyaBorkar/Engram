"""Forensic analysis of response shapes across LongMemEval manifests."""

import json
import sys
from pathlib import Path

# --- config ---
RUNS = Path(r"C:\Users\ameya\Documents\Engram\benchmarks\runs")

# CoT preamble openings (case-sensitive prefix)
COT_PREFIXES = (
    "The user",
    "Let me",
    "Looking at",
    "Based on the memory",
    "I need to",
    "First,",
    "To answer",
    "Memory [",
    "I'll",
    "I will",
)

# Refusal openings (case-insensitive)
REFUSAL_PREFIXES = (
    "i don't know",
    "i do not know",
    "i don't have",
    "i cannot",
    "i can't determine",
)

CLIFF_LO = 3500
CLIFF_HI = 5000


def quantile(values, q):
    if not values:
        return 0
    vs = sorted(values)
    idx = max(0, min(len(vs) - 1, round((len(vs) - 1) * q)))
    return vs[idx]


def is_cot(resp):
    return any(resp.startswith(p) for p in COT_PREFIXES)


def is_refusal(resp):
    s = resp.lower().strip()
    return any(s.startswith(p) for p in REFUSAL_PREFIXES)


def shape_bucket(resp):
    s = resp.strip()
    if len(s) == 0:
        return "empty"
    if is_cot(resp):
        return "cot_preamble"
    if is_refusal(resp):
        return "refusal"
    if len(s) > 200:
        return "verbose_other"
    return "wrong_concrete"


def analyze_manifest(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"path": str(path), "error": str(e)}

    pq = data.get("per_question") or []
    if not pq:
        return None

    rows = []
    for r in pq:
        if "response" not in r:
            continue
        rows.append(r)
    if not rows:
        return None

    lengths = [len(r["response"]) for r in rows]
    n = len(rows)
    n_empty = sum(1 for r in rows if len(r["response"].strip()) == 0)
    n_cliff = sum(1 for L in lengths if CLIFF_LO <= L <= CLIFF_HI)
    n_cot = sum(1 for r in rows if is_cot(r["response"]))
    n_refusal = sum(1 for r in rows if is_refusal(r["response"]))

    p50 = quantile(lengths, 0.50)
    p90 = quantile(lengths, 0.90)
    p95 = quantile(lengths, 0.95)
    p99 = quantile(lengths, 0.99)
    mx = max(lengths) if lengths else 0

    # accuracy
    scores = [r.get("score", 0.0) for r in rows]
    score_mean = sum(scores) / len(scores) if scores else 0.0

    # qtype cliff breakdown
    qtype_total = {}
    qtype_cliff = {}
    qtype_empty = {}
    qtype_fail = {}
    qtype_pass = {}
    for r in rows:
        qt = r.get("question_type", "?")
        qtype_total[qt] = qtype_total.get(qt, 0) + 1
        L = len(r["response"])
        if CLIFF_LO <= L <= CLIFF_HI:
            qtype_cliff[qt] = qtype_cliff.get(qt, 0) + 1
        if len(r["response"].strip()) == 0:
            qtype_empty[qt] = qtype_empty.get(qt, 0) + 1
        if r.get("score", 0.0) >= 1.0:
            qtype_pass[qt] = qtype_pass.get(qt, 0) + 1
        else:
            qtype_fail[qt] = qtype_fail.get(qt, 0) + 1

    # failed-response shape buckets
    failures = [r for r in rows if r.get("score", 0.0) < 1.0]
    fail_buckets = {
        "empty": 0,
        "cot_preamble": 0,
        "refusal": 0,
        "wrong_concrete": 0,
        "verbose_other": 0,
    }
    for r in failures:
        fail_buckets[shape_bucket(r["response"])] += 1

    cfg = data.get("engram_config") or {}
    # extract short key flags
    cfg_keys = sorted(cfg.keys())
    cfg_str = ", ".join(f"{k}={cfg[k]}" for k in cfg_keys)

    return {
        "path": str(path),
        "name": path.name,
        "timestamp": data.get("timestamp"),
        "git_commit": (data.get("git_commit") or "")[:8],
        "provider": data.get("provider"),
        "provider_hash": data.get("provider_hash") or "",
        "engram_config": cfg,
        "engram_config_str": cfg_str,
        "n": n,
        "score": score_mean,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": mx,
        "n_empty": n_empty,
        "n_cliff": n_cliff,
        "n_cot": n_cot,
        "n_refusal": n_refusal,
        "fail_buckets": fail_buckets,
        "n_failures": len(failures),
        "qtype_total": qtype_total,
        "qtype_cliff": qtype_cliff,
        "qtype_empty": qtype_empty,
        "qtype_fail": qtype_fail,
        "qtype_pass": qtype_pass,
    }


def gather():
    # all *-longmemeval.json under runs/, skipping noop/backup
    candidates = []
    for p in RUNS.rglob("*-longmemeval.json"):
        if "noop" in p.name:
            continue
        if ".backup" in p.name:
            continue
        candidates.append(p)
    # also check runs/release explicitly already included via rglob
    # de-dup absolute paths
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return sorted(out)


def main():
    paths = gather()
    summaries = []
    for p in paths:
        s = analyze_manifest(p)
        if s and "error" not in s:
            summaries.append(s)
        elif s and "error" in s:
            print(f"ERROR on {p}: {s['error']}", file=sys.stderr)
    # sort by timestamp
    summaries.sort(key=lambda s: s.get("timestamp") or "")
    return summaries


if __name__ == "__main__":
    summaries = main()
    # write a JSON intermediate for inspection
    out_path = RUNS.parent / "shape_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}; {len(summaries)} manifests analyzed")

    # Headline table
    print("\n=== TIMELINE (n>=50) ===")
    fmt = "{ts:<28} {cm:<10} {n:>4} {sc:>5}  p50={p50:<4} p90={p90:<5} p95={p95:<5} p99={p99:<5} max={mx:<5} empty={e:<3} cliff={c:<3} cot={cot:<3} refuse={r:<3} hash={ph:<24} cfg={cfg}"
    print(
        f"{'TIMESTAMP':<28} {'COMMIT':<10} {'N':>4} {'SCORE':>5}  LEN-DIST  EMPTY  CLIFF  COT  REFUSE  PROVIDER_HASH                CONFIG"
    )
    for s in summaries:
        if s["n"] < 50:
            continue
        print(
            fmt.format(
                ts=s["timestamp"] or "?",
                cm=s["git_commit"],
                n=s["n"],
                sc=f"{s['score']:.3f}",
                p50=s["p50"],
                p90=s["p90"],
                p95=s["p95"],
                p99=s["p99"],
                mx=s["max"],
                e=s["n_empty"],
                c=s["n_cliff"],
                cot=s["n_cot"],
                r=s["n_refusal"],
                ph=(s["provider_hash"] or "?")[:24],
                cfg=s["engram_config_str"][:80],
            )
        )
