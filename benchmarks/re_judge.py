"""Re-judge a LongMemEval manifest with a pinned judge snapshot and
optional strict-fair rubric clarification.

The manifest forensic on the n=500 v3a SOTA run (`bb7c8412`) identified
8-11 questions where the model's answer was semantically correct but
the judge (`openai/gpt-4o`, floating alias) marked it wrong. Common
patterns:

  * "...3 total." judged wrong because gold is bare "3"
  * "did not mention X. Mentioned Y instead." judged wrong against
    "did not mention this; mentioned Y but not X" — same answer
  * "Under the bed." judged wrong against "under my bed"

This tool re-judges using:
  1. A PINNED judge snapshot (`openai/gpt-4o-2024-08-06` by default)
     so the verdict is reproducible — floating aliases drift.
  2. Optionally a `--strict-fair` rubric footer that explicitly tells
     the judge to accept embedded gold values, equivalent abstain
     paraphrases, and minor pronoun drift.

Outputs:
  * `re_judge_<timestamp>.json` -- per-question old vs new verdicts
  * Stdout: summary table + flip count + new accuracy

Usage:
    python -m benchmarks.re_judge \\
        --manifest benchmarks/runs/<...>-longmemeval.json \\
        --judge openrouter \\
        --judge-model openai/gpt-4o-2024-08-06 \\
        [--strict-fair] [--only-failures] [--parallel 10]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suite module lives outside the package; load it the way the bench loader does.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SUITE_PATH = _REPO_ROOT / "benchmarks" / "suites" / "longmemeval.py"


def _load_env_file(path: Path) -> bool:
    """Best-effort `.env` loading. Mirrors engram.bench._cli._load_env_file
    so this script works the same way the bench CLI does without
    importing a private helper. Existing env vars take precedence.
    """
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
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


def _load_suite_module() -> Any:
    if "_re_judge_longmemeval" in sys.modules:
        return sys.modules["_re_judge_longmemeval"]
    sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "_re_judge_longmemeval", _SUITE_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_re_judge_longmemeval"] = mod
    spec.loader.exec_module(mod)
    return mod


_STRICT_FAIR_FOOTER = (
    "\n\nClarifications:\n"
    "(a) If the gold value appears verbatim anywhere in the response, "
    "answer yes -- enumeration prefixes or trailing context do not "
    "make the response wrong.\n"
    "(b) Equivalent abstain patterns are correct: 'did not mention X; "
    "mentioned Y' and 'did not mention this; mentioned Y but not X' "
    "convey the same refusal and should both be judged yes if the gold "
    "is also an abstain that names Y.\n"
    "(c) Minor pronoun drift in non-key tokens (e.g. 'the bed' vs 'my "
    "bed', 'I' vs 'you') is acceptable and should not flip the verdict."
)


@dataclass(slots=True)
class _ReJudgeResult:
    qid: str
    qtype: str
    gold: str
    response: str
    old_score: float
    new_score: float
    new_raw: str
    error: str | None = None


def _build_judge_chat(provider: str, model: str) -> Any:
    """Use the same chat builder the bench uses, so providers behave identically."""
    from engram.bench._real_provider import build_chat
    return build_chat(provider, model=model, max_tokens=None)


def _re_judge_one(
    *,
    suite: Any,
    chat: Any,
    record: dict[str, Any],
    use_strict_fair: bool,
) -> _ReJudgeResult:
    qid = record["question_id"]
    qtype = record["question_type"]
    question = record.get("question", "")
    gold = str(record.get("gold", ""))
    response = (record.get("response") or "").strip()
    old_score = float(record.get("score", 0.0))

    # If the original run errored (empty response + error field), skip
    # re-judging -- those can't flip.
    if "error" in record or not response:
        return _ReJudgeResult(
            qid=qid, qtype=qtype, gold=gold, response=response,
            old_score=old_score, new_score=old_score, new_raw="(skipped: errored or empty)",
        )

    if qtype not in suite._JUDGE_INSTRUCTIONS:
        return _ReJudgeResult(
            qid=qid, qtype=qtype, gold=gold, response=response,
            old_score=old_score, new_score=old_score, new_raw="",
            error=f"unknown qtype {qtype!r}",
        )

    instructions = suite._JUDGE_INSTRUCTIONS[qtype]
    if use_strict_fair:
        instructions = instructions + _STRICT_FAIR_FOOTER

    prompt = suite._read_prompt("judge").format(
        instructions=instructions,
        question=question,
        gold=gold,
        response=response,
    )

    # Build the Message via the suite's Message class; avoids re-importing.
    Message = suite.Message
    try:
        raw = chat.chat([Message(role="user", content=prompt)])
        verdict = suite._parse_judge_verdict(raw)
    except Exception as exc:  # noqa: BLE001 -- surface any provider error in the report
        return _ReJudgeResult(
            qid=qid, qtype=qtype, gold=gold, response=response,
            old_score=old_score, new_score=old_score, new_raw="",
            error=f"{type(exc).__name__}: {exc}",
        )
    new_score = 1.0 if verdict else 0.0
    return _ReJudgeResult(
        qid=qid, qtype=qtype, gold=gold, response=response,
        old_score=old_score, new_score=new_score, new_raw=raw,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Path to the LongMemEval manifest JSON to re-judge.")
    parser.add_argument("--judge", default="openrouter",
                        help="Chat provider for the judge (openrouter, openai, ...).")
    parser.add_argument("--judge-model", default="openai/gpt-4o-2024-08-06",
                        help="Pinned judge model. Use a DATED snapshot; "
                             "floating aliases like 'openai/gpt-4o' drift.")
    parser.add_argument("--strict-fair", action="store_true",
                        help="Append the strict-fair rubric clarification to "
                             "every qtype's instructions.")
    parser.add_argument("--only-failures", action="store_true",
                        help="Only re-judge questions with score=0 (faster, "
                             "but won't catch lenient-FP risk). Default is "
                             "re-judge all questions for an honest delta.")
    parser.add_argument("--parallel", type=int, default=10,
                        help="Concurrent judge calls. Default 10.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output report JSON path. Default: "
                             "benchmarks/re_judge_<manifest-stem>_<ts>.json")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="Path to a .env file to load before resolving "
                             "the provider (default: .env in cwd). Existing "
                             "environment variables take precedence.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    if _load_env_file(args.env_file):
        print(f"loaded {args.env_file}", file=sys.stderr)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("re_judge")

    suite = _load_suite_module()
    manifest_path: Path = args.manifest
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["per_question"]
    log.info("Loaded manifest: %d questions, original accuracy=%.4f",
             len(records), manifest["aggregate_metrics"].get("accuracy", -1))

    targets = [r for r in records if (not args.only_failures or r["score"] == 0.0)]
    log.info("Re-judging %d questions (%s) with judge=%s/%s%s",
             len(targets),
             "score=0 only" if args.only_failures else "all",
             args.judge, args.judge_model,
             " + strict-fair rubric" if args.strict_fair else "")

    chat = _build_judge_chat(args.judge, args.judge_model)
    log.info("Judge chat built. Beginning re-judge...")
    t0 = time.time()

    results: list[_ReJudgeResult] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(_re_judge_one, suite=suite, chat=chat,
                      record=r, use_strict_fair=args.strict_fair): r
            for r in targets
        }
        completed = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            completed += 1
            if completed % 25 == 0:
                log.info("  progress: %d/%d", completed, len(targets))

    elapsed = time.time() - t0
    log.info("Re-judge complete in %.1fs (%.2fs/q)",
             elapsed, elapsed / max(1, len(targets)))

    # Compute flip stats
    pf = [r for r in results if r.old_score == 1.0 and r.new_score == 0.0]  # was right, now wrong
    fp = [r for r in results if r.old_score == 0.0 and r.new_score == 1.0]  # was wrong, now right
    errs = [r for r in results if r.error is not None]
    log.info("Flips: PF (was right, now wrong) = %d", len(pf))
    log.info("       FP (was wrong, now right) = %d", len(fp))
    log.info("       net = %+d", len(fp) - len(pf))
    log.info("Judge errors: %d", len(errs))

    # Compute new accuracy
    old_correct = sum(1 for r in records if r["score"] == 1.0)
    # For records we re-judged, use new score; for the rest (only-failures mode), keep old.
    qid_to_new = {r.qid: r.new_score for r in results}
    new_correct = 0
    for rec in records:
        if "error" in rec:
            continue  # excluded from accuracy_correct
        score = qid_to_new.get(rec["question_id"], rec["score"])
        if score == 1.0:
            new_correct += 1
    n_completed = sum(1 for r in records if "error" not in r)
    new_acc_correct = new_correct / max(1, n_completed)
    old_acc_correct = manifest["aggregate_metrics"].get("accuracy_correct", -1)

    log.info("Accuracy_correct: old=%.4f  new=%.4f  Δ=%+.4f",
             old_acc_correct, new_acc_correct,
             new_acc_correct - old_acc_correct)

    # Output report
    out_path = args.output or (
        manifest_path.parent.parent
        / f"re_judge_{manifest_path.stem}_{int(time.time())}.json"
    )
    report = {
        "source_manifest": str(manifest_path),
        "judge_provider": args.judge,
        "judge_model": args.judge_model,
        "strict_fair_rubric": args.strict_fair,
        "only_failures": args.only_failures,
        "n_records_re_judged": len(targets),
        "elapsed_s": elapsed,
        "flips": {
            "PF_was_right_now_wrong": [
                {"qid": r.qid, "qtype": r.qtype, "gold": r.gold,
                 "response": r.response[:300], "new_raw": r.new_raw[:200]}
                for r in pf
            ],
            "FP_was_wrong_now_right": [
                {"qid": r.qid, "qtype": r.qtype, "gold": r.gold,
                 "response": r.response[:300], "new_raw": r.new_raw[:200]}
                for r in fp
            ],
            "errors": [
                {"qid": r.qid, "qtype": r.qtype, "error": r.error}
                for r in errs
            ],
        },
        "old_accuracy_correct": old_acc_correct,
        "new_accuracy_correct": new_acc_correct,
        "delta_accuracy_correct": new_acc_correct - old_acc_correct,
    }
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Report written: %s", out_path)

    # Print flip table to stdout for quick scan
    if fp:
        print("\n== FP (was wrong, now right) ==")
        for r in fp:
            print(f"  [{r.qtype[:18]:18}] {r.qid[:18]:18}  gold={r.gold[:50]!r}")
            print(f"     resp={r.response[:120]!r}")
    if pf:
        print("\n== PF (was right, now wrong) — INVESTIGATE ==")
        for r in pf:
            print(f"  [{r.qtype[:18]:18}] {r.qid[:18]:18}  gold={r.gold[:50]!r}")
            print(f"     resp={r.response[:120]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
