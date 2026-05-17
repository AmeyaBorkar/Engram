"""LongMemEval benchmark suite (Stage 6 DoD).

Loads `longmemeval_s_cleaned.json` from `benchmarks/datasets/longmemeval/`,
runs the full retrieve -> answer -> judge pipeline against Engram, and
emits a manifest with per-question results plus per-type accuracy.

The dataset file is question-centric. Each record:

    {
      "question_id": "e47becba",
      "question_type": "single-session-user",
      "question": "What degree did I graduate with?",
      "question_date": "2023/05/30 (Tue) 23:40",
      "answer": "Business Administration",
      "answer_session_ids": [...],
      "haystack_session_ids": [...],
      "haystack_dates": [...],
      "haystack_sessions": [[{role, content}, ...], ...]
    }

Pipeline per question (matches the official setup, retrieve-then-answer):

  1. Build a fresh `Memory` for this question.
  2. Observe every turn from `haystack_sessions` as an Event whose source
     records the speaker. (We don't pre-consolidate -- that's an ablation
     to be added later as a separate suite.)
  3. Hierarchical retrieve top-K against the question text.
  4. Generate an answer via the chat provider, given the retrieved
     memory + the question.
  5. Score the answer with an LLM judge whose prompt mirrors the
     official LongMemEval evaluator: yes/no for correctness, with
     question-type-specific guidance for temporal-reasoning,
     knowledge-update, and single-session-preference.

Configuration:

  * `LONGMEMEVAL_MAX_QUESTIONS=N` -- cap the question count for a smoke
    run. Defaults to all 500.
  * `LONGMEMEVAL_K=K`             -- retrieval top-k (default 10).
  * `LONGMEMEVAL_JUDGE_MODEL=name` -- override the judge model. Defaults
    to the bench Provider's chat model (so OpenAI -> gpt-4o-mini judges
    its own answers; users who want gpt-4o as the judge can override).

Cost estimate for one full LongMemEval-S run @ gpt-4o-mini + text-embedding-3-small:
  ~ $1 (500 questions x 2 LLM calls + ~12M embedding tokens).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engram import Memory, SqliteStorage
from engram.bench import Provider, SuiteResult
from engram.integrations._agent import _majority_vote, _strip_cot
from engram.integrations._verify import verify_answer
from engram.providers._message import Message
from engram.providers._protocols import ChatProvider
from engram.retrieve._params import RetrieveParams
from engram.retrieve._reranker import Reranker
from engram.schemas import Embedding, Event, ItemKind

_LOG = logging.getLogger("engram.bench.longmemeval")

DATASET_ROOT = Path("benchmarks/datasets/longmemeval")
DEFAULT_FILENAME = "longmemeval_s_cleaned.json"
PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_VERSIONS: dict[str, str] = {
    "answer": "v1",
    "judge": "v1",
}

# Prompt-variant routing. JOURNEY section 24+ documents the n=500 v2 evaluation
# (commit 0166c0b5): v2's bundled abstain + per-qtype + scratchpad CoT
# REGRESSED by -3.7 pp because the scratchpad caused CoT-leakage / empty
# responses on multi-session and preference. The v2a / v2b / v2c variants
# below bisect the v2 design space:
#   - v2a: abstain only, softened (no "always anchor" hard rule), no qtype hints, no CoT.
#   - v2b: per-qtype FORMAT hints only (no abstain, no scratchpad CoT).
#   - v2c: v2a + v2b combined (abstain + format hints, no scratchpad CoT).
# Each variant gets its own template file and (for v2b / v2c) its own qtype
# hint dict. v1 path unchanged.

# v2 answer-prompt qtype hints. Activated only when `prompt_version="v2"`.
# Each hint is appended to the v2 prompt template via the {qtype_hint} slot.
# Designed to address specific Sonnet-vs-Kimi judge disagreements seen in
# JOURNEY §23: synthesis on preference / multi-session, date-math scratchpad
# on temporal, latest-value resolution on knowledge-update, abstain anchoring
# on sss-user. The base v2 prompt already carries the abstain pattern; these
# hints layer per-qtype expectations on top.
_V2_QTYPE_HINTS: dict[str, str] = {
    "single-session-user": (
        "This question asks about something the user mentioned in a single "
        "session. Locate the most specific mention in the retrieved memory "
        "and answer directly. If the specific fact is not present, anchor "
        "your answer with the closest related thing the user DID mention."
    ),
    "single-session-assistant": (
        "This question asks about something the assistant said in a single "
        "session. Quote or paraphrase the assistant's recommendation "
        "faithfully -- match the form and specifics."
    ),
    "single-session-preference": (
        "This is a PREFERENCE question. The expected answer format is "
        "'The user would prefer [X]' or an equivalent synthesis. Identify "
        "the user's preferences from patterns + explicit statements in the "
        "memory, then synthesize them into a single preference statement. "
        "Partial coverage of the user's preferences is acceptable as long "
        "as your synthesis is consistent with the evidence."
    ),
    "multi-session": (
        "This is a MULTI-SESSION aggregation question. The answer requires "
        "combining information from multiple separate sessions.\n"
        "Steps (show briefly, then state final answer):\n"
        "  (1) Identify each relevant value/event from each session.\n"
        "  (2) Aggregate as the question asks (sum, count, total, "
        "average, difference).\n"
        "  (3) Final answer: a single concrete value, not a CoT preamble."
    ),
    "temporal-reasoning": (
        "This is a TEMPORAL question. Compute durations / intervals "
        "explicitly.\n"
        "Steps (show briefly, then state final answer):\n"
        "  (1) Identify the relevant dates / durations in the memory.\n"
        "  (2) Compute the difference (days, weeks, months) explicitly.\n"
        "  (3) Final answer: a single concrete duration or date.\n"
        "Off-by-one errors on days are tolerated; off-by-many is not."
    ),
    "knowledge-update": (
        "This is a KNOWLEDGE-UPDATE question. Multiple values may be "
        "mentioned over time for the same fact. Pick the MOST RECENT one. "
        "If both the previous and updated value are visible in memory, "
        "stating both with the updated one clearly marked as current is "
        "acceptable; stating only the latest value is also acceptable."
    ),
}

# v2b qtype hints: FORMAT-ONLY (no "show your work" / scratchpad / CoT).
# The v2 evaluation showed the scratchpad instructions caused Kimi K2.6 to
# echo the hint text into the answer and sometimes produce empty responses
# entirely. v2b keeps the per-qtype routing but strips the work-show
# instructions; the model is told the expected answer FORM only.
_V2B_QTYPE_HINTS: dict[str, str] = {
    "single-session-user": (
        "Expected answer form: the specific fact in 1-5 words."
    ),
    "single-session-assistant": (
        "Expected answer form: the assistant's recommendation, faithfully "
        "paraphrased in 1-2 sentences."
    ),
    "single-session-preference": (
        "Expected answer form: a single sentence starting with "
        "'The user would prefer ...' synthesizing the user's preference."
    ),
    "multi-session": (
        "Expected answer form: a single concrete value (number, "
        "duration, name). No reasoning shown."
    ),
    "temporal-reasoning": (
        "Expected answer form: a single concrete date or duration. "
        "No reasoning shown. Off-by-one days are tolerated."
    ),
    "knowledge-update": (
        "Expected answer form: the most recent value for the fact. "
        "1-5 words."
    ),
}

# Map prompt_version -> (template_filename_version, qtype_hints_dict).
# Centralizing this here keeps _answer_with_phase_e simple and gives the
# CLI / test code one place to look for available variants.
# - v1:  original prompt, no qtype hints.
# - v2:  bundled abstain + per-qtype + scratchpad CoT (the n=500 regression).
# - v2a: abstain anchoring only, softened.  No qtype hints.
# - v2b: per-qtype format hints only.  No abstain anchoring, no scratchpad.
# - v2c: abstain (softened) + per-qtype format hints, no scratchpad.
_PROMPT_VARIANTS: dict[str, tuple[str, dict[str, str]]] = {
    "v1": ("v1", {}),
    "v2": ("v2", _V2_QTYPE_HINTS),
    "v2a": ("v2a", {}),
    "v2b": ("v2b", _V2B_QTYPE_HINTS),
    "v2c": ("v2c", _V2B_QTYPE_HINTS),
}

# Question-type-specific hints for the judge, taken verbatim from the
# official `evaluate_qa.py`.
_JUDGE_INSTRUCTIONS: dict[str, str] = {
    "single-session-user": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response is equivalent to the correct answer or contains all "
        "the intermediate steps to get the correct answer, you should also answer yes. "
        "If the response only contains a subset of the information required by the "
        "answer, answer no."
    ),
    "single-session-assistant": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response is equivalent to the correct answer or contains all "
        "the intermediate steps to get the correct answer, you should also answer yes. "
        "If the response only contains a subset of the information required by the "
        "answer, answer no."
    ),
    "multi-session": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response is equivalent to the correct answer or contains all "
        "the intermediate steps to get the correct answer, you should also answer yes. "
        "If the response only contains a subset of the information required by the "
        "answer, answer no."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. Do not penalize off-by-one errors for the number of days. If the "
        "response only contains a subset of the information required by the answer, "
        "answer no."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. A response that "
        "contains the previous information together with the updated answer is also "
        "acceptable. Otherwise, answer no."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric capturing the user's preferences, and a "
        "response from a model. Please answer yes if the response covers the rubric "
        "(partial coverage is acceptable as long as the response is consistent with the "
        "rubric). Otherwise, answer no."
    ),
}


@dataclass
class _QuestionOutcome:
    """One question's pipeline result. Returned by `_run_one_question`.

    The previous code path mutated shared lists in-place (per_question,
    per_type_scores, retrieve_ms, ...) from inside the worker, which
    fights parallel execution. Returning a value here lets the
    ThreadPoolExecutor path collect results without locks and the
    serial path stay bit-identical.
    """

    qid: str
    qtype: str
    question: str
    gold: str
    response: str
    score: float
    k: int
    turns_ingested: int
    ingest_ms: float
    consolidate_ms: float
    retrieve_ms: float
    answer_ms: float
    judge_ms: float
    error_msg: str | None = None


@dataclass
class _Progress:
    """Thread-safe completion / accuracy counter for progress logging.

    The serial and parallel paths both go through `mark()`; the lock
    serializes the running-accuracy log line so concurrent workers
    don't interleave the format string. Logging itself is already
    thread-safe (`logging.Handler.emit` uses an internal lock); the
    lock here is for the counter math, not the I/O.
    """

    total: int
    log_interval: int
    completed: int = 0
    correct: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def mark(self, outcome: _QuestionOutcome) -> None:
        verdict = (
            "ERROR"
            if outcome.error_msg
            else ("PASS" if outcome.score == 1.0 else "FAIL")
        )
        with self.lock:
            self.completed += 1
            self.correct += outcome.score
            done = self.completed
            acc = self.correct / done if done else 0.0
            should_log = done % self.log_interval == 0 or done == self.total
            if should_log:
                _LOG.info(
                    "q %d/%d [%s] -> %s "
                    "(ingest %d turns in %.1fs, ans %.1fs, jud %.1fs; acc=%.3f)",
                    done,
                    self.total,
                    outcome.qtype,
                    verdict,
                    outcome.turns_ingested,
                    outcome.ingest_ms / 1000.0,
                    outcome.answer_ms / 1000.0,
                    outcome.judge_ms / 1000.0,
                    acc,
                )


@dataclass(frozen=True)
class _Question:
    qid: str
    qtype: str
    question: str
    question_date: str
    gold: str
    answer_session_ids: tuple[str, ...]
    haystack_dates: tuple[str, ...]
    haystack_session_ids: tuple[str, ...]
    haystack_sessions: tuple[tuple[dict[str, Any], ...], ...]


def _stratified_sample(
    questions: Sequence[_Question],
    n: int,
    seed: int | None,
) -> list[_Question]:
    """Take a stratified sample of `n` questions across qtypes.

    Each qtype gets a slice proportional to its share of the full
    dataset; rounding remainder is allocated to the largest qtypes
    first. Within a qtype, selection is deterministic given `seed`:
    sort by `qid`, then shuffle with `random.Random(seed)` and take
    the leading slice. Two calls with the same `(questions, n, seed)`
    return the same list -- a prerequisite for reproducibility audits.

    If `n >= len(questions)`, returns all questions in their original
    order (sampling is a no-op).
    """
    if n >= len(questions):
        return list(questions)
    import random as _random

    by_qtype: dict[str, list[_Question]] = {}
    qtype_order: list[str] = []
    for q in questions:
        if q.qtype not in by_qtype:
            qtype_order.append(q.qtype)
            by_qtype[q.qtype] = []
        by_qtype[q.qtype].append(q)

    total = len(questions)
    # Proportional allocation with remainder going to the largest qtypes.
    raw: dict[str, float] = {qt: len(rows) / total * n for qt, rows in by_qtype.items()}
    floor: dict[str, int] = {qt: int(v) for qt, v in raw.items()}
    remainder = n - sum(floor.values())
    fractional = sorted(
        ((raw[qt] - floor[qt], qt) for qt in by_qtype),
        reverse=True,
    )
    for _, qt in fractional[:remainder]:
        floor[qt] += 1

    rng = _random.Random(seed if seed is not None else 0)  # noqa: S311 - sampling, not crypto
    picked: list[_Question] = []
    for qt in qtype_order:
        # Deterministic shuffle within qtype.
        pool = sorted(by_qtype[qt], key=lambda q: q.qid)
        rng.shuffle(pool)
        picked.extend(pool[: floor[qt]])
    return picked


def _load_dataset(path: Path) -> list[_Question]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out: list[_Question] = []
    for r in rows:
        out.append(
            _Question(
                qid=r["question_id"],
                qtype=r["question_type"],
                question=r["question"],
                question_date=r.get("question_date", ""),
                gold=r["answer"],
                answer_session_ids=tuple(r.get("answer_session_ids", ())),
                haystack_dates=tuple(r.get("haystack_dates", ())),
                haystack_session_ids=tuple(r.get("haystack_session_ids", ())),
                haystack_sessions=tuple(
                    tuple(turn for turn in session) for session in r.get("haystack_sessions", ())
                ),
            )
        )
    return out


def _checksum(path: Path) -> str:
    if not path.exists():
        return f"longmemeval/{path.name}/missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"longmemeval/{path.name}/{h.hexdigest()}"


def _read_prompt(name: str, version: str = "v1") -> str:
    """Load a versioned prompt template from PROMPTS_DIR.

    `version` defaults to "v1" so callers that don't specify it get the
    original behavior (zero risk to in-flight benches reading from the
    same module). The bench's `prompt_version` configure() field
    controls which template the answer step reads.
    """
    return (PROMPTS_DIR / f"longmemeval_{name}_{version}.txt").read_text(encoding="utf-8")


def _format_memory(results: Sequence[Any]) -> str:
    """Format retrieved memory as a numbered list, one bullet per result.

    The retrieval surface is `RetrievalResult`; we render
    `[level, score] content` so the answerer sees both the source level
    (event/summary/abstraction) and the relevance score.
    """
    if not results:
        return "(no relevant memory found)"
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] (level={r.level.value}, score={r.score:.2f}) {r.content}")
    return "\n".join(lines)


def _format_memory_grouped(results: Sequence[Any], storage: Any) -> str:
    """Group retrieved events by session_id; order sessions by best
    in-session score; within session, order by turn_index.

    Falls back to flat formatting when no metadata is available (e.g.
    abstractions or events without session_id).  The grouped form
    drops the `level/score` noise that may distract the answerer and
    instead shows session-level context blocks the LLM can reason
    across.  Designed to help on multi-session and _abs questions
    where structural context matters more than rank-ordered bullets.
    """
    if not results:
        return "(no relevant memory found)"

    # Buckets keyed by session_id; abstractions / no-metadata go to
    # an "_other_" bucket rendered first.
    buckets: dict[str, list[tuple[int, Any, dict[str, Any]]]] = {}
    for i, r in enumerate(results):
        event = storage.get_event(r.item_id) if hasattr(r, "item_id") else None
        meta: dict[str, Any] = dict(event.metadata) if event and event.metadata else {}
        session_id = str(meta.get("session_id", "_other_"))
        buckets.setdefault(session_id, []).append((i, r, meta))

    # Sort sessions by the BEST score within each (highest first); within
    # each session, order by turn_index ascending so chronology is
    # preserved.
    def session_best_score(items: list[tuple[int, Any, dict[str, Any]]]) -> float:
        return max(item[1].score for item in items)

    if "_other_" in buckets:
        ordered_sessions = ["_other_"] + sorted(
            (s for s in buckets if s != "_other_"),
            key=lambda s: -session_best_score(buckets[s]),
        )
    else:
        ordered_sessions = sorted(buckets, key=lambda s: -session_best_score(buckets[s]))

    lines: list[str] = []
    for session_idx, sid in enumerate(ordered_sessions, start=1):
        items = buckets[sid]
        # Sort within-session by turn_index when available; otherwise
        # by retrieval rank.
        items.sort(key=lambda triple: triple[2].get("turn_index", triple[0]))
        if sid == "_other_":
            lines.append(f"=== Other memory items ({len(items)}) ===")
        else:
            n_total = items[0][2].get("session_n_turns", "?")
            lines.append(
                f"=== Session {session_idx} (id={sid[:12]}{'...' if len(sid) > 12 else ''}, "
                f"{len(items)} of {n_total} turns retrieved) ==="
            )
        for _, r, meta in items:
            role = meta.get("role", "memory")
            turn_idx = meta.get("turn_index")
            prefix = f"[{role}]"
            if turn_idx is not None:
                prefix = f"[turn {turn_idx} {role}]"
            lines.append(f"{prefix} {r.content}")
        lines.append("")  # blank line between sessions
    return "\n".join(lines).rstrip()


def _generate_answer(
    chat: Any,
    *,
    memory_text: str,
    question: str,
    question_date: str,
) -> str:
    prompt = _read_prompt("answer").format(
        memory=memory_text,
        question=question,
        question_date=question_date or "(date unknown)",
    )
    messages = [Message(role="user", content=prompt)]
    return chat.chat(messages)


_OFFICIAL_JUDGE_YES_RE = re.compile(r"^\s*yes\b")
_OFFICIAL_JUDGE_NO_RE = re.compile(r"^\s*no\b")


def _parse_judge_verdict(raw: str) -> bool:
    """Mirror the official LongMemEval scorer's yes/no parsing.

    The official `evaluate_qa.py` scorer takes the first non-empty line
    of the judge response, lowercases it, and exact-matches "yes" vs
    "no" (with leading whitespace tolerated). CoT preambles like
    "Let me check ... Yes" do NOT count as yes; the verdict line must
    be the first content line. This is stricter than the prior
    substring match and removes parser-driven false positives /
    negatives across the bench (audit H-78).

    Returns True iff the first content line begins with "yes". Any
    response that begins with "no" -- or doesn't begin with either --
    returns False (judge is treated as "not confident yes").
    """
    if not raw:
        return False
    # First non-empty line is the verdict; everything after is rationale.
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if _OFFICIAL_JUDGE_YES_RE.match(lowered):
            return True
        if _OFFICIAL_JUDGE_NO_RE.match(lowered):
            return False
        # First non-empty line is not yes/no -- treat as ambiguous = no.
        return False
    return False


def _judge(
    chat: Any,
    *,
    qtype: str,
    question: str,
    gold: str,
    response: str,
) -> bool:
    # The per-qtype rubrics meaningfully differ (single-session-preference
    # accepts partial coverage, temporal-reasoning tolerates off-by-one
    # days).  Silently falling back to multi-session if the qtype is
    # unknown produces wrong-rubric scoring that's invisible in the
    # aggregate.  Raise so a future LongMemEval version adding a new
    # qtype fails the bench loudly until the rubric is wired in.
    if qtype not in _JUDGE_INSTRUCTIONS:
        raise RuntimeError(
            f"unknown LongMemEval qtype {qtype!r}; rubric must be added to "
            f"_JUDGE_INSTRUCTIONS before scoring"
        )
    instructions = _JUDGE_INSTRUCTIONS[qtype]
    prompt = _read_prompt("judge").format(
        instructions=instructions,
        question=question,
        gold=gold,
        response=response,
    )
    messages = [Message(role="user", content=prompt)]
    raw = chat.chat(messages)
    return _parse_judge_verdict(raw)


_HAYSTACK_DATE_DOW_RE = re.compile(r"\s*\([A-Za-z]+\)\s*")
_AUTO_TEMPORAL_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _build_auto_temporal_filter(question: str) -> str | None:
    """Extract year tokens from the question and build an OR-regex.

    Returns None when no year token is present (skip the filter for
    that question). Years are deduplicated -- "2023 and 2024" yields
    `\b(2023|2024)\b`. The pattern is case-insensitive at the engine
    level so no need to lower-case here.
    """
    years = sorted(set(_AUTO_TEMPORAL_YEAR_RE.findall(question)))
    if not years:
        return None
    alt = "|".join(years)
    return rf"\b({alt})\b"


def _parse_haystack_date(date_str: str) -> datetime | None:
    """Parse a LongMemEval haystack date like "2023/05/30 (Tue) 23:40".

    The day-of-week is decorative; strip it before parsing. Returns
    None when the string doesn't match the canonical shape -- the
    caller falls back to `Event`'s default `created_at = now()`.
    """
    if not date_str:
        return None
    cleaned = _HAYSTACK_DATE_DOW_RE.sub(" ", date_str).strip()
    try:
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _ingest_haystack(memory: Memory, q: _Question) -> int:
    """Batch-embed every turn from the haystack and insert into storage.

    `Memory.observe(...)` embeds and inserts ONE event at a time. For
    LongMemEval that's ~500 single-text embedder calls per question --
    sentence-transformers and OpenAI both pay a per-call overhead that
    dominates at this granularity (model warm-up on GPU, HTTP round
    trip on the API). Batching all turns in one `embedder.embed(...)`
    call cuts per-question ingestion from ~5 min to ~5 s on GPU and
    from ~30 min to ~30 s on CPU for the same haystack size.

    We bypass `Memory.observe` deliberately and call storage directly
    so the batch stays one transaction.

    Each session's haystack date is parsed into `Event.created_at` so
    the temporal-aware retrieve path (recency_lambda, as_of=...) sees
    real timestamps rather than the bench's wall-clock "now".
    """
    events: list[Event] = []
    contents: list[str] = []
    for session_idx, session in enumerate(q.haystack_sessions):
        date = q.haystack_dates[session_idx] if session_idx < len(q.haystack_dates) else ""
        session_id = (
            q.haystack_session_ids[session_idx]
            if session_idx < len(q.haystack_session_ids)
            else None
        )
        session_dt = _parse_haystack_date(date)
        # Pre-count content-bearing turns in this session so the per-turn
        # `is_last_turn` flag and `session_n_turns` are correct even when
        # some turns have empty content (skipped below).
        session_turns_with_content = [t for t in session if t.get("content")]
        session_n_turns = len(session_turns_with_content)
        turn_index = 0
        for turn in session:
            content = turn.get("content")
            if not content:
                continue
            role = turn.get("role", "unknown")
            framed = f"[{date}] [{role}] {content}" if date else f"[{role}] {content}"
            metadata: dict[str, Any] = {}
            if session_id is not None:
                # Tagging events with their source session lets the
                # ablation harness compute retrieval-level recall (did
                # the right haystack session appear in top-k?) without
                # paying for an answer LLM call. The metadata blob is
                # opaque to retrieve / rerank, so this is a free
                # observability hook with no behavior change.
                metadata["session_id"] = session_id
            # Structural position. `turn_index` is 0-based within the
            # session; `session_idx` orders sessions by haystack
            # appearance (≈ chronology when haystack_dates is set).
            # `is_first_turn` / `is_last_turn` make boundary lookups O(1)
            # without re-counting. Downstream consumers (per-session
            # diversity, within-session ranking, qtype-specific
            # heuristics) read these without re-deriving from session_id.
            metadata["turn_index"] = turn_index
            metadata["session_idx"] = session_idx
            metadata["session_n_turns"] = session_n_turns
            metadata["is_first_turn"] = turn_index == 0
            metadata["is_last_turn"] = turn_index == session_n_turns - 1
            metadata["role"] = role
            turn_index += 1
            # `has_answer` is LongMemEval's per-turn ground-truth flag:
            # True  -> this exact turn contains the answer
            # False -> in an answer session but not the answer turn
            # None  -> no label (almost always non-answer-session turn)
            # We preserve it so retrieval evals can compute event-level
            # recall ("did we surface the *actual* gold turn?") in
            # addition to the coarser session-level recall.
            ha = turn.get("has_answer")
            if ha is not None:
                metadata["has_answer"] = bool(ha)
            if session_dt is not None:
                events.append(
                    Event(
                        content=framed,
                        source=role,
                        created_at=session_dt,
                        metadata=metadata,
                    )
                )
            else:
                events.append(Event(content=framed, source=role, metadata=metadata))
            contents.append(framed)
    if not events:
        return 0
    embedder = memory.embedder
    storage = memory.storage
    vectors = embedder.embed(contents)
    with storage.transaction():
        storage.insert_events(events)
        for event, raw_vec in zip(events, vectors, strict=True):
            # L2-normalize so cosine similarity reduces to a dot product
            # in storage. Already-normalized vectors (bge-* via
            # `normalize_embeddings=True`) survive the pass intact.
            norm = math.sqrt(sum(x * x for x in raw_vec))
            vec = [x / norm for x in raw_vec] if norm > 0 else list(raw_vec)
            storage.insert_embedding(
                Embedding(
                    item_id=event.id,
                    item_kind=ItemKind.EVENT,
                    model=embedder.model,
                    dim=embedder.dim,
                    vector=tuple(vec),
                )
            )
    return len(events)


class LongMemEvalSuite:
    name: str = "longmemeval"
    dataset_version: str = "longmemeval-s-cleaned-v1"

    def __init__(
        self,
        *,
        dataset_filename: str = DEFAULT_FILENAME,
        k: int | None = None,
        max_questions: int | None = None,
    ) -> None:
        self._path = DATASET_ROOT / dataset_filename
        self.dataset_checksum: str = _checksum(self._path)
        self._k = int(os.environ.get("LONGMEMEVAL_K", k if k is not None else 10))
        env_max = os.environ.get("LONGMEMEVAL_MAX_QUESTIONS")
        self._max = (
            int(env_max) if env_max else (max_questions if max_questions is not None else None)
        )
        self._provider: Provider | None = None
        # Phase E knobs -- populated via `configure(**)` from the bench CLI.
        self._hyde: bool = False
        self._multi_query_n: int = 1
        self._decompose: bool = False
        self._temporal: bool = False
        self._surface_conflicts: bool = False
        self._reranker: Reranker | None = None
        self._cot: bool = False
        self._self_consistency_n: int = 1
        self._verify: bool = False
        self._verify_max_retries: int = 1
        self._consolidate_chat: ChatProvider | None = None
        self._consolidate: bool = False
        # Consolidation parallelism. The async path's per-cluster
        # abstraction LLM calls fan out via asyncio.gather behind a
        # semaphore; the default 8 is rate-limit-friendly but leaves
        # 5-10x wall-time on the table when the chat provider supports
        # higher concurrency (e.g. Haiku / GPT-4o-mini at 50+).
        self._aconsolidate_concurrency: int = 8
        self._judge_chat: ChatProvider | None = None
        self._seed: int | None = None
        # Sample size for stratified discovery experiments. None = full
        # dataset (subject to LONGMEMEVAL_MAX_QUESTIONS / --limit).
        self._sample_n: int | None = None
        # Parallelism for the question loop. 1 = serial (default,
        # bit-identical to the prior code path). >=2 fans out via
        # ThreadPoolExecutor; the shared embedder and chat provider
        # handle concurrent calls inside their own connection pools.
        self._parallel: int = 1
        # GPU concurrency cap (recorded for manifest reproducibility).
        # The actual enforcement lives in engram._gpu_lock; this just
        # mirrors the value into the manifest.
        self._gpu_concurrency: int = 1
        # Answer-prompt version. "v1" is the original prompt that says
        # "if you don't know, say I don't know"; "v2" adds explicit
        # abstain anchoring (state related context before saying IDK)
        # and per-qtype hints (preference synthesis, multi-session
        # aggregation scratchpad, temporal date-math, latest-value).
        # See JOURNEY §23 for the design rationale.
        self._prompt_version: str = "v1"
        # Phase F retrieve knobs (BM25 + MMR + recency + drill / pool).
        self._bm25_weight: float = 0.0
        self._mmr_lambda: float = 0.0
        self._recency_lambda: float = 0.0
        self._drill_k: int | None = None
        self._confidence_threshold: float | None = None
        self._candidate_multiplier: int | None = None
        self._bm25_k1: float = 1.5
        self._bm25_b: float = 0.75
        self._recency_decay_days: float = 90.0
        self._mmr_pool_size: int = 0
        self._recent_window_k: int = 0
        self._auto_temporal: bool = False
        # Per-session diversity floor in top-k. 0 = off.
        # Set to 3-5 to combat within-session-rank failures where the
        # cross-encoder fills top-k with similar turns from one session.
        self._min_sessions_in_topk: int = 0
        # Within-session over-sampling: promote first/last turn of each
        # session in top-k. False = off (default). Complements
        # min_sessions_in_topk.
        self._within_session_oversample: bool = False
        # Context format passed to the answerer.  "flat" is the
        # original bulleted-rank format; "grouped" groups by session
        # with explicit boundary markers + speaker labels + turn
        # indices, dropping the score/level annotations.  Default
        # "flat" for backward compatibility.
        self._context_format: str = "flat"
        # Answer-Form Forcing (AFF). When "structured", the answer
        # prompt is augmented with a JSON output instruction and the
        # response is parsed to extract just the `final_answer`
        # field. Targets the CoT-leakage failure mode in v2 where
        # Kimi echoed hint instructions instead of answering.
        # "freeform" (default) is the original path.
        self._answer_form: str = "freeform"

    def configure(
        self,
        *,
        k: int | None = None,
        seed: int | None = None,
        hyde: bool = False,
        multi_query_n: int = 1,
        decompose: bool = False,
        temporal: bool = False,
        surface_conflicts: bool = False,
        reranker: Reranker | None = None,
        cot: bool = False,
        self_consistency_n: int = 1,
        verify: bool = False,
        verify_max_retries: int = 1,
        consolidate_chat: ChatProvider | None = None,
        consolidate: bool = False,
        aconsolidate_concurrency: int = 8,
        judge_chat: ChatProvider | None = None,
        sample_n: int | None = None,
        parallel: int = 1,
        gpu_concurrency: int = 1,
        prompt_version: str = "v1",
        bm25_weight: float = 0.0,
        mmr_lambda: float = 0.0,
        recency_lambda: float = 0.0,
        drill_k: int | None = None,
        confidence_threshold: float | None = None,
        candidate_multiplier: int | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        recency_decay_days: float = 90.0,
        mmr_pool_size: int = 0,
        recent_window_k: int = 0,
        auto_temporal: bool = False,
        min_sessions_in_topk: int = 0,
        within_session_oversample: bool = False,
        context_format: str = "flat",
        answer_form: str = "freeform",
    ) -> None:
        """Wire Phase E knobs from the CLI into the suite.

        Called by `bench._runner.run` before `setup`. Every knob is
        opt-in; defaults reproduce the v0.1.0 baseline path so the
        suite stays bit-identical when no flags are passed.
        """
        if k is not None:
            self._k = k
        self._seed = seed
        if seed is not None:
            # Audit H-80: seed every RNG, not just `random`. numpy / torch /
            # torch.cuda / transformers each have their own state; without
            # this, two --seed 1337 runs produced different embeddings,
            # rerank orders, and retrieval pools even on identical code.
            from engram._seed import seed_everything

            seeded = seed_everything(seed)
            _LOG.info(
                "longmemeval: seeded RNGs %s with seed=%d",
                sorted(k for k, v in seeded.items() if v),
                seed,
            )
        self._hyde = hyde
        self._multi_query_n = multi_query_n
        self._decompose = decompose
        self._temporal = temporal
        self._surface_conflicts = surface_conflicts
        self._reranker = reranker
        self._cot = cot
        self._self_consistency_n = self_consistency_n
        self._verify = verify
        self._verify_max_retries = verify_max_retries
        self._consolidate_chat = consolidate_chat
        self._consolidate = consolidate
        if aconsolidate_concurrency < 1:
            raise ValueError(
                f"aconsolidate_concurrency must be >= 1, got {aconsolidate_concurrency}"
            )
        self._aconsolidate_concurrency = aconsolidate_concurrency
        self._judge_chat = judge_chat
        if sample_n is not None and sample_n < 1:
            raise ValueError(f"sample_n must be >= 1, got {sample_n}")
        self._sample_n = sample_n
        if parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {parallel}")
        self._parallel = parallel
        # gpu_concurrency is a recorded copy of the env-var-driven setting
        # (the CLI sets ENGRAM_GPU_CONCURRENCY before suite construction;
        # engram._gpu_lock reads it lazily). Capturing it here gets the
        # value into the manifest's engram_config for reproducibility.
        if gpu_concurrency < 1:
            raise ValueError(f"gpu_concurrency must be >= 1, got {gpu_concurrency}")
        self._gpu_concurrency = gpu_concurrency
        if prompt_version not in _PROMPT_VARIANTS:
            raise ValueError(
                f"prompt_version must be one of {sorted(_PROMPT_VARIANTS)}, "
                f"got {prompt_version!r}"
            )
        self._prompt_version = prompt_version
        self._bm25_weight = bm25_weight
        self._mmr_lambda = mmr_lambda
        self._recency_lambda = recency_lambda
        self._drill_k = drill_k
        self._confidence_threshold = confidence_threshold
        self._candidate_multiplier = candidate_multiplier
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._recency_decay_days = recency_decay_days
        self._mmr_pool_size = mmr_pool_size
        self._recent_window_k = recent_window_k
        self._auto_temporal = auto_temporal
        if min_sessions_in_topk < 0:
            raise ValueError(
                f"min_sessions_in_topk must be >= 0, got {min_sessions_in_topk}"
            )
        self._min_sessions_in_topk = min_sessions_in_topk
        self._within_session_oversample = within_session_oversample
        if context_format not in ("flat", "grouped"):
            raise ValueError(
                f"context_format must be 'flat' or 'grouped', got {context_format!r}"
            )
        self._context_format = context_format
        if answer_form not in ("freeform", "structured"):
            raise ValueError(
                f"answer_form must be 'freeform' or 'structured', got {answer_form!r}"
            )
        self._answer_form = answer_form

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        if self._provider is None:
            raise RuntimeError("setup() must be called before run()")
        embedder = getattr(self._provider, "embedder", None)
        chat = getattr(self._provider, "chat", None)
        if embedder is None or chat is None:
            raise RuntimeError(
                "longmemeval requires a Provider with both `embedder` and `chat` "
                "attributes. Use --embedder openai --chat openai (or similar)."
            )

        questions = _load_dataset(self._path)
        if not questions:
            _LOG.warning(
                "longmemeval dataset not found at %s; emitting placeholder result. "
                "Run `python scripts/fetch_longmemeval.py` first.",
                self._path,
            )
            return SuiteResult(
                name=self.name,
                aggregate_metrics={"accuracy": 0.0, "n_questions": 0.0},
                confidence_intervals={"accuracy": (0.0, 0.0)},
                per_question=[],
                latency_ms={},
            )
        # `--sample N` takes precedence over `--limit M` -- a stratified
        # sample is rarely what the caller wants AND a leading slice
        # both. Warn if both are set so the manifest tells the story.
        if self._sample_n is not None:
            if self._max is not None:
                _LOG.warning(
                    "longmemeval: --sample %d overrides --limit %d "
                    "(taking a stratified sample of the full dataset)",
                    self._sample_n,
                    self._max,
                )
            questions = _stratified_sample(
                questions, self._sample_n, seed=self._seed
            )
            _LOG.info(
                "longmemeval: stratified sample of %d questions (seed=%s)",
                len(questions),
                self._seed,
            )
        elif self._max is not None:
            questions = questions[: self._max]

        # Log every question for small smoke runs; every 10 for full runs
        # so 500-question manifests don't bury the user in INFO lines.
        log_interval = 1 if len(questions) <= 20 else 10

        _LOG.info(
            "longmemeval: starting %d questions (k=%d, %d turn avg, cap=%s, parallel=%d)",
            len(questions),
            self._k,
            sum(len(s) for q in questions for s in q.haystack_sessions) // max(len(questions), 1),
            self._max if self._max is not None else "none",
            self._parallel,
        )

        # Pre-build the per-question RetrieveParams: Phase E flags only
        # have effect if Memory's defaults (or the per-call kwargs) say
        # so. Storing the params once outside the question loop keeps
        # the construction allocation-light.
        base_params_kwargs: dict[str, Any] = {
            "k": self._k,
            "hyde": self._hyde,
            "multi_query_n": self._multi_query_n,
            "decompose": self._decompose,
            "temporal": self._temporal,
            "surface_conflicts": self._surface_conflicts,
            "bm25_weight": self._bm25_weight,
            "mmr_lambda": self._mmr_lambda,
            "recency_lambda": self._recency_lambda,
            "bm25_k1": self._bm25_k1,
            "bm25_b": self._bm25_b,
            "recency_decay_days": self._recency_decay_days,
            "mmr_pool_size": self._mmr_pool_size,
            "recent_window_k": self._recent_window_k,
            "min_sessions_in_topk": self._min_sessions_in_topk,
            "within_session_oversample": self._within_session_oversample,
        }
        # `context_format` is a bench-side rendering choice, not a
        # RetrieveParams field. The suite reads `self._context_format`
        # directly when formatting memory for the answerer.
        if self._drill_k is not None:
            base_params_kwargs["drill_k"] = self._drill_k
        if self._confidence_threshold is not None:
            base_params_kwargs["confidence_threshold"] = self._confidence_threshold
        if self._candidate_multiplier is not None:
            base_params_kwargs["candidate_multiplier"] = self._candidate_multiplier
        default_retrieve_params = RetrieveParams(**base_params_kwargs)
        judge_chat = self._judge_chat if self._judge_chat is not None else chat

        progress = _Progress(total=len(questions), log_interval=log_interval)
        outcomes: list[_QuestionOutcome | None] = [None] * len(questions)

        def _run(q_idx: int, q: _Question) -> tuple[int, _QuestionOutcome]:
            outcome = self._run_one_question(
                q,
                q_idx=q_idx,
                total=len(questions),
                embedder=embedder,
                chat=chat,
                judge_chat=judge_chat,
                default_retrieve_params=default_retrieve_params,
            )
            progress.mark(outcome)
            return q_idx, outcome

        if self._parallel <= 1:
            # Serial path. Bit-identical to the prior code path for any
            # `--parallel 1` (and the default).
            for q_idx, q in enumerate(questions):
                idx, outcome = _run(q_idx, q)
                outcomes[idx] = outcome
        else:
            # Parallel path. Each worker owns its SqliteStorage (created
            # inside `_run_one_question`). The shared embedder, chat,
            # judge_chat, and reranker must be thread-safe -- this is
            # already true for sentence-transformers, OpenAI / Anthropic
            # SDK clients, BGEReranker, and the disk cache (which holds
            # its own internal lock).
            with ThreadPoolExecutor(max_workers=self._parallel) as ex:
                fut_to_idx = {
                    ex.submit(_run, q_idx, q): q_idx
                    for q_idx, q in enumerate(questions)
                }
                for fut in as_completed(fut_to_idx):
                    idx, outcome = fut.result()
                    outcomes[idx] = outcome

        # Assemble per-question manifest entries and latency arrays in
        # original question order. The shape matches the prior code
        # path so downstream tooling reads the manifest the same way.
        per_question: list[dict[str, Any]] = []
        retrieve_ms: list[float] = []
        answer_ms: list[float] = []
        judge_ms: list[float] = []
        ingest_ms_list: list[float] = []
        per_type_scores: dict[str, list[float]] = {}
        for oc in outcomes:
            if oc is None:
                # Defensive: ThreadPoolExecutor / serial loop fill every
                # slot before this aggregation runs. A None here would
                # be a code bug, not a runtime condition.
                raise RuntimeError("longmemeval: outcome slot left unfilled")
            per_type_scores.setdefault(oc.qtype, []).append(oc.score)
            entry: dict[str, Any] = {
                "question_id": oc.qid,
                "question_type": oc.qtype,
                "question": oc.question,
                "gold": oc.gold,
                "response": oc.response,
                "score": oc.score,
                "k": oc.k,
                "turns_ingested": oc.turns_ingested,
                "consolidate_ms": oc.consolidate_ms,
            }
            if oc.error_msg is not None:
                entry["error"] = oc.error_msg
            per_question.append(entry)
            ingest_ms_list.append(oc.ingest_ms)
            retrieve_ms.append(oc.retrieve_ms)
            answer_ms.append(oc.answer_ms)
            judge_ms.append(oc.judge_ms)

        # Audit H-77: errored questions (content-filter rejections, 429s,
        # network blips) used to score 0 and be conflated with wrong-but-
        # completed answers. Split them out:
        #   `accuracy`           -- legacy denominator (n_questions), kept
        #                           for back-compat with prior manifests.
        #   `accuracy_correct`   -- correct / n_completed; the honest
        #                           number to compare across runs.
        #   `n_errored`          -- how many entries carry an `error` key.
        #   `error_rate`         -- n_errored / n_questions.
        # Per-qtype variants land too.
        flat = [s for vals in per_type_scores.values() for s in vals]
        accuracy = sum(flat) / len(flat) if flat else 0.0
        per_type_errored: dict[str, int] = dict.fromkeys(per_type_scores, 0)
        n_errored_total = 0
        for entry in per_question:
            if "error" in entry:
                qt = entry.get("question_type", "")
                per_type_errored[qt] = per_type_errored.get(qt, 0) + 1
                n_errored_total += 1

        def _safe_div(num: float, den: float) -> float:
            return num / den if den > 0 else 0.0

        n_total = len(flat)
        n_completed_total = n_total - n_errored_total
        correct_total = sum(flat)
        accuracy_correct = _safe_div(correct_total, float(n_completed_total))

        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "accuracy_correct": accuracy_correct,
            "n_questions": float(n_total),
            "n_completed": float(n_completed_total),
            "n_errored": float(n_errored_total),
            "error_rate": _safe_div(float(n_errored_total), float(n_total)),
            "k": float(self._k),
        }
        if n_errored_total > 0 and n_total > 0 and n_errored_total / n_total > 0.01:
            _LOG.warning(
                "longmemeval: %d/%d questions errored (%.1f%%); "
                "use accuracy_correct (=%.3f over %d completed) instead of "
                "accuracy (=%.3f over %d total) for cross-run comparison",
                n_errored_total,
                n_total,
                n_errored_total / n_total * 100.0,
                accuracy_correct,
                n_completed_total,
                accuracy,
                n_total,
            )
        for qtype, scores in per_type_scores.items():
            if not scores:
                continue
            qt_correct = sum(scores)
            qt_total = len(scores)
            qt_errored = per_type_errored.get(qtype, 0)
            qt_completed = qt_total - qt_errored
            metrics[f"accuracy_{qtype}"] = _safe_div(qt_correct, float(qt_total))
            metrics[f"accuracy_correct_{qtype}"] = _safe_div(
                qt_correct, float(qt_completed)
            )
            metrics[f"n_{qtype}"] = float(qt_total)
            metrics[f"n_completed_{qtype}"] = float(qt_completed)
            metrics[f"n_errored_{qtype}"] = float(qt_errored)
        cis: dict[str, tuple[float, float]] = {k: (v, v) for k, v in metrics.items()}
        return SuiteResult(
            name=self.name,
            aggregate_metrics=metrics,
            confidence_intervals=cis,
            per_question=per_question,
            latency_ms={
                "retrieve": retrieve_ms,
                "answer": answer_ms,
                "judge": judge_ms,
                "ingest": ingest_ms_list,
            },
        )

    def _run_one_question(
        self,
        q: _Question,
        *,
        q_idx: int,
        total: int,
        embedder: Any,
        chat: Any,
        judge_chat: Any,
        default_retrieve_params: RetrieveParams,
    ) -> _QuestionOutcome:
        """Run the ingest + retrieve + answer + judge pipeline for one
        question, with full per-question exception isolation.

        Thread-safe by construction: creates its own SqliteStorage,
        builds a fresh `Memory` against the shared embedder/chat, and
        returns a value rather than mutating shared state. The shared
        providers must themselves be thread-safe (sentence-transformers,
        OpenAI SDK, Anthropic SDK, BGEReranker, DiskCache all are).

        Wraps the whole body in `try/except Exception`: any failure
        (content-filter rejection, 429, network blip, parse error,
        defensive RuntimeError from the engine) scores the question as
        0 and surfaces the exception message in
        `_QuestionOutcome.error_msg`. The outer loop's aggregation
        treats those as `n_errored` and excludes them from
        `accuracy_correct` (audit H-77).

        KeyboardInterrupt and SystemExit are NOT swallowed so Ctrl+C
        still aborts the run cleanly.
        """
        ingest_ms = 0.0
        consolidate_ms = 0.0
        retrieve_ms_v = 0.0
        answer_ms_v = 0.0
        judge_ms_v = 0.0
        turns = 0
        response = ""
        score = 0.0
        error_msg: str | None = None

        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            try:
                memory = Memory(
                    storage=storage,
                    embedder=embedder,
                    chat=chat,
                    consolidate_chat=self._consolidate_chat,
                    retrieve_params=default_retrieve_params,
                    reranker=self._reranker,
                )
                t_ingest = time.perf_counter()
                turns = _ingest_haystack(memory, q)
                ingest_ms = (time.perf_counter() - t_ingest) * 1000.0

                if self._consolidate:
                    t_consolidate = time.perf_counter()
                    try:
                        cons_result = asyncio.run(
                            memory.aconsolidate(
                                max_concurrent_abstractions=self._aconsolidate_concurrency,
                            )
                        )
                        consolidate_ms = (time.perf_counter() - t_consolidate) * 1000.0
                        _LOG.info(
                            "  q %d/%d consolidated: %d clusters -> %d abstractions in %.1fs",
                            q_idx + 1,
                            total,
                            getattr(cons_result, "clusters_formed", 0),
                            getattr(cons_result, "abstractions_created", 0),
                            consolidate_ms / 1000.0,
                        )
                    except Exception as exc:
                        _LOG.warning(
                            "  consolidate failed for q %d/%d: %s",
                            q_idx + 1,
                            total,
                            exc,
                        )

                t0 = time.perf_counter()
                retrieve_kwargs: dict[str, Any] = {
                    "k": self._k,
                    "reinforce": False,
                }
                if self._recency_lambda > 0:
                    question_dt = _parse_haystack_date(q.question_date)
                    if question_dt is not None:
                        retrieve_kwargs["as_of"] = question_dt
                if self._auto_temporal:
                    filt = _build_auto_temporal_filter(q.question)
                    if filt:
                        retrieve_kwargs["lexical_filter"] = filt
                results = memory.retrieve(q.question, **retrieve_kwargs)
                if (
                    self._auto_temporal
                    and not results
                    and retrieve_kwargs.get("lexical_filter")
                ):
                    retrieve_kwargs.pop("lexical_filter", None)
                    results = memory.retrieve(q.question, **retrieve_kwargs)
                retrieve_ms_v = (time.perf_counter() - t0) * 1000.0

                memory_text = self._format_memory_for_answer(results, storage)

                t0 = time.perf_counter()
                response = self._answer_with_phase_e(
                    chat=chat,
                    memory=memory,
                    memory_text=memory_text,
                    question=q.question,
                    question_date=q.question_date,
                    qtype=q.qtype,
                )
                answer_ms_v = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                correct = _judge(
                    judge_chat,
                    qtype=q.qtype,
                    question=q.question,
                    gold=q.gold,
                    response=response,
                )
                judge_ms_v = (time.perf_counter() - t0) * 1000.0

                score = 1.0 if correct else 0.0
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                _LOG.warning(
                    "q %d/%d [%s] -> ERROR (%s)",
                    q_idx + 1,
                    total,
                    q.qtype,
                    error_msg,
                )
                score = 0.0
        finally:
            storage.close()

        return _QuestionOutcome(
            qid=q.qid,
            qtype=q.qtype,
            question=q.question,
            gold=q.gold,
            response=response,
            score=score,
            k=self._k,
            turns_ingested=turns,
            ingest_ms=ingest_ms,
            consolidate_ms=consolidate_ms,
            retrieve_ms=retrieve_ms_v,
            answer_ms=answer_ms_v,
            judge_ms=judge_ms_v,
            error_msg=error_msg,
        )

    def _format_memory_for_answer(
        self, results: Sequence[Any], storage: Any
    ) -> str:
        """Dispatch on `self._context_format`. 'flat' is bit-identical
        to the legacy `_format_memory`; 'grouped' uses session-grouped
        formatting with metadata."""
        if self._context_format == "grouped":
            return _format_memory_grouped(results, storage)
        return _format_memory(results)

    def _aff_suffix(self) -> str:
        """Answer-Form Forcing JSON suffix for the answer prompt.

        When --answer-form structured, the model is told to output a
        single JSON object with one key: `final_answer`.  The parser
        in `_extract_aff_answer` strips the JSON wrapper before
        scoring.  This blocks CoT-leakage at the output layer: even
        if the model wants to think out loud, it must commit to a
        terminal answer inside the JSON, and the rest is discarded.
        """
        if self._answer_form != "structured":
            return ""
        return (
            "\n\nIMPORTANT: Output ONLY a single JSON object with one key, "
            "no preamble or markdown fences:\n"
            '{"final_answer": "<your concise answer here>"}\n'
        )

    @staticmethod
    def _extract_aff_answer(raw: str) -> str:
        """Pull `final_answer` out of an AFF JSON response.

        Tolerant to common LLM-output deviations:
          - markdown code fences (```json ... ```)
          - leading/trailing prose
          - missing closing brace
        Falls back to the raw response when parsing fails entirely,
        so AFF is monotonic (never worse than freeform).
        """
        if not raw:
            return raw
        text = raw.strip()
        # Strip markdown fences if present.
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline > -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        # Locate the first { and try to parse a JSON object.
        first_brace = text.find("{")
        if first_brace == -1:
            return raw
        last_brace = text.rfind("}")
        if last_brace > first_brace:
            payload = text[first_brace : last_brace + 1]
        else:
            payload = text[first_brace:] + "}"  # tolerate truncation
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return raw
        if isinstance(data, dict) and "final_answer" in data:
            return str(data["final_answer"])
        return raw

    def _answer_with_phase_e(
        self,
        *,
        chat: Any,
        memory: Memory,
        memory_text: str,
        question: str,
        question_date: str,
        qtype: str = "",
    ) -> str:
        """Answer step with optional CoT / self-consistency / verify.

        When every Phase E agent flag is at its default and
        prompt_version="v1", this collapses to a single `_generate_answer`
        call -- bit-identical to the v0.1.0 path. Each opt-in adds work
        in this fixed order:

          * `prompt_version="v2"`: switch to the abstain-anchored
            template and append per-qtype hints from `_V2_QTYPE_HINTS`.
            Targets the failure patterns identified in JOURNEY §23
            (abstain form, preference synthesis, temporal date-math,
            multi-session aggregation, knowledge-update latest-value).
          * `cot`: append a CoT instruction to the answer prompt and
            strip the reasoning prefix from the reply.
          * `self_consistency_n>=2`: take N samples and majority-vote
            (over the post-CoT-strip answer when `cot` is on).
          * `verify`: re-run the verifier; on unsupported, re-retrieve
            and re-answer up to `verify_max_retries` times.
        """
        template_version, qtype_hint_map = _PROMPT_VARIANTS[self._prompt_version]
        qtype_hint = qtype_hint_map.get(qtype, "")
        base_prompt = _read_prompt("answer", version=template_version).format(
            memory=memory_text,
            question=question,
            question_date=question_date or "(date unknown)",
            qtype_hint=qtype_hint,
        )
        cot_suffix = (
            "\n\nFirst think step-by-step about which memories are "
            "relevant. Then write 'Answer:' on a new line followed by "
            "the final answer only."
            if self._cot
            else ""
        )
        aff_suffix = self._aff_suffix()

        def _one_call(prompt_text: str) -> str:
            raw: str = chat.chat([Message(role="user", content=prompt_text)])
            if self._cot:
                raw = _strip_cot(raw)
            if self._answer_form == "structured":
                raw = self._extract_aff_answer(raw)
            return raw

        prompt = base_prompt + cot_suffix + aff_suffix
        if self._self_consistency_n >= 2:
            samples = tuple(_one_call(prompt) for _ in range(self._self_consistency_n))
            response = _majority_vote(samples)
        else:
            response = _one_call(prompt)

        if self._verify:
            current_memory_text = memory_text
            for attempt in range(self._verify_max_retries + 1):
                verdict = verify_answer(
                    question=question,
                    context=current_memory_text,
                    answer=response,
                    chat=chat,
                    max_retries=0,
                )
                if verdict.supported or attempt == self._verify_max_retries:
                    break
                # Re-retrieve (Memory may have updated state between
                # turns) and re-answer. Same retrieve config + prompt
                # version as the first attempt; ReAct-style refinement
                # is the `retrieve_iterative` job, not the verifier's.
                fresh = memory.retrieve(question, k=self._k, reinforce=False)
                current_memory_text = self._format_memory_for_answer(
                    fresh, memory.storage
                )
                fresh_prompt = _read_prompt(
                    "answer", version=template_version
                ).format(
                    memory=current_memory_text,
                    question=question,
                    question_date=question_date or "(date unknown)",
                    qtype_hint=qtype_hint,
                ) + cot_suffix + aff_suffix
                response = _one_call(fresh_prompt)

        return response

    def teardown(self) -> None:
        self._provider = None


SUITE: LongMemEvalSuite = LongMemEvalSuite()
