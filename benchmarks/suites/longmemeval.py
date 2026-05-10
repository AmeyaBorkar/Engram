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

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engram import Memory, SqliteStorage
from engram.bench import Provider, SuiteResult
from engram.providers._message import Message
from engram.schemas import Embedding, Event, ItemKind

_LOG = logging.getLogger("engram.bench.longmemeval")

DATASET_ROOT = Path("benchmarks/datasets/longmemeval")
DEFAULT_FILENAME = "longmemeval_s_cleaned.json"
PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_VERSIONS: dict[str, str] = {
    "answer": "v1",
    "judge": "v1",
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


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"longmemeval_{name}_v1.txt").read_text(encoding="utf-8")


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


def _judge(
    chat: Any,
    *,
    qtype: str,
    question: str,
    gold: str,
    response: str,
) -> bool:
    instructions = _JUDGE_INSTRUCTIONS.get(qtype, _JUDGE_INSTRUCTIONS["multi-session"])
    prompt = _read_prompt("judge").format(
        instructions=instructions,
        question=question,
        gold=gold,
        response=response,
    )
    messages = [Message(role="user", content=prompt)]
    raw = chat.chat(messages).strip().lower()
    # The official scorer uses substring match for "yes" -- we mirror it
    # so verdicts are comparable across runs.
    return "yes" in raw and "no" not in raw.split("yes", 1)[0]


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
    """
    events: list[Event] = []
    contents: list[str] = []
    for session_idx, session in enumerate(q.haystack_sessions):
        date = q.haystack_dates[session_idx] if session_idx < len(q.haystack_dates) else ""
        for turn in session:
            content = turn.get("content")
            if not content:
                continue
            role = turn.get("role", "unknown")
            framed = f"[{date}] [{role}] {content}" if date else f"[{role}] {content}"
            events.append(Event(content=framed, source=role))
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
        if self._max is not None:
            questions = questions[: self._max]

        per_question: list[dict[str, Any]] = []
        retrieve_ms: list[float] = []
        answer_ms: list[float] = []
        judge_ms: list[float] = []
        per_type_scores: dict[str, list[float]] = {}
        # Log every question for small smoke runs; every 10 for full runs
        # so 500-question manifests don't bury the user in INFO lines.
        log_interval = 1 if len(questions) <= 20 else 10

        _LOG.info(
            "longmemeval: starting %d questions (k=%d, %d turn avg, cap=%s)",
            len(questions),
            self._k,
            sum(len(s) for q in questions for s in q.haystack_sessions) // max(len(questions), 1),
            self._max if self._max is not None else "none",
        )

        for q_idx, q in enumerate(questions):
            storage = SqliteStorage(":memory:")
            storage.initialize()
            try:
                memory = Memory(storage=storage, embedder=embedder, chat=chat)
                t_ingest = time.perf_counter()
                turns = _ingest_haystack(memory, q)
                ingest_ms = (time.perf_counter() - t_ingest) * 1000.0

                t0 = time.perf_counter()
                results = memory.retrieve(q.question, k=self._k, reinforce=False)
                retrieve_ms.append((time.perf_counter() - t0) * 1000.0)

                memory_text = _format_memory(results)

                t0 = time.perf_counter()
                response = _generate_answer(
                    chat,
                    memory_text=memory_text,
                    question=q.question,
                    question_date=q.question_date,
                )
                answer_ms.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                correct = _judge(
                    chat, qtype=q.qtype, question=q.question, gold=q.gold, response=response
                )
                judge_ms.append((time.perf_counter() - t0) * 1000.0)

                score = 1.0 if correct else 0.0
                per_type_scores.setdefault(q.qtype, []).append(score)
                per_question.append(
                    {
                        "question_id": q.qid,
                        "question_type": q.qtype,
                        "question": q.question,
                        "gold": q.gold,
                        "response": response,
                        "score": score,
                        "k": self._k,
                        "turns_ingested": turns,
                    }
                )
                if (q_idx + 1) % log_interval == 0 or q_idx == len(questions) - 1:
                    correct_so_far = sum(s for vs in per_type_scores.values() for s in vs)
                    total = sum(len(vs) for vs in per_type_scores.values())
                    _LOG.info(
                        "q %d/%d [%s] -> %s "
                        "(ingest %d turns in %.1fs, ans %.1fs, jud %.1fs; acc=%.3f)",
                        q_idx + 1,
                        len(questions),
                        q.qtype,
                        "PASS" if score == 1.0 else "FAIL",
                        turns,
                        ingest_ms / 1000.0,
                        answer_ms[-1] / 1000.0,
                        judge_ms[-1] / 1000.0,
                        correct_so_far / total if total else 0.0,
                    )
            finally:
                storage.close()

        flat = [s for vals in per_type_scores.values() for s in vals]
        accuracy = sum(flat) / len(flat) if flat else 0.0
        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "n_questions": float(len(flat)),
            "k": float(self._k),
        }
        for qtype, scores in per_type_scores.items():
            if scores:
                metrics[f"accuracy_{qtype}"] = sum(scores) / len(scores)
                metrics[f"n_{qtype}"] = float(len(scores))
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
            },
        )

    def teardown(self) -> None:
        self._provider = None


SUITE: LongMemEvalSuite = LongMemEvalSuite()
