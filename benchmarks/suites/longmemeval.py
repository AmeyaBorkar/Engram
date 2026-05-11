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
        self._judge_chat: ChatProvider | None = None
        self._seed: int | None = None

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
        judge_chat: ChatProvider | None = None,
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
            import random

            random.seed(seed)
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
        self._judge_chat = judge_chat

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

        # Pre-build the per-question RetrieveParams: Phase E flags only
        # have effect if Memory's defaults (or the per-call kwargs) say
        # so. Storing the params once outside the question loop keeps
        # the construction allocation-light.
        default_retrieve_params = RetrieveParams(
            k=self._k,
            hyde=self._hyde,
            multi_query_n=self._multi_query_n,
            decompose=self._decompose,
            temporal=self._temporal,
            surface_conflicts=self._surface_conflicts,
        )
        judge_chat = self._judge_chat if self._judge_chat is not None else chat
        for q_idx, q in enumerate(questions):
            storage = SqliteStorage(":memory:")
            storage.initialize()
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

                # Optional consolidation pass (Engram's novelty):
                # cluster the haystack events and abstract each cluster
                # into a Level.SUMMARY MemoryItem. The retrieve below
                # then surfaces SUMMARY hits alongside raw EVENTs --
                # higher-level memories outrank single turns when the
                # confidence is high. Cost: ~1 chat call per cluster.
                consolidate_ms = 0.0
                if self._consolidate:
                    t_consolidate = time.perf_counter()
                    try:
                        cons_result = memory.consolidate()
                        consolidate_ms = (time.perf_counter() - t_consolidate) * 1000.0
                        _LOG.info(
                            "  consolidated: %d clusters -> %d abstractions in %.1fs",
                            getattr(cons_result, "clusters_formed", 0),
                            getattr(cons_result, "abstractions_created", 0),
                            consolidate_ms / 1000.0,
                        )
                    except Exception as exc:
                        _LOG.warning(
                            "  consolidate failed for q %d/%d: %s",
                            q_idx + 1,
                            len(questions),
                            exc,
                        )

                t0 = time.perf_counter()
                # Per-call kwargs pick up Memory's defaults set above.
                # Explicit reinforce=False keeps decay state unperturbed
                # so a re-run of the same suite is bit-identical.
                results = memory.retrieve(q.question, k=self._k, reinforce=False)
                retrieve_ms.append((time.perf_counter() - t0) * 1000.0)

                memory_text = _format_memory(results)

                t0 = time.perf_counter()
                response = self._answer_with_phase_e(
                    chat=chat,
                    memory=memory,
                    memory_text=memory_text,
                    question=q.question,
                    question_date=q.question_date,
                )
                answer_ms.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                correct = _judge(
                    judge_chat,
                    qtype=q.qtype,
                    question=q.question,
                    gold=q.gold,
                    response=response,
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
                        "consolidate_ms": consolidate_ms,
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

    def _answer_with_phase_e(
        self,
        *,
        chat: Any,
        memory: Memory,
        memory_text: str,
        question: str,
        question_date: str,
    ) -> str:
        """Answer step with optional CoT / self-consistency / verify.

        When every Phase E agent flag is at its default, this collapses
        to a single `_generate_answer` call -- bit-identical to the
        v0.1.0 path. Each opt-in adds work in this fixed order:

          * `cot`: append a CoT instruction to the answer prompt and
            strip the reasoning prefix from the reply.
          * `self_consistency_n>=2`: take N samples and majority-vote
            (over the post-CoT-strip answer when `cot` is on).
          * `verify`: re-run the verifier; on unsupported, re-retrieve
            and re-answer up to `verify_max_retries` times.
        """
        base_prompt = _read_prompt("answer").format(
            memory=memory_text,
            question=question,
            question_date=question_date or "(date unknown)",
        )
        cot_suffix = (
            "\n\nFirst think step-by-step about which memories are "
            "relevant. Then write 'Answer:' on a new line followed by "
            "the final answer only."
            if self._cot
            else ""
        )

        def _one_call(prompt_text: str) -> str:
            raw: str = chat.chat([Message(role="user", content=prompt_text)])
            return _strip_cot(raw) if self._cot else raw

        prompt = base_prompt + cot_suffix
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
                # turns) and re-answer. Same retrieve config as the
                # first attempt; ReAct-style refinement is the
                # `retrieve_iterative` job, not the verifier's.
                fresh = memory.retrieve(question, k=self._k, reinforce=False)
                current_memory_text = _format_memory(fresh)
                fresh_prompt = _read_prompt("answer").format(
                    memory=current_memory_text,
                    question=question,
                    question_date=question_date or "(date unknown)",
                ) + cot_suffix
                response = _one_call(fresh_prompt)

        return response

    def teardown(self) -> None:
        self._provider = None


SUITE: LongMemEvalSuite = LongMemEvalSuite()
