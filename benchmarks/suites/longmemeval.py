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
import time
from collections.abc import Sequence
from dataclasses import dataclass
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


def _seed_everything(seed: int) -> None:
    """Seed every RNG that affects retrieval / reranking determinism.

    Pre-audit `--seed` only seeded stdlib `random`, which is unused on
    the hot retrieve path. NumPy (bootstrap), Torch (BGE reranker +
    sentence-transformers), and HuggingFace Transformers all carry
    their own RNGs that decide GPU sampling kernels, dropout, and
    tokenizer fallbacks. Seeding them all is the only way `--seed N`
    means "rerun is bit-identical" rather than "rerun's bootstrap CI
    is bit-identical."
    """
    import random as _random

    _random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is optional (bench extra); fine to skip when absent.
        pass
    try:
        from transformers import set_seed as _set_seed  # type: ignore[import-not-found]

        _set_seed(seed)
    except ImportError:
        pass


def _bootstrap_ci(values: list[float], *, seed: int = 1337) -> tuple[float, float]:
    """Bootstrap a 95% CI on the mean. Empty / single-value -> (mean, mean).

    Used to back the suite's `confidence_intervals` field with real
    bounds instead of the pre-audit zero-width `(v, v)` placeholder.
    Local copy of `scripts/_stats.bootstrap_mean_ci` so the suite has
    no script-layer dependency.
    """
    if not values:
        return (0.0, 0.0)
    try:
        import numpy as _np
    except ImportError:  # pragma: no cover - numpy is a core dep
        m = sum(values) / len(values)
        return (m, m)
    arr = _np.asarray(values, dtype=_np.float64)
    mean = float(arr.mean())
    if arr.size < 2:
        return (mean, mean)
    rng = _np.random.default_rng(seed)
    n_iters = 5000
    idx = rng.integers(0, arr.size, size=(n_iters, arr.size))
    resamples = arr[idx].mean(axis=1)
    lo = float(_np.quantile(resamples, 0.025))
    hi = float(_np.quantile(resamples, 0.975))
    return (lo, hi)


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
    raw = chat.chat(messages)
    return _judge_parse(raw)


def _judge_parse(raw: str) -> bool:
    """Parse a judge reply into a bool, matching the official LongMemEval scorer.

    The official scorer (`evaluate_qa.py` in the LongMemEval repo)
    strips whitespace, takes the first line, lowercases, and demands
    exact match against "yes". Anything else, including CoT prefaces
    that contain "yes" inside a sentence, scores 0. The previous
    substring search (`"yes" in raw and "no" not in raw.split("yes")[0]`)
    accepted noise the official scorer rejects -- diverged verdicts
    silently inflated our reported accuracy vs published numbers.
    """
    if not raw:
        return False
    first = raw.strip().splitlines()
    if not first:
        return False
    return first[0].strip().lower() == "yes"


_HAYSTACK_DATE_DOW_RE = re.compile(r"\s*\([A-Za-z]+\)\s*")
_AUTO_TEMPORAL_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _build_auto_temporal_filter(question: str) -> str | None:
    """Extract year tokens from the question and build an OR-regex.

    Returns None when no year token is present (skip the filter for
    that question). Years are deduplicated -- "2023 and 2024" yields
    `\b(2023|2024)\b`. The pattern is case-insensitive at the engine
    level so no need to lower-case here.
    """
    years = sorted({m for m in _AUTO_TEMPORAL_YEAR_RE.findall(question)})
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
    # Chunk embed calls to bound peak memory: at ~500 turns and
    # ~4096-dim embeddings, the full materialised result is ~50 MB of
    # Python floats per question -- a real cliff when the harness is
    # already holding the haystack content + reranker model in RAM.
    # 512 items per chunk is a sweet spot for sentence-transformers
    # (saturates GPU batch throughput) and OpenAI (one request well
    # below their per-call token cap).
    chunk_size = 512
    vectors: list[Any] = []
    for start in range(0, len(contents), chunk_size):
        end = start + chunk_size
        vectors.extend(embedder.embed(contents[start:end]))
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
        # Constructor defaults: do NOT read env here. The module-level
        # `SUITE = LongMemEvalSuite()` is created at import time; reading
        # env at import time means in-process re-imports see stale env
        # values (and `--limit` overrides are masked by an earlier
        # invocation's env). Env reads happen in `run()` instead.
        self._ctor_k = k
        self._ctor_max = max_questions
        self._k = k if k is not None else 10
        self._max = max_questions
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

    def configure(
        self,
        *,
        max_questions: int | None = None,
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
    ) -> None:
        """Wire Phase E knobs from the CLI into the suite.

        Called by `bench._runner.run` before `setup`. Every knob is
        opt-in; defaults reproduce the v0.1.0 baseline path so the
        suite stays bit-identical when no flags are passed.
        """
        if max_questions is not None:
            self._max = max_questions
        if k is not None:
            self._k = k
        self._seed = seed
        if seed is not None:
            _seed_everything(seed)
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

        # Read env vars at run() time, not import time: the SUITE singleton
        # is built at module import, which happens before the bench CLI
        # parses --limit. Reading here also lets in-process re-runs pick
        # up changed env between invocations.
        env_max = os.environ.get("LONGMEMEVAL_MAX_QUESTIONS")
        if env_max:
            # Env overrides ctor only when configure(max_questions=...)
            # hasn't already supplied an explicit cap. The audit's H-74
            # fix prefers suite_config (i.e. CLI --limit) over env, so
            # only apply env when no explicit configure() value is set.
            if self._max is None:
                self._max = int(env_max)
        env_k = os.environ.get("LONGMEMEVAL_K")
        if env_k and self._ctor_k is None:
            # Only apply env when ctor didn't set k AND configure() didn't.
            # (configure() always sets self._k when k is non-None, but the
            # ctor stores the original here for the env-vs-default tiebreak.)
            self._k = int(env_k)

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
        ingest_ms_all: list[float] = []
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
        }
        if self._drill_k is not None:
            base_params_kwargs["drill_k"] = self._drill_k
        if self._confidence_threshold is not None:
            base_params_kwargs["confidence_threshold"] = self._confidence_threshold
        if self._candidate_multiplier is not None:
            base_params_kwargs["candidate_multiplier"] = self._candidate_multiplier
        default_retrieve_params = RetrieveParams(**base_params_kwargs)
        judge_chat = self._judge_chat if self._judge_chat is not None else chat
        for q_idx, q in enumerate(questions):
            storage = SqliteStorage(":memory:")
            storage.initialize()
            try:
                self._run_one_question(
                    q,
                    q_idx=q_idx,
                    questions=questions,
                    storage=storage,
                    embedder=embedder,
                    chat=chat,
                    judge_chat=judge_chat,
                    default_retrieve_params=default_retrieve_params,
                    per_question=per_question,
                    per_type_scores=per_type_scores,
                    retrieve_ms=retrieve_ms,
                    answer_ms=answer_ms,
                    judge_ms=judge_ms,
                    ingest_ms_all=ingest_ms_all,
                    log_interval=log_interval,
                )
            finally:
                storage.close()

        flat = [s for vals in per_type_scores.values() for s in vals]
        # Track errored vs completed separately. Pre-audit, infra errors
        # (content-filter rejections, 429s, transient network blips)
        # scored as 0 and were indistinguishable from genuine wrong
        # answers in the headline `accuracy`, silently contaminating
        # SOTA comparisons. Now we expose both `accuracy` (scores / N,
        # the harshest "errors are failures" reading) AND
        # `accuracy_correct` (correct / n_completed, the official
        # LongMemEval reading that drops infra failures).
        n_errored = sum(1 for e in per_question if e.get("error"))
        n_total = len(per_question)
        n_completed = n_total - n_errored
        # `accuracy` keeps the pre-audit reading (errored questions
        # count as wrong) so existing SCOREBOARD claims stay comparable
        # while we transition. `accuracy_correct` is the
        # comparable-to-published-LongMemEval reading: drop errored
        # rows from BOTH numerator and denominator, so the metric isn't
        # contaminated by transient 429s / content-filter rejections.
        completed_scores = [
            float(e.get("score", 0.0))
            for e in per_question
            if not e.get("error")
        ]
        accuracy = sum(flat) / len(flat) if flat else 0.0
        accuracy_correct = (
            sum(completed_scores) / len(completed_scores)
            if completed_scores
            else 0.0
        )
        if n_total > 0 and n_errored / n_total > 0.01:
            _LOG.warning(
                "longmemeval: %d/%d questions errored (%.1f%%); "
                "see per-question `error` fields and use `accuracy_correct` "
                "for the comparable-to-published number.",
                n_errored,
                n_total,
                100.0 * n_errored / n_total,
            )
        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "accuracy_correct": accuracy_correct,
            "n_questions": float(len(flat)),
            "n_completed": float(n_completed),
            "n_errored": float(n_errored),
            "k": float(self._k),
        }
        for qtype, scores in per_type_scores.items():
            if scores:
                metrics[f"accuracy_{qtype}"] = sum(scores) / len(scores)
                metrics[f"n_{qtype}"] = float(len(scores))
        # Real bootstrap CIs on the bounded-mean metrics. Pre-audit these
        # were zero-width (v, v) placeholders, which made every claim
        # in SCOREBOARD impossible to compare against published numbers.
        # Bootstrap reseed is deterministic for reproducibility.
        ci_seed = self._seed if self._seed is not None else 1337
        cis: dict[str, tuple[float, float]] = {}
        cis["accuracy"] = _bootstrap_ci(flat, seed=ci_seed)
        cis["accuracy_correct"] = _bootstrap_ci(completed_scores, seed=ci_seed)
        for qtype, scores in per_type_scores.items():
            if scores:
                cis[f"accuracy_{qtype}"] = _bootstrap_ci(scores, seed=ci_seed)
        # Constants ship with zero-width CIs by construction (count
        # metrics, k) -- there's no resampling distribution for them.
        for name in ("n_questions", "n_completed", "n_errored", "k"):
            cis[name] = (metrics[name], metrics[name])
        for qtype in per_type_scores:
            key = f"n_{qtype}"
            if key in metrics:
                cis[key] = (metrics[key], metrics[key])
        # Record judge + answer prompt versions on every per_question
        # entry so the manifest preserves which prompt template scored
        # each row. Pre-audit the PROMPT_VERSIONS constants existed but
        # never landed in the manifest; reruns with a bumped judge
        # template would silently mix accuracy points across templates.
        # Stamping per-row keeps the historical contract clean even if
        # a future suite uses different prompts per question type.
        for entry in per_question:
            entry["answer_prompt_version"] = PROMPT_VERSIONS["answer"]
            entry["judge_prompt_version"] = PROMPT_VERSIONS["judge"]
        return SuiteResult(
            name=self.name,
            aggregate_metrics=metrics,
            confidence_intervals=cis,
            per_question=per_question,
            latency_ms={
                "ingest": ingest_ms_all,
                "retrieve": retrieve_ms,
                "answer": answer_ms,
                "judge": judge_ms,
            },
        )

    def _run_one_question(
        self,
        q: _Question,
        *,
        q_idx: int,
        questions: Sequence[_Question],
        storage: Any,
        embedder: Any,
        chat: Any,
        judge_chat: Any,
        default_retrieve_params: RetrieveParams,
        per_question: list[dict[str, Any]],
        per_type_scores: dict[str, list[float]],
        retrieve_ms: list[float],
        answer_ms: list[float],
        judge_ms: list[float],
        ingest_ms_all: list[float],
        log_interval: int,
    ) -> None:
        """Run the ingest + retrieve + answer + judge pipeline for one
        question, with full per-question exception isolation.

        Wraps the whole body in `try/except Exception`: any failure
        (content-filter rejection from a chat provider, network blip,
        parse error, even a defensive RuntimeError from the engine)
        scores the question as 0 and records the exception message on
        the per-question manifest entry. The next question proceeds
        normally -- a single bad request never costs more than its own
        slot in the manifest.

        KeyboardInterrupt and SystemExit are NOT swallowed so Ctrl+C
        still aborts the run cleanly.
        """
        ingest_ms = 0.0
        consolidate_ms = 0.0
        turns = 0
        response = ""
        score = 0.0
        error_msg: str | None = None
        auto_temporal_fallback = False
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
            ingest_ms_all.append(ingest_ms)

            if self._consolidate:
                t_consolidate = time.perf_counter()
                try:
                    cons_result = asyncio.run(memory.aconsolidate())
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
                # Record the fallback so per-question analysis can tell
                # which questions the temporal filter emptied -- pre-audit
                # this was silent and the manifest only captured the
                # second result set, hiding the cost of an over-tight
                # filter.
                auto_temporal_fallback = True
                retrieve_kwargs.pop("lexical_filter", None)
                results = memory.retrieve(q.question, **retrieve_kwargs)
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            # Content-filter rejections, rate-limit drops, network blips,
            # parse failures all land here. Record the failure on the
            # manifest, log it, score 0, and let the outer loop move on.
            error_msg = f"{type(exc).__name__}: {exc}"
            _LOG.warning(
                "q %d/%d [%s] -> ERROR (%s)",
                q_idx + 1,
                len(questions),
                q.qtype,
                error_msg,
            )
            score = 0.0
            # Keep latency arrays aligned: append placeholders for any
            # phase that didn't finish so the per-phase latency stats
            # don't grow misaligned with `per_question`.
            if len(retrieve_ms) <= q_idx:
                retrieve_ms.append(0.0)
            if len(answer_ms) <= q_idx:
                answer_ms.append(0.0)
            if len(judge_ms) <= q_idx:
                judge_ms.append(0.0)
            if len(ingest_ms_all) <= q_idx:
                ingest_ms_all.append(0.0)

        per_type_scores.setdefault(q.qtype, []).append(score)
        entry: dict[str, Any] = {
            "question_id": q.qid,
            "question_type": q.qtype,
            "question": q.question,
            "gold": q.gold,
            "response": response,
            "score": score,
            "k": self._k,
            "turns_ingested": turns,
            "consolidate_ms": consolidate_ms,
            "auto_temporal_fallback": auto_temporal_fallback,
        }
        if error_msg is not None:
            entry["error"] = error_msg
        per_question.append(entry)

        if (q_idx + 1) % log_interval == 0 or q_idx == len(questions) - 1:
            correct_so_far = sum(s for vs in per_type_scores.values() for s in vs)
            total = sum(len(vs) for vs in per_type_scores.values())
            verdict = "ERROR" if error_msg else ("PASS" if score == 1.0 else "FAIL")
            _LOG.info(
                "q %d/%d [%s] -> %s "
                "(ingest %d turns in %.1fs, ans %.1fs, jud %.1fs; acc=%.3f)",
                q_idx + 1,
                len(questions),
                q.qtype,
                verdict,
                turns,
                ingest_ms / 1000.0,
                answer_ms[-1] / 1000.0 if answer_ms else 0.0,
                judge_ms[-1] / 1000.0 if judge_ms else 0.0,
                correct_so_far / total if total else 0.0,
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
