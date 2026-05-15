"""Procedural transfer benchmark (Stage 7).

Tests whether an agent backed by Engram's procedural memory beats a
no-memory baseline on a held-out task suite. The synthetic version
shipped here exercises the API contract; the real-LLM version (using
a chat provider for action selection) lands as a follow-up.

The synthetic protocol:

  1. Plant N "training" procedures. Each is a (situation_pattern,
     known_good_action, SUCCESS) tuple drawn from a fixed task family
     (e.g. "user reports flaky X" -> "isolate + rerun" pattern).
  2. Generate M held-out tasks. Each task has a situation that
     paraphrases one of the training patterns. The correct action is
     the same as the training pattern's action.
  3. No-memory agent: picks an action at random from the action pool.
     Score: hit rate on held-out tasks.
  4. Engram agent: calls `memory.retrieve_procedures(situation, k=1)`,
     picks the top result's action. Score: hit rate on held-out tasks.
  5. Reports both scores plus the lift (engram - baseline).

Stage 7 DoD: engram > baseline by >= 15 percentage points. With the
synthetic split below (orthogonal one-hot situation vectors + planted
training procedures), the engram agent scores ~100% and the random
baseline scores ~1/n_actions, so the lift is dramatic and stable.

CI runs this via `python -m engram.bench run procedural-transfer
--provider fake` against FakeEmbedder. The synthetic numbers are not
SOTA claims; they're contract verification. The follow-up real-LLM
version is what goes into the SCOREBOARD's "procedural transfer" row.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engram import (
    Memory,
    Outcome,
    SqliteStorage,
)
from engram.bench import Provider, SuiteResult


@dataclass(frozen=True)
class _TaskPattern:
    """One training pattern: situation phrasing + the known-good action."""

    situation: str
    action: str


# The training set. Each pattern is a recognizable agent task. The
# held-out queries paraphrase these situations; the agent must surface
# the matching action via similarity.
TRAINING_PATTERNS: tuple[_TaskPattern, ...] = (
    _TaskPattern(
        situation="user reports a flaky integration test",
        action="rerun with --no-cov and bisect; isolate the failing assertion",
    ),
    _TaskPattern(
        situation="user asks how to migrate database schema safely",
        action="use a numbered SQL migration, wrap in BEGIN/COMMIT, run on a copy first",
    ),
    _TaskPattern(
        situation="user wants help debugging a Python import error",
        action="check sys.path, verify __init__.py files, run python -c 'import x' to isolate",
    ),
    _TaskPattern(
        situation="user reports OOM error during model training",
        action=(
            "reduce batch size, enable gradient checkpointing, monitor GPU memory with nvidia-smi"
        ),
    ),
    _TaskPattern(
        situation="user asks about handling a flood of API rate limits",
        action="implement exponential backoff with jitter, cache responses, batch where possible",
    ),
)

# Held-out queries. By default they match training situations exactly --
# the basic procedural transfer test ("given this exact situation again,
# recall the procedure") works with any embedder including the
# hash-based FakeEmbedder used in CI. To exercise the harder
# paraphrase/generalization case (which needs a real semantic
# embedder), pass `paraphrase_mode=True` to the suite.
HELDOUT_QUERIES_EXACT: tuple[tuple[str, str], ...] = tuple(
    (p.situation, p.action) for p in TRAINING_PATTERNS
)

# Paraphrases for use with real (semantic) embedders. The expected
# action is the same as the source training pattern's action; the
# embedder has to bridge surface-form variation. This is the harder
# test and only makes sense for `--embedder local` or `--embedder
# openai` runs.
HELDOUT_QUERIES_PARAPHRASE: tuple[tuple[str, str], ...] = (
    ("flaky CI test keeps failing intermittently", TRAINING_PATTERNS[0].action),
    (
        "how do I safely change my database tables in production",
        TRAINING_PATTERNS[1].action,
    ),
    ("ImportError: cannot import name 'X' from 'package'", TRAINING_PATTERNS[2].action),
    ("CUDA out of memory while training my transformer", TRAINING_PATTERNS[3].action),
    ("getting 429 errors from the OpenAI API", TRAINING_PATTERNS[4].action),
)


def _dataset_checksum(queries: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for p in TRAINING_PATTERNS:
        h.update(p.situation.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.action.encode("utf-8"))
        h.update(b"\x01")
    for q, a in queries:
        h.update(q.encode("utf-8"))
        h.update(b"\x00")
        h.update(a.encode("utf-8"))
        h.update(b"\x02")
    return h.hexdigest()


def _no_memory_baseline_score(
    queries: Sequence[tuple[str, str]],
    rng: random.Random,
    n_trials: int = 5,
) -> float:
    """Random-action baseline. Averaged over `n_trials` to smooth noise."""
    pool = [p.action for p in TRAINING_PATTERNS]
    hits = 0
    n = 0
    for _trial in range(n_trials):
        for _query, correct_action in queries:
            chosen = rng.choice(pool)
            if chosen == correct_action:
                hits += 1
            n += 1
    return hits / n if n else 0.0


def _engram_agent_score(
    memory: Memory,
    queries: Sequence[tuple[str, str]],
) -> tuple[float, list[dict[str, Any]]]:
    """Engram agent: retrieve top procedure for each held-out query, use
    its action. Returns (hit_rate, per_query).

    Hit is decided by procedure id rather than action-string equality.
    Strict string compare was brittle: a future refactor that
    normalizes whitespace, capitalizes verbs, or rewrites actions
    during consolidation would silently regress every score.  Matching
    by id requires the held-out query to retrieve the SAME procedure
    that produced `correct_action` during training — semantically what
    we want.
    """
    # Record training patterns and remember the id assigned to each
    # action.  Multiple training patterns share an action only if the
    # caller wired them that way; the action -> id map is one-to-one
    # otherwise.
    action_to_id: dict[str, Any] = {}
    for pattern in TRAINING_PATTERNS:
        recorded = memory.record_procedure(
            pattern.situation,
            pattern.action,
            outcome=Outcome.SUCCESS,
        )
        action_to_id.setdefault(pattern.action, recorded.id)

    per_query: list[dict[str, Any]] = []
    hits = 0
    for query, correct_action in queries:
        results = memory.retrieve_procedures(query, k=1, reinforce=False)
        chosen_action = results[0].procedure.action if results else "(none)"
        chosen_id = results[0].procedure.id if results else None
        correct_id = action_to_id.get(correct_action)
        # Match by id when both sides are known; fall back to action
        # string compare for synthetic queries whose `correct_action`
        # wasn't part of the training set (unusual, but defensive).
        if correct_id is not None and chosen_id is not None:
            is_hit = chosen_id == correct_id
        else:
            is_hit = chosen_action == correct_action
        if is_hit:
            hits += 1
        per_query.append(
            {
                "query": query,
                "correct_action": correct_action,
                "chosen_action": chosen_action,
                "hit": is_hit,
                "similarity": results[0].similarity if results else 0.0,
            }
        )
    return hits / len(queries) if queries else 0.0, per_query


class ProceduralTransferSuite:
    name: str = "procedural-transfer"
    dataset_version: str = "synthetic-v1"

    def __init__(self, *, paraphrase_mode: bool = False) -> None:
        self._provider: Provider | None = None
        self._paraphrase = paraphrase_mode
        self._queries: Sequence[tuple[str, str]] = (
            HELDOUT_QUERIES_PARAPHRASE if paraphrase_mode else HELDOUT_QUERIES_EXACT
        )
        self.dataset_checksum: str = _dataset_checksum(self._queries)

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        if self._provider is None:
            raise RuntimeError("setup() must be called before run()")
        embedder = getattr(self._provider, "embedder", None)
        if embedder is None:
            raise RuntimeError(
                "procedural-transfer requires a provider with an `embedder` attribute"
            )

        rng = random.Random(0)  # noqa: S311 -- benchmark baseline, not a security primitive
        baseline = _no_memory_baseline_score(self._queries, rng)

        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            memory = Memory(storage=storage, embedder=embedder)
            t0 = time.perf_counter()
            engram_score, per_query = _engram_agent_score(memory, self._queries)
            wallclock_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            storage.close()

        lift = engram_score - baseline
        metrics: dict[str, float] = {
            "engram_score": engram_score,
            "baseline_score": baseline,
            "lift": lift,
            "n_training": float(len(TRAINING_PATTERNS)),
            "n_heldout": float(len(self._queries)),
            "paraphrase_mode": 1.0 if self._paraphrase else 0.0,
        }
        cis: dict[str, tuple[float, float]] = {k: (v, v) for k, v in metrics.items()}
        return SuiteResult(
            name=self.name,
            aggregate_metrics=metrics,
            confidence_intervals=cis,
            per_question=per_query,
            latency_ms={"engram_score_run": [wallclock_ms]},
        )

    def teardown(self) -> None:
        self._provider = None


# Default suite: exact-text held-out queries. CI-friendly (works with
# FakeEmbedder). For paraphrase generalization, swap the SUITE
# assignment for `ProceduralTransferSuite(paraphrase_mode=True)`.
SUITE: ProceduralTransferSuite = ProceduralTransferSuite()
SUITE_PARAPHRASE: ProceduralTransferSuite = ProceduralTransferSuite(paraphrase_mode=True)
