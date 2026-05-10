"""LongMemEval benchmark suite.

Stage 6 DoD requires Engram to meet or beat the best public LongMemEval
number on a single fixed split. The scaffold loads `longmemeval-S` /
`longmemeval-M` / `longmemeval-Oracle` from `benchmarks/datasets/`,
runs Engram's hierarchical retrieve, computes the dataset's reported
accuracy metric, and emits a manifest pinning the score to the run.

The dataset is **not vendored**. To run this suite, place the splits at
`benchmarks/datasets/longmemeval/<split>.jsonl`. The reference release
schema (per LongMemEval's repo):

    {
      "session_id": "...",
      "messages": [{"role": "...", "content": "...", "ts": "..."}, ...],
      "questions": [
        {"id": "...", "question": "...", "answer": "...", "type": "..."}
      ]
    }

For CI's smoke run (no dataset available), this suite skips with a
"missing dataset" SuiteResult and a metric of 0. That keeps the harness
contract stable across environments without polluting tracking metrics.

The reference paper to compare against in `SCOREBOARD.md` is:

    Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
    Interactive Memory" (cited at lock-in time in `benchmarks/SOTA.md`).

Engram's score in the manifest is meaningless until a real provider
(Anthropic / OpenAI) drives a real embedding model. The fake-provider
path exists only to validate that the harness wiring works.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engram import Memory, SqliteStorage
from engram.bench import Provider, SuiteResult
from engram.schemas import Embedding, Event, ItemKind

_LOG = logging.getLogger("engram.bench.longmemeval")

DATASET_ROOT = Path("benchmarks/datasets/longmemeval")
DEFAULT_SPLIT = "longmemeval-S"
K = 5


@dataclass(frozen=True)
class _Question:
    id: str
    text: str
    answer: str
    qtype: str


@dataclass(frozen=True)
class _Session:
    id: str
    messages: tuple[dict[str, Any], ...]
    questions: tuple[_Question, ...]


def _load_split(split: str) -> Iterator[_Session]:
    path = DATASET_ROOT / f"{split}.jsonl"
    if not path.exists():
        return iter(())

    def gen() -> Iterator[_Session]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield _Session(
                    id=row["session_id"],
                    messages=tuple(row.get("messages", ())),
                    questions=tuple(
                        _Question(
                            id=q["id"],
                            text=q["question"],
                            answer=q["answer"],
                            qtype=q.get("type", "open"),
                        )
                        for q in row.get("questions", ())
                    ),
                )

    return gen()


def _checksum(split: str) -> str:
    path = DATASET_ROOT / f"{split}.jsonl"
    if not path.exists():
        return f"longmemeval/{split}/missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"longmemeval/{split}/{h.hexdigest()}"


def _exact_match(answer: str, hits_text: list[str]) -> float:
    """Crude relevance proxy until the real evaluator is in: did any
    retrieved item contain the answer string? LongMemEval's official
    scorer uses an LLM judge -- swap that in once the provider seam is
    driven by a real chat model in the suite config."""
    answer_l = answer.strip().lower()
    if not answer_l:
        return 0.0
    return 1.0 if any(answer_l in t.lower() for t in hits_text) else 0.0


class LongMemEvalSuite:
    name: str = "longmemeval"
    dataset_version: str

    def __init__(self, *, split: str = DEFAULT_SPLIT) -> None:
        self.split = split
        self.dataset_version = f"{split}-v1"
        self.dataset_checksum: str = _checksum(split)
        self._provider: Provider | None = None

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        if self._provider is None:
            raise RuntimeError("setup() must be called before run()")
        embedder = getattr(self._provider, "embedder", None)
        if embedder is None:
            raise RuntimeError("longmemeval requires a provider with an `embedder` attribute")

        sessions = list(_load_split(self.split))
        if not sessions:
            _LOG.warning(
                "longmemeval split %r not found at %s; emitting placeholder result",
                self.split,
                DATASET_ROOT,
            )
            return SuiteResult(
                name=self.name,
                aggregate_metrics={"accuracy": 0.0, "n_questions": 0.0},
                confidence_intervals={"accuracy": (0.0, 0.0)},
                per_question=[],
                latency_ms={},
            )

        per_question: list[dict[str, Any]] = []
        retrieve_ms: list[float] = []
        scores: list[float] = []

        for session in sessions:
            storage = SqliteStorage(":memory:")
            storage.initialize()
            try:
                events: list[Event] = []
                for msg in session.messages:
                    text = msg.get("content", "")
                    if not text:
                        continue
                    events.append(Event(content=text, source=msg.get("role")))
                if not events:
                    continue
                storage.insert_events(events)
                vectors = embedder.embed([e.content for e in events])
                with storage.transaction():
                    for e, v in zip(events, vectors, strict=True):
                        storage.insert_embedding(
                            Embedding(
                                item_id=e.id,
                                item_kind=ItemKind.EVENT,
                                model=embedder.model,
                                dim=embedder.dim,
                                vector=tuple(v),
                            )
                        )
                memory = Memory(storage=storage, embedder=embedder)

                for q in session.questions:
                    t0 = time.perf_counter()
                    results = memory.retrieve(q.text, k=K, reinforce=False)
                    retrieve_ms.append((time.perf_counter() - t0) * 1000.0)
                    score = _exact_match(q.answer, [r.content for r in results])
                    scores.append(score)
                    per_question.append(
                        {
                            "session_id": session.id,
                            "question_id": q.id,
                            "qtype": q.qtype,
                            "score": score,
                            "k": K,
                        }
                    )
            finally:
                storage.close()

        accuracy = sum(scores) / len(scores) if scores else 0.0
        return SuiteResult(
            name=self.name,
            aggregate_metrics={
                "accuracy": accuracy,
                "n_questions": float(len(scores)),
                "n_sessions": float(len(sessions)),
            },
            confidence_intervals={"accuracy": (accuracy, accuracy)},
            per_question=per_question,
            latency_ms={"retrieve": retrieve_ms},
        )

    def teardown(self) -> None:
        self._provider = None


SUITE: LongMemEvalSuite = LongMemEvalSuite()
