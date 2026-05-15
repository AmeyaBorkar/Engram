"""LoCoMo benchmark suite.

LoCoMo is a long-conversation memory benchmark with five sub-splits:
single-hop, multi-hop, temporal, open-domain, and adversarial. Stage 6
DoD requires Engram to match the best public RAG-class number on the
temporal and adversarial splits; Stage 8 raises that to "beat".

Like the LongMemEval scaffold, this loads the dataset from
`benchmarks/datasets/locomo/<split>.jsonl` and emits a manifest. CI
runs without the dataset and falls through to a placeholder result.

Reference release schema (per the LoCoMo authors' repo):

    {
      "conversation_id": "...",
      "turns": [{"speaker": "...", "text": "...", "ts": "..."}, ...],
      "questions": [
        {"id": "...", "type": "single_hop|multi_hop|temporal|...",
         "question": "...", "answer": "...", "evidence_turn_ids": [...]}
      ]
    }

Per-split accuracy ends up in `aggregate_metrics["accuracy_<split>"]`
plus an aggregate `accuracy` averaged across types.
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

_LOG = logging.getLogger("engram.bench.locomo")

DATASET_ROOT = Path("benchmarks/datasets/locomo")
DEFAULT_SPLIT = "all"
K = 5

_SPLITS: tuple[str, ...] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "adversarial",
)


@dataclass(frozen=True)
class _Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class _Question:
    id: str
    qtype: str
    text: str
    answer: str
    evidence_turn_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Conversation:
    id: str
    turns: tuple[_Turn, ...]
    questions: tuple[_Question, ...]


def _load_split(split: str) -> Iterator[_Conversation]:
    path = DATASET_ROOT / f"{split}.jsonl"
    if not path.exists():
        return iter(())

    def gen() -> Iterator[_Conversation]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield _Conversation(
                    id=row["conversation_id"],
                    turns=tuple(
                        _Turn(speaker=t.get("speaker", ""), text=t.get("text", ""))
                        for t in row.get("turns", ())
                    ),
                    questions=tuple(
                        _Question(
                            id=q["id"],
                            qtype=q["type"],
                            text=q["question"],
                            answer=q["answer"],
                            evidence_turn_ids=tuple(q.get("evidence_turn_ids", ())),
                        )
                        for q in row.get("questions", ())
                    ),
                )

    return gen()


def _checksum(split: str) -> str:
    path = DATASET_ROOT / f"{split}.jsonl"
    if not path.exists():
        return f"locomo/{split}/missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"locomo/{split}/{h.hexdigest()}"


def _exact_match(answer: str, hits_text: list[str]) -> float:
    answer_l = answer.strip().lower()
    if not answer_l:
        return 0.0
    return 1.0 if any(answer_l in t.lower() for t in hits_text) else 0.0


def _bootstrap_ci(values: list[float], *, seed: int = 1337) -> tuple[float, float]:
    """Bootstrap a 95% CI on the mean. Empty / single -> (mean, mean).

    Local copy of the same helper LongMemEval ships. Kept inline so the
    suite has no cross-suite import dependency -- bench suites are
    designed to be independently loadable.
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


class LoCoMoSuite:
    name: str = "locomo"
    dataset_version: str

    def __init__(self, *, split: str = DEFAULT_SPLIT) -> None:
        self.split = split
        self.dataset_version = f"locomo-{split}-v1"
        self.dataset_checksum: str = _checksum(split)
        self._provider: Provider | None = None
        # Question-count cap, plumbed in via configure(max_questions=).
        # Pre-audit, LoCoMo silently ignored `--limit` because it only
        # honoured a never-set `LOCOMO_MAX_QUESTIONS` env. Now respects
        # suite_config so a single CLI knob covers every suite.
        self._max_questions: int | None = None

    def configure(self, *, max_questions: int | None = None, **_unused: Any) -> None:
        """Pick up `--limit` (or other future knobs) from `suite_config`.

        Extra kwargs are tolerated and ignored so the bench CLI can
        pass the same `suite_config` dict to every suite (LongMemEval
        cares about hyde/MMR/recency, LoCoMo doesn't) without each
        suite having to enumerate every flag the CLI knows about.
        """
        if max_questions is not None:
            self._max_questions = max_questions

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        if self._provider is None:
            raise RuntimeError("setup() must be called before run()")
        embedder = getattr(self._provider, "embedder", None)
        if embedder is None:
            raise RuntimeError("locomo requires a provider with an `embedder` attribute")

        # If the user requested `all`, iterate every split that exists.
        splits_to_run = list(_SPLITS) if self.split == "all" else [self.split]
        per_question: list[dict[str, Any]] = []
        retrieve_ms: list[float] = []
        # Don't pre-populate per_type_scores with the SPLIT filenames.
        # Pre-audit, splits like `single_hop` (filename) and qtypes like
        # `single-hop` (dataset value, naming convention varies) could
        # both end up in the dict as separate buckets, double-counting
        # the same question. Now we only `setdefault` from the canonical
        # `q.qtype` field so the aggregate exactly matches the dataset's
        # own taxonomy.
        per_type_scores: dict[str, list[float]] = {}

        # Per-question cap: count questions across ALL splits and stop
        # once we've reached the limit. Mirrors LongMemEval's `--limit`
        # semantics so a single CLI knob covers every suite.
        questions_seen = 0
        any_data = False
        for split in splits_to_run:
            if (
                self._max_questions is not None
                and questions_seen >= self._max_questions
            ):
                break
            for conversation in _load_split(split):
                if (
                    self._max_questions is not None
                    and questions_seen >= self._max_questions
                ):
                    break
                any_data = True
                storage = SqliteStorage(":memory:")
                storage.initialize()
                try:
                    events = [
                        Event(content=t.text, source=t.speaker)
                        for t in conversation.turns
                        if t.text
                    ]
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
                    for q in conversation.questions:
                        if (
                            self._max_questions is not None
                            and questions_seen >= self._max_questions
                        ):
                            break
                        # Per-question try/except (matches LongMemEval):
                        # one bad retrieval should never abort the
                        # entire LoCoMo split. The error lands on the
                        # per_question entry; the question scores 0 and
                        # the loop moves on. Pre-audit a single failure
                        # killed every subsequent question.
                        error_msg: str | None = None
                        score = 0.0
                        try:
                            t0 = time.perf_counter()
                            results = memory.retrieve(q.text, k=K, reinforce=False)
                            retrieve_ms.append((time.perf_counter() - t0) * 1000.0)
                            score = _exact_match(q.answer, [r.content for r in results])
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as exc:
                            error_msg = f"{type(exc).__name__}: {exc}"
                            _LOG.warning(
                                "locomo [%s/%s] -> ERROR (%s)",
                                conversation.id,
                                q.id,
                                error_msg,
                            )
                        per_type_scores.setdefault(q.qtype, []).append(score)
                        entry: dict[str, Any] = {
                            "conversation_id": conversation.id,
                            "question_id": q.id,
                            "qtype": q.qtype,
                            "score": score,
                            "k": K,
                        }
                        if error_msg is not None:
                            entry["error"] = error_msg
                        per_question.append(entry)
                        questions_seen += 1
                finally:
                    storage.close()

        if not any_data:
            _LOG.warning(
                "no LoCoMo splits found at %s; emitting placeholder result",
                DATASET_ROOT,
            )
            return SuiteResult(
                name=self.name,
                aggregate_metrics={"accuracy": 0.0, "n_questions": 0.0},
                confidence_intervals={"accuracy": (0.0, 0.0)},
                per_question=[],
                latency_ms={},
            )

        per_type_acc: dict[str, float] = {
            qtype: (sum(s) / len(s) if s else 0.0) for qtype, s in per_type_scores.items()
        }
        flat_scores = [s for vals in per_type_scores.values() for s in vals]
        overall = sum(flat_scores) / len(flat_scores) if flat_scores else 0.0
        n_errored = sum(1 for e in per_question if e.get("error"))
        metrics: dict[str, float] = {
            "accuracy": overall,
            "n_questions": float(len(flat_scores)),
            "n_errored": float(n_errored),
        }
        for qtype, acc in per_type_acc.items():
            metrics[f"accuracy_{qtype}"] = acc
        # Real bootstrap CIs on the bounded-mean metrics. Pre-audit
        # these were zero-width (v, v) placeholders -- "Engram beats
        # baseline by 0.03 [0.03, 0.03]" carries no statistical meaning.
        cis: dict[str, tuple[float, float]] = {
            "accuracy": _bootstrap_ci(flat_scores),
        }
        for qtype, s in per_type_scores.items():
            cis[f"accuracy_{qtype}"] = _bootstrap_ci(s)
        # Counts: zero-width by construction (no resampling distribution).
        for name in ("n_questions", "n_errored"):
            cis[name] = (metrics[name], metrics[name])
        return SuiteResult(
            name=self.name,
            aggregate_metrics=metrics,
            confidence_intervals=cis,
            per_question=per_question,
            latency_ms={"retrieve": retrieve_ms},
        )

    def teardown(self) -> None:
        self._provider = None


SUITE: LoCoMoSuite = LoCoMoSuite()
