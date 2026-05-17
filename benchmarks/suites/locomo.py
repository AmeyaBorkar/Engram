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
import math
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


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> tuple[float, float]:
    """Return percentile-bootstrap 95% CI on the mean of ``values``.

    Inlined into the suite (rather than importing from ``scripts/``) so
    the benchmark module has no dependency on the ``scripts/`` package
    at import time -- the same harness is used in test runs that don't
    have the ``scripts/`` tree available.

    Empty input collapses to (0, 0); single-element to the trivial
    interval. ``n_iters=2000`` is a conservative tradeoff between
    interval accuracy and runtime for the typical LoCoMo split
    (~10 ms per ~200-sample bootstrap).
    """
    import random as _random  # local import; suite-level random untouched

    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    vs = list(values)
    if n == 1:
        v = float(vs[0])
        return (v, v)
    rng = _random.Random(seed)
    means: list[float] = []
    for _ in range(n_iters):
        s = 0.0
        for _ in range(n):
            s += vs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = max(0, int(alpha / 2 * n_iters))
    hi_idx = min(n_iters - 1, int((1 - alpha / 2) * n_iters))
    return (means[lo_idx], means[hi_idx])


# Metric names whose CI is "real" (computed via bootstrap on per-question
# scores). Other metric names (counts, error totals) are reported with
# the degenerate (v, v) width-zero CI -- bootstrapping a count would
# be meaningless.
_LOCOMO_PROPORTION_METRICS: frozenset[str] = frozenset({"accuracy"})


def _real_locomo_cis(
    metrics: dict[str, float],
    per_type_scores: dict[str, list[float]],
    flat_scores: list[float],
) -> dict[str, tuple[float, float]]:
    """Compute confidence intervals for the LoCoMo aggregate metrics.

    M-160: pre-fix the suite returned ``(v, v)`` for every metric --
    a zero-width interval that broadcasts certainty we don't have.
    Now: bootstrap CIs for the proportion-style metrics (accuracy,
    per-qtype accuracy), degenerate ``(v, v)`` for counts.
    """
    out: dict[str, tuple[float, float]] = {}
    for name, value in metrics.items():
        if name == "accuracy":
            out[name] = _bootstrap_mean_ci(flat_scores)
        elif name.startswith("accuracy_"):
            qtype = name.removeprefix("accuracy_")
            scores = per_type_scores.get(qtype, [])
            out[name] = _bootstrap_mean_ci(scores)
        elif name in _LOCOMO_PROPORTION_METRICS:  # future-proof
            out[name] = _bootstrap_mean_ci(flat_scores)
        else:
            out[name] = (value, value)
    return out


_TOKEN_RE = __import__("re").compile(r"\w+")
# Subset of NLTK's English stopwords — small enough to be obvious, large
# enough to make token-F1 behave like the official LoCoMo scorer on
# common cases.  Kept inline so the suite has no NLTK runtime dep.
_LOCOMO_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "to",
        "for", "with", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "this", "that", "these",
        "those", "it", "its", "as", "at", "by", "from", "into", "i", "you",
        "we", "they", "he", "she",
    }
)


def _tokenize_for_f1(text: str) -> list[str]:
    return [
        tok
        for tok in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if tok and tok not in _LOCOMO_STOPWORDS
    ]


def _token_f1(gold: str, candidate: str) -> float:
    """Token-level F1 (stopwords removed), matching LoCoMo's standard QA scorer.

    Multi-token answers get partial credit when retrieved passages cover
    some of the gold tokens.  Returns 0 if either side is empty after
    tokenization.
    """
    gold_tokens = _tokenize_for_f1(gold)
    cand_tokens = _tokenize_for_f1(candidate)
    if not gold_tokens or not cand_tokens:
        return 0.0
    # Multiset-style intersection so a repeated gold token has to be
    # matched by the same number of repeated candidate tokens.
    gold_counts: dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = 0
    for t in cand_tokens:
        if gold_counts.get(t, 0) > 0:
            overlap += 1
            gold_counts[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(answer: str, hits_text: list[str]) -> float:
    """Best per-hit token-F1 score for `answer` across `hits_text`.

    Renamed from substring-containment to the standard LoCoMo
    token-F1 metric.  The previous implementation returned 1.0 if the
    gold answer string appeared anywhere inside any retrieved text —
    so 'no' matched any text mentioning 'no', 'Lisbon' matched any
    text mentioning Lisbon, and 'yes' was trivially high-coverage.
    That made LoCoMo numbers incomparable to published scores.
    """
    answer_l = answer.strip()
    if not answer_l or not hits_text:
        return 0.0
    return max(_token_f1(answer_l, t) for t in hits_text)


class LoCoMoSuite:
    name: str = "locomo"
    dataset_version: str

    def __init__(self, *, split: str = DEFAULT_SPLIT) -> None:
        self.split = split
        self.dataset_version = f"locomo-{split}-v1"
        self.dataset_checksum: str = _checksum(split)
        self._provider: Provider | None = None
        self._max_questions: int | None = None
        self._seed: int | None = None

    def configure(
        self,
        *,
        max_questions: int | None = None,
        seed: int | None = None,
        **_ignored: Any,
    ) -> None:
        """Per-run knobs from the CLI's ``suite_config``.

        H-74: ``--limit`` is plumbed via ``suite_config`` instead of
        mutating ``os.environ``. Pre-fix, ``--limit`` set
        ``LOCOMO_MAX_QUESTIONS`` but the suite never read that env
        var, so the cap was silently ignored on LoCoMo.

        Other Phase E knobs are accepted but ignored -- LoCoMo's
        scoring path is intentionally simple (raw retrieval + token-F1)
        and doesn't currently use the BM25/MMR/recency machinery.
        Keeping ``**_ignored`` lets the same CLI flags be passed
        without errors.
        """
        if max_questions is not None and max_questions < 0:
            raise ValueError(f"max_questions must be >= 0, got {max_questions}")
        self._max_questions = max_questions
        self._seed = seed

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
        # Per-qtype scores are populated by setdefault as each question
        # is scored.  Pre-populating with empty lists keyed on _SPLITS
        # used to put empty buckets in the aggregate (`single_hop: []`)
        # for splits that weren't even run, and double-counted when a
        # question's qtype was named differently from its split
        # (e.g. qtype 'single_hop_v2' lived alongside an empty
        # 'single_hop' bucket).
        per_type_scores: dict[str, list[float]] = {}
        # M-159: count per-question errors separately from "wrong
        # answer" so an infra blip (network, parse error) doesn't
        # inflate the headline failure rate. The number is reported
        # as a separate metric.
        n_errored = 0

        any_data = False
        questions_seen = 0
        cap = self._max_questions  # H-74; None = uncapped.
        for split in splits_to_run:
            for conversation in _load_split(split):
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
                            # L2-normalize before storage so cosine
                            # similarity at query time matches what the
                            # in-memory vector index expects (unit-norm
                            # rows).  Mirror of LongMemEval's ingest
                            # path — without this, LoCoMo retrieval
                            # scores were wrong by a model-dependent
                            # constant and comparisons against published
                            # LoCoMo numbers (or against the BM25/Chroma
                            # baselines) were not apples-to-apples.
                            norm = math.sqrt(sum(x * x for x in v))
                            normalized = (
                                tuple(x / norm for x in v)
                                if norm > 0.0
                                else tuple(v)
                            )
                            storage.insert_embedding(
                                Embedding(
                                    item_id=e.id,
                                    item_kind=ItemKind.EVENT,
                                    model=embedder.model,
                                    dim=embedder.dim,
                                    vector=normalized,
                                )
                            )
                    memory = Memory(storage=storage, embedder=embedder)
                    for q in conversation.questions:
                        if cap is not None and questions_seen >= cap:
                            break
                        questions_seen += 1
                        # M-159 fix: per-question try/except. One
                        # bad question (parse error, retrieve
                        # raising on a malformed embedding row,
                        # etc.) used to abort the entire split.
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
                            n_errored += 1
                            _LOG.warning(
                                "locomo: q %s in conversation %s -> %s",
                                q.id,
                                conversation.id,
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
                finally:
                    storage.close()
                if cap is not None and questions_seen >= cap:
                    break
            if cap is not None and questions_seen >= cap:
                break

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
        metrics: dict[str, float] = {
            "accuracy": overall,
            "n_questions": float(len(flat_scores)),
            "n_errored": float(n_errored),
        }
        for qtype, acc in per_type_acc.items():
            metrics[f"accuracy_{qtype}"] = acc
        # M-160: real bootstrap CIs over the per-question score list
        # for proportion-style metrics; non-proportion metrics
        # (n_questions, n_errored) get the degenerate width-zero CI
        # they deserve.
        cis: dict[str, tuple[float, float]] = _real_locomo_cis(
            metrics, per_type_scores, flat_scores
        )
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
