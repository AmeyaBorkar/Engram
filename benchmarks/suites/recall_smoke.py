"""Recall smoke benchmark.

Tiny synthetic corpus; every query is exact text of one indexed document.
Every retriever should score recall@10 = 1.0 here - the suite exists to
validate the harness wiring (suite loading, retriever protocol, baseline
adapters, manifest writing), NOT embedding quality.

Stage 6 plugs in real datasets (LongMemEval, LoCoMo) against real
providers; until then this is what runs in CI when SOTA infrastructure
is exercised end-to-end.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import Any

from engram import Memory, SqliteStorage
from engram.bench import EngramRetriever, Hit, Provider, Retriever, SuiteResult

K = 10

_DOCS: tuple[str, ...] = (
    "user mentioned they have a golden retriever named Max",
    "Max is nine years old and slowing down a bit",
    "user asked about senior dog food brands",
    "Max prefers chicken-based kibble over fish",
    "user prefers tea over coffee in the morning",
    "weekly grocery list now includes oat milk and sourdough",
    "user is learning Spanish on a 30-minute daily streak",
    "the kitchen sink leak was finally fixed last weekend",
    "user is reading a biography of Marie Curie",
    "the next dental cleaning is scheduled for March 12",
    "user's favorite hiking trail is the one above the lake",
    "summer trip plan: a week in Lisbon and four days in Porto",
    "user's running shoes need replacing - over 800 km logged",
    "the spider plant by the window finally sprouted babies",
    "favorite weeknight dinner: lentil soup with crusty bread",
    "user is allergic to shellfish; serious reaction in 2019",
    "weekly therapy session moved to Thursdays at 4pm",
    "user signed up for a watercolor class on Sundays",
    "the new neighbors brought over banana bread on day one",
    "user's first car was a 1998 Honda Civic, manual transmission",
    "monthly book club is reading Borges this month",
    "user's grandmother turns 92 in October",
    "the office moved buildings; new desk near a window",
    "user keeps a paper journal, two pages every evening",
    "running goal: a half marathon by year end",
    "user's favorite indoor plant is the rubber tree",
    "weekend ritual: sourdough start + farmers market + bike ride",
    "the dishwasher is making a knocking sound on rinse cycle",
    "user prefers paperbacks; gives away hardcovers after reading",
    "first attempt at miso paste from scratch took six weeks",
)


def _docs_checksum() -> str:
    h = hashlib.sha256()
    for doc in _DOCS:
        h.update(doc.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


_DATASET_CHECKSUM = _docs_checksum()


def _build_retrievers(provider: Provider) -> dict[str, tuple[Retriever, Any]]:
    """Return name -> (retriever, cleanup) for every system we know how to construct.

    Skips Chroma-based retrievers if `chromadb` is not installed.
    """
    embedder = getattr(provider, "embedder", None)
    if embedder is None:
        raise RuntimeError(
            "recall-smoke suite requires a provider with an `embedder` attribute "
            "(the harness's FakeProvider satisfies this)."
        )

    retrievers: dict[str, tuple[Retriever, Any]] = {}

    storage = SqliteStorage(":memory:")
    storage.initialize()
    memory = Memory(storage=storage, embedder=embedder)
    retrievers["engram"] = (EngramRetriever(memory), storage)

    try:
        from benchmarks.baselines.chroma import ChromaRetriever
        from benchmarks.baselines.chroma_bm25 import ChromaBM25Retriever
    except ImportError:
        return retrievers

    retrievers["chroma"] = (ChromaRetriever(embedder=embedder), None)
    retrievers["chroma_bm25"] = (ChromaBM25Retriever(embedder=embedder), None)
    return retrievers


def _recall(hits: list[Hit], relevant_id: str) -> float:
    return 1.0 if any(h.id == relevant_id for h in hits) else 0.0


class RecallSmokeSuite:
    name: str = "recall-smoke"
    dataset_version: str = "synthetic-conversational-v1"
    dataset_checksum: str = _DATASET_CHECKSUM

    def __init__(self) -> None:
        self._provider: Provider | None = None

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        if self._provider is None:
            raise RuntimeError("setup() must be called before run()")
        retrievers = _build_retrievers(self._provider)

        per_question: list[dict[str, Any]] = []
        recall_per_system: dict[str, list[float]] = {name: [] for name in retrievers}

        for system_name, (retriever, _cleanup) in retrievers.items():
            doc_ids: list[str] = []
            for i, doc in enumerate(_DOCS):
                doc_ids.append(retriever.add(doc, doc_id=f"doc-{i}"))

            for q_idx, query in enumerate(_DOCS[:K]):
                hits = list(retriever.query(query, k=K))
                rec = _recall(hits, relevant_id=doc_ids[q_idx])
                recall_per_system[system_name].append(rec)
                per_question.append(
                    {
                        "system": system_name,
                        "query_idx": q_idx,
                        "query": query,
                        "recall@k": rec,
                        "k": K,
                        "n_hits": len(hits),
                    }
                )

        for _name, (_retriever, cleanup) in retrievers.items():
            if cleanup is not None and hasattr(cleanup, "close"):
                with contextlib.suppress(Exception):
                    cleanup.close()

        aggregate_metrics: dict[str, float] = {
            f"{system}_recall@{K}": (sum(recalls) / len(recalls)) if recalls else 0.0
            for system, recalls in recall_per_system.items()
        }
        confidence_intervals: dict[str, tuple[float, float]] = {
            metric: (value, value) for metric, value in aggregate_metrics.items()
        }

        return SuiteResult(
            name=self.name,
            aggregate_metrics=aggregate_metrics,
            confidence_intervals=confidence_intervals,
            per_question=per_question,
            latency_ms={},
        )

    def teardown(self) -> None:
        self._provider = None


SUITE: RecallSmokeSuite = RecallSmokeSuite()
