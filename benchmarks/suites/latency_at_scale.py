"""Stage 6 latency benchmark @ 100k items.

SCOREBOARD target: `retrieve` (coarse-to-fine) at 100k items hits
P50 < 150 ms, P99 < 500 ms on a laptop with the in-memory vector
index. The benchmark is intentionally read-dominant and warmed up:
the cache rebuild that follows a large write burst is NOT amortized
into the latency window -- agents query in steady state, and the
budget reflects that workload.

Inputs:
  * `n_items` -- corpus size (default 100k).
  * `n_queries` -- the warm-loop sample size (default 200).
  * `dim` -- embedding dimensionality (taken from the provider).

Workload:
  1. Plant `n_items` synthetic events with their embeddings.
  2. Warm the vector index with 5 retrieves.
  3. Time `n_queries` retrieves (k=10), record per-call ms.
  4. Emit P50 / P95 / P99 / max as `aggregate_metrics`. The bench
     harness writes a manifest with the full per-call latencies in
     `latency_ms["retrieve"]`.

Determinism: queries are derived from `i * 113 % n_items` so the
sequence is deterministic but the picks are spread across the corpus
(modular hash with a coprime stride).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from engram import Memory, SqliteStorage
from engram.bench import Provider, SuiteResult
from engram.schemas import Embedding, Event, ItemKind

N_ITEMS_DEFAULT = 100_000
N_QUERIES_DEFAULT = 200
N_WARMUP = 5
K = 10
INSERT_BATCH = 1_000


def _docs(n: int) -> list[str]:
    return [f"sample fact number {i}" for i in range(n)]


def _docs_checksum(n: int) -> str:
    h = hashlib.sha256()
    h.update(f"latency-at-scale:n={n}".encode("utf-8"))
    return h.hexdigest()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    return s[idx]


class LatencyAtScaleSuite:
    name: str = "latency-at-scale"
    dataset_version: str = "synthetic-v1"
    n_items: int = N_ITEMS_DEFAULT
    n_queries: int = N_QUERIES_DEFAULT

    def __init__(
        self,
        *,
        n_items: int = N_ITEMS_DEFAULT,
        n_queries: int = N_QUERIES_DEFAULT,
    ) -> None:
        self.n_items = n_items
        self.n_queries = n_queries
        self.dataset_checksum: str = _docs_checksum(n_items)
        self._provider: Provider | None = None

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        if self._provider is None:
            raise RuntimeError("setup() must be called before run()")
        embedder = getattr(self._provider, "embedder", None)
        if embedder is None:
            raise RuntimeError("latency-at-scale requires a provider with an `embedder` attribute")

        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            docs = _docs(self.n_items)
            events = [Event(content=text) for text in docs]
            storage.insert_events(events)
            for start in range(0, len(events), INSERT_BATCH):
                chunk = events[start : start + INSERT_BATCH]
                vectors = embedder.embed([e.content for e in chunk])
                with storage.transaction():
                    for e, v in zip(chunk, vectors, strict=True):
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
            for _ in range(N_WARMUP):
                memory.retrieve("warmup", k=K, reinforce=False)

            per_call_ms: list[float] = []
            for i in range(self.n_queries):
                idx = (i * 113) % self.n_items
                t0 = time.perf_counter()
                memory.retrieve(f"sample fact number {idx}", k=K, reinforce=False)
                per_call_ms.append((time.perf_counter() - t0) * 1000.0)

            p50 = _percentile(per_call_ms, 50.0)
            p95 = _percentile(per_call_ms, 95.0)
            p99 = _percentile(per_call_ms, 99.0)
            mx = max(per_call_ms)

            metrics: dict[str, float] = {
                "retrieve_p50_ms": p50,
                "retrieve_p95_ms": p95,
                "retrieve_p99_ms": p99,
                "retrieve_max_ms": mx,
                "n_items": float(self.n_items),
                "n_queries": float(self.n_queries),
                "dim": float(embedder.dim),
            }
            cis: dict[str, tuple[float, float]] = {k: (v, v) for k, v in metrics.items()}
            per_question: list[dict[str, Any]] = [
                {"i": i, "ms": ms} for i, ms in enumerate(per_call_ms)
            ]
            return SuiteResult(
                name=self.name,
                aggregate_metrics=metrics,
                confidence_intervals=cis,
                per_question=per_question,
                latency_ms={"retrieve": per_call_ms},
            )
        finally:
            storage.close()

    def teardown(self) -> None:
        self._provider = None


SUITE: LatencyAtScaleSuite = LatencyAtScaleSuite()
