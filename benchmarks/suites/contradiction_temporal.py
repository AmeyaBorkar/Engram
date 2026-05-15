"""Stage 8 adversarial benchmark: contradiction + temporal reasoning.

Two synthetic splits, both designed to expose what flat-RAG memory
systems do badly:

  1. **Contradiction adversarial.** N pairs of contradicting facts
     (A_i, B_i) are seeded as memory items with identical embeddings
     so they always co-retrieve. The CONTRADICT relationship is
     recorded as a `Conflict` row (status=OPEN).
       * Baseline score (pre-reconcile): the agent retrieves the pair
         and gets confused -- both facts are visible, the agent has no
         way to know which to trust. Score = "correctly returns only
         the survivor" rate, which is 0 before reconcile.
       * Engram score (post-reconcile with PREFER_RECENT): the loser
         is invalidated; default retrieve returns only the survivor.
         Score should be 1.0.
     The lift (engram - baseline) is the headline.

  2. **Temporal-shift.** N triples (v1 at t1, v2 at t2 invalidating
     v1, v3 at t3 invalidating v2). All three versions share an
     embedding. The agent is queried at three snapshots (t1.5, t2.5,
     t3.5); the correct answer is whichever version was current at
     that timestamp. Score = accuracy over 3N (item, snapshot) pairs.

DoD: the suite proves that contradictions are *observable* (the
storage layer exposes them via `list_conflicts`) and *resolvable*
(reconcile invalidates the loser; retrieve stops surfacing it). The
temporal split proves that "as of when?" queries return
historically-correct state.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from engram import (
    Conflict,
    Embedding,
    ItemKind,
    Level,
    Memory,
    MemoryItem,
    Resolution,
    SqliteStorage,
)
from engram.bench import Provider, SuiteResult
from engram.reconcile import Reconciler


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pair:
    """One contradicting pair. Both items share the cluster_text used as
    the retrieval query; only the content (`a_text` vs `b_text`) and the
    "winner_is_b" flag differ."""

    cluster_text: str
    a_text: str
    b_text: str
    winner_is_b: bool


CONTRADICTION_PAIRS: tuple[_Pair, ...] = (
    _Pair(
        cluster_text="user has a dog or a cat",
        a_text="user has a cat named Mittens",
        b_text="user has a dog named Rex",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's deployment region",
        a_text="user deploys to us-east-1",
        b_text="user deploys to eu-west-1",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's preferred database",
        a_text="user prefers Postgres for production",
        b_text="user prefers MySQL for production",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's team size",
        a_text="user manages a team of 4 engineers",
        b_text="user manages a team of 12 engineers",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's primary language",
        a_text="user codes mostly in Python",
        b_text="user codes mostly in Rust",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's office location",
        a_text="user works from the SF office",
        b_text="user works fully remote from Vermont",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's release cadence",
        a_text="user ships releases monthly",
        b_text="user ships releases weekly",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's CI provider",
        a_text="user uses GitHub Actions for CI",
        b_text="user uses Buildkite for CI",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's note-taking app",
        a_text="user takes notes in Notion",
        b_text="user takes notes in Obsidian",
        winner_is_b=True,
    ),
    _Pair(
        cluster_text="user's cloud provider",
        a_text="user runs production on AWS",
        b_text="user runs production on GCP",
        winner_is_b=True,
    ),
)


@dataclass(frozen=True)
class _Triple:
    """A three-version temporal fact: v1 at t1, v2 at t2 (invalidates v1),
    v3 at t3 (invalidates v2)."""

    cluster_text: str
    v1: str
    t1: datetime
    v2: str
    t2: datetime
    v3: str
    t3: datetime


TEMPORAL_TRIPLES: tuple[_Triple, ...] = (
    _Triple(
        cluster_text="user's role",
        v1="user is a backend engineer",
        t1=_utc(2026, 1, 1),
        v2="user is a tech lead",
        t2=_utc(2026, 4, 1),
        v3="user is an engineering manager",
        t3=_utc(2026, 7, 1),
    ),
    _Triple(
        cluster_text="user's stack",
        v1="user's stack is Django + Postgres",
        t1=_utc(2026, 1, 1),
        v2="user's stack is FastAPI + Postgres",
        t2=_utc(2026, 4, 1),
        v3="user's stack is FastAPI + DuckDB",
        t3=_utc(2026, 7, 1),
    ),
    _Triple(
        cluster_text="user's company",
        v1="user works at Acme Corp",
        t1=_utc(2026, 1, 1),
        v2="user works at Globex Co",
        t2=_utc(2026, 4, 1),
        v3="user works at Initech",
        t3=_utc(2026, 7, 1),
    ),
    _Triple(
        cluster_text="user's location",
        v1="user is based in Boston",
        t1=_utc(2026, 1, 1),
        v2="user is based in Austin",
        t2=_utc(2026, 4, 1),
        v3="user is based in Lisbon",
        t3=_utc(2026, 7, 1),
    ),
    _Triple(
        cluster_text="user's pet",
        v1="user has a goldfish",
        t1=_utc(2026, 1, 1),
        v2="user has a cat",
        t2=_utc(2026, 4, 1),
        v3="user has a dog",
        t3=_utc(2026, 7, 1),
    ),
)


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def _dataset_checksum() -> str:
    h = hashlib.sha256()
    for p in CONTRADICTION_PAIRS:
        h.update(p.cluster_text.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.a_text.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.b_text.encode("utf-8"))
        h.update(b"\x01")
    for t in TEMPORAL_TRIPLES:
        for s in (t.cluster_text, t.v1, t.v2, t.v3):
            h.update(s.encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\x02")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_vec(vec: Sequence[float]) -> tuple[float, ...]:
    import math

    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return tuple(x / n for x in vec)


def _seed_memory_item(
    storage: SqliteStorage,
    *,
    content: str,
    embedding_text: str,
    model: str,
    dim: int,
    vec_source: Any,
    created_at: datetime,
) -> MemoryItem:
    item = MemoryItem(
        level=Level.SUMMARY,
        content=content,
        created_at=created_at,
        valid_from=created_at,
    )
    storage.insert_memory_item(item)
    raw_vec = vec_source.embed([embedding_text])[0]
    storage.insert_embedding(
        Embedding(
            item_id=item.id,
            item_kind=ItemKind.MEMORY_ITEM,
            model=model,
            dim=dim,
            vector=_norm_vec(raw_vec),
        )
    )
    return item


def _retrieve_ids(
    memory: Memory,
    query: str,
    *,
    k: int,
    as_of: datetime | None = None,
) -> list[UUID]:
    results = memory.retrieve(
        query,
        k=k,
        prefer="general",
        confidence_threshold=0.0,
        reinforce=False,
        as_of=as_of,
    )
    return [r.item_id for r in results]


# ---------------------------------------------------------------------------
# The two scores
# ---------------------------------------------------------------------------


def _contradiction_scores(
    memory: Memory, storage: SqliteStorage
) -> tuple[float, float, float, int, int, bool]:
    """Returns:
      (pre_reconcile_score, post_reconcile_score, lift,
       n_observed_open, n_resolved, n_observed_truncated)

    Naming note: the audit (M-164) called out that "baseline vs Engram"
    was misleading -- both measurements come from the same engine; only
    the reconcile step differs. The honest naming is
    "pre-reconcile vs post-reconcile". A third-party baseline (e.g. a
    raw vector store with no conflict-tracking) would live in a
    separate suite.
    """
    # Seed every pair: A first (older), B second (newer). Record the
    # CONTRADICT. Baseline measurement happens before any reconcile.
    pair_records: list[tuple[_Pair, MemoryItem, MemoryItem, Conflict]] = []
    for i, pair in enumerate(CONTRADICTION_PAIRS):
        a_when = _utc(2026, 1, 1, i % 24)
        b_when = _utc(2026, 4, 1, i % 24)
        a = _seed_memory_item(
            storage,
            content=pair.a_text,
            embedding_text=pair.cluster_text,
            model=memory.embedder.model,
            dim=memory.embedder.dim,
            vec_source=memory.embedder,
            created_at=a_when,
        )
        b = _seed_memory_item(
            storage,
            content=pair.b_text,
            embedding_text=pair.cluster_text,
            model=memory.embedder.model,
            dim=memory.embedder.dim,
            vec_source=memory.embedder,
            created_at=b_when,
        )
        conflict = Conflict(
            source_item_id=b.id, target_item_id=a.id, similarity=1.0
        )
        storage.record_conflict(conflict)
        pair_records.append((pair, a, b, conflict))

    # Observability check: every seeded conflict shows up as OPEN.
    # Use a limit ABOVE len(pair_records) so we never silently truncate;
    # if the result hits the limit anyway, flag it on the manifest so a
    # reader knows the number is a lower bound. Pre-audit a hard 1000
    # cap meant the metric could go stale once seeded pairs exceeded it
    # without any signal.
    from engram import ConflictStatus

    observability_limit = max(2 * len(pair_records) + 1, 1024)
    open_rows = storage.list_conflicts(
        status=ConflictStatus.OPEN, limit=observability_limit
    )
    n_observed_open = len(open_rows)
    n_observed_truncated = bool(n_observed_open >= observability_limit)

    # Pre-reconcile measurement: score is "retrieve surfaces ONLY the
    # winner". With no reconcile, both items are visible at
    # retrieve(k=10), so this is 0. Same metric as the post-reconcile
    # number below to keep them apples-to-apples.
    pre_hits = 0
    for pair, a, b, _ in pair_records:
        ids = _retrieve_ids(memory, pair.cluster_text, k=10)
        expected_winner = b.id if pair.winner_is_b else a.id
        expected_loser = a.id if pair.winner_is_b else b.id
        if expected_winner in ids and expected_loser not in ids:
            pre_hits += 1
    pre_reconcile_score = pre_hits / len(pair_records)

    # Reconcile everything: PREFER_RECENT means B wins (we seeded B at
    # 2026-04 vs A at 2026-01).
    reconciler = Reconciler(storage)
    resolved = 0
    for _, _, _, conflict in pair_records:
        reconciler.reconcile(
            conflict.id,
            resolution=Resolution.PREFER_RECENT,
            now=_utc(2026, 5, 1),
        )
        resolved += 1

    # Post-reconcile measurement: retrieve should surface only the
    # winner. Score = "loser is gone AND winner is on top" rate.
    post_hits = 0
    for pair, a, b, _ in pair_records:
        ids = _retrieve_ids(memory, pair.cluster_text, k=10)
        expected_winner = b.id if pair.winner_is_b else a.id
        expected_loser = a.id if pair.winner_is_b else b.id
        if expected_winner in ids and expected_loser not in ids:
            post_hits += 1
    post_reconcile_score = post_hits / len(pair_records)

    return (
        pre_reconcile_score,
        post_reconcile_score,
        post_reconcile_score - pre_reconcile_score,
        n_observed_open,
        resolved,
        n_observed_truncated,
    )


def _temporal_score(memory: Memory, storage: SqliteStorage) -> float:
    """Build N temporal triples, ask the agent at three snapshots each.
    Return accuracy = correct_snapshots / total_snapshots."""
    reconciler = Reconciler(storage)
    per_triple_correct: list[bool] = []
    for triple in TEMPORAL_TRIPLES:
        v1 = _seed_memory_item(
            storage,
            content=triple.v1,
            embedding_text=triple.cluster_text,
            model=memory.embedder.model,
            dim=memory.embedder.dim,
            vec_source=memory.embedder,
            created_at=triple.t1,
        )
        v2 = _seed_memory_item(
            storage,
            content=triple.v2,
            embedding_text=triple.cluster_text,
            model=memory.embedder.model,
            dim=memory.embedder.dim,
            vec_source=memory.embedder,
            created_at=triple.t2,
        )
        c12 = Conflict(source_item_id=v2.id, target_item_id=v1.id, similarity=1.0)
        storage.record_conflict(c12)
        # Invalidate v1 right when v2 lands -- the conflict resolves at t2.
        reconciler.reconcile(c12.id, resolution=Resolution.PREFER_RECENT, now=triple.t2)

        v3 = _seed_memory_item(
            storage,
            content=triple.v3,
            embedding_text=triple.cluster_text,
            model=memory.embedder.model,
            dim=memory.embedder.dim,
            vec_source=memory.embedder,
            created_at=triple.t3,
        )
        c23 = Conflict(source_item_id=v3.id, target_item_id=v2.id, similarity=1.0)
        storage.record_conflict(c23)
        reconciler.reconcile(c23.id, resolution=Resolution.PREFER_RECENT, now=triple.t3)

        # Snapshot mid-windows. At t1+15d only v1 should win; at t2+15d
        # only v2; at t3+15d only v3.
        from datetime import timedelta

        for snapshot, expected_id in (
            (triple.t1 + timedelta(days=15), v1.id),
            (triple.t2 + timedelta(days=15), v2.id),
            (triple.t3 + timedelta(days=15), v3.id),
        ):
            ids = _retrieve_ids(memory, triple.cluster_text, k=1, as_of=snapshot)
            per_triple_correct.append(bool(ids) and ids[0] == expected_id)

    if not per_triple_correct:
        return 0.0
    return sum(1 for ok in per_triple_correct if ok) / len(per_triple_correct)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class ContradictionTemporalSuite:
    name: str = "contradiction-temporal"
    dataset_version: str = "synthetic-v1"

    def __init__(self) -> None:
        self._provider: Provider | None = None
        self.dataset_checksum: str = _dataset_checksum()

    def setup(self, provider: Provider) -> None:
        self._provider = provider

    def run(self) -> SuiteResult:
        if self._provider is None:
            raise RuntimeError("setup() must be called before run()")
        embedder = getattr(self._provider, "embedder", None)
        if embedder is None:
            raise RuntimeError(
                "contradiction-temporal requires a provider with an `embedder` attribute"
            )

        # --- contradiction split ---
        storage_c = SqliteStorage(":memory:")
        storage_c.initialize()
        try:
            memory_c = Memory(storage=storage_c, embedder=embedder)
            t0 = time.perf_counter()
            (
                pre_reconcile_score,
                post_reconcile_score,
                lift,
                n_open,
                n_resolved,
                n_observed_truncated,
            ) = _contradiction_scores(memory_c, storage_c)
            contradiction_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            storage_c.close()

        # --- temporal split ---
        storage_t = SqliteStorage(":memory:")
        storage_t.initialize()
        try:
            memory_t = Memory(storage=storage_t, embedder=embedder)
            t0 = time.perf_counter()
            temporal = _temporal_score(memory_t, storage_t)
            temporal_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            storage_t.close()

        metrics: dict[str, float] = {
            # Pre-audit naming was "baseline_score" / "engram_score",
            # but both come from the same engine; only the reconcile
            # step differs. The honest naming is pre/post-reconcile.
            # The old keys are kept as aliases so existing SCOREBOARD
            # scrapers don't break; new readers should prefer the
            # `*_reconcile_score` names.
            "pre_reconcile_score": pre_reconcile_score,
            "post_reconcile_score": post_reconcile_score,
            "baseline_score": pre_reconcile_score,
            "engram_score": post_reconcile_score,
            "lift": lift,
            "temporal_accuracy": temporal,
            "n_pairs": float(len(CONTRADICTION_PAIRS)),
            "n_observed_open_conflicts": float(n_open),
            "n_observed_open_truncated": 1.0 if n_observed_truncated else 0.0,
            "n_resolved_conflicts": float(n_resolved),
            "n_temporal_triples": float(len(TEMPORAL_TRIPLES)),
        }
        cis: dict[str, tuple[float, float]] = {k: (v, v) for k, v in metrics.items()}

        # Light per-question shape so the manifest writer has something
        # to bind per-row. One entry per contradiction pair plus one per
        # temporal triple.
        per_question: list[dict[str, Any]] = []
        for i, pair in enumerate(CONTRADICTION_PAIRS):
            per_question.append(
                {
                    "split": "contradiction",
                    "id": i,
                    "cluster_text": pair.cluster_text,
                }
            )
        for j, triple in enumerate(TEMPORAL_TRIPLES):
            per_question.append(
                {
                    "split": "temporal",
                    "id": j,
                    "cluster_text": triple.cluster_text,
                }
            )

        return SuiteResult(
            name=self.name,
            aggregate_metrics=metrics,
            confidence_intervals=cis,
            per_question=per_question,
            latency_ms={
                "contradiction_split_ms": [contradiction_ms],
                "temporal_split_ms": [temporal_ms],
            },
        )

    def teardown(self) -> None:
        self._provider = None


SUITE: ContradictionTemporalSuite = ContradictionTemporalSuite()
