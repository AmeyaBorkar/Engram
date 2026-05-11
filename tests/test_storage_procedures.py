"""SQLite storage surface for procedures (Stage 7).

Covers:
  * `insert_procedure` / `get_procedure` round-trip preserves every field.
  * `list_procedures` orders by created_at desc and filters by outcome.
  * `update_procedure_outcome` flips the row + bumps `updated_at` +
    invalidates the vector index shard for procedures.
  * `count_procedures` / `count_procedures_by_outcome` aggregate correctly.
  * `search_procedure_embeddings` does cosine top-k over situations,
    respects outcome filter and cold filter.
  * The unified decay-state surface (get/update/iter/mark_cold/...) works
    for `ItemKind.PROCEDURE` via the per-kind SQL lookup dicts.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engram import (
    Embedding,
    ItemKind,
    Outcome,
    Procedure,
    SqliteStorage,
)
from engram.providers._fake import FakeEmbedder


@pytest.fixture
def emb() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestInsertGet:
    def test_round_trip_preserves_fields(self, storage: SqliteStorage) -> None:
        p = Procedure(
            situation="user asks for code review of Python file",
            action="read file, list issues by severity",
            outcome=Outcome.SUCCESS,
            weight=0.85,
            metadata={"trace_id": "abc"},
        )
        storage.insert_procedure(p)
        fetched = storage.get_procedure(p.id)
        assert fetched is not None
        assert fetched.id == p.id
        assert fetched.situation == p.situation
        assert fetched.action == p.action
        assert fetched.outcome is Outcome.SUCCESS
        assert fetched.weight == pytest.approx(0.85)
        assert fetched.metadata == {"trace_id": "abc"}

    def test_get_missing_returns_none(self, storage: SqliteStorage) -> None:
        from uuid import uuid4

        assert storage.get_procedure(uuid4()) is None


class TestListProcedures:
    def test_orders_by_created_at_desc(self, storage: SqliteStorage) -> None:
        a = Procedure(
            situation="a", action="a", created_at=datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
        b = Procedure(
            situation="b", action="b", created_at=datetime(2025, 6, 1, tzinfo=timezone.utc)
        )
        c = Procedure(
            situation="c", action="c", created_at=datetime(2025, 3, 1, tzinfo=timezone.utc)
        )
        for p in (a, b, c):
            storage.insert_procedure(p)
        listed = storage.list_procedures()
        # Newest first.
        assert [p.situation for p in listed] == ["b", "c", "a"]

    def test_outcome_filter(self, storage: SqliteStorage) -> None:
        storage.insert_procedure(Procedure(situation="s1", action="a", outcome=Outcome.SUCCESS))
        storage.insert_procedure(Procedure(situation="f1", action="a", outcome=Outcome.FAILURE))
        storage.insert_procedure(Procedure(situation="s2", action="a", outcome=Outcome.SUCCESS))
        successes = storage.list_procedures(outcome=Outcome.SUCCESS)
        assert {p.situation for p in successes} == {"s1", "s2"}

    def test_limit(self, storage: SqliteStorage) -> None:
        for i in range(5):
            storage.insert_procedure(Procedure(situation=f"s{i}", action=f"a{i}"))
        assert len(storage.list_procedures(limit=3)) == 3

    def test_invalid_limit_raises(self, storage: SqliteStorage) -> None:
        with pytest.raises(ValueError, match="limit"):
            storage.list_procedures(limit=0)


class TestUpdateOutcome:
    def test_flips_outcome_and_bumps_updated_at(self, storage: SqliteStorage) -> None:
        p = Procedure(
            situation="s",
            action="a",
            outcome=Outcome.UNKNOWN,
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        storage.insert_procedure(p)
        storage.update_procedure_outcome(p.id, Outcome.SUCCESS)
        fetched = storage.get_procedure(p.id)
        assert fetched is not None
        assert fetched.outcome is Outcome.SUCCESS
        assert fetched.updated_at > p.updated_at

    def test_missing_id_raises_key_error(self, storage: SqliteStorage) -> None:
        from uuid import uuid4

        with pytest.raises(KeyError):
            storage.update_procedure_outcome(uuid4(), Outcome.SUCCESS)


class TestCounts:
    def test_count_procedures(self, storage: SqliteStorage) -> None:
        assert storage.count_procedures() == 0
        for i in range(3):
            storage.insert_procedure(Procedure(situation=f"s{i}", action="a"))
        assert storage.count_procedures() == 3

    def test_count_by_outcome(self, storage: SqliteStorage) -> None:
        storage.insert_procedure(Procedure(situation="s1", action="a", outcome=Outcome.SUCCESS))
        storage.insert_procedure(Procedure(situation="s2", action="a", outcome=Outcome.SUCCESS))
        storage.insert_procedure(Procedure(situation="f1", action="a", outcome=Outcome.FAILURE))
        counts = storage.count_procedures_by_outcome()
        assert counts[Outcome.SUCCESS] == 2
        assert counts[Outcome.FAILURE] == 1
        assert counts[Outcome.PARTIAL] == 0
        assert counts[Outcome.UNKNOWN] == 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchProcedureEmbeddings:
    def test_top_k_by_cosine(self, storage: SqliteStorage, emb: FakeEmbedder) -> None:
        # Plant 3 procedures with distinct situations.
        situations = [
            "user reports flaky test in CI",
            "user asks about Spanish irregular verbs",
            "user wants help debugging a Python import error",
        ]
        ids = []
        for s in situations:
            p = Procedure(situation=s, action="a")
            storage.insert_procedure(p)
            vec = tuple(emb.embed([s])[0])
            storage.insert_embedding(
                Embedding(
                    item_id=p.id,
                    item_kind=ItemKind.PROCEDURE,
                    model=emb.model,
                    dim=emb.dim,
                    vector=vec,
                )
            )
            ids.append(p.id)

        # Query close to situation #0 ("flaky test").
        qvec = tuple(emb.embed(["user reports flaky test in CI"])[0])
        hits = storage.search_procedure_embeddings(qvec, k=3, model=emb.model)
        assert len(hits) == 3
        # Exact match should be top.
        assert hits[0][0] == ids[0]
        assert hits[0][1] == situations[0]
        assert hits[0][2] == pytest.approx(1.0, abs=1e-5)

    def test_outcome_filter(self, storage: SqliteStorage, emb: FakeEmbedder) -> None:
        ok = Procedure(situation="ok", action="a", outcome=Outcome.SUCCESS)
        bad = Procedure(situation="bad", action="a", outcome=Outcome.FAILURE)
        for p, s in ((ok, "ok"), (bad, "bad")):
            storage.insert_procedure(p)
            vec = tuple(emb.embed([s])[0])
            storage.insert_embedding(
                Embedding(
                    item_id=p.id,
                    item_kind=ItemKind.PROCEDURE,
                    model=emb.model,
                    dim=emb.dim,
                    vector=vec,
                )
            )
        qvec = tuple(emb.embed(["ok"])[0])
        # Only successes.
        hits = storage.search_procedure_embeddings(
            qvec, k=10, model=emb.model, outcomes=[Outcome.SUCCESS]
        )
        assert [h[0] for h in hits] == [ok.id]
        # Only failures.
        hits = storage.search_procedure_embeddings(
            qvec, k=10, model=emb.model, outcomes=[Outcome.FAILURE]
        )
        assert [h[0] for h in hits] == [bad.id]

    def test_invalid_k_raises(self, storage: SqliteStorage, emb: FakeEmbedder) -> None:
        with pytest.raises(ValueError, match="k must be"):
            storage.search_procedure_embeddings([0.0] * emb.dim, k=0, model=emb.model)

    def test_outcome_filter_picks_up_update(
        self, storage: SqliteStorage, emb: FakeEmbedder
    ) -> None:
        """update_procedure_outcome must invalidate the procedure index
        shard so the next search sees the new outcome."""
        p = Procedure(situation="x", action="a", outcome=Outcome.UNKNOWN)
        storage.insert_procedure(p)
        vec = tuple(emb.embed(["x"])[0])
        storage.insert_embedding(
            Embedding(
                item_id=p.id,
                item_kind=ItemKind.PROCEDURE,
                model=emb.model,
                dim=emb.dim,
                vector=vec,
            )
        )
        qvec = tuple(emb.embed(["x"])[0])
        # Initially UNKNOWN; filter to SUCCESS returns nothing.
        assert (
            storage.search_procedure_embeddings(
                qvec, k=10, model=emb.model, outcomes=[Outcome.SUCCESS]
            )
            == []
        )
        # Update to SUCCESS; same filter now returns the row.
        storage.update_procedure_outcome(p.id, Outcome.SUCCESS)
        hits = storage.search_procedure_embeddings(
            qvec, k=10, model=emb.model, outcomes=[Outcome.SUCCESS]
        )
        assert [h[0] for h in hits] == [p.id]


# ---------------------------------------------------------------------------
# Decay-state surface works for procedures
# ---------------------------------------------------------------------------


class TestDecayStateForProcedures:
    def test_get_decay_state(self, storage: SqliteStorage) -> None:
        p = Procedure(situation="s", action="a")
        storage.insert_procedure(p)
        state = storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.item_kind is ItemKind.PROCEDURE
        assert state.weight == 1.0
        assert state.reinforcement_count == 0
        assert state.cold_at is None

    def test_mark_cold_and_count(self, storage: SqliteStorage) -> None:
        from datetime import datetime, timezone

        p = Procedure(situation="s", action="a")
        storage.insert_procedure(p)
        storage.mark_cold(p.id, ItemKind.PROCEDURE, at=datetime.now(tz=timezone.utc))
        assert storage.count_cold(ItemKind.PROCEDURE) == 1

    def test_decay_totals(self, storage: SqliteStorage) -> None:
        for i in range(2):
            storage.insert_procedure(Procedure(situation=f"s{i}", action="a"))
        totals = storage.decay_totals(ItemKind.PROCEDURE)
        assert totals["hot_items"] == 2
        assert totals["cold_items"] == 0
        assert totals["reinforcement_total"] == 0
