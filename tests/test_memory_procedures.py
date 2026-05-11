"""Stage 7 Memory-level procedure tests.

Exercises:
  * `record_procedure(situation, action, outcome)` writes + embeds +
    fires the right outcome signal through decay.
  * `retrieve_procedures(situation, k)` finds analogous procedures
    ranked by similarity * weight * outcome boost. Successes outrank
    failures at equal similarity but failures stay visible.
  * `update_outcome(id, outcome)` flips outcome + routes the change
    through decay (success -> reinforce, failure -> contradict).
  * Reinforcement-on-retrieve fires for each surfaced procedure;
    `reinforce=False` opts out.
  * Outcome=UNKNOWN is a clean no-op on the decay engine.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from engram import (
    ItemKind,
    Memory,
    Outcome,
    Procedure,
    ProcedureMatch,
    SqliteStorage,
)
from engram.providers._fake import FakeEmbedder


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=64))


# ---------------------------------------------------------------------------
# record_procedure
# ---------------------------------------------------------------------------


class TestRecordProcedure:
    def test_persists_situation_action_outcome(self, memory: Memory) -> None:
        p = memory.record_procedure(
            "user reports a flaky test",
            "rerun with --no-cov; isolate the failure",
            outcome=Outcome.SUCCESS,
        )
        fetched = memory.storage.get_procedure(p.id)
        assert fetched is not None
        assert fetched.situation == "user reports a flaky test"
        assert fetched.action == "rerun with --no-cov; isolate the failure"
        assert fetched.outcome is Outcome.SUCCESS

    def test_writes_situation_embedding(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a")
        emb = memory.storage.get_embedding(
            p.id, ItemKind.PROCEDURE, model=memory.embedder.model
        )
        assert emb is not None
        assert emb.dim == memory.embedder.dim

    def test_success_outcome_increments_reinforcement(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a", outcome=Outcome.SUCCESS)
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.reinforcement_count == 1

    def test_partial_outcome_also_reinforces(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a", outcome=Outcome.PARTIAL)
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.reinforcement_count == 1

    def test_failure_outcome_increments_contradiction(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a", outcome=Outcome.FAILURE)
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.contradiction_count == 1

    def test_unknown_outcome_fires_no_signal(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a")  # outcome defaults to UNKNOWN
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.reinforcement_count == 0
        assert state.contradiction_count == 0
        assert state.corroboration_count == 0


# ---------------------------------------------------------------------------
# retrieve_procedures
# ---------------------------------------------------------------------------


class TestRetrieveProcedures:
    def test_returns_procedure_match_with_score_and_similarity(
        self, memory: Memory
    ) -> None:
        p = memory.record_procedure("user reports flaky test", "rerun with -p no:cov")
        results = memory.retrieve_procedures("user reports flaky test", k=1)
        assert len(results) == 1
        m = results[0]
        assert isinstance(m, ProcedureMatch)
        assert m.procedure.id == p.id
        assert m.similarity == pytest.approx(1.0, abs=1e-5)

    def test_successes_outrank_failures_at_equal_similarity(self, memory: Memory) -> None:
        # Identical situations -> identical similarity. SUCCESS should
        # outrank FAILURE because of the outcome boost.
        ok = memory.record_procedure("same situation", "a", outcome=Outcome.SUCCESS)
        bad = memory.record_procedure("same situation", "b", outcome=Outcome.FAILURE)
        results = memory.retrieve_procedures("same situation", k=2, reinforce=False)
        assert results[0].procedure.id == ok.id
        assert results[1].procedure.id == bad.id

    def test_outcome_filter(self, memory: Memory) -> None:
        memory.record_procedure("s", "good action", outcome=Outcome.SUCCESS)
        memory.record_procedure("s", "bad action", outcome=Outcome.FAILURE)
        successes = memory.retrieve_procedures(
            "s", k=10, outcomes=[Outcome.SUCCESS], reinforce=False
        )
        assert len(successes) == 1
        assert successes[0].procedure.action == "good action"

    def test_reinforce_on_use_default(self, memory: Memory) -> None:
        p = memory.record_procedure("x", "a")
        memory.retrieve_procedures("x", k=1)
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.reinforcement_count == 1  # one from retrieval

    def test_reinforce_false_does_not_bump(self, memory: Memory) -> None:
        p = memory.record_procedure("x", "a")
        memory.retrieve_procedures("x", k=1, reinforce=False)
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.reinforcement_count == 0

    def test_empty_store_returns_empty(self, memory: Memory) -> None:
        assert memory.retrieve_procedures("anything", k=10) == []

    def test_invalid_k_raises(self, memory: Memory) -> None:
        with pytest.raises(ValueError, match="k must be"):
            memory.retrieve_procedures("anything", k=0)


# ---------------------------------------------------------------------------
# update_outcome
# ---------------------------------------------------------------------------


class TestUpdateOutcome:
    def test_unknown_to_success_fires_reinforce(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a")  # UNKNOWN -> no signal
        assert (
            memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE).reinforcement_count == 0
        )
        updated = memory.update_outcome(p.id, Outcome.SUCCESS)
        assert updated.outcome is Outcome.SUCCESS
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.reinforcement_count == 1

    def test_unknown_to_failure_fires_contradict(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a")
        memory.update_outcome(p.id, Outcome.FAILURE)
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.contradiction_count == 1

    def test_unknown_to_unknown_is_noop(self, memory: Memory) -> None:
        p = memory.record_procedure("s", "a")
        memory.update_outcome(p.id, Outcome.UNKNOWN)
        state = memory.storage.get_decay_state(p.id, ItemKind.PROCEDURE)
        assert state is not None
        assert state.reinforcement_count == 0
        assert state.contradiction_count == 0

    def test_missing_id_raises_key_error(self, memory: Memory) -> None:
        with pytest.raises(KeyError):
            memory.update_outcome(uuid4(), Outcome.SUCCESS)

    def test_returned_procedure_has_new_outcome_and_bumped_updated_at(
        self, memory: Memory
    ) -> None:
        p = memory.record_procedure("s", "a")
        before = p.updated_at
        updated = memory.update_outcome(p.id, Outcome.SUCCESS)
        assert updated.outcome is Outcome.SUCCESS
        assert updated.updated_at >= before


# ---------------------------------------------------------------------------
# Integration: outcome-feedback loop end-to-end
# ---------------------------------------------------------------------------


class TestOutcomeFeedbackLoop:
    def test_failed_procedure_eventually_ranks_below_success_after_retrievals(
        self, memory: Memory
    ) -> None:
        """The agent records two competing procedures; success gets
        reinforced via repeated retrievals; failure decays. After the
        loop, success outranks failure even with equal similarity."""
        ok = memory.record_procedure("same situation", "winning action", outcome=Outcome.SUCCESS)
        bad = memory.record_procedure("same situation", "losing action", outcome=Outcome.FAILURE)

        # Retrieve a few times; reinforce-on-use ticks each.
        for _ in range(3):
            results = memory.retrieve_procedures("same situation", k=2)
            assert results[0].procedure.id == ok.id  # success on top from the start

        # Confirm the decay state reflects the bias.
        ok_state = memory.storage.get_decay_state(ok.id, ItemKind.PROCEDURE)
        bad_state = memory.storage.get_decay_state(bad.id, ItemKind.PROCEDURE)
        assert ok_state is not None
        assert bad_state is not None
        # OK got 1 (record) + 3 (retrieves) = 4; bad got 1 contradict
        # at record + 3 retrieve reinforcements = 3 reinforce.
        assert ok_state.reinforcement_count == 4
        assert bad_state.contradiction_count == 1
        assert bad_state.reinforcement_count == 3


# ---------------------------------------------------------------------------
# Direct Procedure model insertion path still works (low-level API)
# ---------------------------------------------------------------------------


def test_low_level_storage_path_remains(storage: SqliteStorage) -> None:
    """Users who bypass Memory and insert procedures directly through
    storage should still get sensible behavior (the storage protocol
    is the lower layer)."""
    p = Procedure(situation="raw insert", action="a", outcome=Outcome.PARTIAL)
    storage.insert_procedure(p)
    assert storage.count_procedures() == 1
    fetched = storage.get_procedure(p.id)
    assert fetched is not None
    assert fetched.outcome is Outcome.PARTIAL
