"""Hypothesis-driven property tests for Stage 8 invariants.

The headline invariants this file pins:

  * **Validity-window semantics.** Given an item with (valid_from,
    valid_until, invalidated_at), the item is visible at `as_of` iff
    `valid_from <= as_of`, `valid_until is None OR valid_until > as_of`,
    AND `invalidated_at is None OR invalidated_at > as_of`. Mirrors the
    SQL predicate exactly so divergence is a test failure.

  * **Invalidation idempotency.** `invalidate_memory_item` preserves the
    FIRST timestamp under any number of subsequent calls. `as_of`
    queries depend on this for replayability.

  * **Reconcile preserves winner-is-source-or-target.** Regardless of
    `Resolution` and the (random) trust / corroboration / recency
    values, the resolved winner is one of the two parties (or None for
    KEEP_BOTH).

  * **MemoryItem temporal invariants.** valid_from defaults to
    created_at; valid_until >= valid_from is enforced under arbitrary
    setter combinations.

  * **Conflict status machine.** OPEN <-> no resolution/winner/at;
    RESOLVED <-> has resolution + resolved_at; resolved_winner_id None
    iff resolution is KEEP_BOTH.
"""

from __future__ import annotations

import string
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from engram import (
    Conflict,
    ConflictStatus,
    DecayState,
    ItemKind,
    Level,
    MemoryItem,
    Resolution,
    SqliteStorage,
    Storage,
    new_id,
)
from engram.reconcile import Reconciler

_settings = settings(
    max_examples=50,
    deadline=2000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_text = st.text(alphabet=string.printable, min_size=1, max_size=80)


# Datetime range bounded so we don't run into pydantic / SQL limits.
_dt = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
).map(lambda d: d.replace(tzinfo=timezone.utc))


def _is_valid_at(item: MemoryItem, as_of: datetime) -> bool:
    """The visibility predicate that
    `search_memory_item_embeddings_as_of` implements in SQL. Pure
    Python here so the property test reads as a spec."""
    if item.valid_from is not None and as_of < item.valid_from:
        return False
    if item.valid_until is not None and as_of >= item.valid_until:
        return False
    if item.invalidated_at is not None and as_of >= item.invalidated_at:
        return False
    return True


# ---------------------------------------------------------------------------
# MemoryItem temporal invariants
# ---------------------------------------------------------------------------


class TestMemoryItemTemporalInvariants:
    @given(content=_text)
    @_settings
    def test_valid_from_defaults_to_created_at(self, content: str) -> None:
        item = MemoryItem(level=Level.SUMMARY, content=content)
        assert item.valid_from == item.created_at

    @given(
        content=_text,
        vf=_dt,
        delta=st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=365)),
    )
    @_settings
    def test_valid_until_after_valid_from_accepted(
        self, content: str, vf: datetime, delta: timedelta
    ) -> None:
        vu = vf + delta
        item = MemoryItem(level=Level.SUMMARY, content=content, valid_from=vf, valid_until=vu)
        assert item.valid_from == vf
        assert item.valid_until == vu

    @given(
        content=_text,
        vf=_dt,
        delta=st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=365)),
    )
    @_settings
    def test_valid_until_before_valid_from_rejected(
        self, content: str, vf: datetime, delta: timedelta
    ) -> None:
        bad_vu = vf - delta
        with pytest.raises(ValueError, match="precedes"):
            MemoryItem(
                level=Level.SUMMARY,
                content=content,
                valid_from=vf,
                valid_until=bad_vu,
            )

    @given(content=_text, when=_dt)
    @_settings
    def test_invalidated_by_without_at_rejected(self, content: str, when: datetime) -> None:
        winner = new_id()
        with pytest.raises(ValueError, match="invalidated_by"):
            MemoryItem(
                level=Level.SUMMARY,
                content=content,
                invalidated_by=winner,
            )
        # The valid pairing is accepted.
        MemoryItem(
            level=Level.SUMMARY,
            content=content,
            invalidated_at=when,
            invalidated_by=winner,
        )


# ---------------------------------------------------------------------------
# Visibility predicate matches SQL via the storage layer
# ---------------------------------------------------------------------------


class TestVisibilityPredicateAgreesWithStorage:
    @given(
        vf=_dt,
        vu_delta=st.one_of(
            st.none(),
            st.timedeltas(min_value=timedelta(days=1), max_value=timedelta(days=200)),
        ),
        inv_delta=st.one_of(
            st.none(),
            st.timedeltas(min_value=timedelta(days=1), max_value=timedelta(days=200)),
        ),
        as_of_offset=st.timedeltas(min_value=timedelta(days=-365), max_value=timedelta(days=365)),
    )
    @_settings
    def test_python_predicate_matches_sql(
        self,
        vf: datetime,
        vu_delta: timedelta | None,
        inv_delta: timedelta | None,
        as_of_offset: timedelta,
    ) -> None:
        from engram.schemas import Embedding

        # Fresh per-example storage: hypothesis runs many iterations
        # against one test function, but the SQLite vector index would
        # accumulate rows across iterations and our k=1 search could
        # then return a different item with the same score.
        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            vu = vf + vu_delta if vu_delta is not None else None
            inv = vf + inv_delta if inv_delta is not None else None
            item = MemoryItem(
                level=Level.SUMMARY,
                content="x",
                created_at=vf,
                valid_from=vf,
                valid_until=vu,
                invalidated_at=inv,
            )
            storage.insert_memory_item(item)
            vec = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            storage.insert_embedding(
                Embedding(
                    item_id=item.id,
                    item_kind=ItemKind.MEMORY_ITEM,
                    model="fake",
                    dim=8,
                    vector=vec,
                )
            )

            as_of = vf + as_of_offset
            python_predicate = _is_valid_at(item, as_of)
            hits = storage.search_memory_item_embeddings_as_of(
                list(vec), k=1, model="fake", as_of=as_of
            )
            sql_visible = bool(hits) and hits[0][0] == item.id
            assert python_predicate == sql_visible, (
                f"divergence: python={python_predicate} sql={sql_visible} "
                f"vf={vf} vu={vu} inv={inv} as_of={as_of}"
            )
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# Invalidation idempotency
# ---------------------------------------------------------------------------


class TestInvalidationIdempotency:
    @given(
        first=_dt,
        offsets=st.lists(
            st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=365)),
            min_size=1,
            max_size=8,
        ),
    )
    @_settings
    def test_first_timestamp_wins(
        self,
        storage: SqliteStorage,
        first: datetime,
        offsets: list[timedelta],
    ) -> None:
        item = MemoryItem(level=Level.SUMMARY, content="x", created_at=first, valid_from=first)
        storage.insert_memory_item(item)
        storage.invalidate_memory_item(item.id, at=first)
        for offset in offsets:
            storage.invalidate_memory_item(item.id, at=first + offset)
        result = storage.get_memory_item(item.id)
        assert result is not None
        assert result.invalidated_at == first

    @given(first=_dt)
    @_settings
    def test_first_winner_id_wins(self, storage: SqliteStorage, first: datetime) -> None:
        item = MemoryItem(level=Level.SUMMARY, content="x", created_at=first, valid_from=first)
        storage.insert_memory_item(item)
        w1, w2 = new_id(), new_id()
        storage.invalidate_memory_item(item.id, at=first, by=w1)
        storage.invalidate_memory_item(item.id, at=first + timedelta(days=1), by=w2)
        result = storage.get_memory_item(item.id)
        assert result is not None
        assert result.invalidated_by == w1


# ---------------------------------------------------------------------------
# Reconcile winner invariant
# ---------------------------------------------------------------------------


def _seed_reconcile_pair(
    storage: Storage,
    source_at: datetime,
    target_at: datetime,
    source_trust: float | None = None,
    target_trust: float | None = None,
    source_corroboration: int = 0,
    target_corroboration: int = 0,
) -> tuple[MemoryItem, MemoryItem, Conflict]:
    source = MemoryItem(
        level=Level.SUMMARY,
        content="src",
        created_at=source_at,
        valid_from=source_at,
        source_trust=source_trust,
    )
    target = MemoryItem(
        level=Level.SUMMARY,
        content="tgt",
        created_at=target_at,
        valid_from=target_at,
        source_trust=target_trust,
    )
    storage.insert_memory_item(source)
    storage.insert_memory_item(target)
    if source_corroboration:
        existing = storage.get_decay_state(source.id, ItemKind.MEMORY_ITEM)
        assert existing is not None
        storage.update_decay_state(
            DecayState(
                item_id=source.id,
                item_kind=ItemKind.MEMORY_ITEM,
                weight=existing.weight,
                reinforcement_count=existing.reinforcement_count,
                corroboration_count=source_corroboration,
                contradiction_count=existing.contradiction_count,
                last_decayed_at=source_at,
                cold_at=existing.cold_at,
            )
        )
    if target_corroboration:
        existing = storage.get_decay_state(target.id, ItemKind.MEMORY_ITEM)
        assert existing is not None
        storage.update_decay_state(
            DecayState(
                item_id=target.id,
                item_kind=ItemKind.MEMORY_ITEM,
                weight=existing.weight,
                reinforcement_count=existing.reinforcement_count,
                corroboration_count=target_corroboration,
                contradiction_count=existing.contradiction_count,
                last_decayed_at=target_at,
                cold_at=existing.cold_at,
            )
        )
    conflict = Conflict(source_item_id=source.id, target_item_id=target.id, similarity=0.9)
    storage.record_conflict(conflict)
    return source, target, conflict


class TestReconcileWinnerInvariants:
    @given(
        source_dt=_dt,
        delta=st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=365)),
        source_trust=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        target_trust=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        source_corr=st.integers(min_value=0, max_value=20),
        target_corr=st.integers(min_value=0, max_value=20),
        resolution=st.sampled_from(
            [
                Resolution.PREFER_RECENT,
                Resolution.PREFER_TRUSTED,
                Resolution.PREFER_FREQUENT,
            ]
        ),
    )
    @_settings
    def test_winner_is_source_or_target(
        self,
        storage: SqliteStorage,
        source_dt: datetime,
        delta: timedelta,
        source_trust: float | None,
        target_trust: float | None,
        source_corr: int,
        target_corr: int,
        resolution: Resolution,
    ) -> None:
        target_dt = source_dt + delta
        source, target, conflict = _seed_reconcile_pair(
            storage,
            source_at=source_dt,
            target_at=target_dt,
            source_trust=source_trust,
            target_trust=target_trust,
            source_corroboration=source_corr,
            target_corroboration=target_corr,
        )
        out = Reconciler(storage).reconcile(
            conflict.id, resolution=resolution, now=target_dt + timedelta(days=1)
        )
        assert out.resolved_winner_id in (source.id, target.id)
        assert out.status is ConflictStatus.RESOLVED
        assert out.resolution is resolution

    @given(
        source_dt=_dt,
        delta=st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=365)),
    )
    @_settings
    def test_keep_both_no_winner(
        self,
        storage: SqliteStorage,
        source_dt: datetime,
        delta: timedelta,
    ) -> None:
        target_dt = source_dt + delta
        _, _, conflict = _seed_reconcile_pair(storage, source_at=source_dt, target_at=target_dt)
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.KEEP_BOTH,
            now=target_dt + timedelta(days=1),
        )
        assert out.resolved_winner_id is None
        assert out.status is ConflictStatus.RESOLVED

    @given(
        source_dt=_dt,
        delta=st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=365)),
    )
    @_settings
    def test_prefer_recent_picks_newer(
        self,
        storage: SqliteStorage,
        source_dt: datetime,
        delta: timedelta,
    ) -> None:
        """PREFER_RECENT picks the side with the later created_at, every time."""
        target_dt = source_dt + delta
        assume(target_dt > source_dt)
        _source, target, conflict = _seed_reconcile_pair(
            storage, source_at=source_dt, target_at=target_dt
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_RECENT,
            now=target_dt + timedelta(days=1),
        )
        assert out.resolved_winner_id == target.id

    @given(
        source_dt=_dt,
        delta=st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=365)),
        source_trust=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        target_trust=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @_settings
    def test_prefer_trusted_picks_higher_trust(
        self,
        storage: SqliteStorage,
        source_dt: datetime,
        delta: timedelta,
        source_trust: float,
        target_trust: float,
    ) -> None:
        """Unless trust ties, PREFER_TRUSTED picks the higher-trust side.

        Audit M-64 changed the trust comparison from strict `!=` to
        `math.isclose(rel_tol=1e-9, abs_tol=1e-12)` so sub-ulp float
        noise post-JSON-round-trip falls back to PREFER_RECENT instead
        of picking arbitrarily on numerical drift. The property test
        respects the new contract by demanding the two values diverge
        beyond that tolerance.
        """
        import math

        assume(not math.isclose(source_trust, target_trust, rel_tol=1e-9, abs_tol=1e-12))
        target_dt = source_dt + delta
        source, target, conflict = _seed_reconcile_pair(
            storage,
            source_at=source_dt,
            target_at=target_dt,
            source_trust=source_trust,
            target_trust=target_trust,
        )
        out = Reconciler(storage).reconcile(
            conflict.id,
            resolution=Resolution.PREFER_TRUSTED,
            now=target_dt + timedelta(days=1),
        )
        expected = source.id if source_trust > target_trust else target.id
        assert out.resolved_winner_id == expected


# ---------------------------------------------------------------------------
# Loser invalidation invariant
# ---------------------------------------------------------------------------


class TestLoserInvalidationInvariant:
    @given(
        source_dt=_dt,
        delta=st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=365)),
        resolution=st.sampled_from(
            [
                Resolution.PREFER_RECENT,
                Resolution.PREFER_TRUSTED,
                Resolution.PREFER_FREQUENT,
            ]
        ),
    )
    @_settings
    def test_loser_invalidated_by_winner(
        self,
        storage: SqliteStorage,
        source_dt: datetime,
        delta: timedelta,
        resolution: Resolution,
    ) -> None:
        target_dt = source_dt + delta
        source, target, conflict = _seed_reconcile_pair(
            storage,
            source_at=source_dt,
            target_at=target_dt,
            source_trust=0.5,
            target_trust=0.7,
            source_corroboration=2,
            target_corroboration=5,
        )
        when = target_dt + timedelta(days=1)
        out = Reconciler(storage).reconcile(conflict.id, resolution=resolution, now=when)
        loser_id = source.id if out.resolved_winner_id == target.id else target.id
        loser = storage.get_memory_item(loser_id)
        assert loser is not None
        assert loser.invalidated_at == when
        assert loser.invalidated_by == out.resolved_winner_id
        # Winner is NOT invalidated.
        assert out.resolved_winner_id is not None
        winner = storage.get_memory_item(out.resolved_winner_id)
        assert winner is not None
        assert winner.invalidated_at is None
