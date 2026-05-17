"""Recency-boost vectorization correctness (M-98).

The boost was a Python loop with per-candidate `math.exp`; the
audit asked for a numpy-vectorized form. Math must be bit-identical
(within floating-point tolerance) so existing rankings don't shift.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from engram.retrieve._engine import HierarchicalRetriever, _Candidate
from engram.retrieve._params import RetrieveParams
from engram.schemas import ItemKind, Level


class _FakeStorage:
    """Storage stub that returns a known `created_at` per candidate id."""

    def __init__(self, created_at_map: dict[UUID, datetime]) -> None:
        self._created_at_map = created_at_map

    def get_created_at_batch(self, items: Sequence[tuple[UUID, ItemKind]]) -> dict[UUID, datetime]:
        return {iid: self._created_at_map[iid] for iid, _ in items if iid in self._created_at_map}


class _FakeEmbedder:
    model = "fake"


def _make_retriever(created_at_map: dict[UUID, datetime]) -> HierarchicalRetriever:
    storage = _FakeStorage(created_at_map)
    embedder = _FakeEmbedder()
    # The retriever is purely a function-host for `_apply_recency_boost`.
    return HierarchicalRetriever(
        storage=storage,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
    )


def _cand(content: str = "x") -> _Candidate:
    return _Candidate(
        item_id=uuid4(),
        item_kind=ItemKind.EVENT,
        level=Level.EVENT,
        content=content,
        score=1.0,
        supported_by=(),
    )


class TestRecencyBoostMath:
    def test_zero_lambda_passes_scores_through(self) -> None:
        retriever = _make_retriever({})
        cands = [_cand(), _cand()]
        scores = [1.0, 2.0]
        p = RetrieveParams(recency_lambda=0.0)
        out = retriever._apply_recency_boost(cands, scores, p)  # type: ignore[reportPrivateUsage]
        assert out == [1.0, 2.0]

    def test_zero_day_old_gets_full_lambda(self) -> None:
        now = datetime(2026, 5, 15, tzinfo=timezone.utc)
        c = _cand()
        retriever = _make_retriever({c.item_id: now})
        p = RetrieveParams(recency_lambda=0.5, recency_decay_days=10.0, as_of=now)
        out = retriever._apply_recency_boost([c], [1.0], p)  # type: ignore[reportPrivateUsage]
        # exp(0) = 1; 1.0 + 0.5 * 1.0 = 1.5
        assert out[0] == pytest.approx(1.5, abs=1e-9)

    def test_decay_days_old_gets_one_over_e_lambda(self) -> None:
        now = datetime(2026, 5, 15, tzinfo=timezone.utc)
        c = _cand()
        decay_days = 10.0
        old = now - timedelta(days=decay_days)
        retriever = _make_retriever({c.item_id: old})
        p = RetrieveParams(
            recency_lambda=0.5,
            recency_decay_days=decay_days,
            as_of=now,
        )
        out = retriever._apply_recency_boost([c], [1.0], p)  # type: ignore[reportPrivateUsage]
        # exp(-1) = 1/e; 1.0 + 0.5 * 1/e
        assert out[0] == pytest.approx(1.0 + 0.5 / math.e, abs=1e-9)

    def test_missing_created_at_gets_zero_bonus(self) -> None:
        retriever = _make_retriever({})  # No created_at for any id
        c = _cand()
        p = RetrieveParams(
            recency_lambda=0.5,
            recency_decay_days=10.0,
            as_of=datetime(2026, 5, 15, tzinfo=timezone.utc),
        )
        out = retriever._apply_recency_boost([c], [1.0], p)  # type: ignore[reportPrivateUsage]
        # No record -> no bonus.
        assert out[0] == pytest.approx(1.0, abs=1e-9)

    def test_mixed_candidates_preserve_alignment(self) -> None:
        now = datetime(2026, 5, 15, tzinfo=timezone.utc)
        c_recent = _cand()
        c_old = _cand()
        c_missing = _cand()
        decay_days = 5.0
        retriever = _make_retriever(
            {
                c_recent.item_id: now,
                c_old.item_id: now - timedelta(days=10),
            }
        )
        p = RetrieveParams(
            recency_lambda=1.0,
            recency_decay_days=decay_days,
            as_of=now,
        )
        scores = [2.0, 3.0, 4.0]
        out = retriever._apply_recency_boost([c_recent, c_old, c_missing], scores, p)  # type: ignore[reportPrivateUsage]
        assert len(out) == 3
        # recent: 2.0 + 1.0 * exp(0) = 3.0
        assert out[0] == pytest.approx(3.0, abs=1e-9)
        # 10 days old, decay 5 days: 3.0 + 1.0 * exp(-2) ~= 3.135
        assert out[1] == pytest.approx(3.0 + math.exp(-2), abs=1e-9)
        # missing: 4.0
        assert out[2] == pytest.approx(4.0, abs=1e-9)

    def test_empty_candidates_returns_empty(self) -> None:
        retriever = _make_retriever({})
        p = RetrieveParams(recency_lambda=0.5)
        out = retriever._apply_recency_boost([], [], p)  # type: ignore[reportPrivateUsage]
        assert out == []
