"""Schema-level tests for `DecayState`.

The mutating engine and storage paths live in their own commits; this
module just guards the model contract: bounds, frozen-ness, round-trip.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import engram
from engram.ids import new_id
from engram.schemas import DecayState, ItemKind


def _state(**overrides: object) -> DecayState:
    base: dict[str, object] = {
        "item_id": new_id(),
        "item_kind": ItemKind.EVENT,
        "last_decayed_at": datetime.now(tz=timezone.utc),
    }
    base.update(overrides)
    return DecayState(**base)  # type: ignore[arg-type]


class TestDecayState:
    def test_defaults(self) -> None:
        state = _state()
        assert state.weight == 1.0
        assert state.reinforcement_count == 0
        assert state.corroboration_count == 0
        assert state.contradiction_count == 0
        assert state.cold_at is None

    def test_weight_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _state(weight=-0.01)
        with pytest.raises(ValidationError):
            _state(weight=1.01)

    @pytest.mark.parametrize(
        "field",
        ["reinforcement_count", "corroboration_count", "contradiction_count"],
    )
    def test_counters_non_negative(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _state(**{field: -1})

    def test_frozen(self) -> None:
        state = _state()
        with pytest.raises(ValidationError):
            state.weight = 0.5  # type: ignore[misc]

    def test_round_trip(self) -> None:
        state = _state(
            weight=0.42,
            reinforcement_count=3,
            corroboration_count=1,
            contradiction_count=2,
            cold_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
        )
        clone = DecayState.model_validate(state.model_dump())
        assert clone == state


class TestPublicReexports:
    def test_engram_exposes_decay_symbols(self) -> None:
        assert engram.DecayParams is not None
        assert engram.DecayState is not None
        # Sanity check: instantiation works through the package surface.
        params = engram.DecayParams()
        assert 0.0 < params.alpha < 1.0
