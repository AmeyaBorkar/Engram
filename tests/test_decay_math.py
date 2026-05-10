"""Tests for `engram.decay._math`.

Coverage target: 100% line + branch on `engram.decay._math`. The pure
math is the only place we require this; the engine and storage paths are
covered to >=90% per the cross-cutting standards.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from engram.decay import DecayParams, apply, clamp01, is_cold

# --- DecayParams validation --------------------------------------------------


class TestDecayParams:
    def test_defaults_are_sane(self) -> None:
        params = DecayParams()
        assert params.half_life_seconds == 30.0 * 86400.0
        assert params.beta == 0.10
        assert params.gamma == 0.05
        assert params.delta == 0.20
        assert params.threshold == 0.05
        # alpha is the per-second base, derived from half-life.
        assert 0.0 < params.alpha < 1.0
        # alpha^half_life == 0.5 by construction.
        assert math.isclose(params.alpha**params.half_life_seconds, 0.5, rel_tol=1e-9, abs_tol=1e-9)

    @pytest.mark.parametrize("hl", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_half_life(self, hl: float) -> None:
        with pytest.raises(ValueError, match="half_life_seconds"):
            DecayParams(half_life_seconds=hl)

    @pytest.mark.parametrize("name", ["beta", "gamma", "delta"])
    def test_rejects_negative_gain(self, name: str) -> None:
        with pytest.raises(ValueError, match=name):
            DecayParams(**{name: -0.1})

    @pytest.mark.parametrize("name", ["beta", "gamma", "delta"])
    def test_rejects_nonfinite_gain(self, name: str) -> None:
        with pytest.raises(ValueError, match=name):
            DecayParams(**{name: float("nan")})
        with pytest.raises(ValueError, match=name):
            DecayParams(**{name: float("inf")})

    @pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
    def test_rejects_bad_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            DecayParams(threshold=threshold)

    def test_is_frozen(self) -> None:
        params = DecayParams()
        with pytest.raises(AttributeError):
            params.beta = 0.5  # type: ignore[misc]


# --- clamp01 ----------------------------------------------------------------


class TestClamp01:
    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.3, 0.3),
            (1.0, 1.0),
            (1.5, 1.0),
        ],
    )
    def test_clamp(self, inp: float, expected: float) -> None:
        assert clamp01(inp) == expected

    def test_nan_to_zero(self) -> None:
        assert clamp01(float("nan")) == 0.0

    def test_neg_inf_to_zero(self) -> None:
        assert clamp01(float("-inf")) == 0.0

    def test_pos_inf_to_one(self) -> None:
        assert clamp01(float("inf")) == 1.0


# --- apply: explicit cases --------------------------------------------------


class TestApplyExplicit:
    def test_no_dt_no_signals_is_identity(self) -> None:
        params = DecayParams()
        assert apply(weight=0.7, dt_seconds=0.0, params=params) == 0.7

    def test_pure_decay_lowers_weight(self) -> None:
        params = DecayParams(half_life_seconds=10.0, beta=0.0, gamma=0.0, delta=0.0)
        # After exactly one half-life, weight should be half (within fp precision).
        out = apply(weight=1.0, dt_seconds=10.0, params=params)
        assert math.isclose(out, 0.5, rel_tol=1e-9)

    def test_two_half_lives_quarters_weight(self) -> None:
        params = DecayParams(half_life_seconds=10.0)
        out = apply(weight=1.0, dt_seconds=20.0, params=params)
        assert math.isclose(out, 0.25, rel_tol=1e-9)

    def test_reinforcement_raises_weight(self) -> None:
        params = DecayParams(half_life_seconds=1e9, beta=0.30)
        out = apply(weight=0.5, dt_seconds=0.0, reinforcement=1, params=params)
        assert math.isclose(out, 0.80, rel_tol=1e-9)

    def test_reinforcement_clamped_to_one(self) -> None:
        params = DecayParams(half_life_seconds=1e9, beta=0.30)
        out = apply(weight=0.9, dt_seconds=0.0, reinforcement=10, params=params)
        assert out == 1.0

    def test_contradiction_lowers_weight(self) -> None:
        params = DecayParams(half_life_seconds=1e9, delta=0.30)
        out = apply(weight=0.8, dt_seconds=0.0, contradiction=1, params=params)
        assert math.isclose(out, 0.50, rel_tol=1e-9)

    def test_contradiction_clamped_to_zero(self) -> None:
        params = DecayParams(half_life_seconds=1e9, delta=0.99)
        out = apply(weight=0.10, dt_seconds=0.0, contradiction=5, params=params)
        assert out == 0.0

    def test_corroboration_increments(self) -> None:
        params = DecayParams(half_life_seconds=1e9, gamma=0.10)
        out = apply(weight=0.5, dt_seconds=0.0, corroboration=2, params=params)
        assert math.isclose(out, 0.70, rel_tol=1e-9)

    def test_combined_signals(self) -> None:
        params = DecayParams(half_life_seconds=1e9, beta=0.10, gamma=0.05, delta=0.20)
        out = apply(
            weight=0.5,
            dt_seconds=0.0,
            reinforcement=2,
            corroboration=2,
            contradiction=1,
            params=params,
        )
        # 0.5 + 0.10*2 + 0.05*2 - 0.20*1 = 0.5 + 0.20 + 0.10 - 0.20 = 0.60
        assert math.isclose(out, 0.60, rel_tol=1e-9)


# --- apply: input validation ------------------------------------------------


class TestApplyInputs:
    @pytest.mark.parametrize("w", [-0.01, 1.01, float("nan")])
    def test_rejects_out_of_range_weight(self, w: float) -> None:
        with pytest.raises(ValueError, match="weight"):
            apply(weight=w, dt_seconds=0.0, params=DecayParams())

    @pytest.mark.parametrize("dt", [-1.0, float("nan"), float("inf")])
    def test_rejects_bad_dt(self, dt: float) -> None:
        with pytest.raises(ValueError, match="dt_seconds"):
            apply(weight=0.5, dt_seconds=dt, params=DecayParams())

    @pytest.mark.parametrize("name", ["reinforcement", "corroboration", "contradiction"])
    def test_rejects_negative_signal(self, name: str) -> None:
        with pytest.raises(ValueError, match=name):
            apply(weight=0.5, dt_seconds=0.0, params=DecayParams(), **{name: -1})


# --- is_cold ----------------------------------------------------------------


class TestIsCold:
    def test_above_threshold_not_cold(self) -> None:
        params = DecayParams(threshold=0.10)
        assert not is_cold(0.20, params)

    def test_at_threshold_not_cold(self) -> None:
        # is_cold uses strict `<`. An item exactly at the threshold survives.
        params = DecayParams(threshold=0.10)
        assert not is_cold(0.10, params)

    def test_below_threshold_cold(self) -> None:
        params = DecayParams(threshold=0.10)
        assert is_cold(0.09, params)

    def test_threshold_zero_means_only_zero_is_cold(self) -> None:
        # With threshold=0, no positive weight is < 0, so nothing is ever cold.
        params = DecayParams(threshold=0.0)
        assert not is_cold(0.0, params)
        assert not is_cold(0.0001, params)


# --- property tests (DoD invariants) ----------------------------------------


_finite_weight = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_finite_dt = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)
_signal_count = st.integers(min_value=0, max_value=1000)
_finite_gain = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_finite_threshold = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_finite_half_life = st.floats(min_value=1.0, max_value=1e10, allow_nan=False, allow_infinity=False)


@st.composite
def _params(draw: st.DrawFn) -> DecayParams:
    return DecayParams(
        half_life_seconds=draw(_finite_half_life),
        beta=draw(_finite_gain),
        gamma=draw(_finite_gain),
        delta=draw(_finite_gain),
        threshold=draw(_finite_threshold),
    )


class TestProperties:
    @given(
        weight=_finite_weight,
        dt=_finite_dt,
        r=_signal_count,
        c=_signal_count,
        x=_signal_count,
        params=_params(),
    )
    def test_output_in_unit_interval(
        self,
        weight: float,
        dt: float,
        r: int,
        c: int,
        x: int,
        params: DecayParams,
    ) -> None:
        out = apply(
            weight=weight,
            dt_seconds=dt,
            reinforcement=r,
            corroboration=c,
            contradiction=x,
            params=params,
        )
        assert 0.0 <= out <= 1.0

    @given(weight=_finite_weight, dt=_finite_dt, params=_params())
    def test_decay_without_signals_is_non_increasing(
        self,
        weight: float,
        dt: float,
        params: DecayParams,
    ) -> None:
        out = apply(weight=weight, dt_seconds=dt, params=params)
        assert out <= weight + 1e-12  # tolerance for fp identity at dt=0

    @given(
        weight=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        r=st.integers(min_value=1, max_value=100),
        beta=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
        threshold=_finite_threshold,
    )
    def test_reinforcement_strictly_raises_when_room(
        self,
        weight: float,
        r: int,
        beta: float,
        threshold: float,
    ) -> None:
        # alpha=1 (no decay), no contradiction, no corroboration, weight has
        # headroom below 1: reinforcement strictly raises weight.
        params = DecayParams(half_life_seconds=1e15, beta=beta, threshold=threshold)
        out = apply(weight=weight, dt_seconds=0.0, reinforcement=r, params=params)
        assert out > weight or out == 1.0

    @given(
        weight=_finite_weight,
        x=st.integers(min_value=1, max_value=100),
        delta=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_contradiction_never_raises(
        self,
        weight: float,
        x: int,
        delta: float,
    ) -> None:
        params = DecayParams(half_life_seconds=1e15, delta=delta)
        out = apply(weight=weight, dt_seconds=0.0, contradiction=x, params=params)
        assert out <= weight + 1e-12

    @given(
        weight=_finite_weight,
        dt1=_finite_dt,
        dt2=_finite_dt,
        params=_params(),
    )
    def test_pure_decay_is_composable(
        self,
        weight: float,
        dt1: float,
        dt2: float,
        params: DecayParams,
    ) -> None:
        # In the absence of signals, decaying for dt1+dt2 in one step should
        # equal decaying for dt1 then dt2 (mathematically; up to fp noise).
        # Because intermediate clamping doesn't kick in for valid weights.
        params_no_signals = DecayParams(
            half_life_seconds=params.half_life_seconds,
            beta=0.0,
            gamma=0.0,
            delta=0.0,
            threshold=params.threshold,
        )
        one_step = apply(weight=weight, dt_seconds=dt1 + dt2, params=params_no_signals)
        intermediate = apply(weight=weight, dt_seconds=dt1, params=params_no_signals)
        two_step = apply(weight=intermediate, dt_seconds=dt2, params=params_no_signals)
        assert math.isclose(one_step, two_step, rel_tol=1e-9, abs_tol=1e-12)
