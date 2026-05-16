"""Tests for `engram._vec_math` -- normalize, dot, cosine_similarity.

Pure math, no I/O. Property-style coverage with Hypothesis for the
shape invariants (norm == 1 after normalize, similarity in [-1, 1]).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from engram._vec_math import cosine_similarity, dot, normalize

_finite_float = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


class TestNormalize:
    def test_unit_vector_unchanged(self) -> None:
        assert normalize([1.0, 0.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0])

    def test_scales_to_unit(self) -> None:
        out = normalize([3.0, 4.0])
        assert math.isclose(math.sqrt(sum(x * x for x in out)), 1.0, rel_tol=1e-9)
        assert out == pytest.approx([0.6, 0.8])

    def test_zero_vector_raises_by_default(self) -> None:
        with pytest.raises(ValueError, match="zero-norm"):
            normalize([0.0, 0.0, 0.0])

    def test_zero_vector_soft_returns_copy(self) -> None:
        vec = [0.0, 0.0, 0.0]
        out = normalize(vec, raise_on_zero=False)
        assert out == vec
        assert out is not vec  # not aliased

    def test_expected_dim_match(self) -> None:
        out = normalize([1.0, 2.0, 2.0], expected_dim=3)
        assert math.isclose(math.sqrt(sum(x * x for x in out)), 1.0)

    def test_expected_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            normalize([1.0, 0.0], expected_dim=3)
        with pytest.raises(ValueError, match="does not match"):
            normalize([1.0, 0.0, 0.0, 0.0], expected_dim=3)

    def test_returns_list_not_input_type(self) -> None:
        # Accepts tuple, returns list.
        out = normalize((1.0, 0.0))
        assert isinstance(out, list)

    @given(st.lists(_finite_float, min_size=1, max_size=64))
    def test_property_unit_norm(self, vec: list[float]) -> None:
        # Non-zero vector -> norm == 1 after normalize. Loose tolerance
        # because at the float-epsilon end of the range (denormals,
        # ~1e-300 components) the round-trip is bounded by accumulated
        # rounding rather than relative precision.
        if math.sqrt(sum(x * x for x in vec)) == 0.0:
            with pytest.raises(ValueError, match="zero-norm"):
                normalize(vec)
            return
        out = normalize(vec)
        norm_squared = sum(x * x for x in out)
        # If we got here we expect a non-zero result. Loose tolerance
        # caps relative error; the absolute floor catches the denormal
        # case where rel_tol alone is too tight.
        assert math.isclose(math.sqrt(norm_squared), 1.0, rel_tol=1e-6, abs_tol=1e-6)


class TestDot:
    def test_basic(self) -> None:
        assert dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            dot([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_returns_zero(self) -> None:
        # dot is well-defined on zero vectors.
        assert dot([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestCosineSimilarity:
    def test_identical_vectors_one(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_opposite_vectors_minus_one(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_orthogonal_zero(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="zero-norm"):
            cosine_similarity([0.0, 0.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="zero-norm"):
            cosine_similarity([1.0, 2.0], [0.0, 0.0])

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="differ"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_expected_dim_match(self) -> None:
        assert math.isclose(
            cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], expected_dim=3),
            1.0,
        )

    def test_expected_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_dim"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0], expected_dim=3)


def test_public_reexport_in_engram() -> None:
    """`normalize` is reachable from the top-level `engram` package."""
    from engram import normalize as public_normalize

    assert public_normalize is normalize
