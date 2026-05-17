"""Tests for `RetrieveParams.replace(...)` and `field_names()`.

The `replace` helper enforces a pass-through contract for callers
that fan out a single retrieve into per-leaf retrieves
(`_multi_query_retrieve`, `_decomposed_retrieve`). Without it, the
caller must enumerate every field manually -- which silently drops
new fields and quietly breaks leaf retrieves.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engram.retrieve import RetrieveParams


class TestReplaceSemantics:
    def test_returns_a_new_instance(self) -> None:
        a = RetrieveParams(k=5)
        b = a.replace(k=10)
        assert a is not b
        assert a.k == 5
        assert b.k == 10

    def test_unspecified_fields_are_preserved(self) -> None:
        """The whole point: every knob a caller already tuned must
        carry over to the leaf retrieves unless explicitly overridden."""
        original = RetrieveParams(
            k=7,
            prefer="specific",
            bm25_weight=0.7,
            mmr_lambda=0.4,
            recency_lambda=0.1,
            lexical_filter=r"\b2023\b",
            recency_decay_days=30.0,
            recent_window_k=5,
            bm25_k1=1.8,
            bm25_b=0.6,
            mmr_pool_size=20,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        replaced = original.replace(reinforce_on_use=False)
        # Every non-overridden field equals the original.
        for name in original.field_names():
            if name == "reinforce_on_use":
                continue
            assert getattr(replaced, name) == getattr(original, name), name
        assert original.reinforce_on_use is True
        assert replaced.reinforce_on_use is False

    def test_post_init_revalidates(self) -> None:
        """`replace` re-runs `__post_init__`, so a bad combination
        on the replaced copy still raises."""
        p = RetrieveParams(k=5)
        with pytest.raises(ValueError, match="k must be"):
            p.replace(k=0)

    def test_unknown_field_raises_type_error(self) -> None:
        p = RetrieveParams()
        with pytest.raises(TypeError):
            p.replace(not_a_real_field=42)

    def test_field_names_lists_every_field(self) -> None:
        """A sanity check on the field list so future audits can
        bisect a missing field at the test level."""
        p = RetrieveParams()
        names = set(p.field_names())
        # Spot-check a few of the audit-cited fields the old leaf
        # constructors were dropping.
        for needed in (
            "bm25_weight",
            "mmr_lambda",
            "recency_lambda",
            "lexical_filter",
            "bm25_k1",
            "bm25_b",
            "recency_decay_days",
            "mmr_pool_size",
            "recent_window_k",
        ):
            assert needed in names, needed


class TestPassThroughForLeafRetrieves:
    """Scenario tests: simulate the leaf-construction pattern used by
    multi-query / decompose, verify nothing is lost.
    """

    def test_replace_overrides_only_named_fields(self) -> None:
        parent = RetrieveParams(
            k=10,
            prefer="auto",
            bm25_weight=1.2,
            mmr_lambda=0.6,
            lexical_filter="city",
        )
        # The leaf retrieves should disable reinforce_on_use, hyde,
        # multi_query_n, decompose -- but inherit every other field.
        leaf = parent.replace(
            reinforce_on_use=False,
            hyde=False,
            multi_query_n=1,
            decompose=False,
        )
        assert leaf.bm25_weight == 1.2
        assert leaf.mmr_lambda == 0.6
        assert leaf.lexical_filter == "city"
        assert leaf.k == 10
        assert leaf.prefer == "auto"
        # And the explicit overrides took effect.
        assert leaf.reinforce_on_use is False
        assert leaf.hyde is False
        assert leaf.multi_query_n == 1
        assert leaf.decompose is False
