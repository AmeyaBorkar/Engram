"""Smoke + contract tests for the Stage 8 adversarial benchmark.

The suite runs end-to-end against `FakeProvider`. The assertion
discipline matches the Stage 7 procedural-transfer suite -- the
synthetic numbers are not SOTA claims, just contract verification
that:

  * Engram outperforms the no-reconcile baseline by a wide margin on
    the contradiction split (the Stage 8 DoD).
  * Every seeded conflict surfaces as OPEN via `list_conflicts`.
  * Temporal accuracy is high.
"""

from __future__ import annotations

from benchmarks.suites.contradiction_temporal import (
    CONTRADICTION_PAIRS,
    SUITE,
    TEMPORAL_TRIPLES,
)

from engram.bench import FakeProvider


def test_suite_runs_and_lifts() -> None:
    provider = FakeProvider(dim=64)
    SUITE.setup(provider)
    try:
        result = SUITE.run()
    finally:
        SUITE.teardown()

    assert result.name == "contradiction-temporal"
    metrics = result.aggregate_metrics

    # Every seeded conflict was observable as OPEN before reconcile.
    assert metrics["n_observed_open_conflicts"] == float(len(CONTRADICTION_PAIRS))

    # Every conflict got resolved by the suite's reconcile pass.
    assert metrics["n_resolved_conflicts"] == float(len(CONTRADICTION_PAIRS))

    # DoD: engram outperforms baseline by >= 0.5 (in practice we expect
    # ~1.0 vs ~0.0 on this synthetic split).
    assert metrics["lift"] >= 0.5

    # Engram score is close to perfect on the synthetic split.
    assert metrics["engram_score"] >= 0.9

    # Temporal accuracy is also high. With FakeEmbedder all three
    # versions in a triple share the same embedding -- the agent has
    # to use temporal info to disambiguate. Bar at 0.9.
    assert metrics["temporal_accuracy"] >= 0.9

    # Per-question shape covers both splits.
    splits = {row["split"] for row in result.per_question}
    assert splits == {"contradiction", "temporal"}
    assert sum(1 for r in result.per_question if r["split"] == "contradiction") == len(
        CONTRADICTION_PAIRS
    )
    assert sum(1 for r in result.per_question if r["split"] == "temporal") == len(TEMPORAL_TRIPLES)


def test_dataset_checksum_is_stable() -> None:
    """Changing the dataset changes the checksum; the suite singletons
    must have the same checksum across instantiations of identical
    parameters."""
    from benchmarks.suites.contradiction_temporal import ContradictionTemporalSuite

    a = ContradictionTemporalSuite()
    b = ContradictionTemporalSuite()
    assert a.dataset_checksum == b.dataset_checksum
