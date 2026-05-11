"""Tests for the BGE cross-encoder reranker.

Two test classes:

  * `TestBGERerankerAvailability` -- unit tests against the protocol
    surface that don't load the model (cheap).
  * `TestBGERerankerWithModel` -- `@pytest.mark.slow` integration tests
    that actually load `BAAI/bge-reranker-v2-m3` and run a forward
    pass. Skipped in the default test lane; opt in with `pytest -m
    slow`.
"""

from __future__ import annotations

import pytest

from engram import new_id
from engram.retrieve import RerankCandidate, Reranker
from engram.retrieve._bge_reranker import DEFAULT_MODEL, BGEReranker
from engram.schemas import Level, RetrievalResult


def _candidate(content: str, score: float = 0.5) -> RerankCandidate:
    return RerankCandidate(
        result=RetrievalResult(
            item_id=new_id(),
            level=Level.SUMMARY,
            content=content,
            confidence=score,
            score=score,
            supported_by=(),
        ),
        prior_score=score,
    )


class TestBGERerankerLazyImport:
    """The BGE reranker class is exposed lazily via
    `engram.retrieve.__getattr__` so the heavy sentence-transformers
    import only triggers on use."""

    def test_lazy_attribute(self) -> None:
        from engram import retrieve

        cls = retrieve.BGEReranker  # triggers the lazy import
        assert cls is BGEReranker

    def test_unknown_attribute_raises(self) -> None:
        import engram.retrieve as retrieve

        with pytest.raises(AttributeError):
            retrieve.__getattr__("NotAThing")


@pytest.mark.slow
class TestBGERerankerWithModel:
    """Loads the real BGE model. Slow; only runs in the slow lane."""

    @pytest.fixture(scope="class")
    def reranker(self) -> BGEReranker:
        # Force CPU for CI reproducibility; users on GPUs override
        # via the device= arg.
        return BGEReranker(model=DEFAULT_MODEL, device="cpu")

    def test_implements_protocol(self, reranker: BGEReranker) -> None:
        assert isinstance(reranker, Reranker)

    def test_empty_candidates_returns_empty(self, reranker: BGEReranker) -> None:
        assert reranker.rerank("anything", []) == []

    def test_ranks_relevant_higher_than_irrelevant(
        self, reranker: BGEReranker
    ) -> None:
        query = "user has a dog named Rex"
        candidates = [
            _candidate("user has a dog named Rex"),
            _candidate("the speed of light is 299792458 m/s"),
        ]
        scores = reranker.rerank(query, candidates)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_name_includes_model(self, reranker: BGEReranker) -> None:
        assert reranker.name.startswith("bge-reranker:")
        assert DEFAULT_MODEL in reranker.name


class TestDefaultBatchSize:
    """Pure logic; no model required."""

    def test_cpu_default(self) -> None:
        assert BGEReranker._default_batch_size("cpu") == 16

    def test_cuda_default(self) -> None:
        assert BGEReranker._default_batch_size("cuda") == 64

    def test_none_default(self) -> None:
        assert BGEReranker._default_batch_size(None) == 16
