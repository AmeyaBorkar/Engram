"""Reranker protocol + deterministic fake.

A reranker takes the merged candidate list from the coarse-to-fine
pipeline and reorders it. Cross-encoders (BGE, Cohere Rerank, ...) live
behind this protocol so callers can swap them without touching the
retrieve engine.

The fake reranker (used in tests + the bench harness) scores by literal
token overlap between the query and the candidate text, with a small
length-normalization. It's deterministic, dependency-free, and lets us
assert that the rerank seam is wired up without depending on a real
cross-encoder model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engram.schemas import RetrievalResult


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """One candidate passed to a reranker.

    The candidate carries the full `RetrievalResult` (so the reranker
    can read its content) plus the `score` from the prior stage. The
    reranker returns a new score; the engine sorts on it.
    """

    result: RetrievalResult
    prior_score: float


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranker.

    Implementations score candidates against the query text. The
    returned scores must be finite floats; ordering is descending.
    """

    name: str

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> list[float]:
        """Return one score per candidate (same length, same order)."""


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


class FakeReranker:
    """Token-overlap reranker -- deterministic, dependency-free.

    Score = (overlap_count * 1.0 + prior_score * blend) / (sqrt(len(cand_tokens)) + 1)

    The blend term keeps the prior similarity informative when overlap
    is zero (e.g. paraphrase queries). Token tokenization is unicode-
    aware and case-insensitive.
    """

    name: str = "fake-reranker"

    def __init__(self, *, blend: float = 0.5) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend!r}")
        self._blend = blend

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> list[float]:
        q_tokens = set(_tokenize(query))
        scores: list[float] = []
        for cand in candidates:
            c_tokens = _tokenize(cand.result.content)
            overlap = sum(1 for t in c_tokens if t in q_tokens)
            length_norm = (len(c_tokens) ** 0.5) + 1.0
            scores.append((overlap + cand.prior_score * self._blend) / length_norm)
        return scores
