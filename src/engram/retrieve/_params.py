"""Parameters for `Memory.retrieve` / `HierarchicalRetriever.retrieve`.

`RetrieveParams` is a frozen dataclass so two calls with the same
parameters are reproducible by construction. Validation lives in
`__post_init__` -- the model is small enough that pydantic would be
overkill, and we save the per-call validation cost on a hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# `auto`     -- generalize when confident, drill when not (the default).
# `specific` -- always surface events; abstractions are skipped.
# `general`  -- always surface abstractions/summaries; never drill.
RetrievePrefer = Literal["auto", "specific", "general"]


@dataclass(frozen=True, slots=True)
class RetrieveParams:
    """Shape of one retrieval call.

    `k`                    -- final result count.
    `prefer`               -- specific / general / auto (see RetrievePrefer).
    `confidence_threshold` -- in `auto` mode, hits at or above this score
                              are surfaced as the abstraction; below it,
                              the engine drills into supporting events.
    `drill_k`              -- per low-confidence abstraction, how many
                              supporting events to consider when drilling.
    `candidate_multiplier` -- number of memory items to pull from the first
                              stage = `k * candidate_multiplier`. The merge
                              + rerank step picks the final `k`.
    `include_cold`         -- include items pruned by the decay engine.
    `reinforce_on_use`     -- after retrieval, fire reinforcement signals
                              against every surfaced item (closes the loop
                              between retrieval and decay).
    `as_of`                -- if set, return items whose validity window
                              covers this timestamp (Stage 8 temporal
                              retrieve). `None` means "current state":
                              invalidated items are excluded but no
                              backward time travel happens.
    `hyde`                 -- if True (and the Memory has a chat provider),
                              transform the query into a hypothetical
                              answer via the chat provider before
                              embedding. Trades one chat call for a
                              precision boost on questions whose phrasing
                              differs from how memories were stored.
    `multi_query_n`        -- if >= 2 AND a chat provider is configured,
                              expand the query into N total queries
                              (original + N-1 paraphrases), retrieve
                              each independently, and fuse via
                              Reciprocal Rank Fusion. Recall-complete
                              at the cost of N retrievals. 1 = off.
    `rrf_k`                -- smoothing constant for RRF when
                              multi_query_n >= 2. Default 60 is the
                              standard value from the RRF paper.
    """

    k: int = 10
    prefer: RetrievePrefer = "auto"
    confidence_threshold: float = 0.7
    drill_k: int = 3
    candidate_multiplier: int = 3
    include_cold: bool = False
    reinforce_on_use: bool = True
    as_of: datetime | None = None
    hyde: bool = False
    multi_query_n: int = 1
    rrf_k: int = 60
    decompose: bool = False
    surface_conflicts: bool = False
    temporal: bool = False
    # Hybrid retrieval: when > 0, the engine runs BM25 against the
    # event corpus in parallel with the dense path and fuses the two
    # rankings via Reciprocal Rank Fusion. 0.0 (default) disables BM25
    # entirely; the recommended setting is 1.0 (equal-weight fusion).
    # Values other than 0/1 currently scale the BM25 ranking's RRF
    # contribution proportionally -- see `_engine._fuse_dense_bm25`.
    bm25_weight: float = 0.0
    # MMR diversity rerank: when > 0, re-orders the rerank pool to
    # balance relevance with diversity via Maximal Marginal Relevance.
    # `0` (default) is off; `0.7` is a sensible value -- prioritize
    # relevance but suppress near-duplicates.
    mmr_lambda: float = 0.0
    # Time-decay boost on the rerank score: items closer to `as_of`
    # (or `now` if as_of is None) get a small multiplicative bump.
    # Off by default; values around 0.05 - 0.15 work well on
    # LongMemEval's knowledge-update and temporal-reasoning categories.
    recency_lambda: float = 0.0
    # Lexical filter: regex pattern that candidate `content` must match
    # to survive into the rerank pool. None (default) is "no filter".
    # Use for surgical recall on year-anchored queries
    # (`lexical_filter=r"\b2023\b"`) where the embedder smooths the
    # year away. Case-insensitive. Items that fail the regex are
    # dropped BEFORE the rerank step, so the cross-encoder never sees
    # them.
    lexical_filter: str | None = None
    # BM25 hyperparameters. Lucene defaults. `k1` in [1.2, 2.0] and
    # `b` in [0.5, 0.9] are the sane operating ranges; touch only when
    # you have a specific tuning hypothesis.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # Time-decay half-life for `recency_lambda`. 90 days is the default
    # because it matches LongMemEval's typical haystack span; shorter
    # values give the boost a sharper falloff.
    recency_decay_days: float = 90.0
    # Size of the candidate pool MMR re-orders over. `0` means "use
    # `k * candidate_multiplier`" (the default). Raise above that
    # multiplier to give MMR a wider pool to diversify across without
    # also widening the cross-encoder rerank.
    mmr_pool_size: int = 0
    # Recent-window hybrid: include the top-N most-recent events
    # (by created_at desc) as a third RRF stream alongside dense + BM25.
    # `0` (default) is off. Helpful for "lately I..." / "recently..."
    # queries where the answer is in the freshest events regardless of
    # token overlap.
    recent_window_k: int = 0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.prefer not in ("auto", "specific", "general"):
            raise ValueError(
                f"prefer must be 'auto', 'specific', or 'general'; got {self.prefer!r}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold!r}"
            )
        if self.drill_k < 0:
            raise ValueError(f"drill_k must be >= 0, got {self.drill_k}")
        if self.candidate_multiplier < 1:
            raise ValueError(f"candidate_multiplier must be >= 1, got {self.candidate_multiplier}")
        if self.multi_query_n < 1:
            raise ValueError(f"multi_query_n must be >= 1, got {self.multi_query_n}")
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")
        if self.bm25_weight < 0:
            raise ValueError(f"bm25_weight must be >= 0, got {self.bm25_weight}")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda}")
        if self.recency_lambda < 0:
            raise ValueError(f"recency_lambda must be >= 0, got {self.recency_lambda}")
        if self.lexical_filter is not None:
            # Validate the regex at construction time so the engine
            # doesn't pay the cost (or surface the surprise) per call.
            import re as _re

            try:
                _re.compile(self.lexical_filter)
            except _re.error as exc:  # pragma: no cover - propagated to caller
                raise ValueError(f"lexical_filter is not a valid regex: {exc}") from exc
        if self.bm25_k1 < 0:
            raise ValueError(f"bm25_k1 must be >= 0, got {self.bm25_k1}")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError(f"bm25_b must be in [0, 1], got {self.bm25_b}")
        if self.recency_decay_days <= 0:
            raise ValueError(f"recency_decay_days must be > 0, got {self.recency_decay_days}")
        if self.mmr_pool_size < 0:
            raise ValueError(f"mmr_pool_size must be >= 0, got {self.mmr_pool_size}")
        if self.recent_window_k < 0:
            raise ValueError(f"recent_window_k must be >= 0, got {self.recent_window_k}")
