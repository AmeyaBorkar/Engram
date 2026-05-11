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
