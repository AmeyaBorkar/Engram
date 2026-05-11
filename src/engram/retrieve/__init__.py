"""Coarse-to-fine retrieval (Stage 6).

`Memory.retrieve` reads abstractions first and drills into supporting
events when confidence is low or the caller asks for specifics. Each
retrieval result honestly reports its hierarchy `level`.

Public surface (re-exported by `engram`):

  * `RetrieveParams` -- shape of the call (k, prefer, drill_k, ...).
  * `RetrievePrefer` -- `Literal["auto", "specific", "general"]`.
  * `Reranker` -- protocol for an optional cross-encoder pass.
  * `FakeReranker` -- deterministic test double that ranks by surface
    string overlap.
  * `HierarchicalRetriever` -- the engine. `Memory.retrieve` is a thin
    wrapper.

The engine is decoupled from `Memory` itself so non-default
configurations (custom reranker, drill_k, ...) can be passed through
without ballooning the `Memory` constructor.
"""

from __future__ import annotations

from engram.retrieve._engine import HierarchicalRetriever
from engram.retrieve._params import RetrieveParams, RetrievePrefer
from engram.retrieve._reranker import FakeReranker, RerankCandidate, Reranker

__all__ = [
    "FakeReranker",
    "HierarchicalRetriever",
    "RerankCandidate",
    "Reranker",
    "RetrieveParams",
    "RetrievePrefer",
]


def __getattr__(name: str) -> object:
    """Lazy access for `BGEReranker` so the heavy sentence-transformers
    import only happens when the user actually constructs one."""
    if name == "BGEReranker":
        from engram.retrieve._bge_reranker import BGEReranker as _BGEReranker

        return _BGEReranker
    raise AttributeError(f"module 'engram.retrieve' has no attribute {name!r}")
