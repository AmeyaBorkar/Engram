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
from engram.retrieve._reranker import FakeReranker, Reranker

__all__ = [
    "FakeReranker",
    "HierarchicalRetriever",
    "Reranker",
    "RetrieveParams",
    "RetrievePrefer",
]
