"""Engram consolidation engine.

Recent unconsolidated events get clustered, abstracted into general
patterns, and linked into the memory hierarchy through provenance. This
is the README's headline feature: events become summaries, summaries
become abstractions, and the resulting hierarchy is what the Stage 6
coarse-to-fine retriever reads.

Module layout:
  * `_clustering`     — HDBSCAN / agglomerative clustering of unit-norm vectors
  * `_abstraction`    — versioned LLM prompt + JSON schema for generalization
  * `_contradiction`  — pairwise check against existing abstractions
  * `_engine`         — orchestration that ties the above into one pipeline

The engine is opt-in (call `Memory.consolidate(...)`) - the library does
not run consolidation in the background. Stage 9 introduces a worker
that schedules consolidation alongside the decay tick.
"""

from engram.consolidation._abstraction import (
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    AbstractionParseError,
    AbstractionRequest,
    AbstractionResult,
    extract_abstraction,
    parse_response,
    render_prompt,
)
from engram.consolidation._clustering import (
    ClusterAssignment,
    ClusterParams,
    cluster,
    cohesion,
)

__all__ = [
    "PROMPT_VERSION",
    "PROMPT_VERSIONS",
    "AbstractionParseError",
    "AbstractionRequest",
    "AbstractionResult",
    "ClusterAssignment",
    "ClusterParams",
    "cluster",
    "cohesion",
    "extract_abstraction",
    "parse_response",
    "render_prompt",
]
