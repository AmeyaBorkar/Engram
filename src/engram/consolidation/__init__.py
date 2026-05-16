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

The engine is opt-in (call `Memory.consolidate(...)`) — the library does
not run consolidation in the background.  A background scheduler that
runs consolidation alongside the decay tick is a roadmap item, not
shipped.
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
# Re-export Verdict from its canonical home (engram.schemas) so it's
# also reachable via engram.consolidation for back-compat.  The
# duplicate import-path is intentional during the transition; tests
# and downstream callers can move to `from engram import Verdict`.
from engram.consolidation._contradiction import (
    CandidateRow,
    ContradictionParams,
    DetectedConflict,
    JudgeResponse,
    conflicts_to_metadata,
    detect_contradictions,
    judge,
    parse_judge_response,
    render_judge_prompt,
)
from engram.schemas import Verdict
from engram.consolidation._engine import (
    ConsolidationEngine,
    ConsolidationParams,
    ConsolidationResult,
    PromotionParams,
    PromotionResult,
)

__all__ = [
    "PROMPT_VERSION",
    "PROMPT_VERSIONS",
    "AbstractionParseError",
    "AbstractionRequest",
    "AbstractionResult",
    "CandidateRow",
    "ClusterAssignment",
    "ClusterParams",
    "ConsolidationEngine",
    "ConsolidationParams",
    "ConsolidationResult",
    "ContradictionParams",
    "DetectedConflict",
    "JudgeResponse",
    "PromotionParams",
    "PromotionResult",
    "Verdict",
    "cluster",
    "cohesion",
    "conflicts_to_metadata",
    "detect_contradictions",
    "extract_abstraction",
    "judge",
    "parse_judge_response",
    "parse_response",
    "render_judge_prompt",
    "render_prompt",
]
