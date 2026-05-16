"""Engram: hierarchical memory with consolidation and principled decay for LLM systems.

Public surface stability classification
---------------------------------------
Every symbol re-exported below is one of three tiers; the inline group
comments mark them and the audit table tracks compat guarantees.

Tier S (stable): semver-compatible. A breaking change requires a major
version bump. These are the contract users program against:
  - Memory, RetrieveParams, RetrievePrefer
  - Storage, SqliteStorage, StorageStats, stats
  - Schemas: Event, MemoryItem, Procedure, Cluster, Conflict,
    ConflictStatus, ProvenanceLink, RetrievalResult, ProcedureMatch,
    Embedding, Source, DecayState
  - Enums: Level, ItemKind, Outcome, Resolution, Verdict
  - DecayParams
  - new_id, is_preference
  - __version__

Tier E (experimental, pre-1.0): may change in any minor release. Audit
notes track the open design questions. As of 0.2.x:
  - HierarchicalRetriever (the public retriever interface is stable; the
    composition under the hood -- HyDE / multi-query / iterative -- is
    not)
  - Reranker, FakeReranker (the protocol is stable, but the
    sentence-transformers backed reranker may move providers)

Tier I (internal): underscored module paths only. Not re-exported here.
Examples: `engram._otel`, `engram._vec_math`, `engram._preference`,
`engram.consolidation._engine`, `engram.reconcile._engine`. These can
move at any time. The audit forbids new third-party code from importing
them; if you need something here, file an issue and we'll promote it.

Adding a new public symbol
--------------------------
1. Decide the tier and add it to the matching group below.
2. Add a brief docstring entry explaining what stability tier it lives
   in (the same tier rule as the symbol above it).
3. The CHANGELOG entry MUST call out the new public surface.
"""

# Imports are alphabetized; the comments below mark which stability tier
# each symbol falls into (see module docstring). Tier annotations are
# author-facing -- they do not affect runtime behavior.
from engram._preference import is_preference  # Tier S helper
from engram._vec_math import normalize  # Tier S helper
from engram.decay import DecayParams  # Tier S decay surface
from engram.ids import new_id  # Tier S -- UUIDv7 with monotonic counter
from engram.memory import Memory  # Tier S orchestrator
from engram.retrieve import (  # Tier E retrieve composition
    FakeReranker,
    HierarchicalRetriever,
    Reranker,
    RetrieveParams,
    RetrievePrefer,
)
from engram.schemas import (  # Tier S persisted schemas
    Cluster,
    Conflict,
    ConflictStatus,
    DecayState,
    Embedding,
    Event,
    ItemKind,
    Level,
    MemoryItem,
    Outcome,
    Procedure,
    ProcedureMatch,
    ProvenanceLink,
    Resolution,
    RetrievalResult,
    Source,
    Verdict,
)
from engram.storage import SqliteStorage, Storage, StorageStats, stats  # Tier S

# Alphabetically sorted. Stability tier is documented inline above each
# import and in the module docstring; do not introduce a non-alpha order
# here -- ruff RUF022 catches divergence and the grouping serves
# readability via comments rather than via list order.
__all__ = [
    "Cluster",  # Tier S
    "Conflict",  # Tier S
    "ConflictStatus",  # Tier S
    "DecayParams",  # Tier S
    "DecayState",  # Tier S
    "Embedding",  # Tier S
    "Event",  # Tier S
    "FakeReranker",  # Tier E
    "HierarchicalRetriever",  # Tier E
    "ItemKind",  # Tier S
    "Level",  # Tier S
    "Memory",  # Tier S
    "MemoryItem",  # Tier S
    "Outcome",  # Tier S
    "Procedure",  # Tier S
    "ProcedureMatch",  # Tier S
    "ProvenanceLink",  # Tier S
    "Reranker",  # Tier E
    "Resolution",  # Tier S
    "RetrievalResult",  # Tier S
    "RetrieveParams",  # Tier E
    "RetrievePrefer",  # Tier E
    "Source",  # Tier S
    "SqliteStorage",  # Tier S
    "Storage",  # Tier S
    "StorageStats",  # Tier S
    "Verdict",  # Tier S
    "is_preference",  # Tier S helper
    "new_id",  # Tier S helper
    "normalize",  # Tier S helper
    "stats",  # Tier S helper
]
__version__ = "0.2.1"
