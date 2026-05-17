"""Engram: hierarchical memory with consolidation and principled decay for LLM systems."""

from engram._preference import is_preference
from engram._vec_math import normalize
from engram.decay import DecayParams
from engram.ids import new_id
from engram.memory import Memory
from engram.retrieve import (
    FakeReranker,
    HierarchicalRetriever,
    Reranker,
    RetrieveParams,
    RetrievePrefer,
)
from engram.schemas import (
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
from engram.storage import SqliteStorage, Storage, StorageStats, stats

__all__ = [
    "Cluster",
    "Conflict",
    "ConflictStatus",
    "DecayParams",
    "DecayState",
    "Embedding",
    "Event",
    "FakeReranker",
    "HierarchicalRetriever",
    "ItemKind",
    "Level",
    "Memory",
    "MemoryItem",
    "Outcome",
    "Procedure",
    "ProcedureMatch",
    "ProvenanceLink",
    "Reranker",
    "Resolution",
    "RetrievalResult",
    "RetrieveParams",
    "RetrievePrefer",
    "Source",
    "SqliteStorage",
    "Storage",
    "StorageStats",
    "Verdict",
    "is_preference",
    "new_id",
    "normalize",
    "stats",
]
try:
    # Source of truth: the installed distribution's metadata (read once
    # at import).  Falls back to a placeholder if the package isn't
    # installed (e.g., running directly out of a checkout without
    # `pip install -e .`) so import never breaks.
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__: str = _pkg_version("engrampy")
except PackageNotFoundError:  # pragma: no cover - dev tree without install
    __version__ = "0.0.0+unknown"
