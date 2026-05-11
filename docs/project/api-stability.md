# API stability

Pre-`v1.0.0`, Engram's public API may evolve between minor versions. The contract below documents what's stable and what's experimental.

## Stable public surface (v0.3.x)

These names are committed: a removal or breaking change requires a deprecation cycle.

| Name | Module |
|---|---|
| `Memory` | `engram` |
| `Memory.observe` / `aobserve` | — |
| `Memory.retrieve` / `aretrieve` | — |
| `Memory.consolidate` / `aconsolidate` | — |
| `Memory.record_procedure` / `arecord_procedure` | — |
| `Memory.retrieve_procedures` / `aretrieve_procedures` | — |
| `Memory.reconcile` / `areconcile` | — |
| `Memory.list_conflicts` / `alist_conflicts` | — |
| `Memory.update_outcome` / `aupdate_outcome` | — |
| `Memory.tick` / `tick_async` | — |
| `Storage` (protocol) | `engram.storage` |
| `SqliteStorage` | `engram.storage` |
| `Event`, `MemoryItem`, `Procedure`, `Embedding`, `Cluster`, `ProvenanceLink`, `DecayState` | `engram.schemas` |
| `Conflict`, `ConflictStatus`, `Resolution`, `Source`, `Verdict` | `engram.schemas` |
| `Level`, `ItemKind`, `Outcome` | `engram.schemas` |
| `RetrievalResult`, `ProcedureMatch` | `engram.schemas` |
| `EngramAgent`, `format_context` | `engram.integrations` |

## Experimental / may change

| Name | Module | Why |
|---|---|---|
| `EngramRetrieveNode`, `EngramObserveNode` | `engram.integrations.langgraph` | LangGraph's own API still evolving; expect minor adjustments. |
| `EngramLlamaIndexMemory` | `engram.integrations.llamaindex` | LlamaIndex `BaseMemory` shape may shift; we adapt. |
| `engram._otel` | private | Span/counter names may rename pre-v1.0 if OTel semantic conventions for LLM systems change. |
| `engram.reconcile.Reconciler` | `engram.reconcile` | Direct construction is supported; new `Resolution` values may be added (additive only). |

## Internal (use at your own risk)

Modules prefixed with `_` are internal: `engram._security`, `engram._otel`, `engram.providers._fake`, every `*._engine` submodule.

## Deprecation policy

See [Deprecation policy](deprecation-policy.md).
