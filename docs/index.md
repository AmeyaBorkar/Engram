# Engram

**Hierarchical memory with consolidation and principled decay for LLM agents and assistants.**

Engram is a memory layer for LLM systems that does what existing memory libraries don't: it *consolidates*. Raw events get abstracted into general patterns, redundant or contradicted memories decay, and retrieval reads across a hierarchy from specific episodes to compressed knowledge — the way human memory actually works.

It is designed as a single primitive that serves both **agents** (procedural memory: "in situations like this, this approach worked") and **assistants** (semantic memory: "the user has a golden retriever they care about"), with the same algorithm and the same API.

---

## Why hierarchical memory

Most LLM memory libraries are a vector store with a flat list of facts. That works until:

- The conversation gets long enough that "facts" stop being literal events and start being patterns.
- The same fact gets restated three different ways and a flat store can't decide which to surface.
- A new fact contradicts an old one, and the store keeps both.

Engram:

1. **Consolidates.** Many specific events become one general principle. "User mentioned dog in conv 3, vet in conv 7, kibble in conv 12" becomes "user has a dog they actively care about."
2. **Forgets selectively.** Routine, redundant, or contradicted memories decay; surprising, frequently-used, or recently-relevant ones strengthen.
3. **Reads across abstractions.** Retrieval sometimes wants the general pattern, sometimes the specific episode, sometimes both. Flat stores can't do this cleanly.
4. **Reconciles contradictions.** When two memories disagree, you choose a resolution policy (`PREFER_RECENT`, `PREFER_TRUSTED`, `PREFER_FREQUENT`, `MERGE`, `KEEP_BOTH`, `MANUAL`) and Engram invalidates the loser. Time-travel queries (`as_of=...`) still surface historical state.

---

## Quick example

```python
from engram import Memory, SqliteStorage, Outcome
from engram.providers import OpenAIEmbedder, OpenAIChat

memory = Memory(
    storage=SqliteStorage("engram.db"),
    embedder=OpenAIEmbedder(),
    chat=OpenAIChat(),
)

# Observe events.
memory.observe("user prefers tabs over spaces")
memory.observe("user uses Python and Rust")

# Retrieve hierarchically.
results = memory.retrieve("what do I know about the user's coding style?")

# Record procedural learnings.
memory.record_procedure(
    "flaky integration test",
    "rerun with --no-cov and bisect",
    outcome=Outcome.SUCCESS,
)
```

See [Quickstart](getting-started/quickstart.md) for the full tour.

---

## Stage map

| Release | Capability |
|---|---|
| `v0.1.0` | Hierarchical memory on SQLite (consolidate, decay, coarse-to-fine retrieve) |
| `v0.2.0` | Procedural memory (record_procedure / retrieve_procedures / update_outcome) |
| `v0.2.1` | Framework integrations (LangGraph, LlamaIndex, raw OpenAI/Anthropic agent helper) |
| `v0.3.0` | Contradiction + temporal reasoning (reconcile, validity windows, as_of) |
| `v0.3.1` | MERGE resolution policy, prompt-injection corpus for the judge, perf budgets |
| `v0.4.0` | Postgres backend, async API, multi-tenant + RLS, full OpenTelemetry |
| `v1.0.0` | Frozen public API, paper, full docs site |

See the [Roadmap](https://github.com/AmeyaBorkar/Engram/blob/main/ROADMAP.md) for the full DoD.

---

## Next steps

- **Install and run a quickstart** → [Getting started](getting-started/install.md)
- **Understand the model** → [Concepts](getting-started/concepts.md)
- **API surface** → [Memory reference](api/memory.md)
- **Deploy in production** → [Observability](operations/observability.md) / [Multi-tenant](operations/multi-tenant.md)
