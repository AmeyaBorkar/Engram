# Engram

> Hierarchical memory with consolidation and principled decay for LLM agents and assistants.

[![PyPI](https://img.shields.io/badge/pypi-coming_soon-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-in_progress-orange)]()

Engram is a memory layer for LLM systems that does what existing memory libraries don't: it *consolidates*. Raw events get abstracted into general patterns, redundant or contradicted memories decay, and retrieval reads across a hierarchy from specific episodes to compressed knowledge — the way human memory actually works.

It is designed as a single primitive that serves both **agents** (procedural memory: "in situations like this, this approach worked") and **assistants** (semantic memory: "the user has a golden retriever they care about"), with the same algorithm and the same API.

---

## Why Engram exists

Every production LLM system today has a memory problem. The complaint is universal — *"it doesn't remember me"* — and the typical solution is some variation of *"dump everything into a vector store and search by cosine similarity."*

That isn't memory. It's a logbook with a search bar.

Real memory does three things that current systems don't:

1. **Consolidates.** Many specific events become one general principle. "User mentioned dog in conv 3, vet in conv 7, kibble in conv 12" becomes "user has a dog they actively care about."
2. **Forgets selectively.** Routine, redundant, or contradicted memories decay; surprising, frequently-used, or recently-relevant ones strengthen.
3. **Reads across abstractions.** Retrieval sometimes wants the general pattern, sometimes the specific episode, sometimes both. Flat stores can't do this cleanly.

Engram is built around these three principles.

---

## The core idea

Engram organizes memory as a hierarchy with three things flowing through it:

```
                      ┌──────────────────────────────┐
                      │   Consolidated abstractions  │   ← retrieved when general
                      │   (semantic / procedural)    │      patterns suffice
                      └──────────────▲───────────────┘
                                     │  consolidation
                                     │  (clustering + abstraction)
                      ┌──────────────┴───────────────┐
                      │   Mid-level summaries        │
                      │   (episode clusters)         │
                      └──────────────▲───────────────┘
                                     │
                      ┌──────────────┴───────────────┐
                      │   Raw event log              │   ← retrieved when
                      │   (episodic memory)          │      specifics matter
                      └──────────────────────────────┘
                                     │
                                     ▼
                              decay function
                          (recency, reinforcement,
                           corroboration, contradiction)
```

The four moving parts:

- **Event log.** Every observation lands here first, with provenance and timestamp.
- **Consolidation pass.** Periodically (or on trigger), the system clusters related events, extracts abstractions, and links them to their supporting evidence.
- **Decay.** Memories at every level have weights driven by reinforcement (was this retrieved? did it lead to a successful outcome?), corroboration (how many independent events support it?), contradiction (was it ever overruled?), and recency.
- **Hierarchical retrieval.** Queries read across the whole hierarchy, preferring abstractions when they suffice and drilling into specifics when they don't.

---

## Quick start

```bash
pip install engram   # placeholder — package not yet published
```

```python
from engram import Memory

memory = Memory(
    backend="sqlite",                    # or "postgres", "duckdb"
    embedding_model="text-embedding-3-small",
    consolidation_model="gpt-4o-mini",   # used for abstraction extraction
    consolidation_interval=50,           # consolidate every 50 events
)
```

That's the whole setup. Engram is model-agnostic — bring whichever LLM and embedding model you want for the consolidation and retrieval steps.

---

## Usage

### As assistant memory

```python
# Observe events as they happen
memory.observe("User mentioned they have a golden retriever named Max.")
memory.observe("User asked about senior dog food.")
memory.observe("User said Max is 9 years old and slowing down.")

# ... many sessions later ...

context = memory.retrieve("what should I know about the user's pets?")
# Returns:
# [
#   {
#     "level": "abstraction",
#     "content": "User has an aging golden retriever (Max, ~9yo) and is
#                 actively researching senior dog care.",
#     "confidence": 0.91,
#     "supported_by": [event_ids...],
#   },
#   ... lower-level episodes available if needed
# ]
```

### As agent procedural memory

```python
# Record what the agent tried and what happened
memory.observe({
    "type": "procedure",
    "situation": "API returned 429 rate limit",
    "action": "exponential backoff with jitter, retry up to 5x",
    "outcome": "success",
})

# Later, in a new situation
procedures = memory.retrieve_procedures(
    situation="hitting 503 errors from downstream service"
)
# Returns consolidated procedures from analogous past situations,
# ranked by how often they worked.
```

### Manual consolidation and inspection

```python
# Trigger consolidation explicitly (e.g. during downtime)
memory.consolidate()

# Inspect what's been consolidated
memory.summary()

# Resolve a contradiction manually
memory.reconcile(memory_id="...", resolution="prefer_recent")
```

---

## How it works under the hood

### Consolidation

When triggered, the consolidation pass:

1. **Clusters** recent unconsolidated events using embedding similarity, with a configurable cohesion threshold.
2. **Extracts abstractions** from each cluster using a cheap LLM call. The prompt is structured to produce *generalizations*, not summaries.
3. **Links** abstractions to their supporting events (provenance is always preserved).
4. **Detects contradictions** with existing abstractions and flags them for resolution.
5. **Promotes** stable, frequently-corroborated abstractions to higher levels of the hierarchy.

### Decay

Each memory item carries a weight $w \in [0, 1]$. The weight evolves as:

$$
w_{t+1} = w_t \cdot \alpha^{\Delta t} + \beta \cdot r_t + \gamma \cdot c_t - \delta \cdot x_t
$$

Where $r_t$ is reinforcement (was it retrieved and useful?), $c_t$ is new corroboration, $x_t$ is contradiction, and $\alpha, \beta, \gamma, \delta$ are tunable. Items below a threshold are pruned (or kept as cold storage, depending on configuration).

### Retrieval

Retrieval is **coarse-to-fine** by default: search abstractions first, then drill into supporting episodes only if the query demands specifics or the top-level results are low-confidence. This is both faster and more grounded than flat vector search.

---

## What makes Engram different

| | Flat vector store | mem0 / similar | **Engram** |
|---|---|---|---|
| Stores raw events | ✓ | ✓ | ✓ |
| Summarization | — | ✓ | ✓ |
| **Multi-level hierarchy** | — | — | ✓ |
| **Principled decay** | — | partial | ✓ |
| **Contradiction handling** | — | — | ✓ |
| **Provenance tracking** | — | partial | ✓ |
| **Procedural memory** | — | — | ✓ |
| **Coarse-to-fine retrieval** | — | — | ✓ |

The headline difference is that Engram treats memory as a *living hierarchy that changes over time*, not as a static append-only store with a search index. The downstream effects — better recall on long conversations, transferable agent procedures, graceful handling of stale or contradictory information — fall out of that.

---

## Benchmarks

Evaluation is in progress. Planned benchmarks:

- **LongMemEval** — long-horizon conversational memory.
- **LoCoMo** — multi-session dialogue with memory recall.
- **Custom procedural transfer benchmark** — does an agent with Engram do better on tasks it has seen analogues of? (Constructed from agent traces.)

Baselines: flat vector store (Chroma, Pinecone), mem0, Letta/MemGPT, full-context (where feasible).

Results will be published in the paper and tracked in [`/benchmarks`](./benchmarks).

---

## Roadmap

Stage-by-stage breakdown — including cross-cutting standards on speed, quality, and security — in [`ROADMAP.md`](./ROADMAP.md). High-level milestones:

**v0.1 — Core primitive.**
Event log, basic consolidation, decay, coarse-to-fine retrieval. SQLite backend. Reference benchmarks against flat vector store.

**v0.2 — Procedural memory.**
First-class support for action/situation/outcome triples. Procedure retrieval API. Agent-framework integrations (LangGraph, LlamaIndex, raw OpenAI).

**v0.3 — Contradiction and temporal reasoning.**
Trust-weighted conflict resolution, temporal segmentation ("X was true until March"), explicit invalidation.

**v0.4 — Multi-tenant and production.**
Postgres backend, async API, observability, memory inspector UI.

**v1.0 — Paper + stable API.**
Frozen public API, full benchmark suite, peer-reviewed paper.

---

## Research

Engram is an applied research project as much as a library. The paper-track contributions:

- A formal framing of memory as a hierarchical decay process with measurable consolidation quality.
- Algorithmic choices (when to consolidate, what to abstract, how to decay) with ablations.
- A unified primitive for episodic→semantic consolidation and episodic→procedural abstraction.
- Benchmarks against existing memory libraries and flat baselines.

Drafts and notes live in [`/research`](./research).

---

## Citation

If you use Engram in research, please cite (citation will be added when the paper is on arXiv):

```bibtex
@misc{engram2026,
  title  = {Engram: Hierarchical Memory with Consolidation and Decay for LLM Systems},
  author = {[your name]},
  year   = {2026},
  note   = {Preprint forthcoming},
}
```

---

## Contributing

Engram is early. The most useful contributions right now:

- **Benchmark runs** — reproducing baselines, finding failure modes.
- **Algorithmic experiments** — alternative consolidation strategies, decay functions, retrieval policies.
- **Integrations** — bindings for popular agent/RAG frameworks.
- **Edge cases** — adversarial conversations or agent traces that break the current implementation.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for setup and conventions.

---

## License

MIT. See [`LICENSE`](./LICENSE).

---

## Acknowledgments

Engram draws on ideas from cognitive neuroscience (complementary learning systems, episodic-to-semantic consolidation, Ebbinghaus decay), spaced repetition systems, and prior memory libraries (mem0, Letta/MemGPT, Zep). Standing on shoulders.
