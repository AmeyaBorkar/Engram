# Engram

> Hierarchical memory with consolidation and principled decay for LLM agents and assistants.

[![PyPI](https://img.shields.io/pypi/v/engrampy?color=blue&cacheSeconds=600)](https://pypi.org/project/engrampy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-1267_passing-success)](#status)
[![mypy](https://img.shields.io/badge/mypy-strict-success)](#status)

Engram is a memory layer for LLM systems that does what existing memory libraries don't: it *consolidates*. Raw events get abstracted into general patterns, redundant or contradicted memories decay, and retrieval reads coarse-to-fine across the resulting hierarchy.

It's designed as a single primitive that serves both **agents** (procedural memory: "in situations like this, this approach worked") and **assistants** (semantic memory: "the user has a golden retriever named Max").

```bash
pip install engrampy
```

---

## Why Engram exists

Every production LLM system today has a memory problem. The complaint is universal — *"it doesn't remember me"* — and the typical solution is some variation of *"dump everything into a vector database with a search bar."*

That isn't memory. It's a logbook with a search bar.

Real memory does three things that current systems don't:

1. **Consolidates.** Many specific events become one general principle. "User mentioned dog in conv 3, vet in conv 7, kibble in conv 12" becomes "user has a dog they actively care about."
2. **Forgets selectively.** Routine, redundant, or contradicted memories decay; surprising, frequently-used, or recently-relevant ones strengthen.
3. **Reads across abstractions.** Retrieval sometimes wants the general pattern, sometimes the specific episode, sometimes both. Flat stores can't do this cleanly.

Engram is built around these three principles.

---

## The core idea

Engram organizes memory as a hierarchy, with consolidation flowing upward and decay applied at every level:

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

- **Event log.** Every observation lands here first, with provenance and timestamp. Writes are atomic with their embedding.
- **Consolidation pass.** Periodically (or on trigger), the engine clusters related events, extracts abstractions, and links them to their supporting evidence.
- **Decay.** Memories at every level carry weights driven by reinforcement (was this retrieved? was it useful?), corroboration (how many independent events support it?), contradiction detection, and recency.
- **Hierarchical retrieval.** Queries read across the whole hierarchy, preferring abstractions when they suffice and drilling into specifics when they don't. Conflict-aware: contradictory memories co-surface when the resolution matters.

---

## Quick start

```python
from engram import Memory, SqliteStorage
from engram.providers.openai import OpenAIEmbedder, OpenAIChat

storage = SqliteStorage("engram.db")
storage.initialize()                                # applies pending migrations

memory = Memory(
    storage=storage,
    embedder=OpenAIEmbedder(),
    chat=OpenAIChat(),                              # optional; needed for consolidate / merge-reconcile
    # consolidate_chat=OpenAIChat(model="gpt-4o"), # optional; stronger model for abstraction
)
```

Or use the context-manager idiom — `SqliteStorage.__enter__` calls `initialize()` and `__exit__` cleans up:

```python
with SqliteStorage("engram.db") as storage:
    memory = Memory(storage=storage, embedder=OpenAIEmbedder(), chat=OpenAIChat())
    memory.observe("...")
```

Test-mode (no API keys):

```python
from engram import Memory, SqliteStorage
from engram.providers import FakeEmbedder, FakeChat

with SqliteStorage(":memory:") as storage:
    memory = Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=128),
        chat=FakeChat(default="canned reply"),
    )
```

---

## Usage

### Assistant memory

```python
memory.observe("User mentioned they have a golden retriever named Max.")
memory.observe("User asked about senior dog food.")
memory.observe("User said Max is 9 years old and slowing down.")

# Periodically (or on a timer)
memory.consolidate()

# Later — coarse-to-fine retrieval over the hierarchy
results = memory.retrieve("what should I know about the user's pets?", k=5)
for r in results:
    print(r.level, r.score, r.content)
# AbstractionLevel.ABSTRACTION  0.91  "User has an aging golden retriever (Max, ~9yo)
#                                      and is actively researching senior dog care."
# AbstractionLevel.EVENT         0.78  "User said Max is 9 years old and slowing down."
# ...
```

### Agent procedural memory

```python
memory.record_procedure(
    situation="API returned 429 rate limit",
    action="exponential backoff with jitter, retry up to 5x",
    outcome="success",
)

# Later, in a new situation
procedures = memory.retrieve_procedures(
    situation="hitting 503 errors from downstream service",
    k=5,
)
# Returns consolidated procedures from analogous past situations,
# ranked by how often they worked.
```

### Temporal queries

```python
from datetime import datetime, timezone

# State as of a past date — invalidated memories surface again
past = memory.retrieve(
    "what pet does the user have?",
    k=5,
    as_of=datetime(2026, 3, 1, tzinfo=timezone.utc),
)
```

### Conflict-aware retrieval

```python
# Co-surfaces contradictory memory_items so the caller can resolve
# rather than silently picking one side
from engram.retrieve import RetrieveParams

results = memory.retrieve(
    "what is the user's current job?",
    retrieve_params=RetrieveParams(surface_conflicts=True),
)
```

---

## How it works under the hood

### Consolidation

When triggered, the consolidation pass:

1. **Clusters** recent unconsolidated events using embedding similarity, with a configurable cohesion threshold.
2. **Extracts abstractions** from each cluster using a chat provider. The prompt is structured to produce *generalizations*, not summaries.
3. **Detects contradictions** with existing abstractions, persists `Conflict` rows, and gates promotion on resolution.
4. **Links** abstractions to their supporting events — provenance is always preserved.
5. **Promotes** stable, frequently-corroborated abstractions to higher levels of the hierarchy.

`Memory.aconsolidate` runs clusters concurrently via `asyncio.gather` with a semaphore-bounded ceiling — ~30× speedup on typical workloads vs the serial path.

### Decay

Each memory item carries a weight `w ∈ [0, 1]` that evolves with reinforcement, corroboration, contradiction, and time. Items below threshold are pruned. The math is deterministic — given a fixed clock and signal stream, replays produce identical weights.

### Retrieval

Coarse-to-fine by default: search abstractions first, drill into supporting episodes only when the query demands specifics or the top-level results are low-confidence. Hybrid stack underneath: dense embeddings + BM25 lexical + recent-window stream, fused via reciprocal rank fusion, optionally cross-encoder-reranked (`BAAI/bge-reranker-v2-m3` by default behind the `[reranker]` extra), with vectorized MMR for diversity and an additive recency boost.

---

## What's in 0.3.0

`engrampy==0.3.0` (released 2026-05-18) is a correctness, security, and benchmark release. 236 commits since 0.2.1. Highlights:

- **Seven-cluster security + correctness audit** landed real fixes across storage transactions/locks/migrations, retrieval pipeline correctness, consolidation/reconcile invariants, providers/decay/schemas/OTel, and the bench harness. Plus expanded prompt-injection detection (Unicode confusables, RTL marks, base64 payloads, multilingual variants) and a [`SECURITY.md`](./SECURITY.md) threat model.
- **LongMemEval benchmark harness matured significantly**: cap-fix diagnostics, six prompt variants (`v2 / v2a / v2b / v2c / v3 / v3a`), deterministic calculator tools (`--enable-tools`), content-filter fallback (`--chat-fallback`), `--chat-max-tokens` override for any OpenAI-compatible provider, stratified sampling, parallel evaluation, GPU concurrency cap.
- **Analysis tooling** for benchmark forensics: [`benchmarks/re_judge.py`](./benchmarks/re_judge.py) (re-score with pinned judge snapshot), [`benchmarks/compare_manifests.py`](./benchmarks/compare_manifests.py) (per-question diff), [`benchmarks/cum_accuracy.py`](./benchmarks/cum_accuracy.py) (per-question cumulative trajectory).
- **Library hardening**: `Event.content` cap raised 64 KiB → 1 MiB; `SCHEMA_VERSION` + `extra=forbid` on persisted models; `observe_many` batched ingest; agent `achat` async surface; `Verdict` re-exported from `engram.schemas`.
- **Integrations refined**: llamaindex `get_all` + kwargs handling; langgraph `config` arg + `ainvoke`.

See [`CHANGELOG.md`](./CHANGELOG.md) for the full breakdown.

---

## Benchmarks

The success criterion for Engram is **measurably better recall on long-horizon memory tasks**, not just better in principle. We track the runs in [`benchmarks/SCOREBOARD.md`](./benchmarks/SCOREBOARD.md) with full methodology disclosures and link every claim to a committed manifest.

### LongMemEval-S (n=500, full population) — 0.3.0 snapshot

| Judge config | Score | Notes |
|---|---:|---|
| `openai/gpt-4o` (floating alias, as-run on 2026-05-18) | **86.95%** | accuracy_correct (433/498) |
| `openai/gpt-4o-2024-08-06` (paper-default snapshot, no rubric mod) | **87.75%** | Most apples-to-apples vs published systems that report "gpt-4o" judge |

Wilson 95% binomial CI on 433/498: **[83.6%, 89.5%]**. Per-qtype: sss-assistant 100%, sss-user 94.3%, knowledge-update 89.6%, sss-preference 86.2%, temporal-reasoning 84.2%, multi-session 78.9%.

**Config:** Kimi K2.6 actor (open-weight, MIT-modified), BAAI/bge-large-en-v1.5 fp32 embedder, bge-reranker-v2-m3, k=20, `v3a` prompt, `--enable-tools`, `--seed 1337`. Full reproducibility metadata in the manifest at `benchmarks/runs/20260518T033410_441206+0000-bb7c8412-dirty-longmemeval.json`.

### Honest framing

**This is not a SOTA claim.** A 5-agent post-run audit on 2026-05-18 surfaced multiple published systems above us with comparable judge configurations:

| Above us | Score | Actor | Judge |
|---|---:|---|---|
| Honcho (Claude Haiku 4.5) | 90.4% | smaller actor | gpt-4o |
| Mastra-OM (Gemini-3-Flash) | 89.20% | frontier-mini actor | gpt-4o |
| Lumetra Engram | 91.6% | GPT-5 | gpt-4o |
| Honcho (Gemini-3-Pro) | 92.6% | stronger actor | gpt-4o |
| Mastra-OM (Gemini-3-Pro) | 93.27% | stronger actor | gpt-4o |
| Mastra-OM (gpt-5-mini) | 94.87% | stronger actor | gpt-4o |

The defensible framing is **"first published reproducible LongMemEval-S result with an open-weight actor under the paper-default gpt-4o judge protocol"** — a real but narrow first. See [`JOURNEY.md`](./JOURNEY.md) §27 for the full audit.

### Methodology disclosures

Before citing any of these numbers, read the disclosures at the top of [`benchmarks/SCOREBOARD.md`](./benchmarks/SCOREBOARD.md). The load-bearing ones:

- `v3a` prompt injects qtype-conditional hints for `multi-session` and `single-session-preference` (the other four qtypes get the base prompt). Standard LongMemEval protocol does not expose qtype to the actor — this is a documented deviation.
- `--enable-tools` is deterministic regex substitution for `SUM`/`COUNT`/`AVG`/`MIN`/`MAX`/`*_BETWEEN` ops, not external knowledge.
- Kimi K2.6 is open-weight but **frontier-tier reasoning class** (GPQA-Diamond 90.5%, AIME 96.4%), not gpt-4o-class.
- The full n=500 SOTA manifest is `git_dirty=True`; a clean-tree re-run is open work.

### Tracked suites

- **LongMemEval-S** — long-horizon conversational memory (above).
- **LoCoMo** — multi-session dialogue with temporal/adversarial splits. Harness scaffolded; full runs pending.
- **Custom procedural transfer** — does an agent with Engram do better on tasks it has seen analogues of? Constructed from agent traces. Suite scaffolded.
- **Recall-smoke** — harness wiring check against Chroma and Chroma+BM25 baselines. Not a SOTA claim.

The plan, targets, and reproducibility discipline are in [`benchmarks/SOTA.md`](./benchmarks/SOTA.md).

---

## Status

| Quality gate | State |
|---|---|
| `pytest -x -q` | 1267 passed, 1 skipped, 17 deselected (~63s) |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 205 files formatted |
| `mypy --strict src/engram` | No issues across 77 source files |
| `twine check dist/*` | Wheel + sdist PASSED |

---

## Roadmap

Stage-by-stage breakdown — including cross-cutting standards on speed, quality, and security — in [`ROADMAP.md`](./ROADMAP.md). High-level:

**v0.1 — Core primitive.** Event log, basic consolidation, decay, coarse-to-fine retrieval. SQLite backend. Shipped.

**v0.2 — Retrieval-side hybrid stack + procedural memory.** BM25 + RRF + MMR + cross-encoder reranker, recent-window stream, conflict-aware retrieve, async parallel consolidation, persistent disk cache, asymmetric query prompts. First-class procedural memory (`record_procedure` / `retrieve_procedures`). Shipped.

**v0.3 — Correctness, security, benchmark depth.** Seven-cluster audit; LongMemEval harness improvements; prompt variants, tools, fallback; analysis tooling. Shipped (current).

**v0.4 — Path-to-90 + Postgres backend.** Sub-session chunking (highest-leverage retrieval change per the recall diagnostic), temporal qtype hints, multi-session verification, Postgres backend with row-level security for read-side tenant scoping, observability surface.

**v1.0 — Stable API + paper.** Frozen public API, full benchmark suite reproducing all claims on a clean tree, peer-reviewed paper.

---

## Research

Engram is an applied research project as much as a library. The paper-track contributions are:

- A formal framing of memory as a hierarchical decay process with measurable consolidation quality.
- Algorithmic choices (when to consolidate, what to abstract, how to decay) with ablations.
- A unified primitive for episodic→semantic consolidation and episodic→procedural abstraction.
- Reproducible benchmarks against existing memory libraries with full methodology disclosure.

Notes live in [`/research`](./research) and the audit narrative is preserved in [`JOURNEY.md`](./JOURNEY.md).

---

## Citation

```bibtex
@software{engram2026,
  title   = {Engram: Hierarchical Memory with Consolidation and Decay for LLM Systems},
  author  = {Borkar, Ameya},
  year    = {2026},
  url     = {https://github.com/AmeyaBorkar/Engram},
  version = {0.3.0},
}
```

A peer-reviewed paper is in progress; this citation will be updated when it lands on arXiv.

---

## Contributing

Engram is early. The most useful contributions right now:

- **Benchmark runs** — reproducing results on a clean tree, surfacing failure modes, running configurations we haven't (clean v1-prompt baseline, judge-ensemble experiments).
- **Algorithmic experiments** — alternative consolidation strategies, decay functions, retrieval policies. Sub-session chunking is the open lever for path-to-90%.
- **Integrations** — bindings for popular agent / RAG frameworks beyond LangGraph and LlamaIndex.
- **Edge cases** — adversarial conversations or agent traces that break the current implementation.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for setup and conventions.

---

## License

MIT. See [`LICENSE`](./LICENSE).

---

## Acknowledgments

Engram draws on ideas from cognitive neuroscience (complementary learning systems, episodic-to-semantic consolidation, Ebbinghaus decay), spaced repetition systems, and prior memory libraries (MemGPT, Letta, Zep, Graphiti, mem0, A-MEM).
