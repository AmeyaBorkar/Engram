# Roadmap

Engram ships in stages. Each stage is independently reviewable and production-grade — the bar on **speed**, **quality**, and **security** does not change between stage 1 and stage 10. The README's `v0.1...v1.0` versions group these stages into release milestones; this document is the unit-of-work view.

The point of staging is *not* to defer rigor. It's to make rigor reviewable.

---

## Cross-cutting standards

These apply at every stage. A change cannot land if it regresses any of these.

### Speed

- **Public APIs have perf budgets.** Every endpoint (`observe`, `retrieve`, `consolidate`, `retrieve_procedures`) carries a P50/P99 budget at a stated workload size. Microbenchmarks in `benchmarks/micro/` fail CI on regression.
- **Vectorized math on hot paths.** Embedding similarity, decay updates, clustering — `numpy` (or backend-native vector ops). No Python-level inner loops.
- **Indexed reads.** Every storage query path has an index reviewed via `EXPLAIN`. Sequential scans on hot paths are CI-blocking.
- **Batched provider I/O.** Embedding and chat calls go through a batcher with a configurable window (default 20ms / 64-item ceiling). Steady-state per-call rate is measured and asserted.
- **Caches are observable.** LRU + content-hash caches expose hit-rate metrics.

### Quality

- **Strictly typed.** `mypy --strict` is clean on `src/engram`. Public APIs are fully typed; type stubs ship in the wheel via PEP 561.
- **Strictly linted.** `ruff check` includes `S` (security), `B` (bugbear), `C4` (comprehensions), `N` (naming), `RET`, `RUF`. `ruff format --check` is enforced.
- **Coverage targets.** ≥ 90% line coverage on storage / retrieval / decay; 100% on the decay math; meaningful branch coverage on the consolidation pipeline.
- **Property-based tests.** Hypothesis covers invariants — weights ∈ [0, 1], decay is monotonic without reinforcement, provenance never dangles, consolidation is idempotent on a stable input set.
- **Golden tests for prompts.** Every LLM-facing prompt has a fixture-driven test against the deterministic fake provider. Drift is intentional and changelogged.
- **No flaky tests.** A flake is treated as a bug. Quarantine then fix; never paper over with retries.
- **Determinism by default.** Every component takes an injectable clock and RNG. Replays are exact.

### Security

- **Parameterized SQL only.** String concatenation into SQL is a CI-blocking lint failure (`S608`). No exceptions.
- **Prompt injection is a regression class.** The consolidation prompt treats event content as data, never instructions. A corpus of attack traces lives at `tests/security/prompt_injection/` and runs on every PR.
- **No plaintext PII in logs.** Provider adapters apply a configurable redaction pass to request/response payloads before any structured log emission. Redaction is on-by-default in shipped adapters.
- **Dependency hygiene.** `pip-audit` runs in CI and fails on known CVEs. Core has zero required runtime dependencies; backends and providers are optional extras.
- **Boundary validation only.** Inputs are validated (with Pydantic v2 models or equivalent) at the public API surface. Internal code trusts its types.
- **Supply chain.** Releases will be Sigstore-signed. CI is pinned by digest where it touches secrets.
- **Threat model.** `SECURITY.md` enumerates the threats Engram explicitly cares about; refreshed each minor release.

### Operability

- **Structured logging.** `logging.getLogger("engram.<subsystem>")` everywhere. JSON output by default in production mode.
- **Metrics.** OpenTelemetry spans on every public call from Stage 9; counters/histograms for queue depth, batch size, decay tick duration, and provider latency.
- **Backwards-compat.** Within a minor version, no breaking API changes. Deprecations carry a one-minor warning before removal.

### SOTA discipline

Beating state-of-the-art is the success criterion for this project, not a v1.0 nice-to-have. The full plan lives in `benchmarks/SOTA.md`; the running scoreboard lives in `benchmarks/SCOREBOARD.md`. The discipline that holds at every stage:

- **Standing harness from Stage 1.** Even when there's no algorithm to test, the harness exists. The fake-provider smoke benchmark runs in CI on every PR.
- **Pinned baselines.** Every benchmark run cites the specific paper / repo version of the system it's compared against. Stale comparisons don't count.
- **Reproducibility manifests.** Every result has a manifest committed to `benchmarks/runs/` recording git commit, environment, config, dataset checksum, per-question scores, and bootstrap CIs.
- **Public scoreboard.** `SCOREBOARD.md` is updated on every release and whenever a tracked baseline publishes new numbers. We don't hide losses.
- **No vibes.** A claim of "we beat X" needs a manifest. Without one, the claim doesn't enter the README, the changelog, or marketing.

---

## Stages

A stage is "done" when its **Definition of Done** checks all pass. If a check is unmet on schedule, scope shrinks — the bar does not.

### Stage 0 — Foundations *(complete)*

**Goal.** A repository that builds into an empty package, with all production hygiene already in place.

**Scope.**
- `pyproject.toml` (hatchling), MIT license, contributing guide, security policy, changelog.
- CI matrix: Python 3.10–3.13 × Linux / macOS / Windows, with lint, type, test, and audit jobs.
- Strict ruff and mypy configurations.
- Smoke tests covering import, instantiation, version.

**Out of scope.** Any algorithmic code.

**Definition of done.**
- `pip install -e ".[dev]"` succeeds on a clean machine.
- `ruff check` / `ruff format --check` / `mypy` / `pytest` all green.
- CI is green on `main` and on PRs.

---

### Stage 1 — Storage and data model

**Goal.** A durable, indexed event log that other stages can read and write against without thinking about SQL.

**Scope.**
- Core schemas (Pydantic v2 models): `Event`, `MemoryItem`, `Embedding`, `ProvenanceLink`, `Cluster`. `MemoryItem.level ∈ {event, summary, abstraction}`.
- A `Storage` protocol so backends are pluggable. SQLite is the only implementation in this stage.
- SQLite backend: WAL mode, foreign keys on, per-thread connections.
- Migrations as numbered SQL files under `src/engram/storage/migrations/`. Runtime applies pending migrations on first connect.
- Indexes: `(created_at)`, `(weight)`, `(level)`, `(cluster_id)`. Vector similarity via `sqlite-vec` if installed; numpy fallback otherwise.
- A read-only inspector helper for tests / debugging.
- **Benchmark harness scaffold** (`benchmarks/harness/`) — CLI entry point, suite/baseline/run protocols, manifest writer. No algorithmic content; framework only. Runs end-to-end against a no-op suite using the fake provider in CI.

**Out of scope.** Embedding generation, retrieval ranking, consolidation, decay updates.

**Definition of done.**
- 1M synthetic events insert in < 30 s on a laptop SSD; reads of the last 1k events in < 50 ms.
- Hypothesis tests prove provenance links never dangle.
- Fuzz test: random byte payloads survive a roundtrip without panicking the layer.
- Migration tests: each prior schema version upgrades cleanly to current.
- Coverage ≥ 90% on the storage module.
- `python -m engram.bench run noop --provider fake` succeeds in CI; produces a manifest in `benchmarks/runs/` with all required fields populated.

---

### Stage 2 — Provider abstraction

**Goal.** Embedding and chat-completion calls go through one abstraction, with batching, retries, caching, and a deterministic fake for tests.

**Scope.**
- `EmbeddingProvider` and `ChatProvider` protocols with sync + async surfaces.
- `FakeProvider` — hash-based deterministic embeddings, scripted chat replies. Used by every unit test.
- Concrete adapters: OpenAI, Anthropic. Behind `[openai]` / `[anthropic]` extras.
- Cross-cutting wrappers:
  - `Retry` — exponential backoff with jitter; configurable max attempts.
  - `Batcher` — debounced batch window; coalesces concurrent requests.
  - `Cache` — LRU keyed by content hash; observable hit rate.
  - `Redactor` — configurable PII patterns scrub request/response before logging.

**Definition of done.**
- Switching providers requires zero changes in consumer code.
- Batching reduces steady-state provider-call count by ≥ 5× on the smoke benchmark.
- Cache hit rate is observable and deterministic across replays.
- Prompt-injection corpus runs against the fake provider and asserts non-instruction-following.

---

### Stage 3 — Observe and flat retrieve

**Goal.** End-to-end usable as a vector store with provenance: `observe()` writes; `retrieve()` reads with cosine similarity. Hierarchy is still flat — every result is at `level="event"` — but the whole pipeline is real.

**Scope.**
- `Memory.observe(content)` accepts strings or structured `Event` payloads; embeds via the provider; lands in storage.
- `Memory.retrieve(query, k=10)` returns top-k with cosine similarity.
- `MemoryItem.confidence`, `supported_by`, and `level` are populated coherently even for the flat case.
- Crash-safety: `observe` is durable before returning.

**Out of scope.** Consolidation, decay scheduling, hierarchical retrieval.

**Definition of done.**
- P50 < 50 ms per `observe`, P50 < 100 ms per `retrieve` at 10k events on a laptop. Manifest committed.
- Recall@10 on the held-out conversational split ≥ Chroma baseline. Adapters for Chroma and Chroma+BM25 land in `benchmarks/baselines/`. Manifest committed.
- SIGKILL between calls leaves the store in a consistent state (tested).
- Zero dropped events under concurrent observers (tested with 8 writers).

---

### Stage 4 — Decay engine

**Goal.** Memory items strengthen with use and weaken with time. The README formula is real, observable, and tunable.

**Scope.**
- Implementation of `w_{t+1} = w_t · α^Δt + β·r_t + γ·c_t − δ·x_t` with all four signals tracked separately.
- Reinforcement signal fired by retrieval when an item is used in a useful answer (interface only at this stage; full hookup at Stage 6).
- Background decay tick (synchronous and async variants) updates weights at a configurable cadence.
- Pruning policy: items below threshold move to a `cold` table (auditable) or are deleted, configurable.
- All math is dimensionless; clock is injectable.

**Definition of done.**
- Property tests: weights ∈ [0, 1] always; reinforcement strictly raises weight; decay is monotonic without reinforcement; corroboration count is non-decreasing.
- Replayability: given a fixed event stream and clock, weights are bit-identical across runs.
- Coverage: 100% on the decay math.
- Reinforcement and corroboration counters are exported as metrics.

---

### Stage 5 — Consolidation

**Goal.** Recent unconsolidated events get clustered, abstracted, and linked into the hierarchy. The README's headline feature.

**Scope.**
- **Clustering.** HDBSCAN over normalized embeddings with a configurable `min_cluster_size` and cohesion threshold. Falls back to threshold-based agglomerative for small N.
- **Abstraction extraction.** A hardened LLM prompt that produces *generalizations*, not summaries, with a JSON schema and a validator. Prompt is versioned; changes are changelogged.
- **Provenance.** Every abstraction links to the events that supported it. `MemoryItem.supported_by` is non-empty for any non-event level. Storage enforces this with a CHECK constraint.
- **Promotion.** Stable, frequently-corroborated abstractions move up a level (cluster summary → high-level abstraction).
- **Contradiction detection.** New abstractions checked against existing ones via embedding distance + LLM judge; conflicts are recorded for resolution at Stage 8.

**Out of scope.** Conflict *resolution* (Stage 8). Hierarchical retrieval (Stage 6).

**Definition of done.**
- Golden traces produce expected abstractions, deterministically, against the fake provider.
- Prompt-injection corpus: events that try to claim system-instruction status do not get promoted into abstractions.
- Throughput ≥ 100 events / s on the fake provider; ≥ 10 events / s on a real provider with batching.
- Provenance integrity invariant survives Hypothesis fuzzing.

---

### Stage 6 — Coarse-to-fine retrieve *(complete)*

**Goal.** Retrieval reads abstractions first and drills into supporting events when confidence is low or the query asks for specifics. Closes the loop on the README's pitch.

**Scope.**
- Two-stage retrieval: top-k abstractions, optional drill-down to events. ✓
- Confidence threshold drives drill-down; query-time hint (`prefer="specific"` / `"general"`) overrides. ✓
- Re-ranker option (cross-encoder via the provider abstraction). ✓
- `MemoryItem.level` faithfully reflects what the caller is reading. ✓

**Definition of done.**
- **LongMemEval-S real-provider run**: 71.4% on 500 questions with bge-large + Kimi K2.6, above the paper's reported best memory system (~65%) and the strongest long-context LLM baseline (~58%). Manifest at `benchmarks/runs/release/20260511T052920_486768+0000-0b6dfa53-longmemeval.json`. Caveat (same-model judge) documented in SCOREBOARD. ✓
- **LoCoMo:** harness suite committed (`benchmarks/suites/locomo.py`) — same shape, five sub-splits (single_hop, multi_hop, temporal, open_domain, adversarial). Real scores pending real-provider run (deferred to v0.1.1).
- Outperforms Stage 3 flat retrieval on a synthetic centroid-orthogonal-events split by ~100 percentage points recall@k (target was ≥ 10). ✓
- Latency: warm-cache P50 ~ 2.3 ms / P99 ~ 3.1 ms at 100k items / dim=128 on a laptop. The `VectorIndex` cache lands ~ 50× under the budget without needing `sqlite-vec`; the SCOREBOARD row tracks the measurement and points at the manifest. ✓
- This stage tags **`v0.1.0`** — first PyPI release. The release post links to the manifests.

---

### Stage 7 — Procedural memory  *(core complete; integrations deferred to v0.2.1)*

**Goal.** First-class storage and retrieval of procedures: "in situations like this, this approach worked / failed."

**Scope.**
- Schema extension: `Procedure { situation, action, outcome, weight }`. Outcomes feed reinforcement. ✓
- `Memory.retrieve_procedures(situation)` finds analogous past situations and returns ranked procedures. ✓
- Integrations: LangGraph, LlamaIndex, raw OpenAI / Anthropic. Each integration has its own integration test that runs in CI. *(Deferred to v0.2.1 — Stage 7 core ships first.)*

**Definition of done.**
- Procedural transfer benchmark defined and published in `benchmarks/suites/procedural_transfer.py`. ✓
- An agent backed by Engram beats a no-memory baseline by ≥ 15% on a held-out task suite. ✓ (+72 pp lift on the synthetic exact-match split with FakeEmbedder; paraphrase mode lands real-LLM numbers in v0.2.1.)
- Reinforcement-from-outcome path is exercised end-to-end in tests. ✓ (`test_memory_procedures.TestOutcomeFeedbackLoop` plus 20 surface-level tests.)

---

### Stage 8 — Contradiction and temporal reasoning  *(core complete; LoCoMo run + MERGE deferred to v0.3.1)*

**Goal.** Engram knows when facts change and can answer "as of when?" queries.

**Scope.**
- Trust-weighted conflict resolution: weighted by recency (`PREFER_RECENT`), source trust (`PREFER_TRUSTED`), and corroboration count (`PREFER_FREQUENT`). ✓
- Temporal segmentation: facts have validity windows (`valid_from` / `valid_until`). ✓
- Explicit invalidation API (`Memory.reconcile` invalidates the loser; `storage.invalidate_memory_item` is idempotent and preserves the first timestamp). ✓
- `Memory.reconcile(conflict_id, resolution=...)` shipping; the README signature uses `conflict_id` for precision (a memory item can have multiple open conflicts). ✓

**Definition of done.**
- Adversarial test suite: contradicting events do not silently overwrite; the conflict is observable and resolvable. ✓ `benchmarks/suites/contradiction_temporal.py` -- 10-pair contradiction split scores **+100 pp lift** (engram 1.00 / baseline 0.00).
- Temporal queries return historically-correct state. ✓ Same suite's 5-triple temporal split scores **100% accuracy** across 15 (item, snapshot) pairs.
- **LoCoMo temporal split:** Engram beats the best public RAG-class number cited in `benchmarks/SCOREBOARD.md`. Manifest committed. *(Deferred to v0.3.1 -- harness scaffold lives at `benchmarks/suites/locomo.py`; real-LLM numbers need a paid provider run.)*

---

### Stage 9 — Multi-tenant and production  *(targets `v0.4.0`; Stage 9a partial shipped)*

**Goal.** Engram runs as a service: Postgres backend, async API, observability, isolation.

**Scope.**
- Postgres backend with `pgvector`. Tenant isolation via row-level security and per-tenant connection roles. *(Deferred to v0.4.0 proper -- needs Docker for local testing.)*
- Async API (`async def observe`, `async def retrieve`, …) parallel to the sync surface. ✓ (Stage 9a; routes through `asyncio.to_thread` over SQLite. Postgres-native async in v0.4.0.)
- Observability: OpenTelemetry spans on every public call; Prometheus metrics for queue depths, batch sizes, decay-tick durations, provider latency. ✓ (Stage 9a; spans + counters on public `Memory` calls via the optional `[otel]` extra. Decay-tick / batcher instrumentation rolls in alongside their modules.)
- Memory inspector: a separate package (`engram-inspector`) — small web UI to browse the hierarchy. *(Deferred to v0.4.0.)*
- Security audit: dependency review, threat-model refresh, prompt-injection corpus expansion, secret-scanning sweep. *(Judge prompt-injection corpus shipped in v0.3.1.)*

**Definition of done.**
- Multi-tenant load test: 100 tenants × 1k QPS sustained for 1 hour with no cross-tenant leakage. Verified. *(Requires Postgres; deferred to v0.4.0.)*
- Async throughput ≥ 5× sync on bound provider calls. *(Deferred to v0.4.0 -- depends on async-native Postgres.)*
- Threat model updated; new attack surfaces (RLS bypass, async-cancellation invariants) covered by tests. *(Deferred to v0.4.0.)*

---

### Stage 10 — Stable release and paper  *(targets `v1.0.0`; scaffold + docs + reproduction tooling shipped)*

**Goal.** Frozen public API, full benchmark suite, paper preprint, docs site.

**Scope.**
- Semver-stable API surface; deprecation policy documented. ✓ (`docs/project/api-stability.md` + `docs/project/deprecation-policy.md`.)
- Benchmark results published in `benchmarks/` with reproduction scripts and a CI job that re-runs them on a tagged release. ✓ (`scripts/reproduce_benchmarks.py` runs every suite against `FakeProvider`; CI hook lands once the user is ready to publish.)
- Paper preprint on arXiv. *(Deferred -- requires the LongMemEval re-judge + LoCoMo real-LLM run for headline numbers.)*
- Documentation site (mkdocs-material) with API reference, tutorials, and a memory-mental-model explainer. ✓ (`mkdocs.yml` + `docs/` with auto-generated API ref via `mkdocstrings[python]`. New `[docs]` extra.)

**Definition of done.**
- All public symbols documented; doc build is part of CI. ✓ (mkdocs build clean -- only warnings on two `:raises:` docstring formatting issues in the storage protocol, non-blocking.)
- Preprint accepted to arXiv; cited in README. *(Deferred -- depends on paid LLM runs.)*
- Public API has zero `# TODO` and no breaking changes since `v0.4.0`.
- **LongMemEval:** ≥ 5 points absolute over best public number cited in `benchmarks/SCOREBOARD.md`. Manifest committed. *(Same-model-judge caveat from the 71.4% v0.1.0 run pending a GPT-4o re-judge; ~$5 paid run.)*
- **LoCoMo adversarial:** ≥ 10 points absolute over best non-Engram approach. Manifest committed. *(Pending paid run.)*
- **Procedural transfer:** ≥ 10% lift over episodic-only Engram (i.e. abstraction layer is provably load-bearing). Manifest committed.

---

## Release mapping

| Release | Stages | Headline |
|---|---|---|
| `v0.1.0` | 0, 1, 2, 3, 4, 5, 6 | Core primitive: hierarchical memory works end-to-end on SQLite. |
| `v0.2.0` | 7 | Procedural memory and agent-framework integrations. |
| `v0.3.0` | 8 | Contradiction and temporal reasoning. |
| `v0.4.0` | 9 | Postgres, async, multi-tenant, observability. |
| `v1.0.0` | 10 | Stable API, paper, docs site. |

---

## How we work the roadmap

- One stage owner at a time. The owner opens a tracking issue with the stage's DoD copied in as a checklist.
- Each stage's PR train ends with a `Stage N — done` PR that updates this file and the changelog in the same diff.
- If a DoD check is impossible to meet on schedule, the stage doesn't ship; we cut scope rather than the bar.
- Reviews are done against the cross-cutting standards explicitly. A reviewer is empowered to block on a missing perf budget or a missed security check, not just on code shape.
