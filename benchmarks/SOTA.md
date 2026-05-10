# State of the art — targets, baselines, bets

To beat SOTA you have to know where the goalposts are. This file is the running answer for Engram. It is updated when:

- A paper or library publishes new headline numbers on a suite we track.
- We run our own benchmarks and the scoreboard shifts.
- An algorithmic bet pays off (or doesn't).

The scoreboard view — current best public vs. current Engram — lives in `SCOREBOARD.md`. This file explains the *why* behind the numbers there.

---

## Benchmark suites

### LongMemEval

**What.** Long-horizon conversational memory benchmark with ~500 sessions per simulated user; questions are designed to require recall across sessions.

**Source.** Wu et al., 2024 — https://github.com/xiaowu0162/LongMemEval

**Why we care.** Cleanest published probe of multi-session recall. Most failure modes of flat RAG systems show up here: context dilution, recency bias, missed cross-session links.

**Engram targets.**
- `v0.1`: meet or beat the best public number cited in `SCOREBOARD.md` at release time, on a single fixed split, within our latency budget.
- `v1.0`: ≥ 5 points absolute over best public, on the same split, within budget.

**Honest note.** SOTA numbers move. Each run manifest pins the comparison to a specific paper / repo version; stale comparisons don't count.

---

### LoCoMo

**What.** Multi-session dialogue benchmark (~600 sessions) across five question categories: single-hop, multi-hop, temporal, open-domain, adversarial.

**Source.** Maharana et al., 2024 — https://github.com/snap-research/LoCoMo

**Why we care.** Tests temporal reasoning and adversarial recall. Both areas where flat stores trip — temporal because they have no notion of validity windows, adversarial because they retrieve plausibly-relevant-but-wrong matches.

**Engram targets.**
- `v0.1`: meet best public on the splits where flat RAG breaks (temporal, adversarial). Match-or-better on single-hop.
- `v0.3` (after temporal reasoning ships): beat best public on temporal questions.
- `v1.0`: dominate the adversarial split — ≥ 10 points absolute over the best non-Engram approach.

---

### Custom procedural transfer

**What.** A new benchmark we construct: agent traces grouped into "task families." For each family, the agent has seen N analogous tasks; it then attempts a held-out task. Score = success rate of an Engram-backed agent vs. a no-memory baseline and an episodic-only baseline.

**Source.** Built in `benchmarks/suites/procedural/` (Stage 7).

**Why we care.** No existing benchmark probes whether procedural memory transfers across analogous situations. This is the *agent-side* pitch of Engram (semantic memory for assistants vs. procedural memory for agents). If we don't have a benchmark, we can't claim the pitch is real.

**Engram targets.**
- `v0.2`: define and publish the benchmark; beat no-memory baseline by ≥ 15%.
- `v1.0`: beat naive episodic-only retrieval (no abstraction layer) by ≥ 10%.

---

## Baselines we measure against

Every release benchmark run includes a fresh comparison against the best public implementation of each architecture class. Adapters live in `benchmarks/baselines/<name>/`.

| Class | Implementation | Notes |
|---|---|---|
| Flat dense | Chroma + OpenAI embeddings | The "logbook with a search bar" baseline. |
| Hybrid dense+sparse | Chroma + BM25 | Hybrid is widely known to beat pure dense; we measure it explicitly so the comparison is fair. |
| Hierarchical paged | Letta / MemGPT | Page-based hierarchy with recall API. |
| Graph-RAG | Zep / Graphiti | Knowledge-graph memory with temporal edges. |
| Graph-RAG | Cognee | Alternative graph approach. |
| PageRank-RAG | HippoRAG | Multi-hop traversal via PageRank. |
| Summarization | mem0 | Strong reported numbers on LoCoMo. |
| Zettelkasten | A-MEM | Recent paper with linked-note structure. |
| Long-context | Full-context (Sonnet/Opus, 1M) | Upper-bound where feasible; not memory but a useful ceiling. |

We do not bundle every adapter as a runtime dependency of Engram itself. Adapters install via the harness's own extras (`pip install -e "./benchmarks[baselines]"`).

---

## Why we believe we can win

The bet is that **combining several individually-proven techniques cleanly** beats the best single-technique baselines on long-horizon and adversarial questions. Ordered by expected impact:

1. **Hybrid retrieval (dense + BM25) at every level.** Dense captures semantics; sparse catches names, numbers, identifiers, rare tokens. Hybrid almost always beats either alone — but most memory libraries ship dense-only.
2. **Hierarchical reads.** A single abstraction can answer many specific questions at lower cost and higher confidence. We read coarse first, drill fine only when needed. This compounds with hybrid retrieval — abstractions are short and dense; drill-down into events is where sparse pays off.
3. **Decay that's actually principled.** Reinforcement, corroboration, and contradiction signals all feed weights. Stale or contradicted facts fall below threshold instead of haunting retrieval forever. The README formula is real, observable, and tunable.
4. **Trust-weighted contradiction handling.** Most libraries either ignore contradictions or naively prefer recency. We resolve with provenance count, recency, and explicit `source.trust` together.
5. **Procedural path.** Most memory libraries are declarative-only. The procedural path (`situation → action → outcome`) is a separate, optimized code path for agent transfer learning — and the only path that addresses the procedural-transfer benchmark above.
6. **Cross-encoder re-rank.** Cheap and consistently helps when top-k is small. A no-brainer.
7. **Hard-grounded consolidation prompt.** Generalizations only, JSON-validated, with provenance enforced at the storage layer (CHECK constraint on `supported_by` non-empty). Most "summarization" memory libraries lose specificity by collapsing into prose.
8. **Determinism for replay.** Every component takes an injectable clock and RNG. We can re-run a benchmark exactly and bisect regressions — which means we can keep getting better, not just be lucky once.

None of these are individually novel. The bet is that the *combination*, plus the engineering rigor (perf budgets, property tests, prompt-injection corpus, reproducible manifests), is what produces SOTA-crushing numbers.

---

## Reproducibility

Every benchmark run produces a manifest committed to `benchmarks/runs/<date>-<short-sha>-<suite>.json` containing:

- Git commit, dirty flag, Python version, OS, CPU model, RAM.
- Engram config (decay coefficients, consolidation interval, retrieval `k`, hybrid weights, re-rank on/off).
- Provider config (or fake-provider hash for deterministic runs).
- Dataset version + checksum.
- Per-question scores.
- Aggregate metrics with bootstrap confidence intervals (n = 1000).
- Latency histograms (P50, P95, P99).

CI runs the smoke benchmark on every PR with the fake provider. Full benchmarks against real providers run on tagged releases or manual `workflow_dispatch`. **A run that can't be reproduced doesn't count toward the scoreboard.**

---

## How we keep up

The scoreboard (`SCOREBOARD.md`) is updated:

- On every Engram release, after the release benchmark run.
- Whenever a tracked baseline publishes new numbers (we follow the relevant authors / repos).
- Quarterly, manually, by re-checking the leaderboards for each suite.

If a baseline ships a new technique that significantly beats us, we don't hide it. We update the scoreboard, add a roadmap entry to chase it, and tag the commit so the regression is auditable.
