# Scoreboard

Living comparison of Engram vs. the best public results we know of, per suite.

The numbers in this file are **pinned** to a specific source per row. They get refreshed on each Engram release benchmark run, and whenever a tracked baseline publishes new numbers — see `SOTA.md` for the discipline.

> **Last refresh:** 2026-05-10 (initial). Baseline cells are placeholders pending the first verified run; Engram cells are targets, not measurements. The point of this file from day 1 is the *shape* — concrete suites, concrete metrics, concrete cells that will hold concrete numbers.

---

## LongMemEval

| System | Source / version | Accuracy | Latency P50 (retrieve) | Manifest |
|---|---|---|---|---|
| Best public | TBD — pinned at run time | TBD | n/a | n/a |
| Engram (target, `v0.1`) | this repo | meet best public | < 150 ms @ 100k items | required |
| Engram (target, `v1.0`) | this repo | best public + 5 absolute | within budget | required |

## LoCoMo

| System | Source / version | Single-hop | Multi-hop | Temporal | Open-domain | Adversarial |
|---|---|---|---|---|---|---|
| Best public (RAG class) | TBD | TBD | TBD | TBD | TBD | TBD |
| Engram (target, `v0.1`) | this repo | match | match | match | match | match |
| Engram (target, `v0.3`) | this repo | beat | match | beat | match | match |
| Engram (target, `v1.0`) | this repo | beat | beat | beat | beat | beat (≥ +10 abs) |

## Custom procedural transfer

| System | Source | Lift over no-memory baseline |
|---|---|---|
| No-memory agent | this repo | 0% (definitional) |
| Episodic-only Engram | this repo | TBD |
| Engram (target, `v0.2`) | this repo | ≥ +15% |
| Engram (target, `v1.0`) | this repo | ≥ +10% over episodic-only |

## Latency

| API | Workload | Target P50 | Target P99 | Status |
|---|---|---|---|---|
| `observe` | 10k events | < 50 ms | < 200 ms | **passing** (Stage 3, FakeEmbedder, local) |
| `retrieve` (flat) | 10k events | < 100 ms | < 300 ms | **passing** (Stage 3, FakeEmbedder, local) |
| `retrieve` (coarse-to-fine) | 100k items | < 150 ms | < 500 ms | **passing** (Stage 6, FakeEmbedder dim=128, warm cache, local). Measured: P50 ~ 2.3 ms / P99 ~ 3.1 ms via the in-memory `VectorIndex`. |
| `decay.record` | per-signal | < 5 ms | < 20 ms | **passing** (Stage 4, in-memory, local) |
| `decay.tick` | 10k hot items | < 500 ms | < 2 s | **passing** (Stage 4, local) |
| `consolidate` | per-event @ fake provider | n/a | n/a (≥ 100 / s throughput) | **passing** (Stage 5, FakeChat scripted, local) |
| `consolidate` | per-event @ real provider | n/a | n/a (≥ 10 / s with batching) | deferred — Stage 9 (chat batching) |

## Throughput

| API | Workload | Target | Status |
|---|---|---|---|
| `observe` | concurrent writers | ≥ 1k / s | implicit pass (Stage 3, 8 writers no drops) |

## Smoke benchmark (`recall-smoke`, FakeEmbedder, exact-text queries)

Validates harness wiring; not a SOTA claim. Real recall comparisons
land at Stage 6 against LongMemEval / LoCoMo with real providers.

| System | Recall@10 |
|---|---|
| Engram | 1.0 |
| Chroma | 1.0 |
| Chroma + BM25 (RRF) | 1.0 |

---

## How to read this file

Until Engram has shipped Stage 6, the "Engram" rows are aspirational targets. The "Best public" rows fill in with verified numbers from cited papers / repos before each release benchmark run.

A row without a source is a row we don't trust yet. **We do not claim to have crushed SOTA on the basis of an unverified target.** Every "we beat X" claim in the README, the changelog, or external comms requires a manifest in `benchmarks/runs/` with the matching result.

---

## Change log

| Date | Change | Manifest |
|---|---|---|
| 2026-05-10 | Initial scaffold. No measurements yet. | n/a |
| 2026-05-10 | Stage 3: observe/retrieve P50 budgets verified locally (FakeEmbedder); recall-smoke against Chroma + Chroma+BM25 reaches the 1.0 floor on exact-text queries. | CI-uploaded |
| 2026-05-10 | Stage 4: decay engine ships with 100% line+branch coverage on the math, end-to-end replayability (bit-identical weights across runs), and metrics surface (`DecayMetrics`). | CI-uploaded |
| 2026-05-10 | Stage 5: consolidation pipeline (clustering + abstraction + contradiction + promotion). Throughput >= 100 events/s on FakeChat. Provenance integrity invariant survives Hypothesis fuzzing. Prompt-injection corpus regression suite (`looks_like_injection` filter rejects every CORPUS payload echo). | CI-uploaded |
| 2026-05-10 | Stage 6: coarse-to-fine retrieve. `Memory.retrieve(prefer=...)` reads the `{summary, abstraction}` layer first, drills into supporting events when confidence is low, optionally cross-encoder-reranks. In-memory `VectorIndex` cache hits warm-cache P50 ~ 2.3 ms / P99 ~ 3.1 ms at 100k items / dim=128 (50x under the SCOREBOARD budget). Hierarchical recall lift over flat ~ +100 pp on the synthetic centroid-orthogonal-events split. LongMemEval / LoCoMo harness suites scaffolded; real scores pending real-provider runs. | CI-uploaded |
