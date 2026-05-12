# Changelog

All notable changes to Engram are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Once we ship `v1.0.0`, the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html); pre-1.0 releases may break the public API on minor version bumps.

## [Unreleased]

## [0.2.0] — 2026-05-12

### Changed

- **Distribution name moved to `engrampy`**. The original `engram-memory`
  PyPI name was claimed out from under the project on 2026-02-10 by an
  unrelated party publishing 0.4.0 / 0.4.1 / 0.5.0b1 pointing at
  `github.com/Ashish-dwi99/Engram`. The Python import name is unchanged
  (`from engram import Memory` works as before). Users on `engram-memory`
  ≤ 0.1.0 should switch to `pip install engrampy`. PEP 541 reclaim
  requests for both `engram` (placeholder) and `engram-memory` (active
  squat) are pending.

### Added

- **Retrieval-side hybrid stack** (Stage F):
  - `engram.retrieve._bm25.BM25Index` — vectorized (numpy scatter-add) BM25
    with Lucene-style k1/b parameters, lazy index build, frozen-after-search
    invariant.
  - `engram.retrieve._bm25.reciprocal_rank_fusion` — RRF for fusing dense +
    lexical + recent-window candidate streams.
  - `engram.retrieve._mmr.mmr_select` — vectorized greedy MMR with
    min-max-normalized relevance so `λ` controls the diversity trade-off
    as documented (fixed a math bug where unnormalized cross-encoder
    logits dwarfed the redundancy term).
  - Recency boost on the rerank scores: `score + λ·exp(-days/decay)`.
    Additive (not multiplicative) so it preserves direction on negative
    reranker logits.
  - Per-question recent-window stream and per-question auto-temporal year
    extraction in the LongMemEval bench suite, with empty-pool fallback.
- **Cross-encoder reranker**: `engram.retrieve.BGEReranker` (BAAI/bge-reranker-v2-m3
  by default). Reads through the `[reranker]` extra.
- **Async parallel consolidation**: `Memory.aconsolidate` and
  `ConsolidationEngine.aconsolidate` use `asyncio.gather` over clusters with
  a semaphore-bounded concurrency limit. ~30× speedup on typical
  LongMemEval haystacks vs serial.
- **Persistent disk cache** for `(chat, embed)` provider responses:
  `engram.providers._disk_cache.CachedChat` / `CachedEmbedder` /
  `with_disk_cache()`. SQLite-backed; useful for benchmark re-runs.
- **Asymmetric query prompts** in `LocalEmbedder` (e.g. `s2p_query` for
  Stella, instruction prefix for E5), plus a per-instance LRU result cache.
- **Storage performance**: composite indexes (migration `0009_perf_indexes.sql`),
  PRAGMA `cache_size = 64MB` + `mmap_size = 256MB`, batched embedding +
  `created_at` lookups via `get_embeddings_batch` / `get_created_at_batch`.
- **OpenRouter chat + embedder** behind a single `OPENROUTER_API_KEY`
  (`engram.providers.openrouter`). Optional `HTTP-Referer` /
  `X-Title` ranking headers documented in `.env.example`.
- **`Level.GLOBAL` + `Level.TOPIC`** hierarchy levels and aggregate
  `user_state` storage.
- **Conflict-aware retrieval**: `RetrieveParams.surface_conflicts` co-surfaces
  contradictory memory_items when a confident answer exists.
- **Bench harness improvements**:
  - `engram-bench run longmemeval` now wires every retrieve-side knob:
    `--bm25-weight`, `--mmr-lambda`, `--recency-lambda`, `--lexical-filter`,
    `--auto-temporal`, `--recent-window-k`, `--disk-cache`, `--drill-k`,
    `--confidence-threshold`, `--rerank-pool-multiplier`, `--bm25-k1`,
    `--bm25-b`, `--recency-decay-days`, `--mmr-pool-size`.
  - Per-question exception isolation: a content-filter rejection or
    network blip on one question scores it 0 and continues the run instead
    of crashing.
  - `--consolidate` triggers the parallel `aconsolidate` path.
- **Evaluation infrastructure**:
  - `scripts/retrieval_eval.py` — retrieval-only evaluator with
    recall@k / hit@k / multi_recall@k / MRR / first-correct-rank /
    precision@k at multiple k cutoffs. No LLM calls.
  - `scripts/ablate_longmemeval.py` — per-question, per-config ablation
    matrix with retrieval-only or full LLM scoring.
  - `scripts/sweep.py` — single-knob hyperparameter sweep with bootstrap CIs
    and McNemar significance tests vs a baseline value.
  - `scripts/retrieval_trace.py` — per-question per-stage trace dump
    (dense top-N, BM25 top-N, recent-window, plus final top-k per config),
    with answer-session annotation.
  - `scripts/run_all_evals.py` — orchestrator that runs all of the above
    in one command and emits a consolidated `REPORT.md`.
  - `scripts/_stats.py` — bootstrap mean / paired-diff CIs and McNemar's
    exact / chi-square tests.
- **`docs/EVAL_PROTOCOL.md`** — formal per-component evaluation protocol:
  hypothesis, knobs, primary metric, decision rule, known issues. Standard
  sweep grids, statistical-test definitions, reproducibility checklist.

### Fixed

- MMR no longer degenerates to relevance-sort when cross-encoder logits
  have a wide range — relevance is min-max normalized to [0, 1] inside
  `mmr_select` before greedy selection so `λ` has the documented meaning.
- Recency boost no longer inverts on negative reranker logits — switched
  to additive `score + λ·decay`.
- LongMemEval haystack date strings now parsed into `Event.created_at`
  so `--recency-lambda` and recent-window retrieval have real timestamps
  to work with.

### Added

- Project scaffolding: `pyproject.toml` (hatchling build), `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `.gitignore`.
- `ROADMAP.md` — staged plan from foundations through `v1.0`, with cross-cutting standards on speed, quality, security, operability, and SOTA discipline.
- CI workflow (`.github/workflows/ci.yml`): lint (ruff), type (mypy strict), test matrix on Python 3.10–3.13 across Linux/macOS/Windows, smoke benchmark (`engram.bench run noop`), and dependency audit (`pip-audit`).
- `tests/test_smoke.py` — package-level import / instantiation / version checks.
- Empty `Memory` class re-exported from `engram` (implementation forthcoming per `ROADMAP.md` Stage 1+).
- PEP 561 marker (`py.typed`) so consumers receive the package's type information.
- SOTA infrastructure: `benchmarks/SOTA.md` (suites, baselines, algorithmic bets, reproducibility discipline), `benchmarks/SCOREBOARD.md` (running comparison), and placeholder directories `benchmarks/{harness,baselines,suites,runs}/`. Stage DoDs in `ROADMAP.md` now reference scoreboard targets.
- **Stage 1 — Storage and data model**:
  - Core schemas (`engram.schemas`): `Event`, `MemoryItem`, `Embedding`, `ProvenanceLink`, `Cluster`, plus `Level` and `ItemKind` enums. Pydantic v2 models with bounds checks on `weight`/`cohesion` and dim/vector consistency.
  - Time-ordered identifiers (`engram.ids.new_id`): UUIDv7 (RFC 9562) implemented for Python 3.10+ since the stdlib generator only ships in 3.14.
  - `Storage` protocol (`engram.storage.Storage`) — pluggable backend interface; SQLite is the only implementation in Stage 1, Postgres lands in Stage 9.
  - `SqliteStorage` — WAL mode, foreign keys on, per-thread connections, parameterized queries everywhere. CHECK constraints on `weight`, `cohesion`, `level`, `item_kind`, `dim`. Provenance linkage uses `ON DELETE CASCADE` for memory items and `ON DELETE RESTRICT` for events to enforce referential integrity.
  - Migration runner (`engram.storage.migrations`): numbered SQL files with self-recording version inserts, idempotent re-application, atomic-per-migration.
  - `0001_initial.sql` — five tables (`events`, `clusters`, `memory_items`, `embeddings`, `provenance_links`) with indexes on `created_at`, `weight`, `level`, `cluster_id`, `(item_id, item_kind)`, and both directions of provenance.
  - Read-only inspector (`engram.storage.stats`) for tests and manifests.
- **Stage 1 — Benchmark harness scaffold**:
  - `engram.bench` package with CLI (`python -m engram.bench` / `engram-bench` console script) supporting `run <suite> --provider fake --runs-dir <path>`.
  - `Suite` protocol, `SuiteResult` dataclass, `Provider` protocol stub with deterministic `FakeProvider`.
  - `Manifest` writer: captures git commit/dirty, Python version, OS, CPU, RAM (cross-platform), provider hash, dataset version/checksum, aggregate metrics, CIs, per-question scores, and latency histograms. Manifests are JSON-stable (sorted keys, indent=2).
  - `noop` suite — verifies harness end-to-end, runs in CI on every PR.
- **Stage 1 — Tests**: 77 tests covering CRUD, transactions, FK integrity, migration idempotency, `pytest.raises` assertions, plus Hypothesis property tests (provenance never dangles, weights stay bounded, metadata survives arbitrary unicode), fuzz tests against random byte payloads, and opt-in performance tests (`pytest -m slow`) for the DoD numbers (1M inserts < 30 s, last-1k reads < 50 ms). Storage module coverage: 98%.
- **Stage 2 — Provider abstraction**:
  - `engram.providers.Retry` — exponential backoff with optional jitter, sync + async (`call` / `acall`), injectable clock and RNG for deterministic tests.
  - `engram.providers.Cache` — LRU cache with observable `hits` / `misses` / `hit_rate`; `__contains__` doesn't count as a lookup. `content_hash(*parts)` builds keys with NUL-separated SHA-256 to avoid concatenation collisions.
  - `engram.providers.Redactor` — regex-based PII / secret scrubbing with a default pattern set (Anthropic and OpenAI keys, AWS access keys, Bearer tokens, emails, US phones, SSNs, credit-card-shaped digits) and `Redactor.redact_obj(...)` for nested dict / list / tuple structures.
  - `engram.providers.EmbeddingProvider` and `ChatProvider` protocols with sync + async surfaces, plus `Message` / `Role` types.
  - `engram.providers.FakeEmbedder` / `FakeChat` — deterministic fakes for tests. Hash-based unit-norm embeddings; scripted chat with content-hash keys and a default fallback that includes the input hash.
  - `engram.providers.openai.OpenAIEmbedder` / `OpenAIChat` — adapters behind the `[openai]` extra (`text-embedding-3-small` / `gpt-4o-mini` defaults, `dimensions` arg passed only when non-native, opaque `completion_kwargs`).
  - `engram.providers.anthropic.AnthropicChat` — adapter behind the `[anthropic]` extra (`claude-haiku-4-5-20251001` default, system messages extracted to top-level `system` arg, content blocks concatenated, non-text blocks dropped).
  - `engram.providers.Batcher` — thread-based debounced batcher; coalesces concurrent `submit(item)` calls into one `fn(list)`. Observable `call_count` so tests and benchmarks can assert the Stage 2 DoD (≥ 5× call-count reduction). Exception propagation to all waiters in a batch; result-count mismatches are a hard error.
  - `engram._security.prompt_injection.CORPUS` — eight known prompt-injection attack styles with explicit `forbidden_substrings` per entry. Stage 5+ regression-tests the consolidation prompt against this corpus; Stage 2 exercises the parametrized test surface against `FakeChat`.
  - `engram.bench.FakeProvider` upgraded to bundle the Stage 2 `FakeEmbedder` and `FakeChat`; harness's `Provider` protocol still narrow (`name`, `manifest_hash`).
  - `dev` extra now installs `openai` and `anthropic` so mypy can typecheck adapter modules locally; end users still install via `[openai]` / `[anthropic]` extras.
- **Stage 2 — Tests**: 106 new tests covering retry exponential growth + jitter determinism + async paths, cache LRU eviction + promotion + hit-rate accounting, redactor pattern coverage and nested-container traversal, fake-provider determinism + manifest-hash stability + protocol satisfaction, OpenAI / Anthropic adapters via `unittest.mock` (no network), batcher concurrency + exception propagation, prompt-injection corpus shape and FakeChat resistance.
- **CI**: `pip-audit` job now uses `--skip-editable` so the (unpublished) editable `engram` install doesn't cause a 404 lookup; all third-party deps are still audited.
- **Stage 3 — Observe and retrieve**:
  - `engram.Memory(storage=..., embedder=...)` ships its public surface: `observe(content)` (string or `Event`) embeds, normalizes to unit-norm, and persists event + embedding atomically; `retrieve(query, k=10)` returns `RetrievalResult`s ranked by cosine similarity (every result is `level=EVENT` until Stage 5 adds abstractions).
  - `engram.RetrievalResult` schema — frozen Pydantic model with `level`, `content`, `confidence` (clamped to `[0, 1]`), `score`, `supported_by`, `item_id`. The schema is the same shape Stage 6 uses for hierarchical results.
  - `Storage.search_event_embeddings(query_vec, k, model)` — protocol method backing `Memory.retrieve`. SQLite implementation does an `O(N)` brute-force scan with `np.frombuffer` + `argpartition`. Stage 6 swaps to `sqlite-vec` behind the same protocol.
  - `numpy>=1.26` is now a core runtime dependency.
  - **Crash-safety**: `test_observe_durable_across_sigkill` spawns a writer subprocess, kills it (`SIGKILL` / `TerminateProcess`), and verifies the database is consistent on reopen.
  - **Concurrent writers**: `test_concurrent_observers_no_drops` runs 8 threads × 50 observes each through one shared `Memory` and asserts all 400 events land with unique contents.
  - **Perf budgets** (opt-in `pytest -m slow`): observe P50 < 50 ms and retrieve P50 < 100 ms at 10k events. Both pass locally.
- **Stage 3 — Benchmark baselines**:
  - `engram.bench.Retriever` protocol + `Hit` dataclass — common surface every benchmark baseline implements.
  - `engram.bench.EngramRetriever` — adapter from `engram.Memory` to the protocol.
  - `benchmarks/baselines/chroma.py` — Chroma adapter with custom `EmbeddingProvider` injection (apples-to-apples comparison against Engram on the same vectors).
  - `benchmarks/baselines/chroma_bm25.py` — dense + sparse hybrid via Reciprocal Rank Fusion. Self-contained BM25 implementation in `_bm25.py` (no `rank_bm25` dependency).
  - `[bench]` extra (`chromadb>=0.5`) for opt-in install.
- **Stage 3 — Smoke benchmark suite**:
  - `benchmarks/suites/recall_smoke.py` indexes a 30-doc synthetic conversational corpus into every available retriever and computes recall@10 on exact-text queries. Local run shows all three retrievers at recall@10 = 1.0 (the floor).
  - CI now runs `engram-bench run recall-smoke --provider fake` on every PR (Ubuntu 3.13 only, with `[dev,bench]` extras).
  - The runner's suite-name lookup now maps CLI-friendly hyphens to Python module underscores.
- **Stage 4 — Decay engine**:
  - `engram.DecayParams` — frozen dataclass that owns the formula's tunables. Parameterized by `half_life_seconds` (default 30 days), `beta` / `gamma` / `delta` (signal gains), and `threshold` (prune cutoff). `__post_init__` enforces non-negative gains, threshold ∈ [0,1], and a positive finite half-life. The per-second `alpha` derives from half-life as `0.5 ** (1 / half_life_seconds)`.
  - `engram.decay._math.apply()` — pure formula `w_{t+1} = clamp01(w_t * alpha^dt + beta*r + gamma*c - delta*x)` with strict input validation. The `clamp01` helper maps NaN to 0.0 so a corrupted input fails loudly via the prune path rather than silently poisoning every comparison. **100% line + branch coverage** on the math module per the Stage 4 DoD.
  - `engram.DecayState` schema — frozen Pydantic model holding the per-row mutable state (`weight`, three signal counters, `last_decayed_at`, optional `cold_at`). Lives alongside the immutable `Event` / `MemoryItem` row.
  - **Storage migration `0002_decay.sql`** — adds `weight`, `reinforcement_count`, `corroboration_count`, `contradiction_count`, `last_decayed_at`, `cold_at` to both `events` and `memory_items` (memory_items already had `weight`). Indexes on `events.weight`, `events.cold_at` (partial), `memory_items.cold_at` (partial). Backfills `last_decayed_at` from the row's existing timestamp so v1 databases upgrade cleanly.
  - **Storage CRUD** — `get_decay_state`, `iter_decay_states` (batched streaming via `cursor.fetchmany`), `update_decay_state`, `mark_cold` / `unmark_cold`, `count_cold`, `delete_cold_items`, `decay_totals`. Per-kind SQL is pre-built into module-level dicts so the runtime path never f-strings into `execute` (S608). `delete_cold_items(EVENT)` refuses to delete events that participate in provenance links (FK is `ON DELETE RESTRICT`); callers in that situation should use the `cold` policy.
  - `engram.DecayEngine(storage, *, params, prune_policy, clock, kinds, batch_size)`:
    - `record(item_id, kind, *, reinforcement, corroboration, contradiction, now)` — eager path; one transaction reads the state, applies decay-since-last + the new signal, bumps the counter, clamps the weight, and pushes the row cold if it crosses the threshold. `Memory.reinforce` / `corroborate` / `contradict` are thin wrappers.
    - `tick(*, now)` / `tick_async(*, now)` — periodic sweep; iterates every hot item (per-kind transaction), applies pure decay, marks newly-cold items, and under the `delete` prune policy purges cold rows physically. Returns a `TickResult` with `items_processed` / `items_pruned` / `items_deleted` / `duration_ms` / per-kind breakdown.
    - `metrics()` — `DecayMetrics` snapshot with hot / cold gauges, signal totals (over hot rows only - cold rows don't conflate the active surface), the cached `last_tick`, and a per-kind `KindCounters` map.
  - `Memory(decay_params=..., prune_policy=..., clock=...)` — Stage 4 surface on the public API:
    - `reinforce` / `corroborate` / `contradict` / `tick` / `tick_async` / `is_cold` / `metrics`.
    - `retrieve(query, k, *, include_cold=False)` — cold items filtered out by default; explicit override available for audit flows.
  - **Crash-safe and replayable**: every record/tick is one storage transaction, and with an injected clock + pre-seeded UUIDs two runs of the same operation sequence produce bit-identical decay states (weights, counters, timestamps, cold markers all match exactly). Stage 4 DoD verified by `tests/test_decay_replay.py`.
  - **Property tests** (`tests/test_decay_properties.py`): across arbitrary mixes of reinforce / corroborate / contradict / tick, weights stay in [0, 1], the three signal counters are non-decreasing, pure decay is monotonic, and reinforcement with headroom strictly raises the weight (or pins to 1.0).
- **Stage 5 — Consolidation**:
  - `engram.consolidation.cluster(vectors, *, params)` — clusters unit-norm embeddings using HDBSCAN (optional `[consolidation]` extra) or a pure-numpy single-link agglomerative fallback. `ClusterParams.method="auto"` picks HDBSCAN when installed and `N >= auto_hdbscan_min_n`. Stable order across runs (HDBSCAN itself is deterministic; agglomerative uses an explicit i<j scan + lower-index-as-root union).
  - `engram.consolidation.AbstractionRequest` / `AbstractionResult` — the validated I/O of one abstraction call. Pydantic frozen schemas with strict bounds.
  - `engram.consolidation.extract_abstraction(request, chat)` — sends a versioned prompt (`prompts/abstract_v1.txt`) to the chat provider, parses strictly with Pydantic, retries once on malformed JSON.
  - **Prompt-injection defense** (DoD: "events that try to claim system-instruction status do not get promoted into abstractions"): hardened `abstract_v1.txt` frames observations as data, plus a final `looks_like_injection(text)` pattern filter (`engram._security.prompt_injection`) rejects abstraction outputs whose text matches known injection patterns even after JSON validation succeeds. Regression suite `tests/test_consolidation_prompt_injection.py` runs every CORPUS attack through prompt rendering, parser, and end-to-end consolidation.
  - **Storage seams** for the engine: `Storage.iter_unconsolidated_events_with_embeddings(model, limit, batch_size)` streams `(Event, vector)` pairs for events with no provenance link in deterministic `(created_at, id)` order; `Storage.insert_memory_item_with_provenance(item, event_ids, *, cluster, embedding, provenance_weights)` is the atomic write path that refuses to land non-event memory items without supporting events; `Storage.search_memory_item_embeddings(query_vec, *, k, model, levels, exclude_ids, include_cold)` is the vector-recall used by contradiction detection; `Storage.iter_memory_items(*, level, include_cold, batch_size)` and `Storage.update_memory_item_level(item_id, level)` back the promotion pass.
  - `engram.consolidation.ConsolidationEngine.consolidate(*, max_events)` — full pipeline: stream unconsolidated events, cluster, ask the chat provider for one abstraction per cluster, embed the abstraction, atomically write cluster + memory_item + embedding + provenance links. Provenance weights default to 1.0 for events the LLM marked as load-bearing (`AbstractionResult.supports`) and `support_weight` (0.5 by default) for the rest. Each successful row's `metadata["consolidation"]` records `prompt_version`, `confidence`, `cohesion`, `supports`, observation count, and any conflicts.
  - **Contradiction detection** (`engram.consolidation._contradiction`): vector recall + LLM judge. Versioned `prompts/judge_v1.txt`. Verdicts (`agree`/`contradict`/`unrelated`) parsed via Pydantic. Conflicts recorded on the new memory_item's `metadata["consolidation"]["conflicts"]` for Stage 8's resolver. Off by default (`ContradictionParams(enabled=False)`); turn on with similarity threshold (default 0.7) and max_candidates (default 3).
  - **Promotion** (`engram.consolidation.PromotionParams`, `Memory.promote()`): elevates `Level.SUMMARY` items to `Level.ABSTRACTION` when corroboration_count >= min_corroboration (3), contradiction_count <= max_contradiction (0), weight >= min_weight (0.5), and `metadata["consolidation"]["conflicts"]` is empty. Off by default; opt in once the corpus has corroboration history.
  - `Memory(chat=...)` — the public seam grows a chat provider argument. `Memory.consolidate(...)` and `Memory.promote(...)` are the public entry points; `Memory.consolidator` exposes the underlying engine for callers who need fine-grained control.
  - **Throughput**: Stage 5 DoD target ≥ 100 events/s on the fake provider passes on the slow-marked `test_consolidate_throughput_at_least_100_events_per_second`. The "≥ 10 events/s on a real provider with batching" target is deferred to Stage 9 — chat-provider batching is an architectural addition tied to the Stage 9 batcher work.
  - **Provenance integrity** (DoD): Hypothesis property test `test_provenance_integrity_invariant` runs many random consolidation scenarios and asserts every non-event memory item has at least one provenance link. The dual invariant — consolidated events disappear from `iter_unconsolidated_events_with_embeddings` — is also pinned.
  - Hatch wheel build now force-includes `engram/consolidation/prompts/` so the prompt fixtures ship alongside the source.
- **Stage 6 — Coarse-to-fine retrieve**:
  - `engram.HierarchicalRetriever` — the Stage 6 engine. Pipeline: top-`k * candidate_multiplier` over `{Level.SUMMARY, Level.ABSTRACTION}`; per hit, keep the abstraction when `confidence >= confidence_threshold` (or `prefer="general"`), otherwise drill into supporting events, score them fresh against the query, and emit the top `drill_k`. `prefer="specific"` skips the abstraction layer entirely (Stage 3 flat behavior). Empty hierarchy + `prefer="auto"` falls through to events so pure-vector-store callers still get useful answers.
  - `engram.RetrieveParams` — frozen dataclass with `k`, `prefer` (`Literal["auto","specific","general"]`), `confidence_threshold`, `drill_k`, `candidate_multiplier`, `include_cold`, `reinforce_on_use`. `__post_init__` validates bounds.
  - `Memory.retrieve(query, k=None, *, prefer=None, confidence_threshold=None, drill_k=None, include_cold=None, reinforce=None, reranker=None)` — backwards-compatible with the Stage 3 surface; per-call kwargs override Memory-level defaults set via the new `retrieve_params=` and `reranker=` constructor arguments.
  - `engram.Reranker` protocol + `engram.FakeReranker` — optional cross-encoder seam for reordering the merged candidate set. The fake is a deterministic token-overlap scorer with a configurable similarity blend; real cross-encoders (BGE, Cohere Rerank, …) implement the same protocol.
  - **Reinforcement-on-use**: `RetrieveParams.reinforce_on_use=True` (default) plumbs every surfaced item through `DecayEngine.reinforce`. Cold items are silently skipped (the engine refuses to reinforce them by design); raced deletions are caught and logged. Closes the retrieval / decay loop the README pitches.
  - **Vector index** (`engram.storage._vector_index.VectorIndex`): per-`(item_kind, model)` numpy matrix cached in process memory. Lazy build on first search, dirty-flagged on `insert_embedding` / `mark_cold` / `unmark_cold` / `delete_cold_items` / `update_memory_item_level`. After the matrix lookup, one small `SELECT id, content WHERE id IN (...)` joins the content for the top-k -- no content materialization for the discarded `(n - k)` rows.
  - **Latency** (DoD: P50 < 150 ms, P99 < 500 ms at 100k items): warm-cache `retrieve(query, k=10)` measures **P50 ~ 2.3 ms / P99 ~ 3.1 ms** on a laptop in-memory SQLite + FakeEmbedder dim=128 -- ~50× under the budget. Cold-cache rebuild after a 100k-item write burst ~ 450 ms. Slow-marked `test_retrieve_warm_p50_p99_under_budget` asserts the budget in CI; `benchmarks/suites/latency_at_scale.py` emits the SCOREBOARD-pinned manifest.
  - **Recall lift over flat** (DoD: ≥ 10 percentage points): `tests/test_retrieve_recall_lift.py` plants a synthetic split where event vectors are orthogonal to the topic centroid and only the consolidated summary embeds onto the centroid query. Hierarchical recall@k = 1.0; flat recall@k ~ 0; lift ~ 100 percentage points at k = 1 and k = 3.
  - **Benchmark suites**: `benchmarks/suites/longmemeval.py` and `benchmarks/suites/locomo.py` -- harness scaffolds that load the public splits from `benchmarks/datasets/{longmemeval,locomo}/<split>.jsonl`, run the hierarchical retrieve, score against the dataset answer, and emit a manifest. Datasets are not vendored; CI smoke-runs without them and emits placeholder results. Real LLM-judge scoring lands once the bench provider is driven by a real chat model.
  - `engram.bench` re-exports the new types so the harness can introspect a Memory instance without importing internals; the smoke benchmark suite still runs flat retrieve via `prefer="specific"` for apples-to-apples comparison against Chroma.
  - `Storage.score_events_by_ids(query_vec, event_ids, *, model)` -- new protocol method backing the drill-down path. Bypasses the vector index for small bounded id sets.
  - **Tests**: 22 new tests covering all four pillars (empty-hierarchy fallback, `prefer="general"`, `prefer="auto"` with confidence-threshold drill, `prefer="specific"` flat path, `RetrievalResult.level` fidelity, reinforcement on use, reranker plumbing, validation), plus the recall-lift split and two slow-marked latency assertions.
- **Stage 6 — LongMemEval-S real-provider benchmark** (the v0.1.0 receipt):
  - First end-to-end run with real models: **71.4% accuracy** on the full 500-question LongMemEval-S split, using `BAAI/bge-large-en-v1.5` (local, GPU) as embedder and Kimi K2.6 via OpenCode Go for both answer generation and the LongMemEval-style yes/no judge. Manifest at `benchmarks/runs/release/20260511T052920_486768+0000-0b6dfa53-longmemeval.json`.
  - Per-type accuracy: 94.6% single-session-assistant, 84.3% single-session-user, 72.2% temporal-reasoning, 69.2% knowledge-update, 60.2% multi-session, 50.0% single-session-preference.
  - **Context for the headline number**: above the LongMemEval paper's reported best memory system (~65%), above the strongest long-context LLM baseline (Claude-3.5-Sonnet at ~58%), and above mem0's post-paper claim (~67%). With **no reranker, no HyDE, no consolidation** — purely Engram's hierarchical retrieve + bge-large + Kimi.
  - **Honest caveats**: Kimi judges its own answers (self-preference bias likely adds 3-7 absolute points); n=500 gives a bootstrap 95% CI of roughly ±4 points; no contemporaneous re-runs of mem0/Letta on the same machine for apples-to-apples.
  - **Reproducibility infrastructure** that made this run possible: LongMemEval download script (`scripts/fetch_longmemeval.py` pulls `xiaowu0162/longmemeval-cleaned` from HuggingFace), `.env` loading via `python-dotenv` + `.env.example` template, real-provider builders in `engram.bench._real_provider` (OpenAI, Anthropic, Moonshot direct, OpenCode Zen, OpenCode Go), CLI flags `--embedder local --chat opencode-go --chat-model kimi-k2.6 --limit N --embed-device {cuda,cpu}`, `LocalEmbedder` with GPU auto-detect + batch ingestion, per-question progress logging.
  - Runbook at `docs/RUNBOOK_LONGMEMEVAL.md` documents the exact reproduction recipe end-to-end including cost estimates per provider combination.
- **Stage 7 — Procedural memory** (targets `v0.2.0`):
  - **New first-class memory item: `Procedure { situation, action, outcome, weight, metadata }`.** Procedures are how the agent learns from doing -- a SUCCESS reinforces the procedure's weight (it surfaces more next time), a FAILURE contradicts it (the agent stops reaching for the failed pattern), and PARTIAL is treated as a positive observation with the same reinforce signal. `Outcome` is a closed enum (`success` / `partial` / `failure` / `unknown`); UNKNOWN is the default at record time and the outcome-feedback loop flips it later.
  - **`Memory.record_procedure(situation, action, outcome=UNKNOWN, metadata=)`** embeds the situation, atomically inserts procedure + embedding, and fires the right decay signal at insert time. Returns the persisted `Procedure`.
  - **`Memory.retrieve_procedures(situation, k=5, *, outcomes=, include_cold=, reinforce=True)`** finds analogous past procedures ranked by `similarity * weight * outcome_boost`. Successes outrank failures at equal similarity but failures stay visible -- the agent can learn "this didn't work" too. `outcomes=` filters at the index level. Reinforcement-on-use fires for each surfaced procedure unless `reinforce=False`.
  - **`Memory.update_outcome(procedure_id, outcome)`** flips outcome + routes the change through decay. Returns the refetched `Procedure` with the bumped `updated_at`. Raises `KeyError` if the id doesn't exist.
  - **Schema additions:** `Outcome` enum, `Procedure` model (mutable -- outcome transitions), `ProcedureMatch` frozen result type. `ItemKind.PROCEDURE` joins the kind taxonomy so the existing embedding storage and decay-state surface accept procedures without special casing.
  - **Migration 0003** (`storage/migrations/0003_procedures.sql`): adds the `procedures` table with the full decay-column suite from migration 0002, indexes on `created_at` / `weight` / `outcome` / `cold_at`, and widens the `embeddings.item_kind` CHECK to allow `'procedure'`. SQLite's table-rebuild pattern preserves existing event/memory_item embedding rows; UNIQUE(item_id, item_kind, model) survives the rebuild.
  - **Storage protocol** grows `insert_procedure`, `get_procedure`, `list_procedures(outcome=, limit=)`, `update_procedure_outcome`, `count_procedures`, `count_procedures_by_outcome`, `search_procedure_embeddings(query_vec, k, *, outcomes=, include_cold=)`. The vector index gains a third shard kind ("procedure") whose "level" slot is the outcome string, so outcome-filtered search works through the existing `levels=` plumbing.
  - **Decay engine** picks up procedures via `_DEFAULT_KINDS = (EVENT, MEMORY_ITEM, PROCEDURE)`. Every existing per-kind decay-state method (`get_decay_state` / `update_decay_state` / `iter_decay_states` / `mark_cold` / `unmark_cold` / `count_cold` / `delete_cold_items` / `decay_totals`) works for procedures with zero additional code; the per-kind SQL templates regenerate automatically.
  - **Procedural transfer benchmark** at `benchmarks/suites/procedural_transfer.py`. Synthetic exact-match split (5 training patterns + 5 held-out queries) demonstrates the API contract: Engram agent scores 1.0 vs random-action baseline 0.28 = **+72 percentage-point lift** with FakeEmbedder, well above the Stage 7 DoD's 15-point bar. A `paraphrase_mode=True` variant of the same suite plants paraphrased held-out queries for use with real semantic embedders (`--embedder local` / `--embedder openai`).
  - **Tests:** 11 new schema tests, 9 new migration tests, 17 new storage tests, 20 new Memory-level tests. The end-to-end outcome-feedback loop (success and failure compete on the same situation; success outranks after repeated retrievals) is pinned. **597 tests total pass** after Stage 7 lands.
  - **Deferred to v0.2.1**: framework integration tests (LangGraph, LlamaIndex, raw OpenAI/Anthropic). The Stage 7 DoD calls for one per framework with a CI integration test; this needs real agent loops + extra CI surface and ships as a follow-up release.
- **Stage 8 — Contradiction & temporal reasoning** (targets `v0.3.0`):
  - **Public surface: `Memory.reconcile(conflict_id, *, resolution, manual_winner_id=None, now=None)`.** Resolves a detected `Conflict` per a chosen `Resolution` policy: `PREFER_RECENT` (later `created_at` wins), `PREFER_TRUSTED` (higher `source_trust` wins, None treated as 0.0; ties fall back to recent), `PREFER_FREQUENT` (higher corroboration count from decay state wins; ties fall back to recent), `KEEP_BOTH` (no winner, both stay valid), and `MANUAL` (caller picks the winner). The loser gets `invalidate_memory_item`d with the winner's id and the resolution timestamp; the conflict row flips OPEN -> RESOLVED with the resolution/winner/resolved_at fields filled in for audit.
  - **`Memory.list_conflicts(*, status=, memory_item_id=, limit=)`** is the matching read surface; `memory_item_id` walks the conflict graph in both directions (a memory item can be source or target of a conflict; callers don't usually care).
  - **Temporal-aware retrieve: `Memory.retrieve(..., as_of=None)`.** `as_of=None` is the new default and excludes items invalidated by `Memory.reconcile`; `as_of=<datetime>` returns historically-correct state -- items whose validity window covers the timestamp AND whose invalidation (if any) happened after it. `RetrieveParams.as_of` is wired through `HierarchicalRetriever` via the new `storage.search_memory_item_embeddings_as_of` path.
  - **Schema additions:** `Source { name, trust }`, `Verdict` (moved here from consolidation; AGREE / CONTRADICT / UNRELATED), `Resolution` (the five policies above), `ConflictStatus` (OPEN / RESOLVED), `Conflict` model with full status-machine invariants (winner must be source or target, KEEP_BOTH is the only resolution that doesn't need a winner, double-resolve raises). `MemoryItem` grows `valid_from`, `valid_until`, `invalidated_at`, `invalidated_by`, `source_trust` columns plus a model validator that defaults `valid_from = created_at` and enforces `valid_until >= valid_from`.
  - **The previously-internal `consolidation._contradiction.Conflict` is renamed `DetectedConflict`** -- the new persistent storage entity owns the bare name now. `DetectedConflict` stays the transient detector-output dataclass.
  - **Migration 0004** (`storage/migrations/0004_temporal_conflicts.sql`): creates the `conflicts` table with FK cascades to `memory_items` on source and target, full CHECKs on `verdict`/`status`/`resolution`, a CHECK that source != target, and UNIQUE(source_item_id, target_item_id). Adds the five new columns to `memory_items` and backfills `valid_from = created_at` on existing rows. Partial indexes on `valid_until`/`invalidated_at`/`source_trust` keep the NULL-dominant common case cheap.
  - **Storage protocol** grows `record_conflict`, `get_conflict`, `list_conflicts`, `resolve_conflict(id, *, resolution, resolved_winner_id, resolved_at)` (atomic OPEN -> RESOLVED with invariant validation), `count_conflicts`, `count_conflicts_by_status`, `invalidate_memory_item(id, *, at, by=)` (idempotent -- first timestamp wins), `set_validity_window(id, *, valid_from=, valid_until=)`, `set_source_trust`, and `search_memory_item_embeddings_as_of(query_vec, *, k, model, as_of=, levels=, exclude_ids=, include_cold=, candidate_multiplier=4)`. The temporal search over-fetches by `candidate_multiplier` from the in-memory vector index, then SQL-filters the candidates by validity.
  - **Reconciler engine** (`engram.reconcile.Reconciler`): policy implementations + tie-break rules (recency tie -> lex id compare for determinism), loser-invalidation side effect, idempotency guard on already-resolved conflicts.
  - **Consolidation integration:** when contradiction detection is enabled and the LLM judge returns CONTRADICT, the engine now writes a first-class `Conflict` row (status=OPEN) alongside the legacy metadata blob. The metadata path stays for back-compat -- existing readers (the promotion gate, audit traces, legacy callers) keep working.
  - **Adversarial benchmark suite** at `benchmarks/suites/contradiction_temporal.py`. Synthetic contradiction split (10 contradicting fact-pairs) scores **+100 percentage-point lift** -- Engram returns only the survivor post-reconcile (1.0); the no-reconcile baseline returns both items (0.0). Synthetic temporal split (5 three-version chains, 15 query snapshots) scores **100% accuracy** -- `as_of=t` returns the right version at each snapshot. The DoD ("contradicting events do not silently overwrite; the conflict is observable and resolvable" + "temporal queries return historically-correct state") is satisfied.
  - **Tests:** 26 schema tests, 12 migration-0004 tests, 32 storage CRUD tests, 17 reconciler tests, 8 end-to-end Memory.retrieve temporal tests, 2 suite-smoke tests. The consolidation engine integration test now also asserts the storage row is recorded and the graph is walkable from either side. **674 tests total pass** after Stage 8 lands.
  - **Deferred to v0.3.1**: `Resolution.MERGE` (LLM-merged content) -- needs a chat call per merge, not in the DoD; the surface stays additive when it lands. **LoCoMo temporal split real numbers** -- the harness scaffold exists, real-LLM scores depend on a paid run.
- **v0.3.1 — Polish & MERGE** (post-Stage 8 hardening):
  - **`Resolution.MERGE`** — the reconciler now synthesizes a new memory item via the chat provider when callers ask for `MERGE`. Both originals are invalidated pointing to the merged item; the conflict resolves with `resolved_winner_id=None` (the merged-into id is reachable via either parent's `invalidated_by`). Prompt at `reconcile/prompts/merge_v1.txt` is hardened the same way as `abstract_v1.txt`/`judge_v1.txt`: payloads inlined, JSON-only output, parse failure falls back to statement-b verbatim per the prompt's guidance. **Migration 0005** widens the `conflicts.resolution` CHECK to accept `'merge'` by rebuilding the table.
  - **Judge prompt-injection corpus** (`tests/test_consolidation_judge_prompt_injection.py`) — 17 tests covering prompt rendering (CRITICAL RULES precedes STATEMENTS), parse-time rejection (malformed JSON, missing/invalid verdict, case mismatch), the `judge(...)` wrapper's UNRELATED fallback on retry exhaustion, and end-to-end CORPUS payload sweep as A or B. Closes the Stage 5 standards gap that mandated a corpus per LLM-facing prompt.
  - **Hypothesis property tests** (`tests/test_stage8_properties.py`) — 12 properties × 50 examples each (~600 randomized cases) covering: visibility predicate matches SQL bit-for-bit, invalidation idempotency under N retries, winner-is-source-or-target invariant under random policy/trust/corroboration inputs, recency tie-break determinism via lex id compare, KEEP_BOTH produces no winner.
  - **Stage 8 perf budgets** (`tests/test_stage8_perf.py`) — `@pytest.mark.slow` budgets for `search_memory_item_embeddings_as_of` (P50 < 225 ms @ 100k items, 1% invalidated), `Memory.reconcile` (P50 < 25 ms, 3 storage round-trips), and `Memory.list_conflicts` (P50 < 10 ms @ 5k pairs).
  - **Coverage gap-fill** — `engram.schemas` 100%, `engram.reconcile._engine` 97%, `engram.reconcile._merge` 100%, `engram.memory` 99%, `engram.storage.sqlite` 97%. All Stage 8 modules above the 90% cross-cutting bar.
- **v0.2.1 — Framework integrations** (Stage 7 deferred item):
  - **`engram.integrations.format_context(results, ...)`** — framework-agnostic prompt-context formatter that turns RetrievalResult or ProcedureMatch sequences into bulleted strings.
  - **`engram.integrations.EngramAgent(memory, chat, ...)`** — opinionated agent wrapper. `agent.chat(message)` retrieves relevant memories, prepends them as system context, calls the chat provider, optionally auto-observes the user turn + reply. Returns an `EngramAgentTurn` documenting the full trace.
  - **`engram.integrations.langgraph`** — `EngramRetrieveNode` + `EngramObserveNode`. Callable `StateGraph` nodes; configurable state keys; framework deps imported lazily. New `[langgraph]` extra.
  - **`engram.integrations.llamaindex`** — `EngramLlamaIndexMemory` duck-types LlamaIndex's `BaseMemory` shape (put/get/get_all/reset). `get(input=q)` returns a single system-role pseudo-message; `as_chat_message()` converts to a real LlamaIndex `ChatMessage`. New `[llamaindex]` extra.
  - **Tests**: 28 integration tests covering format_context (6), EngramAgent (12), LangGraph nodes (7) including a full StateGraph end-to-end, LlamaIndex adapter (9) including pseudo-message conversion.
- **Stage 9a — Production layer (partial)** (targets `v0.4.0`; Postgres backend + full multi-tenant read-side defer to v0.4.0 proper):
  - **Async surface on `Memory`**: every public sync method has an `async def` parallel (`aobserve`, `aretrieve`, `aconsolidate`, `areconcile`, `arecord_procedure`, `aretrieve_procedures`, `aupdate_outcome`, `areinforce`, `acorroborate`, `acontradict`, `apromote`, `alist_conflicts`). All route through `asyncio.to_thread` so the SQLite per-thread connection model continues to apply. The async-native Postgres path lands in v0.4.0 alongside the backend.
  - **OpenTelemetry instrumentation** (`engram._otel`): lazy wrapper around `opentelemetry-api`. With the `[otel]` extra and a configured TracerProvider, every public `Memory` call emits spans + counters with stable attributes (`k`, `prefer`, `resolution`, etc). Without the extra, all calls are no-ops (zero cost). Histograms: `engram.retrieve.latency_ms`. Counters: observe/retrieve/consolidate/reconcile calls. 6 instrumentation tests use an in-memory span exporter.
  - **Multi-tenant `tenant_id` write-side**: `Event`, `MemoryItem`, `Procedure` all gain optional `tenant_id: str | None`. **Migration 0006** adds the column to each table with partial indexes (`WHERE tenant_id IS NOT NULL`). `Memory(..., tenant_id="acme")` injects on writes; caller-explicit overrides are honored (escape hatch for cross-tenant admin tools). Read-side enforcement (filtering `retrieve` by tenant) deferred to v0.4.0 alongside Postgres + RLS where it's an actual security boundary; for the single-process SQLite library, filter-based isolation isn't a real boundary anyway.
  - **Async memory tests**: 11 tests via `asyncio.run`. Uses a tempfile `file_storage` fixture because cross-thread SQLite `:memory:` databases are per-connection and don't share schema.
  - **Multi-tenant tests**: 17 tests covering schema round-trip, write-side injection, caller-explicit override, untenanted Memory leaves tenant_id NULL, cross-tenant writes don't collide, migration 0006 upgrade preserves rows.
- **Stage 10 — Docs site + reproduction tooling (no paper)**:
  - **`mkdocs.yml` + `docs/`** scaffold using `mkdocs-material` + `mkdocstrings[python]` for auto-generated API reference from docstrings. Sections: Getting started, Guides (observe/retrieve, consolidation, decay, procedural, contradiction/temporal, integrations), API reference (Memory, schemas, storage, reconcile, integrations), Operations (observability, multi-tenant, performance), Project (api-stability, deprecation-policy, security). New `[docs]` extra brings `mkdocs`/`mkdocs-material`/`mkdocstrings[python]`.
  - **`scripts/reproduce_benchmarks.py`** — one-shot CLI that runs every suite against `FakeProvider` and emits manifests to `benchmarks/runs/ci/<timestamp>/`. CI on tagged release uses this to re-emit reproducibility receipts. `--only` flag selects a subset.
  - **API stability + deprecation policy docs** at `docs/project/api-stability.md` + `docs/project/deprecation-policy.md`. Documents what's stable in v0.3.x, what's experimental, the deprecation cycle for pre- and post-v1.0 releases, and the breaking-vs-additive change rules.

[Unreleased]: https://github.com/AmeyaBorkar/Engram/compare/HEAD...HEAD
