# Changelog

All notable changes to Engram are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Once we ship `v1.0.0`, the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html); pre-1.0 releases may break the public API on minor version bumps.

## [Unreleased]

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

[Unreleased]: https://github.com/AmeyaBorkar/Engram/compare/HEAD...HEAD
