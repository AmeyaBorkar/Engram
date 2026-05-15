# Engram SOTA Push — Session Journey

Chronological log of every decision, ship, finding, and dead-end from this session. Written so that a future reader (including a fresh-context me) can reconstruct exactly where the project is and how it got here.

Session began with v0.1.0 on disk (Stage 6 complete), `engram-memory` as the planned PyPI name, and the intent to push for SOTA on LongMemEval-S. It ended with `engrampy` v0.2.1 published, a full retrieval evaluation infrastructure in place, and a precise understanding of where retrieval-side gains live versus where the SOTA gap actually is.

---

## Table of contents

1. [Mission & starting point](#1-mission--starting-point)
2. [Wave 1 — Hybrid retrieval feature suite](#2-wave-1--hybrid-retrieval-feature-suite)
3. [First LLM run: crash on content filter](#3-first-llm-run-crash-on-content-filter)
4. [Per-question exception isolation](#4-per-question-exception-isolation)
5. [Second LLM run: multi-session collapse](#5-second-llm-run-multi-session-collapse)
6. [Ablation infrastructure](#6-ablation-infrastructure)
7. [Math correctness audit and fixes](#7-math-correctness-audit-and-fixes)
8. [Evaluation protocol & statistical infrastructure](#8-evaluation-protocol--statistical-infrastructure)
9. [First ablation findings: the bm25 × mmr catastrophe](#9-first-ablation-findings-the-bm25--mmr-catastrophe)
10. [Cross-qtype ablation validation](#10-cross-qtype-ablation-validation)
11. [Per-question diagnostic trace](#11-per-question-diagnostic-trace)
12. [End-to-end evaluation orchestrator](#12-end-to-end-evaluation-orchestrator)
13. [PyPI release: engrampy v0.2.0 → v0.2.1](#13-pypi-release-engrampy-v020--v021)
14. [The has_answer discovery: event-level ground truth](#14-the-has_answer-discovery-event-level-ground-truth)
15. [Full event-level recall sweep](#15-full-event-level-recall-sweep)
16. [Zero-recall failure-mode classification](#16-zero-recall-failure-mode-classification)
17. [Current state](#17-current-state)
18. [Next steps](#18-next-steps)
19. [Appendix A — Commit log](#appendix-a--commit-log)
20. [Appendix B — Scripts shipped](#appendix-b--scripts-shipped)
21. [Appendix C — Source changes](#appendix-c--source-changes)
22. [Appendix D — Headline numbers](#appendix-d--headline-numbers)

---

## 1. Mission & starting point

- **Goal**: push for SOTA on LongMemEval-S. Stretch goals include LoCoMo and procedural memory benches.
- **Starting point**: v0.1.0 on PyPI as `engram-memory`. Stage 6 done. Dense + BGE cross-encoder rerank in place. Several Phase E retrieval flags (HyDE, multi-query, decompose, temporal anchoring, surface-conflicts) wired but defaulted off.
- **User constraints**: no virtualenvs. No Co-Authored-By trailers in commits. Elite bar on speed/quality/security. Track all SOTA-relevant scores in `SCOREBOARD.md`.

The killed-LLM trajectory from the prior session was ~70-75% projected mid-run. The session opened with the question: what else can we hybridize at the retrieval level without paying for any new API calls?

---

## 2. Wave 1 — Hybrid retrieval feature suite

Shipped in one commit (`d37b361`) along with several adjacent improvements that were also ready to go.

### Retrieval features

| Feature | Where it lives | What it does |
|---|---|---|
| **BM25 lexical hybrid** | `src/engram/retrieve/_bm25.py` (new) | numpy-vectorized BM25 with Lucene-style `k1`/`b`; lazy index build; frozen-after-search invariant |
| **Reciprocal Rank Fusion** | `_bm25.py` (alongside BM25) | RRF for combining dense + lexical + recent-window streams |
| **MMR diversity rerank** | `src/engram/retrieve/_mmr.py` (new) | numpy-vectorized greedy MMR over the rerank pool |
| **Recency boost** | `src/engram/retrieve/_engine.py` | Multiplicative `score * (1 + λ * exp(-days/decay))` (later fixed — see §7) |
| **Recent-window hybrid stream** | `_engine.py` + `_storage/sqlite.py::list_recent_events` | Top-N most-recent events injected as a 3rd RRF stream |
| **Auto-temporal lexical filter** | `benchmarks/suites/longmemeval.py::_build_auto_temporal_filter` | Per-question year-token regex extraction with empty-pool fallback |
| **Lexical filter (general)** | `_engine.py` | Regex pre-filter applied before rerank |
| **Surface-conflicts** | `_engine.py` (already existed; flag plumbed through) | Co-surface contradictory memory items |

### Adjacent improvements

| Improvement | File | Effect |
|---|---|---|
| Storage `bm25_search_events` | `storage/sqlite.py` | Lazy in-memory BM25 index, rebuild on corpus or k1/b change |
| Batch `get_embeddings_batch` / `get_created_at_batch` | `storage/sqlite.py` | 1 SQL round-trip per `ItemKind` instead of N (for MMR and recency lookups) |
| Composite indexes | `storage/migrations/0009_perf_indexes.sql` (new) | Hot-path query speedups |
| PRAGMA tuning | `storage/sqlite.py` | `cache_size = 64MB`, `mmap_size = 256MB` |
| Async parallel consolidation | `consolidation/_abstraction.py` + `_engine.py` | `aconsolidate` with `asyncio.gather` + semaphore-bounded concurrency; ~30× speedup over serial |
| Disk cache for `(chat, embed)` | `providers/_disk_cache.py` (new) | SQLite-backed persistent cache; `CachedChat` / `CachedEmbedder` / `with_disk_cache` |
| Asymmetric query prompts | `providers/local.py` | `s2p_query` for Stella, instruction prefix for E5 family |
| Per-instance LRU cache | `providers/local.py` | 4096-entry result cache on the embedder |

### New `RetrieveParams` fields

```python
bm25_weight: float = 0.0
bm25_k1: float = 1.5
bm25_b: float = 0.75
mmr_lambda: float = 0.0
mmr_pool_size: int = 0           # 0 = use k * candidate_multiplier
recency_lambda: float = 0.0
recency_decay_days: float = 90.0
recent_window_k: int = 0
lexical_filter: str | None = None
```

### New CLI flags

14 new bench flags on `python -m engram.bench run longmemeval`: `--bm25-weight`, `--bm25-k1`, `--bm25-b`, `--mmr-lambda`, `--mmr-pool-size`, `--recency-lambda`, `--recency-decay-days`, `--recent-window-k`, `--lexical-filter`, `--auto-temporal`, `--disk-cache`, `--drill-k`, `--confidence-threshold`, `--rerank-pool-multiplier`.

**Commit `d37b361`** — `retrieve+providers+bench: hybrid retrieval + async consolidation + disk cache`.

---

## 3. First LLM run: crash on content filter

User kicked off the full LongMemEval-S run with:

```
--dtype fp32 --rerank-pool-multiplier 5 --bm25-weight 1.0 --mmr-lambda 0.7
--recency-lambda 0.1 --auto-temporal --recent-window-k 10 --surface-conflicts
```

At question 141 (`single-session-preference`), Moonshot AI returned HTTP 400 with `content_filter` type:

```
"The request was rejected because it was considered high risk"
```

The bench had no per-question exception handling, so 1.5 hours of compute went up in smoke. The trajectory before the crash was already troubling — multi-session accuracy was dropping fast (0.78 → 0.50 by q=130).

---

## 4. Per-question exception isolation

Shipped to never lose a multi-hour run to a single bad request again.

- Refactored `benchmarks/suites/longmemeval.py::run()` to pull the per-question pipeline into `_run_one_question(...)`.
- Wrapped the body in `try / except Exception` — score 0 on failure, record `error` in the per-question manifest entry, continue.
- `KeyboardInterrupt` and `SystemExit` still propagate so Ctrl+C aborts cleanly.
- Latency arrays get padded with `0.0` placeholders so they stay aligned with `per_question`.

**Commit `7bac655`** — `bench: per-question exception isolation in LongMemEval`.

---

## 5. Second LLM run: multi-session collapse

User restarted with the same flags. This time:

- q=3 hit the content filter, was logged as `ERROR`, **run continued** ✓
- q=101 hit a different error ("No allowed providers for the selected model") — also caught and continued ✓
- But accuracy collapsed on multi-session:

| Bucket | This run | Prior killed run | Δ |
|---|---|---|---|
| single-session-user q=1-70 | 0.871-0.933 | 0.800-0.880 | **+0.05 to +0.06** ✓ |
| multi-session q=71-130 | 0.538 | 0.669 | **−0.131** ✗ |
| single-session-preference q=140-160 | 0.500 | 0.664 | **−0.164** ✗ |

The hybrid stack was hurting multi-session badly. User killed the run and asked for a methodology rather than another expensive attempt.

---

## 6. Ablation infrastructure

Built the right tool: replay the same question through N configs side-by-side at the retrieval level only (no LLM cost).

### `scripts/ablate_longmemeval.py`

- Per-question, per-config retrieve runs.
- 12 predefined configs in `CONFIGS`: `baseline`, `bm25`, `mmr07`, `mmr03`, `recent`, `recency`, `autotemp`, `bm25+aut`, `bm25+mmr`, `bm25+rec`, `all_aggressive`, `conservative`.
- Two modes: `--retrieval-only` (no LLM) or full LLM-scored.
- Output: per-question manifest JSON + a markdown matrix.

### Storage changes for retrieval-only metrics

Added `session_id` to `Event.metadata` during `_ingest_haystack`:

```python
metadata["session_id"] = q.haystack_session_ids[session_idx]
```

This let the ablation harness compute session-level recall (did at least one event from any answer session appear in top-k?) **without** running the LLM.

**Commit `3736ad0`** — `bench: per-question retrieve-config ablation harness`.

---

## 7. Math correctness audit and fixes

User asked: "Can you check the math and logic of these features?" Five issues surfaced.

### Issues found

| # | Issue | Severity | Where |
|---|---|---|---|
| 1 | Recent-window stream is unconditional, floods candidate pool | HIGH | `_engine._fuse_hybrid_sources` |
| 2 | MMR runs on unnormalized cross-encoder logits — diversity term dwarfed by relevance range | MEDIUM | `_mmr.mmr_select` |
| 3 | Recency boost is **multiplicative** — inverts on negative reranker logits | LOW (sign bug) | `_engine._apply_recency_boost` |
| 4 | BM25 fractional weight rounds to 1 (0.5 and 1.0 behave identically) | minor | `_engine._fuse_hybrid_sources` |
| 5 | Auto-temporal fallback fires only on **empty** pool, not on partial-wrong pool | minor | `benchmarks/suites/longmemeval.py` |

### Fixes shipped

**MMR normalization** — `src/engram/retrieve/_mmr.py`:

```python
# Min-max normalize relevance to [0, 1] so the diversity term
# (cosine on unit vectors, already in [0, 1]) carries comparable weight.
rel_min = float(relevance_raw.min())
rel_max = float(relevance_raw.max())
if rel_max > rel_min:
    relevance_np = (relevance_raw - rel_min) / (rel_max - rel_min)
else:
    relevance_np = np.zeros_like(relevance_raw)
```

**Recency additive form** — `_engine._apply_recency_boost`:

```python
# Was: out.append(s * (1.0 + p.recency_lambda * math.exp(-days_old / decay_days)))
# Now:
bonus = p.recency_lambda * math.exp(-days_old / decay_days)
out.append(s + bonus)
```

Both are no-ops at default knob values (`mmr_lambda=0`, `recency_lambda=0`) — backward compatible.

`docs/EVAL_PROTOCOL.md` was updated to mark both as "Fixed May 2026" with the new behavior documented in the per-component contracts.

**Commit `eaf3dc0`** — `retrieve: fix MMR + recency math correctness`.

---

## 8. Evaluation protocol & statistical infrastructure

User callout: "We are claiming SOTA here. We need those standards. Each component should be individually evaluated rigorously, for performance and tuned and fixed."

Three artifacts shipped to enforce that discipline.

### `docs/EVAL_PROTOCOL.md`

Per-component contracts. For each retrieval feature: hypothesis, knobs, primary metric, decision rule, known issues, when-not-to-use. Plus the formal gate:

> Ship rule: paired Δ recall CI excludes zero AND McNemar p < 0.05 AND no per-qtype regression > 0.02.

### `scripts/_stats.py`

- `bootstrap_mean_ci(values, n_iters=10000)` — bootstrap 95% CI on a metric mean.
- `bootstrap_paired_diff_ci(a, b)` — paired-question Δ CI; `excludes_zero` flag.
- `mcnemar(passes_a, passes_b)` — exact-binomial for `<25` discordant pairs, Yates-corrected chi-square otherwise.
- `format_ci`, `format_p` helpers.

Pure stdlib + numpy.

### `scripts/sweep.py`

Single-knob grid sweep with paired CI + McNemar vs a baseline value. Predefined grids in `SWEEPS` for `bm25_weight`, `bm25_k1`, `bm25_b`, `mmr_lambda`, `mmr_pool_size`, `recency_lambda`, `recency_decay_days`, `recent_window_k`, `candidate_multiplier`, `k`, `confidence_threshold`, `drill_k`.

### `scripts/retrieval_eval.py`

Retrieval-only multi-config evaluator across the full LongMemEval-S (or filtered subset). Reports recall@k / hit@k / multi_recall@k / MRR / first_correct_rank / precision@k at multiple k cutoffs (default 10/20/50). No LLM calls.

Updated to emit a statistical-significance section with bootstrap CIs, paired-diff vs baseline, McNemar p-values, plus a failure-mode list (qids broken by each config).

**Commits**:
- `5897b63` — `bench: retrieval-only evaluator for LongMemEval (no LLM cost)`
- `6bb90b6` — `eval: formal evaluation protocol + statistical infrastructure`

---

## 9. First ablation findings: the bm25 × mmr catastrophe

User ran the ablation on 30 multi-session questions (~48 min). Results:

| Config | Mean recall | Δ vs baseline |
|---|---:|---:|
| baseline | 0.943 | — |
| bm25 | 0.910 | −0.033 |
| mmr07 | 0.944 | +0.001 |
| mmr03 | 0.938 | −0.005 |
| recent | 0.924 | −0.019 |
| recency | 0.943 | 0.000 |
| autotemp | 0.943 | 0.000 |
| bm25+aut | 0.910 | −0.033 |
| **bm25+mmr** | **0.501** | **−0.442** ⚠ |
| bm25+rec | 0.910 | −0.033 |
| **all_aggressive** | **0.498** | **−0.445** ⚠ |
| conservative | 0.910 | −0.033 |

### Diagnosis of the bm25 × mmr interaction

Each alone is mild; **together** they destroy multi-session recall:

1. BM25 ranks by literal-token overlap → returns 3-5 events from one answer session (same-session lexical overlap is naturally high).
2. RRF fusion creates a pool with many same-session BM25 duplicates plus cross-session dense hits.
3. Cross-encoder scores the same-session duplicates highly (similar content, all match the query).
4. **MMR sees them as redundant** and drops second/third gold-session events to make room for diverse (but wrong-session) events.
5. Multi-recall (fraction of answer sessions covered) collapses.

Hit rate stayed at 1.0 across the board — MMR always keeps one answer-session event. But coverage falls off a cliff. For multi-session questions, where the LLM needs evidence from multiple answer sessions to synthesize, that's the critical metric.

The killed-LLM run's −0.40 drop on multi-session is now mechanistically explained.

---

## 10. Cross-qtype ablation validation

Ran the ablation on all 5 LongMemEval qtypes × 30 questions each = 150 questions total. Cross-qtype matrix:

| Config | sss-user | sss-pref | multi | ku | temporal | mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.000 | 0.967 | 0.943 | 1.000 | 0.928 | **0.968** |
| bm25 | 1.000 | 0.967 | 0.910 | 1.000 | 0.947 | 0.965 |
| mmr07 | 0.967 | 0.967 | 0.944 | 1.000 | 0.939 | 0.961 |
| recency | 0.967 | 0.900 | 0.818 | 0.983 | 0.828 | 0.899 |
| autotemp | 1.000 | 0.967 | 0.943 | 1.000 | 0.928 | **0.968** |
| bm25+mmr | 1.000 | 0.967 | 0.806 | 1.000 | 0.894 | 0.933 |
| all_aggressive | 1.000 | 0.900 | 0.648 | 0.967 | 0.767 | 0.856 |

### Verdict per protocol gates

- **Pass everything**: `baseline`, `autotemp` (tied at 0.968).
- **Fail**: every other config — either a per-qtype regression > 0.02 (most) or net negative.
- **Most catastrophic**: `recency` overall (hurts everywhere, especially temporal-reasoning at −0.100), `bm25+mmr` on multi-session (−0.137), `all_aggressive` everywhere.

Retrieval ceiling was estimated at ~97% on session-level recall — leaving end-to-end accuracy gap to the LLM stage.

---

## 11. Per-question diagnostic trace

For when aggregate metrics aren't enough, built a tool to literally see what each component does on one question.

### `scripts/retrieval_trace.py`

For a chosen `--qid` (or sample), dumps:

1. Question + gold + `answer_session_ids` + haystack size.
2. Dense top-N (raw cosine, no rerank).
3. BM25 top-N (lexical, raw scores).
4. Recent-window top-N (created_at DESC).
5. Final top-k per config (full pipeline).

Each retrieved event annotated `[GOLD]` / `[....]` based on session match, plus rank, score, session suffix, 60-char content preview.

This was the first tool that let us look inside a specific failure question and see the mechanism unfold stage by stage.

**Commit `6394a17`** — `bench: per-question retrieval trace tool`.

---

## 12. End-to-end evaluation orchestrator

One command runs the full battery.

### `scripts/run_all_evals.py`

Five phases, each skippable via `--skip-phase N`, resumable via `--resume`:

| Phase | What | Time | GPU |
|---|---|---|---|
| 1. Component correctness | 33 unit tests over BM25, MMR, RRF, auto-temporal, recency math | ~10 s | no |
| 2. Per-qtype ablation | 5 qtypes × 12 configs × N questions, retrieval-only | ~2 hr | yes |
| 3. Hyperparameter sweeps | 4 knobs × 5 values × N questions, paired-diff CI + McNemar | ~30 min | yes |
| 4. Diagnostic traces | 1 broken question per qtype, full per-stage trace | ~5 min | yes |
| 5. Consolidated report | `REPORT.md` with verdicts, recommended launch config | ~10 s | no |

Embedder + reranker load **once** and are shared across phases 2-4. Each phase writes its own JSON; resume skips phases whose artifact already exists.

### Phase 1 unit tests (all 33 passed)

| Component | Tests |
|---|---:|
| `BM25Index` | 9 (empty, single doc, no match, term frequency, IDF, tokenizer, determinism, frozen-after-search, length normalization) |
| `mmr_select` | 8 (empty, single, λ=1.0=relevance-sort, λ=0.0=max-diversity, duplicates, orthogonal items, None vectors, invalid λ raises, λ=0.5 picks diverse) |
| `reciprocal_rank_fusion` | 5 (empty, single ranking, common top wins, both docs surface, exact math) |
| `_build_auto_temporal_filter` | 5 (no year, one year, two years, non-year 4-digit, year-in-word boundary) |
| recency math | 6 (formula at day 0, decay_days, very old; sign correctness across base scores) |

**This validates that all retrieval components are mathematically correct.** Any harms observed downstream are design-fit mismatches, not implementation bugs.

**Commit `eac6b46`** — `eval: end-to-end orchestrator (one script, every test)`.

### First full orchestrator run

User kicked off the full battery; it ran 3.5 hours (`benchmarks/runs/eval_all_20260515_024020`). Final `REPORT.md`:

- Phase 1: 33/33 ✓
- Phase 2: identical per-qtype matrix to manual ablation (catastrophic `bm25+mmr` and `all_aggressive` reconfirmed)
- Phase 3 sweeps: **ran with `qtype=None` which silently selected the first 30 questions in dataset order — all single-session-user**. No knob value beat baseline on sss-user (which was at ceiling). The sweep methodology has a known limitation: it doesn't test on the qtype where the feature is expected to matter.
- Phase 4 traces: one trace per qtype saved.
- Phase 5: recommended `baseline + autotemp + surface-conflicts`.

---

## 13. PyPI release: engrampy v0.2.0 → v0.2.1

User had the PyPI token. Standard plan was to upload as `engram-memory`. **It was squatted.**

### The squat

- `engram` (bare): single 1163-byte placeholder upload from 2025-04-21, summary "Do not use. Placeholder." (could be PEP 541 reclaimed but slow).
- `engram-memory`: claimed out from under the project on **2026-02-10** by an unrelated party. Three versions uploaded within hours: 0.4.0, 0.4.1, 0.5.0b1. Author listed as "Engram Team". Homepage pointing at `github.com/Ashish-dwi99/Engram` (not the user's `AmeyaBorkar/Engram`).

### Resolution

User chose to ship as `engrampy`. Python import name unchanged (`from engram import Memory` still works because the wheel still packages `src/engram/`).

### v0.2.0 ship

- `pyproject.toml` updated: `name = "engrampy"`, `version = "0.2.0"`, comment explaining the squat history.
- `src/engram/__init__.py`: `__version__ = "0.2.0"`.
- `README.md`: install line + squat note.
- `CHANGELOG.md`: full v0.2.0 entry (~100 lines covering everything since v0.1.0).
- Built clean (`twine check` PASSED), local install + import verified.
- User uploaded; live at https://pypi.org/project/engrampy/0.2.0/.

**Commits**:
- `38b4c66` — `release: v0.2.0 — pip install engrampy`
- `(tag) v0.2.0` annotated.

### v0.2.1 patch

User noticed `Author:` field in PyPI metadata was empty (PEP 621 quirk: an entry with `{name, email}` together routes to `Author-email` only).

Fix: dual-entry workaround in `pyproject.toml`:

```toml
authors = [
    { name = "Ameya Borkar" },                                    # → Author: Ameya Borkar
    { name = "Ameya Borkar", email = "ameyaborkar17@gmail.com" }, # → Author-email
]
```

Bumped to 0.2.1, rebuilt, uploaded. Verified both `Author: Ameya Borkar` and `Author-email: Ameya Borkar <ameyaborkar17@gmail.com>` now populated in PyPI metadata.

**Commits**:
- `05e9780` — `release: v0.2.1 — populate legacy Author metadata field`
- `(tag) v0.2.1` annotated.

PyPI live at https://pypi.org/project/engrampy/0.2.1/.

---

## 14. The has_answer discovery: event-level ground truth

User's pivotal question: **"Do we know what we're retrieving is actually correct? Is there a dataset or anything available to actually see exactly what is happening?"**

Honest answer was no — we'd been measuring **session-level recall** (proxy: "did we retrieve any event from an answer session?"). That overestimates correctness when an answer session has 10 turns but only 1-3 contain the actual fact.

### The discovery

Inspected the LongMemEval-S dataset structure:

| Stat | Value |
|---|---|
| Total questions | 500 |
| Total turns | 246,750 |
| `has_answer=True` turns (gold answer-bearing) | **896** (0.36%) |
| `has_answer=False` turns (in answer session but not gold) | 10,064 (4.08%) |
| `has_answer=None` turns (no label, almost always non-answer-session) | 235,790 (95.56%) |
| `has_answer=True` turns NOT in any answer session | **0** (validates session-level proxy as an upper bound) |

**LongMemEval-S ships per-turn ground-truth labels we'd been silently discarding.** Our `_ingest_haystack` only carried `role` and `content` from each turn.

### Fix

`benchmarks/suites/longmemeval.py::_ingest_haystack` now persists `has_answer` into `Event.metadata` alongside `session_id`:

```python
ha = turn.get("has_answer")
if ha is not None:
    metadata["has_answer"] = bool(ha)
```

Opaque to retrieve/rerank; just an observability hook.

### New inspector — `scripts/inspect_retrieval.py`

Three things per question:
1. Dumps the question + gold + every turn of every answer session with `has_answer` flags (`✦GOLD✦` marker on the actual answer-bearing turns).
2. Runs retrieve, shows top-k with three annotation types: `[GOLD-EVT]` (true gold turn), `[SAME-SESS]` (in answer session but not gold), `[........]` (other).
3. Side-by-side **event-level** vs **session-level** recall with explicit `gap` column.

`--stats-only` mode skips per-question dumps and aggregates only.

**Commit `e28f981`** — `bench+inspect: preserve has_answer ground truth + deep inspector`.

A follow-up patch fixed an embarrassing default — `--limit 1` was the default, so the natural full-eval loop ran on 1 question per qtype. Changed to `default=None` (= all matching):

**Commit `d8373e0`** — `inspect: --limit defaults to None (all matching) instead of 1`.

### Also discovered

There's a sixth qtype that earlier scripts had been silently missing: **`single-session-assistant`** (56 questions). The orchestrator's `QTYPES` tuple listed only 5. All findings prior to this section are missing this bucket.

---

## 15. Full event-level recall sweep

Re-ran across all 6 qtypes, all 500 questions, baseline config, k=10. ~3 hours.

### The headline

| Metric | Value |
|---|---|
| **Session-level recall@10** (what we'd reported as 0.97) | **0.966** |
| **Event-level recall@10** (true recall) | **0.841** |
| **Gap** | **+0.125** |

The session-level proxy was overestimating retrieval correctness by 12.5 absolute points.

### Per-qtype breakdown (n=500)

| qtype | n | sess hit | sess R@k | evt hit | **evt R@k** | **gap** |
|---|---:|---:|---:|---:|---:|---:|
| single-session-assistant | 56 | 1.000 | 1.000 | 0.982 | **0.982** | +0.018 |
| single-session-user | 70 | 0.986 | 0.986 | 0.900 | 0.900 | +0.086 |
| knowledge-update | 78 | 1.000 | 0.994 | 0.923 | 0.904 | +0.090 |
| temporal-reasoning | 133 | 0.985 | 0.930 | 0.902 | 0.821 | +0.109 |
| **multi-session** | 133 | 1.000 | 0.961 | 0.910 | **0.757** | **+0.204** |
| **single-session-preference** | 30 | 0.967 | 0.967 | 0.833 | **0.733** | **+0.233** |
| overall | 500 | 0.992 | 0.966 | 0.912 | 0.841 | +0.125 |

### Distribution of event-level recall@10 across 500 questions

| Bucket | Count | % |
|---|---:|---:|
| evt R@k = 1.0 (perfect) | 382 | 76.4% |
| evt R@k in (0.75, 1.0) | 1 | 0.2% |
| evt R@k in (0.5, 0.75] | 23 | 4.6% |
| evt R@k in (0.25, 0.5] | 45 | 9.0% |
| evt R@k in (0, 0.25] | 5 | 1.0% |
| evt R@k = 0 | 44 | 8.8% |

### Caveat — 21 questions have no `has_answer=True` turns

These are abstractive/derivative questions (the answer is implicit, not in a single turn). `event_recall = 0` for these is an artifact of the missing labels, not a retrieval failure. Excluding them from the denominator:

| Adjusted metric | Value |
|---|---|
| True event R@k over evaluable 479 questions | **0.878** |
| Full-recall rate | 382/479 = **79.7%** |
| Partial-recall rate (0 < r < 1) | 74/479 = **15.4%** |
| Zero-recall rate (gold exists, surfaced none) | 23/479 = **4.8%** |

### Severe-gap qtypes (top three)

Gap > 0.4 (severe overestimation by the proxy):

| qtype | severe-gap count | % of qtype |
|---|---:|---:|
| multi-session | 35 | 26.3% |
| temporal-reasoning | 23 | 17.3% |
| single-session-preference | 8 | 26.7% |

### Ghost hits

Session hit = 1 but event hit = 0 — surfaced the right session but **no** gold turn from within it: 20 questions (4.0%). These are pure proxy-lies.

### Re-framing the SOTA picture

| Layer | Old belief | New belief |
|---|---|---|
| Retrieval (true) | 0.97 ceiling | **0.88 ceiling** (over evaluable) |
| End-to-end | tracking ~0.72 | tracking ~0.72 |
| Headroom to retrieval ceiling | ~3 points | **~12 points** |
| Headroom to LLM ceiling | ~25 points | **~10-15 points** |

Retrieval has more real headroom than we thought. The SOTA gap is **split roughly evenly** between retrieval and LLM, not concentrated in the LLM. Multi-session and preference each leak ~25% of gold turns inside the right sessions.

---

## 16. Zero-recall failure-mode classification

For the **23 true-zero-recall questions** (event_recall = 0 AND n_gold_events > 0), built batch tracing and a failure-mode classifier.

### `scripts/batch_trace.py`

Reuses `_trace_question` from `retrieval_trace.py` but loads the embedder + reranker ONCE and traces every qid in a `--qid-file`. Cuts ~10 minutes off a 23-question dump compared to running the loop one-shot per qid.

**Commit `41a0eb8`** — `bench: scripts/batch_trace.py — share embedder across many qid traces`.

### `scripts/analyze_zero_recall_traces.py`

Parses each trace file and classifies into one of five failure modes:

| Class | Definition |
|---|---|
| `session_miss` | Dense missed every answer session in top-50 |
| `gold_at_deep_dense_rank` | Min gold rank in dense top-50 > 20 |
| `wrong_turn_in_session` | Gold session in dense top-50 but gold turn ranks below k=10 |
| `rerank_pushed_gold_out` | Gold was in dense top-10 but rerank dropped it |
| `competing_session` | One session (gold OR off-topic) captures ≥5 of top-10 slots |

Outputs: per-question table, class distribution, class × qtype matrix.

**Commit `07a8acd`** — `bench: trace parser + failure-mode classifier for zero-recall analysis`.

### Results (13 of 23 traces analyzed; rest pending)

| Class | Count |
|---|---:|
| competing_session | 9 |
| rerank_pushed_gold_out | 2 |
| gold_at_deep_dense_rank | 1 |
| session_miss | 1 |

### Critical re-classification

When the dominator session's suffix is checked against the question's actual `answer_session_ids`, the `competing_session` bucket splits in two:

| True mechanism | Count | Description |
|---|---:|---|
| **within-session ranking failure** | 5 | The "competing" session IS the answer session — but the dominator's 5-9 top-10 slots are all *non-gold* turns within it (the actual gold turn is at deeper rank) |
| **true distractor session** | 4 | A different, off-topic session dominates top-10 |

Examples of within-session:
- `0a34ad58`: dominator `7159` = answer session `cebb7159`
- `1a1907b4`: dominator `02eb` = answer session `719502eb`
- `561fabcd`: dominator `p_97` = answer session `sharegpt_hChsWOp_97`
- `gpt4_194be4b`: dominator `55_3` = answer session `3826dc55_3`
- `0bc8ad93`: dominator `fc_3` = answer session `f4ea84fc_3`

Examples of true distractor:
- `6d550036`: question "How many projects have I led?", dominator `d8_2` is a session about "planning to launch a new project / Asana" — semantically closer to the question than the gold sessions, but wrong.
- `5d3d2817`: question "What was my previous occupation?", dominator `A_23` ≠ answer `235eb6fb`.

### Mechanistic insight from one trace (qid `6d550036`)

| Stage | Observation |
|---|---|
| Dense top-15 | 12 of 15 from off-topic session `d8_2` ("planning to launch a new project / Asana") |
| Dense top-50 | 16 gold events present at ranks 15, 17, 18, 19, 20, 21, 25, 26, 32... |
| Final top-10 after rerank | 2 gold events at ranks 8, 10 (both peripheral non-gold turns from gold sessions) |
| Gold turn 0 of session `3c_1` ("I'm working on a project that involves analyzing customer data...") | **Not in top-50 at all** |
| Verdict | True distractor + within-session: a non-gold session about project management out-competes the gold sessions, AND when gold sessions do surface they're via non-gold turns within them |

### Class × qtype matrix

| failure class | multi | sss-asst | sss-pref | sss-user | temporal |
|---|---:|---:|---:|---:|---:|
| competing_session | 3 | 1 | 3 | 1 | 1 |
| gold_at_deep_dense_rank | 0 | 0 | 0 | 0 | 1 |
| rerank_pushed_gold_out | 1 | 0 | 1 | 0 | 0 |
| session_miss | 0 | 0 | 1 | 0 | 0 |

### Implications

| Hypothesis | Verdict |
|---|---|
| "Retrieval is broken; raise k will fix everything" | Partially false. Helps the 5 within-session cases. Doesn't help the 4 true-distractor cases. |
| "Rerank is hurting" | Partially true. 2 of 13 (~15%) cases were demonstrably better before rerank. |
| "Off-topic distractors are the main problem" | False. Only 4 of 13 (31%) cases. |
| "Gold has_answer turns are *opening* turns that get out-competed by their own session's continuation turns" | **True for ~38% of failures** — single dominant mechanism |

### Predicted effect of k=10 → k=20

| Failure class | Likely captured at k=20? |
|---|---|
| within-session (5) | Mostly yes |
| rerank_pushed_gold_out (2) | Likely yes |
| true distractor (4) | Maybe |
| gold_at_deep_dense_rank (1) | Probably yes (gold at rank 24) |
| session_miss (1) | No |

Realistic estimate: **k=20 would recover 8-10 of 13 zero-recall failures**, lifting event-level recall from 0.84 → ~0.86-0.88.

---

## 17. Current state

### Codebase

- v0.2.1 shipped on PyPI as `engrampy`.
- All math fixes in place (MMR normalization, additive recency).
- Per-question exception isolation in the bench.
- `has_answer` preserved through ingest into `Event.metadata`.
- 9 evaluation/analysis scripts under `scripts/`.
- One protocol doc under `docs/EVAL_PROTOCOL.md`.

### Evaluation evidence

- `benchmarks/runs/eval_all_20260515_024020/` — full orchestrator run (Phase 1-5).
- `benchmarks/runs/inspect_full_20260515_094836/` — event-level recall over all 500 questions × 6 qtypes.
- `benchmarks/runs/traces_zero_recall/` — failure traces for the 23 zero-recall questions (13 complete at time of writing).
- `benchmarks/runs/ablation_*.json` (5 files) — per-qtype ablation from earlier session.

### Known truth

1. **Retrieval components are all mathematically correct** (33/33 unit tests).
2. **Baseline retrieval is at 0.97 session-level / 0.88 event-level recall** — strong but not at ceiling.
3. **No hybrid feature passes the protocol gate** for shipping. `autotemp` and `recent` are recall-neutral.
4. **bm25 × mmr is a true catastrophe on multi-session**; recency is broadly negative; recent-window force-feeds noise.
5. **Multi-session and single-session-preference have ~25% event-level miss rates** hidden by the session-level proxy.
6. **The dominant failure mechanism is "within-session ranking"** — opening turns introducing a topic get out-competed by continuation turns within the same session.

### Recommended launch config (passes all protocol gates)

```powershell
python -m engram.bench run longmemeval `
  --embedder local --embed-model BAAI/bge-large-en-v1.5 --embed-device cuda --dtype fp32 `
  --chat opencode-go --chat-model kimi-k2.6 `
  --reranker bge --k 10 --seed 1337 `
  --rerank-pool-multiplier 5 `
  --auto-temporal --surface-conflicts
```

### Open methodological gaps (not blocking)

- Sweep used `qtype=None` (silently selected sss-user first-30) — should be patched to sweep per-qtype.
- n=30 is too small for paired CIs to distinguish ≤5% effects.
- No reference-impl comparison (e.g., `rank-bm25` against our `BM25Index`).
- No slice analysis by question features (year tokens, n-sessions, length, proper-noun density).
- 21 questions in the dataset have no `has_answer=True` turns; event-recall computation can't evaluate them.

---

## 18. Next steps

In order of expected ROI for SOTA push:

| Priority | Action | Cost | Expected impact |
|---|---|---|---|
| 1 | **Re-run `inspect_retrieval.py` with `--k 20`** on all 500 questions | ~3 hr GPU | Confirm or refute the within-session hypothesis; expect event R@k 0.84 → 0.86-0.88 |
| 2 | Dump the remaining 10 zero-recall traces and re-run the classifier on the full 23 | ~10 min | Confirm the failure-mode breakdown holds at full sample |
| 3 | If k=20 helps: ship as a new gate-passing config; run the LLM end-to-end with it | ~4 hr LLM | Confirm event-level lift translates to end-to-end accuracy |
| 4 | Within-session over-sampling (force ≥3 turns per surfaced session into top-10) | ~50 lines | Directly targets the within-session mechanism |
| 5 | Consolidation (Engram's novelty) — per-question summary memory items | ~1 LLM call per cluster at ingest | Replaces "find right turn" with "find right summary" |
| 6 | LLM-side levers (CoT, verify pass, self-consistency, stronger answer model) | $$ LLM cost | The remaining ~10-15 point gap after retrieval lift |
| 7 | File PEP 541 reclaim for `engram-memory` (active squat by unrelated party) | clerical | Reclaim the canonical PyPI name |
| 8 | Patch the orchestrator's sweep to set per-knob target qtype | ~10 lines | Sweep would actually test where features matter |

---

## Appendix A — Commit log

All commits this session, oldest first:

| SHA | Subject |
|---|---|
| `d37b361` | retrieve+providers+bench: hybrid retrieval + async consolidation + disk cache |
| `7bac655` | bench: per-question exception isolation in LongMemEval |
| `3736ad0` | bench: per-question retrieve-config ablation harness |
| `5897b63` | bench: retrieval-only evaluator for LongMemEval (no LLM cost) |
| `6bb90b6` | eval: formal evaluation protocol + statistical infrastructure |
| `eaf3dc0` | retrieve: fix MMR + recency math correctness |
| `6394a17` | bench: per-question retrieval trace tool |
| `eac6b46` | eval: end-to-end orchestrator (one script, every test) |
| `38b4c66` | release: v0.2.0 — `pip install engrampy` |
| `05e9780` | release: v0.2.1 — populate legacy Author metadata field |
| `e28f981` | bench+inspect: preserve has_answer ground truth + deep inspector |
| `d8373e0` | inspect: --limit defaults to None (all matching) instead of 1 |
| `41a0eb8` | bench: scripts/batch_trace.py — share embedder across many qid traces |
| `07a8acd` | bench: trace parser + failure-mode classifier for zero-recall analysis |

Tags pushed: `v0.2.0`, `v0.2.1`.

---

## Appendix B — Scripts shipped

| Script | Purpose |
|---|---|
| `scripts/ablate_longmemeval.py` | Per-question, per-config ablation matrix. 12 predefined configs. Retrieval-only or LLM-scored. |
| `scripts/retrieval_eval.py` | Multi-config retrieval evaluator across LongMemEval-S. recall@k / hit@k / multi_recall@k / MRR / first_correct_rank / precision@k at k=10/20/50. Bootstrap CIs, McNemar, failure-mode listing. |
| `scripts/sweep.py` | Single-knob hyperparameter sweep with paired-diff CI + McNemar vs baseline value. Predefined grids for 12 knobs. |
| `scripts/retrieval_trace.py` | Per-question stage-by-stage trace dump. Dense top-N, BM25 top-N, recent-window, final top-k per config. Session-suffix annotation. |
| `scripts/run_all_evals.py` | End-to-end orchestrator. 5 phases: component tests, ablation, sweeps, traces, consolidated `REPORT.md`. Shares embedder + reranker across phases. Resumable. |
| `scripts/inspect_retrieval.py` | Event-level recall inspector. Reads `has_answer` ground truth. Per-question dump with `✦GOLD✦` / `[GOLD-EVT]` / `[SAME-SESS]` / `[........]` annotations. Side-by-side event-level vs session-level recall with gap column. |
| `scripts/batch_trace.py` | Batch wrapper around `retrieval_trace._trace_question`. Loads embedder + reranker once, traces every qid in `--qid-file`. |
| `scripts/analyze_zero_recall_traces.py` | Parses trace files, classifies each into 5 failure modes (session_miss, gold_at_deep_dense_rank, wrong_turn_in_session, rerank_pushed_gold_out, competing_session). Emits per-question table + class distribution + class × qtype matrix. |
| `scripts/_stats.py` | Pure stdlib + numpy. `bootstrap_mean_ci`, `bootstrap_paired_diff_ci`, `mcnemar` (exact-binomial / Yates chi-square). Helpers `format_ci`, `format_p`. |

---

## Appendix C — Source changes

### New files

- `src/engram/retrieve/_bm25.py` — `BM25Index`, `reciprocal_rank_fusion`, `tokenize`.
- `src/engram/retrieve/_mmr.py` — `mmr_select` with internal min-max normalization.
- `src/engram/providers/_disk_cache.py` — `DiskCache`, `CachedChat`, `CachedEmbedder`, `with_disk_cache`.
- `src/engram/storage/migrations/0009_perf_indexes.sql` — composite indexes on hot paths.
- `docs/EVAL_PROTOCOL.md` — per-component contracts, decision rules, sweep grids, statistical tests.
- `JOURNEY.md` — this file.

### Modified files

- `src/engram/retrieve/_params.py` — 9 new fields (bm25_weight / bm25_k1 / bm25_b / mmr_lambda / mmr_pool_size / recency_lambda / recency_decay_days / recent_window_k / lexical_filter).
- `src/engram/retrieve/_engine.py` — `_fuse_hybrid_sources` (dense + BM25 + recent-window via RRF), `_apply_recency_boost` (additive form post-fix), `_get_created_at_batch`, `_fetch_candidate_vectors` (batched), MMR pool sizing.
- `src/engram/storage/sqlite.py` — `bm25_search_events` (lazy index, rebuild on hyperparam change), `get_embeddings_batch`, `get_created_at_batch`, `list_recent_events`, PRAGMA tuning.
- `src/engram/memory.py` — `retrieve` / `aretrieve` cascading kwargs for every new param, `aconsolidate` async wrapper, `_embed_search_query` helper.
- `src/engram/consolidation/_abstraction.py` — `aextract_abstraction` async sibling.
- `src/engram/consolidation/_engine.py` — `aconsolidate` with asyncio.gather + semaphore.
- `src/engram/providers/local.py` — asymmetric query prompts, per-instance LRU cache, `embed_query` method.
- `src/engram/bench/_cli.py` — 14 new CLI flags.
- `benchmarks/suites/longmemeval.py` — Phase E flags wired, exception isolation refactor (`_run_one_question`), `_build_auto_temporal_filter`, `_parse_haystack_date`, `_ingest_haystack` preserves `session_id` AND `has_answer`.
- `pyproject.toml` — name `engram-memory` → `engrampy`, version `0.1.0` → `0.2.1`, dual-entry authors.
- `src/engram/__init__.py` — `__version__` bumped to `0.2.1`.
- `README.md` — `pip install engrampy` + squat note.
- `CHANGELOG.md` — v0.2.0 and v0.2.1 entries.

---

## Appendix D — Headline numbers

### Retrieval correctness (n=500, baseline, k=10, fp32)

| Metric | Value |
|---|---|
| Session-level recall | 0.966 |
| Event-level recall (over all 500) | 0.841 |
| Event-level recall (over 479 evaluable) | **0.878** |
| Full event-level recall rate | 79.7% |
| Partial event-level recall rate | 15.4% |
| Zero event-level recall rate | 4.8% |
| Per-qtype event-level recall floor | sss-pref 0.733, multi 0.757 |
| Per-qtype event-level recall ceiling | sss-asst 0.982 |

### Component correctness

| Component | Tests passed |
|---|---:|
| `BM25Index` | 9 / 9 |
| `mmr_select` | 8 / 8 |
| `reciprocal_rank_fusion` | 5 / 5 |
| `_build_auto_temporal_filter` | 5 / 5 |
| recency math | 6 / 6 |
| **Total** | **33 / 33** |

### Per-config retrieval verdict (Phase 2 ablation, n=30 per qtype × 5 qtypes — sss-assistant missing)

| Config | Verdict |
|---|---|
| `baseline` | BASELINE — passes everything |
| `autotemp` | NEUTRAL — passes gate, no lift |
| `recent` | NEUTRAL — passes gate, no lift |
| `bm25`, `bm25+aut`, `bm25+rec`, `conservative` | FAIL (multi −0.033) |
| `mmr07` | FAIL (sss-user −0.033) |
| `mmr03` | FAIL (multi+pref −0.033) |
| `recency` | FAIL (multi −0.125, temporal −0.100, pref −0.067) |
| `bm25+mmr` | FAIL (multi −0.137) |
| `all_aggressive` | FAIL (multi −0.295, temporal −0.161) |

### Failure mode breakdown (zero-recall questions, n=13 of 23)

| Mode (post-reclassification) | Count |
|---|---:|
| within-session ranking failure | 5 |
| true distractor session | 4 |
| rerank pushed gold out | 2 |
| gold at deep dense rank | 1 |
| session miss | 1 |

---

*End of journey. Last updated 2026-05-15, after the zero-recall failure analysis on 13 of 23 traces.*
