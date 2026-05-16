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
17. [Full-23 trace classification — within-session vs true distractor](#17-full-23-trace-classification--within-session-vs-true-distractor)
18. [Security audit (between sessions)](#18-security-audit-between-sessions)
19. [k=20 and rerank-off experiments — the breakthrough](#19-k20-and-rerank-off-experiments--the-breakthrough)
20. [Audit findings and SOTA implications](#20-audit-findings-and-sota-implications)
21. [Current state](#21-current-state)
22. [Next steps](#22-next-steps)
23. [Honest-judge experiment — n=100 stratified, Kimi-self vs Sonnet cross](#23-honest-judge-experiment--n100-stratified-kimi-self-vs-sonnet-cross)
24. [Appendix A — Commit log](#appendix-a--commit-log)
25. [Appendix B — Scripts shipped](#appendix-b--scripts-shipped)
26. [Appendix C — Source changes](#appendix-c--source-changes)
27. [Appendix D — Headline numbers](#appendix-d--headline-numbers)

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

## 17. Full-23 trace classification — within-session vs true distractor

After the JOURNEY's first cut (which covered 13 of 23 traces), the remaining traces completed and the analyzer was re-run on the full set. The picture is cleaner.

### Failure class distribution (full n=23)

| Class | Count | % |
|---|---:|---:|
| `competing_session` | 14 | 61% |
| `rerank_pushed_gold_out` | 6 | 26% |
| `session_miss` | 2 | 9% |
| `gold_at_deep_dense_rank` | 1 | 4% |

`competing_session` (some session captures ≥5 of top-10 slots) is the dominant class. But the analyzer's bucket name conflates two very different mechanisms — checking the dominator's session-suffix against each question's `answer_session_ids` reveals which.

### Reclassified by what's actually competing (n=23)

| True mechanism | Count | % | What's happening |
|---|---:|---:|---|
| **within-session ranking failure** | **12** | **52%** | The "dominator" session IS the answer session. We retrieved 5-9 turns from it, but none are the `has_answer=True` turn — the gold turn ranks below k=10 within the same session. |
| **true distractor session** | 11 | 48% | An off-topic session captures top-10. Gold session may be in dense top-50 (and rerank pushed it out, or gold sits at deep rank) but a different session wins the rerank. |

### Mechanism × qtype (full 23)

| | multi | sss-asst | sss-pref | sss-user | temporal | total |
|---|---:|---:|---:|---:|---:|---:|
| within-session | 3 | 1 | 2 | 0 | 6 | **12** |
| true distractor | 1 | 0 | 2 | 1 | 4 | 8 |
| session miss | 0 | 0 | 1 | 0 | 1 | 2 |
| gold deep | 0 | 0 | 0 | 0 | 1 | 1 |
| **total** | **4** | **1** | **5** | **1** | **12** | **23** |

### What the 23 reveal that the 13 didn't

| Observation | Why it matters |
|---|---|
| `rerank_pushed_gold_out` is 26% of zero-recall (6 of 23) | The cross-encoder is meaningfully misordering on a quarter of these failures — worth ablating |
| Temporal-reasoning is the dominant failing qtype (12 of 23 = 52%) | Within-session/true-distractor split is 6/5 here — temporal pain is structural, not one-sided |
| 3 of 6 `rerank_pushed_gold_out` cases are within-session | Even when gold is at dense rank 1 in the right session, rerank can drop it — the cross-encoder isn't blanket-biased toward gold sessions |

### Updated k=20 recovery prediction (made before running the experiment)

| Failure mechanism | Count | Recover at k=20? |
|---|---:|:---:|
| within-session | 12 | mostly yes — gold turn likely at rank 11-25 dense; bigger pool catches it |
| true distractor + rerank pushed gold out | 3 | likely yes — gold WAS dense top-10; bigger pool puts it back |
| true distractor — competing dominates | 5 | maybe partial |
| gold at deep dense rank (1, gold @ rank 24) | 1 | yes (just inside k=20 boundary) |
| session miss | 2 | no |

Predicted realistic recovery: 13-17 of 23 zero-recall failures. Predicted lift: 0.841 → 0.86-0.87 event-level recall.

---

## 18. Security audit (between sessions)

User did a security audit and fix pass between iterations of this journey. Five commits visible in `git log`, all keyed to M-numbered issues:

| SHA | Subject |
|---|---|
| `5bd4b3f` | agent: verify retry symmetry — no double-reinforce, vote on retry (M-47, M-187) |
| `7214504` | memory: protect _USER_STATE_FLAG from caller metadata overlay (M-181, M-188) |
| `7a518e5` | verify: document the locally-scoped last_response in verify loop (M-196) |
| `61ce3d1` | docs(security): rewrite SECURITY.md as full threat model |
| `ad894b3` | memory: serial-fallback _parallel_leaf_retrieves on :memory: (M-25) |

Adjacent quality-of-life changes during the same window (visible in working-tree diffs):

| Change | File |
|---|---|
| Dependency upper bounds added (`pydantic<3`, `numpy<3`, `openai<3`, `anthropic<1`) so a future install can't pull a breaking major | `pyproject.toml` |
| Removed `[postgres]`, `[duckdb]`, `[sqlite-vec]` extras (no in-tree consumer) | `pyproject.toml` |
| `__version__` now read from `importlib.metadata` (`engrampy`) instead of hardcoded | `src/engram/__init__.py` |

None of these changed retrieval behaviour, so prior evaluation results remain valid.

---

## 19. k=20 and rerank-off experiments — the breakthrough

User ran two retrieval-only experiments in parallel directions to disentangle the levers, each ~3 hours on all 500 questions:

- **k=20**: same config as baseline, top-k raised from 10 to 20. Tests the within-session hypothesis.
- **no rerank, k=10**: `--reranker none` to test whether the reranker is net positive (we knew it pushed gold out in 6 of 23 zero-recall failures; question was whether it salvages more cases than it breaks).

### Headline (n=500, baseline config + the tested change, no LLM)

| Run | Sess R@10 | Evt R@10 (all 500) | Evt R@k (evaluable 479) | Full recall (382 was baseline) | Zero recall |
|---|---:|---:|---:|---:|---:|
| baseline k=10 | 0.966 | 0.841 | 0.878 | 382 (76.4%) | 23 |
| **k=20** | **0.986** | **0.898** | **0.937** | **423 (84.6%)** | **11** |
| no rerank, k=10 | 0.939 | 0.796 | 0.831 | 353 (70.6%) | 35 |

**k=20 lifts event-level recall by +5.7 absolute points (0.841 → 0.898)**. Within 6 points of the theoretical ceiling on evaluable questions (0.937 vs 1.000).

**No-rerank loses 4.5 points and introduces 22 new zero-recall regressions**. The reranker is decisively net-positive.

### Per-qtype impact (event-level recall)

| qtype | n | baseline | k=20 | norerank | k=20 Δ | norerank Δ |
|---|---:|---:|---:|---:|---:|---:|
| sss-user | 70 | 0.900 | 0.900 | 0.886 | +0.000 | −0.014 |
| sss-assistant | 56 | 0.982 | 0.982 | 0.929 | +0.000 | −0.054 |
| **sss-preference** | 30 | 0.733 | **0.878** | 0.778 | **+0.144** | +0.044 |
| **multi-session** | 133 | 0.757 | **0.852** | 0.721 | **+0.095** | −0.036 |
| knowledge-update | 78 | 0.904 | 0.923 | 0.823 | +0.019 | −0.081 |
| **temporal-reasoning** | 133 | 0.821 | **0.896** | 0.755 | **+0.074** | −0.067 |
| **overall** | 500 | 0.841 | **0.898** | 0.796 | **+0.057** | −0.045 |

Hardest qtypes (preference, multi-session, temporal) benefit the most from k=20 — exactly what the within-session hypothesis predicted.

### Recovery of the 23 baseline zero-recall failures

| Config | Full recovery | Partial recovery | Still zero | NEW zero-recall introduced |
|---|---:|---:|---:|---:|
| k=20 | 5 | 7 | 11 | **0** |
| no rerank | 4 | 6 | 13 | **22** |

k=20 recovers 12 of 23 with zero collateral damage. No-rerank recovers 10 but breaks 22 others — net loss.

### Distribution shift (event-level recall buckets across 500 questions)

| Bucket | baseline | k=20 | norerank |
|---|---:|---:|---:|
| evt R = 0 | 44 | **32** | 56 |
| evt R in (0, 0.25] | 5 | 2 | 5 |
| evt R in (0.25, 0.5] | 45 | **20** | 71 |
| evt R in (0.5, 0.75] | 23 | 19 | 14 |
| evt R in (0.75, 1.0) | 1 | 4 | 1 |
| evt R = 1.0 | 382 | **423** | 353 |

k=20 moves the distribution sharply right: **−25 in the (0.25, 0.5] bucket, +41 to full recall**.

### Verdict per protocol gates

| Gate | k=20 | no-rerank |
|---|:---:|:---:|
| Paired Δ recall CI excludes zero | ✓ +0.057 (positive side) | ✗ −0.045 (negative side) |
| McNemar on zero-recall recovery | ✓ 12 recoveries, 0 regressions | ✗ 10 recoveries, 22 regressions |
| No per-qtype regression > 0.02 | ✓ no qtype regresses | ✗ 5 qtypes regress |

**k=20 passes all three gates. Ship it. No-rerank fails all three. Keep the reranker.**

### What's still zero at k=20 (the hard wall, 11 questions)

- 2 session-miss cases (gold not in dense top-50 at all)
- ~5-7 true-distractor cases where the off-topic session dominates even k=20
- Possibly some embedder mismatches

Need: stronger embedder, query rewriting (HyDE / decompose), or consolidation. Pure k-bump won't help.

### Updated launch config

```powershell
python -m engram.bench run longmemeval `
  --embedder local --embed-model BAAI/bge-large-en-v1.5 --embed-device cuda --dtype fp32 `
  --chat opencode-go --chat-model kimi-k2.6 `
  --reranker bge --k 20 --seed 1337 `
  --rerank-pool-multiplier 5 `
  --auto-temporal --surface-conflicts
```

Single new flag vs the JOURNEY's earlier recommendation: `--k 20` (was `--k 10`).

---

## 20. Audit findings and SOTA implications

`audit/audit_2026-05-15.md` — 13 specialized agents producing ~1000 raw / ~620 unique findings across the codebase. Severity distribution: 5 CRITICAL, 96 HIGH, 211 MEDIUM, 245 LOW, 63 INFO. Five M-numbered fixes have landed (see Section 18); the rest are still open.

This section catalogues only the audit items that affect what we **claimed** in this session, separating "still valid" from "now invalidated" from "needs a fix before any equivalent claim is honest."

### SOTA-relevant findings, by impact on our work

| Audit ID | Title | Impact on our session findings | Status |
|---|---|---|---|
| **C-03** | LoCoMo "exact match" is substring containment | Would invalidate any LoCoMo SOTA claim. We made none this session, but it pre-empts the planned LoCoMo run. | open |
| **C-04** | LoCoMo embeddings stored un-normalized — cosine wrong | Same as C-03 — invalidates any LoCoMo number until fixed. | open |
| **C-05** | Bench-baseline BM25 is O(qt × N × scan) pure Python | Any "Engram BM25 beats baseline X" claim is inflated by a slow baseline, not a fast engine. We made no direct head-to-head this session. | open |
| **H-76** | `engram_config` field is empty by default; ~30 retrieval knobs not captured in manifest | Reproducibility of every bench manifest is degraded — we have to read git log + script source to know what produced a number. Our `inspect_retrieval.py` outputs include CLI args via `config_args` so our event-level findings are self-describing, but bench-suite manifests aren't. | open |
| **H-77** | LongMemEval per-question Exception swallowed → counted as 0 (mixes infra failure with wrong answer) | Our commit `7bac655` added exception isolation but still scores 0 on error — the audit says we should track `n_errored / n_completed` separately and quote `correct / n_completed`. Killed-LLM runs from earlier in this session aggregated errors-as-zero. The retrieval-only inspect path doesn't make LLM calls and is unaffected. | open |
| **H-78** | Judge yes/no parser diverges from official LongMemEval scorer (`"yes" in raw and "no" not in raw.split("yes", 1)[0]`) | The end-to-end accuracy projections we cited (the killed LLM runs at mid-70s, the predicted 0.78-0.83 at k=20) all used this non-canonical judge. They're directionally indicative but **not comparable to published LongMemEval scores**. | open |
| **H-79** | Auto-temporal lexical_filter fallback un-recorded | Our `--auto-temporal` runs don't record which questions actually had the filter fire and which had the fallback. Doesn't invalidate the aggregate numbers — just hurts post-hoc forensics. | open |
| **H-80** | Seed only seeds Python `random`; numpy / torch / transformers not seeded | Reranker outputs (torch) and numpy operations are not deterministic between runs even with `--seed`. Absolute event-level recall numbers may shift ~0.5-1% across re-runs. Paired-diff comparisons (k=10 vs k=20, baseline vs no-rerank) ARE robust because both arms see the same noise. | open |
| **H-83** | `ablate_longmemeval.py` default output clobbers previous runs | We routed around this by passing explicit `--output` paths every time. Not a current-evidence issue. | open |
| M-25 | `_parallel_leaf_retrieves` race on `:memory:` storage | **Fixed** in `ad894b3`. The fix forces serial fallback on `:memory:`. Our inspect_retrieval runs use `:memory:` and may have had marginal non-determinism pre-fix; post-fix runs are stable. | fixed |
| M-152 | LongMemEval qtype rubric fallback to multi-session | Affects judge rubric selection, only matters for LLM-scored runs. | open |
| M-153 | LongMemEval `confidence_intervals` are zero-width `(v, v)` | We computed our own CIs via `scripts/_stats.py`, so our protocol-gate verdicts are not affected. The suite's manifest field is just unused. | open |
| M-154 | Judge prompt version not in manifest | Reproducibility, not invalidation. | open |
| M-156 | `ingest_ms` not in `latency_ms` manifest | Forensic only. | open |
| M-170 | `analyze_zero_recall_traces.py` regex parser is brittle to trace format | If the trace format changes, the parser silently misclassifies. We froze the format mid-session; current findings stand. | open |

### What's still valid

| Claim from this session | Why it survives the audit |
|---|---|
| Phase 1: 33/33 component unit tests pass | Synthetic data, no provider, no bench suite. Unaffected. |
| Per-qtype ablation matrix (Section 9-10) | Retrieval-only path; doesn't use the suite-side judge or LLM. |
| `bm25 × mmr` catastrophic on multi-session (−0.137 to −0.442) | Effect size dominates any seeding noise; paired comparison robust. |
| 23 zero-recall failures + mechanism classification | Based on `inspect_retrieval` + traces, not the bench judge. |
| **k=20 lifts event-level recall by +5.7 absolute** | n=500 paired diff; H-80 noise insufficient to flip the sign. |
| **No-rerank loses 4.5 points; reranker net-positive** | Same. The 22 new zero-recall regressions are a large signal vs the noise floor. |
| Failure-mode classifier outputs (within-session vs distractor) | Mechanistic findings from the traces, not aggregate scores. |
| Recommended retrieval-side launch config (`--k 20 --auto-temporal --surface-conflicts`) | All retrieval-only validation. |

### What is now suspect or needs a fix before claiming

| Claim | What's wrong | What to do |
|---|---|---|
| **End-to-end LongMemEval accuracy** (killed-LLM-run mid-70s, predicted 0.78-0.83 at k=20) | H-78 judge parser ≠ official; H-77 mixes infra failures into wrong-answer count | Fix H-78 and H-77; re-run end-to-end; only THEN cite a number as comparable to published LongMemEval |
| **Any LoCoMo number we might run next** | C-03 (substring not exact match) + C-04 (un-normalized embeddings) | Fix both before invoking the LoCoMo suite |
| **Any "engram BM25 beats baseline" claim** | C-05 baseline is slow Python; lead is inflated | Fix C-05 (vectorize / reuse engine BM25Index) before any direct comparison |
| **Reproducibility of any retrieval recall number** | H-80 incomplete seeding | Seed numpy/torch/transformers; re-run for record (paired Δs remain valid in the meantime) |
| **Bench manifest as evidence of record** | H-76 empty `engram_config` | Populate `engram_config` from CLI args before next bench run |

### Net effect on the session's takeaway

Most of what we shipped this session is **retrieval-side and judge-independent**, so the audit doesn't unwind the headlines:

- The component-level math is correct (Phase 1).
- The catastrophic interactions (`bm25 × mmr`, recency, recent-window) are real.
- The `--k 20` launch addition is gate-passing.
- The reranker is net-positive.
- Event-level recall ceiling at k=20 is ~0.94 on evaluable questions.

What it DOES unwind is **the link from retrieval to end-to-end accuracy**:

- The mid-70s baseline trajectory came from runs whose judge wasn't the official one.
- The 0.78-0.83 predicted lift at k=20 should be re-expressed as "expected magnitude" rather than a comparable LongMemEval score.

Before any SOTA-claiming LLM run lands on `SCOREBOARD.md`, **H-78 (judge) and H-77 (error accounting) MUST be fixed**. H-80 (seeding) and H-76 (manifest config) should land alongside for the same record.

### 20a. Joined retrieval × LLM 2×2 — the SOTA gap is 4:1 toward LLM, not 50/50

A separate post-audit analysis joined `inspect_full_20260515_094836` (event-level recall, k=10) with the 71.4% v0.1.0 release manifest (`0b6dfa53`, same k=10 config). The cross-tab over all 500 questions:

| | n | % | meaning |
|---|---:|---:|---|
| retrieve hits ≥1 gold turn AND LLM correct | 343 | 68.6% | working as intended |
| retrieve hits ≥1 gold turn AND LLM wrong | 113 | **22.6%** | **LLM-stage loss** — retrieval did its job, LLM failed |
| retrieve miss all gold AND LLM correct | 14 | 2.8% | lucky / world-knowledge |
| retrieve miss all gold AND LLM wrong | 30 | **6.0%** | **retrieval-stage loss** — gold not in top-10 |

**This invalidates the earlier "split roughly evenly between retrieval and LLM" claim** at the top of this section. The actual ratio is ~3.77:1 in favor of LLM-stage loss. Retrieval is ~4× smaller a lever than the LLM stage at the end-to-end level.

#### The 81-question hard wall — perfect retrieval, LLM still failed

Of the 113 LLM-stage losses, **81 had event_recall = 1.0** — every single `has_answer=True` turn was in the LLM's top-10 context. The LLM still produced a wrong answer.

| Failure shape within the 81 | Count |
|---|---:|
| Said "I don't know" with the answer literally in the prompt | 52 (64%) |
| Confidently produced a wrong number / value | 19 |
| Partial / off-by-one answer | 10 |

By qtype:

| qtype | hard-wall count | what the question actually wants |
|---|---:|---|
| multi-session | 24 | arithmetic over multiple turns ("total weight", "how many hours", "total distance") |
| temporal-reasoning | 21 | date arithmetic ("how many weeks since", "how long before") |
| knowledge-update | 19 | pick the latest value when many are present |
| single-session-preference | 10 | synthesize a preference statement ("user would prefer X") |
| sss-user / sss-assistant | 7 | refusal under clear evidence |

#### "I don't know" is the single biggest leak

| | count / rate |
|---|---:|
| Total "I don't know" responses in the 500-question run | 100 (20%) |
| Of those, scored wrong (gold existed) | 93 |
| Refusal rate among LLM-stage losses (temporal-reasoning) | 78.6% |
| Refusal rate among LLM-stage losses (sss-user) | 75.0% |
| Refusal rate among LLM-stage losses (multi-session) | 46.7% |
| Conditional accuracy when retrieval is perfect (event_recall=1.0) | **78.8%** |

Even with the ground truth in context, the answer model refuses 21% of the time.

#### What k=20 actually buys end-to-end (revised)

The §19 "k=20 lifts event-level recall by +5.7" is correct at the retrieval level. The end-to-end translation is **much smaller**, bounded by conditional accuracy on the questions whose retrieval changes:

| | Value |
|---|---:|
| Retrieval-stage losses at k=10 | 30 |
| Of those, recovered at k=20 (event_hit 0 → ≥1) | 5 |
| Conditional accuracy on questions where retrieval hits | 75.2% |
| Theoretical end-to-end ceiling from k=20 lift | ~30 × 0.75 ≈ +4.4 pts |
| Realistic end-to-end lift estimate | **+1-2 pts** |

The earlier "predicted 0.78-0.83 end-to-end" was unfounded extrapolation and is **retracted**. The honest claim is "+1-2 pts at the end-to-end level" — bounded by how often the LLM converts a newly-retrieved gold turn into a correct answer.

#### Three contaminants making the 71.4% benchmark number soft

| Contaminant | Effect on reported 71.4% |
|---|---|
| Same-model judge (Kimi K2.6 answers AND judges) | +3-7 pts self-preference inflation (per SCOREBOARD notes) |
| H-77 swallowed exceptions counted as score=0 | unknown fraction of 143 wrong answers are infra failures, not wrong answers |
| H-78 judge parser divergence from official LongMemEval | of the 93 wrong-scored "I don't know" responses, some may be false negatives |

Estimated "real" baseline under official scoring + 3rd-party judge: **64-68%** (a band, not a number).

#### Strategic implication — where the SOTA budget should go

The 81 hard-wall failures cluster into mechanisms, none of which retrieval tuning can address:

| Mechanism | Hard-wall count | Engram feature that targets it |
|---|---:|---|
| Arithmetic / aggregation | 24 | Consolidation — pre-computed rollups per topic |
| Temporal arithmetic | 21 | Consolidation — events with explicit date math at ingest |
| Latest-value resolution | 19 | Contradiction resolver with `valid_until` / PREFER_RECENT |
| Preference synthesis | 10 | Consolidation — explicit preference extraction |
| Refusal under evidence | 7 | Prompt engineering on the answer side; not Engram-shaped |

This reframes the project thesis: **the next 10-15 pts live in consolidation, not retrieval tuning**. Specifically the move "find right turn → find right summary" — replacing N raw turns the LLM must aggregate, with 1 pre-aggregated abstraction the LLM only needs to recognize.

This is also the only headline lift that is **attributable to the Engram hierarchy** rather than to "bigger model" or "better retriever." (a) bigger LLM helps any vector DB. (c) tool-use / CoT is generic LLM engineering. (b) consolidation is the only one that says "the hierarchy matters."

---

## 21. Current state

### Codebase

- v0.2.1 shipped on PyPI as `engrampy`.
- All math fixes in place (MMR normalization, additive recency).
- Per-question exception isolation in the bench.
- `has_answer` preserved through ingest into `Event.metadata`.
- 9 evaluation/analysis scripts under `scripts/`.
- One protocol doc under `docs/EVAL_PROTOCOL.md`.
- Security audit pass complete (5 M-numbered fix commits).
- Dependency upper bounds + pruned dead extras (`pyproject.toml`).
- `__version__` now sourced from `importlib.metadata`.

### Evaluation evidence

- `benchmarks/runs/eval_all_20260515_024020/` — full orchestrator run (Phase 1-5).
- `benchmarks/runs/inspect_full_20260515_094836/` — event-level recall, baseline k=10 over all 500 questions × 6 qtypes.
- `benchmarks/runs/inspect_k20_20260516_004707/` — event-level recall, k=20 over all 500 questions.
- `benchmarks/runs/inspect_norerank_20260516_040506/` — event-level recall, rerank disabled over all 500 questions.
- `benchmarks/runs/traces_zero_recall/` — failure traces for the 23 zero-recall questions (full set + analyzer output).
- `benchmarks/runs/ablation_*.json` (5 files) — per-qtype ablation from earlier session.

### Known truth (post-audit-reframe + honest-judge experiment §23)

1. **Retrieval components are all mathematically correct** (33/33 unit tests).
2. **Best-config retrieval is at 0.99 session-level / 0.94 event-level recall** (k=20) — within 6 points of ceiling.
3. **k=20 ships at all three protocol gates** (Δ +0.057 with no per-qtype regression).
4. **Reranker is decisively net-positive** — disabling it loses 4.5 points and adds 22 new zero-recall failures.
5. **No hybrid feature** (bm25, mmr, recency, recent-window) passes the gate; `autotemp` and `surface-conflicts` are recall-neutral keepers.
6. **The end-to-end gap is 4:1 in the LLM's favor** — 22.6% LLM-stage loss vs 6.0% retrieval-stage loss on the v0.1.0 release manifest. The k=20 retrieval lift translates to ~+1-2 pts end-to-end, not the +5.7 the retrieval-only number suggests.
7. **81 of 113 LLM-stage losses are hard-wall** — retrieval was perfect (event_recall=1.0), LLM still failed, 64% by refusing ("I don't know" with answer in prompt).
8. **The next big lift is consolidation**, not retrieval tuning — it's the only Engram-attributable mechanism that targets the hard-wall failure modes (aggregation, latest-value, preference synthesis).
9. **Self-preference inflation is NOT a major contaminant** (revised — §23). The hypothesis that Kimi-self inflated v0.1.0 by 3-7 pts was disproved: on a stratified n=100 sample, Kimi-self lands at 66.0% and Sonnet 4.5 cross-judge lands at 68.0% — judges agree 94% and the 2-pt aggregate gap points the *other* direction. The honest baseline is ~67% (averaged across judges, at k=20 with autotemp+surface-conflicts on this sample).
10. **The 4-pt drop from v0.1.0's 71.4% to ~67% on the 100q stratified sample is sample distribution + retrieval config, not judge bias** (§23). The 100q stratified is ~4 pts harder than the leading 500q; k=20+autotemp+surface-conflicts is roughly net-neutral on this slice.
11. **The single highest-ROI prompt fix is the abstain pattern** (§23) — 8 of 32 Sonnet-judged failures (25%) are `_abs` questions where Engram correctly says "I don't know" but the LongMemEval scorer wants the richer "you didn't mention X but you did mention Y" form. Estimated lift: +5-7 pts on the same 100q for ~$0.25 OpenRouter spend.

### Recommended launch config (passes all protocol gates)

```powershell
python -m engram.bench run longmemeval `
  --embedder local --embed-model BAAI/bge-large-en-v1.5 --embed-device cuda --dtype fp32 `
  --chat opencode-go --chat-model kimi-k2.6 `
  --reranker bge --k 20 --seed 1337 `
  --rerank-pool-multiplier 5 `
  --auto-temporal --surface-conflicts
```

### Open methodological gaps (not blocking)

- Sweep used `qtype=None` (silently selected sss-user first-30) — should be patched to sweep per-qtype.
- n=30 is too small for paired CIs to distinguish ≤5% effects (the n=500 k=20 experiment doesn't have this problem).
- No reference-impl comparison (e.g., `rank-bm25` against our `BM25Index`).
- No slice analysis by question features (year tokens, n-sessions, length, proper-noun density).
- 21 questions in the dataset have no `has_answer=True` turns; event-recall computation can't evaluate them.
- k=30 / k=50 untested — diminishing returns expected but unconfirmed.

---

## 22. Next steps

The §23 honest-judge experiment reframed the priority stack again. Bench hygiene (H-76/77/78/80) has shipped (commit `9130085`). GPU concurrency cap shipped (commit `815b953`). Self-preference inflation hypothesis is disproved; the honest baseline is 66-68% on n=100, not the predicted 64-68%. The abstain-prompt fix is now the cheapest +5-7pt lever in the project.

### Critical path (now mostly done)

| # | Action | Status | Notes |
|---|---|---|---|
| 1 | Fix H-78 (judge parser) — match official LongMemEval scorer | ✓ shipped `9130085` | First-line lowercase exact-match `^yes\b` / `^no\b` |
| 2 | Fix H-77 (error accounting) — split `n_errored` from `n_completed` | ✓ shipped `9130085` | `accuracy_correct` / `n_completed` / `error_rate` per qtype |
| 3 | Fix H-80 (seeding) — seed numpy/torch/transformers/sentence-transformers | ✓ shipped `9130085` | `engram._seed.seed_everything()` |
| 4 | Fix H-76 (populate `engram_config` in manifest) | ✓ shipped `9130085` | Knobs + provider descriptors in every manifest |
| 5 | Re-judge with non-self LLM | ✓ done (§23) | Sonnet 4.5 cross-judge at n=100 → 68.0%; Kimi self at n=100 → 66.0%; agreement 94%. Self-preference is NOT the contaminant. |
| 5a | Bench infrastructure: parallel eval + stratified sample + GPU lock | ✓ shipped `9130085` + `815b953` | `--parallel`, `--sample`, `--gpu-concurrency`, ThreadPoolExecutor + process-wide BoundedSemaphore |

### Cheapest next lever — abstain-prompt fix

| # | Action | Cost | Expected outcome |
|---|---|---|---|
| 6 | **Add abstain-handling instruction to answer prompt** — when retrieved memory doesn't contain the answer, instruct the model to (a) state what *was* mentioned in related context, (b) then clearly say it doesn't know about the specific thing | ~10 lines of prompt edit | Recovers 6-7 of the 8 `_abs` failures on the same 100q sample → 68% → 73-74% Sonnet-judged. Total OpenRouter spend: ~$0.25 (judge calls only; answer + ingest hit cache). |

### The decisive experiment for the Engram thesis

| # | Action | Cost | Expected outcome |
|---|---|---|---|
| 7 | **Consolidation-on-100q experiment** — same `--sample 100 --seed 1337` but with `--consolidate --consolidate-chat openrouter --consolidate-chat-model anthropic/claude-haiku-4-5 --aconsolidate-concurrency 50`. Compare against §23 baseline. | ~$3-5 OpenRouter (Haiku consolidation at ~38 clusters/q × 100 q net of cached 18). Plus ~$0.25 Sonnet judge. | Predicted lift: +5-10 pts on the hard-wall qtypes (preference synthesis, multi-hop aggregation, latest-value resolution). **This is the only experiment that can attribute lift to the Engram hierarchy specifically** rather than to a bigger model or better retriever. |
| 8 | If (7) lands ≥5 pts: full 500q run with the same config | ~$15-25 LLM (mostly Haiku) | The headline SOTA bench run |

### Retrieval-side follow-ups (lower priority now)

| # | Action | Cost | Realistic impact |
|---|---|---|---|
| 9 | Test k=30 / k=50 retrieval-only | ~3 hr each, no LLM | Maybe +0.5-1 pt end-to-end on top of k=20 |
| 10 | Within-session over-sampling | ~50 lines | Raises precision; could lift conditional accuracy when retrieval is partial |
| 11 | Slice analysis on the 11 still-zero-at-k=20 failures | ~30 min | Identifies whether the remaining hard wall is embedder-bounded or query-rewriting-bounded |

### LLM-side follow-ups (for the remaining 15-20 pt headroom)

| # | Action | Cost | When |
|---|---|---|---|
| 12 | Verify pass on answers (we already have the plumbing) | 1.5× LLM | After (7) — see if verify catches the 19 confident-wrong answers in the hard wall |
| 13 | Self-consistency N=3 majority vote on the refusal cluster | 3× LLM | Specifically target the 52 "I don't know with answer in prompt" cases |
| 14 | Per-qtype answer prompt routing (preference / temporal / multi-hop / factual) | ~150 lines | Predicted +2-4 pts; Sonnet sometimes lenient on preference, Kimi sometimes lenient on temporal — explicit per-qtype instructions can close both gaps |

### Future suites (blocked on audit fixes)

| # | Action | Why blocked |
|---|---|---|
| 15 | Fix C-05 (vectorize bench BM25 baseline) | Required before any "engram-vs-baseline" comparison is honest |
| 16 | Fix C-03 + C-04 (LoCoMo exact-match + normalization) | Required before any LoCoMo number is comparable to published |
| 17 | Run LoCoMo suite | After (16) |

### Housekeeping

| # | Action |
|---|---|
| 18 | File PEP 541 reclaim for `engram-memory` (active squat by unrelated party) |
| 19 | Patch the orchestrator's sweep to set per-knob target qtype |

---

## 23. Honest-judge experiment — n=100 stratified, Kimi-self vs Sonnet cross

After the bench hygiene fixes shipped (commit `9130085`) and the GPU concurrency cap shipped (commit `815b953`), we ran the same `--sample 100 --seed 1337` configuration twice with only the judge swapped. Same Kimi K2.6 answers (cached on disk, identical across the two runs), same retrieval (k=20 + autotemp + surface-conflicts), same gpu_concurrency=1. The only knob that changed: the judge model.

The motivation: the SCOREBOARD has carried a "Kimi self-judge inflates by 3-7 pts" caveat since v0.1.0. This was a hypothesis, never measured. We finally measured it.

### Setup

- **Sample**: stratified 100q with `seed=1337`. Composition: sss-user 14, multi-session 27, sss-preference 6, temporal 27, knowledge-update 15, sss-assistant 11.
- **Retrieval**: bge-large-en-v1.5 + bge-reranker-v2-m3 (fp32, CUDA), k=20, `--auto-temporal --surface-conflicts --rerank-pool-multiplier 5`.
- **Answer**: Kimi K2.6 via opencode-go.
- **Judges (two runs)**:
  - Cross-family: Claude Sonnet 4.5 via OpenRouter (`anthropic/claude-sonnet-4-5`).
  - Self: Kimi K2.6 via opencode-go.
- **No consolidation** in either run — clean baseline for the comparison.
- Manifests:
  - Sonnet: `benchmarks/runs/20260516T190353_627654+0000-815b953f-dirty-longmemeval.json`
  - Kimi-self: `benchmarks/runs/20260516T194247_734439+0000-815b953f-dirty-longmemeval.json`

### Headline

| Judge | accuracy_correct | n_errored |
|---|---:|---:|
| Claude Sonnet 4.5 | **0.680** | 0 |
| Kimi K2.6 (self) | **0.660** | 0 |
| **Gap** | **−0.020** | — |

**The self-preference inflation hypothesis is disproved on this sample.** Kimi grades its own answers *more harshly* than Sonnet does, by 2 absolute points. Both judges land near 67% — the honest baseline.

### Per-qtype agreement

| qtype | n | Kimi-self | Sonnet | Δ (Kimi − Sonnet) |
|---|---:|---:|---:|---:|
| single-session-assistant | 11 | 1.000 | 1.000 | 0.000 |
| single-session-user | 14 | 0.857 | 0.786 | **+0.071** (Kimi-lenient) |
| multi-session | 27 | 0.556 | 0.630 | **−0.074** (Sonnet-lenient) |
| knowledge-update | 15 | 0.667 | 0.667 | 0.000 |
| temporal-reasoning | 27 | 0.667 | 0.630 | **+0.037** (Kimi-lenient) |
| single-session-preference | 6 | 0.000 | 0.333 | **−0.333** (Sonnet-lenient; n=6 noisy) |

The qtype split is bidirectional: Kimi is more permissive on factual / temporal questions, Sonnet is more permissive on synthesis-heavy (preference / multi-session) questions. Net effect across qtypes cancels to a 2-pt aggregate gap.

### Judge agreement matrix (n=100)

| | | |
|---|---:|---|
| Both PASS | 64 | — |
| Both FAIL | 30 | — |
| Kimi PASS / Sonnet FAIL | **2** | Kimi-lenient (`f4f1d8a4_abs`, `gpt4_4929293b`) |
| Kimi FAIL / Sonnet PASS | **4** | Sonnet-lenient (`1c0ddc50`, `ef9cf60a`, `a89d7624`, `2318644b`) |
| **Agreement** | **94/100 = 94%** | — |

### Disagreement mechanism

The 6 disagreements concentrate exactly where the LongMemEval rubric is most subjective:

**Sonnet-lenient (4 cases) — preference / multi-session synthesis:**
- `1c0ddc50` (sss-pref) — Engram lists podcasts/audiobooks; gold rubric says "user would prefer suggestions related to listening to new podcasts or audiobooks". Sonnet accepts as covering the rubric; Kimi rejects.
- `a89d7624` (sss-pref) — Engram lists Denver attractions; gold expects "responses that take into account prior Denver experience". Sonnet accepts partial coverage; Kimi rejects.
- `ef9cf60a` (multi-session) — Engram says "$300" with verbose CoT prefix; Sonnet extracts the answer, Kimi rejects on form.
- `2318644b` (multi-session) — Engram says "Over $270 more per night"; gold is "$270". Sonnet treats "over X" as equivalent; Kimi rejects.

**Kimi-lenient (2 cases) — abstain / temporal:**
- `f4f1d8a4_abs` (sss-user abstain) — Engram says "I don't know"; gold expects the long-form "you didn't mention X but you mentioned Y". Kimi accepts the bare refusal; Sonnet rejects.
- `gpt4_4929293b` (temporal) — Engram's response is a CoT preamble that never reaches a concrete answer. Kimi accepts; Sonnet rejects.

The official rubric says "partial coverage is acceptable" for sss-preference and "do not penalize off-by-one" for temporal. Sonnet enforces those rubric clauses literally; Kimi reads them more conservatively on synthesis and more liberally on form.

### What this means for the v0.1.0 71.4% number

The v0.1.0 release manifest (n=500, k=10, Kimi-self) scored 71.4%. This experiment isolates two confounders against that:

1. **Judge bias**: ±2 pts maximum on this sample, direction unclear. The "3-7 pt inflation" caveat in SCOREBOARD overstated the contaminant. The honest cross-family number on the same sample (n=100, k=20, autotemp+surface-conflicts) is 68%.
2. **Sample distribution**: the stratified n=100 is materially harder than the leading 500q at v0.1.0 settings. Both judges land near 67% on this 100q, vs 71.4% on the full 500. Difference ~4 pts.

So the v0.1.0 71.4% is *probably real* — neither a self-judge mirage nor reproducible by re-running with stricter scoring. It came from a different sample distribution where the leading-N slice happened to be easier.

### Implications for the SOTA push

- **Honest baseline is 67%** (averaged across judges on the n=100 sample). Not 64-68% as previously estimated.
- **The defensible SOTA bar (75%) is +8 pts away**, not +12. Reachable with abstain-prompt fix (+5-7) plus one Tier A lever (+1-3).
- **The "crushing SOTA / paper-worthy" 80% bar is +13 pts away**. Reachable with abstain + per-qtype prompts + consolidation, *without* an answer-model upgrade.
- **Abstain-prompt fix is the cheapest +5-7pt lever in the project.** 8 of 32 Sonnet-judged fails (25%) are `_abs` questions where Engram refused correctly but the judge wanted the richer pattern. One prompt edit, ~$0.25 to re-evaluate.
- **The judge-disagreement pattern argues for per-qtype answer prompts.** If the answer prompt told Kimi to (a) cover preference rubrics with explicit alignment, (b) give concrete numbers on multi-hop math without CoT preamble, both judges would agree more often and on more questions.
- **Consolidation remains the load-bearing experiment.** The 67% baseline is anchored; any consolidation lift on the same n=100 with the same cached answers/judges becomes attributable.

### Cost / wall time

| Run | OpenRouter spend | OpenCode spend | Wall time |
|---|---:|---:|---:|
| Sonnet judge (first run, cold cache) | ~$0.20 | ~$0.05 (Kimi answers) | ~30-45 min (gpu_concurrency=1 serialized 100 ingest+rerank ops) |
| Kimi-self judge (second run, cache warm) | $0 | ~$0.05 (Kimi judges, answers cached) | ~20-25 min (judge phase dominated) |

OpenRouter budget remaining after both runs: **~$2.80 of $3**.

The cache machinery (`benchmarks/runs/cache.sqlite`) earned its keep here: the second run did zero embedding work, zero Kimi answer work, and only paid for 100 fresh judge calls.

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
| `7c3da59` | docs: add JOURNEY.md — session record of the engrampy v0.2 push |
| `5bd4b3f` | agent: verify retry symmetry — no double-reinforce, vote on retry (M-47, M-187) |
| `7214504` | memory: protect _USER_STATE_FLAG from caller metadata overlay (M-181, M-188) |
| `7a518e5` | verify: document the locally-scoped last_response in verify loop (M-196) |
| `61ce3d1` | docs(security): rewrite SECURITY.md as full threat model |
| `ad894b3` | memory: serial-fallback _parallel_leaf_retrieves on :memory: (M-25) |
| `093b8fc` | docs: extend JOURNEY.md with k=20 + rerank-off breakthrough |
| `3327a12` | docs(JOURNEY): catalog audit findings affecting SOTA claims |
| `cebf6dc` | docs(JOURNEY): reframe SOTA gap as 4:1 LLM-stage vs retrieval-stage |
| `9130085` | bench: parallel eval + stratified sample + honest scoring (H-76/77/78/80) |
| `815b953` | gpu: process-wide concurrency cap to decouple --parallel from VRAM |

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
- `src/engram/_seed.py` — `seed_everything(N)` helper that seeds Python random + PYTHONHASHSEED + numpy + torch + torch.cuda + transformers (H-80 / commit `9130085`).
- `src/engram/_gpu_lock.py` — process-wide `BoundedSemaphore` decoupling `--parallel` from VRAM; `gpu_section()` context manager wraps all torch CUDA forward passes (commit `815b953`).
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
- `src/engram/bench/_cli.py` — 14 Phase E/F CLI flags plus `--parallel`, `--sample`, `--gpu-concurrency`, `--aconsolidate-concurrency` (commits `9130085` + `815b953`).
- `src/engram/bench/_runner.py` — manifest now captures the resolved `suite_config` as `engram_config` via `_serialize_for_manifest` (H-76 / commit `9130085`).
- `src/engram/providers/local.py` — `LocalEmbedder._encode` wrapped with `gpu_section()` to enforce the process-wide CUDA semaphore (commit `815b953`).
- `src/engram/retrieve/_bge_reranker.py` — `BGEReranker.rerank` wrapped with `gpu_section()` (commit `815b953`).
- `benchmarks/suites/longmemeval.py` — Phase E flags wired, exception isolation refactor (`_run_one_question`), `_build_auto_temporal_filter`, `_parse_haystack_date`, `_ingest_haystack` preserves `session_id`, `has_answer`, `turn_index`, `is_first_turn`, `is_last_turn`, `session_idx`, `session_n_turns`, `role`. `_parse_judge_verdict` matches official LongMemEval scorer (H-78). `accuracy_correct` split from `accuracy` (H-77). `seed_everything()` plumbed for full RNG determinism (H-80). `_QuestionOutcome` dataclass + `_Progress` thread-safe counter + `ThreadPoolExecutor` parallel path. `_stratified_sample` with deterministic per-qtype proportional allocation.
- `pyproject.toml` — name `engram-memory` → `engrampy`, version `0.1.0` → `0.2.1`, dual-entry authors.
- `src/engram/__init__.py` — `__version__` bumped to `0.2.1`.
- `README.md` — `pip install engrampy` + squat note.
- `CHANGELOG.md` — v0.2.0 and v0.2.1 entries.

---

## Appendix D — Headline numbers

### Retrieval correctness (n=500, baseline + reranker, fp32)

| Metric | k=10 (was) | **k=20 (now)** |
|---|---|---|
| Session-level recall | 0.966 | **0.986** |
| Event-level recall (over all 500) | 0.841 | **0.898** |
| Event-level recall (over 479 evaluable) | 0.878 | **0.937** |
| Full event-level recall rate | 76.4% | **84.6%** |
| Partial event-level recall rate | 14.8% | 9.0% |
| Zero event-level recall rate (when gold exists) | 4.8% | **2.3%** |
| Per-qtype event-level recall floor | sss-pref 0.733 | **sss-pref 0.878** |
| Per-qtype event-level recall ceiling | sss-asst 0.982 | sss-asst 0.982 |

### Rerank impact (norerank vs baseline, k=10, n=500)

| Metric | baseline + reranker | norerank | Δ |
|---|---|---|---|
| Event-level recall (all 500) | 0.841 | 0.796 | **−0.045** |
| Full recall count | 382 | 353 | −29 |
| Zero recall count | 44 | 56 | +12 |
| Zero-recall regressions added | n/a | 22 | n/a |

Reranker is decisively net positive.

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

### Failure mode breakdown (zero-recall questions, full n=23)

| Mode (post-reclassification) | Count | % |
|---|---:|---:|
| within-session ranking failure | 12 | 52% |
| true distractor session | 11 | 48% |
| — of which rerank pushed gold out | 3 | — |
| — of which gold at deep dense rank | 1 | — |
| — of which complete session miss | 2 | — |

### k=20 recovery (of the 23 baseline zero-recall failures)

| Outcome | Count |
|---|---:|
| Full recovery (event recall went to 1.0) | 5 |
| Partial recovery (event recall went to (0, 1)) | 7 |
| Still zero at k=20 | 11 |
| New zero-recall introduced | **0** |

### End-to-end gap decomposition (n=500, v0.1.0 release manifest, k=10)

| Cell | n | % |
|---|---:|---:|
| retrieve ≥1 gold ∧ LLM correct | 343 | 68.6% |
| retrieve ≥1 gold ∧ LLM wrong (LLM-stage loss) | **113** | **22.6%** |
| retrieve miss all gold ∧ LLM correct (lucky) | 14 | 2.8% |
| retrieve miss all gold ∧ LLM wrong (retrieval-stage loss) | **30** | **6.0%** |
| **Total** | **500** | **100.0%** |
| End-to-end accuracy | 357 | 71.4% |

LLM-stage loss / retrieval-stage loss ratio = **22.6 / 6.0 ≈ 3.77** — the SOTA gap is ~4:1 toward LLM-stage improvements.

### The 81 hard-wall failures (perfect retrieval, LLM still failed)

| Failure shape | Count |
|---|---:|
| "I don't know" with answer in prompt | 52 (64%) |
| Confidently wrong number / value | 19 |
| Partial / off-by-one | 10 |

| qtype | hard-wall count | mechanism needed |
|---|---:|---|
| multi-session | 24 | arithmetic / rollup |
| temporal-reasoning | 21 | date arithmetic |
| knowledge-update | 19 | latest-value resolution |
| sss-preference | 10 | preference synthesis |
| sss-user / assistant | 7 | refusal under evidence |
| **total** | **81** | (mostly consolidation-shaped) |

### Conditional accuracy by retrieval state

| Retrieval state | n | Conditional accuracy |
|---|---:|---:|
| Perfect (event_recall = 1.0) | 382 | (382 − 81) / 382 = **78.8%** |
| Hits at least one gold session | 456 | 343 / 456 = 75.2% |
| Misses all gold sessions | 44 | 14 / 44 = 31.8% (lucky / world-knowledge) |

### k=20 → end-to-end translation (the honest number)

| | Value |
|---|---:|
| Retrieval-stage losses at k=10 | 30 |
| Of those, recovered at k=20 (event_hit 0 → ≥1) | 5 |
| Theoretical end-to-end ceiling from k=20 | ~30 × 0.75 = +4.4 pts |
| Realistic end-to-end lift | **+1-2 pts** |

The earlier "predicted 0.78-0.83 end-to-end at k=20" was retracted.

### Three contaminants on the 71.4% headline

| Contaminant | Estimated Impact (pre-§23) | Measured Impact (post-§23) |
|---|---|---|
| Same-model judge (Kimi answers, Kimi judges) | +3-7 pts inflation | **±2 pts, direction unclear** — Kimi-self at 66.0% vs Sonnet cross-judge at 68.0% on same n=100 sample; 94% verdict agreement |
| H-77 swallowed exceptions counted as score=0 | unknown count of infra failures mis-labeled as wrong | **fixed** in commit `9130085`; `accuracy_correct` over `n_completed` is now reported |
| H-78 judge parser ≠ official LongMemEval | some 93 wrong-scored refusals may be parser-related | **fixed** in commit `9130085`; first-line `^yes\b` / `^no\b` exact match |

Revised estimate of "real" 71.4% under official scoring with cross-family judge: **~67%** on a stratified n=100 sample (k=20 + autotemp + surface-conflicts). The 71.4% v0.1.0 number is *probably real* — the ~4-pt gap is sample distribution (leading-500 was easier than the stratified-100), not self-judge bias or parser drift.

### Honest baseline on n=100 stratified (k=20 + autotemp + surface-conflicts, no consolidation)

| Judge | accuracy_correct | n_errored | OpenRouter spend |
|---|---:|---:|---:|
| Claude Sonnet 4.5 (OpenRouter) | **0.680** | 0 | ~$0.20 |
| Kimi K2.6 (self, opencode-go) | **0.660** | 0 | $0 |
| Agreement | 94/100 | — | — |

This is the new comparison floor for every consolidation / prompt-fix experiment.

### Per-qtype n=100 baseline (Sonnet judge)

| qtype | n | accuracy_correct | v0.1.0 (k=10 Kimi-self n=500) | Δ |
|---|---:|---:|---:|---:|
| single-session-assistant | 11 | 1.000 | 0.946 | +0.054 |
| single-session-user | 14 | 0.786 | 0.843 | −0.057 |
| multi-session | 27 | 0.630 | 0.602 | +0.028 |
| knowledge-update | 15 | 0.667 | 0.692 | −0.025 |
| temporal-reasoning | 27 | 0.630 | 0.722 | −0.092 |
| single-session-preference | 6 | 0.333 | 0.500 | −0.167 (n=6 noisy) |
| **overall** | **100** | **0.680** | 0.714 | −0.034 |

### Path-to-SOTA budget (post-§23)

| Lever | Predicted Δ on honest baseline | Cost | Status |
|---|---:|---:|---|
| Honest baseline (n=100, k=20, autotemp+surface-conflicts) | — | — | **67%** (established §23) |
| + abstain-prompt fix | +5 to +7 | ~$0.25 OR (Sonnet judge re-eval) | next experiment |
| + consolidation on same 100q | +5 to +10 | ~$3-5 OR (Haiku consolidation) | thesis experiment |
| + per-qtype answer prompts | +2 to +4 | ~$0.25 OR (re-eval) | Tier A |
| + Tier B retrieval polish (embedder swap, k=30) | +1 to +3 | ~$0.50 OR | Tier B |
| **Achievable on n=100** | | | **~77-87%** |

The 75% defensible-SOTA bar is reachable with just the abstain fix + one Tier A lever. The 80%+ crushing-SOTA bar is reachable with the full stack, no answer-model upgrade.

---

*Last updated 2026-05-17, after the n=100 honest-judge experiment (§23) disproved the self-preference inflation hypothesis and established 67% as the cross-family-judged baseline. The 71.4% v0.1.0 number is probably real and reflects easier sample distribution, not self-judge bias. The next decisive experiment is abstain-prompt fix + consolidation on the same n=100 sample.*
