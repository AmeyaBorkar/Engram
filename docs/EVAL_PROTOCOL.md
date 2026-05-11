# Evaluation protocol

This document is the standard we hold every retrieval-side claim to.
If a feature lands in a SOTA-claiming bench run, it has passed this
protocol against the version of the codebase we ran.

## The bar

Every retrieval-side feature must satisfy:

1. **Contract** — a one-sentence statement of what the feature claims
   to improve, on which question types, and at what cost.
2. **Isolated evaluation** — the feature is benchmarked alone against
   the baseline (everything else off) on the full LongMemEval-S set
   (500 questions).
3. **Hyperparameter sweep** — every knob is swept across a grid; the
   best-on-dev setting is recorded with confidence intervals.
4. **Statistical significance** — overall recall@10 improvement vs
   baseline has a bootstrap 95% CI that excludes zero, AND a McNemar
   exact test (p < 0.05) on the per-question hit@10 pass/fail
   matrix.
5. **Per-type honesty** — the feature reports per-qtype impact; a
   feature that improves overall by hurting one bucket >5% is called
   out, not hidden.
6. **Failure mode analysis** — for every config that fails the bar,
   the script prints the questions the feature broke (passed in
   baseline, failed under the feature) so we can diagnose.

## Per-component contracts (current state)

### BM25 hybrid retrieval

- **Hypothesis**: literal-token overlap (years, codes, names) is
  preserved by BM25 that the embedder smooths away. Helps queries
  with proper nouns / dates / specific entities.
- **Knobs**: `bm25_weight` ∈ {0, 0.5, 1.0, 1.5, 2.0}; `bm25_k1` ∈
  {1.0, 1.5, 2.0}; `bm25_b` ∈ {0.5, 0.75, 0.9}.
- **Primary metric**: overall recall@10. Secondary: temporal-reasoning,
  knowledge-update recall@10.
- **Decision rule**: ship the best `(bm25_weight, k1, b)` if overall
  recall@10 lift CI excludes 0 AND multi-session doesn't drop > 0.02.
- **Cost**: O(corpus_size) build + O(query_terms × hits) query.
- **When NOT to use**: corpora with stop-word-heavy queries; tasks
  where literal tokens are irrelevant.

### MMR diversity rerank

- **Hypothesis**: when multiple top-ranked candidates are near-duplicates,
  greedy MMR replaces some with diverse but still-relevant items,
  improving the LLM's evidence coverage.
- **Knobs**: `mmr_lambda` ∈ {0.3, 0.5, 0.7, 0.9}; `mmr_pool_size` ∈
  {k×3, k×5, k×10}.
- **Primary metric**: multi_recall@10 (proxy for evidence coverage).
- **Decision rule**: ship if multi_recall@10 lift CI excludes 0 AND
  overall hit@10 doesn't drop > 0.01.
- **Fixed May 2026**: relevance scores are now min-max normalized to
  [0, 1] inside `mmr_select` so the diversity term (cosine in [0, 1])
  has equal weight to relevance. Previously the wide-range
  cross-encoder logits dwarfed the diversity penalty.
- **When NOT to use**: queries that legitimately need multiple
  same-source supporting turns (multi-session over one fact).

### Recency boost

- **Hypothesis**: recently-stated information should rank above
  older mentions when the question implies "now" / "current" / "lately".
- **Knobs**: `recency_lambda` ∈ {0, 0.05, 0.1, 0.2}; `recency_decay_days`
  ∈ {30, 60, 90, 180}.
- **Primary metric**: knowledge-update recall@10; temporal-reasoning
  recall@10. Secondary: overall recall@10.
- **Decision rule**: ship if knowledge-update lift CI excludes 0
  AND overall hit@10 doesn't drop.
- **Fixed May 2026**: boost is now additive (`score + λ·decay`) so
  it never inverts on negative reranker logits. λ is in the same
  units as the reranker score; sane values are 0.1-0.3 for typical
  BGE-reranker output.
- **When NOT to use**: questions about historical sequences, "first
  time I…" queries.

### Recent-window hybrid stream

- **Hypothesis**: the latest N events are always relevant to "what
  did I last do" queries; injecting them as a parallel RRF stream
  improves knowledge-update recall.
- **Knobs**: `recent_window_k` ∈ {0, 5, 10, 20, 50}.
- **Primary metric**: knowledge-update recall@10.
- **Decision rule**: ship ONLY if multi-session recall@10 doesn't
  drop > 0.02 AND knowledge-update lift CI excludes 0.
- **Confirmed failure (May 2026)**: at `recent_window_k = 10` on
  the v0.1.0 codebase, multi-session recall@10 dropped from ~0.78
  to a projected ~0.45 over 130 questions. **Currently flagged for
  removal from launch configs** pending either:
    - a gate that only activates the stream when the question
      contains a recency cue (`lately|recent|now|last|latest`), or
    - per-qtype weighting (off by default for multi-session).

### Auto-temporal lexical filter

- **Hypothesis**: questions naming a year are answered by events
  from that year. A surgical lexical filter improves precision
  without recall loss when fallback triggers on empty pool.
- **Knobs**: filter pattern (regex); fallback policy.
- **Primary metric**: temporal-reasoning recall@10.
- **Decision rule**: ship if temporal-reasoning lift CI excludes 0
  AND overall recall@10 doesn't drop AND the fallback rate is
  reported alongside (so we know how often we're filtering).
- **Known issue (May 2026)**: fallback only fires on EMPTY pool.
  If the filter keeps some-but-wrong events, no fallback. Worth
  monitoring fallback rate per qtype.

### Cross-encoder rerank

- **Hypothesis**: cross-encoder relevance is more accurate than
  dense cosine for the top-k step.
- **Knobs**: model id, `candidate_multiplier`, dtype.
- **Primary metric**: MRR (catches "found but ranked deep" failures).
- **Decision rule**: ship if MRR lift CI excludes 0 AND overall
  recall@10 doesn't drop.

### Hierarchical retrieval (consolidation + drill)

- **Hypothesis**: consolidated abstractions surface high-level
  answers efficiently; drill recovers when confidence is low.
- **Knobs**: `confidence_threshold`, `drill_k`, `prefer`.
- **Primary metric**: overall recall@10 with consolidation on.
- **Decision rule**: ship if overall lift CI excludes 0 (the
  consolidation work has to actually help; otherwise we're paying
  for nothing).
- **Cost**: 1 LLM call per cluster at consolidation time.

## Standard sweep grids

| Knob | Grid |
|---|---|
| `bm25_weight` | 0, 0.5, 1.0, 1.5, 2.0 |
| `bm25_k1` | 1.0, 1.5, 2.0 |
| `bm25_b` | 0.5, 0.75, 0.9 |
| `mmr_lambda` | 0.3, 0.5, 0.7, 0.9 |
| `mmr_pool_size` | 30, 50, 100 |
| `recency_lambda` | 0, 0.05, 0.1, 0.2 |
| `recency_decay_days` | 30, 60, 90, 180 |
| `recent_window_k` | 0, 5, 10, 20 |
| `candidate_multiplier` | 3, 5, 10 |
| `k` (top-k) | 5, 10, 20 |

## Statistical tests

### Bootstrap 95% CI

For each metric M reported on N questions:
- Resample the N per-question values with replacement, 10,000 times.
- For each resample, compute the mean.
- Report `[mean, mean - p2.5, mean + p97.5]`.
- Lift vs baseline = difference of means; lift CI excludes 0 → the
  feature is statistically improving.

### McNemar's exact test

For pairwise hit@10 (binary pass/fail) over the same N questions:
- Build the 2x2 table:
    - AA: pass on both
    - AB: pass on A only
    - BA: pass on B only
    - BB: fail on both
- Test stat = (|AB - BA| - 1)² / (AB + BA) [Yates-corrected],
  distributed as chi-square with 1 df under H0.
- Exact binomial for small `(AB + BA)`.

Both tests must pass for a "ship it" decision.

## Evaluation order (recommended)

For each new feature:

1. **Build the contract entry** in this file before writing code.
2. **Sweep alone vs baseline** — `scripts/sweep.py --knob X`.
3. **Confirm the best setting** vs baseline with the eval rules above.
4. **Pairwise interaction** — add to the conservative baseline; check
   no regression.
5. **Full stack interaction** — include in the proposed launch config;
   re-sweep one more time.
6. **Failure mode analysis** — list the questions the feature broke
   in step 3. Decide whether to gate, drop, or accept the trade.
7. **Commit** — the manifest + the diff + the failure list go in.

## Failure mode logging

Every component evaluation MUST output:

| Required output | Where it goes |
|---|---|
| Per-question pass/fail diff vs baseline | JSON manifest `diff_vs_baseline` field |
| List of qids the feature broke | Markdown summary `### broken` section |
| Fallback / activation rates (auto-temporal, recent-window) | Per-config stats panel |
| Bootstrap CIs on all aggregates | Markdown summary `(±)` |
| McNemar p-value vs baseline | Markdown summary `(p=)` |

## Reproducibility checklist

- Set `--seed 1337` (or matching value).
- Pin embedder + reranker + dtype.
- Record git commit hash in the manifest.
- Record provider names + model ids (provider-side model drift is the
  one non-deterministic axis we can't pin; record it for forensics).
- For end-to-end LLM runs, also record the OpenCode-Zen / Moonshot /
  OpenRouter provider routing flags.

## What this protocol does NOT cover (yet)

- **K-fold cross-validation**: LongMemEval-S is a single set. We
  evaluate on the full set every time; no held-out test. This is a
  weakness of our pipeline, not the protocol. When LongMemEval-M or
  -L lands, we move to dev/test splits.
- **Multiple seeds**: retrieval is deterministic given (embedder,
  dtype, params). The only stochastic axis is LLM sampling (verify /
  self-consistency / answer with temperature > 0). For those, run
  3-5 seeds per config and report mean ± std.
- **Cost accounting**: token costs, GPU-seconds. Tracked separately.
