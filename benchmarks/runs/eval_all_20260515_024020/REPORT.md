# Engram Retrieval Evaluation — Full Report

Generated: 2026-05-15 06:03:02 India Standard Time

Output dir: `benchmarks\runs\eval_all_20260515_024020`


## Configuration

```
{
  "out_dir": "benchmarks\\runs\\eval_all_20260515_024020",
  "limit": 30,
  "sweep_limit": 30,
  "k": 10,
  "embed_model": "BAAI/bge-large-en-v1.5",
  "embed_device": "cuda",
  "dtype": "fp32",
  "reranker": "BAAI/bge-reranker-v2-m3",
  "skip_phase": [],
  "resume": false,
  "started_at": "2026-05-15 02:40:21 India Standard Time"
}
```


## Phase 1: Component correctness

Overall: **33/33 passed** (0 failed)

| Component | Passed | Failed |
|---|---:|---:|
| `BM25Index` | 9 | 0 |
| `mmr_select` | 8 | 0 |
| `reciprocal_rank_fusion` | 5 | 0 |
| `_build_auto_temporal_filter` | 5 | 0 |
| `recency_boost_math` | 6 | 0 |

## Phase 2: Per-qtype ablation

### Mean recall@k by (config, qtype)

| config | single-session | single-session | multi-session | knowledge-upda | temporal-reaso |
|---|---:|---:|---:|---:|---:|
| baseline | 1.000 | 0.967 | 0.943 | 1.000 | 0.928 |
| bm25 | 1.000 | 0.967 | 0.910 (-0.033) | 1.000 | 0.947 |
| mmr07 | 0.967 (-0.033) | 0.967 | 0.932 | 1.000 | 0.939 |
| mmr03 | 1.000 | 0.933 (-0.033) | 0.910 (-0.033) | 1.000 | 0.922 |
| recent | 1.000 | 0.967 | 0.924 | 1.000 | 0.928 |
| recency | 0.967 (-0.033) | **0.900** (-0.067) | **0.818** (-0.125) | 0.983 | **0.828** (-0.100) |
| autotemp | 1.000 | 0.967 | 0.943 | 1.000 | 0.928 |
| bm25+aut | 1.000 | 0.967 | 0.910 (-0.033) | 1.000 | 0.947 |
| bm25+mmr | 1.000 | 0.967 | **0.806** (-0.137) | 1.000 | 0.894 (-0.033) |
| bm25+rec | 1.000 | 0.967 | 0.910 (-0.033) | 1.000 | 0.947 |
| all_aggressive | 1.000 | **0.900** (-0.067) | **0.648** (-0.295) | 0.967 (-0.033) | **0.767** (-0.161) |
| conservative | 1.000 | 0.967 | 0.910 (-0.033) | 1.000 | 0.947 |

### Verdict per config (gate: per-qtype regression ≤ 0.02)

| Config | Verdict |
|---|---|
| `all_aggressive` | FAIL (multi-session Δ=-0.295) |
| `autotemp` | NEUTRAL (no lift, no regression) |
| `baseline` | BASELINE |
| `bm25` | FAIL (multi-session Δ=-0.033) |
| `bm25+aut` | FAIL (multi-session Δ=-0.033) |
| `bm25+mmr` | FAIL (multi-session Δ=-0.137) |
| `bm25+rec` | FAIL (multi-session Δ=-0.033) |
| `conservative` | FAIL (multi-session Δ=-0.033) |
| `mmr03` | FAIL (single-session-preference Δ=-0.033) |
| `mmr07` | FAIL (single-session-user Δ=-0.033) |
| `recency` | FAIL (multi-session Δ=-0.125) |
| `recent` | NEUTRAL (no lift, no regression) |

## Phase 3: Hyperparameter sweeps


### `bm25_weight`

Baseline value: `0.0`, n_questions = 30

| value | n | recall@k | hit@k | mrr | Δ recall (paired CI) | sig? |
|---|---:|---:|---:|---:|---|---|
| 0.0 | 30 | 1.000 | 1.000 | 0.970 | (baseline) | — |
| 0.5 | 30 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | no |
| 1.0 | 30 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | no |
| 1.5 | 30 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | no |
| 2.0 | 30 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | no |

### `mmr_lambda`

Baseline value: `0.0`, n_questions = 30

| value | n | recall@k | hit@k | mrr | Δ recall (paired CI) | sig? |
|---|---:|---:|---:|---:|---|---|
| 0.0 | 30 | 1.000 | 1.000 | 0.970 | (baseline) | — |
| 0.3 | 30 | 1.000 | 1.000 | 0.970 | +0.000 [+0.000, +0.000] | no |
| 0.5 | 30 | 0.967 | 0.967 | 0.967 | -0.033 [-0.100, +0.000] | no |
| 0.7 | 30 | 0.967 | 0.967 | 0.967 | -0.033 [-0.100, +0.000] | no |
| 0.9 | 30 | 0.967 | 0.967 | 0.967 | -0.033 [-0.100, +0.000] | no |

### `recency_lambda`

Baseline value: `0.0`, n_questions = 30

| value | n | recall@k | hit@k | mrr | Δ recall (paired CI) | sig? |
|---|---:|---:|---:|---:|---|---|
| 0.0 | 30 | 1.000 | 1.000 | 0.970 | (baseline) | — |
| 0.1 | 30 | 0.967 | 0.967 | 0.967 | -0.033 [-0.100, +0.000] | no |
| 0.3 | 30 | 0.967 | 0.967 | 0.967 | -0.033 [-0.100, +0.000] | no |
| 0.5 | 30 | 0.967 | 0.967 | 0.939 | -0.033 [-0.100, +0.000] | no |
| 1.0 | 30 | 0.933 | 0.933 | 0.933 | -0.067 [-0.167, +0.000] | no |

### `recent_window_k`

Baseline value: `0`, n_questions = 30

| value | n | recall@k | hit@k | mrr | Δ recall (paired CI) | sig? |
|---|---:|---:|---:|---:|---|---|
| 0 | 30 | 1.000 | 1.000 | 0.970 | (baseline) | — |
| 5 | 30 | 1.000 | 1.000 | 0.970 | +0.000 [+0.000, +0.000] | no |
| 10 | 30 | 1.000 | 1.000 | 0.970 | +0.000 [+0.000, +0.000] | no |
| 20 | 30 | 0.967 | 0.967 | 0.967 | -0.033 [-0.100, +0.000] | no |
| 50 | 30 | 0.967 | 0.967 | 0.967 | -0.033 [-0.100, +0.000] | no |

## Phase 4: Diagnostic traces

Per-qtype trace of one question where any config flipped a baseline pass to a failure.

| qtype | qid | trace file |
|---|---|---|
| single-session-user | `5d3d2817` | `traces\single-session-user__5d3d2817.txt` |
| single-session-preference | `06f04340` | `traces\single-session-preference__06f04340.txt` |
| multi-session | `3a704032` | `traces\multi-session__3a704032.txt` |
| knowledge-update | `c4ea545c` | `traces\knowledge-update__c4ea545c.txt` |
| temporal-reasoning | `gpt4_59149c7` | `traces\temporal-reasoning__gpt4_59149c7.txt` |

## Recommended launch config

Configs that pass the per-qtype regression gate (Δ ≥ −0.02 everywhere):

- `autotemp` — NEUTRAL (no lift, no regression)
- `recent` — NEUTRAL (no lift, no regression)

If you want the safest launch: just `baseline` (no flags). If you want
recall-neutral enhancements that might pay off elsewhere (LLM stage): pick a
`NEUTRAL` config and add `--surface-conflicts --auto-temporal` for free.

```powershell
python -m engram.bench run longmemeval `
  --embedder local --embed-model BAAI/bge-large-en-v1.5 --embed-device cuda --dtype fp32 `
  --chat opencode-go --chat-model kimi-k2.6 `
  --reranker bge --k 10 --seed 1337 `
  --rerank-pool-multiplier 5 `
  --auto-temporal --surface-conflicts
```