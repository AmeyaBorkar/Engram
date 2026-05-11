# Performance

Per-call latency budgets pinned on the hot paths. CI gates regression on the slow lane.

## Budgets (laptop-grade)

| Operation | Workload | P50 budget | P99 budget |
|---|---|---|---|
| `Memory.observe` | single write | < 50 ms | — |
| `Memory.retrieve` | 10k events | < 100 ms | — |
| `Memory.retrieve` (warm cache, 100k items) | dim=128 | < 150 ms | < 500 ms |
| `Memory.retrieve` (cold cache rebuild) | 100k items | < 1500 ms | — |
| `search_memory_item_embeddings_as_of` | 100k items, 1% invalidated | < 225 ms | < 750 ms |
| `Memory.reconcile` (non-MERGE) | — | < 25 ms | < 100 ms |
| `Memory.list_conflicts` | 5k pairs | < 10 ms | < 50 ms |

All from `tests/test_*_perf.py`, marked `@pytest.mark.slow`. CI runs the slow lane on tagged releases.

## Vector index

The in-memory vector index caches the embedding matrix per `(item_kind, model)`. Lazy build on first search; dirty flag on every write. Once warm, similarity is a single numpy matmul; cold-cache rebuilds cost ~1 s at 100k items.

`sqlite-vec` integration is opt-in via the `[sqlite-vec]` extra (Stage 6); when installed it replaces the numpy matmul with native vector index queries. The default numpy path stays the reference.

## Decay tick

The decay sweep streams every hot item in batches of 1k (configurable). At 1M items the tick takes ~few seconds on a laptop — well within the cadence callers typically run it (every minutes to hours, depending on the application).

## Provider batching

Embedding and chat calls go through a configurable `Batcher` (default: 20 ms window, 64-item ceiling). The Stage 2 DoD asserts a ≥ 5× call-count reduction on the smoke benchmark; in practice we see 10-50× depending on the access pattern.

## Microbenchmarks

Run the slow lane:

```bash
pytest -m slow
```

Run a specific perf test:

```bash
pytest tests/test_stage8_perf.py -v
```
