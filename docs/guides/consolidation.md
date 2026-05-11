# Consolidation

Cluster recent events, extract abstractions, link via provenance.

## Run a consolidation pass

```python
result = memory.consolidate(max_events=100)
print(f"clusters: {result.clusters_formed}")
print(f"abstractions: {result.abstractions_created}")
print(f"conflicts: {result.conflicts_detected}")
```

The pipeline:

1. Pull up to `max_events` unconsolidated events from storage.
2. Cluster the embeddings (HDBSCAN by default; agglomerative fallback for small N).
3. Per cluster: call the chat provider to produce a generalization; parse strictly.
4. Atomically insert the resulting `MemoryItem` + provenance links.
5. Optionally run the contradiction detector against existing abstractions.

## Promotion

Stable summaries get promoted to abstractions:

```python
promotion = memory.promote()
print(f"promoted: {promotion.promoted}")
```

The promotion gate requires:

- `corroboration_count ≥ min_corroboration` (default 3)
- `contradiction_count ≤ max_contradiction` (default 0)
- `weight ≥ min_weight` (default 0.5)
- No recorded conflicts in metadata

Enable it via:

```python
from engram.consolidation import ConsolidationParams, PromotionParams

memory = Memory(
    storage=...,
    embedder=...,
    chat=...,
    consolidation_params=ConsolidationParams(
        promotion_params=PromotionParams(enabled=True),
    ),
)
```

## Contradiction detection

Off by default. Enable when you want the detector to flag conflicts during consolidation:

```python
from engram.consolidation import ContradictionParams

memory = Memory(
    storage=...,
    embedder=...,
    chat=...,
    consolidation_params=ConsolidationParams(
        contradiction_params=ContradictionParams(
            enabled=True,
            similarity_threshold=0.7,
            max_candidates=3,
        ),
    ),
)
```

The judge LLM classifies pairs as `agree` / `contradict` / `unrelated`. `contradict` verdicts get written both to the new memory item's metadata blob (back-compat) AND to a first-class `Conflict` row (Stage 8) with `status=OPEN`.

Resolve those conflicts via `Memory.reconcile`. See [Contradiction & temporal](contradiction-temporal.md).
