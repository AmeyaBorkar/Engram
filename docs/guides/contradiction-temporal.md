# Contradiction and temporal reasoning

Stage 8. The headline: facts change, contradictions get observed and resolved, and time-travel queries return historically-correct state.

## Detecting contradictions

The Stage 5 contradiction detector (opt-in) flags pairs of memory items that disagree. Verdicts come from an LLM judge:

```python
from engram.consolidation import ConsolidationParams, ContradictionParams

memory = Memory(
    storage=...,
    embedder=...,
    chat=...,
    consolidation_params=ConsolidationParams(
        contradiction_params=ContradictionParams(
            enabled=True,
            similarity_threshold=0.7,
        ),
    ),
)
```

When the detector classifies a pair as `contradict`, a `Conflict` row is written with `status=OPEN`.

## Listing conflicts

```python
from engram import ConflictStatus

for c in memory.list_conflicts(status=ConflictStatus.OPEN):
    src = memory.storage.get_memory_item(c.source_item_id)
    tgt = memory.storage.get_memory_item(c.target_item_id)
    print(f"conflict {c.id}: {src.content!r} vs {tgt.content!r}")
```

## Resolving a conflict

Pick a policy:

```python
from engram import Resolution

memory.reconcile(c.id, resolution=Resolution.PREFER_RECENT)
```

Available policies:

| Policy | Winner |
|---|---|
| `PREFER_RECENT` | the more recently-created item |
| `PREFER_TRUSTED` | the higher `source_trust` (None treated as 0.0); ties fall to PREFER_RECENT |
| `PREFER_FREQUENT` | the higher corroboration count from decay state; ties fall to PREFER_RECENT |
| `KEEP_BOTH` | no winner; both stay valid. The conflict is marked RESOLVED so audits stop re-surfacing it |
| `MANUAL` | caller specifies via `manual_winner_id` |
| `MERGE` | LLM synthesizes a new memory item with content from both; both originals invalidated, new item carries provenance from both |

The loser gets `invalidated_at` + `invalidated_by` set. Default retrieve drops invalidated items; `as_of=` queries can still surface them.

## Temporal validity

Every `MemoryItem` carries:

- `valid_from` (defaults to `created_at`)
- `valid_until` (None = still current)
- `invalidated_at` / `invalidated_by` (set during reconcile)

Visibility at time $t$:

$$
\text{valid\_from} \le t \quad \text{AND} \quad \text{valid\_until} > t \quad \text{AND} \quad \text{invalidated\_at} > t
$$

(missing bounds count as $+\infty$.)

## Time-travel queries

```python
from datetime import datetime, timezone

# What did Engram believe at 2026-03-01?
past = memory.retrieve(
    "user's deployment region",
    as_of=datetime(2026, 3, 1, tzinfo=timezone.utc),
)

# Current state (excludes invalidated items):
current = memory.retrieve("user's deployment region")
```

Three-version chains work: $v_1 \to v_2 \to v_3$ where each version invalidates the previous. Snapshots at three points in time each return the right version. The Stage 8 adversarial suite at `benchmarks/suites/contradiction_temporal.py` pins this at 100% accuracy on 15 (item, snapshot) pairs.

## Source trust

For the `PREFER_TRUSTED` policy:

```python
memory.storage.set_source_trust(item_id, 0.9)
```

Higher `source_trust` wins. None is treated as 0.0 (untrusted).

You can model this via the `Source` schema:

```python
from engram import Source

vetted_source = Source(name="claims-from-product", trust=0.9)
guess_source = Source(name="user-said-once", trust=0.3)
```

(The Source model is the policy-level concept; storage tracks the float on `MemoryItem.source_trust`.)
