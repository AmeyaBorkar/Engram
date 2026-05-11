# Decay

Memory items have a weight in $[0, 1]$ that strengthens with use and weakens with time.

## The formula

$$
w_{t+1} = w_t \cdot \alpha^{\Delta t} + \beta \cdot r_t + \gamma \cdot c_t - \delta \cdot x_t
$$

- $r_t$ — reinforcement (was this surfaced and useful?)
- $c_t$ — corroboration (other evidence agrees)
- $x_t$ — contradiction (something disagreed)
- $\alpha$, $\beta$, $\gamma$, $\delta$ — tunable rates on `DecayParams`

The signals are tracked separately so the engine can attribute weight changes to specific causes.

## Tunable parameters

```python
from engram import DecayParams, Memory

memory = Memory(
    storage=...,
    embedder=...,
    decay_params=DecayParams(
        alpha=0.99,    # base decay per day
        beta=0.05,     # reinforcement
        gamma=0.03,    # corroboration
        delta=0.10,    # contradiction
        threshold=0.1, # below this -> cold
    ),
)
```

## Tick

The decay tick applies the formula across every hot item:

```python
result = memory.tick()
print(f"processed: {result.processed}, cold: {result.newly_cold}")
```

Or async:

```python
result = await memory.tick_async()
```

Typically run on a schedule: every minute for interactive systems, every hour for batch.

## Cold + prune policies

When weight drops below the threshold, the item moves to a cold state:

- `prune_policy="cold"` (default): item stays in storage with `cold_at` set; retrieval excludes it.
- `prune_policy="delete"`: item is hard-deleted. Events with provenance links can't be deleted (storage refuses) — use `cold` if you need to prune events.

## Manual signals

```python
from engram import ItemKind

memory.reinforce(item_id, ItemKind.EVENT)
memory.corroborate(item_id, ItemKind.MEMORY_ITEM)
memory.contradict(item_id, ItemKind.MEMORY_ITEM)
```

These methods apply the decay formula since the last update PLUS the fresh signal. They return the post-update `DecayState`.

## Property invariants

The Hypothesis test suite pins:

- Weight stays in $[0, 1]$ across every signal sequence.
- Reinforcement strictly raises weight.
- Decay is monotonic without reinforcement.
- Corroboration count is non-decreasing.
- Replayability: same event stream + same clock → bit-identical weights.

100% coverage on the decay math, by policy.
