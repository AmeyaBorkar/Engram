# Concepts

The mental model behind Engram.

## The hierarchy

Three levels of memory items, each higher than the last:

```
event      one observation, never modified
  ↓ (consolidate)
summary    a cluster of related events, with provenance to each
  ↓ (promote)
abstraction stable, frequently-corroborated summaries
```

Retrieval reads across all three; specific events surface when an abstraction's confidence is low or when the caller passes `prefer="specific"`.

## Decay

Every memory item carries a `weight ∈ [0, 1]` that evolves over time:

$$
w_{t+1} = w_t \cdot \alpha^{\Delta t} + \beta \cdot r_t + \gamma \cdot c_t - \delta \cdot x_t
$$

- $r_t$ — reinforcement (was this surfaced and useful?)
- $c_t$ — corroboration (other evidence agrees)
- $x_t$ — contradiction (something disagreed)
- $\alpha, \beta, \gamma, \delta$ — tunable rates on `DecayParams`

Items below a threshold move to a `cold` table (auditable) or are deleted, per the configured `prune_policy`.

## Provenance

Anything above the event level keeps a non-empty `supported_by` link to the events that justified it. Storage enforces this with a CHECK constraint: a non-event memory item without provenance can't be inserted.

## Procedural memory

Agents observe `(situation, action, outcome)` triples. The outcome maps onto the same decay engine:

| Outcome | Decay signal |
|---|---|
| `SUCCESS` | reinforce |
| `PARTIAL` | reinforce |
| `FAILURE` | contradict |
| `UNKNOWN` | no-op |

Retrieval ranks by `similarity * weight * outcome_boost`, so failures stay visible (the agent benefits from "this didn't work" lessons) but successes outrank them at equal similarity.

## Temporal validity

Stage 8 adds time bounds on every memory item:

- `valid_from` (defaults to `created_at`)
- `valid_until` (None = still current)
- `invalidated_at` / `invalidated_by` (set when reconcile picks the other side)

A query at time $t$ sees an item iff:
$$\text{valid\_from} \le t < \text{valid\_until} \text{ AND } t < \text{invalidated\_at}$$

(missing bounds count as $+\infty$.)

This lets you answer "as of when?" queries:

```python
results = memory.retrieve("what was X?", as_of=datetime(2026, 3, 1, tzinfo=timezone.utc))
```

## Conflicts

When two memory items contradict, Stage 5's detector creates a `Conflict` row (status=OPEN). The Stage 8 reconciler resolves it: it picks a winner per the chosen `Resolution` policy, invalidates the loser, and marks the conflict RESOLVED.

This is what makes "memory that changes over time" work — old beliefs aren't deleted, they're versioned.

## Providers

Engram doesn't know about specific LLMs. Two protocols:

- `EmbeddingProvider.embed(texts) -> list[list[float]]`
- `ChatProvider.chat(messages) -> str`

Real adapters live in `engram.providers.openai`, `engram.providers.anthropic`, and the moonshot / OpenCode Zen variants used in the benchmarks. Tests use `FakeEmbedder` + `FakeChat`.

Cross-cutting wrappers (`Retry`, `Cache`, `Batcher`, `Redactor`) sit between the user and the adapter and are configurable per-provider.

## Storage

`Storage` is a protocol. SQLite is the only implementation through v0.3.x; Postgres lands in v0.4.0 against the same protocol with `pgvector` and RLS. Switching backends doesn't require code changes in the consumer.
