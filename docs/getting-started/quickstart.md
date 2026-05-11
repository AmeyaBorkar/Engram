# Quickstart

A 10-minute tour of the public API.

## 1. Construct a Memory

```python
from engram import Memory, SqliteStorage
from engram.providers import OpenAIEmbedder, OpenAIChat

memory = Memory(
    storage=SqliteStorage("engram.db"),
    embedder=OpenAIEmbedder(),
    chat=OpenAIChat(),  # optional; required for consolidate / reconcile-MERGE
)
```

For tests, swap to deterministic fakes:

```python
from engram.providers._fake import FakeEmbedder, FakeChat

memory = Memory(
    storage=SqliteStorage(":memory:"),
    embedder=FakeEmbedder(dim=128),
    chat=FakeChat(default="canned reply"),
)
```

## 2. Observe events

```python
event = memory.observe("user mentioned they have a golden retriever named Rex")
print(event.id, event.created_at)
```

`observe` accepts a string (wraps into an `Event` with defaults) or a fully-formed `Event`. The event lands in storage atomically with its embedding.

## 3. Retrieve

Coarse-to-fine over the hierarchy:

```python
results = memory.retrieve("what pet does the user have?", k=5)
for r in results:
    print(r.level, r.score, r.content)
```

Stage 8 layers in temporal validity:

```python
from datetime import datetime, timezone

# State as of 2026-03-01:
past = memory.retrieve(
    "what pet does the user have?",
    k=5,
    as_of=datetime(2026, 3, 1, tzinfo=timezone.utc),
)
```

## 4. Consolidate

Cluster recent events, extract abstractions:

```python
result = memory.consolidate(max_events=100)
print(result.clusters_formed, result.abstractions_created)
```

## 5. Procedural memory

Agents learn from doing:

```python
from engram import Outcome

# Record a procedure with its outcome.
memory.record_procedure(
    "user reports flaky integration test",
    "rerun with --no-cov and bisect to isolate",
    outcome=Outcome.SUCCESS,
)

# Look up analogous past procedures.
matches = memory.retrieve_procedures(
    "flaky CI test keeps failing intermittently",
    k=3,
)
```

## 6. Reconcile contradictions

When two memory items disagree, choose a policy:

```python
from engram import Resolution, ConflictStatus

# Find open conflicts (Stage 5 records them during consolidation).
for conflict in memory.list_conflicts(status=ConflictStatus.OPEN):
    # Resolve with the policy that fits your trust model.
    memory.reconcile(conflict.id, resolution=Resolution.PREFER_RECENT)
```

Available policies:

- `PREFER_RECENT` — newer item wins
- `PREFER_TRUSTED` — higher `source_trust` wins
- `PREFER_FREQUENT` — higher corroboration count wins
- `KEEP_BOTH` — neither invalidated
- `MANUAL` — caller picks via `manual_winner_id`
- `MERGE` — LLM synthesizes a new item from both; both originals invalidated

## 7. Use the agent wrapper

```python
from engram.integrations import EngramAgent

agent = EngramAgent(memory, chat=memory.chat)  # type: ignore[attr-defined]
turn = agent.chat("Should I rerun the flaky test the same way as last time?")
print(turn.reply)
print(turn.retrieved_context)
```

## 8. Async surface

For event-loop callers:

```python
event = await memory.aobserve("x")
results = await memory.aretrieve("x", k=5)
```

Every public sync method has an `async def` parallel: `aobserve`, `aretrieve`, `aconsolidate`, `areconcile`, etc.
