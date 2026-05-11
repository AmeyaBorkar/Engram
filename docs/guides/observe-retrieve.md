# Observe and retrieve

The two primitives.

## `observe`

Record an event:

```python
event = memory.observe("user mentioned they ship to us-east-1")
```

You can also pre-build the Event for explicit control:

```python
from engram import Event

event = Event(
    content="user mentioned they ship to us-east-1",
    source="conversation-2026-05-12",
    metadata={"channel": "slack", "thread_id": "t123"},
)
memory.observe(event)
```

The Event lands in storage atomically with its embedding. The embedding model is whatever you passed to `Memory(embedder=...)`.

## `retrieve`

Coarse-to-fine over the hierarchy:

```python
results = memory.retrieve("where does the user deploy?", k=5)
for r in results:
    print(r.level, r.score, r.content)
```

The default behavior is `prefer="auto"`:

- High-confidence abstractions surface as-is.
- Low-confidence ones drill into their supporting events.

Override:

- `prefer="general"` — return abstractions/summaries only.
- `prefer="specific"` — return raw events only.

## Temporal queries

```python
from datetime import datetime, timezone

# What did Engram believe at 2026-03-01?
past = memory.retrieve(
    "where does the user deploy?",
    k=5,
    as_of=datetime(2026, 3, 1, tzinfo=timezone.utc),
)
```

`as_of=None` (the default) returns current state. Items invalidated via `reconcile` are excluded; `as_of=<datetime>` returns items whose validity window covers that timestamp.

## Reranking

Pass an optional reranker:

```python
from engram.retrieve import FakeReranker  # for tests; real reranker via providers

results = memory.retrieve("query", k=5, reranker=FakeReranker())
```
