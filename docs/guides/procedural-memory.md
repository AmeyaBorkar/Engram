# Procedural memory

The agent learns from doing: `(situation, action, outcome)` triples.

## Record

```python
from engram import Outcome

memory.record_procedure(
    "user reports a flaky integration test",
    "rerun with --no-cov and bisect to isolate the failing assertion",
    outcome=Outcome.SUCCESS,
)
```

Outcomes drive the decay engine:

| Outcome | Signal |
|---|---|
| `SUCCESS` | reinforce |
| `PARTIAL` | reinforce |
| `FAILURE` | contradict |
| `UNKNOWN` | no-op (typical at record time when the outcome isn't observed yet) |

## Retrieve

```python
matches = memory.retrieve_procedures(
    "flaky CI test keeps failing intermittently",
    k=3,
)
for m in matches:
    print(m.score, m.procedure.action)
```

Ranking is `similarity * weight * outcome_boost`:

| Outcome | Boost |
|---|---|
| `SUCCESS` | 1.0 |
| `PARTIAL` | 0.8 |
| `UNKNOWN` | 0.7 |
| `FAILURE` | 0.6 |

Failures stay visible (the agent benefits from "this didn't work" lessons) but successes outrank failures at equal similarity.

Filter by outcome:

```python
successes = memory.retrieve_procedures(
    "flaky CI test",
    k=3,
    outcomes=[Outcome.SUCCESS, Outcome.PARTIAL],
)
```

## Update an outcome

A procedure starts as `UNKNOWN` if you don't know yet. Flip later:

```python
proc = memory.record_procedure("s", "a")  # UNKNOWN
# ... agent runs the action, observes the outcome ...
updated = memory.update_outcome(proc.id, Outcome.SUCCESS)
```

The decay engine gets the right signal automatically (`SUCCESS`/`PARTIAL` → reinforce, `FAILURE` → contradict, `UNKNOWN` → no-op).

## Outcome-feedback loop

Each `retrieve_procedures` call reinforces every surfaced procedure (unless you pass `reinforce=False`). Combined with outcome-driven reinforcement/contradiction at update time, this creates the loop the README pitches: useful procedures get stronger, failed ones get weaker.

See the procedural-transfer benchmark at `benchmarks/suites/procedural_transfer.py` for the +72pp lift over random-action baseline on the synthetic exact-match split.
