# Observability

Engram emits OpenTelemetry spans and counters from every public `Memory` call when the optional `[otel]` extra is installed and a `TracerProvider` is configured.

## Install

```bash
pip install "engram-memory[otel]"
```

The extra brings `opentelemetry-api` and `opentelemetry-sdk`. Without it, the instrumentation short-circuits to no-ops — there's zero cost for users who don't opt in.

## Configure a provider

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider(
    resource=Resource.create({"service.name": "your-agent"})
)
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)
trace.set_tracer_provider(provider)
```

## What gets emitted

### Spans

Every public `Memory` method that touches storage or providers emits a span. All spans come from the `engram` instrumentation library.

| Span | Attributes |
|---|---|
| `engram.memory.observe` | `engram.event_id`, `engram.embedder.model` |
| `engram.memory.retrieve` | `k`, `prefer`, `engram.retrieve.n_results`, `engram.retrieve.latency_ms`, optional `engram.retrieve.as_of` |
| `engram.memory.consolidate` | `max_events`, `engram.consolidate.events_processed`, `engram.consolidate.clusters_formed`, `engram.consolidate.abstractions_created`, `engram.consolidate.conflicts_detected` |
| `engram.memory.reconcile` | `resolution`, `engram.reconcile.conflict_id`, `engram.reconcile.winner_id` |

### Counters

| Counter | Tags |
|---|---|
| `engram.observe.calls` | — |
| `engram.retrieve.calls` | `k` |
| `engram.consolidate.calls` | — |
| `engram.reconcile.calls` | `resolution` |

### Histograms

| Histogram | Unit | Tags |
|---|---|---|
| `engram.retrieve.latency_ms` | ms | `k` |

## Tracing across async boundaries

The async methods (`aobserve`, `aretrieve`, etc.) inherit the trace context from the caller through `asyncio.to_thread`'s context-var propagation. No special handling needed.

## Verifying in tests

The dev extras include `opentelemetry-sdk`. Tests register an in-memory exporter:

```python
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
otel_trace.set_tracer_provider(provider)

memory.observe("x")
assert any(s.name == "engram.memory.observe" for s in exporter.get_finished_spans())
```
