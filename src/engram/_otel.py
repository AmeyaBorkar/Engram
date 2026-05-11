"""OpenTelemetry instrumentation helpers.

Stage 9 instruments every public Memory call with spans and emits
counters/histograms for observable signals. The implementation is
*lazy*:

  * If `opentelemetry-api` is installed AND the user has configured a
    `TracerProvider` / `MeterProvider`, the helpers return real
    tracers/meters and spans are emitted.
  * If `opentelemetry-api` is not installed, the helpers return no-op
    objects with the same surface. Memory code can call `with
    span("memory.observe"): ...` unconditionally; cost is zero when
    OTel isn't present.

The user opts in via the `[otel]` extra (`pip install
engram-memory[otel]`) and then configures their own provider.

No SDK imports here -- this module touches only the `opentelemetry-api`
surface, which is stable and lightweight. The SDK is for callers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-only for typing
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Span, Tracer

INSTRUMENTATION_NAME = "engram"
INSTRUMENTATION_VERSION = "0.3.1"


try:
    from opentelemetry import metrics as _metrics_mod
    from opentelemetry import trace as _trace_mod

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without otel installed
    _OTEL_AVAILABLE = False
    _trace_mod = None  # type: ignore[assignment]
    _metrics_mod = None  # type: ignore[assignment]


def get_tracer() -> Tracer:
    """Return a tracer for the engram instrumentation.

    When OTel isn't installed, returns the API's documented no-op
    tracer (the api module always provides this; importing succeeds
    even without a configured provider).
    """
    if _trace_mod is None:  # pragma: no cover - otel absent at import time
        raise RuntimeError(
            "opentelemetry-api is not installed; install engram-memory[otel]"
        )
    return _trace_mod.get_tracer(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def get_meter() -> Meter:
    """Return a meter for the engram instrumentation."""
    if _metrics_mod is None:  # pragma: no cover - otel absent at import time
        raise RuntimeError(
            "opentelemetry-api is not installed; install engram-memory[otel]"
        )
    return _metrics_mod.get_meter(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def is_otel_available() -> bool:
    """True if opentelemetry-api is importable.

    Memory code that emits spans calls this lazily; without OTel the
    instrumentation short-circuits.
    """
    return _OTEL_AVAILABLE


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span | None]:
    """Wrap a block of code in a span if OTel is available.

    Usage:

        with span("memory.observe", k=10) as s:
            event = self._do_observe(content)
            if s is not None:
                s.set_attribute("memory.event_id", str(event.id))

    The `s` yielded is the underlying `Span` or `None`. Callers can
    add attributes conditionally; the no-op path costs almost nothing.
    """
    if not _OTEL_AVAILABLE:
        yield None
        return
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            if value is not None:
                s.set_attribute(key, value)
        yield s


class _Metrics:
    """Lazy-initialized counters/histograms used across the codebase.

    Construction is deferred until first access so that importing this
    module is free even without OTel configured. Counter / histogram
    methods are thread-safe per the OTel API contract.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._retrieve_count: Counter | None = None
        self._retrieve_latency_ms: Histogram | None = None
        self._observe_count: Counter | None = None
        self._consolidate_count: Counter | None = None
        self._reconcile_count: Counter | None = None

    def _init(self) -> None:
        if self._initialized or not _OTEL_AVAILABLE:
            return
        meter = get_meter()
        self._retrieve_count = meter.create_counter(
            "engram.retrieve.calls",
            description="Total number of Memory.retrieve calls",
        )
        self._retrieve_latency_ms = meter.create_histogram(
            "engram.retrieve.latency_ms",
            description="Memory.retrieve wall-clock latency in milliseconds",
            unit="ms",
        )
        self._observe_count = meter.create_counter(
            "engram.observe.calls",
            description="Total number of Memory.observe calls",
        )
        self._consolidate_count = meter.create_counter(
            "engram.consolidate.calls",
            description="Total number of Memory.consolidate runs",
        )
        self._reconcile_count = meter.create_counter(
            "engram.reconcile.calls",
            description="Total number of Memory.reconcile calls",
        )
        self._initialized = True

    def retrieve_call(self, k: int) -> None:
        self._init()
        if self._retrieve_count is not None:
            self._retrieve_count.add(1, {"k": k})

    def retrieve_latency(self, ms: float, *, k: int) -> None:
        self._init()
        if self._retrieve_latency_ms is not None:
            self._retrieve_latency_ms.record(ms, {"k": k})

    def observe_call(self) -> None:
        self._init()
        if self._observe_count is not None:
            self._observe_count.add(1)

    def consolidate_call(self) -> None:
        self._init()
        if self._consolidate_count is not None:
            self._consolidate_count.add(1)

    def reconcile_call(self, *, resolution: str) -> None:
        self._init()
        if self._reconcile_count is not None:
            self._reconcile_count.add(1, {"resolution": resolution})


METRICS = _Metrics()

__all__ = [
    "INSTRUMENTATION_NAME",
    "INSTRUMENTATION_VERSION",
    "METRICS",
    "get_meter",
    "get_tracer",
    "is_otel_available",
    "span",
]
