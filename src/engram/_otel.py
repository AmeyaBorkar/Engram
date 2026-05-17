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
engrampy[otel]`) and then configures their own provider.

No SDK imports here -- this module touches only the `opentelemetry-api`
surface, which is stable and lightweight. The SDK is for callers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-only for typing
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Span, Tracer

INSTRUMENTATION_NAME = "engram"
# Mirror the package version so OTel telemetry attribution doesn't drift
# from the installed distribution's metadata.  Read once at import.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    INSTRUMENTATION_VERSION: str = _pkg_version("engrampy")
except PackageNotFoundError:  # pragma: no cover - dev tree without install
    INSTRUMENTATION_VERSION = "0.0.0+unknown"


_LOG = logging.getLogger("engram.otel")


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

    Raises RuntimeError if opentelemetry-api is not installed.  Callers
    that want graceful no-op behavior when OTel is absent should gate
    on `is_otel_available()` first or use the `span()` context manager,
    which already short-circuits.  (The previous docstring promised a
    no-op tracer here but the implementation always raised — this
    aligns the contract with the behavior.)
    """
    if _trace_mod is None:  # pragma: no cover - otel absent at import time
        raise RuntimeError("opentelemetry-api is not installed; install engram-memory[otel]")
    return _trace_mod.get_tracer(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def get_meter() -> Meter:
    """Return a meter for the engram instrumentation."""
    if _metrics_mod is None:  # pragma: no cover - otel absent at import time
        raise RuntimeError("opentelemetry-api is not installed; install engram-memory[otel]")
    return _metrics_mod.get_meter(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def is_otel_available() -> bool:
    """True if opentelemetry-api is importable.

    Memory code that emits spans calls this lazily; without OTel the
    instrumentation short-circuits.
    """
    return _OTEL_AVAILABLE


def _safe_set_attribute(span: Span, key: str, value: Any) -> None:
    """Set `key=value` on `span`, swallowing type errors.

    OTel rejects attribute values outside `bool | int | float | str |
    Sequence[bool | int | float | str]`.  Caller code passes mixed
    payloads (UUIDs, datetimes, enums, model objects) — letting the
    `TypeError` / `ValueError` propagate kills the span, the wrapped
    operation succeeds without telemetry, and the surface error is
    confusing.  We log at DEBUG and move on so a bad attribute key
    doesn't break the user-visible call.
    """
    try:
        span.set_attribute(key, value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        _LOG.debug("otel: rejected attribute %s=%r: %s", key, value, exc)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span | None]:
    """Wrap a block of code in a span if OTel is available.

    Usage:

        with span("memory.observe", k=10) as s:
            event = self._do_observe(content)
            if s is not None:
                s.set_attribute("memory.event_id", str(event.id))

    Common attributes the rest of the codebase passes here:

      * `k` — retrieve top-k count
      * `mode` — retrieve routing path (`base`/`hyde`/`multi`/...)
      * `tenant_id` — multi-tenant scope key (memory.py retrieve / observe
        call sites pass this when set so dashboards can break down the
        per-tenant call rate without scraping each Memory instance)
      * `engram.event_id` / `engram.retrieve.n_results` — emitted by
        the wrapped operation, not at span entry.

    The `s` yielded is the underlying `Span` or `None`. Callers can
    add attributes conditionally; the no-op path costs almost nothing.
    Attribute values that OTel rejects (anything outside `bool | int |
    float | str | Sequence[...]`) are swallowed by `_safe_set_attribute`
    so a bad attribute key doesn't tear down the surrounding span.
    """
    if not _OTEL_AVAILABLE:
        yield None
        return
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            if value is not None:
                _safe_set_attribute(s, key, value)
        yield s


class _Metrics:
    """Lazy-initialized counters/histograms used across the codebase.

    Construction is deferred until first access so that importing this
    module is free even without OTel configured. Counter / histogram
    methods are thread-safe per the OTel API contract.
    """

    def __init__(self) -> None:
        self._initialized = False
        # Two threads hitting their first retrieve_call concurrently could
        # both pass the `if self._initialized` check and create duplicate
        # counters from the same meter.  The lock makes _init() effectively
        # double-checked-locking — fast path is a lock-free flag read, slow
        # path serializes the create_counter / create_histogram calls.
        self._init_lock = threading.Lock()
        self._retrieve_count: Counter | None = None
        self._retrieve_latency_ms: Histogram | None = None
        self._observe_count: Counter | None = None
        self._consolidate_count: Counter | None = None
        self._reconcile_count: Counter | None = None
        # I-05: observability gaps — counters added so dashboards can
        # alert on transient failure modes that the engine previously
        # swallowed without signal.
        self._reinforce_error_count: Counter | None = None
        self._auto_temporal_fallback_count: Counter | None = None
        self._verify_parse_fallback_count: Counter | None = None
        self._merge_fallback_count: Counter | None = None

    def _init(self) -> None:
        if self._initialized or not _OTEL_AVAILABLE:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._init_locked()

    def _init_locked(self) -> None:
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
        self._reinforce_error_count = meter.create_counter(
            "engram.reinforce.errors",
            description=(
                "Reinforcement signals that the storage layer rejected "
                "(item missing, already cold, etc.).  A non-zero rate "
                "indicates retrieve↔storage drift."
            ),
        )
        self._auto_temporal_fallback_count = meter.create_counter(
            "engram.retrieve.auto_temporal_fallback",
            description=(
                "Auto-temporal retrieves that produced zero hits and "
                "fell back to dropping the lexical_filter for a re-run."
            ),
        )
        self._verify_parse_fallback_count = meter.create_counter(
            "engram.verify.parse_fallback",
            description=(
                "Verify calls that failed to parse the judge response "
                "and fell back to the supported=True default."
            ),
        )
        self._merge_fallback_count = meter.create_counter(
            "engram.reconcile.merge_fallback",
            description=(
                "Merge resolutions that exhausted retries and fell back "
                "to one parent's content (audit-trail signal: a non-zero "
                "rate means human review is overdue)."
            ),
        )
        self._initialized = True

    def retrieve_call(self, k: int, *, mode: str | None = None) -> None:
        """Record one retrieve call.

        `mode` distinguishes the routing path
        (`"base"`/`"hyde"`/`"multi"`/`"decompose"`/`"temporal"`) so a
        dashboard can break the call rate down by feature without
        re-instrumenting Memory.retrieve.  Defaults to `"base"` when
        the caller doesn't specify — the unannotated retrieves still
        attribute to a known bucket.
        """
        self._init()
        if self._retrieve_count is not None:
            attrs: dict[str, Any] = {"k": k, "mode": mode or "base"}
            self._retrieve_count.add(1, attrs)

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

    def reinforce_error(self, *, reason: str | None = None) -> None:
        """Record a swallowed reinforcement error.

        `reason` is a short tag (`"missing"`, `"cold"`, `"transient"`)
        for downstream attribution; omitted when the caller can't
        cleanly classify.
        """
        self._init()
        if self._reinforce_error_count is not None:
            attrs: dict[str, Any] = {}
            if reason is not None:
                attrs["reason"] = reason
            self._reinforce_error_count.add(1, attrs)

    def auto_temporal_fallback(self) -> None:
        """Record one auto-temporal lexical-filter drop fallback."""
        self._init()
        if self._auto_temporal_fallback_count is not None:
            self._auto_temporal_fallback_count.add(1)

    def verify_parse_fallback(self) -> None:
        """Record one verify-judge parse failure that defaulted to supported=True."""
        self._init()
        if self._verify_parse_fallback_count is not None:
            self._verify_parse_fallback_count.add(1)

    def merge_fallback(self) -> None:
        """Record one merge resolution that exhausted retries."""
        self._init()
        if self._merge_fallback_count is not None:
            self._merge_fallback_count.add(1)


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
