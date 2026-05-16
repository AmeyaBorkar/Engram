"""OpenTelemetry instrumentation helpers.

Stage 9 instruments every public Memory call with spans and emits
counters/histograms for observable signals. The implementation is
*lazy*:

  * If `opentelemetry-api` is installed AND the user has configured a
    `TracerProvider` / `MeterProvider`, the helpers return real
    tracers/meters and spans are emitted.
  * If `opentelemetry-api` is not installed, `span()` yields `None`
    immediately so Memory code can call `with span("memory.observe"):
    ...` unconditionally; cost is one branch + one yield when OTel is
    absent.

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

_log = logging.getLogger(__name__)

INSTRUMENTATION_NAME = "engram"


def _resolve_version() -> str:
    """Read the package version from importlib.metadata.

    Pinned via installed distribution metadata so a release that
    forgets to update both `engram.__version__` and an
    `INSTRUMENTATION_VERSION` literal here cannot fall out of sync.
    Falls back to "unknown" in the dev case where `engrampy` is not
    yet installed (e.g. running from a source tree without `pip
    install -e .`).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("engrampy")
        except PackageNotFoundError:
            # Source-tree run without an installed distribution. The
            # tests pin via the package __init__'s __version__, so
            # accept the literal here as a last-resort default rather
            # than crash on import.
            return "0.0.0+unknown"
    except ImportError:  # pragma: no cover - importlib.metadata always present on >=3.10
        return "0.0.0+unknown"


INSTRUMENTATION_VERSION = _resolve_version()


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

    Always raises `RuntimeError` when `opentelemetry-api` is not
    installed. The lazy `is_otel_available()` check is the public way
    to test for OTel; callers that want a no-op should use the `span()`
    contextmanager, which yields `None` in that case.
    """
    if _trace_mod is None:  # pragma: no cover - otel absent at import time
        raise RuntimeError(
            "opentelemetry-api is not installed; install engrampy[otel]"
        )
    return _trace_mod.get_tracer(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def get_meter() -> Meter:
    """Return a meter for the engram instrumentation.

    Always raises `RuntimeError` when `opentelemetry-api` is not
    installed; see `get_tracer()` for the rationale.
    """
    if _metrics_mod is None:  # pragma: no cover - otel absent at import time
        raise RuntimeError(
            "opentelemetry-api is not installed; install engrampy[otel]"
        )
    return _metrics_mod.get_meter(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def is_otel_available() -> bool:
    """True if opentelemetry-api is importable.

    Memory code that emits spans calls this lazily; without OTel the
    instrumentation short-circuits.
    """
    return _OTEL_AVAILABLE


def safe_set_attribute(span_obj: Span | None, key: str, value: Any) -> None:
    """Set an attribute on `span_obj` tolerating any (TypeError, ValueError).

    OTel restricts attribute values to bool / str / int / float / sequence
    thereof. A value that fails the type check (a UUID, a datetime, a
    complex container) would otherwise raise during instrumentation and
    propagate out of the user's call. We log at DEBUG and drop the
    attribute -- observability MUST NOT break the wrapped operation.

    `None` values are silently dropped to match the `span()` helper.
    """
    if span_obj is None or value is None:
        return
    try:
        span_obj.set_attribute(key, value)
    except (TypeError, ValueError) as exc:
        _log.debug(
            "skipping span attribute %r=%r: %s", key, value, exc, exc_info=False
        )


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span | None]:
    """Wrap a block of code in a span if OTel is available.

    Usage::

        with span("memory.observe", k=10) as s:
            event = self._do_observe(content)
            safe_set_attribute(s, "memory.event_id", str(event.id))

    The `s` yielded is the underlying `Span` or `None`. Callers can
    add attributes conditionally; the no-op path costs almost nothing.

    Set attributes via `safe_set_attribute` rather than `s.set_attribute`
    directly so a malformed value (a UUID, a datetime) does not propagate
    a TypeError out of the instrumented call.
    """
    if not _OTEL_AVAILABLE:
        # Fast-path: when OTel is absent we still go through the
        # contextmanager protocol (one yield), but we never construct a
        # tracer or evaluate the attributes dict beyond what `**` already
        # forces. Allocation here is the empty-state generator only.
        yield None
        return
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            safe_set_attribute(s, key, value)
        yield s


class _Metrics:
    """Lazy-initialized counters/histograms used across the codebase.

    Construction is deferred until first access so that importing this
    module is free even without OTel configured. Counter / histogram
    methods are thread-safe per the OTel API contract, but the lazy
    `_init()` itself is guarded by a `threading.Lock` so concurrent
    first-callers do not race to construct duplicate counters.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._lock = threading.Lock()
        self._retrieve_count: Counter | None = None
        self._retrieve_latency_ms: Histogram | None = None
        self._observe_count: Counter | None = None
        self._consolidate_count: Counter | None = None
        self._reconcile_count: Counter | None = None
        # Counters added in I-05 for the observability gaps the audit
        # flagged: reinforce errors, fallback rates on auto-temporal /
        # verify-parse / merge. All instruments are nullable until the
        # first _init() call lands.
        self._reinforce_error_count: Counter | None = None
        self._auto_temporal_fallback_count: Counter | None = None
        self._verify_parse_fallback_count: Counter | None = None
        self._merge_fallback_count: Counter | None = None

    def _init(self) -> None:
        # Fast path under the read of a plain bool (no lock cost in steady
        # state). Lock only on the rare race where two threads see
        # `_initialized=False` simultaneously.
        if self._initialized or not _OTEL_AVAILABLE:
            return
        with self._lock:
            # Double-check under the lock: another thread may have raced
            # ahead while we waited.
            if self._initialized:
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
            # Gap-fill counters (I-05).
            self._reinforce_error_count = meter.create_counter(
                "engram.reinforce.errors",
                description=(
                    "Errors raised by Memory.reinforce / corroborate / "
                    "contradict (item missing, cold pool, value errors)"
                ),
            )
            self._auto_temporal_fallback_count = meter.create_counter(
                "engram.retrieve.auto_temporal_fallback",
                description=(
                    "Times the auto-temporal retrieve path could not "
                    "extract a temporal anchor and fell back to the "
                    "non-temporal retrieve"
                ),
            )
            self._verify_parse_fallback_count = meter.create_counter(
                "engram.verify.parse_fallback",
                description=(
                    "Times the EngramAgent verify step could not parse "
                    "the verifier's response and fell back to the "
                    "unverified answer"
                ),
            )
            self._merge_fallback_count = meter.create_counter(
                "engram.reconcile.merge_fallback",
                description=(
                    "Times Resolution.MERGE could not synthesize a merged "
                    "memory item and fell back to PREFER_RECENT"
                ),
            )
            self._initialized = True

    def retrieve_call(self, k: int, *, mode: str | None = None) -> None:
        """Record one Memory.retrieve call.

        `mode` is the dispatch path the retrieve took: "base",
        "decompose", "multi_query", "iterative", etc. Caller may pass
        `None` for the legacy code path, in which case the attribute is
        omitted from the counter labels.
        """
        self._init()
        if self._retrieve_count is not None:
            attrs: dict[str, Any] = {"k": k}
            if mode is not None:
                attrs["mode"] = mode
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

    # -- I-05 gap-fill counters ---------------------------------------------

    def reinforce_error(self, *, kind: str, error: str) -> None:
        """Record one error from a reinforce / corroborate / contradict call.

        `kind` is the item kind ("event", "memory_item", "procedure"); `error`
        is a short tag for the failure mode (e.g. "missing", "cold",
        "validation").
        """
        self._init()
        if self._reinforce_error_count is not None:
            self._reinforce_error_count.add(1, {"kind": kind, "error": error})

    def auto_temporal_fallback(self) -> None:
        """Record one auto-temporal fallback (no anchor extracted)."""
        self._init()
        if self._auto_temporal_fallback_count is not None:
            self._auto_temporal_fallback_count.add(1)

    def verify_parse_fallback(self) -> None:
        """Record one verify-step parse fallback."""
        self._init()
        if self._verify_parse_fallback_count is not None:
            self._verify_parse_fallback_count.add(1)

    def merge_fallback(self, *, reason: str = "unknown") -> None:
        """Record one merge fallback to PREFER_RECENT.

        `reason` is a short tag for why merge could not proceed
        (e.g. "no_chat", "empty_response", "parse_error").
        """
        self._init()
        if self._merge_fallback_count is not None:
            self._merge_fallback_count.add(1, {"reason": reason})


METRICS = _Metrics()

__all__ = [
    "INSTRUMENTATION_NAME",
    "INSTRUMENTATION_VERSION",
    "METRICS",
    "get_meter",
    "get_tracer",
    "is_otel_available",
    "safe_set_attribute",
    "span",
]
