"""Stage 9 OpenTelemetry instrumentation tests.

Sets up an in-memory span exporter on the test runner, exercises every
instrumented Memory call, and verifies the expected spans + attributes
are emitted. Counter / histogram emission is exercised but not deeply
asserted (the API contract is fully delegated to the SDK).

Tests rely on `opentelemetry-sdk` from the dev extras. End users who
install `engram-memory[otel]` get the same surface; users who install
plain `engram-memory` get free no-ops.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import engram as _engram_pkg
from engram import (
    Conflict,
    Level,
    Memory,
    MemoryItem,
    Resolution,
    SqliteStorage,
)
from engram._otel import (
    INSTRUMENTATION_NAME,
    INSTRUMENTATION_VERSION,
    METRICS,
    is_otel_available,
    safe_set_attribute,
)
from engram.providers._fake import FakeEmbedder


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Install a fresh TracerProvider with an in-memory exporter.

    Each test gets isolated spans; teardown resets the global provider
    by replacing it with a fresh one for the next test (the OTel
    sdk's `set_tracer_provider` is a one-time call by design, but the
    NoOpTracerProvider sentinel allows replacement in tests).
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "engram-test"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Force-override the global provider (OTel allows this once normally;
    # we explicitly poke the internal slot for repeat-set in tests).
    otel_trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined,unused-ignore]
    otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined,unused-ignore]
    otel_trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


def _span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


class TestObserveInstrumentation:
    def test_observe_emits_span(
        self, memory: Memory, span_exporter: InMemorySpanExporter
    ) -> None:
        event = memory.observe("hello world")
        names = _span_names(span_exporter)
        assert "engram.memory.observe" in names
        # The span has the event id and embedder.model attributes.
        obs_span = next(
            s
            for s in span_exporter.get_finished_spans()
            if s.name == "engram.memory.observe"
        )
        attrs = dict(obs_span.attributes or {})
        assert attrs.get("engram.event_id") == str(event.id)
        assert "engram.embedder.model" in attrs


class TestRetrieveInstrumentation:
    def test_retrieve_emits_span_with_k_and_latency(
        self, memory: Memory, span_exporter: InMemorySpanExporter
    ) -> None:
        memory.observe("once upon a time there was a cat")
        memory.retrieve("cat", k=5)
        names = _span_names(span_exporter)
        assert "engram.memory.retrieve" in names
        retr_span = next(
            s
            for s in span_exporter.get_finished_spans()
            if s.name == "engram.memory.retrieve"
        )
        attrs = dict(retr_span.attributes or {})
        assert attrs.get("k") == 5
        assert "engram.retrieve.n_results" in attrs
        assert "engram.retrieve.latency_ms" in attrs

    def test_retrieve_with_as_of_records_attribute(
        self, memory: Memory, span_exporter: InMemorySpanExporter
    ) -> None:
        memory.observe("x")
        when = datetime(2030, 1, 1, tzinfo=timezone.utc)
        memory.retrieve("x", k=1, as_of=when)
        retr_span = next(
            s
            for s in span_exporter.get_finished_spans()
            if s.name == "engram.memory.retrieve"
        )
        attrs = dict(retr_span.attributes or {})
        assert attrs.get("engram.retrieve.as_of") == when.isoformat()


class TestReconcileInstrumentation:
    def test_reconcile_emits_span_with_resolution(
        self, memory: Memory, storage: SqliteStorage,
        span_exporter: InMemorySpanExporter,
    ) -> None:
        older = MemoryItem(
            level=Level.SUMMARY,
            content="older",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        newer = MemoryItem(
            level=Level.SUMMARY,
            content="newer",
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            valid_from=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        storage.insert_memory_item(older)
        storage.insert_memory_item(newer)
        c = Conflict(source_item_id=newer.id, target_item_id=older.id, similarity=0.9)
        storage.record_conflict(c)
        memory.reconcile(
            c.id,
            resolution=Resolution.PREFER_RECENT,
            now=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
        rec_span = next(
            s
            for s in span_exporter.get_finished_spans()
            if s.name == "engram.memory.reconcile"
        )
        attrs = dict(rec_span.attributes or {})
        assert attrs.get("resolution") == "prefer_recent"
        assert attrs.get("engram.reconcile.winner_id") == str(newer.id)
        assert attrs.get("engram.reconcile.conflict_id") == str(c.id)


class TestInstrumentationName:
    def test_tracer_named_engram(self, span_exporter: InMemorySpanExporter) -> None:
        # Every span comes from the `engram` instrumentation library.
        memory = Memory(
            storage=SqliteStorage(":memory:"),
            embedder=FakeEmbedder(dim=8),
        )
        memory.storage.initialize()
        memory.observe("x")
        for s in span_exporter.get_finished_spans():
            scope = s.instrumentation_scope
            assert scope is not None
            assert scope.name == INSTRUMENTATION_NAME


def test_otel_available() -> None:
    """Smoke: the dev environment ships opentelemetry-api."""
    assert is_otel_available()


class TestInstrumentationVersion:
    """`INSTRUMENTATION_VERSION` must equal `engram.__version__`.

    Drifting these is the classic observability bug -- a release ships
    one but forgets the other and downstream metrics pivot off the
    wrong tag. Pin them via importlib.metadata so they cannot drift.
    """

    def test_matches_package_version(self) -> None:
        assert INSTRUMENTATION_VERSION == _engram_pkg.__version__


class TestSafeSetAttribute:
    """`safe_set_attribute` tolerates malformed values without raising."""

    def test_none_value_is_noop(self) -> None:
        # Should not raise even with None span.
        safe_set_attribute(None, "k", "v")
        # Should not raise with None value.
        safe_set_attribute(_DummySpan(), "k", None)

    def test_bad_type_logged_and_dropped(self) -> None:
        s = _DummySpan(raises=TypeError("bad"))
        # No exception should escape.
        safe_set_attribute(s, "key", object())

    def test_value_error_swallowed(self) -> None:
        s = _DummySpan(raises=ValueError("bad"))
        safe_set_attribute(s, "key", object())

    def test_unrelated_exception_propagates(self) -> None:
        s = _DummySpan(raises=RuntimeError("not a value error"))
        # safe_set_attribute should NOT swallow non-(TypeError, ValueError).
        with pytest.raises(RuntimeError):
            safe_set_attribute(s, "key", "v")


class _DummySpan:
    """Minimal Span-like object for safe_set_attribute tests."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self._raises = raises

    def set_attribute(self, key: str, value: object) -> None:  # noqa: ARG002
        if self._raises is not None:
            raise self._raises


class TestMetricsGapFillCounters:
    """The I-05 gap-fill counters are reachable and call into OTel.

    We don't assert on the SDK's emitted values (the API contract is
    delegated to the SDK); we assert the API is callable and
    short-circuits cleanly when OTel is absent.
    """

    def test_reinforce_error_callable(self) -> None:
        # Should not raise.
        METRICS.reinforce_error(kind="event", error="missing")

    def test_auto_temporal_fallback_callable(self) -> None:
        METRICS.auto_temporal_fallback()

    def test_verify_parse_fallback_callable(self) -> None:
        METRICS.verify_parse_fallback()

    def test_merge_fallback_callable(self) -> None:
        METRICS.merge_fallback(reason="no_chat")

    def test_retrieve_call_with_mode(self) -> None:
        # Mode label is an opt-in dimension on retrieve.calls.
        METRICS.retrieve_call(k=5, mode="base")
        METRICS.retrieve_call(k=5, mode="decompose")
        METRICS.retrieve_call(k=5)  # no mode is also fine
