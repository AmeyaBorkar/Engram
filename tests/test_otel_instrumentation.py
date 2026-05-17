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

from engram import (
    Conflict,
    Level,
    Memory,
    MemoryItem,
    Resolution,
    SqliteStorage,
)
from engram._otel import INSTRUMENTATION_NAME, is_otel_available
from engram.providers._fake import FakeEmbedder


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Install a fresh TracerProvider with an in-memory exporter.

    The OTel SDK's `set_tracer_provider` is a one-time call by design.
    For tests we explicitly poke the internal slots; monkeypatch's
    auto-teardown restores them after the test so the patched state
    doesn't leak into any later test that imports otel_trace.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "engram-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # `monkeypatch.setattr` records the original value and restores it
    # at fixture teardown — so the test patch doesn't bleed into
    # subsequent tests that import otel_trace.
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", provider, raising=False)
    monkeypatch.setattr(
        otel_trace._TRACER_PROVIDER_SET_ONCE,
        "_done",
        False,
        raising=False,
    )
    otel_trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def memory(storage: SqliteStorage) -> Memory:
    return Memory(storage=storage, embedder=FakeEmbedder(dim=8))


def _span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


class TestObserveInstrumentation:
    def test_observe_emits_span(self, memory: Memory, span_exporter: InMemorySpanExporter) -> None:
        event = memory.observe("hello world")
        names = _span_names(span_exporter)
        assert "engram.memory.observe" in names
        # The span has the event id and embedder.model attributes.
        obs_span = next(
            s for s in span_exporter.get_finished_spans() if s.name == "engram.memory.observe"
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
            s for s in span_exporter.get_finished_spans() if s.name == "engram.memory.retrieve"
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
            s for s in span_exporter.get_finished_spans() if s.name == "engram.memory.retrieve"
        )
        attrs = dict(retr_span.attributes or {})
        assert attrs.get("engram.retrieve.as_of") == when.isoformat()


class TestReconcileInstrumentation:
    def test_reconcile_emits_span_with_resolution(
        self,
        memory: Memory,
        storage: SqliteStorage,
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
            s for s in span_exporter.get_finished_spans() if s.name == "engram.memory.reconcile"
        )
        attrs = dict(rec_span.attributes or {})
        assert attrs.get("resolution") == "prefer_recent"
        assert attrs.get("engram.reconcile.winner_id") == str(newer.id)
        assert attrs.get("engram.reconcile.conflict_id") == str(c.id)


class TestInstrumentationName:
    def test_tracer_named_engram(self, memory: Memory, span_exporter: InMemorySpanExporter) -> None:
        # Every span comes from the `engram` instrumentation library.
        # Uses the conftest-derived `memory` fixture (audit M-134) so
        # the underlying SqliteStorage is closed on teardown.
        memory.observe("x")
        for s in span_exporter.get_finished_spans():
            scope = s.instrumentation_scope
            assert scope is not None
            assert scope.name == INSTRUMENTATION_NAME


def test_otel_available() -> None:
    """Smoke: the dev environment ships opentelemetry-api."""
    assert is_otel_available()


# ---------------------------------------------------------------------------
# I-05: new counters + mode attribute on retrieve_call
# ---------------------------------------------------------------------------


class TestObservabilityCounters:
    def test_retrieve_call_accepts_mode_keyword(self) -> None:
        """METRICS.retrieve_call(k, mode=...) must accept all modes
        without raising, even before OTel is wired."""
        from engram._otel import METRICS

        for mode in ("base", "hyde", "multi", "decompose", "temporal", None):
            METRICS.retrieve_call(k=10, mode=mode)

    def test_reinforce_error_counter_exists(self) -> None:
        from engram._otel import METRICS

        METRICS.reinforce_error(reason="missing")
        METRICS.reinforce_error()  # reason omitted

    def test_auto_temporal_fallback_counter_exists(self) -> None:
        from engram._otel import METRICS

        METRICS.auto_temporal_fallback()

    def test_verify_parse_fallback_counter_exists(self) -> None:
        from engram._otel import METRICS

        METRICS.verify_parse_fallback()

    def test_merge_fallback_counter_exists(self) -> None:
        from engram._otel import METRICS

        METRICS.merge_fallback()


# ---------------------------------------------------------------------------
# M-202: span.set_attribute guarded against TypeError / ValueError
# ---------------------------------------------------------------------------


class TestSpanSafeSetAttribute:
    def test_span_swallows_unsupported_attribute_type(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        """OTel rejects values outside its supported scalar/sequence types;
        the helper must log + continue rather than tearing down the span.
        """
        from engram._otel import span as span_cm

        class _Weird:
            def __repr__(self) -> str:
                return "<weird>"

        # `set_attribute(..., _Weird())` raises TypeError in OTel; the
        # helper must catch it (the assertion is that no exception
        # surfaces here).
        with span_cm("engram.test.attr_guard", weird=_Weird()) as s:
            assert s is not None
            # Inside the block, a stray set_attribute on a weird value
            # also must not propagate.
            from engram._otel import _safe_set_attribute

            _safe_set_attribute(s, "ok", "yes")
            _safe_set_attribute(s, "bad", _Weird())

        spans = span_exporter.get_finished_spans()
        # The span finished cleanly — its name + the safe attributes
        # land in the export; the bad ones are dropped.
        attr_guard_span = next(sp for sp in spans if sp.name == "engram.test.attr_guard")
        attrs = dict(attr_guard_span.attributes or {})
        assert attrs.get("ok") == "yes"
        assert "bad" not in attrs
