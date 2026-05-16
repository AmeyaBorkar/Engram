"""Hypothesis-driven invariants for the storage layer.

The headline invariant: provenance links never dangle. Anything we can do
to add, remove, or reorganize memory items and events must leave provenance
either valid or absent — never pointing into the void.
"""

from __future__ import annotations

import sqlite3
import string
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engram.schemas import Event, Level, MemoryItem
from engram.storage import SqliteStorage

# Bound the runtime of property tests in CI.
#
# `max_examples` was 50, which only sampled the small-graph corner of
# the input space.  Bumping to 200 quadruples coverage; combined with
# the wider `n_events`/`n_links` strategy below, the suite still
# completes well under the 2s deadline on a laptop.
_settings = settings(
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


_text = st.text(alphabet=string.printable, min_size=0, max_size=200)


@given(content=_text, source=st.one_of(st.none(), _text))
@_settings
def test_event_roundtrip(storage: SqliteStorage, content: str, source: str | None) -> None:
    e = Event(content=content, source=source)
    storage.insert_event(e)
    got = storage.get_event(e.id)
    assert got is not None
    assert got.content == content
    assert got.source == source


@given(
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    level=st.sampled_from(list(Level)),
    content=_text,
)
@_settings
def test_memory_item_weight_round_trip(
    storage: SqliteStorage, weight: float, level: Level, content: str
) -> None:
    item = MemoryItem(level=level, content=content, weight=weight)
    storage.insert_memory_item(item)
    got = storage.get_memory_item(item.id)
    assert got is not None
    assert got.weight == weight
    assert got.level == level


@given(
    # Widened from `max_value=10` so Hypothesis explores larger
    # provenance graphs.  Combined with `max_examples=200` above,
    # the property runs over many more shapes.
    n_events=st.integers(min_value=1, max_value=50),
    n_links=st.integers(min_value=1, max_value=50),
)
@_settings
def test_provenance_never_dangles(storage: SqliteStorage, n_events: int, n_links: int) -> None:
    """All provenance links must reference live rows on both sides."""
    events = [Event(content=f"e{i}") for i in range(n_events)]
    for e in events:
        storage.insert_event(e)
    item = MemoryItem(level=Level.SUMMARY, content="x")
    storage.insert_memory_item(item)

    seen: set[tuple[bytes, bytes]] = set()
    for i in range(n_links):
        target = events[i % len(events)]
        key = (item.id.bytes, target.id.bytes)
        if key in seen:
            continue
        seen.add(key)
        storage.link_provenance(item.id, target.id)

    # Raw `_connect()` here: the property verifies a cross-table
    # invariant (every provenance link's endpoints exist on both sides)
    # that the public API doesn't surface — it only returns linked
    # rows, not links pointing at missing rows.  Asking the SQL layer
    # directly is the only way to see "ghost" links if any ever
    # appeared.  noqa: SLF001
    rows = (
        storage._connect()
        .execute(
            "SELECT pl.memory_item_id, pl.event_id, "
            "       (SELECT 1 FROM memory_items mi WHERE mi.id = pl.memory_item_id) AS mi_ok, "
            "       (SELECT 1 FROM events ev WHERE ev.id = pl.event_id) AS ev_ok "
            "FROM provenance_links pl"
        )
        .fetchall()
    )
    for row in rows:
        assert row["mi_ok"] == 1
        assert row["ev_ok"] == 1


@given(content=_text, source=st.one_of(st.none(), _text))
@_settings
def test_event_metadata_handles_arbitrary_unicode(
    storage: SqliteStorage, content: str, source: str | None
) -> None:
    metadata: dict[str, Any] = {"k": content, "n": len(content)}
    e = Event(content=content, source=source, metadata=metadata)
    storage.insert_event(e)
    got = storage.get_event(e.id)
    assert got is not None
    assert got.metadata == metadata


def test_provenance_referential_integrity_fuzz(storage: SqliteStorage) -> None:
    """Direct fuzz: try linking provenance to garbage UUIDs; must always raise."""
    import os
    from uuid import UUID

    item = MemoryItem(level=Level.SUMMARY, content="x")
    storage.insert_memory_item(item)

    for _ in range(100):
        garbage = UUID(bytes=os.urandom(16))
        try:
            storage.link_provenance(item.id, garbage)
        except sqlite3.IntegrityError:
            continue
        # If we reach here, a provenance link was created with no event row.
        # That's a referential-integrity bug.
        raise AssertionError(f"provenance link to non-existent event {garbage} succeeded")
