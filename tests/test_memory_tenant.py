"""Stage 9a multi-tenant write-side tests.

Memory(tenant_id="...") injects the tenant onto every write. Read-side
filtering by tenant is deferred to v0.4.0 (Postgres + RLS) -- the schema
and write surface are ready ahead of that.
"""

from __future__ import annotations

import pytest

from engram import (
    Level,
    Memory,
    MemoryItem,
    Outcome,
    SqliteStorage,
)
from engram.providers._fake import FakeEmbedder


@pytest.fixture
def memory_a(storage: SqliteStorage) -> Memory:
    return Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=8),
        tenant_id="tenant-a",
    )


@pytest.fixture
def memory_b(storage: SqliteStorage) -> Memory:
    return Memory(
        storage=storage,
        embedder=FakeEmbedder(dim=8),
        tenant_id="tenant-b",
    )


class TestTenantPropertyExposed:
    def test_default_is_none(self, storage: SqliteStorage) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=8))
        assert memory.tenant_id is None

    def test_constructor_sets_it(self, memory_a: Memory) -> None:
        assert memory_a.tenant_id == "tenant-a"


class TestObserveInjectsTenant:
    def test_event_tagged(self, memory_a: Memory) -> None:
        event = memory_a.observe("hello")
        fetched = memory_a.storage.get_event(event.id)
        assert fetched is not None
        assert fetched.tenant_id == "tenant-a"

    def test_caller_preset_tenant_preserved(self, memory_a: Memory) -> None:
        from engram.schemas import Event

        # If the caller explicitly sets a different tenant, Memory does
        # NOT override it. (This is the escape hatch for cross-tenant
        # admin tools that operate over multiple tenants.)
        pre = Event(content="x", tenant_id="other-tenant")
        result = memory_a.observe(pre)
        fetched = memory_a.storage.get_event(result.id)
        assert fetched is not None
        assert fetched.tenant_id == "other-tenant"

    def test_untenanted_memory_does_not_set(self, storage: SqliteStorage) -> None:
        memory = Memory(storage=storage, embedder=FakeEmbedder(dim=8))
        event = memory.observe("x")
        fetched = memory.storage.get_event(event.id)
        assert fetched is not None
        assert fetched.tenant_id is None


class TestRecordProcedureInjectsTenant:
    def test_procedure_tagged(self, memory_a: Memory) -> None:
        proc = memory_a.record_procedure("flaky CI", "rerun", outcome=Outcome.SUCCESS)
        fetched = memory_a.storage.get_procedure(proc.id)
        assert fetched is not None
        assert fetched.tenant_id == "tenant-a"


class TestMemoryItemCanCarryTenant:
    def test_round_trip(self, storage: SqliteStorage) -> None:
        item = MemoryItem(level=Level.SUMMARY, content="x", tenant_id="alpha")
        storage.insert_memory_item(item)
        fetched = storage.get_memory_item(item.id)
        assert fetched is not None
        assert fetched.tenant_id == "alpha"


class TestCrossTenantWriteIsolation:
    def test_both_writes_persist_with_correct_tenant(
        self, memory_a: Memory, memory_b: Memory
    ) -> None:
        """Two Memory instances pointing at the same SQLite both write
        successfully; each row carries its own tenant."""
        e_a = memory_a.observe("from a")
        e_b = memory_b.observe("from b")
        f_a = memory_a.storage.get_event(e_a.id)
        f_b = memory_b.storage.get_event(e_b.id)
        assert f_a is not None
        assert f_a.tenant_id == "tenant-a"
        assert f_b is not None
        assert f_b.tenant_id == "tenant-b"
