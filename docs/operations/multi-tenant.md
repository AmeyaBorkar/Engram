# Multi-tenant

Engram supports per-tenant scoping of writes through the `tenant_id` field on `Event`, `MemoryItem`, and `Procedure`, with full read-side enforcement landing in `v0.4.0` alongside the Postgres backend with RLS.

## Current status (v0.3.x)

**Write side:** ✓ shipped.

`Memory(..., tenant_id="acme")` injects the tenant onto every write:

```python
from engram import Memory, SqliteStorage
from engram.providers._fake import FakeEmbedder

memory_acme = Memory(
    storage=SqliteStorage("engram.db"),
    embedder=FakeEmbedder(dim=128),
    tenant_id="acme",
)

# All writes get tagged with tenant_id="acme".
memory_acme.observe("acme-specific fact")
memory_acme.record_procedure("acme task", "acme action")
```

The caller can override per-call by passing a pre-built `Event` with an explicit `tenant_id` (the escape hatch for cross-tenant admin tools).

**Read side:** ⚠️ deferred to v0.4.0.

Today's SQLite-backed `Memory.retrieve(...)` returns rows regardless of tenant. The schema has the column, the indexes are in place, but the retrieve path doesn't filter on tenant_id yet. For a single-process library this is consistent with "the storage isn't a security boundary." For the production multi-tenant deployment, v0.4.0 lands the Postgres backend with row-level security (RLS) where the filter is enforced by the database — which is the right place for an isolation guarantee.

## Migration 0006

Tenant columns were added in migration 0006 with partial indexes:

```sql
ALTER TABLE events       ADD COLUMN tenant_id TEXT;
ALTER TABLE memory_items ADD COLUMN tenant_id TEXT;
ALTER TABLE procedures   ADD COLUMN tenant_id TEXT;

CREATE INDEX idx_events_tenant_id
    ON events(tenant_id) WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_memory_items_tenant_id
    ON memory_items(tenant_id) WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_procedures_tenant_id
    ON procedures(tenant_id) WHERE tenant_id IS NOT NULL;
```

Pre-migration rows end up with `tenant_id=NULL` ("untenanted" / "global"). Tenant-scoped queries (in v0.4.0) do NOT match NULL — legacy data stays invisible to tenant-scoped reads, which is the safe default.

## v0.4.0 preview

With the Postgres backend:

```python
from engram import Memory
from engram.storage.postgres import PostgresStorage  # v0.4.0

memory = Memory(
    storage=PostgresStorage(
        dsn="postgresql://engram_app@db/engram",  # per-tenant role
        tenant_id_setter="SELECT set_config('engram.tenant_id', $1, false)",
    ),
    embedder=...,
    tenant_id="acme",
)
```

Per-tenant connection roles + RLS policies on `events` / `memory_items` / `procedures` enforce tenant isolation at the DB level. Application code cannot bypass it.
