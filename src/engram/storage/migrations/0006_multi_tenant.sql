-- Stage 9 — multi-tenant tenant_id columns.
--
-- Adds an optional `tenant_id TEXT` column to every owner-scoped table
-- (events, memory_items, procedures). Conflicts inherit tenancy through
-- their source / target memory_items so don't need their own column.
--
-- Semantics:
--   * NULL tenant_id => "untenanted" / "global". Visible only when the
--     caller doesn't filter by tenant; tenant-scoped queries do NOT
--     match NULL. This is the safe default: legacy data (pre-0006) ends
--     up untenanted and tenant-scoped reads can't see it accidentally.
--   * non-NULL tenant_id => the row belongs to that tenant; visible only
--     when the caller filters on the matching tenant_id.
--
-- Indexes on (tenant_id) keep the filter cheap. They're partial so the
-- common NULL-dominant single-tenant case adds no overhead.
--
-- Stage 9's Postgres backend layers RLS on top of this for actual
-- security isolation; SQLite gets filter-based isolation only (sufficient
-- for development + the single-process library case).

BEGIN;

ALTER TABLE events       ADD COLUMN tenant_id TEXT;
ALTER TABLE memory_items ADD COLUMN tenant_id TEXT;
ALTER TABLE procedures   ADD COLUMN tenant_id TEXT;

CREATE INDEX idx_events_tenant_id
    ON events(tenant_id) WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_memory_items_tenant_id
    ON memory_items(tenant_id) WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_procedures_tenant_id
    ON procedures(tenant_id) WHERE tenant_id IS NOT NULL;

INSERT INTO schema_migrations (version) VALUES (6);

COMMIT;
