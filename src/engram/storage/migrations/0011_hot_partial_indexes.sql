-- Partial indexes for hot-item scans + tenant length cap.
--
-- The decay tick iterates `iter_decay_states(..., include_cold=False)` —
-- a full-table scan filtered by `cold_at IS NULL`.  The existing
-- `idx_*_cold_at WHERE cold_at IS NOT NULL` indexes only help the
-- *cold* subset (decay-totals, audit reads).  Add the mirror partial
-- index covering hot rows so the tick walks an index slice instead of
-- the whole table.
--
-- For events, the consolidation path also benefits: `iter_unconsolidated_
-- events_with_embeddings` filters by `cold_at IS NULL` then `NOT EXISTS
-- (provenance_links ...)`.  The partial index on hot rows keeps the
-- planner from scanning cold events that will never make it past the
-- first filter.
--
-- Tenant length cap: the schema accepted arbitrary-length TEXT for
-- `tenant_id`, but every retrieve / log entry that prints it would
-- emit megabytes of garbage if a caller passed a malformed slug.  Cap
-- at 256 characters via a CHECK constraint.  Sub-256 is well over the
-- longest sensible tenant identifier (UUIDs, slugs, account ids).
--
-- SQLite can't add a CHECK to an existing column; we'd need to rebuild
-- the table.  Defer the structural change to a later migration and
-- enforce the cap at the Python layer (memory.py already rejects
-- empty/whitespace tenant_id).

BEGIN;

-- Hot events: covers the consolidation scan + decay tick.
CREATE INDEX IF NOT EXISTS idx_events_cold_at_null
    ON events(id) WHERE cold_at IS NULL;

-- Hot memory_items: covers the decay tick on the abstraction tier.
CREATE INDEX IF NOT EXISTS idx_memory_items_cold_at_null
    ON memory_items(id) WHERE cold_at IS NULL;

-- Hot procedures: same shape, smaller table; still a measurable win
-- when the procedure store grows.
CREATE INDEX IF NOT EXISTS idx_procedures_cold_at_null
    ON procedures(id) WHERE cold_at IS NULL;

INSERT INTO schema_migrations (version) VALUES (11);

COMMIT;
