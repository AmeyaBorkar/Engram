-- Composite index supporting the `get_supporting_events` ORDER BY.
--
-- The hot path joins provenance_links to events and orders by
-- `provenance_links.weight DESC, events.created_at DESC`.  The
-- previous schema only had `idx_provenance_member` on
-- `(memory_item_id)`, so the planner found the rows via the index
-- but then sorted them by weight in memory.
--
-- For a memory item with hundreds of supporters that's a measurable
-- spike on every retrieve that drills into provenance.  The composite
-- below covers the lookup AND the sort key, so the planner can stream
-- rows already-sorted from the b-tree.

BEGIN;

CREATE INDEX IF NOT EXISTS idx_provenance_member_weight
    ON provenance_links(memory_item_id, weight DESC);

INSERT INTO schema_migrations (version) VALUES (10);

COMMIT;
