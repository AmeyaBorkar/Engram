-- Composite indexes on the hot retrieve / consolidation paths.
--
-- Three workloads benefit:
--
-- 1. Vector index rebuild scans (`_INDEX_REBUILD_SQL` in sqlite.py).
--    The rebuild reads every embedding row matching a fixed
--    `(item_kind, model)` pair to hydrate the in-memory ANN index on
--    the first search after a write. With ~500 events per LongMemEval
--    haystack the rebuild is hot, and the existing UNIQUE(item_id,
--    item_kind, model) sorts by item_id first -- ordering the lookup
--    by (item_kind, model) first lets sqlite jump straight to the
--    contiguous slice.
--
-- 2. Unconsolidated-events scan (`iter_unconsolidated_events_with_embeddings`).
--    Joins events + embeddings + filters by model + cold_at + NOT EXISTS
--    on provenance_links. A composite on embeddings(item_kind, model)
--    plus the existing event indexes lets the planner walk a tight slice.
--
-- 3. Conflict lookup by member (`list_conflicts(memory_item_id=X, status=...)`).
--    Currently merges idx_conflicts_source_item OR target_item_id +
--    idx_conflicts_status; the composite on (status, source_item_id)
--    and mirror (status, target_item_id) makes the lookup a direct
--    btree probe.

BEGIN;

-- 1. Vector index rebuild + unconsolidated scan.
CREATE INDEX IF NOT EXISTS idx_embeddings_kind_model
    ON embeddings(item_kind, model);

-- 2. Provenance reverse-walk (event -> memory_item). Already covered by
--    idx_provenance_event for membership tests; this index adds the
--    composite to short-circuit the NOT EXISTS subquery on the
--    consolidation path.
CREATE INDEX IF NOT EXISTS idx_provenance_event_memory
    ON provenance_links(event_id, memory_item_id);

-- 3. Conflict lookup by member + status.
CREATE INDEX IF NOT EXISTS idx_conflicts_source_status
    ON conflicts(source_item_id, status);
CREATE INDEX IF NOT EXISTS idx_conflicts_target_status
    ON conflicts(target_item_id, status);

-- 4. memory_items level + cluster_id composite -- the promotion pass
--    iterates summaries grouped by cluster.
CREATE INDEX IF NOT EXISTS idx_memory_items_level_cluster
    ON memory_items(level, cluster_id);

INSERT INTO schema_migrations (version) VALUES (9);

COMMIT;
