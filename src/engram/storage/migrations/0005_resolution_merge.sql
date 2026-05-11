-- Stage 8 follow-up (v0.3.1) -- widen conflicts.resolution CHECK.
--
-- Migration 0004 shipped with the original five resolution values
-- (prefer_recent, prefer_trusted, prefer_frequent, keep_both, manual).
-- v0.3.1 adds `merge`, where the reconciler synthesizes a new memory
-- item via the chat provider; both originals get invalidated pointing
-- to the new item. `resolved_winner_id` is NULL for MERGE (mirroring
-- KEEP_BOTH); the merged-into id is reachable via either original's
-- `invalidated_by` field.
--
-- SQLite cannot ALTER a CHECK constraint in place, so we rebuild the
-- conflicts table. Pre-existing rows survive verbatim; the indexes are
-- recreated alongside.

BEGIN;

CREATE TABLE conflicts_new (
    id                  BLOB PRIMARY KEY,
    source_item_id      BLOB NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    target_item_id      BLOB NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    similarity          REAL NOT NULL CHECK (similarity >= -1.0 AND similarity <= 1.0),
    verdict             TEXT NOT NULL DEFAULT 'contradict'
        CHECK (verdict IN ('agree', 'contradict', 'unrelated')),
    status              TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'resolved')),
    resolution          TEXT
        CHECK (resolution IS NULL OR resolution IN (
            'prefer_recent', 'prefer_trusted', 'prefer_frequent',
            'keep_both', 'manual', 'merge'
        )),
    resolved_winner_id  BLOB,
    resolved_at         TEXT,
    detected_at         TEXT NOT NULL,
    CHECK (source_item_id != target_item_id),
    UNIQUE(source_item_id, target_item_id)
);

INSERT INTO conflicts_new
    (id, source_item_id, target_item_id, similarity, verdict, status,
     resolution, resolved_winner_id, resolved_at, detected_at)
    SELECT id, source_item_id, target_item_id, similarity, verdict, status,
           resolution, resolved_winner_id, resolved_at, detected_at
    FROM conflicts;

DROP TABLE conflicts;
ALTER TABLE conflicts_new RENAME TO conflicts;

CREATE INDEX idx_conflicts_status         ON conflicts(status);
CREATE INDEX idx_conflicts_source_item    ON conflicts(source_item_id);
CREATE INDEX idx_conflicts_target_item    ON conflicts(target_item_id);
CREATE INDEX idx_conflicts_resolved_winner ON conflicts(resolved_winner_id)
    WHERE resolved_winner_id IS NOT NULL;

INSERT INTO schema_migrations (version) VALUES (5);

COMMIT;
