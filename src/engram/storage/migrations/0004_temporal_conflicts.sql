-- Stage 8 — Contradiction & temporal reasoning.
--
-- Two extensions to the schema:
--
-- 1. A first-class `conflicts` table. The Stage 5 contradiction detector
--    already records detections in `MemoryItem.metadata` as a transient
--    blob; Stage 8 promotes them to rows so the reconciler can manage a
--    lifecycle (OPEN -> RESOLVED) and so audits can replay decisions.
--    The metadata blob is kept for back-compat (legacy reads still work).
--
-- 2. Temporal validity, explicit invalidation, and denormalized source
--    trust on `memory_items`:
--      * `valid_from`        - default = `created_at`; backfilled on
--                              upgrade. NOT NULL. Used by `as_of` queries.
--      * `valid_until`       - NULL for facts that are still current.
--      * `invalidated_at`    - NULL means "not invalidated."
--      * `invalidated_by`    - id of the item that won the conflict
--                              that invalidated this one (NULL allowed
--                              for TTL-style invalidations with no
--                              replacement).
--      * `source_trust`      - denormalized from the introducing Source's
--                              trust score in [0,1] for the
--                              `Resolution.PREFER_TRUSTED` policy.
--
-- The conflicts table uses ON DELETE CASCADE on the memory_item FKs:
-- if a memory item is hard-deleted (via `delete_cold_items` from Stage
-- 4), its dangling conflict rows go with it.

BEGIN;

CREATE TABLE conflicts (
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
            'keep_both', 'manual'
        )),
    resolved_winner_id  BLOB,
    resolved_at         TEXT,
    detected_at         TEXT NOT NULL,
    CHECK (source_item_id != target_item_id),
    UNIQUE(source_item_id, target_item_id)
);
CREATE INDEX idx_conflicts_status         ON conflicts(status);
CREATE INDEX idx_conflicts_source_item    ON conflicts(source_item_id);
CREATE INDEX idx_conflicts_target_item    ON conflicts(target_item_id);
CREATE INDEX idx_conflicts_resolved_winner ON conflicts(resolved_winner_id)
    WHERE resolved_winner_id IS NOT NULL;

-- Temporal validity + invalidation + source trust on memory_items.
-- `valid_from` is added nullable, backfilled with `created_at`. SQLite
-- can't add a NOT NULL column without a default; we leave the column
-- nullable in the schema and rely on the storage layer to always write
-- it (the row-to-model mapping reads `valid_from` and falls back to
-- `created_at` if NULL, mirroring the model_validator on the Pydantic
-- side).
ALTER TABLE memory_items ADD COLUMN valid_from      TEXT;
ALTER TABLE memory_items ADD COLUMN valid_until     TEXT;
ALTER TABLE memory_items ADD COLUMN invalidated_at  TEXT;
ALTER TABLE memory_items ADD COLUMN invalidated_by  BLOB;
ALTER TABLE memory_items ADD COLUMN source_trust    REAL
    CHECK (source_trust IS NULL OR (source_trust >= 0.0 AND source_trust <= 1.0));

UPDATE memory_items SET valid_from = created_at WHERE valid_from IS NULL;

-- Indexes for the temporal-aware retrieval path. Both columns are
-- typically NULL, so partial indexes keep them small.
CREATE INDEX idx_memory_items_valid_until    ON memory_items(valid_until)
    WHERE valid_until IS NOT NULL;
CREATE INDEX idx_memory_items_invalidated_at ON memory_items(invalidated_at)
    WHERE invalidated_at IS NOT NULL;
CREATE INDEX idx_memory_items_source_trust   ON memory_items(source_trust)
    WHERE source_trust IS NOT NULL;

INSERT INTO schema_migrations (version) VALUES (4);

COMMIT;
