-- Stage 9 (E.7 + E.8) -- multi-granularity hierarchy: TOPIC + GLOBAL.
--
-- Widens `memory_items.level` to accept 'topic' and 'global' in
-- addition to the existing event / summary / abstraction / preference
-- (the latter added in 0007). Rebuild pattern same as 0007; FK
-- toggling protects conflict rows from cascade-delete.

PRAGMA foreign_keys = OFF;

BEGIN;

CREATE TABLE memory_items_new (
    id                  BLOB PRIMARY KEY,
    level               TEXT NOT NULL CHECK (level IN (
        'event', 'summary', 'abstraction', 'preference', 'topic', 'global'
    )),
    content             TEXT NOT NULL,
    weight              REAL NOT NULL DEFAULT 1.0
        CHECK (weight >= 0.0 AND weight <= 1.0),
    cluster_id          BLOB REFERENCES clusters(id) ON DELETE SET NULL,
    metadata            TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    reinforcement_count INTEGER NOT NULL DEFAULT 0 CHECK (reinforcement_count >= 0),
    corroboration_count INTEGER NOT NULL DEFAULT 0 CHECK (corroboration_count >= 0),
    contradiction_count INTEGER NOT NULL DEFAULT 0 CHECK (contradiction_count >= 0),
    last_decayed_at     TEXT,
    cold_at             TEXT,
    valid_from          TEXT,
    valid_until         TEXT,
    invalidated_at      TEXT,
    invalidated_by      BLOB,
    source_trust        REAL
        CHECK (source_trust IS NULL OR (source_trust >= 0.0 AND source_trust <= 1.0)),
    tenant_id           TEXT
);

-- Explicit column lists on both sides protect against silent
-- column-order drift if a future migration adds/reorders columns on
-- either table.  Positional INSERT...SELECT with matching counts
-- would still 'succeed' but write data into the wrong slots.
INSERT INTO memory_items_new (
    id, level, content, weight, cluster_id, metadata, created_at, updated_at,
    reinforcement_count, corroboration_count, contradiction_count,
    last_decayed_at, cold_at,
    valid_from, valid_until, invalidated_at, invalidated_by, source_trust,
    tenant_id
) SELECT
    id, level, content, weight, cluster_id, metadata, created_at, updated_at,
    reinforcement_count, corroboration_count, contradiction_count,
    last_decayed_at, cold_at,
    valid_from, valid_until, invalidated_at, invalidated_by, source_trust,
    tenant_id
FROM memory_items;

DROP TABLE memory_items;
ALTER TABLE memory_items_new RENAME TO memory_items;

CREATE INDEX idx_memory_items_created_at    ON memory_items(created_at);
CREATE INDEX idx_memory_items_weight        ON memory_items(weight);
CREATE INDEX idx_memory_items_level         ON memory_items(level);
CREATE INDEX idx_memory_items_cluster_id    ON memory_items(cluster_id) WHERE cluster_id IS NOT NULL;
CREATE INDEX idx_memory_items_cold_at       ON memory_items(cold_at) WHERE cold_at IS NOT NULL;
CREATE INDEX idx_memory_items_valid_until    ON memory_items(valid_until)
    WHERE valid_until IS NOT NULL;
CREATE INDEX idx_memory_items_invalidated_at ON memory_items(invalidated_at)
    WHERE invalidated_at IS NOT NULL;
CREATE INDEX idx_memory_items_source_trust   ON memory_items(source_trust)
    WHERE source_trust IS NOT NULL;
CREATE INDEX idx_memory_items_tenant_id      ON memory_items(tenant_id)
    WHERE tenant_id IS NOT NULL;

INSERT INTO schema_migrations (version) VALUES (8);

COMMIT;

PRAGMA foreign_keys = ON;
