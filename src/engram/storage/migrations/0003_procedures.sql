-- Stage 7 — Procedures.
--
-- A `procedure` row is "in this situation, this action had that
-- outcome." Retrieval embeds the `situation` column; `action` and
-- `outcome` come along as payload. The decay engine ticks procedures
-- alongside events and memory_items, so the column shape mirrors what
-- migration 0002 added to those tables (weight, reinforcement_count,
-- corroboration_count, contradiction_count, last_decayed_at, cold_at).
--
-- `outcome` is a closed text enum. `UNKNOWN` is the default so callers
-- can record a procedure they haven't seen the outcome of yet; the
-- outcome-feedback loop on `Memory.update_outcome` flips it later.
--
-- Embeddings for procedures use the existing `embeddings` table with
-- `item_kind = 'procedure'`. The migration extends the CHECK to allow
-- that value.

BEGIN;

CREATE TABLE procedures (
    id                  BLOB PRIMARY KEY,
    situation           TEXT NOT NULL,
    action              TEXT NOT NULL,
    outcome             TEXT NOT NULL DEFAULT 'unknown'
        CHECK (outcome IN ('success', 'partial', 'failure', 'unknown')),
    weight              REAL NOT NULL DEFAULT 1.0
        CHECK (weight >= 0.0 AND weight <= 1.0),
    reinforcement_count INTEGER NOT NULL DEFAULT 0 CHECK (reinforcement_count >= 0),
    corroboration_count INTEGER NOT NULL DEFAULT 0 CHECK (corroboration_count >= 0),
    contradiction_count INTEGER NOT NULL DEFAULT 0 CHECK (contradiction_count >= 0),
    last_decayed_at     TEXT,
    cold_at             TEXT,
    metadata            TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

-- Indexes mirror events / memory_items so retrieval / decay sweeps stay
-- on the same access patterns.
CREATE INDEX idx_procedures_created_at ON procedures(created_at);
CREATE INDEX idx_procedures_weight     ON procedures(weight);
CREATE INDEX idx_procedures_outcome    ON procedures(outcome);
CREATE INDEX idx_procedures_cold_at    ON procedures(cold_at) WHERE cold_at IS NOT NULL;

-- Embeddings table previously CHECK'd `item_kind IN ('event', 'memory_item')`.
-- SQLite can't ALTER a CHECK constraint in place, so we rebuild the table.
-- The `embeddings_new` -> swap -> recreate-indexes pattern is the standard
-- migration shape; we wrap it in the same BEGIN/COMMIT so the embeddings
-- table is never visible in an inconsistent state.
CREATE TABLE embeddings_new (
    id           BLOB PRIMARY KEY,
    item_id      BLOB NOT NULL,
    item_kind    TEXT NOT NULL CHECK (item_kind IN ('event', 'memory_item', 'procedure')),
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL CHECK (dim > 0),
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE(item_id, item_kind, model)
);

INSERT INTO embeddings_new (id, item_id, item_kind, model, dim, vector, created_at)
    SELECT id, item_id, item_kind, model, dim, vector, created_at FROM embeddings;

DROP TABLE embeddings;
ALTER TABLE embeddings_new RENAME TO embeddings;

CREATE INDEX idx_embeddings_item ON embeddings(item_id, item_kind);

INSERT INTO schema_migrations (version) VALUES (3);

COMMIT;
