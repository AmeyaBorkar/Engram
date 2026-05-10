-- Stage 4 — Decay state.
--
-- Every item that the decay engine touches needs four extra columns:
--   weight                     — running [0,1] score
--   reinforcement_count        — non-negative integer counter
--   corroboration_count        — non-negative integer counter
--   contradiction_count        — non-negative integer counter
--   last_decayed_at            — ISO-8601 UTC; the dt input to the formula
--                                is `now - last_decayed_at`
--   cold_at                    — ISO-8601 UTC; non-NULL once the item has
--                                dropped below the prune threshold
--
-- `events` did not have `weight` in the v1 schema (events were treated as
-- immutable observations); Stage 4 adds it because retrieval at the flat
-- level operates on events directly, and decay must apply there.
--
-- `memory_items` already has `weight` from v1. We extend it with the same
-- counter / timestamp columns.
--
-- Backfill rule: existing rows have last_decayed_at = created_at (events)
-- or updated_at (memory_items). cold_at stays NULL.

BEGIN;

ALTER TABLE events ADD COLUMN weight              REAL    NOT NULL DEFAULT 1.0
    CHECK (weight >= 0.0 AND weight <= 1.0);
ALTER TABLE events ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0
    CHECK (reinforcement_count >= 0);
ALTER TABLE events ADD COLUMN corroboration_count INTEGER NOT NULL DEFAULT 0
    CHECK (corroboration_count >= 0);
ALTER TABLE events ADD COLUMN contradiction_count INTEGER NOT NULL DEFAULT 0
    CHECK (contradiction_count >= 0);
ALTER TABLE events ADD COLUMN last_decayed_at     TEXT;
ALTER TABLE events ADD COLUMN cold_at             TEXT;

UPDATE events SET last_decayed_at = created_at WHERE last_decayed_at IS NULL;

CREATE INDEX idx_events_weight  ON events(weight);
CREATE INDEX idx_events_cold_at ON events(cold_at) WHERE cold_at IS NOT NULL;

ALTER TABLE memory_items ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0
    CHECK (reinforcement_count >= 0);
ALTER TABLE memory_items ADD COLUMN corroboration_count INTEGER NOT NULL DEFAULT 0
    CHECK (corroboration_count >= 0);
ALTER TABLE memory_items ADD COLUMN contradiction_count INTEGER NOT NULL DEFAULT 0
    CHECK (contradiction_count >= 0);
ALTER TABLE memory_items ADD COLUMN last_decayed_at     TEXT;
ALTER TABLE memory_items ADD COLUMN cold_at             TEXT;

UPDATE memory_items SET last_decayed_at = updated_at WHERE last_decayed_at IS NULL;

CREATE INDEX idx_memory_items_cold_at ON memory_items(cold_at) WHERE cold_at IS NOT NULL;

INSERT INTO schema_migrations (version) VALUES (2);

COMMIT;
