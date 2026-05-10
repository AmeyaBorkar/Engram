-- Engram core schema.
--
-- Tables:
--   events            - immutable raw observations
--   clusters          - groupings produced by consolidation
--   memory_items      - hierarchy (event / summary / abstraction)
--   embeddings        - dense vectors for events and memory items
--   provenance_links  - memory_item -> event support
--
-- Conventions:
--   IDs are 16-byte BLOBs (UUIDv7), giving time-ordered locality on insert.
--   Timestamps are ISO-8601 strings in UTC.
--   JSON metadata is stored as TEXT; deserialized at the python boundary.

BEGIN;

CREATE TABLE events (
    id           BLOB PRIMARY KEY,
    content      TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    source       TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX idx_events_created_at ON events(created_at);
CREATE INDEX idx_events_source     ON events(source) WHERE source IS NOT NULL;

CREATE TABLE clusters (
    id           BLOB PRIMARY KEY,
    cohesion     REAL NOT NULL CHECK (cohesion >= 0.0 AND cohesion <= 1.0),
    created_at   TEXT NOT NULL
);

CREATE TABLE memory_items (
    id           BLOB PRIMARY KEY,
    level        TEXT NOT NULL CHECK (level IN ('event', 'summary', 'abstraction')),
    content      TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0 AND weight <= 1.0),
    cluster_id   BLOB REFERENCES clusters(id) ON DELETE SET NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX idx_memory_items_created_at ON memory_items(created_at);
CREATE INDEX idx_memory_items_weight     ON memory_items(weight);
CREATE INDEX idx_memory_items_level      ON memory_items(level);
CREATE INDEX idx_memory_items_cluster_id ON memory_items(cluster_id) WHERE cluster_id IS NOT NULL;

CREATE TABLE embeddings (
    id           BLOB PRIMARY KEY,
    item_id      BLOB NOT NULL,
    item_kind    TEXT NOT NULL CHECK (item_kind IN ('event', 'memory_item')),
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL CHECK (dim > 0),
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE(item_id, item_kind, model)
);
CREATE INDEX idx_embeddings_item ON embeddings(item_id, item_kind);

CREATE TABLE provenance_links (
    id              BLOB PRIMARY KEY,
    memory_item_id  BLOB NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    event_id        BLOB NOT NULL REFERENCES events(id)        ON DELETE RESTRICT,
    weight          REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at      TEXT NOT NULL,
    UNIQUE(memory_item_id, event_id)
);
CREATE INDEX idx_provenance_memory_item ON provenance_links(memory_item_id);
CREATE INDEX idx_provenance_event       ON provenance_links(event_id);

INSERT INTO schema_migrations (version) VALUES (1);

COMMIT;
