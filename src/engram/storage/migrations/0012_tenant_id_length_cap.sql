-- Cap `tenant_id` at 256 characters.
--
-- The 0006 migration added `tenant_id TEXT` to events / memory_items /
-- procedures with no length cap.  A caller passing a multi-megabyte
-- string (mistakenly using the field for free-form metadata, an
-- attacker exhausting disk via the public surface) would bloat every
-- row in the table.
--
-- SQLite doesn't support `ALTER TABLE ... ADD CONSTRAINT CHECK`, and
-- rebuilding three tables for a single CHECK is invasive for a
-- non-structural bound.  TRIGGERs on BEFORE INSERT / BEFORE UPDATE
-- raise an explicit error when the length exceeds 256 — equivalent
-- enforcement, no rebuild.
--
-- 256 is well above any sensible tenant identifier (UUID = 36, slug =
-- ~64, account id = ~32).  Callers exceeding this cap are doing
-- something wrong and the trigger will tell them so.

BEGIN;

-- One pair (insert + update) per table, three tables total.

CREATE TRIGGER trg_events_tenant_id_length_insert
    BEFORE INSERT ON events
    WHEN NEW.tenant_id IS NOT NULL AND length(NEW.tenant_id) > 256
BEGIN
    SELECT RAISE(ABORT, 'tenant_id exceeds 256 characters');
END;

CREATE TRIGGER trg_events_tenant_id_length_update
    BEFORE UPDATE ON events
    WHEN NEW.tenant_id IS NOT NULL AND length(NEW.tenant_id) > 256
BEGIN
    SELECT RAISE(ABORT, 'tenant_id exceeds 256 characters');
END;

CREATE TRIGGER trg_memory_items_tenant_id_length_insert
    BEFORE INSERT ON memory_items
    WHEN NEW.tenant_id IS NOT NULL AND length(NEW.tenant_id) > 256
BEGIN
    SELECT RAISE(ABORT, 'tenant_id exceeds 256 characters');
END;

CREATE TRIGGER trg_memory_items_tenant_id_length_update
    BEFORE UPDATE ON memory_items
    WHEN NEW.tenant_id IS NOT NULL AND length(NEW.tenant_id) > 256
BEGIN
    SELECT RAISE(ABORT, 'tenant_id exceeds 256 characters');
END;

CREATE TRIGGER trg_procedures_tenant_id_length_insert
    BEFORE INSERT ON procedures
    WHEN NEW.tenant_id IS NOT NULL AND length(NEW.tenant_id) > 256
BEGIN
    SELECT RAISE(ABORT, 'tenant_id exceeds 256 characters');
END;

CREATE TRIGGER trg_procedures_tenant_id_length_update
    BEFORE UPDATE ON procedures
    WHEN NEW.tenant_id IS NOT NULL AND length(NEW.tenant_id) > 256
BEGIN
    SELECT RAISE(ABORT, 'tenant_id exceeds 256 characters');
END;

INSERT INTO schema_migrations (version) VALUES (12);

COMMIT;
