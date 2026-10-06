-- ============================================================
-- Migration 001: retraining_events outcome + error, UTC convention
-- ============================================================
--
-- Run once against an EXISTING Sentinel-AI database, BEFORE deploying
-- the application version that writes retraining_events.status:
--
--     psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/001_retraining_events_status.sql
--
-- Idempotent and additive only. Safe when:
--   * the table exists and the columns are missing  -> columns added, rows labelled
--   * the columns already exist                     -> every statement is a no-op
--   * the database is brand new                     -> Sentinel.sql already creates
--                                                     the full table; this is a no-op
-- Nothing is dropped, recreated, renamed or retyped.
--
-- NOT VERIFIED against a live PostgreSQL from this repository: run it on
-- a dev database first.
--
-- Wrapped in one transaction so a failure leaves the table untouched.

BEGIN;


-- ------------------------------------------------------------
-- 1. Columns
--
-- status: one of 'promoted', 'rejected', 'failed'.
--   DEFAULT NULL on purpose: a writer that omits status cannot be
--   labelled truthfully by a default (promoted and rejected rows would
--   look alike), so NULL means "not labelled yet" and step 3 labels it.
--   The application always writes status explicitly.
-- error_message: nullable; set only for status = 'failed', already
--   sanitized and bounded (<= 500 chars) by the orchestrator.
-- ------------------------------------------------------------

ALTER TABLE retraining_events
    ADD COLUMN IF NOT EXISTS status TEXT DEFAULT NULL
        CHECK (status IN ('promoted', 'rejected', 'failed')),
    ADD COLUMN IF NOT EXISTS error_message TEXT
        CHECK (char_length(error_message) <= 500);


-- ------------------------------------------------------------
-- 2. OPTIONAL, NOT DEFAULT: normalize historical triggered_at to UTC
--
-- Convention: retraining_events.triggered_at represents UTC.
-- New rows get an explicit UTC timestamp from the orchestrator. Rows
-- written before this migration used DEFAULT NOW(), i.e. the server's
-- TimeZone setting at the time.
--
-- The docker-compose postgres:15 service sets no TZ/PGTZ, so the
-- official image's default (UTC) most likely applied and NO conversion
-- is needed. Confirm on the live server before doing anything:
--
--     SHOW timezone;
--
-- Only if the old rows were written under a known non-UTC zone, replace
-- <OLD_SERVER_TZ> (e.g. 'Asia/Kolkata') and uncomment. It must run here,
-- BEFORE step 3: until then, status IS NULL selects exactly the legacy
-- DEFAULT NOW() rows (new code always writes status), and after step 3
-- it matches nothing, so a re-run cannot shift a row twice.
-- If the old zone cannot be established, leave this commented and treat
-- legacy rows as "UTC, unverified".
--
-- UPDATE retraining_events
-- SET    triggered_at = (triggered_at AT TIME ZONE '<OLD_SERVER_TZ>')
--                      AT TIME ZONE 'UTC'
-- WHERE  status IS NULL;
-- ------------------------------------------------------------


-- ------------------------------------------------------------
-- 3. Label historical rows from existing data (deterministic)
--
-- Every row written before status existed was a completed attempt, so
-- promoted TRUE -> 'promoted', FALSE -> 'rejected'. Never 'failed', and
-- error_message is left NULL. Rows already labelled are untouched.
-- ------------------------------------------------------------

UPDATE retraining_events
SET    status = CASE WHEN promoted THEN 'promoted' ELSE 'rejected' END
WHERE  status IS NULL
  AND  promoted IS NOT NULL;


-- ------------------------------------------------------------
-- 4. New-row default in the same time basis as the convention
--
-- The application always passes triggered_at; this only affects a
-- writer that omits it, which would otherwise get server-local NOW().
-- ------------------------------------------------------------

ALTER TABLE retraining_events
    ALTER COLUMN triggered_at SET DEFAULT (NOW() AT TIME ZONE 'UTC');


-- ------------------------------------------------------------
-- 5. Document the convention in the catalog
-- ------------------------------------------------------------

COMMENT ON COLUMN retraining_events.triggered_at IS
    'Start of the retraining attempt. Represents UTC wall-clock time '
    '(TIMESTAMP WITHOUT TIME ZONE). Rows written before migration 001 '
    'used server-local NOW(); see migrations/001_retraining_events_status.sql.';

COMMENT ON COLUMN retraining_events.status IS
    'promoted | rejected | failed. NULL only for unlabelled legacy rows.';

COMMENT ON COLUMN retraining_events.error_message IS
    'Failed attempts only: sanitized (secrets redacted), single line, <= 500 chars.';


COMMIT;
