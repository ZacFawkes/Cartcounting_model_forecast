/* ==========================================================================
   03_verification.sql
   Smoke checks to run in SSMS after deploying 01 + 02 and sending traffic.
   Read-only: safe to run on production.
   ========================================================================== */

USE CartResults;
GO

-- 1. All five tables exist --------------------------------------------------
SELECT TABLE_NAME
FROM   INFORMATION_SCHEMA.TABLES
WHERE  TABLE_SCHEMA = N'cart'
ORDER  BY TABLE_NAME;
-- Expect: backtest_results, calibration_results, received_events,
--         scoring_results, training_results

-- 2. Identifier columns really are binary-collated --------------------------
SELECT c.TABLE_NAME, c.COLUMN_NAME, c.COLLATION_NAME
FROM   INFORMATION_SCHEMA.COLUMNS c
WHERE  c.TABLE_SCHEMA = N'cart'
  AND  c.COLUMN_NAME IN (N'event_id', N'entity_id', N'model_id', N'job_id',
                         N'upload_id', N'site_key', N'event_type')
ORDER  BY c.TABLE_NAME, c.COLUMN_NAME;
-- Every row must read Latin1_General_BIN2

-- 3. Event counts by type ---------------------------------------------------
SELECT event_type, COUNT(*) AS events, MAX(received_at) AS newest
FROM   cart.received_events
GROUP  BY event_type
ORDER  BY event_type;

-- 4. Every event has its projection row (no orphans) ------------------------
SELECT e.event_type, COUNT(*) AS missing_projection
FROM   cart.received_events e
LEFT   JOIN cart.training_results    t ON t.event_id = e.event_id
LEFT   JOIN cart.calibration_results c ON c.event_id = e.event_id
LEFT   JOIN cart.backtest_results    b ON b.event_id = e.event_id
LEFT   JOIN cart.scoring_results     s ON s.event_id = e.event_id
WHERE  COALESCE(t.event_id, c.event_id, b.event_id, s.event_id) IS NULL
GROUP  BY e.event_type;
-- Expect: zero rows

-- 5. No duplicate decisions (the ux_scoring_decision guarantee) -------------
SELECT entity_id, decision_time, model_id, COUNT(*) AS n
FROM   cart.scoring_results
WHERE  entity_id IS NOT NULL AND decision_time IS NOT NULL AND model_id IS NOT NULL
GROUP  BY entity_id, decision_time, model_id
HAVING COUNT(*) > 1;
-- Expect: zero rows

-- 6. Stored payloads are valid JSON -----------------------------------------
SELECT COUNT(*) AS invalid_payloads
FROM   cart.received_events
WHERE  ISJSON(payload_json) <> 1;
-- Expect: 0

-- 7. Latest activity per entity ---------------------------------------------
SELECT entity_id,
       COUNT(*)            AS events,
       MIN(received_at)    AS first_seen,
       MAX(received_at)    AS last_seen
FROM   cart.received_events
WHERE  entity_id IS NOT NULL
GROUP  BY entity_id
ORDER  BY last_seen DESC;

-- 8. Index health -----------------------------------------------------------
SELECT OBJECT_NAME(i.object_id) AS table_name, i.name AS index_name,
       i.type_desc, i.is_unique, i.has_filter
FROM   sys.indexes i
WHERE  i.object_id IN (OBJECT_ID(N'cart.received_events'),
                       OBJECT_ID(N'cart.training_results'),
                       OBJECT_ID(N'cart.calibration_results'),
                       OBJECT_ID(N'cart.backtest_results'),
                       OBJECT_ID(N'cart.scoring_results'))
  AND  i.name IS NOT NULL
ORDER  BY table_name, index_name;
