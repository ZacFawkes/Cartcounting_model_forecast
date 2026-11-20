/* ==========================================================================
   02_create_tables.sql
   Run ONCE against the CartResults database, after 01_create_database.sql.

   Faithful port of cart_platform/receiver/schema.sql (SQLite) to T-SQL.

   Type mapping
     TEXT key / identifier  -> NVARCHAR(n) COLLATE Latin1_General_BIN2
     TEXT JSON blob         -> NVARCHAR(MAX)
     TEXT ISO-8601 instant  -> DATETIME2(3)
     INTEGER PK AUTOINCREMENT -> BIGINT IDENTITY(1,1)
     REAL                   -> FLOAT(53)
     sha-256 hex            -> CHAR(64)

   WHY THE BIN2 COLLATION IS MANDATORY
     SQLite compares TEXT case-sensitively; the SQL Server default
     (SQL_Latin1_General_CP1_CI_AS) does not. Identifiers here carry hex
     suffixes (mdl_300_2_20261013035036_c1f270), so under a case-insensitive
     collation two distinct ids could collide: a replay would be recorded
     against the wrong model, or a valid event rejected as a 409 conflict.

   This script is idempotent and safe to re-run.
   ========================================================================== */

USE CartResults;
GO

/* ====================================================================
   1. received_events - the immutable audit log. Every event lands here
      first; the four projection tables are derived from it.
   ==================================================================== */
IF OBJECT_ID(N'cart.received_events', N'U') IS NULL
CREATE TABLE cart.received_events (
    event_id        NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    event_type      NVARCHAR(50)  COLLATE Latin1_General_BIN2 NOT NULL,
    schema_version  INT           NOT NULL,
    entity_id       NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key        NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    upload_id       NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    job_id          NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id        NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    occurred_at     DATETIME2(3)  NULL,
    received_at     DATETIME2(3)  NOT NULL
        CONSTRAINT DF_received_events_received_at DEFAULT (SYSUTCDATETIME()),
    payload_hash    CHAR(64)      NOT NULL,
    payload_json    NVARCHAR(MAX) NOT NULL,
    CONSTRAINT PK_received_events PRIMARY KEY CLUSTERED (event_id),
    CONSTRAINT CK_received_events_payload_json CHECK (ISJSON(payload_json) = 1)
);
GO

/* event_id is capped at 512 chars by the API contract, but a clustered index
   key may not exceed 900 bytes and NVARCHAR(512) is 1024. NVARCHAR(450) is
   exactly 900 bytes; real ids are ~70 chars, so there is ~6x headroom. */

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_events_entity'
                 AND object_id = OBJECT_ID(N'cart.received_events'))
    CREATE NONCLUSTERED INDEX ix_events_entity
        ON cart.received_events (entity_id, occurred_at DESC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_events_type'
                 AND object_id = OBJECT_ID(N'cart.received_events'))
    CREATE NONCLUSTERED INDEX ix_events_type
        ON cart.received_events (event_type, occurred_at DESC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_events_job'
                 AND object_id = OBJECT_ID(N'cart.received_events'))
    CREATE NONCLUSTERED INDEX ix_events_job
        ON cart.received_events (job_id);
GO


/* ====================================================================
   2. training_results
   ==================================================================== */
IF OBJECT_ID(N'cart.training_results', N'U') IS NULL
CREATE TABLE cart.training_results (
    id                    BIGINT IDENTITY(1,1) NOT NULL,
    event_id              NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    job_id                NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    entity_id             NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key              NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    upload_id             NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id              NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    status                NVARCHAR(50)  NULL,
    occurred_at           DATETIME2(3)  NULL,
    received_at           DATETIME2(3)  NULL,
    metrics_json          NVARCHAR(MAX) NULL,
    metadata_json         NVARCHAR(MAX) NULL,
    config_hash           NVARCHAR(64)  NULL,
    frozen_config_json    NVARCHAR(MAX) NULL,
    effective_config_json NVARCHAR(MAX) NULL,
    CONSTRAINT PK_training_results PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_training_results_event UNIQUE (event_id),
    CONSTRAINT FK_training_results_event FOREIGN KEY (event_id)
        REFERENCES cart.received_events (event_id)
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_training_entity'
                 AND object_id = OBJECT_ID(N'cart.training_results'))
    CREATE NONCLUSTERED INDEX ix_training_entity
        ON cart.training_results (entity_id, occurred_at DESC);
GO


/* ====================================================================
   3. calibration_results
   ==================================================================== */
IF OBJECT_ID(N'cart.calibration_results', N'U') IS NULL
CREATE TABLE cart.calibration_results (
    id            BIGINT IDENTITY(1,1) NOT NULL,
    event_id      NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    job_id        NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    entity_id     NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key      NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    upload_id     NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id      NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    calibrator_id NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    status        NVARCHAR(50)  NULL,
    scope         NVARCHAR(50)  NULL,
    method        NVARCHAR(50)  NULL,
    occurred_at   DATETIME2(3)  NULL,
    received_at   DATETIME2(3)  NULL,
    metrics_json  NVARCHAR(MAX) NULL,
    metadata_json NVARCHAR(MAX) NULL,
    config_hash   NVARCHAR(64)  NULL,
    CONSTRAINT PK_calibration_results PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_calibration_results_event UNIQUE (event_id),
    CONSTRAINT FK_calibration_results_event FOREIGN KEY (event_id)
        REFERENCES cart.received_events (event_id)
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_calib_entity'
                 AND object_id = OBJECT_ID(N'cart.calibration_results'))
    CREATE NONCLUSTERED INDEX ix_calib_entity
        ON cart.calibration_results (entity_id, occurred_at DESC);
GO


/* ====================================================================
   4. backtest_results
   ==================================================================== */
IF OBJECT_ID(N'cart.backtest_results', N'U') IS NULL
CREATE TABLE cart.backtest_results (
    id                    BIGINT IDENTITY(1,1) NOT NULL,
    event_id              NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    job_id                NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    entity_id             NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key              NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    upload_id             NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id              NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    report_id             NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    backtest_type         NVARCHAR(50)   NOT NULL,
    status                NVARCHAR(50)   NULL,
    reason                NVARCHAR(1000) NULL,
    occurred_at           DATETIME2(3)   NULL,
    received_at           DATETIME2(3)   NULL,
    metrics_json          NVARCHAR(MAX)  NULL,
    assumptions_json      NVARCHAR(MAX)  NULL,
    caveat                NVARCHAR(1000) NULL,
    artifacts_json        NVARCHAR(MAX)  NULL,
    config_hash           NVARCHAR(64)   NULL,
    frozen_config_json    NVARCHAR(MAX)  NULL,
    effective_config_json NVARCHAR(MAX)  NULL,
    CONSTRAINT PK_backtest_results PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_backtest_results_event UNIQUE (event_id),
    CONSTRAINT FK_backtest_results_event FOREIGN KEY (event_id)
        REFERENCES cart.received_events (event_id),
    CONSTRAINT CK_backtest_results_type CHECK (backtest_type IN
        (N'forecast_accuracy', N'policy_replay', N'closed_loop_robustness'))
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_backtest_entity'
                 AND object_id = OBJECT_ID(N'cart.backtest_results'))
    CREATE NONCLUSTERED INDEX ix_backtest_entity
        ON cart.backtest_results (entity_id, backtest_type, occurred_at DESC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_backtest_type'
                 AND object_id = OBJECT_ID(N'cart.backtest_results'))
    CREATE NONCLUSTERED INDEX ix_backtest_type
        ON cart.backtest_results (backtest_type);
GO


/* ====================================================================
   5. scoring_results - the high-volume table
   ==================================================================== */
IF OBJECT_ID(N'cart.scoring_results', N'U') IS NULL
CREATE TABLE cart.scoring_results (
    id                    BIGINT IDENTITY(1,1) NOT NULL,
    event_id              NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    request_id            NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    entity_id             NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    kdvr                  INT           NULL,
    chan_no               INT           NULL,
    site_key              NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    decision_time         DATETIME2(3)  NULL,
    current_cart          FLOAT(53)     NULL,
    simulated_cart        FLOAT(53)     NULL,
    pred_cart_20min       FLOAT(53)     NULL,
    pred_cart_40min       FLOAT(53)     NULL,
    pred_min_20min        FLOAT(53)     NULL,
    pred_min_40min        FLOAT(53)     NULL,
    lower_bound_min_40min FLOAT(53)     NULL,
    shortage_risk         FLOAT(53)     NULL,
    alert_state           NVARCHAR(20)  NULL,
    quantity              INT           NULL,
    model_id              NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    calibration_id        NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    scored_at             DATETIME2(3)  NULL,
    CONSTRAINT PK_scoring_results PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_scoring_results_event UNIQUE (event_id),
    CONSTRAINT FK_scoring_results_event FOREIGN KEY (event_id)
        REFERENCES cart.received_events (event_id)
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_scoring_entity'
                 AND object_id = OBJECT_ID(N'cart.scoring_results'))
    CREATE NONCLUSTERED INDEX ix_scoring_entity
        ON cart.scoring_results (entity_id, decision_time DESC);
GO

/* One decision per (entity, decision_time, model): the database-level
   guarantee that a retried score cannot create a second order.

   The filtered WHERE clause reproduces SQLite, where NULLs do not collide in
   a unique index. SQL Server treats NULL as a value, so without the filter a
   second event with a NULL model_id would be rejected. */
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ux_scoring_decision'
                 AND object_id = OBJECT_ID(N'cart.scoring_results'))
    CREATE UNIQUE NONCLUSTERED INDEX ux_scoring_decision
        ON cart.scoring_results (entity_id, decision_time, model_id)
        WHERE entity_id IS NOT NULL
          AND decision_time IS NOT NULL
          AND model_id IS NOT NULL;
GO
