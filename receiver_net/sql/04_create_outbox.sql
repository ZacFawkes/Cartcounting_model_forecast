/* ==========================================================================
   04_create_outbox.sql
   The SENDER side: a durable transactional outbox.

   Run this against the database used by whatever produces result events
   (the ML server). It may be CartResults or a separate database - the outbox
   belongs to the sender, not to the receiver.

   Port of the result_outbox table in cart_platform/registry/schema.sql.

   Why an outbox: an event is written here in the same breath as the local
   result it describes, and a background worker delivers it over HTTP. Receiver
   downtime therefore cannot fail training or scoring, and nothing is lost when
   the sending process restarts.

   Idempotent and safe to re-run.
   ========================================================================== */

USE CartResults;
GO

IF SCHEMA_ID(N'cart') IS NULL
    EXEC(N'CREATE SCHEMA cart AUTHORIZATION dbo;');
GO

IF OBJECT_ID(N'cart.result_outbox', N'U') IS NULL
CREATE TABLE cart.result_outbox (
    id              BIGINT IDENTITY(1,1) NOT NULL,
    event_id        NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    event_type      NVARCHAR(50)  COLLATE Latin1_General_BIN2 NOT NULL,
    destination     NVARCHAR(500) NOT NULL,
    entity_id       NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    job_id          NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id        NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    payload_json    NVARCHAR(MAX) NOT NULL,
    status          NVARCHAR(20)  NOT NULL
        CONSTRAINT DF_result_outbox_status DEFAULT (N'pending'),
    attempts        INT           NOT NULL
        CONSTRAINT DF_result_outbox_attempts DEFAULT (0),
    next_attempt_at DATETIME2(3)  NULL,
    claimed_at      DATETIME2(3)  NULL,
    last_error      NVARCHAR(MAX) NULL,
    response_code   INT           NULL,
    created_at      DATETIME2(3)  NOT NULL
        CONSTRAINT DF_result_outbox_created_at DEFAULT (SYSUTCDATETIME()),
    sent_at         DATETIME2(3)  NULL,
    CONSTRAINT PK_result_outbox PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_result_outbox_event UNIQUE (event_id),
    CONSTRAINT CK_result_outbox_status CHECK (status IN
        (N'pending', N'sending', N'sent', N'dead_letter')),
    CONSTRAINT CK_result_outbox_payload CHECK (ISJSON(payload_json) = 1)
);
GO

/* UQ_result_outbox_event is what makes enqueue idempotent: the same event_id
   can never be queued twice, so a retried training run cannot deliver the same
   result as two separate events. */

/* The worker's claim query: status + due time, ordered by id. */
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_outbox_status'
                 AND object_id = OBJECT_ID(N'cart.result_outbox'))
    CREATE NONCLUSTERED INDEX ix_outbox_status
        ON cart.result_outbox (status, next_attempt_at)
        INCLUDE (attempts);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'ix_outbox_entity'
                 AND object_id = OBJECT_ID(N'cart.result_outbox'))
    CREATE NONCLUSTERED INDEX ix_outbox_entity
        ON cart.result_outbox (entity_id, created_at DESC);
GO

/* claimed_at has no counterpart in the SQLite original, which detected stuck
   rows using created_at and so could resurrect a row that was legitimately
   still in flight. Stamping the claim time makes recovery exact. */


/* --------------------------------------------------------------------------
   Operational queries
   -------------------------------------------------------------------------- */

-- Queue depth by status
-- SELECT status, COUNT(*) AS n FROM cart.result_outbox GROUP BY status;

-- Anything stuck or failing
-- SELECT TOP (50) id, event_id, status, attempts, response_code, last_error,
--        next_attempt_at, created_at
-- FROM   cart.result_outbox
-- WHERE  status IN (N'dead_letter', N'sending')
-- ORDER  BY id DESC;

-- Requeue every dead letter (the API does this via POST .../outbox/retry)
-- UPDATE cart.result_outbox
-- SET    status = N'pending', attempts = 0,
--        next_attempt_at = SYSUTCDATETIME(), last_error = NULL
-- WHERE  status = N'dead_letter';
