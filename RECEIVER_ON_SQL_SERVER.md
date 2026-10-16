# Cart Result Receiver on Microsoft SQL Server

How to stand up the Result Receiver backed by **Microsoft SQL Server** instead of
SQLite, what the HTTP API looks like, and how the ML server talks to it.

**Status of this document.** Every table, column, route, status code and sample
payload below was read out of the running system (`cart_platform/receiver/`) or
captured from a real training run on 2026-10-13. The T-SQL was written for this
document and checked with a T-SQL parser (`sqlglot`, `tsql` dialect): 37 of 38
statements parse, and the 38th is the computed-column fragment in §4.3, which
was validated inside a complete `CREATE TABLE`. It has **not been executed
against a live SQL Server instance** — none was available in the build
environment. Treat §10 as the acceptance checklist to run on your own instance
before go-live.

---

## 1. Where SQL Server sits

```
┌────────────────────────┐        HTTP (JSON)        ┌──────────────────────────┐
│   ML server  :8100     │ ────────────────────────▶ │  Result Receiver :8200   │
│  cart_platform/server  │   POST /v1/results/...    │  cart_platform/receiver  │
│                        │                           │                          │
│  durable outbox        │ ◀──────────────────────── │  201 / 200 / 409         │
│  (platform.db)         │        ack + outcome      └────────────┬─────────────┘
└────────────────────────┘                                        │ pyodbc
                                                                  ▼
                                                     ┌──────────────────────────┐
                                                     │  Microsoft SQL Server    │
                                                     │  DB: CartResults         │
                                                     │  schema: cart            │
                                                     └──────────────────────────┘
```

Two rules that make this swap safe, both already true in the code:

1. The ML server **never** opens the receiver's database. It only makes HTTP
   calls. Changing SQLite to SQL Server therefore touches the receiver only —
   no training, scoring or backtesting code changes.
2. The receiver is **append-only for history**. An `event_id` that arrives twice
   with the same payload is a replay (`200`); the same `event_id` with a
   *different* payload is rejected (`409`). Historical results are never
   overwritten.

---

## 2. Prerequisites

| Item | Requirement | Notes |
|---|---|---|
| SQL Server | 2016 SP1 or later | Needs `OFFSET … FETCH` (2012+) and 1700-byte nonclustered index keys (2016+) |
| | Azure SQL / Managed Instance also fine | |
| ODBC driver | **ODBC Driver 18 for SQL Server** | Driver 17 works; adjust the connection string |
| Python package | `pyodbc >= 5.0` | `pip install pyodbc` |
| Collation | see §4.2 — **binary collation on identifier columns** | This is not optional; see the warning |
| Permissions | `db_datareader`, `db_datawriter`, plus `CREATE TABLE` for the initial deploy | The service account needs no `db_owner` at runtime |


Verify the driver is visible to Python:

```bash
python -c "import pyodbc; print(pyodbc.drivers())"
# expect: ['ODBC Driver 18 for SQL Server', ...]
```

---

## 3. Create the database, login and schema

Run as an administrator once:

```sql
-- 3.1 Database -------------------------------------------------------------
IF DB_ID(N'CartResults') IS NULL
    CREATE DATABASE CartResults;
GO

ALTER DATABASE CartResults SET READ_COMMITTED_SNAPSHOT ON;   -- readers never block writers
GO

USE CartResults;
GO

-- 3.2 Schema ---------------------------------------------------------------
IF SCHEMA_ID(N'cart') IS NULL
    EXEC(N'CREATE SCHEMA cart AUTHORIZATION dbo;');
GO

-- 3.3 Login + user for the receiver service --------------------------------
USE master;
GO
IF SUSER_ID(N'cart_receiver') IS NULL
    CREATE LOGIN cart_receiver WITH PASSWORD = N'<strong-password-from-your-vault>';
GO

USE CartResults;
GO
IF USER_ID(N'cart_receiver') IS NULL
    CREATE USER cart_receiver FOR LOGIN cart_receiver;
GO

ALTER ROLE db_datareader ADD MEMBER cart_receiver;
ALTER ROLE db_datawriter ADD MEMBER cart_receiver;
GRANT EXECUTE ON SCHEMA::cart TO cart_receiver;
GO
```

> Do **not** put the password in source control, a Dockerfile, or an `.env` that
> is committed. Read it from your secret store and export it at process start.

---

## 4. Table design

### 4.1 Type mapping

| SQLite (today) | SQL Server | Why |
|---|---|---|
| `TEXT` primary/foreign key | `NVARCHAR(450)` **BIN2 collation** | 450 × 2 bytes = the 900-byte index key limit |
| `TEXT` short attribute | `NVARCHAR(100…200)` | Measured max lengths are small (see below) |
| `TEXT` JSON blob | `NVARCHAR(MAX)` | Largest observed payload is 8,817 chars |
| `TEXT` ISO-8601 timestamp | `DATETIME2(3)` | Real range queries for `since` / `until` |
| `INTEGER PRIMARY KEY AUTOINCREMENT` | `BIGINT IDENTITY(1,1)` | |
| `INTEGER` 0/1 flag | `BIT` | |
| `REAL` | `FLOAT(53)` | |
| `TEXT` sha-256 hex | `CHAR(64)` | Fixed 64 hex chars, ASCII only |

Measured maximum lengths from a real run (38 events, 3 entities):

| Column | Observed max | Allocated |
|---|---|---|
| `event_id` | 69 | 450 |
| `event_type` | 21 | 50 |
| `entity_id` | 5 | 100 |
| `site_key` | 3 | 50 |
| `upload_id` | 21 | 100 |
| `job_id` | 27 | 100 |
| `model_id` | 31 | 200 |
| `payload_json` | 8,817 | `MAX` |

### 4.2 ⚠ Collation — the one thing that will silently corrupt your data

SQLite compares `TEXT` **case-sensitively**. The SQL Server default
(`SQL_Latin1_General_CP1_CI_AS`) is **case-insensitive**. Identifiers in this
system contain hex suffixes, e.g.:

```
mdl_300_2_20261013035036_c1f270
train_20261013043353_8cef49:215:0:training:candidate_ready:v1
```

Under a case-insensitive collation, two genuinely different ids that differ only
in letter case would be treated as the **same key** — a replay would be recorded
against the wrong model, or a legitimate event would be rejected as a `409`
conflict. Every identifier column below therefore carries an explicit
`COLLATE Latin1_General_BIN2`.

### 4.3 The `event_id` length limit

`event_id` is capped at 512 characters by the API contract
(`ResultEventIn.event_id`, `max_length=512`). A SQL Server **clustered** index
key cannot exceed 900 bytes, and `NVARCHAR(512)` is 1,024 bytes. The DDL below
uses `NVARCHAR(450)` (900 bytes exactly) and adds a `CHECK` so an over-long id
fails loudly instead of being truncated.

Real ids are ~70 characters, so there is roughly 6× headroom. If you ever need
the full 512, keep `event_id` as `NVARCHAR(512)` and make the primary key a
persisted hash instead:

```sql
event_key AS CONVERT(CHAR(64), HASHBYTES('SHA2_256', event_id), 2) PERSISTED
```

### 4.4 DDL — run this once, in order

```sql
USE CartResults;
GO

-- ====================================================================
-- 1. received_events — the immutable audit log. Every event lands here
--    first; the four projection tables are derived views of it.
-- ====================================================================
IF OBJECT_ID(N'cart.received_events', N'U') IS NULL
CREATE TABLE cart.received_events (
    event_id        NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    event_type      NVARCHAR(50)  COLLATE Latin1_General_BIN2 NOT NULL,
    schema_version  INT            NOT NULL,
    entity_id       NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key        NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    upload_id       NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    job_id          NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id        NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    occurred_at     DATETIME2(3)   NULL,
    received_at     DATETIME2(3)   NOT NULL
        CONSTRAINT DF_received_events_received_at DEFAULT (SYSUTCDATETIME()),
    payload_hash    CHAR(64)       NOT NULL,
    payload_json    NVARCHAR(MAX)  NOT NULL,
    CONSTRAINT PK_received_events PRIMARY KEY CLUSTERED (event_id),
    CONSTRAINT CK_received_events_payload_json
        CHECK (ISJSON(payload_json) = 1)
);
GO

CREATE NONCLUSTERED INDEX ix_events_entity
    ON cart.received_events (entity_id, occurred_at DESC);
CREATE NONCLUSTERED INDEX ix_events_type
    ON cart.received_events (event_type, occurred_at DESC);
CREATE NONCLUSTERED INDEX ix_events_job
    ON cart.received_events (job_id);
GO

-- ====================================================================
-- 2. training_results
-- ====================================================================
IF OBJECT_ID(N'cart.training_results', N'U') IS NULL
CREATE TABLE cart.training_results (
    id                    BIGINT IDENTITY(1,1) NOT NULL,
    event_id              NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    job_id                NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    entity_id             NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key              NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    upload_id             NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id              NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    status                NVARCHAR(50)   NULL,
    occurred_at           DATETIME2(3)   NULL,
    received_at           DATETIME2(3)   NULL,
    metrics_json          NVARCHAR(MAX)  NULL,
    metadata_json         NVARCHAR(MAX)  NULL,
    config_hash           NVARCHAR(64)   NULL,
    frozen_config_json    NVARCHAR(MAX)  NULL,
    effective_config_json NVARCHAR(MAX)  NULL,
    CONSTRAINT PK_training_results PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_training_results_event UNIQUE (event_id),
    CONSTRAINT FK_training_results_event FOREIGN KEY (event_id)
        REFERENCES cart.received_events (event_id)
);
GO

CREATE NONCLUSTERED INDEX ix_training_entity
    ON cart.training_results (entity_id, occurred_at DESC);
GO

-- ====================================================================
-- 3. calibration_results
-- ====================================================================
IF OBJECT_ID(N'cart.calibration_results', N'U') IS NULL
CREATE TABLE cart.calibration_results (
    id             BIGINT IDENTITY(1,1) NOT NULL,
    event_id       NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    job_id         NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    entity_id      NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key       NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    upload_id      NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    model_id       NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    calibrator_id  NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    status         NVARCHAR(50)   NULL,
    scope          NVARCHAR(50)   NULL,
    method         NVARCHAR(50)   NULL,
    occurred_at    DATETIME2(3)   NULL,
    received_at    DATETIME2(3)   NULL,
    metrics_json   NVARCHAR(MAX)  NULL,
    metadata_json  NVARCHAR(MAX)  NULL,
    config_hash    NVARCHAR(64)   NULL,
    CONSTRAINT PK_calibration_results PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_calibration_results_event UNIQUE (event_id),
    CONSTRAINT FK_calibration_results_event FOREIGN KEY (event_id)
        REFERENCES cart.received_events (event_id)
);
GO

CREATE NONCLUSTERED INDEX ix_calib_entity
    ON cart.calibration_results (entity_id, occurred_at DESC);
GO

-- ====================================================================
-- 4. backtest_results
-- ====================================================================
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

CREATE NONCLUSTERED INDEX ix_backtest_entity
    ON cart.backtest_results (entity_id, backtest_type, occurred_at DESC);
CREATE NONCLUSTERED INDEX ix_backtest_type
    ON cart.backtest_results (backtest_type);
GO

-- ====================================================================
-- 5. scoring_results — the high-volume table
-- ====================================================================
IF OBJECT_ID(N'cart.scoring_results', N'U') IS NULL
CREATE TABLE cart.scoring_results (
    id                 BIGINT IDENTITY(1,1) NOT NULL,
    event_id           NVARCHAR(450) COLLATE Latin1_General_BIN2 NOT NULL,
    entity_id          NVARCHAR(100) COLLATE Latin1_General_BIN2 NULL,
    site_key           NVARCHAR(50)  COLLATE Latin1_General_BIN2 NULL,
    model_id           NVARCHAR(200) COLLATE Latin1_General_BIN2 NULL,
    decision_time      DATETIME2(3)   NULL,
    occurred_at        DATETIME2(3)   NULL,
    received_at        DATETIME2(3)   NULL,
    shortage_risk      FLOAT(53)      NULL,
    alert_state        NVARCHAR(20)   NULL,
    should_replenish   BIT            NULL,
    quantity           INT            NULL,
    result_json        NVARCHAR(MAX)  NULL,
    warnings_json      NVARCHAR(MAX)  NULL,
    config_hash        NVARCHAR(64)   NULL,
    frozen_config_json NVARCHAR(MAX)  NULL,
    CONSTRAINT PK_scoring_results PRIMARY KEY CLUSTERED (id),
    CONSTRAINT UQ_scoring_results_event UNIQUE (event_id),
    CONSTRAINT FK_scoring_results_event FOREIGN KEY (event_id)
        REFERENCES cart.received_events (event_id)
);
GO

CREATE NONCLUSTERED INDEX ix_scoring_entity
    ON cart.scoring_results (entity_id, decision_time DESC);

-- One decision per (entity, decision_time, model). This is the database-level
-- guarantee that a retried score cannot create a second order.
CREATE UNIQUE NONCLUSTERED INDEX ux_scoring_decision
    ON cart.scoring_results (entity_id, decision_time, model_id)
    WHERE entity_id IS NOT NULL AND decision_time IS NOT NULL AND model_id IS NOT NULL;
GO
```

> The filtered `WHERE` clause on `ux_scoring_decision` reproduces SQLite's
> behaviour, where `NULL`s do not collide in a unique index. SQL Server treats
> `NULL` as a value in unique indexes, so without the filter a second event with
> a `NULL` `model_id` would be rejected.

---

## 5. Point the receiver at SQL Server

Only **three** places in the receiver are SQLite-specific. Everything else
already uses `?` placeholders, which pyodbc also uses.

| File | Line | SQLite | SQL Server |
|---|---|---|---|
| `receiver/database.py` | `connect()` | `sqlite3.connect(path)` | `pyodbc.connect(dsn)` |
| `receiver/database.py` | `_migrate()` | `PRAGMA table_info(t)` | `sys.columns` query |
| `receiver/store.py` | `_filtered()` | `LIMIT ? OFFSET ?` | `OFFSET ? ROWS FETCH NEXT ? ROWS ONLY` |

### 5.1 Connection

```python
import os, pyodbc

def _dsn() -> str:
    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.environ['CART_MSSQL_HOST']},{os.getenv('CART_MSSQL_PORT', '1433')};"
        f"DATABASE={os.getenv('CART_MSSQL_DB', 'CartResults')};"
        f"UID={os.environ['CART_MSSQL_USER']};"
        f"PWD={os.environ['CART_MSSQL_PASSWORD']};"
        "Encrypt=yes;TrustServerCertificate=no;"
        "MultipleActiveResultSets=False;"
    )

def connect():
    conn = pyodbc.connect(_dsn(), autocommit=False, timeout=10)
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf-8")
    conn.setencoding(encoding="utf-8")
    return conn
```

For Windows integrated auth, replace `UID`/`PWD` with `Trusted_Connection=yes`.

### 5.2 Pagination

```python
rows = query(
    f"SELECT * FROM cart.{table} WHERE {clause} "
    f"ORDER BY {order_col} DESC, id DESC "
    f"OFFSET ? ROWS FETCH NEXT ? ROWS ONLY",
    tuple(params) + (offset, limit),      # note: offset first, then limit
)
```

The argument order is **reversed** relative to `LIMIT ? OFFSET ?`. Getting this
wrong silently returns the wrong page rather than erroring.

### 5.3 Column-existence check (replaces `PRAGMA table_info`)

```python
def _columns(conn, table: str) -> set[str]:
    cur = conn.execute(
        "SELECT c.name FROM sys.columns c "
        "JOIN sys.tables t ON t.object_id = c.object_id "
        "JOIN sys.schemas s ON s.schema_id = t.schema_id "
        "WHERE s.name = ? AND t.name = ?", ("cart", table))
    return {r[0] for r in cur.fetchall()}
```

### 5.4 Idempotent insert under concurrency

SQLite is single-writer, so the current `SELECT`-then-`INSERT` cannot race. SQL
Server **can**. Let the unique constraint arbitrate and catch the violation:

```python
try:
    cur.execute(
        "INSERT INTO cart.received_events "
        "(event_id, event_type, schema_version, entity_id, site_key, upload_id, "
        " job_id, model_id, occurred_at, received_at, payload_hash, payload_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,SYSUTCDATETIME(),?,?)", params)
except pyodbc.IntegrityError as exc:
    if "2627" in str(exc) or "2601" in str(exc):    # PK / unique violation
        existing = query_one(
            "SELECT payload_hash FROM cart.received_events WHERE event_id = ?",
            (event_id,))
        return (REPLAY if existing["payload_hash"] == digest else CONFLICT), existing
    raise
```

This preserves the exact three-outcome contract (`created` / `replay` /
`conflict`) that the ML server's outbox depends on.

### 5.5 Timestamps

The API sends ISO-8601 UTC strings such as `2026-10-13T04:34:30.463Z`. pyodbc
will not bind the trailing `Z`. Convert on the way in and format on the way out:

```python
from datetime import datetime, timezone

def to_dt(s):                       # inbound: str -> datetime
    if s is None: return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(
        timezone.utc).replace(tzinfo=None)      # DATETIME2 is naive UTC

def to_iso(dt):                     # outbound: datetime -> str
    if dt is None: return None
    return dt.replace(tzinfo=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")
```

**Store UTC only.** Never let the server's local time zone touch these columns.
A behaviour change to be aware of: `since` / `until` filters become real
datetime comparisons instead of string comparisons — more correct, but callers
who relied on lexical prefix matching will see different results.

---

## 6. The API — how your server talks to it

### 6.1 Write endpoints

All four accept the **same envelope** and differ only in the `event_type` they
will accept.

| Method | Path | Required `event_type` |
|---|---|---|
| POST | `/v1/results/training` | `training.completed` |
| POST | `/v1/results/calibration` | `calibration.completed` |
| POST | `/v1/results/backtest` | `backtest.completed` (also requires `payload.backtest_type`) |
| POST | `/v1/results/scoring` | `scoring.recorded` |

### 6.2 The envelope

```jsonc
{
  "event_id":       "string, 1..512 chars, REQUIRED — the idempotency key",
  "event_type":     "string, REQUIRED — must match the route",
  "payload":        { },          // object, REQUIRED — category-specific
  "schema_version": 1,            // int, default 1
  "occurred_at":    "2026-10-13T04:34:30.463Z",
  "entity_id":      "215:0",
  "site_key":       "163",
  "upload_id":      "upl_20261013_002b9ca1",
  "job_id":         "train_20261013043353_8cef49",
  "model_id":       "mdl_215_0_20261013043402_5f1924"
}
```

Everything except `event_id`, `event_type` and `payload` is nullable.

### 6.3 Response codes

| Code | Meaning | What the ML server does |
|---|---|---|
| **201** | Created — first time this `event_id` was seen | Marks the outbox row `sent` |
| **200** | Replay — same `event_id`, byte-identical payload | Marks the outbox row `sent` (success) |
| **409** | Conflict — same `event_id`, **different** payload | Dead-letters the row; does **not** retry. History is never overwritten |
| **422** | `event_type` does not match the route, or `backtest_type` missing | Dead-letters (non-retryable) |
| **401** | Missing/invalid bearer token | Dead-letters (non-retryable) |
| **408, 425, 429, 500, 502, 503, 504** | Transient | Retries with backoff, honouring `Retry-After` |

Success body:

```json
{
  "event_id": "train_20261013043353_8cef49:215:0:training:candidate_ready:v1",
  "outcome": "created",
  "stored_record": {
    "event_id": "...", "event_type": "training.completed",
    "entity_id": "215:0", "job_id": "train_...", "model_id": "mdl_...",
    "received_at": "2026-10-13T04:34:30.512Z"
  }
}
```

### 6.4 How `event_id` is built — your idempotency key

Deterministic by construction, so a retry always produces the same id:

```
training     {job_id}:{entity_id}:training:{status}:v{schema_version}
calibration  {job_id}:{entity_id}:calibration:{calibration_id}:v{schema_version}
backtest     {job_id}:{entity_id}:{backtest_type}:{status}:v{schema_version}
scoring      score:{entity_id}:{decision_time}:{model_id}:v{schema_version}
```

Real examples:

```
train_20261013043353_8cef49:215:0:training:candidate_ready:v1
train_20261013043353_8cef49:215:0:calibration:cal_215_0_d02625ed:v1
train_20261013043353_8cef49:215:0:policy_replay:completed:v1
score:300:2:2026-10-11T23:43:52:mdl_300_2_20261013035036_c1f270:v1
```

Note `entity_id` is `KDVR:ChanNo`, so **colons appear inside the id**. Never
parse an `event_id` by splitting on `:` — treat it as an opaque key.

### 6.5 Payload shapes (observed keys from a real run)

| Event | Payload keys |
|---|---|
| `training.completed` | `status`, `error`, `detail`, `n_features`, `selected_models`, `frozen_thresholds`, `split_windows`, `metrics`, `config_snapshot`, `config_hash`, `effective_config`, `promotable`, `blocking_reasons`, `promoted`, `warnings`, `artifacts`, `caveat` |
| `calibration.completed` | `calibrator_id`, `status`, `scope`, `method`, `metrics`, `window_start`, `window_end`, `calibration_red`, `config_hash`, `warnings`, `artifacts`, `caveat` |
| `backtest.completed` | `backtest_type`, `report_id`, `status`, `reason`, `metrics`, `assumptions`, `frozen_model_config`, `effective_operational_config`, `config_hash`, `artifacts`, `window_start`, `window_end`, `caveat` |
| `scoring.recorded` | `decision_time`, `current_cart`, `simulated_cart`, `pending_carts`, `forecast`, `shortage_risk`, `alert_state`, `new_predictive_alert`, `reactive_critical`, `action`, `resolved_config`, `frozen_model_config`, `frozen_thresholds`, `config_hash`, `calibration_scope`, `warnings`, `artifact`, `caveat` |

Only metadata and server-local URI references are ever published — **never a
pickled model**. `artifacts` entries look like
`{"kind": "model_metadata_json", "uri": "logs/training/.../metrics.json"}`.

### 6.6 Field-by-field storage map — what lands where

The receiver stores your event in **two layers**:

| Layer | Table | What goes in | Purpose |
|---|---|---|---|
| 1. Audit log | `cart.received_events` | The envelope + the **entire payload verbatim** as `payload_json` | Nothing is ever dropped; the complete event is replayable |
| 2. Projection | `cart.training_results`, `calibration_results`, `backtest_results`, `scoring_results` | A **selected subset** promoted to typed, indexed columns | Fast filtering and joins without parsing JSON |

So: **every field your ML server sends is persisted.** A field that has no
dedicated column is still queryable via `JSON_VALUE` / `OPENJSON` against the
JSON column, just more slowly.

#### 6.6.1 Envelope → `cart.received_events` (all four event types)

| Your field | Example | DB column | Type |
|---|---|---|---|
| `event_id` | `train_...:215:0:training:candidate_ready:v1` | `event_id` | `NVARCHAR(450)` PK |
| `event_type` | `training.completed` | `event_type` | `NVARCHAR(50)` |
| `schema_version` | `1` | `schema_version` | `INT` |
| `entity_id` | `215:0` | `entity_id` | `NVARCHAR(100)` |
| `site_key` | `163` | `site_key` | `NVARCHAR(50)` |
| `upload_id` | `upl_20261013_002b9ca1` | `upload_id` | `NVARCHAR(100)` |
| `job_id` | `train_20261013043353_8cef49` | `job_id` | `NVARCHAR(100)` |
| `model_id` | `mdl_215_0_20261013043402_5f1924` | `model_id` | `NVARCHAR(200)` |
| `occurred_at` | `2026-10-13T04:34:30.463Z` | `occurred_at` | `DATETIME2(3)` |
| `payload` (whole object) | `{...}` | `payload_json` | `NVARCHAR(MAX)` |
| — *server-generated* | arrival time | `received_at` | `DATETIME2(3)` |
| — *server-generated* | SHA-256 of canonical payload | `payload_hash` | `CHAR(64)` |

`payload_hash` is what decides replay vs conflict: same `event_id` + same hash →
`200`; same `event_id` + different hash → `409`.

#### 6.6.2 `training.completed` → `cart.training_results`

| Your payload field | DB column | Notes |
|---|---|---|
| `status` | `status` | e.g. `candidate_ready`, `invalid_config`, `failed` |
| `metrics` | `metrics_json` | Per-target object, keys `y_cart_3`, `y_cart_6`, `y_min_3`, `y_min_6` |
| `config_hash` | `config_hash` | |
| `frozen_thresholds` | `frozen_config_json` | **Fallback chain:** `frozen_model_config` → `frozen_thresholds` |
| `effective_config` | `effective_config_json` | **Fallback chain:** `effective_operational_config` → `effective_config_at_training` → `effective_config` |
| *entire payload* | `metadata_json` | Full copy, so nothing is lost |
| envelope `job_id`, `entity_id`, `site_key`, `upload_id`, `model_id`, `occurred_at` | same names | Taken from the envelope, **not** the payload |

JSON-only (no dedicated column): `error`, `detail`, `n_features`,
`selected_models`, `split_windows`, `config_snapshot`, `promotable`,
`blocking_reasons`, `promoted`, `warnings`, `artifacts`, `caveat`.

```sql
SELECT entity_id,
       JSON_VALUE(frozen_config_json, '$.red_model')        AS red_model,
       JSON_VALUE(metadata_json,      '$.n_features')       AS n_features,
       JSON_VALUE(metadata_json,      '$.promotable')       AS promotable,
       JSON_VALUE(metrics_json,       '$.y_cart_3.selected') AS model_for_cart_3
FROM cart.training_results;
```

#### 6.6.3 `calibration.completed` → `cart.calibration_results`

| Your payload field | DB column |
|---|---|
| `calibrator_id` | `calibrator_id` |
| `status` | `status` |
| `scope` | `scope` |
| `method` | `method` (e.g. `split_conformal`) |
| `metrics` | `metrics_json` (`brier`, `ece`, `coverage`, `n_samples`, `n_positive`) |
| `config_hash` | `config_hash` |
| *entire payload* | `metadata_json` |
| envelope `job_id`, `entity_id`, `site_key`, `upload_id`, `model_id`, `occurred_at` | same names |

JSON-only: `window_start`, `window_end`, `calibration_red`, `warnings`,
`artifacts`, `caveat`.

> Note this table has **no** `frozen_config_json` / `effective_config_json`
> columns — only `config_hash`. Join back to `training_results` on
> `config_hash` to recover the full configuration.

#### 6.6.4 `backtest.completed` → `cart.backtest_results`

The richest mapping — most fields are promoted to columns.

| Your payload field | DB column |
|---|---|
| `backtest_type` | `backtest_type` **(required; 422 if missing)** |
| `report_id` | `report_id` |
| `status` | `status` |
| `reason` | `reason` |
| `metrics` | `metrics_json` |
| `assumptions` | `assumptions_json` |
| `caveat` | `caveat` |
| `artifacts` | `artifacts_json` |
| `config_hash` | `config_hash` |
| `frozen_model_config` | `frozen_config_json` |
| `effective_operational_config` | `effective_config_json` |
| envelope `job_id`, `entity_id`, `site_key`, `upload_id`, `model_id`, `occurred_at` | same names |

JSON-only: `window_start`, `window_end`.

```sql
SELECT entity_id, backtest_type, status,
       JSON_VALUE(frozen_config_json,    '$.red_model')     AS frozen_red,
       JSON_VALUE(effective_config_json, '$.red_threshold') AS operational_red,
       JSON_VALUE(effective_config_json, '$.target_stock')  AS target_stock,
       JSON_VALUE(metrics_json, '$.attribution.prevented_onset') AS prevented,
       JSON_VALUE(metrics_json, '$.attribution.historical_events') AS historical
FROM cart.backtest_results;
```

#### 6.6.5 `scoring.recorded` → `cart.scoring_results`

| Your payload field | DB column | Notes |
|---|---|---|
| `decision_time` | `decision_time` | Also part of the unique key |
| `shortage_risk` | `shortage_risk` | `FLOAT(53)` |
| `alert_state` | `alert_state` | `NORMAL` / `AMBER` / `RED` |
| `action.should_replenish` | `should_replenish` | **Flattened** from the nested `action` object, cast to `BIT` |
| `action.quantity` | `quantity` | **Flattened**, `INT` |
| `warnings` | `warnings_json` | |
| `config_hash` | `config_hash` | |
| `frozen_model_config` | `frozen_config_json` | Wins over `frozen_thresholds` when both are sent |
| *entire payload* | `result_json` | |
| envelope `entity_id`, `site_key`, `model_id`, `occurred_at` | same names | |

JSON-only: `current_cart`, `simulated_cart`, `pending_carts`, `forecast`,
`new_predictive_alert`, `reactive_critical`, `resolved_config`,
`frozen_thresholds`, `calibration_scope`, `artifact`, `caveat`, and the rest of
`action` (`reason`, `urgency`, `order_id`, `minutes_to_predicted_breach`,
`expected_completion_time`, `actionable_before_breach`).

⚠ **`scoring_results` deliberately has no `job_id` or `upload_id` column** — a
scoring event is not produced by a training job, and the ML server sends both as
`null` in the envelope.

```sql
SELECT TOP (50) entity_id, decision_time, shortage_risk, alert_state,
       should_replenish, quantity,
       JSON_VALUE(result_json, '$.action.reason')          AS reason,
       JSON_VALUE(result_json, '$.action.urgency')         AS urgency,
       JSON_VALUE(result_json, '$.forecast.cart_at_20min') AS cart_at_20min,
       JSON_VALUE(result_json, '$.resolved_config.red_threshold.value') AS red_used
FROM cart.scoring_results
ORDER BY decision_time DESC;
```

#### 6.6.6 Summary — promoted columns per event type

| | training | calibration | backtest | scoring |
|---|---|---|---|---|
| Envelope → columns | 6 | 6 | 6 | 4 |
| Payload → own column | 5 | 6 | 11 | 8 |
| Full-payload JSON column | `metadata_json` | `metadata_json` | — | `result_json` |
| Config columns | hash + frozen + effective | hash only | hash + frozen + effective | hash + frozen |

The `backtest_results` table has no full-payload copy because every field is
either promoted to a column or part of `metrics_json` / `assumptions_json`. The
complete original is always available in `received_events.payload_json`.

### 6.7 A complete POST example

```bash
curl -i -X POST https://results.internal:8200/v1/results/backtest \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $CART_RESULT_RECEIVER_TOKEN" \
  -d '{
    "event_id": "train_20261013043353_8cef49:300:2:policy_replay:completed:v1",
    "event_type": "backtest.completed",
    "schema_version": 1,
    "occurred_at": "2026-10-13T04:34:30.462Z",
    "entity_id": "300:2",
    "site_key": "177",
    "upload_id": "upl_20261013_002b9ca1",
    "job_id": "train_20261013043353_8cef49",
    "model_id": "mdl_300_2_20261013035036_c1f270",
    "payload": {
      "backtest_type": "policy_replay",
      "report_id": "bt-300_2-20261013-043430-ab12c",
      "status": "completed",
      "config_hash": "549777393e8a3846",
      "frozen_model_config":  {"red_model": 25, "amber_model": 35},
      "effective_operational_config": {
        "red_threshold": 25, "amber_threshold": 35,
        "bay_capacity": 120, "target_stock": 50, "lead_time_min": 10.0
      },
      "metrics": {"alerts": 14, "trips": 7, "carts": 124},
      "caveat": "simulation estimate, conditional on assumptions"
    }
  }'
```

### 6.8 Read endpoints

| Method | Path | Filters |
|---|---|---|
| GET | `/v1/results/training` | `entity_id`, `site_key`, `upload_id`, `job_id`, `model_id`, `status`, `since`, `until`, `limit`, `offset` |
| GET | `/v1/results/calibration` | as above plus `scope` |
| GET | `/v1/results/backtest` | as above plus `backtest_type` |
| GET | `/v1/results/scoring` | `entity_id`, `site_key`, `model_id`, `since`, `until`, `limit`, `offset` |
| GET | `/v1/results/entities/{entity_id}` | per-entity roll-up across all four categories |
| GET | `/v1/results/events/{event_id}` | the raw stored event, full payload |
| GET | `/health` | liveness + row counts |

`limit` is 1–500 (default 50); `offset` ≥ 0. Response:

```json
{ "total": 12, "limit": 50, "offset": 0, "items": [ ... ] }
```

Examples:

```bash
curl "https://results.internal:8200/v1/results/backtest?entity_id=300:2&backtest_type=policy_replay"
curl "https://results.internal:8200/v1/results/scoring?entity_id=215:0&since=2026-10-13T00:00:00Z&limit=100"
curl "https://results.internal:8200/v1/results/entities/300:2"
```

Because `entity_id` contains a colon, the route uses a `:path` converter — pass
`300:2` directly, unencoded.

---

## 7. How the ML server delivers (no change needed)

Publishing is **transactional**: a result is written to the ML server's own
`result_outbox` table in the same transaction that records the result. A
background worker then drains it.

| Setting | Env var | Default |
|---|---|---|
| Receiver URL | `CART_RESULT_RECEIVER_URL` | `http://127.0.0.1:8200` |
| Enabled | `CART_RESULT_RECEIVER_ENABLED` | `1` |
| Bearer token | `CART_RESULT_RECEIVER_TOKEN` | unset (auth off) |
| Timeout | `CART_RESULT_PUBLISH_TIMEOUT_SEC` | `10` |
| Max attempts | `CART_RESULT_PUBLISH_MAX_ATTEMPTS` | `8` |
| Base backoff | `CART_RESULT_PUBLISH_BACKOFF_SEC` | `2` |
| Max backoff | `CART_RESULT_PUBLISH_MAX_BACKOFF_SEC` | `300` |
| Worker interval | `CART_RESULT_WORKER_INTERVAL_SEC` | `5` |

Behaviour you can rely on:

- A receiver outage **never** fails a scoring or training request — events queue.
- Exponential backoff with jitter; `Retry-After` is honoured when present.
- Non-retryable responses (`4xx` other than the retryable set) are dead-lettered
  rather than hammered.
- In-flight rows are recovered after an ML-server restart.

Operate it with:

```bash
curl http://127.0.0.1:8100/v1/integrations/outbox
curl -X POST http://127.0.0.1:8100/v1/integrations/outbox/retry \
  -H 'Content-Type: application/json' -d '{"flush_now": true}'
```

---

## 8. Setup runbook

```bash
# 1. Database objects (as admin, once)
sqlcmd -S sqlserver.internal -U sa -i 01_create_database.sql
sqlcmd -S sqlserver.internal -U sa -d CartResults -i 02_create_tables.sql

# 2. Receiver host
pip install pyodbc
export CART_MSSQL_HOST=sqlserver.internal
export CART_MSSQL_PORT=1433
export CART_MSSQL_DB=CartResults
export CART_MSSQL_USER=cart_receiver
export CART_MSSQL_PASSWORD="$(vault read -field=password secret/cart/receiver)"
export CART_RESULT_RECEIVER_TOKEN="$(vault read -field=token secret/cart/receiver)"
export CART_RECEIVER_HOST=0.0.0.0
export CART_RECEIVER_PORT=8200

cd cart_platform && bash receiver/run_server.sh

# 3. ML server host — point it at the receiver
export CART_RESULT_RECEIVER_URL=https://results.internal:8200
export CART_RESULT_RECEIVER_ENABLED=1
export CART_RESULT_RECEIVER_TOKEN="$(vault read -field=token secret/cart/receiver)"

cd cart_platform && bash server/run_server.sh

# 4. Smoke test
curl https://results.internal:8200/health
curl -X POST http://127.0.0.1:8100/v1/integrations/outbox/retry \
  -H 'Content-Type: application/json' -d '{"flush_now": true}'
```

---

## 9. Verification queries

```sql
USE CartResults;

-- Did anything arrive?
SELECT event_type, COUNT(*) AS n, MAX(received_at) AS latest
FROM cart.received_events GROUP BY event_type;

-- Idempotency holding? (the two numbers must be equal)
SELECT COUNT(*) AS rows_stored, COUNT(DISTINCT event_id) AS distinct_events
FROM cart.received_events;

-- Each entity's frozen thresholds, from the training events
SELECT entity_id, config_hash,
       JSON_VALUE(frozen_config_json, '$.red_model')   AS red_model,
       JSON_VALUE(frozen_config_json, '$.amber_model') AS amber_model,
       occurred_at
FROM cart.training_results
ORDER BY entity_id, occurred_at DESC;

-- All three backtests present per entity, with the config they ran under
SELECT entity_id, backtest_type, status, config_hash,
       JSON_VALUE(effective_config_json, '$.target_stock')  AS target_stock,
       JSON_VALUE(effective_config_json, '$.bay_capacity')  AS bay_capacity
FROM cart.backtest_results
ORDER BY entity_id, backtest_type;

-- Recent decisions
SELECT TOP (20) entity_id, decision_time, shortage_risk, alert_state,
       should_replenish, quantity, config_hash
FROM cart.scoring_results
ORDER BY decision_time DESC;

-- Orphan check: a projection row with no audit row (should return 0)
SELECT COUNT(*) FROM cart.backtest_results b
LEFT JOIN cart.received_events e ON e.event_id = b.event_id
WHERE e.event_id IS NULL;
```

Expected shape after training three entities (measured on the real system):

| Table | Rows |
|---|---|
| `received_events` | 3 training + 3 calibration + 9 backtest + N scoring |
| `training_results` | 3 — one per entity, three distinct `config_hash` values |
| `backtest_results` | 9 — three types × three entities |

---

## 10. Acceptance checklist

Run these on your instance; none have been executed here.

- [ ] DDL applies cleanly; all indexes build
- [ ] `ISJSON` check accepts a real payload
- [ ] POST the same event twice → `201` then `200`, one row stored
- [ ] POST the same `event_id` with a changed payload → `409`, original intact
- [ ] POST to the wrong route → `422`
- [ ] POST without the bearer token (token configured) → `401`
- [ ] Two ids differing only in case remain distinct (collation check)
- [ ] `OFFSET/FETCH` paging returns correct pages at `offset=0` and `offset=50`
- [ ] `since` / `until` filters return the expected window
- [ ] Stop SQL Server mid-run: scoring still succeeds, events queue, and drain
      after recovery
- [ ] An 8,817-character payload round-trips without truncation

---

## 11. Operational notes and risks

**Sizing.** Mean payload 4.2 KB, max 8.8 KB. Scoring dominates volume: one event
per decision per entity. At a 6.7-minute median interval that is ~215
decisions/entity/day, so 1,000 entities ≈ 215k rows/day ≈ **900 MB/day** in
`received_events`. Plan retention before go-live.

**Retention.** `received_events` is the audit log and grows fastest. Consider
monthly partitioning on `received_at`, or archiving events older than N months
while keeping the four projection tables. Do not delete projection rows that
backtest reports still reference.

**Compression.** `ALTER INDEX … REBUILD WITH (DATA_COMPRESSION = PAGE)` on
`scoring_results` and `received_events` typically pays for itself given the
repetitive JSON.

**Security.** Use `Encrypt=yes` with a real certificate; `TrustServerCertificate=yes`
disables validation and should not reach production. The receiver's bearer token
and the SQL password are independent secrets — rotating one does not rotate the
other.

**Known risks.**

1. The T-SQL here has not been executed against a live instance. §10 exists for
   that reason.
2. Collation (§4.2) is the highest-impact detail. A case-insensitive deployment
   will appear to work and will quietly conflate ids.
3. `decision_time` in `ux_scoring_decision` is now `DATETIME2(3)`. If any caller
   sends sub-millisecond precision, two decisions could round into one. Observed
   `decision_time` values are second-resolution, so this is currently safe.
4. The `since`/`until` semantics change from string to datetime comparison.
5. Entity ids contain colons; anything that parses ids by splitting will break.
