# receiver_net — Cart Result pipeline on SQL Server + ASP.NET Core

A C# / ASP.NET Core 8 port of the Cart result pipeline, backed by **Microsoft
SQL Server** instead of SQLite. It covers **both ends**:

| Service | Port | Role | Python original |
|---|---|---|---|
| **ReceiverNet** | 8200 | receives results, stores them | `cart_platform/receiver/` |
| **PublisherHost** | 8300 | durable outbox, delivers results | `cart_platform/integrations/` |

```
producer ──▶ PublisherHost :8300 ──HTTP──▶ ReceiverNet :8200 ──▶ SQL Server
             (cart.result_outbox)          (cart.received_events + 4 projections)
```

Both are wire-compatible with the Python originals, so you can adopt either end
on its own. To keep the Python ML server and swap only the receiver:

```bash
export CART_RESULT_RECEIVER_URL=http://127.0.0.1:8200
```

No training, scoring or backtesting code changes. Nothing ever opens the
receiver's database directly; it is reached only over HTTP.

---

## Layout

```
receiver_net/
├── ReceiverNet.sln
├── sql/
│   ├── 01_create_database.sql   database, schema [cart], login cart_receiver
│   ├── 02_create_tables.sql     receiver tables + indexes (idempotent)
│   ├── 03_verification.sql      read-only smoke checks for SSMS
│   └── 04_create_outbox.sql     sender-side cart.result_outbox
├── src/
│   ├── CartEvents/              shared: envelope, routes, event ids, canonical hashing
│   ├── ReceiverNet/             the receiver service (port 8200)
│   │   ├── Program.cs           host, DI, Swagger, SQL-error middleware
│   │   ├── Configuration/       ReceiverOptions
│   │   ├── Models/              ResultEventIn, StoreAck, PageOut, ResultQuery
│   │   ├── Data/                connection factory, startup schema check
│   │   ├── Services/            EventStore (idempotent writes + queries)
│   │   ├── Security/            BearerTokenFilter (write routes only)
│   │   └── Endpoints/           HealthEndpoints, ResultsEndpoints
│   ├── CartPublisher/           outbox library: store, HTTP publisher, retry worker
│   └── PublisherHost/           the publisher service (port 8300)
├── requests.http                ready-to-send samples for both services
├── run_server.sh                start the receiver
└── run_publisher.sh             start the publisher
```

### Python → C# map

| Python | C# |
|---|---|
| `receiver/schema.sql` | `sql/02_create_tables.sql` |
| `receiver/database.py` | `ReceiverNet/Data/*.cs` |
| `receiver/store.py` | `ReceiverNet/Services/EventStore.cs` |
| `receiver/app.py` | `ReceiverNet/Program.cs`, `Endpoints/*`, `Models/*`, `Security/*` |
| `integrations/result_events.py` | `CartEvents/ResultEvent.cs` (+ `CanonicalJson.cs`) |
| `registry/schema.sql` → `result_outbox` | `sql/04_create_outbox.sql` |
| `integrations/outbox.py` | `CartPublisher/Outbox.cs` |
| `integrations/publisher.py` | `CartPublisher/ResultPublisher.cs`, `OutboxRetryWorker.cs` |
| `server/ondemand.py` → outbox routes | `CartPublisher/OutboxEndpoints.cs` |

---

## 1. Create the database (SSMS)

Open **SQL Server Management Studio**, connect to the server, then run in order:

1. `sql/01_create_database.sql` — creates `CartResults`, schema `cart`, and the
   `cart_receiver` login. **Replace `<strong-password-from-your-vault>` first.**
2. `sql/02_create_tables.sql` — creates the five receiver tables and indexes.
   Idempotent, safe to re-run.
3. `sql/04_create_outbox.sql` — creates `cart.result_outbox` for the sender.
   Only needed if you run PublisherHost.
4. `sql/03_verification.sql` — optional read-only checks.

From the command line instead:

```bash
sqlcmd -S localhost,1433 -U sa -P "$SA_PASSWORD" -C -i sql/01_create_database.sql
sqlcmd -S localhost,1433 -U sa -P "$SA_PASSWORD" -C -d CartResults -i sql/02_create_tables.sql
sqlcmd -S localhost,1433 -U sa -P "$SA_PASSWORD" -C -d CartResults -i sql/04_create_outbox.sql
```

### Tables

| Table | Holds |
|---|---|
| `cart.received_events` | immutable audit log — every event lands here first |
| `cart.training_results` | projection of `training.completed` |
| `cart.calibration_results` | projection of `calibration.completed` |
| `cart.backtest_results` | projection of `backtest.completed` |
| `cart.scoring_results` | projection of `scoring.recorded` (high volume) |
| `cart.result_outbox` | **sender side** — the delivery queue |

Two design points that are **not** optional:

- **`COLLATE Latin1_General_BIN2` on every identifier column.** SQLite compares
  text case-sensitively; the SQL Server default does not. Identifiers here carry
  hex suffixes (`mdl_300_2_20261013035036_c1f270`), so a case-insensitive
  collation would let two distinct ids collide — a replay recorded against the
  wrong model, or a valid event rejected as a 409.
- **`ux_scoring_decision` is a *filtered* unique index.** SQL Server treats
  `NULL` as a value in unique indexes, SQLite does not. The
  `WHERE ... IS NOT NULL` clause restores SQLite's behaviour.

---

## 2. Run the API

Requires the [.NET 8 SDK](https://dotnet.microsoft.com/download).

```bash
export CART_RECEIVER_SQL_CONNECTION='Server=localhost,1433;Database=CartResults;User Id=cart_receiver;Password=***;Encrypt=True;TrustServerCertificate=True'
export CART_RESULT_RECEIVER_TOKEN='...'   # optional; omit to disable auth

./run_server.sh                            # http://127.0.0.1:8200/docs
```

Or directly:

```bash
dotnet run --project src/ReceiverNet/ReceiverNet.csproj --urls http://127.0.0.1:8200
```

Swagger UI is at `/docs`.

### Configuration

| Setting | Environment variable | Default |
|---|---|---|
| `Receiver:ConnectionString` | `CART_RECEIVER_SQL_CONNECTION` | *(required)* |
| `Receiver:Token` | `CART_RESULT_RECEIVER_TOKEN` | unset → auth disabled |
| `Receiver:Schema` | — | `cart` |
| `Receiver:AutoMigrate` | — | `false` (`true` in Development) |
| `Receiver:DefaultLimit` / `MaxLimit` | — | `50` / `500` |

Environment variables override `appsettings.json`. Never commit a password or
token; `appsettings.Local.json` and `*.env` are git-ignored.

`AutoMigrate` runs `02_create_tables.sql` at startup — convenient locally, but
leave it off in production and deploy the script through SSMS, since the runtime
account needs only `db_datareader` + `db_datawriter`.

---

## 3. The API

### Write — POST

| Method | Path | Required `event_type` |
|---|---|---|
| POST | `/v1/results/training` | `training.completed` |
| POST | `/v1/results/calibration` | `calibration.completed` |
| POST | `/v1/results/backtest` | `backtest.completed` (plus `payload.backtest_type`) |
| POST | `/v1/results/scoring` | `scoring.recorded` |

Envelope:

```json
{
  "event_id": "train_...:300:2:training:candidate_ready:v1",
  "event_type": "training.completed",
  "schema_version": 1,
  "occurred_at": "2026-10-13T04:33:53.120Z",
  "entity_id": "300:2",
  "site_key": "177",
  "upload_id": "upl_...",
  "job_id": "train_...",
  "model_id": "mdl_...",
  "payload": { }
}
```

| Status | Meaning |
|---|---|
| `201` | created — first time this `event_id` was seen |
| `200` | replay — same `event_id`, identical payload |
| `409` | conflict — same `event_id`, **different** payload; nothing is overwritten |
| `401` | missing/invalid bearer token (only when a token is configured) |
| `422` | `event_type` does not match the route, or a contract field is invalid |
| `503` | SQL Server unavailable — the ML server's outbox retries on 5xx |

Idempotency is decided by a SHA-256 of the canonically serialised payload
(sorted keys, no whitespace, non-ASCII escaped), which reproduces Python's
`json.dumps(..., sort_keys=True, separators=(",", ":"))`. An existing
`results_receiver.db` can therefore be migrated here without replays turning
into conflicts.

### Read — GET

| Method | Path | Filters |
|---|---|---|
| GET | `/v1/results/training` | `entity_id`, `site_key`, `upload_id`, `job_id`, `model_id`, `status`, `since`, `until`, `limit`, `offset` |
| GET | `/v1/results/calibration` | as above plus `scope` |
| GET | `/v1/results/backtest` | as above plus `backtest_type` |
| GET | `/v1/results/scoring` | `entity_id`, `site_key`, `model_id`, `since`, `until`, `limit`, `offset` |
| GET | `/v1/results/entities/{entity_id}` | per-entity roll-up across all four categories |
| GET | `/v1/results/events/{event_id}` | the raw stored event with its full payload |
| GET | `/health` | liveness, database target, rows stored |

`limit` is 1–500 (default 50), `offset` ≥ 0. List responses:

```json
{ "total": 12, "limit": 50, "offset": 0, "items": [] }
```

`entity_id` contains a colon and is passed unencoded: `/v1/results/entities/300:2`.

```bash
curl "http://127.0.0.1:8200/v1/results/backtest?entity_id=300:2&backtest_type=policy_replay"
curl "http://127.0.0.1:8200/v1/results/scoring?entity_id=300:2&since=2026-10-13T00:00:00Z&limit=100"
```

See `requests.http` for runnable examples of every route.

### No update route — by design

The receiver is **append-only**, matching the Python original: history is never
rewritten. Correcting a result means emitting a new event with a new `event_id`;
re-POSTing an old `event_id` with changed content returns `409`.

---

## 4. The sender — PublisherHost (durable outbox)

Needed only if your producer is **not** the Python ML server. The Python side
already has its own outbox; point it at the receiver and you are done.

```bash
export CART_OUTBOX_SQL_CONNECTION='Server=localhost,1433;Database=CartResults;User Id=cart_receiver;Password=***;Encrypt=True;TrustServerCertificate=True'
export CART_RESULT_RECEIVER_URL='http://127.0.0.1:8200'
export CART_RESULT_RECEIVER_TOKEN='...'   # must match the receiver's token

./run_publisher.sh                         # http://127.0.0.1:8300/docs
```

### Why an outbox

An event is written to `cart.result_outbox` in the same breath as the local
result it describes, and a background worker delivers it later. Receiver
downtime therefore cannot fail a training or scoring request, and a restart
loses nothing.

A row moves `pending → sending → sent`, or to `dead_letter` once it is
beyond saving. `UQ_result_outbox_event` makes enqueue idempotent: the same
`event_id` can never be queued twice.

### Delivery rules (identical to `publisher.py`)

| Receiver says | Publisher does |
|---|---|
| `200` / `201` | mark `sent` |
| `409` conflict | `dead_letter` — the receiver will never accept it |
| `408, 425, 429, 500, 502, 503, 504` | retry with backoff; `Retry-After` wins when present |
| any other `4xx` | `dead_letter` |
| network error / timeout | retry with backoff |
| unknown `event_type` | `dead_letter`, without sending |

Backoff is `min(base × 2^(attempt-1), max)` with ±50% jitter, up to
`MaxAttempts` (default 8), after which the row is dead-lettered.

### Sender API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/integrations/outbox` | queue an event (`202` queued, `200` already queued) |
| GET | `/v1/integrations/outbox` | counts, `worker_running`, recent events (`status`, `entity_id`, `event_type`, `limit`) |
| GET | `/v1/integrations/outbox/{event_id}` | one event with its delivery history |
| POST | `/v1/integrations/outbox/flush` | deliver a batch now |
| POST | `/v1/integrations/outbox/retry` | requeue dead letters (all, or one `event_id`) then flush |
| GET | `/health` | queue depth, worker state, receiver target |

In-process alternative for a .NET producer: reference `CartPublisher`, call
`services.AddCartPublisher(Configuration)`, then `Outbox.EnqueueAsync(...)` in
the same transaction as the result you are recording.

### Sender configuration

| Setting | Environment variable | Default |
|---|---|---|
| `Publisher:ConnectionString` | `CART_OUTBOX_SQL_CONNECTION` | *(required)* |
| `Publisher:ReceiverUrl` | `CART_RESULT_RECEIVER_URL` | `http://127.0.0.1:8200` |
| `Publisher:Token` | `CART_RESULT_RECEIVER_TOKEN` | unset |
| `Publisher:Enabled` | `CART_RESULT_RECEIVER_ENABLED` | `true` |
| `Publisher:MaxAttempts` | — | `8` |
| `Publisher:BaseBackoffSeconds` / `MaxBackoffSeconds` | — | `2` / `300` |
| `Publisher:WorkerIntervalSeconds` | — | `5` |
| `Publisher:BatchSize` | — | `20` |
| `Publisher:StuckAfterSeconds` | — | `120` |
| `Publisher:RunWorker` | — | `true` (set `false` to enqueue only) |

---

## 5. Notes

- Timestamps are stored as `DATETIME2(3)` in UTC and returned as
  `2026-10-13T04:33:53.120Z`, matching the Python receiver's wire format.
- The envelope row and its projection row are written in one transaction, so a
  failure cannot leave an event without its projection.
- A duplicate-key race between two workers is caught and re-classified as
  replay or conflict rather than failing the request.
- The outbox claim is a single `UPDATE … OUTPUT` over a `TOP/ORDER BY` CTE with
  `READPAST, UPDLOCK, ROWLOCK`, so several publisher instances can share one
  queue without blocking or double-sending. The Python version relied on
  SQLite's single writer and could not do this.
- `result_outbox.claimed_at` has no SQLite counterpart: the original detected
  stuck rows using `created_at` and so could resurrect a row that was still
  legitimately in flight.
- The retry worker swallows every iteration error. Letting one escape would
  stop the whole host under .NET's default
  `BackgroundServiceExceptionBehavior.StopHost`.
- Background on the SQL Server migration, field-by-field storage maps and the
  go-live checklist: [`RECEIVER_ON_SQL_SERVER.md`](../RECEIVER_ON_SQL_SERVER.md).
