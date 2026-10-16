# Result Receiver (:8200) — Backend Logic Specification

Implementation specification for the service that receives results from the ML
server. Written for a backend engineer who must maintain, port or reimplement
it.

**Source of truth.** Behaviour described here was read from
`cart_platform/receiver/{app.py,store.py,database.py,schema.sql}` and verified
by executing the real code. Where a behaviour is a **defect that must be fixed
on reimplementation**, it is labelled ⚠ and the evidence is shown.

---

## 1. What this service is

| | |
|---|---|
| **Role** | A durable, append-only sink for ML results |
| **Direction** | **Inbound only.** It never calls the ML server |
| **Ownership** | Owns its own database. The ML server never opens it directly |
| **Guarantee** | Idempotent ingestion; history is never overwritten |

Verified: `receiver/` imports no HTTP client (`requests`, `httpx`, `urllib`
are absent). There is no outbound path, no callback, no polling. If you need
8200 to trigger work on 8100, that is a **new component** — it does not exist.

```
ML server :8100 ───POST /v1/results/{category}───▶ Receiver :8200 ───▶ database
                ◀──── 201 / 200 / 409 / 422 / 401 ────
```

The ML server holds a durable outbox and retries. Your job is therefore narrow
and strict: **decide the outcome correctly, persist exactly once, answer with
the right status code.** The ML server's retry correctness depends entirely on
your status code being accurate.

---

## 2. Request lifecycle

```
POST /v1/results/training | calibration | backtest | scoring
   │
   ├─ S1  Authenticate ..................... 401 if token configured and wrong
   ├─ S2  Validate envelope schema ......... 422 on malformed body
   ├─ S3  Match event_type to the route .... 422 on mismatch
   ├─ S4  Category preconditions ........... 422 (backtest: backtest_type required)
   ├─ S5  Canonicalise payload → SHA-256
   ├─ S6  Idempotency decision
   │        ├── not seen before ........... CREATED
   │        ├── seen, same hash ........... REPLAY
   │        └── seen, different hash ...... CONFLICT → 409, stop
   ├─ S7  Persist (ONE transaction)
   │        ├── INSERT received_events     (audit log, full payload)
   │        └── INSERT <category>_results  (projection, typed columns)
   └─ S8  Respond 201 (created) or 200 (replay)
```

---

## 3. Stage-by-stage logic

### S1 — Authentication

```
function authenticate(request):
    if RECEIVER_TOKEN is not configured:
        return OK                     # auth disabled (dev only)
    if request.header["Authorization"] != "Bearer " + RECEIVER_TOKEN:
        raise Http(401, "invalid or missing bearer token")
    return OK
```

Applied to **write routes only**. Read routes are currently unauthenticated —
if your deployment exposes them beyond a trusted network, add the same guard.

### S2 — Envelope validation

```
schema ResultEventIn:
    event_id       : string, required, 1..512 chars
    event_type     : string, required
    payload        : object, required
    schema_version : int,    default 1
    occurred_at    : string | null
    entity_id      : string | null
    site_key       : string | null
    upload_id      : string | null
    job_id         : string | null
    model_id       : string | null
```

Anything else in the body is ignored. A missing required field → `422`.

### S3 — Route / type match

Each route accepts exactly one `event_type`. This prevents a
misrouted event from silently landing in the wrong table.

```
EXPECTED = {
    "/v1/results/training"    : "training.completed",
    "/v1/results/calibration" : "calibration.completed",
    "/v1/results/backtest"    : "backtest.completed",
    "/v1/results/scoring"     : "scoring.recorded",
}

if event.event_type != EXPECTED[route]:
    raise Http(422, "event_type '<got>' does not match this route (expected '<want>')")
```

### S4 — Category preconditions

```
if route == "/v1/results/backtest":
    if not event.payload.backtest_type:
        raise Http(422, "payload.backtest_type is required")
```

`backtest_type` is `NOT NULL` in the schema, so this check turns a would-be
database error into a clear, non-retryable client error.

### S5 — Canonical hash

This is the heart of idempotency. The hash must be **stable across
serialisations** — key order and whitespace must not change it.

```
function canonical_hash(payload):
    canonical = json_encode(payload,
                            sort_keys   = true,
                            separators  = (",", ":"),   # no whitespace
                            default     = to_string)    # non-JSON types → str
    return sha256_hex(canonical)
```

Three rules you must not change, or every stored hash becomes invalid:

1. Keys sorted recursively.
2. No whitespace between tokens.
3. Only the `payload` is hashed — **never the envelope**. `occurred_at` moves
   on every retry attempt; including it would turn every retry into a false
   `409`.

### S6 — Idempotency decision

```
function decide(event_id, new_hash):
    existing = SELECT * FROM received_events WHERE event_id = event_id
    if existing is null:            return CREATED
    if existing.payload_hash == new_hash: return REPLAY
    return CONFLICT
```

| Outcome | HTTP | Meaning | Effect on database |
|---|---|---|---|
| `CREATED` | **201** | First time this `event_id` is seen | Inserts audit + projection row |
| `REPLAY` | **200** | Identical resend (network retry) | **Nothing written** |
| `CONFLICT` | **409** | Same id, different content | **Nothing written**, original preserved |

The `409` body returns the stored `payload_hash` so the caller can diagnose:

```json
{
  "message": "event_id already stored with a different payload; historical results are never overwritten",
  "event_id": "...",
  "stored_payload_hash": "9f2c..."
}
```

> **Why conflict is not an update.** A backtest report, once published, is a
> historical fact. Allowing a rewrite would make past results irreproducible. If
> the ML server legitimately needs to publish a revised result, it must emit a
> **new `event_id`** (the id contains `status`, so a changed status yields a new
> id automatically).

### S7 — Persist

Two inserts, in this order (the projection has a foreign key to the audit row):

```
BEGIN TRANSACTION
    INSERT INTO received_events (event_id, event_type, schema_version,
        entity_id, site_key, upload_id, job_id, model_id,
        occurred_at, received_at, payload_hash, payload_json)
    VALUES (..., now_utc(), hash, json_encode(payload))

    store_projection(event)        # see §4
COMMIT
```

`received_at` is **server-generated**, never taken from the request.
`payload_json` stores the payload **verbatim** — nothing is dropped, so any
field without a dedicated column remains recoverable.

### S8 — Response

```
201 Created  (or 200 OK on replay)
{
  "event_id": "...",
  "outcome": "created" | "replay",
  "stored_record": { event_id, event_type, entity_id, job_id, model_id, received_at }
}
```

---

## 4. What gets inserted, per event type

Two layers on every successful write:

| Layer | Table | Content |
|---|---|---|
| Audit | `received_events` | Envelope + **entire payload verbatim** |
| Projection | `<category>_results` | Selected fields as typed, indexed columns |

Shared resolution rules used by the projection step:

```
config_hash = payload.config_hash

frozen      = payload.frozen_model_config
           ?? payload.frozen_thresholds

effective   = payload.effective_operational_config
           ?? payload.effective_config_at_training
           ?? payload.effective_config
```

The fallback chains exist because the four publishers use different field names
for the same concept. Preserve the **order** — for scoring events both
`frozen_model_config` and `frozen_thresholds` are present, and the richer
`frozen_model_config` must win.

Envelope fields (`job_id`, `entity_id`, `site_key`, `upload_id`, `model_id`,
`occurred_at`) always come from the **envelope**, never from inside `payload`.

### 4.1 `training.completed` → `training_results`

```
INSERT INTO training_results
  (event_id, job_id, entity_id, site_key, upload_id, model_id, status,
   occurred_at, received_at, metrics_json, metadata_json,
   config_hash, frozen_config_json, effective_config_json)
VALUES
  (envelope.event_id, envelope.job_id, envelope.entity_id, envelope.site_key,
   envelope.upload_id, envelope.model_id, payload.status,
   envelope.occurred_at, now_utc(),
   json(payload.metrics),
   json(payload),                 -- full copy
   config_hash, json(frozen), json(effective))
```

Dedicated columns: 5. Full copy in `metadata_json`. JSON-only fields: `error`,
`detail`, `n_features`, `selected_models`, `split_windows`, `config_snapshot`,
`promotable`, `blocking_reasons`, `promoted`, `warnings`, `artifacts`, `caveat`.

### 4.2 `calibration.completed` → `calibration_results`

```
INSERT INTO calibration_results
  (event_id, job_id, entity_id, site_key, upload_id, model_id,
   calibrator_id, status, scope, method, occurred_at, received_at,
   metrics_json, metadata_json, config_hash)
VALUES
  (envelope..., payload.calibrator_id, payload.status, payload.scope,
   payload.method, envelope.occurred_at, now_utc(),
   json(payload.metrics), json(payload), config_hash)
```

Dedicated columns: 6. **No frozen/effective config columns** — only
`config_hash`. To recover the configuration, join to `training_results` on
`config_hash`.

### 4.3 `backtest.completed` → `backtest_results`

```
INSERT INTO backtest_results
  (event_id, job_id, entity_id, site_key, upload_id, model_id, report_id,
   backtest_type, status, reason, occurred_at, received_at,
   metrics_json, assumptions_json, caveat, artifacts_json,
   config_hash, frozen_config_json, effective_config_json)
VALUES
  (envelope..., payload.report_id, payload.backtest_type, payload.status,
   payload.reason, envelope.occurred_at, now_utc(),
   json(payload.metrics), json(payload.assumptions), payload.caveat,
   json(payload.artifacts), config_hash, json(frozen), json(effective))
```

Dedicated columns: 11 — the richest projection. **No full-payload copy**,
because every field is either a column or inside `metrics_json` /
`assumptions_json`. The original always remains in
`received_events.payload_json`.

### 4.4 `scoring.recorded` → `scoring_results`

```
action = payload.action ?? {}

INSERT INTO scoring_results
  (event_id, entity_id, site_key, model_id, decision_time, occurred_at,
   received_at, shortage_risk, alert_state, should_replenish, quantity,
   result_json, warnings_json, config_hash, frozen_config_json)
VALUES
  (envelope.event_id, envelope.entity_id, envelope.site_key, envelope.model_id,
   payload.decision_time, envelope.occurred_at, now_utc(),
   payload.shortage_risk, payload.alert_state,
   to_int(to_bool(action.should_replenish)),     -- FLATTENED from nested object
   action.quantity,                              -- FLATTENED
   json(payload), json(payload.warnings ?? []),
   config_hash, json(frozen))
```

Dedicated columns: 8. Full copy in `result_json`.

Two traps:

- **`should_replenish` and `quantity` are flattened out of `payload.action`**,
  not top-level fields. The rest of `action` (`reason`, `urgency`, `order_id`,
  `minutes_to_predicted_breach`, `expected_completion_time`,
  `actionable_before_breach`) stays JSON-only.
- **This table has no `job_id` or `upload_id` column.** A score is not produced
  by a training job; the ML server sends both as `null`.

### 4.5 Summary

| | training | calibration | backtest | scoring |
|---|---|---|---|---|
| Envelope → columns | 6 | 6 | 6 | 4 |
| Payload → own column | 5 | 6 | 11 | 8 |
| Full-payload copy | `metadata_json` | `metadata_json` | — | `result_json` |
| Config columns | hash, frozen, effective | hash | hash, frozen, effective | hash, frozen |

---

## 5. ⚠ Defect to fix on reimplementation: non-atomic persist

### The problem

The current implementation commits **per statement**:

```python
def execute(sql, params):
    conn = connect()
    try:
        cur = conn.execute(sql, params)
        conn.commit()          # ← commits immediately
        return cur
    except Exception:
        conn.rollback(); raise
```

So S7's two inserts are **two separate transactions**. If the projection insert
fails after the audit insert committed, the event is permanently half-stored —
and because the audit row now exists with a matching hash, every retry is
classified `REPLAY` and returns `200`, so the projection row is **never
backfilled**.

### Measured evidence

Forcing the projection step to throw:

```
1st POST raised : projection failed
   received_events  : 1        ← audit row committed
   backtest_results : 0        ← projection row missing

2nd POST (retry) : replay      ← ML server sees 200 and stops retrying
   received_events  : 1
   backtest_results : 0        ← still missing, permanently
```

The ML server believes delivery succeeded. The result is invisible to every
`GET /v1/results/backtest` query, while sitting intact in `received_events`.

### Required fix

Wrap both inserts in one transaction:

```
function persist(event, base, hash):
    BEGIN
    try:
        insert_audit_row(base, hash, event.payload)
        store_projection(event, base)
        COMMIT
    except:
        ROLLBACK
        raise          # → 5xx, ML server retries the whole event
```

On SQL Server, combine this with the unique-violation pattern so the
`CREATED` / `REPLAY` / `CONFLICT` decision stays race-free under concurrency:

```sql
BEGIN TRANSACTION;
BEGIN TRY
    INSERT INTO cart.received_events (...) VALUES (...);
    INSERT INTO cart.backtest_results (...) VALUES (...);
    COMMIT TRANSACTION;                       -- → 201
END TRY
BEGIN CATCH
    ROLLBACK TRANSACTION;
    IF ERROR_NUMBER() IN (2627, 2601)         -- PK / unique violation
        -- re-read the stored hash, then return REPLAY or CONFLICT
    ELSE
        THROW;                                -- → 5xx, retryable
END CATCH
```

A detached recovery job can repair any rows already damaged:

```sql
SELECT e.event_id, e.event_type
FROM cart.received_events e
LEFT JOIN cart.backtest_results b ON b.event_id = e.event_id
WHERE e.event_type = 'backtest.completed' AND b.event_id IS NULL;
-- rebuild the projection from e.payload_json
```

This is the strongest argument for keeping the full payload in
`received_events`: the projection is always reconstructible.

---

## 6. Concurrency

SQLite is single-writer, so the current `SELECT`-then-`INSERT` cannot race. **Any
multi-writer database can.** Two concurrent POSTs with the same `event_id` can
both read "not found" and both attempt the insert.

Do not solve this with a lock or a higher isolation level. Let the unique
constraint arbitrate, then re-read to classify:

```
try:
    INSERT ...                       → CREATED
catch unique_violation:
    stored = SELECT payload_hash WHERE event_id = ?
    return (REPLAY if stored == new_hash else CONFLICT)
```

This is correct under any isolation level and needs no coordination.

---

## 7. Error handling matrix

Your status code directly controls the ML server's retry behaviour.

| Condition | Status | ML server reaction |
|---|---|---|
| Stored successfully | `201` | Marks `sent` |
| Identical replay | `200` | Marks `sent` |
| Same id, different payload | `409` | **Dead-letters**, no retry |
| `event_type` / route mismatch | `422` | Dead-letters |
| `backtest_type` missing | `422` | Dead-letters |
| Bad or missing token | `401` | Dead-letters |
| Database down, bug, timeout | `5xx` | **Retries** with backoff |
| Overloaded | `429` + `Retry-After` | Retries after the stated delay |

The critical rule: **never return 2xx when you did not persist, and never
return 4xx for a transient failure.** A wrong 2xx loses data silently; a wrong
4xx dead-letters a recoverable event.

If you add load shedding, return `429` with `Retry-After` — it is already
honoured by the publisher.

---

## 8. Read-side logic

All four list endpoints share one implementation:

```
function list(table, filters, limit, offset, order_col = "occurred_at"):
    where  = ["1=1"]; params = []
    for (col, value) in filters where value is not null:
        if col == "since":  where += [order_col + " >= ?"]
        elif col == "until": where += [order_col + " <= ?"]
        else:                where += [col + " = ?"]
        params += [value]

    total = SELECT COUNT(*) FROM <table> WHERE <where>
    rows  = SELECT * FROM <table> WHERE <where>
            ORDER BY <order_col> DESC, id DESC
            LIMIT ? OFFSET ?                      -- SQLite
    return { total, limit, offset, items: rows }
```

Notes for a port:

- `scoring_results` orders by `decision_time`, the others by `occurred_at`.
- Filter column names are **whitelisted by the endpoint signature**, never taken
  from arbitrary query keys — this is what keeps the dynamic `WHERE` safe. If
  you accept caller-supplied column names, you reintroduce SQL injection.
- `limit` 1..500 (default 50), `offset` ≥ 0 — enforce it, or a caller can ask
  for the whole table.
- T-SQL: replace `LIMIT ? OFFSET ?` with
  `OFFSET ? ROWS FETCH NEXT ? ROWS ONLY` and **swap the argument order**.

Two aggregate endpoints:

- `GET /v1/results/entities/{entity_id}` — counts per category, all backtests,
  and the latest training / calibration / scoring row for that entity.
- `GET /health` — liveness plus `events_stored` (a `COUNT(*)` on
  `received_events`). Keep it cheap; it is polled.

---

## 9. Invariants to preserve

A reimplementation is correct if all of these hold:

1. The service never initiates a call to the ML server.
2. `event_id` is the sole idempotency key and is treated as **opaque** — never
   parsed or split (it contains `:` from `KDVR:ChanNo`).
3. The payload hash covers the payload only, never the envelope.
4. A stored event is never mutated. Conflicts are rejected, not merged.
5. `received_at` is server time; `occurred_at` is the caller's.
6. The full payload is always retained verbatim.
7. Audit row and projection row are written atomically (§5).
8. The status code accurately reflects whether data was persisted.
9. Binary artifacts are never accepted or stored — only metadata and URI
   references.
10. All timestamps are UTC.

---

## 10. Test plan

| # | Scenario | Expected |
|---|---|---|
| 1 | POST a new event | `201`, 1 audit row + 1 projection row |
| 2 | POST the same event again | `200`, still exactly 1 row each |
| 3 | POST same `event_id`, changed payload | `409`, original untouched |
| 4 | POST with keys reordered in the payload | `200` replay (canonical hash stable) |
| 5 | POST to the wrong route | `422`, nothing stored |
| 6 | POST backtest without `backtest_type` | `422`, nothing stored |
| 7 | POST without the bearer token | `401`, nothing stored |
| 8 | Projection insert fails | **nothing** committed; retry succeeds (§5) |
| 9 | Two concurrent POSTs, same id | One `201`, one `200`; exactly one row |
| 10 | Database unavailable | `5xx`; ML server retries and eventually succeeds |
| 11 | Paging at `offset=0` and `offset=50` | No duplicates, no gaps |
| 12 | `since` / `until` filters | Correct window |
| 13 | 9 KB payload | Round-trips without truncation |
| 14 | Event with no config fields | Accepted; config columns `NULL` |

Tests 1–7 and 11–14 exist today in
`cart_platform/tests/test_result_receiver.py` (23 tests). **Tests 8, 9 and 10
do not exist** and should be added with the §5 fix.

---

## 11. Reference pseudocode

```
function handle_post(route, request):
    # S1
    authenticate(request)

    # S2
    event = validate_schema(request.body)            # 422 on failure

    # S3
    expected = EXPECTED_TYPE[route]
    if event.event_type != expected:
        return 422, "event_type does not match this route"

    # S4
    if route == BACKTEST and not event.payload.backtest_type:
        return 422, "payload.backtest_type is required"

    # S5
    hash = canonical_hash(event.payload)

    # S6 + S7 (atomic)
    BEGIN
    try:
        existing = SELECT payload_hash FROM received_events
                   WHERE event_id = event.event_id
        if existing is not null:
            ROLLBACK
            if existing.payload_hash == hash:
                return 200, {outcome: "replay"}
            return 409, {message: "...", stored_payload_hash: existing.payload_hash}

        INSERT INTO received_events (envelope..., received_at = now_utc(),
                                     payload_hash = hash,
                                     payload_json = encode(event.payload))
        store_projection(event)
        COMMIT
    catch unique_violation:
        ROLLBACK
        stored = SELECT payload_hash FROM received_events WHERE event_id = ...
        return (200 if stored == hash else 409)
    catch any_other_error:
        ROLLBACK
        return 5xx                      # retryable; ML server will resend

    # S8
    return 201, {event_id, outcome: "created", stored_record: {...}}


function store_projection(event):
    p        = event.payload
    config   = p.config_hash
    frozen   = p.frozen_model_config ?? p.frozen_thresholds
    effective= p.effective_operational_config
            ?? p.effective_config_at_training
            ?? p.effective_config

    switch event.event_type:
        case "training.completed":
            insert training_results(envelope, p.status, json(p.metrics),
                                    json(p), config, json(frozen), json(effective))
        case "calibration.completed":
            insert calibration_results(envelope, p.calibrator_id, p.status,
                                       p.scope, p.method, json(p.metrics),
                                       json(p), config)
        case "backtest.completed":
            insert backtest_results(envelope, p.report_id, p.backtest_type,
                                    p.status, p.reason, json(p.metrics),
                                    json(p.assumptions), p.caveat,
                                    json(p.artifacts), config,
                                    json(frozen), json(effective))
        case "scoring.recorded":
            a = p.action ?? {}
            insert scoring_results(envelope, p.decision_time, p.shortage_risk,
                                   p.alert_state, to_bool(a.should_replenish),
                                   a.quantity, json(p), json(p.warnings ?? []),
                                   config, json(frozen))
```

---

## 12. Configuration

| Variable | Purpose | Default |
|---|---|---|
| `CART_RECEIVER_HOST` | Bind address | `127.0.0.1` |
| `CART_RECEIVER_PORT` | Port | `8200` |
| `CART_RECEIVER_DB` | Database location | `data/results_receiver.db` |
| `CART_RESULT_RECEIVER_TOKEN` | Bearer token; unset disables auth | unset |

Bind to `127.0.0.1` unless the ML server is on another host. Always set the
token in any shared environment — unset means **no authentication at all**.

---

## 13. Known gaps

1. **Non-atomic persist** (§5) — measured, fix required before production.
2. **Read endpoints are unauthenticated** — results are readable by anyone who
   can reach the port.
3. **No retention or archiving.** `received_events` grows without bound; at
   ~4.2 KB mean payload and ~215 scoring decisions per entity per day, 1,000
   entities is roughly 900 MB/day.
4. **No rate limiting.** The publisher honours `429` + `Retry-After`, but the
   receiver never issues one.
5. **No schema-version negotiation.** `schema_version` is stored but not
   branched on. A future `v2` payload would be accepted and projected using v1
   rules.
