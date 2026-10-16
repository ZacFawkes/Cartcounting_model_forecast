# Implementation Specification — Dynamic Multi-Site / Multi-Bay Cart Forecasting API

## Role and objective

Act as a senior Python backend engineer, MLOps engineer, and data scientist. **Extend the existing Cart Forecasting & Replenishment API; do not rebuild it or change validated forecasting, feature engineering, risk calibration, or replenishment mathematics without a documented reason.** Implement a production-oriented module that automatically discovers cart-counting streams from a database, assembles the exact historical context required by the existing model, invokes the same scoring engine, and stores **a separate CSV for every scoring invocation**.

**Critical design constraint:** A site may contain **any number of bays**, and bays can be configured, renamed, added, or removed at any time. **Never assume a fixed number of bays per site, a fixed `ChanNo` range, or that one site corresponds to one bay.** Identify each stream using the composite key **`(KDVR, ChanNo)`**, exactly as provided by source telemetry. `siteKey` is optional descriptive metadata, not the primary stream identity. Never aggregate bays unless an explicit separately tested site-level aggregation feature is requested.

## 1. Inspect existing implementation before writing code

Review `api/server/app.py`, `engine.py`, `schemas.py`, `request_log.py`, `settings.py`, `run_server.sh`, `api/tests/`, `cart_forecasting/` and the actual database/query integration if present. Find and document:

1. The **real feature-generation function** and all of its lookback dependencies: lags, rolling windows, differencing, warm-up, interval handling, required columns, and target alignment.
2. The minimum number of **completed consecutive records** needed to produce one valid score. Do **not** assume 12 records are sufficient merely because the existing README says “at least 12.” Derive `required_history_records` from the actual feature pipeline, include any additional warm-up/lag rows, and verify with a test comparing online features to offline features.
3. The actual model artifact location, artifact type, and serialized structure. The current README references `cart_forecasting/outputs/models/cart_forecasting_models.pkl`; **verify that file and inspect its metadata and object keys rather than assuming the filename, number of models, or whether models are global/per-site/per-bay.** Do not load untrusted pickle files. Prefer an explicit versioned model registry/manifest.
4. Whether the existing models were trained on one site/bay or multiple streams, and whether a new `(KDVR, ChanNo)` pair is eligible for scoring. **Discovering a bay does not mean a valid model exists for it.** Do not silently use a site-163-trained model on arbitrary bays. If a globally shared model is supported, require an explicit applicability policy and mark cross-bay generalization as unvalidated until tested.
5. The timestamp at which each source record becomes fully available, particularly `DVRDate`, `next_DVRDate`, `Count`, `Count_IN`, and `Count_OUT`. Treat interval traffic as unavailable until `next_DVRDate` (or verified source commit/availability time).
6. The existing `plan_replenishment()` behavior and state dependencies, so the new module uses exactly the same logic as `/score` and `/backtest`.

Create `docs/multi_bay_discovery.md` recording the inspected file paths, artifact names, feature requirements, and any assumptions or unresolved questions. If source files are missing, implement configurable adapters and clearly report what could not be verified; do not invent internal function names or database schema.

## 2. Dynamic stream discovery — `(KDVR, ChanNo)`

Create a `StreamRegistry` that discovers distinct `(KDVR, ChanNo)` pairs from the source database. Use a safe parameterized query appropriate to the existing DB dialect. Conceptual example (adapt table/column names after inspection):

```sql
SELECT DISTINCT KDVR, ChanNo
FROM <verified_cart_count_table>
WHERE KDVR IS NOT NULL AND ChanNo IS NOT NULL;
```

Do **not** execute this full historical distinct scan for every incoming record. Instead:

- On startup, load known streams from a persisted registry or an indexed discovery query.
- On each incremental ingestion batch, upsert newly observed `(KDVR, ChanNo)` pairs automatically.
- Optionally reconcile with the database on a configurable schedule to discover streams not yet seen by the worker.
- Expose stream lifecycle states: `DISCOVERED`, `WARMING_UP`, `READY`, `MODEL_UNAVAILABLE`, `STALE`, `ERROR`, `DISABLED`.
- Preserve stream identity even if the bay label changes. Support optional metadata: `siteKey`, display name, location, capacity, timezone, active status.
- Never register thousands of hardcoded FastAPI routes. **Use one parameterized route** and dynamically resolve the stream from the registry. The URL path is a stable API contract, not an individual Python function generated per bay.

Required endpoints:

| Method | Endpoint | Behavior |
|---|---|---|
| GET | `/streams` | List discovered streams and statuses; filter by KDVR, siteKey, status |
| GET | `/streams/{KDVR}/{ChanNo}` | Stream metadata, history readiness, model eligibility, latest record |
| GET | `/streams/{KDVR}/{ChanNo}/model` | Resolved artifact path, version, scope, compatibility, features |
| POST | `/streams/{KDVR}/{ChanNo}/score` | Score this bay using validated history or latest DB records |
| GET | `/streams/{KDVR}/{ChanNo}/latest` | Latest forecast/action, run ID, output CSV path |
| GET | `/streams/{KDVR}/{ChanNo}/results` | Paginated scoring runs with links to per-call CSVs |
| GET | `/streams/{KDVR}/{ChanNo}/results/{run_id}` | Full decision and metadata for one invocation |
| GET | `/streams/{KDVR}/{ChanNo}/results/{run_id}/csv` | Download the exact CSV produced by that invocation |
| GET | `/streams/{KDVR}/{ChanNo}/health` | Freshness, worker status, model readiness, errors |

Maintain existing `/health`, `/model/info`, `/score`, `/backtest`, and logging endpoints for backward compatibility. Validate KDVR/ChanNo as source-appropriate path identifiers; do not trust arbitrary path strings when building filesystem paths.

## 3. Model registry and artifact loading

Introduce a versioned model manifest, e.g. `config/model_registry.yaml` or JSON. Populate it by **inspecting the existing `.pkl` artifact and actual training scope**. Example schema (illustrative only):

```yaml
model_registry_version: 1
models:
  - model_id: cart_forecast_site163_v1
    artifact_path: cart_forecasting/outputs/models/cart_forecasting_models.pkl
    scope: explicit_streams
    eligible_streams:
      - {KDVR: 215, ChanNo: 1} # REPLACE with verified mapping; do not assume this pair is correct
    feature_schema_version: existing_pipeline_verified_version
    model_version: verified_from_artifact_or_config
    enabled: true
```

Resolve artifacts relative to a configured project root, not the current working directory. Load each trusted artifact once per process, cache by immutable version, and fail clearly if missing, corrupt, incompatible, or unapproved for a stream. Never silently train a new model at API request time. If there is a valid global model, define its scope explicitly and keep a trace of which streams it has actually been evaluated on. Include `model_id`, `model_version`, `artifact_path`, `feature_schema_version`, and `eligibility_reason` in each score result.

## 4. Database ingestion and automatic scoring

Build a standalone `DatabaseIngestionWorker` (not a background loop hidden in each FastAPI web worker) that watches for **newly committed, completed** telemetry records. Choose the best available source mechanism after inspecting infrastructure:

1. CDC / change stream / message queue if available;
2. otherwise indexed incremental polling with a durable cursor.

Use source ordering by a stable monotonically increasing key (e.g. `CountID`) plus interval timestamp where applicable. If the source has no reliable global monotonic key, implement a composite watermark and an overlap window with deduplication. Acknowledge the watermark **only after the score and its outputs are durably persisted**. Handle late arrivals, corrections, duplicate messages, out-of-order records, worker restarts, and database reconnections. Do not score partial records whose traffic window has not closed.

For every new completed record:

1. Read `(KDVR, ChanNo)` from the record and upsert the stream registry.
2. Determine the stream's required historical lookback **from the verified feature pipeline**, not a guessed constant.
3. Query the latest `X_lag` **completed records for that same exact `(KDVR, ChanNo)`**, ending with the triggering record. Query enough additional rows for feature warm-up if needed. Never mix another bay's history into a stream.
4. Order history chronologically and validate timestamp continuity, uniqueness, nonnegative counts where applicable, required fields, and missing/invalid values.
5. If insufficient history, record `WARMING_UP` with `available_history` and `required_history`, and do not generate a fabricated forecast.
6. If model unavailable/ineligible, log `MODEL_UNAVAILABLE` and do not score.
7. Reuse the existing feature engineering, inference, calibration, alerting, and `plan_replenishment()` code without duplicating their implementations.
8. Maintain an **independent** persistent online session for each `(KDVR, ChanNo)`: pending orders, cooldown, alert state, last processed source record, and idempotency keys. Do not let one bay's pending orders affect another bay.
9. Persist the scoring result and one CSV for this invocation, then advance the durable cursor/acknowledgment.
10. Make the latest result available through the bay-specific endpoints.

Support configurable `DB_POLL_INTERVAL_SECONDS` (e.g. 5–15 seconds as a starting point; actual setting based on DB load). Ensure exactly-once **effective decisions** via idempotency and durable state even if message delivery is at-least-once. Multiple worker processes must not double-score the same source record: use leases/locks or unique transactional claims.

**Important:** The user should not have to manually send `X_lag` records on every database update. The worker fetches history automatically. Keep manual HTTP scoring supported for testing and integrations.

## 5. Decision time, source time, and replay parity

Store separately:

- `record_start_time` = `DVRDate`
- `record_end_time` = `next_DVRDate`
- `record_available_time` = verified database commit/availability timestamp, if available
- `request_received_at` = API arrival time (manual requests)
- `scored_at` = model execution time
- `decision_effective_at` = when the action could actually be issued
- `expected_completion_time` = decision effective time + lead time

Never use future interval traffic at an earlier decision time. Historical replay must use **historical decision-effective timestamps**, not today's wall-clock time. Live API may use actual ingestion/processing time and should report ingestion lag. Replaying a historical record today must not produce a fictitious delivery date today unless explicitly requested as a live simulation.

Add a parity test: for the same eligible stream and same history prefix, the database-triggered path, manual `/streams/{KDVR}/{ChanNo}/score`, and existing `/score` must produce the same features, forecasts, risk, and policy recommendation when supplied with the same session state and effective timestamp. Allow differences only in request IDs and measured wall-clock latency.

## 6. Per-invocation CSV output — mandatory

**Create one distinct CSV file for every successful scoring invocation, including NO_ACTION decisions.** Also write a structured JSON sidecar containing the full nested response and metadata. Do not overwrite prior runs. Use an immutable `run_id` and atomic file creation.

Proposed storage layout:

```text
api/logs/streams/
  KDVR=<safe_kdvr>/
    ChanNo=<safe_channo>/
      YYYY-MM-DD/
        <timestamp>_<run_id>_score.csv
        <timestamp>_<run_id>_score.json
      latest.json
```

Each per-call CSV must have **one row for the decision**, at minimum:

```text
run_id,request_id,KDVR,ChanNo,siteKey,source_record_id,
record_start_time,record_end_time,record_available_time,
scored_at,decision_effective_at,ingestion_lag_seconds,
model_id,model_version,feature_schema_version,
history_count,required_history_records,
current_cart,forecast_cart_20min,forecast_cart_40min,
forecast_min_20min,forecast_min_40min,
forecast_lower_bound_min_40min,shortage_risk,
alert_state,risk_threshold,should_replenish,
recommended_quantity,urgency,order_id,
expected_completion_time,pending_carts,
policy_version,decision_reason,status,error_code
```

Use the **actual outputs available in the existing engine**. If a field is not produced, use null and document it; do not fabricate a forecast. Preserve exact full-precision numeric values in JSON, and use stable, documented rounding in CSV. Include `status=NO_ACTION` when the decision is valid but no replenishment is needed.

For skipped/failed invocations (insufficient history, missing model, malformed record), create a separate structured **event/error CSV or JSON** with the reason; do not pretend it was a successful scoring decision. Maintain the existing append-only `score_results.csv` as a summary index in addition to per-invocation CSVs. Ensure CSV quoting and formula-injection-safe exports if spreadsheet users can open them.

Also maintain a stream-specific cumulative index:

```text
api/logs/streams/KDVR=<...>/ChanNo=<...>/scores_index.csv
```

This is **in addition to**, not instead of, individual CSV files. Index writes must be concurrency-safe and recoverable.

## 7. Example behavior

Assume the database contains these **illustrative** stream identities:

```text
(KDVR=215, ChanNo=1)
(KDVR=215, ChanNo=2)
(KDVR=301, ChanNo=7)
```

They are discovered automatically. The server exposes the same parameterized routes for all of them:

```text
GET  /streams/215/1
POST /streams/215/1/score
GET  /streams/215/1/latest

GET  /streams/215/2
POST /streams/215/2/score
GET  /streams/215/2/latest

GET  /streams/301/7
POST /streams/301/7/score
GET  /streams/301/7/latest
```

**These IDs are examples, not verified source values.** A new `(KDVR, ChanNo)` automatically becomes discoverable when new data appears, but its status is `WARMING_UP` until enough history exists and `MODEL_UNAVAILABLE` unless an eligible model is registered.

Example scoring event:

```text
Database commits completed record CountID=987654
    -> identify (KDVR=215, ChanNo=2)
    -> fetch verified X_lag history for ONLY (215, 2)
    -> validate completion/feature readiness
    -> resolve eligible model from manifest
    -> compute features with original code
    -> score forecast + risk
    -> apply isolated bay session/order book
    -> persist one immutable per-call CSV and JSON
    -> publish latest result and log request/worker metrics
```

Example response (values are **illustrative**, not model outputs):

```json
{
  "run_id": "example-uuid",
  "stream": {"KDVR": 215, "ChanNo": 2, "siteKey": 163},
  "source_record_id": 987654,
  "status": "SCORED",
  "history_count": 20,
  "required_history_records": 20,
  "model": {"model_id": "example-model", "version": "example-version"},
  "forecast": {"min_within_40min": 12.6},
  "shortage_risk": 0.76,
  "action": {"should_replenish": true, "quantity": 16},
  "outputs": {
    "csv": "/streams/215/2/results/example-uuid/csv",
    "json": "/streams/215/2/results/example-uuid"
  }
}
```

## 8. Implementation layout

Extend the existing repository with clear modules (adjust names to match current conventions):

```text
api/
  server/
    app.py                        # existing routes + stream-specific routes
    engine.py                     # existing shared scoring engine
    schemas.py                    # stream and result schemas
    settings.py                   # DB/model/worker/output config
    stream_registry.py            # dynamic stream discovery + lifecycle
    model_registry.py             # verified artifact resolution/eligibility
    history_provider.py           # fetch exact per-stream X_lag records
    db_ingestion.py              # CDC/polling adapter + durable cursor
    stream_worker.py              # independent ingestion/scoring process
    session_store.py              # durable isolated per-stream online state
    result_store.py               # atomic per-invocation CSV/JSON + index
    request_log.py               # existing logs
  config/
    model_registry.yaml
  tests/
    test_stream_registry.py
    test_history_provider.py
    test_model_registry.py
    test_stream_worker.py
    test_stream_api.py
    test_replay_parity.py
  docs/
    multi_bay_discovery.md
    multi_bay_api.md
    operations.md
  logs/streams/                  # generated at runtime
```

Do not put database passwords in source control. Configure DB URL/credentials through environment variables or secret manager. Document required indexes, deployment topology, restart recovery, and how to run the API and worker independently.

## 9. Tests and acceptance criteria

Write automated tests for:

1. Zero, one, two, and arbitrarily many bays per site; nonconsecutive `ChanNo` values.
2. Same `ChanNo` under different `KDVR` remains isolated.
3. Newly discovered streams become accessible without restarting the API.
4. Unknown stream returns a clear 404 or discovery status; known stream without model returns `MODEL_UNAVAILABLE`, not an invented score.
5. Correct lookback length derived from real feature code, including warm-up; exact online/offline feature parity.
6. Only completed intervals are scored; no future traffic leakage.
7. No history mixing across bays and correct chronological ordering.
8. One newly committed record triggers exactly one effective score and exactly one per-call CSV; retries never duplicate replenishment orders.
9. A valid `NO_ACTION` score still produces a CSV.
10. Independent pending orders, cooldowns, alert states, and session resets for each bay.
11. Multiple worker processes cannot double-score the same source record.
12. Durable restart/recovery, cursor rollback/replay, late arrivals, duplicates, and corrected records.
13. Missing/corrupt/unapproved model artifact produces an explicit error.
14. CSV/JSON consistency, immutable filenames, atomic writes, safe file paths, and correct download endpoints.
15. Replay uses historical effective time, while live scoring records actual ingestion delay.
16. Existing 18 API tests still pass; add new tests without removing old ones.
17. Multi-bay stress test with realistic ingestion volume and performance metrics.

**Acceptance demo:** Use a fixture database containing at least 3 sites with unequal bay counts (e.g. 1, 2, and 4 bays), plus one newly appearing bay. Insert completed records one by one; show discovery, warm-up, model resolution, automatic X_lag fetch, automatic scoring, isolated per-bay sessions, a separate CSV for each successful call, and retrieval via `/streams/{KDVR}/{ChanNo}/results/{run_id}/csv`. Demonstrate restart and replay without duplicate decisions. For fixture streams without an approved model, demonstrate correct `MODEL_UNAVAILABLE` behavior rather than forcing a prediction.

## 10. Execution order and deliverables

1. Inspect actual code, schema, artifact files, and training scope; write `multi_bay_discovery.md`.
2. Implement registry + model eligibility and tests.
3. Implement DB history provider and derive actual X_lag from original feature engineering.
4. Implement isolated persistent session state and durable ingestion worker.
5. Add parameterized stream endpoints while preserving existing APIs.
6. Implement immutable per-invocation CSV/JSON and searchable indexes.
7. Validate time semantics and online/backtest feature parity.
8. Run all old and new tests; demonstrate multi-bay ingestion end-to-end.
9. Update README and deployment instructions with real commands, configuration variables, artifact paths, sample CSV output, and observed test results.

**Final response must state:** actual model artifact filename/path and contents discovered; whether it is global or stream-specific; verified X_lag history requirement; exact DB table and cursor strategy; discovered `(KDVR, ChanNo)` pairs; endpoints implemented; sample generated per-call CSV paths; tests passed/failed; and any blockers or unsupported streams. **Do not claim these facts until they have been verified from the repository and database.**

**Important distinction:** Forecast accuracy can be validated on observed historical telemetry. Shortage reduction remains an assumption-dependent simulation estimate; do not describe it as verified real-world business impact.
