# Cart Platform — On-Demand Dataset Registration, Training & Per-Entity Scoring

**Status:** Implementation specification for an AI coding agent. **Replaces the training workflow proposed in `CART_PLATFORM_TRAINING_MODULE_SPEC.md`.**

## 1. User intent — DO NOT MISINTERPRET

Build **one generic, reusable Python server module**. The caller uploads **a dataset and configuration for that upload**. Only **after the upload** does the server inspect **that uploaded dataset** and discover its distinct `(KDVR, ChanNo)` pairs. It registers each discovered stream and makes its corresponding scoring URL available. Training is **on demand for the submitted dataset/entities**, not a single global training pass and **not a scheduled scan of the entire source database**.

**Never** scan SQL Server or all existing registry entities to decide what to train. **Never** automatically retrain all sites. **Never** assume a fixed set of KDVR/ChanNo values or hard-code API routes per entity.

Example: an uploaded dataset contains `(215,0)`, `(215,1)`, `(300,2)`. The server creates/updates three entity registrations and makes these URLs resolvable:

- `POST /v1/entities/215/0/score`
- `POST /v1/entities/215/1/score`
- `POST /v1/entities/300/2/score`

Use **one parameterized FastAPI route** `POST /v1/entities/{kdvr}/{chan_no}/score` backed by a registry, not three runtime-added route definitions. Unregistered pairs return 404. Existing `/v1/score` must remain compatible.

The user may upload **only one entity**, or **many entities in the same file**. There is **no requirement to train every entity at once**. Each entity is independently registered, trainable, deployable, scored and logged. `siteKey` is metadata; entity identity remains `(KDVR, ChanNo)` unless existing `CART_ENTITY_INCLUDE_SITE=1` requires site-scoped uniqueness. Honor that existing convention consistently and reject ambiguous routing.

## 2. Required lifecycle

```text
Caller uploads dataset + JSON config
        |
        v
POST /v1/datasets/upload (multipart: file + config)
        |
        +--> Save immutable upload snapshot, validate schema, generate upload_id
        +--> Discover DISTINCT (KDVR, ChanNo) FROM THIS UPLOAD ONLY
        +--> Return discovered entities, counts, quality, conflicts
        +--> Register eligible entity identities (initially untrained/shadow)
        |
        v
POST /v1/training/jobs (upload_id + selected entity_ids + training config)
        |
        +--> Train selected entity(s) using original cart_forecasting code
        +--> Time-aware validation/calibration, metrics, versioned .pkl artifacts
        +--> Register CANDIDATE model bindings per entity
        |
        v
POST /v1/entities/{kdvr}/{chan_no}/models/{model_id}/promote
        |
        +--> Validate feature schema/config/calibrator, atomically activate
        |
        v
POST /v1/entities/{kdvr}/{chan_no}/score
        |
        +--> Latest >=12 completed records for THAT SAME entity (20 preferred)
        +--> Resolve active entity-specific model/config, build old features
        +--> Predict + shortage risk + advisory replenishment
        +--> Persist scoring_results and per-call CSV artifact
        +--> Return JSON + CSV path/artifact ID
```

**Key separation:** Upload/discovery ≠ training ≠ promotion ≠ scoring. Upload may register URLs immediately, but scoring returns a clear `409 MODEL_NOT_READY` until a compatible model has been promoted (unless the caller explicitly opts into an existing compatible fallback model, clearly marked degraded/shadow). Do not claim a model is ready simply because the route resolves.

## 3. Upload contract

`POST /v1/datasets/upload` accepts multipart form fields:

- `file`: CSV initially (optionally Parquet later); streamed to a configured server-side directory; no arbitrary client-supplied server filesystem paths.
- `config`: JSON describing dataset mapping, entity defaults/overrides and intended training behavior. Validate with Pydantic.

Illustrative config:

```json
{
  "name": "oct_cart_batch_01",
  "timezone": "Asia/Ho_Chi_Minh",
  "column_mapping": {
    "KDVR": "KDVR", "ChanNo": "ChanNo", "siteKey": "siteKey",
    "DVRDate": "DVRDate", "next_DVRDate": "next_DVRDate",
    "Count": "Count", "Count_IN": "Count_IN", "Count_OUT": "Count_OUT"
  },
  "entity_defaults": {
    "max_sequence_gap_min": 15,
    "risk_threshold": 0.55,
    "cooldown_min": 30,
    "lead_time_min": 10
  },
  "entity_overrides": {
    "215:0": {"bay_capacity": 60, "red_threshold": 15, "amber_threshold": 20, "target_stock": 28},
    "215:1": {"bay_capacity": 90, "red_threshold": 20, "amber_threshold": 28, "target_stock": 40}
  },
  "auto_start_training": false
}
```

- `entity_overrides` may omit unknown entities; **do not invent** physical bay capacity. Missing critical config keeps entity untrained/shadow or blocks promotion as appropriate.
- Preserve three-level resolution: request override → entity/site config → system default, with source audit. Preserve frozen `red_model/amber_model` versus `red_threshold/amber_threshold` at evaluation.
- `Count_IN` and `Count_OUT` are **customer traffic**, not carts moved.
- Dataset must contain `KDVR`, `ChanNo`, `DVRDate`, `next_DVRDate`, `Count`, `Count_IN`, `Count_OUT`; `siteKey` required if configured identity is site-scoped or needed for LOSO. Reject missing/malformed fields; do not silently cross-join external tables.
- Validate duplicates, timestamp order, gaps, incomplete intervals, negative/impossible values, inconsistent site mapping, and repeated entity keys. Exclude incomplete intervals from scoring/training. Persist per-entity quality report.
- Upload idempotency: same file checksum + config fingerprint should not create duplicate training or entity side effects; allow explicit new version.
- Registry discovery must be scoped to `upload_id`. Never discover from the entire platform DB by default.

Example upload response:

```json
{
  "upload_id": "upl_20261012_0001",
  "status": "registered",
  "entities": [
    {"kdvr": 215, "chan_no": 0, "entity_id": "215:0", "records": 1500, "score_url": "/v1/entities/215/0/score", "model_status": "not_ready"},
    {"kdvr": 215, "chan_no": 1, "entity_id": "215:1", "records": 1450, "score_url": "/v1/entities/215/1/score", "model_status": "not_ready"}
  ],
  "warnings": []
}
```

## 4. Training is explicitly scoped to an upload

`POST /v1/training/jobs`:

```json
{
  "upload_id": "upl_20261012_0001",
  "entity_ids": ["215:0"],
  "strategy": "per_entity",
  "validation": {"method": "chronological_purged", "holdout_fraction": 0.2},
  "promote": false
}
```

- **Default** `strategy=per_entity`: train **only** selected entities from **only** the supplied `upload_id`; never train other registered entities.
- `entity_ids` optional: if omitted, use only the entities in **this upload**, never all known entities. Return discovered entity list for confirmation when the upload contains multiple streams; allow selection before training.
- `strategy=pooled` is optional and must be explicitly requested; never silently pool unrelated bays.
- A dataset can be uploaded, inspected, then trained later; optional `auto_start_training=true` may enqueue jobs **for this upload only** after validation, never run heavy work inside the upload HTTP handler.
- Reuse `cart_forecasting` feature/target construction, gap-aware entity grouping, chronological splits, clipping and model training. Keep train/validation/holdout isolated; calibrate using out-of-training predictions only.
- If a stream lacks enough days, samples, or shortage events for trustworthy training/calibration, return a detailed `INSUFFICIENT_TRAINING_DATA` result and keep it shadow; do not manufacture metrics or claim the old single-site model is validated for it.
- Model artifacts are immutable/versioned, e.g. `cart_platform/models/per_entity/215_0/<model_id>/bundle.pkl` plus `metadata.json` and calibrator artifact. Store path, dataset checksum, upload_id, feature schema, frozen thresholds, training windows, metrics, model status in the registry.
- Promotion is explicit; candidate training must not replace the active champion. Rollback must restore the prior entity binding.
- Run training in a durable worker/job with status/progress/error reporting; restart must not leave a false SUCCESS.

## 5. Entity scoring API

Use the parameterized path:

`POST /v1/entities/{kdvr}/{chan_no}/score`

```json
{
  "records": [
    {"DVRDate": "2026-10-12T10:00:00", "next_DVRDate": "2026-10-12T10:06:40", "Count": 30, "Count_IN": 12, "Count_OUT": 8}
  ],
  "params": {},
  "write_to_db": true
}
```

The example shows one row only to illustrate schema; real calls require at least **12 usable consecutive completed rows** (20 recommended), or the actual minimum determined by the active model's frozen feature contract. Validate path identity against any KDVR/ChanNo fields in body; reject mismatches. Derive decision time from newest **completed** interval; do not use server receipt time as event time.

Return predictions (20/40-min horizons and min, as supported by actual model), lower bound, risk, alert, advisory replenishment, resolved config + sources, warnings, entity_id, model_id, calibration_id, request_id, idempotency status, and CSV artifact location. Do not claim a recommendation was executed unless a separate action log confirms it.

**State isolation:** each entity has its own durable session/cooldown/order book; existing idempotency must prevent duplicate orders on retries/restarts. Preserve `/v1/score` by internally delegating to the same service function.

**CSV on every score call:** create a unique immutable artifact per **new successful scoring decision**, e.g. `logs/scores/215_0/2026-10-12/<request_id>.csv`, one row with all outputs, identifiers, timestamp, model/version, config sources, warnings, and artifact metadata. Maintain append-only `scoring_results` DB table and optionally a per-entity aggregate CSV. On an idempotent retry, return the original result/artifact path rather than creating a duplicate decision. Use atomic file writes and a cleanup/retention policy. Protect artifact retrieval endpoints.

## 6. How new DB records enter scoring (separate optional adapter)

The server **does not** need to scan the database for training. After registration and promotion, a caller/collector may fetch the latest 20 completed rows **for the specified `(KDVR, ChanNo)`** when a new telemetry record arrives and POST them to that entity's scoring URL. This is a **separate integration**, not a prerequisite for upload-driven training.

If building a DB collector later, scope queries to the registered entity, use checkpoints, retry/idempotency and never mix streams. Do not infer that `tbl_CNN_Count` alone contains joined customer traffic columns; inspect actual SQL query.

## 7. Required endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/datasets/upload` | Upload file+config; discover pairs in this file |
| GET | `/v1/datasets/{upload_id}` | Snapshot, discovered pairs, quality |
| GET | `/v1/datasets/{upload_id}/entities` | List entities discovered from this upload |
| POST | `/v1/training/jobs` | Train selected pairs from one upload |
| GET | `/v1/training/jobs/{job_id}` | Job status, errors, metrics, artifacts |
| GET | `/v1/entities/{kdvr}/{chan_no}/model` | Active/candidate model binding and readiness |
| POST | `/v1/entities/{kdvr}/{chan_no}/models/{model_id}/promote` | Activate validated candidate for this entity |
| POST | `/v1/entities/{kdvr}/{chan_no}/score` | Entity-specific score endpoint |
| GET | `/v1/scoring/artifacts/{request_id}` | Authenticated CSV artifact retrieval |

Keep existing routes working; route count can change. If `CART_ENTITY_INCLUDE_SITE=1`, add unambiguous site-scoped route (`/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/score`) or require an explicit site identifier, and document precedence.

## 8. Storage and security

- Use existing `cart_platform/data/platform.db` and schema migrations for development; do not overwrite the file or reset live tables. Add `dataset_uploads`, `upload_entities`, `training_jobs`, `entity_model_bindings` and artifact references as needed; adapt to existing model registry rather than duplicating equivalent tables.
- Keep uploaded datasets under a configured data directory and model artifacts under a separate configured models directory; never use untrusted paths from HTTP input.
- Apply file size/type limits, authentication/authorization to upload/train/promote/score/artifact routes, parameterized SQL, safe artifact loading from trusted server-produced models only (pickle is unsafe for untrusted files), and atomic promotion with rollback.
- Persist complete audit trail: `upload_id`, `entity_id`, config snapshot, model_id, job_id, request_id, input checksum, scoring output and CSV path.

## 9. Tests / acceptance criteria

1. Upload a CSV with two pairs; response lists **exactly those two pairs** and URLs, even if registry contains ten unrelated pairs.
2. Upload only `(215,0)` and train it: `(215,1)` and all other entities are **not trained or modified**.
3. An entity-specific score URL exists as a parameterized route but returns `409 MODEL_NOT_READY` until a compatible model is promoted.
4. After training/promoting `(215,0)`, `/v1/entities/215/0/score` uses the correct model and config; `(215,1)` cannot accidentally use that model.
5. Training is asynchronous and versioned; prior `.pkl` and champions remain unchanged until promotion.
6. Interleaved KDVR/ChanNo records never contaminate features, targets, state or CSV logs.
7. Score request with 20 completed rows succeeds; insufficient usable history is rejected clearly.
8. Every successful new decision creates one entity-specific CSV; retries reuse the same artifact and do not duplicate orders.
9. Server restart retains model bindings, sessions and scoring results.
10. Existing `cart_platform` tests pass, and original single-site regression outputs remain unchanged.
11. Uploaded CSV with conflicting site mapping, invalid schema or missing physical config is rejected or kept shadow with clear diagnostics.
12. No implicit database-wide discovery, no automatic all-site training, and no automatic promotion anywhere in the implementation.

## 10. Coding agent execution instructions

1. Inspect `MULTI_SITE_SYSTEM_DESIGN.md`, `cart_platform/README.md`, `cart_platform/registry/*`, `cart_platform/server/*`, `cart_forecasting/*`, existing SQLite schema, model bundle format and tests. **Do not assume paths/functions in this spec already exist.**
2. Summarize current reusable components and identify precise integration points before editing.
3. Implement in phases: upload + discovery → scoped training → registry binding/promotion → parameterized score route → CSV per decision → integration tests.
4. Use actual code to determine minimum history, feature schema, model targets and trainability; never silently change the trained feature definition.
5. Provide a minimal example CSV/config and exact `curl`/Python commands for upload → inspect → train → promote → score → inspect CSV and SQLite from PyCharm.
6. Run tests and show results; distinguish executed checks from assumptions. Do not start scheduled retraining or auto-dispatch replenishment.

**Definition of done:** I can send one dataset and its config to the Python server, get back the unique `(KDVR, ChanNo)` values found **in that dataset**, train **only the entities I choose**, promote a model for each, call its stable `/KDVR/ChanNo` scoring endpoint, and inspect a distinct CSV output for each scoring decision. No global DB scan or train-all workflow is involved.

## 10. MANDATORY ADDENDUM — Three backtest result types and automatic calibration (2026-10-12)

**These requirements override any earlier wording that treats calibration as optional or backtesting as a single report.** After training an entity, the server must automatically produce **three clearly distinguished backtest results** for that trained candidate, using only that upload's data for the selected entity. Do not run backtests on unrelated registered entities. All results must be saved and retrievable before promotion.

### 10.1 Exactly three backtest types per candidate

Provide a stable `backtest_type` enumeration and three separate reports:

1. **Forecast / predictive backtest (`forecast_accuracy`)**: strictly chronological, out-of-sample walk-forward or purged holdout evaluation of the predicted cart counts and shortage probabilities. Report per horizon (including 20/40 minutes where supported) MAE, RMSE, bias, baseline comparison (persistence and other existing baselines), sample counts, interval coverage and shortage classification metrics (precision, recall, PR-AUC/Brier where defined). Plot actual vs forecast, intervals and shortage events. This is a **measured predictive performance** report, not a replenishment benefit claim.
2. **Historical policy replay (`policy_replay`)**: run the existing `/v1/backtest` `mode="replay"` against the held-out historical telemetry using the candidate model and its **automatically fitted calibrator**. Report alert timing, lead time, baseline vs simulated shortage count/severity, trips, cart quantities, pending orders and decision ledger. Clearly identify replenishment benefits as **simulation estimates conditional on assumptions**; do not call them observed reductions.
3. **Closed-loop & assumption-sensitivity (`closed_loop_robustness`)**: run the existing `mode="closed_loop"`, rebuilding features after simulated interventions, plus a conservative assumption sensitivity check (and, where supported, best-case/audited default). Report a comparable scenario table with shortage events, severity, trips, carts, assumptions and caveats. Conservative outcomes must be visible even if they show zero benefit. Never present one percentage as the definitive outcome.

If the existing design document defines a different canonical set of "three backtest types", **inspect it and retain its original semantics**, document the mapping, and do not silently substitute new labels. The three types above are the default interpretation for this addendum.

**Leakage rule:** use time-aware train/calibration/test partitions. Do not train, calibrate, tune thresholds or select models on the final backtest holdout. Forecast backtest metrics must be computed on predictions from a model that did not train on those targets. Reuse the original feature engineering, threshold feature freeze, scoring core and simulation audit logic; do not duplicate conflicting decision rules.

### 10.2 Calibration MUST be automatic

- On every successful training job, fit a calibrator **automatically**, with no manual API call, using out-of-training, time-respecting calibration predictions. Store immutable calibrator artifact, method, training/calibration date ranges, sample count, positive event count, reliability diagnostics and version linked to `model_id` and `entity_id`.
- For `per_entity`, prefer per-entity calibration **when sufficient samples and both outcome classes exist**. If data are inadequate, use a **compatible explicitly validated global calibrator only if one is available**, mark `calibration_scope="global_fallback"` and emit a warning. Otherwise set `calibration_status="insufficient_data"`, do not invent probabilities and **block promotion for calibrated risk scoring**. Raw point forecast may remain available in shadow if its contract permits it.
- For `pooled` (only if explicitly requested), fit pooled/global calibration from out-of-training pooled predictions and optionally entity-specific calibration when sample support is sufficient. Never leak entity test outcomes into calibration.
- Evaluate calibration with Brier score, reliability bins / calibration curve, and, when meaningful, ECE and event counts; compare pre/post calibration on a disjoint holdout. No guarantees of improved calibration: record regressions transparently.
- Calibration must match model feature schema, target horizon, risk event definition and frozen `red_model/amber_model`. Reject mismatched model/calibrator bindings at promotion and scoring. Preserve separate configurable evaluation thresholds.
- Promotion stays **manual**. A finished training job is a candidate, never an automatic champion. The job response includes calibrator metadata and links to all three backtests.

### 10.3 API contracts

Keep the existing upload, train, promote and score APIs. Extend training jobs with optional:

```json
{
  "upload_id": "upl_20261012_0001",
  "entity_ids": ["215:0"],
  "strategy": "per_entity",
  "auto_calibrate": true,
  "run_backtests": ["forecast_accuracy", "policy_replay", "closed_loop_robustness"],
  "promote": false
}
```

`auto_calibrate=true` and all three backtests are **mandatory defaults**; callers must not accidentally disable them for a promotable candidate. If a report cannot run (e.g. insufficient holdout events, simulation prerequisites absent), mark that report `insufficient_data` or `failed` with exact reasons, not fake metrics; candidate remains unpromotable until required gates pass or an explicit audited exception is recorded.

Add:

- `GET /v1/training/jobs/{job_id}/backtests` — returns the **three** backtest report statuses and summary metrics, with distinct report IDs and artifact paths.
- `GET /v1/training/jobs/{job_id}/backtests/{backtest_type}` — report JSON and references to plots/CSV files.
- `GET /v1/training/jobs/{job_id}/calibration` — scope, method, quality metrics, artifact ID and fallback/warnings.

A completed job response should include:

```json
{
  "job_id": "train_001",
  "entity_id": "215:0",
  "model_id": "model_001",
  "status": "candidate_ready",
  "calibration": {"status": "fitted", "scope": "entity", "calibrator_id": "cal_001"},
  "backtests": {
    "forecast_accuracy": {"status": "completed", "report_id": "bt_f_001"},
    "policy_replay": {"status": "completed", "report_id": "bt_r_001"},
    "closed_loop_robustness": {"status": "completed", "report_id": "bt_c_001"}
  },
  "promoted": false
}
```

### 10.4 Artifact layout and acceptance tests

For each `upload_id/entity_id/model_id`, persist separate immutable outputs:

```text
logs/training/<job_id>/<entity_id>/
  training_metrics.json
  calibration/metadata.json
  calibration/reliability.csv
  backtests/forecast_accuracy/report.json
  backtests/forecast_accuracy/predictions.csv
  backtests/policy_replay/report.json
  backtests/policy_replay/decisions.csv
  backtests/closed_loop_robustness/report.json
  backtests/closed_loop_robustness/scenarios.csv
```

Tests MUST verify: (a) training one selected entity yields exactly three separately queryable backtest statuses and an automatically created calibration record; (b) no calibration/holdout leakage; (c) a low-event entity does not receive fabricated calibration; (d) all three reports reference the correct entity and model version; (e) backtest failure cannot silently mark a candidate promotable; (f) existing 29+ tests and prior single-site regression gates still pass. Retain API auth and atomic artifact writes.

---

## 11. P7 — Standalone local result receiver API (MANDATORY)

### 11.1 Purpose and boundary

Implement a **second local FastAPI service** that represents a future external backend. It owns a **separate SQLite database** at `cart_platform/data/results_receiver.db` and listens on `127.0.0.1:8200` by default. The existing ML server listens on `127.0.0.1:8100` by default and continues to own `data/platform.db`.

**All training, automatic calibration, three backtest results, and live scoring results must be published from the ML server to the receiver through HTTP**, never by opening the receiver's SQLite database directly. Receiver is an integration/test target now; later the caller may replace its URL with an actual backend without changing training or scoring logic. The ML server must remain functional if the receiver is offline.

```text
Caller -- dataset + config --> ML API :8100
                              | upload-scoped entity discovery
                              | selected-entity train -> auto calibration
                              | 3 backtests -> candidate -> manual promotion
                              | entity scoring
                              v
                         local platform.db + durable outbox
                              |
                         background HTTP publisher (retry)
                              |
                              v
                       Receiver API :8200
                              |
                       results_receiver.db
```

Do not merge receiver tables into `platform.db`. Do not automatically scan external SQL Server or train all registered entities. Do not auto-promote models. Keep the existing `/v1/score`, `/v1/backtest`, registry APIs and single-site compatibility.

### 11.2 Receiver API contract

Expose these routes under `/v1/results` (plus `/health` and `/docs`):

| Method | Route | Behavior |
|---|---|---|
| POST | `/v1/results/training` | Upsert-on-event-ID training outcome and candidate model metadata |
| POST | `/v1/results/calibration` | Store auto-calibration status, scope, diagnostics and artifact metadata |
| POST | `/v1/results/backtest` | Store exactly one backtest report/event; `backtest_type` required |
| POST | `/v1/results/scoring` | Store one scoring decision and its forecast/risk/order recommendation |
| GET | `/v1/results/training` | List/filter training results |
| GET | `/v1/results/calibration` | List/filter calibration results |
| GET | `/v1/results/backtest` | List/filter backtest reports, including `backtest_type` |
| GET | `/v1/results/scoring` | List/filter scoring decisions |
| GET | `/v1/results/entities/{entity_id}` | Aggregate results for an entity (URL-encode entity IDs if needed) |
| GET | `/v1/results/events/{event_id}` | Retrieve an exact received event |
| GET | `/health` | Receiver liveness + SQLite connectivity |

All list endpoints support pagination and filters when applicable: `entity_id`, `site_key`, `upload_id`, `job_id`, `model_id`, `backtest_type`, time range. Return stable JSON response contracts, explicit validation errors and OpenAPI schemas. Prefer one common envelope for all POSTs:

```json
{
  "event_id": "train_001:215:0:forecast_accuracy:completed:v1",
  "event_type": "backtest.completed",
  "schema_version": 1,
  "occurred_at": "2026-10-12T14:00:00Z",
  "entity_id": "215:0",
  "site_key": "163",
  "upload_id": "upl_001",
  "job_id": "train_001",
  "model_id": "model_001",
  "payload": {
    "backtest_type": "forecast_accuracy",
    "status": "completed",
    "metrics": {"mae": 2.9, "rmse": 4.9},
    "artifacts": [{"kind": "predictions_csv", "uri": "logs/training/train_001/215:0/backtests/forecast_accuracy/predictions.csv"}],
    "caveat": "Forecast accuracy is measured on held-out observations; not an observed replenishment improvement."
  }
}
```

Values above are **illustrative only**. For the other two backtests, use `backtest_type` = `policy_replay` or `closed_loop_robustness`; each must be stored as an independently queryable report. Include scenario assumptions and simulation caveats in both policy reports. For failed/insufficient-data runs, publish the real status and reason rather than fabricated metrics. Calibration events include method, scope (`entity`, `global_fallback`, or `insufficient_data`), sample/event counts, Brier/reliability diagnostics, and linked calibrator ID. Scoring events include decision time, resolved config and sources, frozen model thresholds, forecast horizons, risk/alert, order recommendation, warnings, and model/calibrator versions. Avoid storing secrets or unnecessary raw telemetry in the receiver.

POST success returns `201 Created` for a new event or `200 OK` for an identical replay, both with the same `event_id` and stored record reference. Same `event_id` with a different canonical payload must return `409 Conflict`, not silently overwrite historical results. Use a unique constraint on `event_id` and optionally `(entity_id, decision_time, model_id)` for scoring where contractually appropriate.

### 11.3 SQLite receiver schema

Create schema and migrations for `results_receiver.db`. Suggested tables:

- `received_events`: `event_id` PK, `event_type`, `schema_version`, `entity_id`, `site_key`, `upload_id`, `job_id`, `model_id`, `occurred_at`, `received_at`, `payload_hash`, `payload_json`.
- `training_results`: `event_id` FK/UNIQUE, `job_id`, `entity_id`, `model_id`, `status`, `metrics_json`, `metadata_json`.
- `calibration_results`: `event_id` FK/UNIQUE, `job_id`, `entity_id`, `model_id`, `calibrator_id`, `status`, `scope`, `metrics_json`, `metadata_json`.
- `backtest_results`: `event_id` FK/UNIQUE, `job_id`, `entity_id`, `model_id`, `backtest_type`, `status`, `metrics_json`, `assumptions_json`, `caveat`, `artifacts_json`.
- `scoring_results`: `event_id` FK/UNIQUE, `entity_id`, `model_id`, `decision_time`, `result_json`, `warnings_json`.

Use SQLite WAL mode, transactions, indexed filter columns, UTC timestamps, parameterized SQL, and idempotent initialization. Never assume the receiver can open ML-server-local artifact paths; those paths are references only. Expose authenticated artifact-download endpoints on the ML server if retrieval from a separate host is required; do not return an inaccessible filesystem path as a public URL.

### 11.4 ML-side publisher, durable outbox and retry

Add an ML-side `result_outbox` table in **`platform.db`** (migration required) with `event_id` unique, destination, event_type, payload JSON, status (`pending`, `sending`, `sent`, `dead_letter`), attempts, next_attempt_at, last_error, created_at, sent_at. Enqueue the event **in the same transaction as the corresponding local result whenever feasible**, or implement an explicit transactional reconciliation strategy to prevent lost events.

- Emit events on **completed training**, **automatic calibration**, **each of the three backtest reports**, and **every successfully persisted scoring decision**. Publish failed/insufficient-data statuses too where useful for diagnosis.
- Run a background delivery worker independent of API response latency. Retry transient HTTP/network/5xx/429 errors with exponential backoff + jitter; honor `Retry-After` when present; classify non-retryable 4xx as dead-letter (except transient cases). Recover pending/sending messages after process restart. Avoid duplicate sends when multiple workers run.
- Receiver downtime **must not fail training or scoring** once the local result/outbox commit succeeds. Do not block scoring on remote network latency.
- Provide outbox observability (`GET /v1/integrations/outbox` with filters/status/counts) and a safe manual retry operation for dead letters. Do not log API keys.
- Use environment variables such as `CART_RESULT_RECEIVER_URL=http://127.0.0.1:8200`, `CART_RESULT_RECEIVER_ENABLED=1`, `CART_RESULT_RECEIVER_TOKEN`, `CART_RESULT_PUBLISH_TIMEOUT_SEC`, `CART_RESULT_PUBLISH_MAX_ATTEMPTS`. Require bearer-token authentication for receiver writes when enabled; configure local-dev token via environment, not committed code.
- Keep publishing configurable/off for isolated tests. Document the behavior if receiver URL is missing.

### 11.5 Suggested project layout

```text
cart_platform/
  integrations/
    result_events.py           # typed event envelope and versioned schemas
    publisher.py               # HTTP client
    outbox.py                  # transactional queue
    retry_worker.py            # retry/recovery worker
  receiver/
    app.py                     # standalone FastAPI service
    schemas.py                 # POST/GET contracts
    database.py                # receiver SQLite connection and migrations
    store.py                   # idempotent inserts and filtered queries
    run_server.sh              # port 8200
  data/
    platform.db                # existing ML DB + outbox table
    results_receiver.db        # receiver DB, separate file
  tests/
    test_result_receiver.py
    test_result_publishing.py
```

Use the repository's actual conventions if different; do not assume the proposed modules already exist.

### 11.6 End-to-end acceptance tests (mandatory)

1. Start ML server and receiver server independently. Upload a CSV with multiple `(KDVR, ChanNo)` pairs; assert only pairs from **that upload** are registered. No database-wide scan.
2. Train only one selected entity. Assert auto-calibration and **exactly three backtest records** (`forecast_accuracy`, `policy_replay`, `closed_loop_robustness`) are generated locally and delivered to the receiver, with matching `upload_id`, `entity_id`, `job_id`, `model_id` and accurate statuses.
3. Verify `GET /v1/results/entities/{entity_id}` and backtest filtering return only that entity's records; unrelated entities are untouched.
4. Manually promote the candidate after gates pass, score via `/v1/entities/{kdvr}/{chan_no}/score`, verify local score CSV and a matching receiver `scoring_results` row.
5. Send the same event twice: exactly one receiver row; change the payload with the same ID: HTTP 409.
6. Stop the receiver, perform a scoring decision, restart ML server and receiver: score persists locally and the outbox eventually delivers **exactly one** receiver record.
7. Test receiver 5xx/timeout, invalid token, validation error, partial job failure, insufficient calibration data, outbox retry/dead-letter behavior, and recovery after restart.
8. Confirm model artifacts are not sent as binary `.pkl` in every event; only metadata and safe references. All existing platform tests and single-site regression gates still pass.

### 11.7 Developer commands and deliverables

Document exact commands based on actual project environment, e.g.:

```bash
# Terminal A: existing ML server
cd cart_platform
bash server/run_server.sh

# Terminal B: standalone result receiver
cd cart_platform
bash receiver/run_server.sh

# Health / Swagger
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8200/health
# Open http://127.0.0.1:8100/docs and http://127.0.0.1:8200/docs
```

Provide example upload/config, selected-entity train request, promotion request, score request, a representative receiver POST, GET queries for each of the **three backtest types**, and SQL queries for both databases. Supply schema migrations, automated tests, a small end-to-end smoke script, a `.env.example` without real secrets, and an API contract (`openapi.json` or documented Pydantic schemas) that another backend can implement later.

**Final definition of done:** A caller uploads dataset + config; the ML server discovers only uploaded KDVR/ChanNo pairs; selected entities are trained with automatic calibration and all three backtests; after manual promotion they are scored via entity-specific parameterized URLs; all result categories are retained locally and **automatically delivered over HTTP to a standalone SQLite-backed result API** on port 8200, resilient to receiver outages. No source-DB scan, no forced train-all, no auto-promotion, no cross-entity leakage.
