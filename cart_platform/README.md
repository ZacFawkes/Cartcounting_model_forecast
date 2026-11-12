# Cart Forecasting Platform

Multi-entity forecasting, shortage early-warning and replenishment recommendation,
with **on-demand dataset upload, upload-scoped training and a standalone result
receiver**.

Implements `MULTI_SITE_SYSTEM_DESIGN.md` phases **P0, P1, P3, P4** and the whole
of `CART_PLATFORM_ON_DEMAND_ENTITY_MODULE_SPEC.md` (including §10 three
backtests + automatic calibration and §11 P7 result receiver).

```
cart_platform/
├── settings.py              system defaults (resolution level 3)
├── registry/                HOSTED — registry and resolution
│   ├── schema.sql           19 tables: config, snapshots, models, operational, logs
│   ├── database.py          SQLite access, thread-local connections
│   ├── migrations.py        additive column migrations for existing databases
│   ├── entities.py          entity/site CRUD, discovery, threshold derivation
│   ├── models.py            versioned model + per-entity calibrator registry
│   └── resolver.py          3-level config resolution, red_model/red_eval split
├── server/                  HOSTED — the API
│   ├── app.py               20 routes
│   ├── service.py           scoring + backtesting (shared decision core)
│   ├── store.py             durable sessions, order ledger, score write-back
│   ├── logs.py              entity-keyed logging
│   ├── schemas.py           request/response contracts
│   └── run_server.sh
├── client/                  NOT HOSTED — fake callers
│   ├── platform_client.py   HTTP wrapper + client-side call log
│   ├── stream_caller.py     replays telemetry to /v1/score
│   ├── backtest_caller.py   6 assumption scenarios against /v1/backtest
│   └── run_demo.py          9-step end-to-end walkthrough
├── training/                HOSTED — on-demand training
│   ├── model_config.py      per-entity config resolver, frozen snapshot + hash
│   ├── splits.py            chronological train/calibration/holdout + purging
│   ├── trainer.py           per-entity training on the audited pipeline code
│   ├── calibration.py       automatic conformal calibration + diagnostics
│   ├── backtests.py         the three mandatory reports
│   └── jobs.py              durable job runner, promotion gates
├── integrations/            HOSTED — result publishing
│   ├── result_events.py     versioned event envelope
│   ├── events.py            event construction per result category
│   ├── outbox.py            transactional outbox in platform.db
│   └── publisher.py         HTTP delivery + background retry worker
├── receiver/                SEPARATE SERVICE (port 8200)
│   ├── app.py  schema.sql  database.py  store.py  run_server.sh
├── scripts/
│   ├── make_sample_dataset.py   3-entity sample CSV
│   └── smoke_e2e.py             end-to-end walkthrough against both servers
├── data/platform.db         ML registry + operational data + outbox
├── data/results_receiver.db receiver store (separate file, never merged)
├── logs/scores/<entity>/<date>/<request_id>.csv   one CSV per decision
├── logs/training/<job_id>/<entity>/               metrics, calibration, backtests
└── tests/                   155 tests
    ├── test_platform.py                29  registry, resolution, durability
    ├── test_on_demand.py               40  upload, training, backtests, scoring
    ├── test_result_receiver.py         23  receiver contract, idempotency, config
    ├── test_result_publishing.py       17  outbox, retry, dead-letter, recovery
    ├── test_entity_config_isolation.py 30  per-entity config isolation (A–I)
    └── test_alert_bands.py              16  independent WARNING/CRITICAL bands
```

## Quick start — two servers

```bash
cd /Users/Zac/PycharmProjects/Cartcounting_model_forecast/cart_platform
server/run_server.sh   
pip install -r requirements.txt
cp .env.example .env              # optional; defaults work for local dev
      # http://127.0.0.1:8100/docs

# Terminal B — standalone result receiver (owns data/results_receiver.db)
bash receiver/run_server.sh       # http://127.0.0.1:8200/docs

# Terminal C — health + end-to-end demo
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8200/health
python scripts/make_sample_dataset.py     # 3 bays: 215:0, 215:1, 300:2
python scripts/smoke_e2e.py               # upload → train → promote → score
```

The ML server stays fully functional if the receiver is offline: results are
committed locally and the outbox delivers them when it returns.

## On-demand workflow

```
upload dataset + config
   └── discovers (KDVR, ChanNo) pairs IN THAT FILE ONLY — no DB-wide scan
          └── train the entities YOU select from that upload
                 └── automatic calibration  (never a manual call)
                 └── three backtest reports (forecast / replay / robustness)
                        └── candidate — never auto-promoted
                               └── manual promotion
                                      └── POST /v1/entities/{kdvr}/{chan_no}/score
                                             └── one immutable CSV per decision
                                             └── published to the receiver
```

### 1. Upload

```bash
curl -X POST http://127.0.0.1:8100/v1/datasets/upload \
  -F "file=@data/sample_multi_entity.csv" \
  -F 'config={
        "name": "oct_batch_01",
        "timezone": "Asia/Ho_Chi_Minh",
        "entity_defaults": {"risk_threshold": 0.55, "cooldown_min": 30,
                            "lead_time_min": 10, "max_sequence_gap_min": 15},
        "entity_overrides": {
          "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8": {
                    "bay_capacity": 60, "red_threshold": 15,
                    "amber_threshold": 20, "target_stock": 28},
          "163:215:1:1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0": {
                    "bay_capacity": 90, "red_threshold": 20,
                    "amber_threshold": 28, "target_stock": 40}
        },
        "auto_start_training": false}'
```

The dataset must carry a `BayID` column (map any source name with
`config.column_mapping.BayID`); bay keys are normalised, so the GUID
`06053B44-0A47-42FC-BE3E-F88CB85F99D8` becomes
`06053B44_0A47_42FC_BE3E_F88CB85F99D8`.

Returns the bays found **in that file**, their record counts, data-quality
report, trainability and score URL. An entity without `bay_capacity` is reported
untrainable — physical values are never invented. Re-uploading the same file
with the same config is idempotent (`duplicate_of` is set, nothing is retrained).

### 2. Train only the entities you choose

```bash
curl -X POST http://127.0.0.1:8100/v1/training/jobs \
  -H 'Content-Type: application/json' \
  -d '{"upload_id": "upl_20261019_f669dbc9", "entity_ids": ["163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8"], "strategy": "per_entity"}'



curl http://127.0.0.1:8100/v1/training/jobs/train_20261019025655_2485e2

```

Runs asynchronously. `auto_calibrate` and all three backtests are mandatory
defaults — a caller cannot accidentally disable them for a promotable candidate.
`strategy="pooled"` returns 501 rather than silently pooling unrelated bays.

Each entity in the job resolves and freezes **its own** configuration first, so
the job result reports the thresholds actually used:

```json
"results": {
  "300:2": {
    "frozen_thresholds": {"red_model": 25, "amber_model": 35},
    "effective_config": {"red_threshold": 25, "amber_threshold": 35,
                         "bay_capacity": 120, "target_stock": 50},
    "config_hash": "549777393e8a3846"
  }
}
```

An entity whose configuration is incomplete or contradictory fails with
`status="invalid_config"` and an explicit reason; the rest of the job continues.

### 3. Inspect calibration and the three backtests

```bash
curl http://127.0.0.1:8100/v1/training/jobs/{job_id}/calibration
curl http://127.0.0.1:8100/v1/training/jobs/{job_id}/backtests
curl http://127.0.0.1:8100/v1/training/jobs/{job_id}/backtests/forecast_accuracy
curl http://127.0.0.1:8100/v1/training/jobs/{job_id}/backtests/policy_replay
curl http://127.0.0.1:8100/v1/training/jobs/{job_id}/backtests/closed_loop_robustness
```

| `backtest_type` | What it is | How to read it |
|---|---|---|
| `forecast_accuracy` | Chronological out-of-sample MAE/RMSE/bias per horizon vs a persistence baseline, plus shortage PR-AUC/Brier and interval coverage | **Measured** predictive performance |
| `policy_replay` | `mode="replay"` on the holdout with the fitted calibrator: alerts, trips, carts, order ledger, event attribution | **Simulation estimate**, conditional on assumptions |
| `closed_loop_robustness` | `mode="closed_loop"` plus best-case / audited-default / conservative scenarios | **Scenario envelope** — the conservative row is as valid as the optimistic one |

A report that cannot run is stored as `insufficient_data` or `failed` with the
exact reason, and that blocks promotion. Metrics are never fabricated.

Each report also stores the `config_hash`, the frozen model configuration and
the effective operational configuration it used, so a result read months later
still states which thresholds, target stock, bay capacity and lead time
produced it. Scenario-specific overrides in `closed_loop_robustness` are
recorded alongside, not merged into, the entity configuration.

### 4. Promote (always manual)

```bash
curl -X POST http://127.0.0.1:8100/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/models/mdl_163_215_0_20261015141054_160b75/promote \
  -H 'Content-Type: application/json' -d '{"promoted_by": "you"}'
```

Gates: calibration must be `fitted`, all three backtests `completed`, and the
model must have been trained for **this** entity. A blocked promotion returns
409 with the reasons. `audited_exception_reason` records an explicit override.
`POST /v1/sites/{siteKey}/entities/{kdvr}/{chan_no}/bays/{bay_id}/models/rollback`
restores the prior binding.

### 5. Score

```bash
curl -X POST http://127.0.0.1:8100/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score \
  -H 'Content-Type: application/json' \
  -d '{"records": [ ...at least 12, 20 recommended... ], "params": {}}'
```

One parameterized route backed by the registry — not one route per entity.
A URL without the site or the bay returns **410 ROUTE_MOVED** with the correct
URL, because neither can be inferred. Unregistered bays return 404; registered but unpromoted return
**409 MODEL_NOT_READY**. Each new decision writes
`logs/scores/<entity>/<date>/<request_id>.csv`; a retry returns the original
artifact and places no second order.

```bash
curl http://127.0.0.1:8100/v1/scoring/artifacts/{request_id}
curl "http://127.0.0.1:8100/v1/scoring/artifacts/{request_id}?download=true" -o decision.csv
```

## Result receiver (P7)

A second FastAPI service on `:8200` with its own SQLite file, representing a
future external backend. The ML server writes to it **only over HTTP**.

```bash
curl "http://127.0.0.1:8200/v1/results/training?entity_id=215:0"
curl "http://127.0.0.1:8200/v1/results/calibration?entity_id=215:0"
curl "http://127.0.0.1:8200/v1/results/backtest?backtest_type=policy_replay"
curl "http://127.0.0.1:8200/v1/results/scoring?entity_id=215:0"
curl "http://127.0.0.1:8200/v1/results/entities/215:0"
curl "http://127.0.0.1:8200/v1/results/events/{event_id}"
```

POST semantics: **201** new event, **200** identical replay, **409** same
`event_id` with a different payload (historical results are never overwritten).

Delivery uses a transactional outbox in `platform.db` plus a background worker
with exponential backoff and jitter, `Retry-After` support, dead-lettering of
non-retryable 4xx, and recovery of in-flight rows after a restart.

```bash
curl http://127.0.0.1:8100/v1/integrations/outbox
curl -X POST http://127.0.0.1:8100/v1/integrations/outbox/retry \
  -H 'Content-Type: application/json' -d '{"flush_now": true}'
```

Only metadata and server-local URI references are published — never a pickled
model.

Every published event carries the configuration it was produced under
(`config_hash`, the frozen model config and the effective operational config),
and the receiver projects those into queryable columns on `training_results`,
`calibration_results`, `backtest_results` and `scoring_results`. The field
names of the existing event contracts are unchanged — the config fields are
additive, and events sent without them are still accepted and stored. A
receiver database created before this change is migrated in place on startup.

## Inspecting the databases

```bash
sqlite3 data/platform.db \
  "SELECT entity_id, model_id, role, promotable FROM entity_model_bindings;"
sqlite3 data/platform.db \
  "SELECT entity_id, red_model, amber_model, config_hash FROM config_snapshots;"
sqlite3 data/platform.db \
  "SELECT backtest_type, status, report_id, config_hash FROM backtest_reports WHERE job_id='train_...';"
sqlite3 data/platform.db \
  "SELECT entity_id, decision_time, shortage_risk, should_replenish, artifact_path
     FROM scoring_results ORDER BY decision_time DESC LIMIT 5;"
sqlite3 data/platform.db \
  "SELECT status, COUNT(*) FROM result_outbox GROUP BY status;"

sqlite3 data/results_receiver.db \
  "SELECT backtest_type, status, entity_id, config_hash FROM backtest_results;"
sqlite3 data/results_receiver.db \
  "SELECT event_type, COUNT(*) FROM received_events GROUP BY event_type;"
```

To confirm an entity is serving its own frozen thresholds:

```bash
curl -s localhost:8100/v1/entities/300/2/model | jq '.active.model | {red_model, amber_model}'
curl -s localhost:8100/v1/entities/300:2 | jq '.resolved_config | {red_model, red_threshold}'
```

## Security

- Bearer auth on the ML server when `CART_PLATFORM_API_TOKEN` is set; on the
  receiver when `CART_RESULT_RECEIVER_TOKEN` is set. Tokens come from the
  environment, never from committed code, and are never logged.
- Uploads are size- and extension-limited, streamed to a configured directory;
  no client-supplied server paths are ever used.
- All SQL is parameterized. Model artifacts are loaded only from
  server-produced, registry-recorded paths.
- Snapshots, bundles and scoring CSVs are written atomically and made read-only.

Environment: `CART_PLATFORM_HOST`, `CART_PLATFORM_PORT`, `CART_PLATFORM_DB`,
`CART_PLATFORM_LOG_DIR`, `CART_ENTITY_INCLUDE_SITE`, `CART_ENTITY_INCLUDE_BAY`,
`CART_MIN_HISTORY`.

## Entity grain

```
entity_id = "siteKey:KDVR:ChanNo:BayID"
            e.g. "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8"
```

One camera stream watching one cart bay.

- `CART_ENTITY_INCLUDE_SITE=1` (default): DVR ids restart per store, so the
  site is part of the key, not an attribute of it (decision D9b).
- `CART_ENTITY_INCLUDE_BAY=1` (default): a channel can watch more than one
  bay, so the bay is part of the key (decision D9c). The dataset must carry
  `BayID`, and entity routes become
  `/v1/sites/{siteKey}/entities/{KDVR}/{ChanNo}/bays/{BayID}/...`.

Set either to `0` for a deployment where that part is not needed; ids and URLs
then drop the corresponding segment.

**P0 guarantee:** `block_id` is keyed by entity, so no rolling window, lag or
target can mix two streams. Verified two ways: a dedicated unit test, and a
regression gate showing the refactored pipeline reproduces the pre-refactor
single-site results byte-for-byte across 8 output files.

## Parameter resolution

Request time — scoring, backtesting, policy decisions:

```
1. request `params`      explicit per-call override
2. entity_config         per-KDVR:ChanNo value
   site_config           store-level value (lead time, reserve, trip sizes)
3. settings.SYSTEM_DEFAULTS   fallback, flagged "system_default"
```

### Alert bands

`risk_threshold` opens the **WARNING** band and `critical_risk_threshold` opens
the **CRITICAL** one. They are independent, so a bay can warn early without
calling a middling probability critical:

```bash
curl -X PUT http://127.0.0.1:8100/v1/entities \
  -H 'Content-Type: application/json' \
  -d '{"entity_id":"215:0","kdvr":215,"chan_no":0,"site_key":163,
       "risk_threshold":0.15,"critical_risk_threshold":0.70}'
# => NORMAL < 0.15 <= WARNING < 0.70 <= CRITICAL
```

Both can be set per bay in `entity_overrides` at upload time, changed later
through `PUT /v1/entities`, or overridden for a single call in `params` — none
of which requires retraining. Leaving `critical_risk_threshold` unset
reproduces the original coupling (`risk_threshold + 0.25`, capped at 0.95) and
the source is reported as `derived_from:...` rather than as an operator choice.

`risk_threshold` also gates replenishment, so lowering it increases trips as
well as alerts. `critical_risk_threshold` affects **only** the label.

Training time — the configuration that gets frozen into the model:

```
1. entity_config         per-KDVR:ChanNo value
2. site_config           store-level value
3. settings.SYSTEM_DEFAULTS   fallback (rejected for physical parameters)
```

A request override cannot reach training, and the previous model's frozen
configuration is deliberately not an input. Simulations resolve in the order
scenario override → training snapshot → live entity config → system default,
so a historical backtest stays reproducible after the entity is reconfigured.

Every response carries the resolved value **and its source**:

```json
"resolved_config": {
  "target_stock":  {"value": 28,  "source": "request_override"},
  "red_threshold": {"value": 18,  "source": "entity_config"},
  "lead_time_min": {"value": 12,  "source": "site_config"},
  "usage":         {"value": "MEDIUM", "source": "system_default"}
}
```

Unknown entities are rejected with **404**, never scored with default thresholds.

### The RED/AMBER split (decision D10)

17 of the 127 served features are threshold-derived (`below_red`,
`frac_below_amber_*`, `ttb_red_*`, `ttb_amber_*`), so the thresholds are baked
into the feature vector. The platform therefore separates:

| Role | Name | Resolved when | Overridable per call |
|---|---|---|---|
| Feature construction | `red_model`, `amber_model` — frozen with the model version | at training, from that entity's config | ❌ |
| Evaluation & policy | `red_threshold`, `amber_threshold` — risk, alerts, quantity | at every request | ✅ |

`red_model` is **not a global constant**. Each training job resolves the
thresholds for its own entity and freezes them, so three entities trained from
one upload produce three different frozen configurations:

| entity | `red_model` | `amber_model` | `target_stock` | `bay_capacity` | `config_hash` |
|---|---|---|---|---|---|
| 215:0 | 15 | 20 | 28 | 60 | `34144e293f9d3388` |
| 215:1 | 20 | 28 | 40 | 90 | `3de462682213845c` |
| 300:2 | 25 | 35 | 50 | 120 | `549777393e8a3846` |

Overriding the evaluation threshold works and is useful ("what if we called 30
a shortage?"), but it is never silent:

```
"warnings": ["red_threshold=30 differs from red_model=25; features were built
              with the trained threshold, while risk, alerts and quantity use 30.
              Retrain to change the feature-time threshold."]
```

### Configuration snapshots

Every training job writes an immutable snapshot to `config_snapshots` before
the first feature is built: entity id, effective values, the source of each
value, the model-dependent subset, the operational subset, a deterministic
`config_hash`, the job id and a timestamp. That hash then travels with the
model, the calibrator, all three backtest reports, every scoring decision and
every published event.

Consequences, all covered by regression tests:

- Changing an entity's RED **after** training does not alter the existing
  model, its frozen config, or any historical backtest. The next training job
  picks the new value up.
- An existing model's frozen config is **never** consulted when resolving
  configuration for a new candidate — otherwise a stale threshold would
  propagate forever.
- A missing physical parameter (`bay_capacity`) fails that entity's training
  with `invalid_config` instead of silently borrowing a system default.
- Concurrent jobs for different entities cannot share mutable state; the
  snapshot is passed explicitly down the whole call chain.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, champion model, entity counts |
| GET | `/v1/model/info` | Model provenance, frozen thresholds, history requirements |
| GET | `/v1/models` | Model registry |
| GET/PUT | `/v1/sites` | Site registry |
| GET/PUT | `/v1/entities` | Entity registry (filter by `site_key`, `status`) |
| GET | `/v1/entities/{id}` | Entity + resolved config + warnings |
| POST | `/v1/entities/{id}/status` | active / shadow / disabled |
| POST | `/v1/registry/bootstrap` | Discover entities, adopt the pipeline model |
| POST | `/v1/score` | One live decision, written back to the DB |
| POST | `/v1/backtest` | Backtest one entity, many, or a whole site |
| GET | `/v1/sessions` · POST `/v1/sessions/reset` | Durable online state |
| GET | `/v1/orders` · POST `/v1/orders/{id}/execute` | Order ledger + execution log |
| GET | `/v1/logs/requests` · `/scores` · `/backtests` | Entity-filtered logs |
| GET | `/v1/runs/{run_id}` | Stored backtest artefacts |

### POST /v1/score

```json
{
  "entity_id": "215:0",
  "records": [ {"DVRDate": "...", "next_DVRDate": "...",
                "Count": 22, "Count_IN": 18, "Count_OUT": 14} ],
  "params": {"target_stock": 28},
  "write_to_db": true
}
```

Or identify by `kdvr` + `chan_no` (+ `site_key`) instead of `entity_id`.

**How many records?** Minimum **12** enforced, **10** is the mathematical floor
(features reach back to `t-9`), **20** recommended. Measured: output is
identical from 12 upward and latency is flat, so the margin is free — and after
a telemetry gap you need 10 *fresh* records, which exactly-10 cannot absorb.

Records must be consecutive, chronological, last = newest **completed** record.
Never send an interval still in progress.

### POST /v1/backtest

```json
{"entity_id": "215:0", "start": "2026-10-10", "end": "2026-10-12",
 "mode": "replay", "params": {"risk_threshold": 0.55}, "label": "weekly review"}
```

`site_key` backtests every entity in a store; `entity_ids` takes a list.
`mode`: `replay` (~120 ms) or `closed_loop` (~19 s, rebuilds features from the
simulated state each step).

## Durability

| Concern | How it is handled |
|---|---|
| Server restart | Order book lives in SQLite; session state rebuilt from it |
| Duplicate orders | Pending quantity netted before ordering; order ids unique |
| Replayed record | No-op with reason `"record already processed"` |
| Duplicate write-back | `UNIQUE(entity_id, decision_time)` at the DB level |
| Daily reserve budget | Tracked per entity per day, resets on date change |

All four are covered by tests, including a genuine restart simulation.

## Logging

Every table and log line carries `entity_id` and `site_key`.

| Sink | Content |
|---|---|
| `api_requests` | request_id, entity, endpoint, latency, status, error |
| `scoring_results` | one row per decision + resolved config + warnings |
| `replenishment_orders` | full order lifecycle **+ execution log columns** |
| `backtest_runs` | one row per run + artefact path |
| `logs/runs/<run_id>/` | decisions, order book, events, attribution, series, forecasts |
| `logs/server.log` | application log |

`POST /v1/orders/{id}/execute` records what staff actually did
(`executed_by`, `actual_carts`, `notes`). **This is the gap the simulation audit
identified as blocking** — it is what eventually converts a simulated benefit
into a measured one.

## What the numbers mean

Forecast accuracy is **measured**. Shortage reduction is **simulator output**.
The same policy on the same data, varying only unobservable assumptions:

| Scenario | Events | Severity reduction | Trips | Carts |
|---|---|---|---|---|
| Best case | 13 → 1 | 99.2% | 2 | 31 |
| Audited default | 13 → 2 | 94.6% | 7 | 124 |
| Closed-loop | 13 → 4 | 82.5% | 7 | 106 |
| **Conservative** | 13 → 13 | **0.0%** | 9 | 135 |

Never quote a single reduction percentage. Every response keeps the `caveat`
field and the resolved assumption set.

## Tests

```bash
python -m pytest cart_platform/tests -q          # 139
python -m pytest cart_forecasting/tests -q       #  49
python -m pytest api/tests -q                    #  18
```

Run the three suites separately: `cart_forecasting/tests` and `api/tests` are
both packages named `tests`, so pytest cannot collect them in one invocation.

Upload discovery scoped to one file · untrainable entity without physical config ·
upload idempotency · schema rejection · quality reporting · 409 MODEL_NOT_READY
before promotion · training only the selected entity · other entities untouched ·
entities outside the upload rejected · pooled strategy refused · no auto-promotion ·
immutable versioned artifacts · automatic calibration · calibration window disjoint
from train and holdout · no fabricated calibration for a low-event stream · exactly
three queryable backtests · failed backtest blocks promotion · entity-specific
scoring · cross-entity model borrowing refused · path/body identity mismatch ·
one CSV per decision · retry reuses the artifact · interleaved streams do not
contaminate features · restart durability · orphaned job not marked success ·
receiver 201/200/409 semantics · filtering and pagination · outbox retry, backoff,
dead-letter and restart recovery · receiver outage does not fail scoring ·
backward compatibility of `/v1/score` and the registry routes.

Per-entity configuration isolation is covered end to end: three entities freeze
three different threshold pairs · feature engineering uses each entity's frozen
values · all three backtests record the configuration they ran under · a
post-training threshold change leaves the model and historical reports untouched
while operations move · promotion affects only the promoted entity · concurrent
jobs share no mutable configuration · legacy model artifacts without a frozen
config still load · published events carry the configuration without breaking
the existing event contracts · an invalid override fails loudly instead of
falling back.

## Not yet built

`strategy="pooled"` (returns 501 by design), leave-one-site-out validation,
scheduled retraining with champion–challenger promotion, and drift monitoring.
The registry schema already has `model_registry.loso_metrics` and
`drift_metrics`, so they slot in without a migration.

No source-database scan, no train-all workflow and no automatic promotion exist
anywhere in the implementation.
