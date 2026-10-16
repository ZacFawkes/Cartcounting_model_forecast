# Python server — scoring and training workflow

What the ML server (`:8100`) does when the backend sends it telemetry — and
how a model gets trained, calibrated, backtested and promoted in the first
place. Every packet below is a real request captured against the running
service.

**Part A — scoring** (§1–§10): the hot path the backend calls continuously.
**Part B — training** (§11–§17): the lifecycle path, run by a human.

---

# Part A — Scoring

---

## 1. The short answer

```
Backend ──POST /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/score──▶ Python server
        ◀──────── HTTP 200 + the full decision ─────────
                     (synchronous, 75–125 ms)
                                 │
                                 └─▶ the server then pushes a copy to the
                                     receiver :8200 on its own, in the
                                     background. The backend does nothing.
```

**The HTTP response *is* the result.** There is no callback, no webhook and
nothing to poll. If you got `200`, you have the decision.

The second path exists only to archive results for reporting. The backend never
waits for it.

---

## 2. The endpoint

```
POST /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/score
Authorization: Bearer <CART_PLATFORM_API_TOKEN>
Content-Type: application/json
```

One parameterised route for every bay — not one route per entity. A variant
carrying the site is available when DVR ids are not globally unique:

```
POST /v1/sites/{site_key}/entities/{kdvr}/{chan_no}/score
```

### Request packet

```json
{
  "records": [
    {"DVRDate": "2026-10-11T15:03:22", "next_DVRDate": "2026-10-11T15:10:03",
     "Count": 11.0, "Count_IN": 4.0, "Count_OUT": 1.0}
  ],
  "write_to_db": true,
  "client": "backend-poller",
  "params": {"risk_threshold": 0.15}
}
```

| Field | Required | Rule |
|---|---|---|
| `records[]` | yes | **Consecutive, chronological, one entity.** Minimum 12, 20 recommended. The last record is the newest *completed* interval. |
| `write_to_db` | no | Default `true`. `false` = dry run: scores and returns, persists nothing, publishes nothing. |
| `client` | no | Free-text caller name, written to the request log. |
| `params` | no | Per-call config override. Highest precedence — see §6. |
| `kdvr`, `chan_no` | no | If present they **must match the path**, otherwise `422`. |

`Count` is the carts in the bay at the **end** of the interval. `Count_IN` /
`Count_OUT` are customers during it.

---

## 3. What the server does, in order

```
 1  auth middleware          401 if the bearer token is wrong
 2  trace middleware         assigns request_id, starts the timer
 3  path/body agreement      422 if body kdvr/chan_no differ from the path
 4  entity lookup            404 unregistered · 503 disabled
 5  model resolution         409 MODEL_NOT_READY when nothing is promoted
 6  config resolution        params > entity_config > site_config > defaults
 7  records → frame          422 if < 12 records or more than one entity
 8  feature build (127)      422 if the window has a gap and a feature is NaN
 9  load session             alert state + simulated surplus, per entity
10  predict                  4 targets: cart@20, cart@40, min@20, min@40
11  calibrate                conformal → shortage_risk, lower bound
12  alert engine             NORMAL / WARNING / CRITICAL, new_predictive_alert
13  replenishment plan       quantity, urgency, cooldown, capacity, reserve
14  idempotency check        (entity_id, decision_time) already scored?
15  write CSV artifact       one immutable file per decision
16  persist                  20 columns into scoring_results
17  enqueue outbox           same transaction as step 16
18  respond                  200 + the full decision
19  (background)             outbox worker POSTs to the receiver :8200
```

Steps 1–13 always run. Steps 15–17 are skipped when the decision already
exists (step 14) or when `write_to_db` is `false`.

**Important:** the feature vector is rebuilt with the thresholds **frozen into
the model**, never with the current operational values. That is the trained
contract.

---

## 4. The response packet

```json
{
  "request_id": "req-1955401830",
  "entity_id": "215:0", "kdvr": 215, "chan_no": 0, "site_key": 163,
  "entity_status": "active",
  "decision_time": "2026-10-11T15:10:03",
  "current_cart": 11.0,
  "simulated_cart": 11.0,

  "forecast": {
    "cart_at_20min": 9.42,
    "cart_at_40min": 7.88,
    "min_within_20min": 8.90,
    "min_within_40min": 6.73,
    "lower_bound_min_within_40min": 1.10,
    "horizon_note": "horizons are counted in records; 3 ≈ 20 min, 6 ≈ 40 min"
  },

  "shortage_risk": 0.99,
  "alert_state": "CRITICAL",
  "new_predictive_alert": true,
  "reactive_critical": false,

  "action": {
    "should_replenish": true,
    "quantity": 20,
    "urgency": "URGENT",
    "order_id": "ord-215-0-000042",
    "decision_time": "2026-10-11T15:10:03",
    "expected_completion_time": "2026-10-11T15:20:03",
    "actionable_before_breach": true,
    "minutes_to_predicted_breach": 13.3,
    "reason": "risk 0.99 >= 0.15; predicted 40-minute minimum 6.7 carts against a target of 28; ordered 20 after netting 0 already in transit"
  },

  "pending_carts": 20.0,
  "pending_orders": [],

  "model_id": "mdl_215_0_20261013165052_053efa",
  "calibration_id": "cal_215_0_fbe60a3f",
  "calibration_scope": "entity",
  "config_hash": "9aafb11427c7cc52",
  "resolved_config": { "risk_threshold": {"value": 0.15, "source": "entity_config"} },
  "warnings": ["..."],
  "caveat": "Advisory recommendation. No execution is implied ...",

  "persisted": true,
  "idempotent_replay": false,
  "artifact_path": "logs/scores/215_0/2026-10-11/req-1955401830.csv"
}
```

### The fields that decide what the backend does

| Field | Use it for |
|---|---|
| `new_predictive_alert` | **Page staff on this.** True only at the moment a new alert fires. |
| `alert_state` | Display only. It stays elevated for the whole episode — paging on it pages on every record. |
| `reactive_critical` | The bay is **already** below RED. Word the message differently. |
| `action.should_replenish` | Whether to dispatch a trip. |
| `action.quantity` / `urgency` | How many carts, how soon. |
| `action.expected_completion_time` | The deadline to show staff. |
| `persisted` | `true` = this decision was written. |
| `idempotent_replay` | `true` = this timestamp was already scored; the original decision was returned. |
| `warnings` | Must be surfaced, not swallowed. |

### Response headers

```
X-Request-ID: req-1955401830
X-Response-Time-ms: 78.16
```

---

## 5. Status codes — real captures

### `200` success

Measured: **75–125 ms** for a 20-record window.

### `422` too few records

```json
{"detail": "5 usable records supplied; at least 12 consecutive records are required (hard floor 10, recommended 20)."}
```

### `422` body disagrees with the path

```json
{"detail": "body kdvr 300 does not match path 215"}
```

### `422` broken window

Raised when the **tail** of the window is too short after a sequence break, so
a feature cannot be computed:

```json
{"detail": "feature window incomplete (93 unavailable, e.g. ['cart_lag_2', 'cart_lag_3', 'cart_lag_6']). Records must be consecutive with no gap above 15 minutes."}
```

A gap in the *middle* is tolerated as long as the final contiguous run is long
enough to fill the 9-record windows — verified: 12 contiguous records after a
break still score `200`. Do not rely on this; send one clean run.

### `404` entity not registered

```json
{"detail": "entity '999:9' is not registered. Register it before scoring; scoring with default thresholds would be meaningless."}
```

### `409` no promoted model

```json
{"detail": {"code": "MODEL_NOT_READY",
            "message": "no promoted model is bound to entity '215:0'...",
            "readiness": {"candidates": 1, "blocking_reasons": []}}}
```

### `503` entity disabled · `401` bad token · `500` unhandled

```json
{"detail": "internal error", "request_id": "req-1955401830"}
```

### How the backend should treat each

| Code | Retry? | Action |
|---|---|---|
| `200` | — | Use the decision. |
| `401` | No | Fix the token. |
| `404` | No | Register the entity first. |
| `409` | **Not in a tight loop** | Nothing is promoted. Alert operations. |
| `422` | No | Data problem — fix the query that built the window. |
| `503` | Yes, with backoff | Entity disabled or temporarily unscoreable. |
| `5xx` | Yes, with backoff | Server fault. |
| timeout | **Yes — safe** | See §7. |

---

## 6. Configuration precedence

```
params (this call)  >  entity_config  >  site_config  >  SYSTEM_DEFAULTS
```

Every parameter comes back in `resolved_config` with the source that won:

```json
"target_stock":            {"value": 28,     "source": "entity_config"},
"risk_threshold":          {"value": 0.1853, "source": "model_kmeans"},
"critical_risk_threshold": {"value": 0.5707, "source": "model_kmeans"},
"red_model":               {"value": 15.0,   "source": "model_registry"}
```

Sources: `request_override`, `entity_config`, `site_config`, `model_kmeans`,
`model_registry`, `derived_from:<source>`, `system_default`.

**The two alert bands are not configuration.** They are clustered from the
entity's own calibrated risk at training time (K-Means, K=3) and frozen into
the model — writing them to `entity_config` has no effect. A single call may
still override them for diagnostics or an audited exception; the override is
never persisted and is announced in `warnings`.

`band_warning` (= `risk_threshold`) also gates replenishment, so alerting and
ordering read the same measured number.

---

## 7. Idempotency — why retries are safe

The key is `(entity_id, decision_time)`, where
`decision_time = next_DVRDate of the last record in the window`.

If the backend times out after the server already committed, retrying returns
the **original** decision:

```json
{"persisted": false, "idempotent_replay": true, "request_id": "req-…"}
```

No second order is placed, no second artifact is written, no second event is
published. **Retry freely.**

The corollary: re-scoring the same window after changing configuration will
**not** produce a new decision. The earlier rows must be deleted first — which
is what `scripts/run_experiment.py` does.

---

## 8. The background path

After responding, the server enqueues the decision into an outbox table that
lives in the **same database** as `scoring_results`. A worker then delivers it:

```
scoring_results  ──commit #1──┐
result_outbox    ──commit #2──┤   (see the caveat below)
                              │
        └─ worker ─▶ POST :8200/v1/results/scoring
                        201 new · 200 identical replay · 409 payload conflict
```

> **Caveat — not yet a true transactional outbox.** `registry/database.py`
> commits once per statement, so the decision and the outbox row are written in
> two separate transactions. A crash between them leaves a decision that is
> never published, and no reconciliation job exists to detect it. Before
> production either wrap both writes in one transaction, or add a job that
> finds `scoring_results` rows with no matching `result_outbox` entry.

- Exponential backoff with jitter; `Retry-After` honoured.
- Non-retryable `4xx` → `dead_letter`, visible at `GET /v1/integrations/outbox`.
- In-flight rows are recovered after a restart.
- **If the receiver is down, scoring still returns `200`.** The decision is
  committed locally and delivered later.

The backend reads archived results from the receiver, never from the ML
server's database:

```
GET :8200/v1/results/scoring?entity_id=215:0&since=…&limit=…
```

---

## 9. Observability

| Endpoint | Shows |
|---|---|
| `GET /health` | status, champion model, entity counts |
| `GET /v1/logs/requests` | every call with latency and outcome |
| `GET /v1/logs/scores` | recent decisions |
| `GET /v1/scoring/artifacts/{request_id}` | the immutable CSV for one decision |
| `GET /v1/integrations/outbox` | pending / sent / dead-letter counts |
| `GET /v1/sites/{site}/entities/{kdvr}/{chan}/model` | active, candidates, history |

---

## 10. What the backend must get right

1. **Send consecutive records of one entity.** Mixing entities is rejected.
2. **Send at least 12, ideally 20.** Below the floor the window cannot be built.
3. **Page on `new_predictive_alert`, not on `alert_state`.**
4. **Retry on timeout** — it is safe and will not duplicate an order.
5. **Treat `409` differently from `5xx`.** It means a human must promote a
   model, not that the server is unwell.
6. **Surface `warnings`.** They carry shadow-mode, calibration-scope and
   threshold-mismatch notices.
7. **Read archived results from the receiver**, not from the ML server's tables.

---

# Part B — Training and retraining

Scoring is automatic and continuous. Training is **deliberate**: a human
uploads data, selects entities, reviews the evidence and promotes. Nothing is
ever promoted automatically.

---

## 11. The lifecycle

```
POST /v1/datasets/upload          1. upload history + per-bay config
   └── discovers the (KDVR, ChanNo) pairs IN THAT FILE ONLY
POST /v1/training/jobs            2. train the entities YOU select
   └── automatic calibration      (never a manual call)
   └── three backtest reports     (mandatory, cannot be disabled)
          └── candidate — NEVER auto-promoted
GET  .../calibration, .../backtests   3. review the evidence
POST .../models/{model_id}/promote    4. promote, manually
POST .../models/rollback              5. undo if it misbehaves
```

Retraining is the same flow. The new model becomes a **candidate** alongside
the active one; scoring keeps using the old model until you promote.

---

## 12. Step 1 — Upload

```
POST /v1/datasets/upload
Content-Type: multipart/form-data
  file   = <CSV, tab-separated, utf-8-sig>
  config = <JSON>
```

```json
{
  "name": "oct_batch_01",
  "timezone": "Asia/Ho_Chi_Minh",
  "entity_defaults": {"max_sequence_gap_min": 15, "cooldown_min": 30,
                      "lead_time_min": 10},
  "entity_overrides": {
    "163:215:0": {"bay_capacity": 60, "red_threshold": 15,
                  "amber_threshold": 20, "target_stock": 28}
  },
  "auto_start_training": false
}
```

`entity_overrides` is written into `entity_config` — this is how a bay is
configured for the first time. `entity_defaults` applies to every pair found in
the file.

### Response

```json
{
  "upload_id": "upl_20261014_8ba8da5e",
  "status": "registered",
  "n_rows": 2543, "n_rows_usable": 2542, "n_entities": 1,
  "duplicate_of": null,
  "entities": [...],
  "quality": {...},
  "warnings": ["dropped 1 row(s): unparseable timestamps"]
}
```

### Upload is idempotent — verified

Posting the **same file with the same config** returns the same `upload_id`
with `duplicate_of` set. Nothing is re-stored and nothing is retrained:

```
1st call →  upload_id: upl_20261014_8ba8da5e   duplicate_of: null
2nd call →  upload_id: upl_20261014_8ba8da5e   duplicate_of: upl_20261014_8ba8da5e
```

The fingerprint covers the file checksum **and** the config. Changing `name`
alone makes it a different upload.

### Trainability is decided here

```
GET /v1/datasets/{upload_id}/entities
```

```
215:0    n=2542  days=14  trainable=True   []
215:1    n=2542  days=14  trainable=True   []
300:2    n=2542  days=14  trainable=False  ['bay_capacity is not configured
                                            (physical value, never invented)']
```

Per-entity quality is reported too: `n_days`, `shortage_events`,
`median_interval_min`, `n_blocks`, `cart_min/max/median`.

A bay with no `bay_capacity` is **untrainable by design** — the platform will
not guess a physical value.

---

## 13. Step 2 — Train

```
POST /v1/training/jobs
{"upload_id": "upl_...", "entity_ids": ["163:215:0"], "strategy": "per_entity"}
```

Returns immediately with a `job_id`; training runs asynchronously (~40 s per
entity on 2.5k records).

```
POST /v1/training/jobs  strategy="pooled"
 → 501 {"detail": "strategy='pooled' must be requested explicitly and is not
        implemented; unrelated bays are never silently pooled"}
```

### What happens inside the job, per entity

```
 1  resolve + FREEZE the configuration    -> config_snapshot, config_hash
 2  prepare frame                          entity_id, decision_time, block_id
 3  build 127 features                     with THIS entity's red/amber
 4  build targets                          y_cart_3/6, y_min_3/6 (by RECORD)
 5  chronological split                    train / calibration / holdout + purge
 6  feature selection                      on the training rows only
 7  fit the value clipper                  training rows only, applied to all
 8  train 3 candidates per target          Persistence, Ridge, LightGBM
 9  select per target                      composite of MAE, near-shortage MAE,
                                           and shortage recall for min targets
10  write the bundle + metadata.json
11  automatic calibration                  split-conformal against frozen RED
12  alert bands                            K-Means K=3 on the calibration risk
13  three backtests                        forecast / replay / robustness
14  bind as CANDIDATE                      never active
15  evaluate promotion gates
```

The configuration is frozen **before** training, so the job result always
states the thresholds the model was actually built with.

### Polling

```
GET /v1/training/jobs/{job_id}
```

```json
{
  "job_id": "train_20261014030730_1e3219",
  "status": "succeeded",
  "upload_id": "upl_...", "entity_ids": ["163:215:0"], "strategy": "per_entity",
  "progress": {...}, "started_at": "...", "finished_at": "...",
  "results": {
    "163:215:0": {
      "status": "candidate_ready",
      "model_id": "mdl_215_0_20261013165052_053efa",
      "promotable": true, "blocking_reasons": [], "promoted": false,
      "n_features": 127,
      "config_hash": "9aafb11427c7cc52",
      "frozen_thresholds": {"red_model": 15.0, "amber_model": 20.0},
      "effective_config": {...},
      "selected_models": {"y_cart_3": "LightGBM", "y_min_6": "Ridge", ...},
      "split_windows": {"train": {...}, "calibration": {...}, "holdout": {...}},
      "metrics": {...},
      "calibration": {"status": "fitted", "scope": "entity", "metrics": {...}},
      "alert_bands": {"status": "fitted", "source": "model_kmeans",
                      "bands": {"warning": 0.1853, "critical": 0.5707},
                      "clusters": [...]},
      "backtests": {"forecast_accuracy": {"status": "completed"},
                    "policy_replay": {"status": "completed"},
                    "closed_loop_robustness": {"status": "completed"}},
      "warnings": [...]
    }
  }
}
```

Job `status`: `queued` → `running` → `succeeded` | `failed`.

### Per-entity failure does not stop the job

A bad entity fails on its own and the rest continue:

```json
{"status": "invalid_config",
 "error": "INVALID_ENTITY_CONFIG for '300:2': 'bay_capacity' is not configured
           for this entity (resolved to the system default); physical values
           are never invented",
 "model_id": null}
```

| `status` | Meaning |
|---|---|
| `candidate_ready` | Trained, calibrated, backtested — awaiting promotion |
| `invalid_config` | Configuration incomplete or contradictory |
| `insufficient_training_data` | Not enough usable rows after warm-up and purging |
| `failed` | Unexpected error; traceback included |

---

## 14. Step 3 — Review the evidence

```
GET /v1/training/jobs/{job_id}/calibration
GET /v1/training/jobs/{job_id}/backtests
GET /v1/training/jobs/{job_id}/backtests/forecast_accuracy
GET /v1/training/jobs/{job_id}/backtests/policy_replay
GET /v1/training/jobs/{job_id}/backtests/closed_loop_robustness
```

### Alert bands

```
kmeans_k3  fitted  seed=0
NORMAL    380  75.7%   risk 0.0000 .. 0.1813   centre 0.0200
WARNING    70  13.9%   risk 0.1892 .. 0.5598   centre 0.3478
CRITICAL   52  10.4%   risk 0.5817 .. 0.9741   centre 0.7787

=>  NORMAL < 0.1853 <= WARNING < 0.5707 <= CRITICAL
```

The band edge sits in the gap the data left between clusters, not at a
centroid. Guards refuse to band a distribution that cannot support three
groups — below 100 records, below 10 records at `risk >= 0.10`, fewer than 3
distinct values, or a resulting WARNING band under 0.02. A refusal serves the
documented default and says so; it never invents a number.

### Calibration

```
split_conformal  fitted  scope=entity  calibrated against RED = 15.0
n=502  events=74  Brier 0.0784 (uncalibrated 0.1135)  ECE 0.0327  coverage 89.8%
```

Below the minimum samples or positives the status is `insufficient_data` and a
global fallback may be used — flagged loudly as *not calibrated to this
entity*. Probabilities are never fabricated.

### The three backtests — read them differently

| Report | What it is | How to read it |
|---|---|---|
| `forecast_accuracy` | Out-of-sample MAE/RMSE/bias per horizon vs a persistence baseline, plus shortage PR-AUC and interval coverage | **MEASURED.** The only one independent of the simulation assumptions |
| `policy_replay` | Replay on the holdout with the fitted calibrator: alerts, trips, carts, order ledger | **SIMULATION ESTIMATE**, conditional on assumptions |
| `closed_loop_robustness` | best-case / audited-default / conservative scenarios | **SCENARIO ENVELOPE** — the conservative row is as valid as the optimistic one |

Real numbers from `mdl_215_0_20261013165052_053efa` (215:0, risk_threshold 0.15):

```
forecast_accuracy : 17.0% better than persistence on cart_at_20min
                    shortage PR-AUC 0.800, ROC-AUC 0.937, Brier 0.083
policy_replay     : 9 trips, 136 carts, severity reduction 96.2%
robustness        : severity reduction spans 3.0% .. 96.2% across 4 scenarios
```

That 3–96% spread is the honest summary: the benefit is **not robust** to the
assumptions. Quote `forecast_accuracy` as measured; quote the rest as estimates.

A report that cannot run is stored as `insufficient_data` or `failed` with the
exact reason — and that **blocks promotion**.

---

## 15. Step 4 — Promote (always manual)

```
POST /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/models/{modelId}/promote
{"promoted_by": "your-name"}
```

### Gates

1. Calibration status is `fitted`
2. All three backtests are `completed`
3. The model was trained **for this entity**

```json
{"entity_id": "215:0", "model_status": "ready",
 "active": {"model_id": "mdl_...", "promotable": true,
            "blocking_reasons": [], "promoted_by": "your-name",
            "promoted_at": "..."},
 "candidates": [...], "history": [...]}
```

### Refusals — real captures

```
409  {"detail": {"message": "...", "blocking_reasons": [...],
                 "hint": "fix the gate or record an audited exception via
                          audited_exception_reason"}}

404  {"detail": "model 'mdl_215_0_...' is not a candidate for entity '215:1'"}
```

A model trained for one bay can never be promoted onto another.

`audited_exception_reason` records an explicit, attributed override when a gate
must be bypassed.

### Before promotion, scoring is refused

```
409 MODEL_NOT_READY — "no promoted model is bound to entity '215:0'"
```

Training a new model does **not** change what is served. The previous model
keeps scoring until promotion.

---

## 16. Step 5 — Rollback

```
POST /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/models/rollback
{"promoted_by": "your-name"}
```

Restores the previous binding. The current model becomes `retired`, the
previous one `active`. Scoring continues uninterrupted.

To jump to a specific earlier model, promote it by `model_id` instead.

---

## 17. When to retrain

| Trigger | Why |
|---|---|
| You want the bands re-measured | They are frozen per model; new history means a new risk distribution |
| `red_threshold` or `amber_threshold` changed | They build 17 of the 127 features. Changing them via the API only moves the evaluation threshold; the server warns that features still use the trained value |
| `feature_windows` or horizons changed | They define the feature vector and the targets |
| Bay physically changed | Capacity, layout or traffic pattern |
| Accuracy drifts | Compare `forecast_accuracy` across models on a common holdout |
| New history accumulated | More shortage events give a better calibrator |

**No retraining needed** for `cooldown_min`, `target_stock`, `lead_time_min`,
trip sizes or `reserve_per_day` — change them with `PUT /v1/entities` and they
apply on the next call.

The alert bands are a different matter: they are **measured**, so they cannot be
changed by configuration at all. Retrain to re-measure them, or override them
for a single call when you need an audited exception.

### One command, end to end

```bash
python scripts/run_experiment.py \
    --risk-threshold 0.15 --critical-risk-threshold 0.90 \
    --out outputs/risk015_crit090.csv
```

Uploads, trains, promotes, scores every record and exports a CSV with the exact
columns of `scoring_results`, plus a `.config.json` sidecar recording the
parameters, `model_id` and `config_hash`.

---

## 18. Training endpoints at a glance

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/datasets/upload` | Upload history + per-bay config |
| `GET` | `/v1/datasets` · `/v1/datasets/{id}` | List / inspect uploads |
| `GET` | `/v1/datasets/{id}/entities` | Discovered pairs + trainability |
| `POST` | `/v1/training/jobs` | Start a job for selected entities |
| `GET` | `/v1/training/jobs` · `/{job_id}` | Poll status and results |
| `GET` | `/v1/training/jobs/{id}/calibration` | Calibration diagnostics |
| `GET` | `/v1/training/jobs/{id}/backtests[/{type}]` | The three reports |
| `GET` | `/v1/sites/{site}/entities/{kdvr}/{chan}/model` | Active, candidates, history |
| `POST` | `/v1/sites/{site}/entities/{kdvr}/{chan}/models/{id}/promote` | Promote (manual) |
| `POST` | `/v1/sites/{site}/entities/{kdvr}/{chan}/models/rollback` | Restore the previous |

The site-less forms of all four entity routes return **410 Gone** with the
replacement URL.
