# Cart Counting — Forecasting, Shortage Warning and Replenishment

Forecasts how many carts will be available in a bay over the next 20–40
minutes, converts that into a **calibrated shortage probability**, and turns
the probability into an alert and a replenishment recommendation.

Input is cart-counting telemetry from one camera stream watching one bay:

```
CountID  DVRDate  KDVR  ChanNo  CatID  AreaID  Count  CNN_Index
siteKey  next_DVRDate  Count_IN  Count_OUT
```

`Count` is the carts observed in the bay at the end of the interval;
`Count_IN` / `Count_OUT` are customers entering and leaving during it.

---

## Repository layout

Three modules, each independently runnable and tested.

```
Cartcounting_model_forecast/
├── cart_forecasting/   the audited modelling pipeline        49 tests
├── api/                single-entity scoring API             18 tests
└── cart_platform/      multi-entity platform + receiver     193 tests
```

| Module | What it is | When to use it |
|---|---|---|
| **`cart_forecasting/`** | The offline pipeline: features, targets, model selection, conformal calibration, backtests, audit report. Source of truth for the maths. | Research, re-auditing, regenerating reports |
| **`api/`** | A FastAPI service that serves one validated model over HTTP. | The simpler single-bay deployment |
| **`cart_platform/`** | Per-entity registry, on-demand upload/training, promotion gates, scoring API, and a standalone result receiver. | The multi-bay, multi-site deployment |

`cart_platform` imports the pipeline code from `cart_forecasting/src` rather
than copying it, so the online decision and the offline backtest can never
drift apart.

**Entity grain:** `entity_id = "siteKey:KDVR:ChanNo"`, e.g. `163:215:0` — one
camera stream, one bay. The site is part of the key because DVR ids are not
guaranteed unique across stores. Routes are site-scoped to match:

```
POST /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/score
GET  /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/model
POST /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/models/{modelId}/promote
POST /v1/sites/{siteKey}/entities/{kdvr}/{chanNo}/models/rollback
```

The site-less forms return **410 Gone** with the replacement URL.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r cart_platform/requirements.txt
```

### Offline pipeline

```bash
cd cart_forecasting
python main.py            # models, backtests, figures, report -> outputs/
python audit_main.py      # independent audit  -> simulation_audit/
```

### Multi-entity platform

```bash
cd cart_platform
bash server/run_server.sh        # terminal A — :8100  (owns data/platform.db)
bash receiver/run_server.sh      # terminal B — :8200  (owns data/results_receiver.db)
curl http://127.0.0.1:8100/health
```

The ML server stays functional if the receiver is down: results are committed
locally and a transactional outbox delivers them when it returns.

### One parameterised experiment, end to end

```bash
cd cart_platform
python scripts/run_experiment.py --entity 163:215:0 \
    --out outputs/run.csv
```

Uploads → trains → promotes → scores every record → writes a CSV with the exact
columns of the `scoring_results` table, plus a `.config.json` sidecar recording
the parameters, `model_id` and `config_hash`.

---

## The pipeline

```
telemetry
  └── prepare            entity_id, decision_time, contiguous block_id
        └── features     127 columns: lags, rolling stats, slopes, time-to-breach
              └── targets  y_cart_3/6 = Count(t+h);  y_min_3/6 = min(Count(t+1..t+h))
                    └── chronological split  train / calibration / holdout (+ purge)
                          └── model selection per target
                                └── conformal calibration  -> P(min < RED)
                                      └── alert + replenishment policy
```

**Horizons are counted in records, not minutes.** `y_min_6` is the minimum of
the next 6 records; 3 records ≈ 20 min and 6 ≈ 40 min are naming conventions.
Real intervals range from 3.3 to 36 minutes, so a `horizon_reliable` flag marks
rows whose realised span deviates more than 50% from nominal.

`decision_time` is the record's **`next_DVRDate`** — the moment its counting
interval closed. A decision cannot exist before its interval ends.

Four models compete per target (Persistence, Ridge, LightGBM) and are selected
on the calibration split by a composite of overall MAE, near-shortage MAE and —
for minimum targets — shortage recall.

**Measured on the 215:0 holdout:** 17.0% better than persistence on the 20-minute
cart forecast; shortage classification PR-AUC 0.80, ROC-AUC 0.937, Brier 0.083.

---

## Alerts

The label depends on **one** number, the calibrated probability `risk`:

| Label | Condition |
|---|---|
| `CRITICAL` | `risk >= band_critical` |
| `WARNING` | `band_warning <= risk < band_critical` |
| `NORMAL` | `risk < band_warning` |

**The two bands are measured, not configured.** After calibration the trainer
runs K-Means (K=3) on this entity's own calibrated risk over the calibration
split and takes the band edges from where the clusters actually separate. The
result is frozen into the model; `resolved_config` reports the source as
`model_kmeans`.

Measured on 215:0 (`red=15`), from 502 calibration records:

```
NORMAL    380  75.7%   risk 0.0000 .. 0.1813   centre 0.0200
WARNING    70  13.9%   risk 0.1892 .. 0.5598   centre 0.3478
CRITICAL   52  10.4%   risk 0.5817 .. 0.9741   centre 0.7787

=>  NORMAL < 0.1853 <= WARNING < 0.5707 <= CRITICAL
```

On the untouched holdout those bands give:

| Band | Recall | Precision |
|---|---|---|
| WARNING >= 0.1853 | 77.9% | 52.2% |
| CRITICAL >= 0.5707 | 51.9% | 74.1% |
| *(old default 0.55)* | *53.2%* | *71.9%* |

The clustering finds an operating point close to one that previously took
several hand-tuned attempts to locate.

`band_warning` **also gates replenishment**, so alerting and ordering share one
measured number instead of two independent constants.

A distribution that cannot support three bands is refused rather than banded
badly — see "Degenerate clustering" below.

Three things that surprise people:

- **The current cart count does not affect the label.** A bay can hold 3 carts
  and still be `NORMAL`. "Already below RED" is the separate `reactive_critical`
  column.
- **Hysteresis makes the label sticky.** Once elevated it returns to `NORMAL`
  only when `risk < band_warning - 0.10`.
- **`cooldown_min` does not touch `alert_state`.** It gates
  `new_predictive_alert`. `alert_state` is the ongoing state;
  `new_predictive_alert` is the moment a new alert fires.

The engine is stateful per entity and persisted; `POST /v1/sessions/reset`
clears it. Re-scoring the same timestamps without a reset returns the stored
state.

### Degenerate clustering

K-Means will happily band noise. Measured on a short window with no high-risk
records it returns `0.006 / 0.042`, which would mark almost every record
elevated; on an all-zero distribution it raises outright. The trainer therefore
refuses to band unless all of these hold:

| Guard | Default |
|---|---|
| calibration records | >= 100 |
| records with `risk >= 0.10` | >= 10 |
| distinct risk values | >= 3 |
| resulting `band_warning` | >= 0.02 |

When a guard trips the status is `insufficient_data`, the documented default
`0.55 / 0.80` is served, and the source reads `system_default` so a fallback is
never mistaken for a measured value.

### Calibration sets the probability, the clustering sets the bands

Calibration fits the conformal mapping **prediction → probability**:

```
split_conformal   fitted   calibrated against RED = 15.0
n=502  events=74  Brier 0.0784 (uncalibrated 0.1135)  ECE 0.0327  coverage 89.8%
```

It answers *"risk = 0.73"*; the clustering answers *"what do we call 0.73"*.
Both are derived from the same calibration split, and both are frozen into the
model together.

The one link: the calibrator is fitted against a specific `RED`. Change
`red_threshold` afterwards and the probabilities are no longer calibrated for
that definition — the server says so in `warnings`.

---

## Parameters

### Changeable at any time — no retraining

| Parameter | Effect |
|---|---|
| `cooldown_min` | paces ordering and `new_predictive_alert` |
| `target_stock`, `lead_time_min`, `min/max_carts_per_trip`, `reserve_per_day`, `quantity_rule` | sizing and pacing |

```bash
curl -X PUT http://127.0.0.1:8100/v1/entities \
  -H 'Content-Type: application/json' \
  -d '{"entity_id":"163:215:0","kdvr":215,"chan_no":0,"site_key":163,
       "target_stock":30}'
```

Resolution order, reported per key in `resolved_config[k].source`:

```
params  >  entity_config  >  site_config  >  SYSTEM_DEFAULTS
```

### Measured, not configurable

`risk_threshold` and `critical_risk_threshold` are **not** in that chain. They
come off the model's K-Means clustering (`source: model_kmeans`) and cannot be
written to `entity_config` or supplied at upload time — doing so has no effect.

A single call may still override them for diagnostics or an audited exception:

```json
{"records": [...], "params": {"risk_threshold": 0.30}}
```

The override applies to that call only, is never persisted, and is announced in
`warnings` with `source: request_override`, so a bypass is always visible in the
decision record.

### Requires retraining

| Parameter | Why |
|---|---|
| `red_threshold`, `amber_threshold` | they build 17 of the 127 features — and they define what "shortage" means, so the risk distribution and therefore the bands move with them |
| `feature_windows`, `primary_horizon`, `planning_horizon` | they define the feature vector and the targets |

Changing them through the API is accepted but affects only risk, alerts and
quantity; the features keep their trained values and the server warns.

### One number drives both alerting and ordering

`band_warning` opens the WARNING band **and** gates replenishment:

```python
# plan_replenishment()
if risk < policy.risk_threshold:   # = band_warning
    return None                    # no order at all
```

That is deliberate: a bay that is worth warning about is a bay worth a trip.
It also means the measured band cannot be tuned for alert noise without moving
the ordering gate with it.

**Quantity is not derived from the label.** It comes from the stock gap:

```
need = target_stock - pred_min_6 - pending
q    = ceil(max(0, need))   clamped by min/max per trip, bay headroom, reserve
```

and `urgency` from the forecast path versus lead time:

| Urgency | Condition |
|---|---|
| `URGENT` | `lower_min_6 < RED` **and** breach within the lead time |
| `HIGH` | `pred_min_6 < RED` |
| `MEDIUM` | `pred_min_6 < AMBER` |
| `LOW` | otherwise |

## On-demand workflow (`cart_platform`)

```
upload dataset + config
   └── discovers (KDVR, ChanNo) pairs IN THAT FILE ONLY
          └── train the entities YOU select
                 └── automatic calibration   (never a manual call)
                 └── three backtest reports
                        └── candidate — never auto-promoted
                               └── manual promotion
                                      └── POST /v1/entities/{kdvr}/{chan_no}/score
                                             └── one immutable CSV per decision
                                             └── published to the receiver
```

An entity without `bay_capacity` is reported untrainable — physical values are
never invented. Scoring an unpromoted entity returns **409 MODEL_NOT_READY**.

| Backtest | What it is | How to read it |
|---|---|---|
| `forecast_accuracy` | Out-of-sample MAE/RMSE/bias per horizon vs persistence, plus shortage PR-AUC and interval coverage | **Measured** performance |
| `policy_replay` | Replay on the holdout with the fitted calibrator: alerts, trips, carts, order ledger | **Simulation estimate**, conditional on assumptions |
| `closed_loop_robustness` | Best-case / audited-default / conservative scenarios | **Scenario envelope** — the conservative row is as valid as the optimistic one |

A report that cannot run is stored as `insufficient_data` or `failed` with the
reason, and that blocks promotion. Metrics are never fabricated.

Scoring is **idempotent on `(entity_id, decision_time)`**: a retry returns the
original artifact and places no second order. Re-scoring the same window after
changing configuration therefore requires deleting the earlier decisions first
— which is what `scripts/run_experiment.py` does.

---

## Known limitation: simulated carts

`simulated_cart = current_cart + surplus`, where `surplus` is replenishment the
simulator **assumed was delivered**. `order_book.deliver_due()` marks an order
`COMPLETED` once the lead time elapses — no human confirmation is required, and
`POST /v1/orders/{id}/execute` is not a precondition.

With `consumption_model = "unmet_demand"` that surplus is drawn down only when
real demand exceeds real stock, which is rare, so it accumulates:

```
215:0 at 0.15 / 0.90 : surplus median 8.0, present in 86% of decisions
                       actual records below RED     135
                       simulated records below RED   35
```

`service.py` used to add this surplus to the forecast before computing risk,
which suppressed risk to ~0 while the bay was actually empty. Risk,
`alert_state`, quantity sizing and capacity headroom now use the **camera
count and the model-only forecast**; carts still in transit are netted via
`pending_quantity()`. A camera count at or below RED forces
`alert_state = CRITICAL` (`reactive_critical = true`).

`simulated_cart` is still reported but is now informational only. Treat it as a
**simulation under an assumption**, not as measured stock.

Gating delivery on `POST /v1/orders/{id}/execute` instead of on elapsed lead
time would make `simulated_cart` meaningful again.

---

## Backtesting an exported run

`scripts/run_experiment.py` writes the full `scoring_results` schema — 20
columns on the ML server, the same 20 plus `event_id` on the receiver:

```
id, request_id, entity_id, kdvr, chan_no, site_key, decision_time,
current_cart, simulated_cart,
pred_cart_20min, pred_cart_40min, pred_min_20min, pred_min_40min,
lower_bound_min_40min,
shortage_risk, alert_state, quantity, model_id, calibration_id, scored_at
```

The table keeps what production needs plus enough to monitor. Everything else
stays in the table that owns it instead of being copied onto every decision:

| Not stored here | Still available in |
|---|---|
| `should_replenish`, `urgency`, `reason`, `order_id`, `deliver_by` | the API response, and `replenishment_orders` |
| `resolved_config`, `config_hash`, `frozen_config_json` | `config_snapshots`, joined on `model_id` |
| `warnings` | the API response, and `received_events.payload_json` |
| `new_predictive_alert`, `reactive_critical`, `pending_carts` | the API response |
| `latency_ms` | `api_requests` |
| `artifact_path` | `scoring_artifacts` and the CSV on disk |
| `upload_id`, `job_id` | `model_registry`, joined on `model_id` |

`quantity > 0` is equivalent to the old `should_replenish`. The receiver's
projection is flattened from the event payload rather than keeping a second
copy of it in a `result_json` column.

Labels can be re-derived offline from the raw `shortage_risk`:

```python
import pandas as pd, numpy as np

d = pd.read_csv("outputs/risk015_crit090.csv", parse_dates=["decision_time"])
d = d.sort_values("decision_time").reset_index(drop=True)

# the bands the model measured are in the sidecar written next to the CSV
import json
bands = json.load(open("outputs/run.config.json"))["alert_bands"]["bands"]
d["my_label"] = np.select(
    [d.shortage_risk >= bands["critical"], d.shortage_risk >= bands["warning"]],
    ["CRITICAL", "WARNING"], default="NORMAL")
```

Two caveats when comparing against `alert_state`:

- `alert_state` carries hysteresis, so it stays `WARNING` slightly longer than a
  pure threshold rule. On the run above: 133 vs 106 `WARNING`.
- Re-labelling with a different WARNING band cannot reproduce a different
  **run**: that band also gates ordering, so changing it changes which trips
  were placed and therefore the simulated trajectory that produced
  `shortage_risk`. Only the CRITICAL band can be explored purely offline.

Scoring against reality — on the **record** grid, since intervals are uneven:

```python
tel = pd.read_csv("cart_platform/data/real_data.csv", sep="\t", encoding="utf-8-sig")
tel["t"] = pd.to_datetime(tel["next_DVRDate"])
tel = tel[["t", "Count"]].dropna().sort_values("t")
d = d.merge(tel, left_on="decision_time", right_on="t", how="left")

a, t = tel.Count.to_numpy(float), tel.t.to_numpy()
d["actual_min_6"] = [
    a[i:i + 6].min() if (i := np.searchsorted(t, x, "right")) < len(a) else np.nan
    for x in d.decision_time.to_numpy()
]

scope = d[d.Count >= 15]                 # exclude records already in shortage
must_fire = scope.actual_min_6 < 15
fired = scope.alert_state != "NORMAL"
recall = (fired & must_fire).sum() / must_fire.sum()
precision = (fired & must_fire).sum() / fired.sum()
```

Excluding records already below RED matters: those alerts are reactive, not
predictive, and counting them inflates the score.

---

## Tests

```bash
cd cart_forecasting && python -m pytest tests -q     #  49
cd api              && python -m pytest tests -q     #  18
cd cart_platform    && python -m pytest tests -q     # 193
```

The platform suite covers the registry and three-level resolution, on-demand
upload/training/backtests, the result receiver contract and idempotency, the
outbox with retry and dead-lettering, per-entity configuration isolation, the
independent WARNING/CRITICAL bands, and the disk and database retention job.

**P0 guarantee:** `block_id` is keyed by entity, so no rolling window, lag or
target can mix two streams — verified by a dedicated unit test and by a
regression gate reproducing the pre-refactor single-site results byte-for-byte
across 8 output files.

---

## Databases

```bash
sqlite3 cart_platform/data/platform.db \
  "SELECT entity_id, model_id, role FROM entity_model_bindings;"
sqlite3 cart_platform/data/platform.db \
  "SELECT entity_id, decision_time, shortage_risk, alert_state, should_replenish
     FROM scoring_results ORDER BY decision_time DESC LIMIT 5;"
sqlite3 cart_platform/data/results_receiver.db \
  "SELECT event_type, COUNT(*) FROM received_events GROUP BY event_type;"
```

`received_events` on the receiver is the immutable log; the other tables are
projections built from it. Deleting a projection directly cannot be repaired by
re-sending, because the receiver answers a known `event_id` with `REPLAY` and
skips the projection — rebuild it from `received_events` instead.

### Starting over

`scripts/reset_databases.py` empties both databases and keeps their structure.
Destructive, so it is a dry-run until `--yes`.

```bash
cd cart_platform
python scripts/reset_databases.py                      # show what would go
python scripts/reset_databases.py --yes                # reset both
python scripts/reset_databases.py --db receiver --yes  # just one
python scripts/reset_databases.py --yes --with-artifacts
```

Structure is rebuilt by each module's own `init_db()` — its `schema.sql`
followed by its additive migrations — so the result is exactly the schema the
code expects rather than a copy that can drift.

| Mode | Behaviour |
|---|---|
| `recreate` (default) | delete the file and rebuild it from `schema.sql` |
| `truncate` | keep the file, delete every row, restart AUTOINCREMENT ids |

In `recreate` mode the table and column inventory is captured first and
compared afterwards; anything that existed before and is not recreated aborts
the run, so structure added by hand is never dropped silently. A timestamped
`.bak` snapshot is taken first unless `--no-backup` — it is written with
SQLite's online backup API, so it includes the WAL that a file copy misses.

The script refuses to run while another process holds a write lock: a reset
under a live server leaves that process writing to an unlinked inode, where the
work is invisible in the new file. Stop the servers first, or pass `--force`.

`--with-artifacts` also empties `models/`, `logs/` and `data/uploads/`. Without
it those files survive a reset registry that can no longer explain them.

---

## Retention — reclaiming disk and database space

A `retired` model can never be scored again — the resolver binds only a
`champion`, and a rollback promotes a `candidate` — so its bundle, training
logs, calibration and backtest artifacts and its dependent rows are dead
weight. A fortnightly job deletes them, along with operational records that
have outlived their usefulness.

```bash
cd cart_platform
python scripts/purge_retired_artifacts.py                 # dry-run: show the plan
python scripts/purge_retired_artifacts.py --apply         # free the space
python scripts/purge_retired_artifacts.py --grace-days 30 # keep things longer
python scripts/purge_retired_artifacts.py --files-only    # artifacts, no rows
python scripts/purge_retired_artifacts.py --keep-decisions  # leave the ledgers
```

Everything is gated on one age cutoff, **one week** by default.

### Pass 1 — retired models

| What | Deleted when |
|---|---|
| `models/per_entity/<entity>/<model_id>/` | always |
| `logs/training/<job_id>/<entity>/` | no non-retired model and no unfinished job uses that job |
| `data/uploads/<upload_id>/` | every model trained from that upload is retired |
| rows in `scoring_artifacts`, `calibration_records`, `calibrator_registry`, `backtest_reports`, `backtest_runs`, `config_snapshots`, `entity_model_bindings` | always |
| rows in `training_jobs`, `dataset_uploads`, `upload_entities` | the job / upload is unshared |

### Pass 2 — decisions the receiver already has

`scoring_results` and `replenishment_orders` grow without bound: one row per
record per entity, forever. Both are forwarded to the result receiver, which is
their system of record, so once a decision is **acknowledged there** and older
than the cutoff the local copy is redundant and goes.

| What | Rule |
|---|---|
| `scoring_results` | `delivered_at` is set and `decision_time` older than the cutoff |
| `replenishment_orders` | same, and the order is not `PENDING` |

`delivered_at` is stamped on the decision by the outbox the moment the receiver
acknowledges the event — the outbox row that proves delivery is itself purged
later, so the proof is copied onto the decision. A decision whose event is
still `pending`, `sending` or `dead_letter` has **not** reached the other
server: it is the only copy that exists and it is kept, however old, and the
run reports how many were held back. An open (`PENDING`) replenishment order is
outstanding work on the floor and is never deleted either.

```bash
# how much is waiting on delivery
sqlite3 cart_platform/data/platform.db \
  "SELECT status, COUNT(*) FROM result_outbox GROUP BY status;"
```

If the receiver has been down, start it and let the outbox drain — the next run
clears everything it confirms. `--force-undelivered` overrides the gate and
destroys undelivered decisions; use it only when records reach the other server
by some route this platform cannot see.

### Pass 3 — operational logs, any model

| What | Rule |
|---|---|
| `logs/scores/<entity>/<date>/req-*.csv` + its `scoring_artifacts` row | older than the cutoff — every column is already a `scoring_results` row |
| `result_outbox` where `status = 'sent'` | the receiver has acknowledged it |
| `result_outbox` where `status = 'dead_letter'` and the model is retired | cannot be retried into anything meaningful |
| `api_requests` | HTTP traffic log |

Score logs age by `decision_time` — the same moment as `created_at` in
production, since a decision is written when it is made; the two diverge only
for a backfill or an experiment replay, where it is still the age of the
decision that matters.

### What is never deleted

* **anything belonging to a champion or candidate** — bundles, training
  artifacts, calibration, backtests, bindings and registry rows are protected
  by a live-model guard re-checked on every statement;
* **the `model_registry` row** of a retired model, so the audit trail of what
  was trained, promoted and retired survives — it is stamped with `purged_at` /
  `purged_bytes` instead, and every deleted path is recorded in
  `artifact_purges`;
* **a decision the receiver has not acknowledged**, and any order attached to
  it — that data exists nowhere else;
* **open (`PENDING`) replenishment orders**, at any age;
* **undelivered outbox work** — `pending` and `sending`, at any age.

Nothing outside `models/`, `logs/` and `data/uploads/` can be touched: a path
that resolves elsewhere — or to a root itself — is refused, so a stale absolute
`artifact_path` copied from another machine cannot delete anything.

`VACUUM` runs last, because SQLite keeps freed pages inside the file otherwise.
It needs an exclusive lock: with the server running it is skipped with a
warning and the next run reclaims the pages — the deletions themselves are
unaffected.

### Fortnightly schedule (macOS `launchd`)

```bash
bash scripts/install_purge_schedule.sh              # install + load, every 14 days
bash scripts/install_purge_schedule.sh --run-now    # trigger immediately
bash scripts/install_purge_schedule.sh --uninstall
GRACE_DAYS=30 bash scripts/install_purge_schedule.sh
```

Agent `com.cartplatform.purge-retired`, log `logs/retention/purge.log`. The run
is idempotent — an already-purged model is skipped — so a missed or repeated
fire is harmless.

```bash
sqlite3 cart_platform/data/platform.db \
  "SELECT COUNT(*), SUM(purged_bytes)/1048576 AS mb
     FROM model_registry WHERE purged_at IS NOT NULL;"
sqlite3 cart_platform/data/platform.db \
  "SELECT kind, COUNT(*), SUM(bytes)/1048576 AS mb
     FROM artifact_purges GROUP BY kind;"
```

On Linux the same command runs from cron:
`0 3 */14 * * cd /path/to/cart_platform && ../.venv/bin/python scripts/purge_retired_artifacts.py --apply`

---

## Further reading

| Document | Covers |
|---|---|
| `MULTI_SITE_SYSTEM_DESIGN.md` | multi-site architecture, phases P0–P7 |
| `CART_PLATFORM_ON_DEMAND_ENTITY_MODULE_SPEC.md` | upload, scoped training, backtests, receiver |
| `CART_MULTI_BAY_API_MODULE_SPEC.md` | multi-bay API contract |
| `RECEIVER_BACKEND_LOGIC.md`, `RECEIVER_ON_SQL_SERVER.md` | receiver internals and SQL Server port |
| `cart_platform/SCORING_API_WORKFLOW.md` | **request workflow for the backend**: order of operations, packets, status codes, idempotency |
| `cart_platform/README.md` | platform internals, routes, security |
| `api/README.md` | single-entity API |
| `optimization.md`, `instruction.md` | optimisation notes and working instructions |

---

## Caveat

Forecast accuracy is measured on held-out telemetry. Replenishment benefits are
**simulation estimates conditional on documented assumptions** — not measured
outcomes. No execution is implied unless a separate action log confirms it.
