# Multi-Site Cart Forecasting Platform — System Design

**Status:** DRAFT FOR REVIEW — no code written yet
**Supersedes:** single-site pipeline in `cart_forecasting/`, API in `api/`
**Author:** generated from the existing codebase + simulation audit findings

---

## 0. How to review this document

Sections marked **[DECISION]** need your explicit sign-off before I write code.
There are **11** of them. Everything else is a consequence of those choices.

Read in this order: §1 (what we're building) → §2 (all decisions in one table) →
§3–§9 (detail) → §12 (build plan). If you only have 10 minutes, read §2 and §12.

---

## 1. Goal

Turn the validated single-site pipeline into a platform that:

1. **Auto-trains** forecasting models across many `(KDVR, ChanNo)` streams
2. **Auto-calibrates** the shortage-risk score per stream
3. Applies a **per-stream configuration** (thresholds, capacity, lead time)
4. Exposes a **backtest API** to evaluate any stream over any window
5. Exposes a **streaming scoring API** that receives the latest records when a
   new log arrives and **writes the decision back to the database**
6. Produces **logs keyed by store/channel** for everything above

### 1.1 What already exists and stays

| Component | State | Reused? |
|---|---|---|
| Feature engineering (143 candidates) | Validated, no leakage | Yes, needs entity grouping |
| Target builder, chronological splits, purging | Validated | Yes, needs entity grouping |
| Forecasting models (LightGBM + Ridge) | Validated, MAE 3.16 on 40-min min | Yes |
| Conformal calibration | Validated, 503 residuals | Yes, becomes per-entity |
| Alert engine (states, cooldown, hysteresis) | Validated | Yes, per-entity instance |
| Audited replenishment engine (`plan_replenishment`, `OrderBook`) | Audited and fixed | Yes, per-entity instance |
| `/score` + `/backtest` API | Working, 18 tests pass | Yes, extended |

### 1.2 What is new

Entity registry · per-entity config · pooled training orchestrator ·
per-entity calibration · model registry with versions · DB write-back ·
scheduled retraining · drift monitoring · per-entity logging.

---

## 2. Decisions required **[DECISION]**

| # | Decision | Options | My recommendation |
|---|---|---|---|
| D1 | Model strategy | Per-site / Pooled / Hybrid | **Hybrid** — one pooled forecast model, per-entity calibration + config |
| D2 | Records sent per streaming call | 10 / 12 / 20 | **20** (see §7.2 — 10 works but breaks on any telemetry gap) |
| D3 | Thresholds per site | Fixed 15/20 / Business rule / Derived from history | **Derived + override** (see §4.2) |
| D4 | Retrain cadence | Continuous / Weekly / Monthly / Manual | **Monthly**, champion–challenger, manual promote |
| D5 | Promotion gate | Auto / Manual | **Manual** for the first 2 cycles, then auto with guardrails |
| D6 | DB write-back | API writes / Caller writes / Both | **API writes** to its own table, also returns payload |
| D7 | Cold start for a new site | Block / Pooled model + global calibration / Shadow only | **Shadow only** until it has 14 days |
| D8 | Who owns replenishment execution | System auto-dispatch / Staff advisory | **Advisory only** (audit blocker, see §11) |
| D9 | Entity grain | `siteKey` / `KDVR+ChanNo` / `+AreaID` | **`KDVR + ChanNo`** — the physical camera stream (see §3.1) |
| D9b | Is `KDVR` globally unique? | Yes → `"215:0"` / No → `"163:215:0"` | **Needs your answer** (see §3.1) |
| D10 | `red`/`amber` override safety | Split `red_model`/`red_eval` / Rewrite features as relative | **Split now**, relative rewrite in P2 (see §4.4) |

---

## 3. Entity model

### 3.1 Grain

```
entity_id = f"{KDVR}:{ChanNo}"             e.g. "215:0"
```

An entity is **one camera stream watching one cart bay**. `KDVR` identifies the
DVR device; `ChanNo` identifies the channel on that device. The pair is what
physically produces the telemetry, so it is the correct modelling grain.

**The hierarchy:**

```
siteKey   (store)          163
  └── KDVR (DVR device)      215, 216, ...        a store may have several
        └── ChanNo (channel)   0, 1, 2, ...       each watching a bay
              └── entity = the modelled unit
```

**Why not `siteKey + ChanNo`:** channel 0 on DVR 215 and channel 0 on DVR 216
are different cameras pointing at different bays. Keying on the site would
collide them and mix unrelated inventories.

**`siteKey` is not discarded** — it becomes a first-class *attribute* used for:

- grouping entities when pooling training data
- store-level reporting and roll-ups
- shared operational parameters (staff, lead time, reserve pool are usually
  store-level, not camera-level)
- leave-one-**site**-out validation (§5.5), which must hold out a whole store,
  not a single camera, or the test leaks

`AreaID` / `CatID` are carried as metadata. If you later find they split a
stream further, the grain extends without redesign.

**[DECISION D9b] Is `KDVR` globally unique across stores?**

- If **yes** → `entity_id = "215:0"` is sufficient
- If **no** (DVR ids restart per store) → `entity_id = "163:215:0"` is required

I cannot determine this from the sample data (one site, one DVR, one channel).
Getting it wrong silently merges two stores' bays, so this needs a definite
answer. **If unsure, use the 3-part key** — it is always safe.

### 3.2 The one blocking code change

Today `block_id` is computed from a **single globally time-sorted stream**
(`data_loader.py:101`). With multiple entities interleaved, every rolling window
would mix DVR 215 channel 0 with DVR 216 channel 1.

```
NOW:     sort by DVRDate            → block_id = cumsum(gap > 15min)
NEEDED:  sort by entity_id, DVRDate → block_id = entity_id + cumsum(gap > 15min)
                                      where entity_id = KDVR:ChanNo
```

Everything downstream (`build_features`, `build_targets`, `shortage_events`)
already groups by `block_id`, so this single change makes the whole pipeline
entity-correct. **This is the highest-value, lowest-risk change in the project.**

### 3.3 Mandatory regression test

After the change, re-running the pipeline on the existing single-site data must
reproduce **today's numbers exactly** (MAE 3.16, 13 baseline events, 84.6% event
reduction). If it doesn't, the refactor is wrong. This is a hard gate.

---

## 4. Per-entity configuration

### 4.1 Registry

A table (not a Python file — it must be editable without deployment):

```
entity_config
├── entity_id            TEXT PK        "215:0"   (or "163:215:0" — see D9b)
├── kdvr                 INT            215       DVR device
├── chan_no              INT            0         channel on that device
├── site_key             INT            163       store (grouping attribute)
├── area_id              INT NULL
├── display_name         TEXT           "Store 163 — DVR 215 ch0 — North Bay"
├── timezone             TEXT
├── status               TEXT           active | shadow | disabled
│
│   -- physical, per BAY (REQUIRED, cannot be guessed)
├── bay_capacity         INT
│
│   -- physical, per STORE (inherited from site_config, overridable)
├── reserve_per_day      INT NULL       NULL = unconstrained, must be labelled
├── lead_time_min        FLOAT
│
│   -- thresholds, per BAY
├── red_threshold        INT
├── amber_threshold      INT
├── target_stock         INT
├── threshold_source     TEXT           business | derived | manual_override
│
│   -- operations
├── min_carts_per_trip   INT
├── max_carts_per_trip   INT
├── cooldown_min         FLOAT
├── risk_threshold       FLOAT
├── quantity_rule        TEXT           point | risk
│
│   -- data handling
├── max_sequence_gap_min FLOAT          default 15
├── model_id             TEXT           FK → model registry
├── calibration_id       TEXT           FK → calibrator registry
├── updated_at, updated_by
└── UNIQUE (kdvr, chan_no)
```

**Two-level config.** Some parameters belong to the **bay** (capacity,
thresholds), others to the **store** (staff reserve, retrieval lead time — one
team serves all bays in a store). A small `site_config` table holds the
store-level defaults; `entity_config` inherits and may override them. Without
this split you would duplicate the same lead time across every channel and have
no single place to correct it.

### 4.2 Threshold derivation **[DECISION D3]**

`RED=15` is a property of *that* bay. A 200-cart bay cannot use it.

Proposed rule when no business value is supplied:

```
red_threshold    = max(5, round(P10 of historical Count))
amber_threshold  = round(P25 of historical Count)
target_stock     = round(P60 of historical Count)
bay_capacity     = round(1.05 × max observed Count)     -- fallback only
```

With a mandatory manual override column, and `threshold_source` recording which
path was used so no report ever presents a derived number as a business rule.

**Risk:** percentile-derived thresholds encode *current* (possibly bad) service
levels as the target. Business-supplied values are strongly preferred.
**Needs your input: do you have real per-store cart counts and shortage
definitions?**

### 4.3 Parameter resolution — API override → entity default → system default

This is exactly the behaviour you asked for, formalised.

**Three-level precedence, highest wins:**

```
1. API request parameter        explicit per-call override (optional)
2. entity_config / site_config  YOUR per-KDVR:ChanNo default
3. system default               last-resort fallback, logged as "unconfigured"
```

If a request omits a parameter, the entity's configured value is used. If the
entity has no value, the system default applies **and the response is flagged**
`"config_source": {"red_threshold": "system_default", ...}` so nobody mistakes a
fallback for a deliberate setting.

The current API already does levels 1 and 3 (`PolicyOverrides.merged()` falls
back to `DEFAULT_POLICY`). The work is inserting level 2 and making the resolved
values auditable.

**Override matrix:**

| Parameter | Used by | Request override | Note |
|---|---|---|---|
| `target_stock` | policy only | ✅ free | |
| `lead_time_min` | policy + simulator | ✅ free | |
| `retention_halflife_min` | simulator only | ✅ free | backtest only, ignored by `/score` |
| `cooldown_min` | alert + policy | ✅ free | |
| `risk_threshold` | policy | ✅ free | |
| `min/max_carts_per_trip` | policy | ✅ free | |
| `reserve_per_day` | simulator | ✅ free | |
| `consumption_model`, `usage` | simulator only | ✅ free | backtest only |
| `bay_capacity` | simulator + normalisation | ⚠️ restricted | feeds relative features (§5.2) |
| **`red_threshold`** | **features + targets + policy + metrics** | ⚠️ **restricted** | see below |
| **`amber_threshold`** | **features + policy** | ⚠️ **restricted** | see below |

### 4.4 The RED/AMBER trap **[DECISION D10]**

`RED_THRESHOLD` and `AMBER_THRESHOLD` are not just policy numbers — they are
**baked into the feature vector**. Measured in the current code:

```
build_features() imports RED_THRESHOLD / AMBER_THRESHOLD at module level
  → 17 of the 127 served features are threshold-derived:
    below_red, below_amber, frac_below_amber_{3,6,9},
    ttb_red_{3,6,9}, ttb_red_{3,6,9}_finite,
    ttb_amber_{3,6,9}, ttb_amber_{3,6,9}_finite
```

So if a request passes `red_threshold = 25`, naively honouring it would build a
feature vector the model **has never seen** — `below_red` would fire constantly,
`ttb_red_*` would collapse — while the model's learned relationships assume 15.
The forecast would be silently wrong, with no error raised.

**Resolution — split the parameter into its two roles:**

| Role | Name | Scope | Overridable |
|---|---|---|---|
| Feature construction | `red_model` | Frozen with the model version, per entity | ❌ never at request time |
| Evaluation & policy | `red_eval` | What counts as a shortage; drives risk, alerts, quantity | ✅ per request |

`red_model` is recorded in the model registry alongside the feature list. If a
request's `red_eval` differs from the entity's `red_model`, the response carries
a warning:

```json
"warnings": ["red_eval=25 differs from red_model=15; features were built with
             the trained threshold. Risk and policy use 25."]
```

This is safe and useful — you can ask *"what if we called 25 a shortage?"*
without retraining — but it must never be silent.

**Long-term fix (recommended, folds into §5.2):** replace the 17 absolute
threshold features with **relative** ones — `fill_ratio = Count / bay_capacity`,
time-to-reach-percentile instead of time-to-reach-15. Then the feature vector is
threshold-free, `red` becomes purely an evaluation parameter, and the same
pooled model serves a 40-cart bay and a 200-cart bay without this caveat at all.

**Decision needed:** accept the `red_model` / `red_eval` split now (fast, works
with the existing model), with the relative-feature rewrite in P2? Or do the
rewrite first and retrain?

### 4.5 Traceability

Every `/score` and `/backtest` response, every `scoring_results` row and every
log line records the **fully resolved** parameter set plus its source:

```json
"resolved_config": {
  "red_threshold":   {"value": 15,  "source": "entity_config"},
  "target_stock":    {"value": 28,  "source": "request_override"},
  "lead_time_min":   {"value": 10,  "source": "site_config"},
  "cooldown_min":    {"value": 30,  "source": "system_default"}
}
```

Without this you cannot answer "why did store 163 get a different
recommendation than store 164 at the same cart level?" — which is the first
question anyone asks when the system goes live.

---

## 5. Auto-training module

### 5.1 Model strategy **[DECISION D1]**

| Option | Requirement | Assessment |
|---|---|---|
| A. Per-site model | 3+ months per site | Cleanest; you don't have the data |
| B. Pooled, one model for all | Many sites | Loses site-specific dynamics |
| C. **Hybrid (recommended)** | — | Pooled forecaster + per-entity calibration + per-entity config |

**Why hybrid:** depletion *dynamics* (slope, momentum, traffic pressure) likely
transfer across stores; absolute *levels* definitely do not. Conformal
calibration needs only a few hundred residuals, so every entity can afford its
own — giving each store a correctly-scaled risk even from a shared model.

### 5.2 Feature normalization (required for pooling)

Current features are absolute (`distance_to_red`, `cart_mean_9`). For a pooled
model, add relative counterparts:

```
cart_fill_ratio      = Count / bay_capacity
cart_vs_site_median  = Count / site_median_count
distance_to_red_norm = (Count − red) / bay_capacity
pressure_norm        = traffic_in_sum_W / bay_capacity
```

Absolute features are kept — the model can use both — but relative features are
what make transfer possible. `entity_id` itself is **never** a feature (it would
memorise sites and destroy cold-start).

### 5.3 Pipeline

```
1. Ingest          all entities, grouped, validated
2. Entity gate     skip entities with < MIN_DAYS or < MIN_EVENTS → shadow
3. Features        entity-grouped, absolute + relative
4. Splits          chronological per entity, purged; a global cut date
                   keeps all entities on the same calendar split
5. Screen          constants/missing/redundant, on TRAIN only
6. Clip            robust bounds fitted on TRAIN only
7. Train           pooled, 4 targets, same model grid as today
8. Calibrate       per entity, on that entity's VALIDATION residuals
9. Evaluate        per entity + leave-one-site-out (§5.5)
10. Policy search  per entity, on VALIDATION, with audited simulator
11. Register       model + per-entity calibrators + per-entity policy
12. Gate           champion vs challenger → promote or reject
```

### 5.4 Auto-calibration

Calibration is cheaper than training and should run more often.

| Trigger | Action |
|---|---|
| Scheduled (weekly) | Refit conformal residuals on the last N days per entity |
| Drift detected (§10) | Refit that entity, flag for review |
| New entity reaches 14 days | Promote from global to its own calibrator |
| Brier score degrades > X% | Alert, do not auto-refit |

**Hard rule:** calibration residuals must come from a window the model was
**not** trained on. Auto-calibration that silently uses training data destroys
the risk score. This is the single easiest thing to get wrong.

### 5.5 Leave-one-site-out validation **[mandatory]**

Hold out a **whole store** (every `KDVR:ChanNo` belonging to one `siteKey`),
train on the rest, measure on the held-out store, rotate. Holding out a single
*channel* is not a valid test — sibling channels in the same store share staff,
weather and trading patterns, so the result would leak. Without this there is no evidence
the approach transfers — only that it fits. The audit already lists "single-site
evaluation" as a limitation; multi-site fixes it **only** if validated this way.

**Expect worse numbers than today's 3.16 MAE.** That is the honest cost of
generalization, and the report must show it.

### 5.6 Model registry

```
model_registry                    calibrator_registry
├── model_id PK                   ├── calibration_id PK
├── version, trained_at           ├── entity_id, model_id
├── entities_trained_on      -- list of KDVR:ChanNo
├── sites_trained_on         -- list of siteKey           ├── fitted_at, window_start/end
├── train/val/holdout window      ├── n_residuals, alpha
├── feature_list, clip_bounds     ├── lower_offset
├── metrics_per_entity            ├── empirical_coverage
├── loso_metrics                  ├── brier, pr_auc
├── status: candidate|champion|   └── status
│           retired
└── artifact_path
```

Replaces today's single overwritten `cart_forecasting_models.pkl` — which has no
version and no provenance.

### 5.7 Promotion gate **[DECISION D5]**

A challenger is promoted only if, on a **fresh** holdout:

- MAE (40-min minimum) not worse than champion by > 5%
- PR-AUC not worse by > 5%
- Conformal coverage within [0.85, 0.95] for ≥ 80% of entities
- No entity degrades by more than 20%
- Leakage checks pass (reuse the 13 existing probes)

First two cycles: human approves. After that: auto-promote if all gates pass,
with one-click rollback.

---

## 6. Backtest API

Extends the existing `/backtest`.

```
POST /v1/backtest
{
  "entity_id": "215:0",              // KDVR:ChanNo, or "entity_ids": [...]
  // alternatively select a whole store:  "site_key": 163
  "start": "...", "end": "...",
  "mode": "replay" | "closed_loop",
  "model_id": "mdl-2026-10-v3",      // optional; default = champion
  "policy_overrides": {...},
  "inventory_overrides": {...}
}
```

Returns per entity: measured `forecast_accuracy`, simulated `metrics`,
`attribution_summary`, decision list, `run_id`.

```
GET  /v1/backtest/runs?entity_id=&site_key=&from=&to=
GET  /v1/backtest/runs/{run_id}
POST /v1/backtest/compare     { "run_ids": [...] }     // champion vs challenger
```

**Carried-forward caveat (non-negotiable):** forecast accuracy is *measured*;
shortage-reduction figures are *simulator output*. The audit showed the same
policy produces between **4.2% and 100%** severity reduction depending on the
inventory consumption assumption. Every response keeps the `caveat` field and
every scenario its assumption block.

---

## 7. Streaming scoring API

### 7.1 Flow

```
DB insert (new log row)
      │
      ▼  trigger / CDC / poller
Collector  ── fetch last N records for that entity ──►  POST /v1/score
                                                              │
                                              ┌───────────────┤
                                              ▼               ▼
                                    scoring_results      response to caller
                                      (database)
```

### 7.2 How many records **[DECISION D2]**

Measured, not assumed:

| Records | Result |
|---|---|
| 8, 10, 11 | **422 rejected** (API enforces 12) |
| 12 | risk 0.3082, min40 17.140, qty 11 — 69 ms |
| 20 / 30 / 60 / 120 | **identical output**, 55–62 ms |

- **10** = hard mathematical floor (`cart_lag_9`, `cart_momentum_9`,
  `cart_diff_volatility_9` all need `t-9`)
- **12** = currently enforced
- **20** = recommended

You asked for 10. It works *only* if the 10 records are consecutive with no gap
> 15 min. After any telemetry outage the window resets and you need 10 fresh
records — with exactly 10 in flight, one dropped record means a rejected call.
Output and latency are identical at 20, so the margin is free.

**If you still want exactly 10:** lower `MIN_HISTORY_RECORDS` to 10 and accept
that gaps produce `422 INSUFFICIENT_HISTORY`, which the collector must handle.

### 7.3 Contract

```
POST /v1/score
{
  "entity_id": "215:0",              // KDVR:ChanNo
  "records": [ {DVRDate, next_DVRDate, Count, Count_IN, Count_OUT}, ... ],
  "write_to_db": true,
  "idempotency_key": "215:0@2026-10-11T10:53:08"
}
```

Response: forecast (4 horizons + lower bound), `shortage_risk`, `alert_state`,
`action` (should_replenish, quantity, urgency, order_id, deliver_by, reason),
`pending_carts`, `model_id`, `calibration_id`.

**Ordering guarantee:** records must be consecutive and chronological, last =
newest **completed** record. Never send an interval still in progress — its
`Count_IN`/`Count_OUT` are incomplete, which is the one leakage the pipeline
cannot detect for you.

**Idempotency:** re-sending a processed record is already a no-op today and must
stay that way (prevents duplicate orders on retry). Key on
`entity_id + decision_time`.

### 7.4 Write-back schema **[DECISION D6]**

```
scoring_results
├── id PK, request_id
├── entity_id, kdvr, chan_no, site_key     -- denormalised for query speed
├── decision_time        (unique with entity_id)
├── current_cart, simulated_cart
├── pred_cart_20min, pred_cart_40min
├── pred_min_20min, pred_min_40min, lower_bound_min_40min
├── shortage_risk, alert_state, new_predictive_alert, reactive_critical
├── should_replenish, quantity, urgency, order_id, deliver_by
├── pending_carts, reason
├── model_id, calibration_id, policy_version
├── scored_at, latency_ms
└── UNIQUE (entity_id, decision_time)      -- idempotency at DB level
```

```
replenishment_orders
├── order_id PK
├── entity_id, kdvr, chan_no, site_key
├── decision_time, quantity, expected_completion_time
├── status            PENDING | COMPLETED | CANCELLED
├── delivered_quantity, delivered_at
├── urgency, risk_at_decision, predicted_min_at_decision
└── executed_by, executed_at, actual_carts      -- ⚠ see §11
```

The last three columns are the **replenishment action log** the audit named as
the single blocking gap. Even if staff fill them manually at first, having the
column is what eventually converts a simulated benefit into a measured one.

### 7.5 Session state

Today's session state (pending orders, cooldown, alert state) is **in-memory** —
a server restart loses it, and duplicate orders become possible.

Must move to Redis or a DB table keyed by `entity_id`, with the order book as
the source of truth. Pending orders are already persisted in
`replenishment_orders`, so the store can be rebuilt from it on startup.

---

## 8. Logging — per store/channel

Every log line, every table, every file carries `entity_id`.

| Sink | Content | Retention |
|---|---|---|
| `api_requests` (table or JSONL) | request_id, entity_id, endpoint, latency, status, error | 90 days |
| `scoring_results` (table) | one row per decision (§7.4) | permanent |
| `replenishment_orders` (table) | order lifecycle | permanent |
| `backtest_runs` (table) | one row per run + artifact path | permanent |
| `logs/runs/{run_id}/` (files) | decisions, order book, events, attribution, series | 1 year |
| `training_runs` (table) | model_id, entities, metrics, LOSO, gate result | permanent |
| `drift_metrics` (table) | per entity, per day: feature drift, Brier, coverage | 1 year |

Queries this must answer without a data engineer:

- "Show every replenishment recommendation for store 163 channel 0 last week"
- "Which entities have Brier degrading month over month?"
- "Why did the system recommend 16 carts at 10:53 on 11 Oct?" → `request_id` →
  full input + model_id + calibration_id + reason
- "Which entities are in shadow and why?"

---

## 9. Failure modes

| Failure | Handling |
|---|---|
| Fewer than 10 usable records | `422 INSUFFICIENT_HISTORY`; collector retries next record |
| Telemetry gap > 15 min | Window resets; entity returns NO_ACTION until 10 fresh records |
| Record arrives out of order | Reject if `decision_time <= last processed`; log it |
| Duplicate record | Idempotency key → return the stored result, no new order |
| Model/calibrator missing for entity | Fall back to pooled model + global calibrator, flag `degraded` |
| Entity not in registry | `404 UNKNOWN_ENTITY`; never score with default thresholds |
| DB write fails | Still return the response; queue the write for retry; never lose a decision |
| Server restart | Rebuild session state from `replenishment_orders` |
| Clock skew between site and server | Trust telemetry timestamps, never server time |

---

## 10. Drift monitoring

Per entity, daily:

- **Feature drift** — PSI of top-15 features vs training distribution
- **Prediction drift** — mean/std of predicted 40-min minimum
- **Calibration drift** — rolling Brier + conformal coverage vs the 90% target
- **Operational drift** — alerts/day, trips/day, carts/day

Thresholds trigger: `warn` → review; `degraded` → auto-recalibrate; `critical` →
drop the entity to shadow and page a human.

---

## 11. Risks carried over from the audit **[must stay visible]**

These do not disappear by going multi-site — several get **worse**.

1. **No replenishment action log.** Still the single blocking gap. Without it
   the baseline is not a no-intervention counterfactual and no benefit can be
   measured. §7.4 adds the columns; filling them is an operational commitment.

2. **Feedback contamination.** Once staff act on recommendations, future `Count`
   is no longer a natural trajectory. Auto-training on that data makes the model
   learn its own actions. **This is why D4 recommends monthly retraining with a
   manual gate, and D8 recommends advisory-only.** Mitigation: record which
   decisions were acted on and either exclude or explicitly model those periods.

3. **Benefit magnitude is not credible as a point estimate.** Audit range:
   4.2%–100% severity reduction depending on an unobservable assumption. Every
   API response and report must keep the scenario range.

4. **Thin data.** 14 days, 1 site today. Per-site models are not viable; pooling
   is mandatory until there are months of history.

5. **Cold start.** A new entity has no calibration. D7 keeps it in shadow —
   scored and logged but no recommendation surfaced — for its first 14 days.

---

## 12. Build plan

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | Entity-aware refactor (`block_id` = entity + gap) | **Reproduces today's numbers exactly** |
| **P1** | Entity registry + per-entity config + threshold derivation | Config-driven single-site run matches P0 |
| **P2** | Pooled training + per-entity calibration + model registry | LOSO validation report |
| **P3** | Backtest API v1 (multi-entity, model selection, compare) | Audit numbers reproduced via API |
| **P4** | Streaming scoring API + DB write-back + persistent sessions | Restart test: no duplicate orders |
| **P5** | Scheduled retrain + champion–challenger + promotion gate | Two manual cycles before auto |
| **P6** | Drift monitoring + per-entity dashboards | Alerts fire on injected drift |

P0 is non-negotiable and comes first: it is small, it is testable against known
numbers, and everything else depends on it.

---

## 13. What I need from you

1. Sign off the **11 decisions** in §2
2. **The `KDVR` uniqueness answer** (D9b) and the full entity inventory:
   which `(KDVR, ChanNo)` pairs exist, and which `siteKey` each belongs to
3. **Per-bay physical data**: bay capacity, real cart counts, and whether a
   business definition of "shortage" exists (§4.2). Note capacity is per *bay*,
   so it is per `(KDVR, ChanNo)`, not per store
4. **Database access pattern**: can we add tables, or must we write to an
   existing schema? Is there CDC/trigger capability, or do we poll? (§7.1)
5. **How many entities** are in scope, and how much history does each have?
6. Confirm **advisory-only** operation for the first phase (§11.2)

---

## 14. Open questions

- **Is `KDVR` globally unique across stores, or does it restart per store?**
  (D9b — determines a 2-part or 3-part entity key. If unsure, use 3-part.)
- Do multiple `ChanNo` on the **same** `KDVR` watch the **same** bay — e.g. two
  cameras covering one large bay from different angles? If so those channels
  must be merged (summed or deduplicated) **before** modelling, otherwise the
  same carts are counted twice and every threshold is wrong.
- Conversely, can one bay be split across **two different DVRs**? That would
  break the `KDVR:ChanNo` grain and require an explicit bay mapping table.
- Are store opening hours known? Overnight periods currently inflate "safe"
  records and dilute risk metrics.
- Is there a cart-return/collection process that already runs on a schedule? It
  would show up as a recurring jump in `Count` and should be modelled explicitly
  rather than treated as noise.
- What is the real retrieval lead time, and does it vary by store size?
