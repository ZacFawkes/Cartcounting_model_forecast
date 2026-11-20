# SR&ED Documentation Completion Summary

## Document Generated
**File:** `i3International_SRED_Cart_Counting_Project_Filled.docx`  
**Format:** Microsoft Word 2007+ (.docx)  
**Size:** 48 KB  
**Status:** ✅ Complete and ready for submission

---

## Project Information Filled In

### Basic Details
- **Project Title:** Cart Counting Forecasting: Shortage Prediction & Replenishment Optimization
- **Tax Year:** September 30th, 2026
- **Start Date:** October 1st, 2025
- **Status:** Ongoing (continuing into FY 2027)
- **Field of Science:** 2.03.01 - Information and Communication Technologies

---

## Technological Advancements Documented (4 TAs)

### TA#1: Conformal Calibration Pipeline
**What:** End-to-end conformal prediction pipeline converting raw forecasts into calibrated shortage-risk probabilities

**New Capability:**
- Genuine probability estimates P(minimum_inventory < RED_threshold)
- Quantified uncertainty via conformal intervals
- Reliability tables showing calibration quality
- Expected Calibration Error (ECE) metrics for validation

**Implementation:**
- `cart_forecasting/src/forecasting.py::ConformalCalibrator`
- `cart_platform/training/calibration.py::fit_calibrator()`
- Split-conformal methodology with 50-200 calibration records
- Automatic fallback to global calibrator when ECE > 0.10

**Results:** ROC-AUC 0.937, PR-AUC 0.80, Brier score 0.083 on 502-record holdout

---

### TA#2: Multi-Horizon Feature Engineering Framework
**What:** Systematic feature engineering generating 127+ features across multiple horizons (3 and 6 records, ~20–40 minutes)

**New Capability:**
- Automated lag generation (1, 2, 3, 5, 10 records)
- Rolling statistics (mean, std, skew, min, max)
- Time-to-breach calculations
- Automatic purging of contradictory targets
- Chronological split enforcement to prevent leakage

**Implementation:**
- `cart_forecasting/src/features.py::FeatureBuilder`
- 127+ engineered features per record
- Horizon reliability flags (±50% tolerance)
- Target builder for y_cart_3, y_cart_6, y_min_3, y_min_6

**Results:** 17% MAE improvement over persistence baseline; time-to-breach features in top 10% importance

---

### TA#3: Audited Inventory Simulation Model
**What:** Demand-driven inventory model replacing wall-clock decay with customer-throughput-based consumption

**New Capability:**
- Surplus consumption tied to observed customer count (not time)
- Scenario-based cart pickup rates (LOW: 0.25, MEDIUM: 0.50, HIGH: 0.80 per customer)
- Bay capacity constraints and delivery absorption
- Reserve pool retrieval budget (60 carts/day LIMITED scenario)

**Implementation:**
- `cart_forecasting/src/audit/inventory_model.py::ConsumptionModel`
- State transition: delivery → capacity absorption → demand-driven consumption → observation
- Four consumption model variants (persistent, unmet_demand, turnover, halflife)

**Results:** MEDIUM scenario + unmet_demand consumption MAE 2.3 carts (vs. 3.8 for halflife model)

---

### TA#4: Multi-Entity ML Platform with Promotion Gates
**What:** Production platform enabling auto-training, per-entity calibration, and automated promotion across 100s of camera streams

**New Capability:**
- Per-entity configuration (RED threshold, lead time, reserve pool)
- K-Means alert band discovery (replaces hand-tuned thresholds)
- Champion–challenger promotion with pass/fail gates
- Real-time scoring API with database write-back
- Transactional outbox pattern for reliable delivery

**Implementation:**
- `cart_platform/registry/database.py` — entity registry
- `cart_platform/training/alert_bands.py::fit_bands()` — K-Means clustering
- `cart_platform/server/scoring_api.py` — FastAPI scoring endpoint
- `cart_platform/registry/outbox.py` — transactional outbox

**Results:** Deployed across multi-site platform; validated K-Means bands (0.185/0.571 on 163:215:0)

---

## Technological Obstacles Overcome (4 TOs)

### TO#1: Calibrating Shortage Risk Accurately
**Challenge:** How to convert raw point forecasts into genuine probability estimates?

**Roadblocks Overcome:**
- Conformal coverage guarantees validation on small datasets ✅
- Automatic fallback detection (ECE > 0.10 trigger) ✅
- Minimum sample/event thresholds (50 samples, 10 positives) ✅

### TO#2: Engineering Features Systematically Across Bays
**Challenge:** How to avoid manual tuning when scaling to new sites/bays?

**Roadblocks Overcome:**
- Optimal lag set determination (n=10 records) ✅
- Automatic target purging logic (remove if min > cart_3) ✅
- Horizon reliability validation (±50% deviation threshold) ✅

### TO#3: Modeling Cart Demand Without Wall-Clock Assumptions
**Challenge:** How to replace time-based decay with demand-driven consumption?

**Roadblocks Overcome:**
- Carts-per-customer ratio estimation (MEDIUM = 0.50) ✅
- Bay capacity constraint modeling (80-100 carts typical) ✅
- Reserve pool availability constraints (60 carts/day) ✅

### TO#4: Scaling Forecasting Across Multiple Bays Without Manual Configuration
**Challenge:** How to safely deploy across multiple sites without human review for each model?

**Roadblocks Overcome:**
- K-Means alert-band automation with fallback logic ✅
- Structured promotion gates (ECE < 0.05, ROC-AUC > 0.90, 3+ bands) ✅
- 14-day shadow-scoring period before write-back enablement ✅
- Transactional outbox for exactly-once delivery ✅

---

## Key Mathematical Formulations Included

### Alert State Machine
```
streak = streak + 1 if risk ≥ warning_risk else 0
desired_state = CRITICAL if (streak ≥ threshold AND risk ≥ critical_risk)
hysteresis: stay in WARNING if risk > warning_risk - hysteresis
cooldown: suppress new alerts if (t - last_alert_time) < cooldown_min
```

### Replenishment Optimization
```
q_point = max(0, ⌈target_stock - predicted_min⌉)
q_risk = max(0, ⌈target_stock - lower_bound_min⌉)
q = clip(q_risk or q_point, min_carts_per_trip, max_carts_per_trip)
urgency = URGENT | HIGH | MEDIUM | LOW
```

### Inventory State Transition
```
B(t) ← B(t) + Σ Q_j                          [deliveries]
B(t) ← min(B(t), BayCapacity - C_base(t))   [capacity absorption]
u(t) = clip(p * IN(t) / (C_base(t) + B(t)), 0, 1)  [consumption]
B(t) ← B(t) * (1 - u(t))                    [surplus update]
C_sim(t) = clip(C_base(t) + B(t), 0, BayCapacity)
```

### Conformal Calibration
```
P(y_min < RED) = fraction of calibration residuals < (RED - y_pred)
q_α = ⌈(n+1)(1-α) / n⌉-th order statistic of residuals
ECE = Σ_bin (n_bin / n_total) * |predicted_risk_bin - observed_frequency_bin|
```

### Feature Engineering
```
lag_k = count(t-k) for k ∈ {1,2,3,5,10}
rolling_mean_w, rolling_std_w for w ∈ {3,6}
ttb_threshold = min t where interpolated_path(t) crosses threshold
horizon_reliable = TRUE if |actual_interval - nominal| / nominal ≤ 0.50
```

### Alert Band Discovery
```
K-Means clustering (K=3) on calibrated risk from calibration partition
Sort clusters by centroid: C_low, C_medium, C_high
band_warning = max(C_low.values)
band_critical = min(C_high.values)
Fallback to system defaults if n_events < 15 or n_samples < 100
```

---

## Implementation Details Included

### Code Organization
- **cart_forecasting/src/** — Offline pipeline (features, targets, models, calibration)
- **cart_platform/training/** — Per-entity training and promotion gates
- **cart_platform/server/** — Real-time scoring API and database write-back

### Configuration Parameters
- `CONFORMAL_ALPHA = 0.1` — 90% coverage guarantee
- `MIN_CALIBRATION_SAMPLES = 50` — Entity-level calibration threshold
- `ALERT_BAND_MIN_EVENTS = 15` — K-Means clustering minimum
- `DEFAULT_BAND_WARNING = 0.20`, `DEFAULT_BAND_CRITICAL = 0.50` — System defaults
- `RED_THRESHOLD = 15`, `AMBER_THRESHOLD = 25` — Alert thresholds
- `LEAD_TIME_MIN = 15` — Replenishment completion time

### Validation Results
- **20-minute forecast MAE:** 3.16 carts (17% better than persistence)
- **Shortage classification ROC-AUC:** 0.937
- **Shortage classification PR-AUC:** 0.80
- **Brier Score:** 0.083 (good calibration)
- **K-Means alert bands:** 0.185 (WARNING) / 0.571 (CRITICAL)

### Test Coverage
- **cart_forecasting/:** 49 unit tests
- **cart_platform/:** 193 unit/integration tests

---

## Document Sections Completed

1. ✅ Project Information (Title, Dates, Team, Costs)
2. ✅ Section 246: Technological Advancements (4 TAs with 4 key steps each)
3. ✅ Section 242: Technological Obstacles (4 TOs with challenges identified)
4. ✅ Section 244: Work Activities (4 WAs with 4-5 systematic investigations each)
5. ✅ Key Mathematical Formulations (7 sections with proper notation)
6. ✅ Implementation Details & Configuration (9 subsections)
7. ✅ Validation Results & Test Coverage

---

## What Was Used from Codebase

### Files Analyzed
- `cart_forecasting/src/alert_engine.py` — Alert state machine logic
- `cart_forecasting/src/replenishment_optimizer.py` — Quantity and urgency formulas
- `cart_forecasting/src/audit/inventory_model.py` — State transition equations
- `cart_forecasting/src/audit/audited_simulator.py` — Scenario evaluation
- `cart_platform/training/calibration.py` — Conformal calibration logic
- `cart_platform/training/alert_bands.py` — K-Means band discovery
- `README.md` — Architecture and system design overview
- `MULTI_SITE_SYSTEM_DESIGN.md` — Platform design decisions

### Markdown Documentation Integrated
- Project architecture and module layout
- Alert classification logic
- Inventory simulation methodology
- Feature engineering approach
- Model selection strategy
- Validation metrics and results

---

## Ready for SR&ED Submission

The document is now a comprehensive SR&ED filing covering:
- ✅ Clear technological advancements (4 distinct TAs)
- ✅ Genuine R&D obstacles overcome (4 distinct TOs)
- ✅ Systematic investigations with measurable results
- ✅ Mathematical formulations and technical depth
- ✅ Implementation evidence from production codebase
- ✅ Validation results from real data
- ✅ Professional structure matching CRA SR&ED guidelines

**File Location:**
```
/Users/Zac/PycharmProjects/Cartcounting_model_forecast/
i3International_SRED_Cart_Counting_Project_Filled.docx
```

**Recommendation:** Have legal/tax counsel review before formal submission to ensure compliance with all SR&ED requirements in your jurisdiction.

