MASTER PROJECT

Cart Availability Forecasting, Shortage Early Warning & Replenishment Optimization

YOUR ROLE

Act as a Senior Data Scientist, Machine Learning Engineer, and Operations Research Specialist.

Build a complete, reproducible, production-oriented analytical pipeline for predicting cart shortages and recommending proactive cart replenishment.

The solution must combine:

1. Time-series forecasting.
2. Advanced statistical feature engineering.
3. Shortage risk estimation.
4. Replenishment quantity optimization.
5. Replenishment timing optimization.
6. Chronological backtesting.
7. Operational simulation.
8. Model explainability.
9. Automated professional DOCX reporting.

The primary business objective is:

Determine WHEN employees should replenish carts and HOW MANY carts they should retrieve to minimize customer-facing cart shortages while avoiding unnecessary replenishment work.

Do not optimize forecasting accuracy alone.

The final system must quantify forecasting quality, early-warning effectiveness, and simulated operational benefits.

⸻

PART 1 — DATA UNDERSTANDING

1.1 Input

Dataset:

Thinh(2).csv

Existing implementation:

forecast_20min_holdout.py

Inspect both files before modifying the project.

Preserve the original implementation as a benchmark.

The dataset contains telemetry from Site 163.

Expected columns:

* CountID
* DVRDate
* next_DVRDate
* KDVR
* ChanNo
* CatID
* AreaID
* Count
* Count_IN
* Count_OUT
* CNN_Index
* siteKey

Inspect delimiter, data types, missing values, duplicates, and timestamp ordering.

1.2 Data Semantics

Each record represents:

[DVRDate, next_DVRDate)

Where:

* Count_IN: total incoming customer traffic during the interval.
* Count_OUT: total outgoing customer traffic during the interval.
* Count: cart availability measurement associated with the interval.

Treat next_DVRDate as the earliest availability timestamp for completed interval traffic.

Verify the measurement timestamp semantics of Count.

Do not use incomplete future traffic in historical features.

1.3 Frequency Assumption

For this project, assume:

3 records ≈ 20 minutes.

Use:

* 3 records ≈ 20 minutes.
* 6 records ≈ 40 minutes.
* 9 records ≈ 60 minutes.

Preserve original timestamps.

Do not resample into fixed 20-minute bins.

Calculate actual elapsed time for every prediction.

Report deviations from nominal horizons.

If unusually long gaps occur, mark affected predictions as unreliable or exclude them according to documented rules.

⸻

PART 2 — FORECASTING STRATEGY

2.1 Event-Driven Scoring

Run inference whenever a new completed telemetry record becomes available.

Expected scoring frequency:

Approximately every 6–8 minutes.

Training and inference must remain separate.

Do not retrain the model after every record.

2.2 Forecast Targets

Implement direct forecasting for:

Target A — Cart Count at +3 Records

[
Y_{cart,3}=C_{t+3}
]

Target B — Cart Count at +6 Records

[
Y_{cart,6}=C_{t+6}
]

Target C — Minimum Cart Count Within 3 Records

[
Y_{min,3}=\min(C_{t+1},C_{t+2},C_{t+3})
]

Target D — Minimum Cart Count Within 6 Records

[
Y_{min,6}=\min(C_{t+1},…,C_{t+6})
]

The minimum forecast is especially important for shortage prevention.

A shortage can occur between forecast endpoints.

Do not estimate the future minimum using only the two endpoint predictions.

Train direct minimum-forecast models or another explicitly validated approach.

2.3 Shortage Targets

Use:

RED_THRESHOLD = 15
AMBER_THRESHOLD = 20

Create:

shortage_next_3
shortage_next_6

Where:

shortage_next_3 = 1

if any future Count among the next 3 records is below 15.

shortage_next_6 = 1

if any future Count among the next 6 records is below 15.

Distinguish:

* Shortage already active.
* New shortage onset.
* Shortage recovery.

For predictive early-warning metrics, focus on new shortage onsets.

⸻

PART 3 — FEATURE ENGINEERING

Use one unified feature configuration.

Do not create feature ablation experiments A, B, C, D, and E.

Use:

Cart Count + Traffic IN/OUT + Temporal Features + Advanced Statistical Features.

Feature windows:

W = [3, 6, 9]

Every feature must use only data available at prediction time.

3.1 Current State

Create:

cart_current
traffic_in_current
traffic_out_current
traffic_net_current
interval_duration
traffic_in_rate
traffic_out_rate

Calculate:

[
NetTraffic=IN-OUT
]

[
INRate=\frac{IN}{Duration}
]

[
OUTRate=\frac{OUT}{Duration}
]

3.2 Cart Lag Features

Create:

cart_lag_1
cart_lag_2
cart_lag_3
cart_lag_6
cart_lag_9

3.3 Traffic Lag Features

Create:

traffic_in_lag_1
traffic_in_lag_2
traffic_in_lag_3
traffic_in_lag_6
traffic_out_lag_1
traffic_out_lag_2
traffic_out_lag_3
traffic_out_lag_6

3.4 Rolling Statistics

For each W in [3, 6, 9], calculate:

Cart:

* Mean.
* Median.
* Minimum.
* Maximum.
* Standard deviation.
* Range.

Traffic:

* IN sum.
* OUT sum.
* Net traffic sum.
* Mean traffic rate.
* Maximum traffic rate.
* Traffic volatility.

3.5 Traffic Ratios

IN/OUT Ratio

[
R_{inout,W}=
\frac{IN_{sum,W}+\epsilon}
{OUT_{sum,W}+\epsilon}
]

Net Traffic Ratio

[
R_{net,W}=
\frac{IN_{sum,W}-OUT_{sum,W}}
{IN_{sum,W}+OUT_{sum,W}+\epsilon}
]

Traffic Surge Ratio

[
R_{surge}=
\frac{INRate_3}
{INRate_9+\epsilon}
]

Traffic Pressure

[
R_{pressure,W}=
\frac{IN_{sum,W}}
{C_t+\epsilon}
]

Use safe division and robust clipping.

3.6 Cart Momentum

Calculate:

[
Momentum_W=C_t-C_{t-W}
]

For:

* W=3.
* W=6.
* W=9.

3.7 Cart Depletion Rate

Calculate:

[
DepletionRate_W=
\frac{C_t-C_{t-W}}
{Time_t-Time_{t-W}}
]

Use actual elapsed minutes.

Negative values indicate declining cart availability.

3.8 Cart Linear Slope

Fit a rolling linear regression:

[
C_i=\alpha+\beta Time_i+\epsilon_i
]

Generate:

cart_slope_3
cart_slope_6
cart_slope_9

Also calculate rolling trend R².

3.9 Traffic Slope

Fit rolling linear regressions using traffic rates.

Generate:

traffic_in_slope_3
traffic_in_slope_6
traffic_in_slope_9
traffic_out_slope_3
traffic_out_slope_6
traffic_out_slope_9

3.10 Acceleration

Calculate changes in recent slope or depletion velocity.

Generate:

cart_acceleration
traffic_in_acceleration
traffic_out_acceleration

Normalize comparisons for elapsed time.

3.11 Volatility

Calculate:

cart_volatility_3
cart_volatility_6
cart_volatility_9

Also calculate volatility of Cart Count differences.

3.12 Threshold Distance

Generate:

distance_to_red
distance_to_amber
below_red
below_amber

Where:

[
DistanceToRed=C_t-15
]

[
DistanceToAmber=C_t-20
]

3.13 Time-to-Threshold Heuristic

Calculate:

[
TTB=
\frac{C_t-Threshold}
{DepletionMagnitude+\epsilon}
]

Only return a finite positive time-to-breach when cart availability is above the threshold and depletion is occurring.

This is a heuristic, not a calibrated forecast.

3.14 Temporal Features

Create:

hour
minute
day_of_week
is_weekend
hour_sin
hour_cos
weekday_sin
weekday_cos

3.15 Feature Quality

Remove:

* Constant features.
* Invalid features.
* Excessively unstable ratios.
* Features with severe missingness.
* Features that create leakage.

Use conservative feature selection based on training and validation data.

Do not use the final holdout for feature selection.

⸻

PART 4 — MODEL TRAINING

4.1 Models

Compare:

Model A — Persistence

Predict future Cart Count using current Cart Count.

For minimum forecasting, use current Count as a simple reference baseline.

Model B — Ridge Regression

Use standardized numerical features.

Model C — LightGBM Regression

Use conservative hyperparameters.

Suggested search space:

num_leaves = [7, 15, 31]
max_depth = [3, 5]
learning_rate = [0.01, 0.03, 0.05]
min_child_samples = [20, 40, 80]
n_estimators = [100, 200, 400]

Avoid excessive hyperparameter tuning.

4.2 Model Selection

Select models using chronological validation.

Evaluate:

* MAE.
* RMSE.
* Bias.
* MAE near shortage.
* Minimum forecast accuracy.
* Shortage warning performance.

Do not select a model solely because it has the lowest overall MAE.

4.3 Prediction Uncertainty

Generate uncertainty estimates for future minimum Cart Count.

Investigate residual-based quantile calibration or conformal prediction.

Use training and calibration periods only.

Report empirical coverage on holdout.

Do not assume normal residuals without validation.

⸻

PART 5 — SHORTAGE RISK ESTIMATION

5.1 Risk Definition

Estimate:

[
P\left(\min(C_{t+1:t+6})<15\mid X_t\right)
]

Use calibrated probabilities where possible.

Do not convert point forecasts directly into probabilities without an uncertainty model.

5.2 Risk Modeling

Start with forecast-derived risk.

If sufficient independent shortage events exist, evaluate a direct classifier.

Candidate models:

* Logistic Regression.
* LightGBM Classifier.

Use chronological validation.

Evaluate:

* PR-AUC.
* Recall.
* Precision.
* Brier Score.
* Calibration.
* Event-level warning performance.

⸻

PART 6 — REPLENISHMENT OPTIMIZATION

This is the most important business component.

The system must recommend:

1. When replenishment should begin.
2. How many carts should be retrieved.
3. When replenishment must be completed.
4. How much shortage risk remains after replenishment.

6.1 Initial Operational Parameters

Use:

RED_THRESHOLD = 15
AMBER_THRESHOLD = 20
TARGET_STOCK = 25
PRIMARY_HORIZON = 3
PLANNING_HORIZON = 6
REPLENISHMENT_LEAD_TIME_MIN = 10

Treat these as configurable assumptions.

6.2 Forecast Minimum

Estimate:

[
M_{t,6}=\min(C_{t+1},…,C_{t+6})
]

Generate:

* Predicted minimum.
* Lower prediction bound.
* Estimated shortage probability.

6.3 Initial Top-Up Recommendation

Calculate:

[
Q_{point}=
\max(0,TargetStock-\hat M_{t,6})
]

Calculate a conservative alternative:

[
Q_{risk}=
\max(0,TargetStock-LowerBound(M_{t,6}))
]

Round quantities upward to whole carts.

Do not automatically use the conservative quantity without validating its operational cost.

6.4 Timing-Aware Replenishment

Do not assume replenishment happens instantly.

Use:

REPLENISHMENT_LEAD_TIME_MIN = 10

A replenishment decision made at time T becomes effective at approximately T+10 minutes.

The optimizer must account for cart availability before and after replenishment.

If shortage is predicted before replenishment can be completed, classify the situation as urgent.

6.5 Dynamic Inventory Simulation

Model the impact of replenishment using:

[
C^{policy}_{t+k}

C^{baseline}{t+k}
+
\sum{j:\tau_j+L_j\leq t+k}Q_j
]

Where:

* (C^{baseline}): estimated cart availability without additional interventions.
* (Q_j): carts replenished by intervention j.
* (\tau_j): intervention decision time.
* (L_j): replenishment lead time.

This additive model is only a simplified initial simulator.

Account for bay capacity and possible interactions between replenishment and future cart availability.

Do not equate customer traffic with cart demand one-to-one.

Do not claim simulated outcomes are observed causal effects.

6.6 Optimization Objective

Select replenishment quantity and timing by minimizing a weighted operational cost.

Use:

[
J=
\lambda_1 ShortagePenalty
+
\lambda_2 ReplenishmentTrips
+
\lambda_3 CartsMoved
+
\lambda_4 ExcessInventoryPenalty
]

Where:

* ShortagePenalty measures shortage severity and duration.
* ReplenishmentTrips penalizes staff workload.
* CartsMoved penalizes retrieval effort.
* ExcessInventoryPenalty penalizes unnecessary replenishment.

Use configurable cost coefficients.

If actual business costs are unavailable, perform sensitivity analysis.

6.7 Candidate Policy Search

Evaluate candidate policies using validation data.

Search over:

* Risk threshold.
* Target stock.
* Replenishment quantity.
* Replenishment lead time assumptions.
* Alert cooldown.
* Minimum carts per retrieval trip.

Avoid an excessively large search space.

Select the best feasible policy based on validation performance.

Freeze the selected policy before final holdout evaluation.

6.8 Operational Constraints

Support:

bay_capacity
reserve_cart_inventory
max_carts_per_trip
replenishment_lead_time

If these values are unknown, explicitly mark recommendations as unconstrained.

Do not invent available reserve carts or staff capacity.

⸻

PART 7 — EARLY-WARNING ENGINE

7.1 Alert States

Implement:

NORMAL
WARNING
CRITICAL

Use risk thresholds calibrated on validation data.

7.2 Direct Alerts

If observed Cart Count is already below 15:

Generate an immediate critical operational alert.

Do not count it as a successful predictive warning.

7.3 Predictive Alerts

Generate warnings based on forecast shortage risk.

Use:

* Deduplication.
* Cooldown.
* Hysteresis.
* Escalation.
* Recovery conditions.

7.4 Shortage Events

Group consecutive observations below RED into independent shortage events.

Use an explicit recovery rule.

Report:

* Event ID.
* Onset timestamp.
* Recovery timestamp.
* Minimum Cart Count.
* Event duration.

7.5 Warning Lead Time

Calculate:

[
LeadTime=ShortageOnset-FirstWarning
]

Report:

* Mean lead time.
* Median lead time.
* Event recall.
* False alerts per day.
* Percentage warned at least 10 minutes early.
* Percentage warned at least 20 minutes early.

⸻

PART 8 — CHRONOLOGICAL BACKTESTING

8.1 Data Splitting

Never use random train-test splitting.

Use chronological evaluation.

Suggested initial split:

* Earliest 60% of complete days: training.
* Next 20%: validation.
* Latest 20%: final holdout.

Inspect daily completeness and shortage event distribution before finalizing splits.

Use walk-forward validation when feasible.

8.2 Leakage Prevention

Prevent:

* Future traffic leakage.
* Future Count leakage.
* Target leakage across split boundaries.
* Rolling feature leakage.
* Hyperparameter tuning on holdout.
* Policy tuning on holdout.

Use purging or appropriate temporal gaps when overlapping target windows cross split boundaries.

8.3 Forecast Backtest

Generate predictions for every eligible historical scoring origin.

Compare:

* Persistence.
* Ridge.
* LightGBM.

Evaluate both endpoint and future-minimum forecasts.

8.4 Alert Backtest

Replay historical telemetry chronologically.

At each prediction origin:

1. Build available features.
2. Generate forecasts.
3. Estimate shortage risk.
4. Update alert state.
5. Record warnings.
6. Compare warnings with future shortage events.

Never use future observations when generating historical alerts.

8.5 Replenishment Policy Backtest

Simulate replenishment decisions chronologically.

At each eligible origin:

1. Observe current available information.
2. Predict future cart availability.
3. Estimate shortage risk.
4. Determine whether replenishment is needed.
5. Select replenishment quantity.
6. Schedule replenishment completion.
7. Update simulated inventory when replenishment completes.
8. Continue to the next telemetry event.

Avoid double-counting replenishments.

Do not repeatedly recommend the same top-up while an earlier replenishment is pending.

Track pending replenishment orders.

8.6 Historical Baseline Versus Simulated Policy

Compare:

Baseline

Observed historical cart availability.

Proposed Policy

Simulated cart availability after recommended interventions.

Clearly state that historical data may already contain replenishment activity.

Without recorded historical replenishment actions, the baseline is not necessarily a no-intervention scenario.

Therefore, estimated shortage reduction must be presented as a scenario-based simulation, not a proven causal effect.

⸻

PART 9 — BACKTEST EVALUATION

9.1 Forecasting Metrics

Calculate:

* MAE.
* RMSE.
* Bias.
* MAE near shortage.
* Minimum forecast MAE.
* Prediction interval coverage.

9.2 Warning Metrics

Calculate:

* Number of shortage events.
* Event recall.
* Event precision.
* Median warning lead time.
* Mean warning lead time.
* False alerts per store per day.
* Percentage of events warned at least 10 minutes early.
* Percentage of events warned at least 20 minutes early.

9.3 Replenishment Metrics

Calculate:

* Number of recommended replenishment actions.
* Average carts per action.
* Total carts recommended.
* Number of potentially prevented shortage events.
* Simulated shortage duration.
* Simulated shortage severity.
* Simulated shortage reduction percentage.
* Unnecessary replenishment actions.
* Average staff retrieval workload.

9.4 Policy Robustness

Evaluate sensitivity to:

* Replenishment lead time.
* Target stock.
* Forecast uncertainty.
* Risk threshold.
* Replenishment capacity.

Do not optimize these values using the final holdout.

⸻

PART 10 — AUTOMATED DOCX REPORT

Generate:

Cart_Counting_Forecasting_Replenishment_Backtest_Report.docx

Use Python and python-docx.

The report must communicate both technical model performance and operational business value.

Target audience:

* Data Science Team.
* Engineering Managers.
* Product Managers.
* Retail Operations Managers.

Use professional formatting.

Include:

* Executive summary.
* Key performance indicators.
* Data overview.
* Methodology.
* Forecasting results.
* Shortage early-warning results.
* Replenishment optimization.
* Policy backtesting.
* Operational recommendations.
* Limitations.
* Next steps.

Do not fabricate metrics.

⸻

PART 11 — REQUIRED REPORT STRUCTURE

SECTION 1 — EXECUTIVE SUMMARY

Answer:

1. Can future cart shortages be predicted?
2. How accurate are the forecasts?
3. How early can shortages be detected?
4. How many carts should employees replenish?
5. How much operational improvement is estimated?
6. Is the approach ready for shadow deployment?

Include a KPI summary table.

SECTION 2 — DATA OVERVIEW

Include:

* Date range.
* Total records.
* Number of days.
* Average Cart Count.
* Minimum Cart Count.
* Maximum Cart Count.
* Traffic statistics.
* Independent shortage events.
* Data quality issues.

Include a historical Cart Count and Traffic timeline.

SECTION 3 — FORECASTING RESULTS

Include model comparison:

Model	Horizon	MAE	RMSE	Bias
Persistence	+3	Computed	Computed	Computed
Ridge	+3	Computed	Computed	Computed
LightGBM	+3	Computed	Computed	Computed
Persistence	+6	Computed	Computed	Computed
Ridge	+6	Computed	Computed	Computed
LightGBM	+6	Computed	Computed	Computed

Include separate evaluation for minimum Cart Count forecasting.

Create Actual vs Predicted charts.

SECTION 4 — SHORTAGE EARLY-WARNING RESULTS

Include:

Metric	Result
Actual shortage events	Computed
Detected events	Computed
Event recall	Computed
Median lead time	Computed
False alerts/day	Computed
Events warned ≥10 min early	Computed
Events warned ≥20 min early	Computed

Include a shortage warning timeline.

SECTION 5 — REPLENISHMENT RECOMMENDATIONS

For each recommended intervention, report:

Timestamp	Current Carts	Forecast Minimum	Top-Up	Action Time
Computed	Computed	Computed	Computed	Computed

Explain:

* Why replenishment was recommended.
* How many carts were recommended.
* When replenishment should occur.
* Whether the action could be completed before predicted shortage onset.

SECTION 6 — REPLENISHMENT POLICY BACKTEST

Compare observed historical cart availability against simulated policy outcomes.

Include:

Metric	Historical Baseline	Simulated Policy
Shortage events	Computed	Computed
Shortage duration	Computed	Computed
Minimum Cart Count	Computed	Computed
Replenishment actions	Unknown unless observed	Computed
Carts retrieved	Unknown unless observed	Computed

Calculate simulated shortage reduction.

Clearly distinguish observed outcomes from simulated estimates.

SECTION 7 — MODEL EXPLAINABILITY

Include:

* Top 15 features.
* Feature importance.
* Cart depletion slope analysis.
* Traffic surge analysis.
* Shortage risk interpretation.

Use SHAP only if appropriate and available.

Do not interpret predictive feature importance as causality.

SECTION 8 — OPERATIONAL RECOMMENDATIONS

Answer:

* When should employees replenish carts?
* How many carts should they retrieve?
* What signals indicate urgent shortages?
* What is the recommended alert threshold?
* How can false alerts be reduced?
* What additional data would improve the system?

SECTION 9 — LIMITATIONS

Discuss:

* Limited historical data.
* Single-store evaluation.
* Approximate record-based horizons.
* Unknown historical replenishment actions.
* Assumed replenishment lead time.
* Unknown reserve inventory.
* Forecast uncertainty.
* Simulation assumptions.
* Need for multi-store validation.

SECTION 10 — NEXT STEPS

Recommend:

1. More historical telemetry.
2. Replenishment action logging.
3. Multi-store validation.
4. Shadow deployment.
5. Alert calibration.
6. Operational A/B testing.

⸻

PART 12 — REQUIRED OUTPUT FILES

Create:

cart_forecasting/
│
├── data/
│   └── Thinh(2).csv
│
├── src/
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── target_builder.py
│   ├── forecasting.py
│   ├── shortage_risk.py
│   ├── replenishment_optimizer.py
│   ├── alert_engine.py
│   ├── backtesting.py
│   ├── simulator.py
│   └── report_generator.py
│
├── outputs/
│   ├── forecasts.csv
│   ├── shortage_predictions.csv
│   ├── shortage_events.csv
│   ├── replenishment_recommendations.csv
│   ├── model_comparison.csv
│   ├── policy_backtest.csv
│   │
│   ├── figures/
│   │   ├── historical_cart_traffic.png
│   │   ├── forecast_20m.png
│   │   ├── forecast_40m.png
│   │   ├── minimum_cart_forecast.png
│   │   ├── shortage_warning_timeline.png
│   │   ├── replenishment_timeline.png
│   │   ├── policy_comparison.png
│   │   ├── feature_importance.png
│   │   └── lead_time_distribution.png
│   │
│   └── Cart_Counting_Forecasting_Replenishment_Backtest_Report.docx
│
├── config.py
├── requirements.txt
└── main.py

⸻

PART 13 — IMPLEMENTATION REQUIREMENTS

Use:

* Python.
* pandas.
* numpy.
* scikit-learn.
* LightGBM.
* matplotlib.
* python-docx.

Implement modular code.

Include:

* Logging.
* Configuration management.
* Model persistence.
* Reproducible random seeds.
* Data validation.
* Leakage checks.
* Unit tests.
* Error handling.

Create tests for:

1. Rolling features.
2. Target alignment.
3. Temporal leakage.
4. Shortage event grouping.
5. Replenishment timing.
6. Pending replenishment tracking.
7. Alert deduplication.
8. Backtest consistency.

⸻

PART 14 — EXECUTION SEQUENCE

Follow this order.

Step 1 — Audit

Inspect the existing code and dataset.

Step 2 — Baseline

Reproduce the existing forecasting results.

Step 3 — Features

Implement unified advanced feature engineering.

Step 4 — Forecasting

Train and validate direct Cart Count and future-minimum forecasting models.

Step 5 — Risk

Estimate shortage probabilities.

Step 6 — Optimization

Implement timing-aware replenishment recommendations.

Step 7 — Backtesting

Replay historical telemetry chronologically.

Step 8 — Simulation

Evaluate the proposed replenishment policy under documented assumptions.

Step 9 — Reporting

Generate all CSV outputs, figures, and the final DOCX report.

Step 10 — Final Assessment

Explain:

* Which model was selected.
* Why it was selected.
* Whether the 20-minute or 40-minute horizon is more useful.
* How early shortages can be detected.
* How many carts should be replenished.
* Whether the simulated policy improves outcomes.
* What limitations prevent production deployment.

⸻

FINAL ACCEPTANCE CRITERIA

The project is complete only when:

1. The forecasting pipeline runs successfully.
2. All predictions are generated without future information leakage.
3. Chronological backtesting is completed.
4. Forecast accuracy is compared against baselines.
5. Shortage event recall and warning lead time are reported.
6. Replenishment recommendations include quantity and timing.
7. The replenishment policy is evaluated through simulation.
8. All results are saved to CSV.
9. All required figures are generated.
10. A professional DOCX report is generated.
11. The report explicitly distinguishes measured performance from simulated operational benefits.
12. All numerical conclusions are supported by computed results.

The final business question is:

How many carts should store employees replenish, when should they act, and how much shortage risk can the proposed policy reduce?

Begin with data and code auditing, then implement the full pipeline incrementally.