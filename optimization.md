# TASK: AUDIT AND IMPROVE CART REPLENISHMENT SIMULATION

## ROLE

Act as a Senior Data Scientist, Machine Learning Engineer,
Operations Research Specialist, and Simulation Validation Engineer.

You are working on an existing Cart Availability Forecasting,
Shortage Early-Warning, and Replenishment Optimization project.

The project already includes:

- Cart Count forecasting
- Future minimum Cart Count forecasting
- Shortage probability estimation
- Alert generation
- Replenishment quantity recommendations
- Replenishment timing recommendations
- Historical backtesting
- Inventory simulation
- Automated DOCX reporting

Your task is NOT to rebuild the entire project.

Your primary responsibility is to AUDIT, VALIDATE, and IMPROVE
the existing replenishment simulation and historical backtesting.

Everything must be performed OFFLINE using Python and
the existing historical telemetry.

Do NOT require real-world deployment, employee participation,
live camera integration, or A/B testing at this stage.


============================================================
1. PROJECT INPUTS
============================================================

Inspect all existing project files, especially:

1. Cart_Counting_Forecasting_Replenishment_Backtest_Report.docx
2. Existing forecasting source code
3. Existing replenishment optimizer
4. Existing simulation engine
5. Existing backtesting implementation
6. Historical telemetry dataset
7. Existing CSV prediction and recommendation outputs

Do not assume that the reported results are correct.

Reproduce them from the source code and data.

Preserve the original implementation as a benchmark.

Do not overwrite existing results.


============================================================
2. CURRENT REPORTED BASELINE
============================================================

The current report contains:

Minimum Cart Forecast MAE (+40 min): 3.16 carts

Shortage Risk PR-AUC: 0.843

Shortage Event Recall: 53.8%

Detected Shortage Events: 7/13

Median Warning Lead Time: 33.4 minutes

False Alerts per Day: 4.53

Recommended Replenishment Actions: 14

Average Top-Up: 10.5 carts

Simulated Shortage Event Reduction: 92.3%

Simulated Shortage Severity Reduction: 98.9%

These are reported results, not independently verified facts.

The main concern is:

The warning engine detects only 7 of 13 shortage events,
while the replenishment simulation claims to prevent
12 of 13 events.

This difference may be explainable, but it must be audited.

Determine whether the simulated improvement is supported
by the implementation and its assumptions.


============================================================
3. PRIORITY 1 — TEMPORAL DATA LEAKAGE AUDIT
============================================================

Inspect the complete data pipeline.

Each telemetry record represents:

[DVRDate, next_DVRDate)

Count_IN and Count_OUT represent customer traffic
aggregated over that interval.

The record must not be considered fully available
before the interval has completed.

Verify that:

1. Forecast features use only available historical data.
2. Future Count_IN and Count_OUT are never used.
3. Future Cart Count is never used in model inputs.
4. Rolling features are correctly shifted and aligned.
5. Prediction timestamps represent actual decision times.
6. Forecast targets are correctly aligned.
7. Training and validation target windows do not overlap.
8. Hyperparameters are not selected using holdout results.
9. Replenishment decisions never use future observations.
10. Model calibration never uses future holdout residuals.

Generate:

simulation_audit/data_leakage_audit.csv

For each issue, include:

- Component
- Test performed
- Expected behavior
- Observed behavior
- PASS / FAIL
- Severity
- Recommended fix

Fix confirmed leakage issues.

Create automated unit tests to prevent regression.


============================================================
4. PRIORITY 2 — REPLENISHMENT TIMING AUDIT
============================================================

The current replenishment lead time is assumed to be:

10 minutes.

A replenishment action issued at time T must not
increase cart availability immediately.

Example:

Decision time: 10:00

Recommended quantity: 15 carts

Lead time: 10 minutes

Effective replenishment time: 10:10

The simulated inventory must remain unchanged by
this action before 10:10.

Verify:

1. Decision timestamp.
2. Replenishment start timestamp.
3. Expected completion timestamp.
4. Actual simulation delivery timestamp.
5. Number of carts delivered.
6. Whether the action arrived before shortage onset.
7. Whether the action was operationally feasible.

Pay special attention to irregular telemetry intervals.

Do not round replenishment completion to an earlier
record timestamp.

Use actual timestamps for delivery events.

If the simulator uses discrete records, explicitly
document how actions are applied between records.


============================================================
5. PRIORITY 3 — PENDING ORDER AND DOUBLE-COUNTING AUDIT
============================================================

The simulator must maintain a pending replenishment queue.

Example:

10:00:
Recommend +15 carts.

10:07:
A new telemetry record arrives.

The previous +15 carts have not yet been delivered.

The optimizer must know that 15 carts are already pending.

It must not blindly recommend another +15 carts.

Implement or verify a pending-order state containing:

- Order ID
- Decision timestamp
- Quantity
- Expected completion timestamp
- Status
- Delivered quantity

Supported statuses:

PENDING
COMPLETED
CANCELLED

At every scoring timestamp:

1. Process completed replenishment orders.
2. Update simulated inventory.
3. Update remaining pending orders.
4. Generate forecasts.
5. Evaluate shortage risk.
6. Calculate additional replenishment requirements.
7. Avoid duplicate replenishment recommendations.

Do not count the same replenishment twice.

Create unit tests for overlapping and delayed orders.


============================================================
6. PRIORITY 4 — FIXED-ACTION REPLAY VS CLOSED-LOOP SIMULATION
============================================================

Determine whether the current simulator is:

A. Fixed-action replay

or

B. Closed-loop simulation.

A fixed-action replay generates recommendations from
historical observations and applies them to a modified
historical inventory trajectory.

A closed-loop simulation updates the simulated state
after each intervention and uses that updated state
to determine subsequent actions.

These are different evaluation methods.

Audit the existing implementation and identify
which approach it actually uses.

Do not label fixed-action replay as closed-loop simulation.

Implement both evaluation modes where technically justified.

------------------------------------------------------------
MODE A — FIXED-ACTION REPLAY
------------------------------------------------------------

Use the existing historical forecasts and recommendations.

Replay replenishment actions with correct timing.

Apply delivered carts to the historical inventory trajectory.

Enforce capacity limits.

Track all interventions.

This mode provides a transparent scenario estimate.

Clearly state that the historical trajectory may already
contain unobserved employee replenishment activity.

------------------------------------------------------------
MODE B — CLOSED-LOOP SIMULATION
------------------------------------------------------------

Implement a stateful inventory simulator.

At each simulation step:

1. Update the simulated inventory.
2. Process pending replenishment deliveries.
3. Recalculate state-dependent features.
4. Generate new predictions when valid.
5. Evaluate shortage risk.
6. Optimize replenishment actions.
7. Schedule new replenishment orders.
8. Advance the simulation clock.

IMPORTANT:

Do not simply replace historical Cart Count with
simulated Cart Count and assume the resulting
feature distribution remains valid.

Historical traffic may be replayed as an exogenous
input only under an explicit assumption that
replenishment does not alter customer traffic.

A defensible state-transition model is required
to simulate cart availability after interventions.

If no reliable transition model can be identified
from the available data, do not fabricate one.

In that case:

- Implement a transparent simplified simulator.
- Label it as assumption-driven.
- Run sensitivity analysis.
- Explain why closed-loop results remain uncertain.

Never claim causal impact from offline simulation alone.


============================================================
7. PRIORITY 5 — INVENTORY TRANSITION VALIDATION
============================================================

Audit the inventory update equation.

A simplified candidate is:

SimulatedCart(t)
=
BaselineCart(t)
+
EffectiveReplenishment(t)
-
AdditionalCartLosses(t)

However, do not assume this equation is complete.

Investigate:

- Customer cart pickup.
- Customer cart returns.
- Cart retrieval.
- Existing employee replenishment.
- Cart redistribution.
- Cart capacity.
- Delayed replenishment.
- Potential measurement noise.

Count_IN and Count_OUT are customer traffic counts.

They are NOT equivalent to cart pickup and cart returns.

Do not assume:

1 customer entering = 1 cart removed.

Do not assume:

1 customer leaving = 1 cart returned.

If cart pickup and return rates cannot be identified,
treat them as uncertain parameters.

Evaluate multiple plausible scenarios.

For example:

LOW cart usage
MEDIUM cart usage
HIGH cart usage

Do not present arbitrary assumptions as measured facts.

Validate inventory conservation where applicable.

Enforce:

0 <= SimulatedCart <= BayCapacity

Track carts entering and leaving the simulated system.

Do not create carts from nothing.

If reserve inventory is unknown, report replenishment
recommendations as conditionally feasible.


============================================================
8. PRIORITY 6 — REPLENISHMENT FEASIBILITY
============================================================

Audit all operational constraints.

Current assumptions:

RED_THRESHOLD = 15

AMBER_THRESHOLD = 20

TARGET_STOCK = 28

REPLENISHMENT_LEAD_TIME = 10 minutes

MAX_CARTS_PER_TRIP = 20

BAY_CAPACITY = 60

Verify:

1. Top-up quantity is non-negative.
2. Top-up quantity is an integer.
3. Quantity does not exceed trip capacity.
4. Inventory does not exceed bay capacity.
5. Pending deliveries are considered.
6. Replenishment cannot happen before completion.
7. The same reserve carts cannot be assigned twice.
8. Urgent actions are correctly identified.
9. Actions arriving after shortage onset are not
   counted as preventing the onset.
10. Actions that reduce shortage duration but do not
    prevent onset are reported separately.

If reserve inventory is unknown, do not assume
unlimited physical availability without labeling it.

Create a feasibility audit table.


============================================================
9. PRIORITY 7 — EVENT-LEVEL ACTION ATTRIBUTION
============================================================

This is a mandatory deliverable.

For every historical shortage event, determine:

1. Was the event predicted?
2. Was an alert issued?
3. Was a replenishment action recommended?
4. When was the action recommended?
5. When would replenishment be completed?
6. How many carts would be delivered?
7. Did the action arrive before shortage onset?
8. Did the simulated policy prevent the event?
9. Did it only reduce shortage duration?
10. Was the outcome affected by an earlier replenishment?

Generate:

simulation_audit/event_action_attribution.csv

Required columns:

event_id
actual_onset
actual_end
actual_minimum_cart
actual_duration_minutes

first_warning_time
warning_lead_minutes
warning_generated

replenishment_order_id
decision_time
delivery_time
recommended_quantity

delivered_before_onset
effective_inventory_before_onset

prevented_onset
reduced_duration
residual_shortage_minutes

contributing_previous_orders
attribution_explanation

Every prevented event must have a traceable explanation.

Do not claim an event was prevented merely because
the final simulated inventory is above RED.

Evaluate the entire event trajectory.

Distinguish:

- Prevented onset
- Delayed onset
- Reduced severity
- Reduced duration
- Unchanged event

Also report new shortage events that appear under
the simulated policy, if any.


============================================================
10. PRIORITY 8 — VALIDATE SHORTAGE EVENT METRICS
============================================================

Audit the shortage event grouping logic.

Verify:

- RED threshold semantics.
- Event onset.
- Event end.
- Recovery conditions.
- Event merging.
- Short interruptions.
- Missing intervals.
- Irregular timestamps.

Use the same event definition for historical
and simulated trajectories.

Calculate:

Historical shortage events

Simulated shortage events

Shortage duration

Shortage severity

Minimum Cart Count

Event-level reduction

Duration reduction

Severity reduction

Do not calculate event reduction using inconsistent
event grouping rules.

Ensure that event matching does not assign multiple
simulated outcomes incorrectly to one historical event.


============================================================
11. PRIORITY 9 — SIMULATION SENSITIVITY ANALYSIS
============================================================

The current report claims:

92.3% simulated shortage event reduction.

Determine how sensitive this result is to
operational assumptions.

Evaluate:

Replenishment lead time:

5, 10, 15, 20 minutes

Target stock:

20, 25, 28, 30 carts

Maximum carts per trip:

10, 15, 20 carts

Replenishment availability:

Unlimited theoretical reserve
Limited reserve
Delayed reserve availability

Cart usage assumptions:

Low
Medium
High

Inventory retention assumptions:

Short
Medium
Long

Use plausible ranges and clearly document them.

Do not treat the most optimistic scenario
as the expected operational outcome.

Report:

Best-case scenario

Base-case scenario

Conservative scenario

For each scenario calculate:

- Shortage event reduction
- Shortage duration reduction
- Shortage severity reduction
- Retrieval trips
- Carts moved
- Operational cost

Generate a sensitivity heatmap.

Identify which assumptions have the greatest
impact on estimated shortage reduction.


============================================================
12. PRIORITY 10 — POLICY RE-OPTIMIZATION
============================================================

Only after completing the simulation audit,
re-evaluate the replenishment policy.

Do not change the forecasting model unless
the audit identifies a specific need.

Evaluate candidate policies based on:

- Risk threshold
- Target stock
- Replenishment quantity
- Replenishment timing
- Cooldown
- Lead time
- Pending orders

Use the existing cost-sensitive objective:

J =
lambda_shortage * ShortagePenalty
+
lambda_trips * RetrievalTrips
+
lambda_carts * CartsMoved
+
lambda_excess * ExcessInventoryPenalty

Use validation data for policy selection.

Freeze the policy before evaluation.

Do not optimize against the final holdout.

Because the current holdout has already been reviewed,
treat further experiments on it as exploratory.

A genuinely untouched future dataset is required
for independent confirmation.

Do not optimize solely for maximum shortage reduction.

A policy that eliminates nearly all shortages by
retrieving excessive carts may be operationally poor.

Find a practical balance between shortage prevention
and employee workload.


============================================================
13. REQUIRED BEFORE/AFTER RESULTS
============================================================

Create:

simulation_audit/before_after_comparison.csv

Include:

Metric
Original implementation
Audited implementation
Difference
Explanation

Required metrics:

Shortage events

Shortage duration

Shortage severity

Simulated event reduction

Simulated duration reduction

Simulated severity reduction

Retrieval trips

Carts moved

Average carts per trip

Unnecessary replenishment actions

Actions arriving before shortage onset

Actions arriving after shortage onset

Operational cost

Do not fabricate improvements.

If the audited simulator produces worse results,
report them honestly.

The objective is reliable evaluation,
not artificially higher shortage reduction.


============================================================
14. REQUIRED VISUALIZATIONS
============================================================

Generate:

1. Historical vs simulated inventory timeline.

2. Replenishment decision and delivery timeline.

3. Pending replenishment queue visualization.

4. Event-level shortage prevention timeline.

5. Fixed-action replay vs closed-loop comparison.

6. Lead-time sensitivity chart.

7. Target-stock sensitivity chart.

8. Shortage reduction vs retrieval workload.

9. Before/after simulation comparison.

10. Operational cost comparison.

All figures must use actual computed results.

Clearly distinguish:

Observed data
Model predictions
Simulated outcomes


============================================================
15. PROFESSIONAL DOCX REPORT
============================================================

Generate:

Cart_Counting_Simulation_Audit_Report.docx

Use python-docx.

The report must be detailed, technical,
and understandable to business stakeholders.

Include the following sections.


SECTION 1 — EXECUTIVE SUMMARY

Answer:

- Is the original 92.3% shortage reduction credible?
- Were any implementation errors found?
- What changed after the audit?
- What is the revised simulated shortage reduction?
- How sensitive are results to operational assumptions?
- Is the system ready for shadow deployment?


SECTION 2 — ORIGINAL IMPLEMENTATION

Explain:

- Forecasting architecture
- Risk model
- Alert engine
- Replenishment policy
- Inventory simulator
- Original reported results


SECTION 3 — AUDIT FINDINGS

Create a table:

Issue
Severity
Evidence
Impact
Resolution
Status

Distinguish:

Critical errors
Major assumptions
Minor issues
Validated components


SECTION 4 — TEMPORAL LEAKAGE AUDIT

Explain all timestamp and feature alignment checks.

Document every confirmed issue and fix.


SECTION 5 — REPLENISHMENT TIMING

Show:

Decision time
Completion time
Delivery quantity
Inventory before delivery
Inventory after delivery

Include an example timeline.


SECTION 6 — INVENTORY SIMULATION

Document the exact state-transition equations.

Explain:

- Inventory conservation
- Pending replenishments
- Capacity constraints
- Reserve inventory assumptions
- Customer traffic assumptions
- Model limitations


SECTION 7 — FIXED-ACTION VS CLOSED-LOOP

Explain which simulation mode is implemented.

Compare both methods if defensible.

Discuss why their results differ.


SECTION 8 — EVENT-LEVEL AUDIT

Include all historical shortage events.

For each event, report:

- Actual shortage onset
- Warning lead time
- Replenishment decision
- Delivery time
- Quantity
- Simulated outcome
- Attribution explanation

Include detailed case studies.


SECTION 9 — SENSITIVITY ANALYSIS

Show how shortage reduction changes under:

- Different lead times
- Different target stocks
- Different reserve assumptions
- Different cart usage assumptions

Identify fragile assumptions.


SECTION 10 — BEFORE/AFTER RESULTS

Compare the original simulator
against the audited simulator.

Explain all major differences.


SECTION 11 — OPERATIONAL IMPLICATIONS

Answer:

1. How many carts should employees replenish?
2. When should replenishment begin?
3. How much lead time is required?
4. How many retrieval trips are expected?
5. What happens when replenishment is delayed?
6. What happens when reserve carts are unavailable?
7. Which recommendations are robust across scenarios?


SECTION 12 — LIMITATIONS

Explicitly discuss:

- Single-site data
- Limited historical period
- Unknown historical replenishment actions
- Unknown reserve inventory
- Unknown cart pickup and return behavior
- Assumed replenishment lead time
- Simulation model uncertainty
- Lack of observed intervention outcomes


SECTION 13 — FINAL RECOMMENDATION

Provide a clear conclusion:

A. Simulation validated sufficiently for shadow mode.

B. Simulation requires additional offline improvements.

C. Simulation assumptions are too weak for reliable
   operational benefit estimation.

Explain the decision using evidence.


============================================================
16. REQUIRED OUTPUT STRUCTURE
============================================================

simulation_audit/
│
├── data_leakage_audit.csv
├── timing_audit.csv
├── pending_orders_audit.csv
├── inventory_conservation_audit.csv
├── event_action_attribution.csv
├── historical_vs_simulated.csv
├── fixed_action_replay.csv
├── closed_loop_simulation.csv
├── sensitivity_analysis.csv
├── policy_comparison.csv
├── before_after_comparison.csv
│
├── figures/
│   ├── inventory_timeline.png
│   ├── replenishment_timing.png
│   ├── pending_orders.png
│   ├── event_attribution.png
│   ├── replay_vs_closed_loop.png
│   ├── lead_time_sensitivity.png
│   ├── target_stock_sensitivity.png
│   ├── shortage_vs_workload.png
│   └── before_after_comparison.png
│
└── Cart_Counting_Simulation_Audit_Report.docx


============================================================
17. ENGINEERING REQUIREMENTS
============================================================

Use Python.

Prefer:

pandas
numpy
scikit-learn
matplotlib
python-docx

Preserve the existing project structure.

Do not rewrite working components unnecessarily.

Create modular simulation functions.

Include:

- Unit tests
- Reproducible configurations
- Deterministic execution
- Logging
- Input validation
- Error handling

All numerical results must be reproducible.

Add tests for:

1. Replenishment delivery timing.
2. Pending order handling.
3. Duplicate order prevention.
4. Inventory capacity.
5. Shortage event grouping.
6. Event attribution.
7. No future information leakage.
8. Fixed-action replay consistency.
9. Closed-loop state updates.
10. Inventory conservation.


============================================================
18. EXECUTION ORDER
============================================================

Execute the project in this order:

STEP 1:
Inspect the existing source code and report.

STEP 2:
Reproduce the reported baseline metrics.

STEP 3:
Audit temporal leakage.

STEP 4:
Audit replenishment timing.

STEP 5:
Audit pending orders and double counting.

STEP 6:
Identify the existing simulation method.

STEP 7:
Validate inventory transition equations.

STEP 8:
Audit every historical shortage event.

STEP 9:
Fix confirmed implementation errors.

STEP 10:
Run fixed-action replay.

STEP 11:
Implement and evaluate closed-loop simulation
only if defensible.

STEP 12:
Perform sensitivity analysis.

STEP 13:
Re-evaluate replenishment policies.

STEP 14:
Generate before/after comparisons.

STEP 15:
Generate all figures and CSV files.

STEP 16:
Generate the professional DOCX report.


============================================================
19. FINAL ACCEPTANCE CRITERIA
============================================================

The task is complete only when:

1. Original results have been reproduced or discrepancies explained.

2. All temporal leakage checks have been completed.

3. Replenishment lead times are handled correctly.

4. Pending orders cannot be double-counted.

5. Inventory capacity constraints are enforced.

6. Every simulated prevented shortage is traceable
   to a specific replenishment action or previous delivery.

7. Fixed-action and closed-loop simulation are
   correctly distinguished.

8. Simulation assumptions are documented.

9. Sensitivity analysis is completed.

10. Original and audited results are compared.

11. All CSV files and figures are generated.

12. A professional DOCX report is generated.

13. No simulated outcome is presented as an
    experimentally verified business improvement.

14. The final report answers:

    "Is the estimated shortage reduction credible,
     under which assumptions, and what operational
     actions are justified by the available evidence?"


============================================================
20. MOST IMPORTANT INSTRUCTION
============================================================

DO NOT attempt to make the shortage reduction percentage
as high as possible.

Your goal is to determine whether the current reported
92.3% simulated shortage event reduction is trustworthy.

A lower but defensible improvement estimate is more
valuable than an unrealistic 90%+ improvement.

Prioritize:

Correctness
Reproducibility
Operational feasibility
Interpretability
Honest uncertainty reporting

Do not require any real-world deployment.

Complete the entire audit and report generation
using the existing historical dataset.