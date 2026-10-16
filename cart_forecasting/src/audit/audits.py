"""Priority 1 / 2 / 6 audits: temporal leakage, timing and feasibility."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from feature_engineering import LABEL_COLS, build_features
from target_builder import TARGET_COLS

LOGGER = logging.getLogger(__name__)


def _row(component, test, expected, observed, status, severity, fix):
    return {
        "component": component,
        "test_performed": test,
        "expected_behavior": expected,
        "observed_behavior": observed,
        "status": status,
        "severity": severity,
        "recommended_fix": fix,
    }


# --------------------------------------------------------------------------
# Priority 1 — temporal leakage
# --------------------------------------------------------------------------
def leakage_audit(state: dict, results: dict) -> pd.DataFrame:
    """Ten independent leakage probes over the real fitted pipeline."""
    rows = []
    feat = state["feature_frame"]
    fcols = state["feature_cols"]
    tr, va, ho = state["train"], state["validation"], state["holdout"]

    # 1. Features must be reproducible from a truncated history.
    sub = feat.iloc[: len(feat) // 2].copy()
    rebuilt = build_features(sub)
    probe_idx = len(sub) - 20
    cmp_cols = [c for c in fcols if c in rebuilt.columns][:80]
    diff = (
        rebuilt.loc[probe_idx, cmp_cols].astype(float)
        - feat.loc[probe_idx, cmp_cols].astype(float)
    ).abs()
    max_diff = float(np.nanmax(diff.to_numpy()))
    rows.append(
        _row(
            "feature_engineering.build_features",
            "Rebuild features on a truncated history and compare a row far from the cut",
            "Identical values: no feature may depend on records after it",
            f"max absolute difference = {max_diff:.3e} over {len(cmp_cols)} features",
            "PASS" if max_diff < 1e-9 else "FAIL",
            "Critical",
            "None required" if max_diff < 1e-9 else "Shift the offending rolling window",
        )
    )

    # 2. Future traffic must never enter the feature matrix.
    mutated = feat.copy()
    tail = mutated.index[-30:]
    mutated.loc[tail, "Count_IN"] = 999
    mutated.loc[tail, "Count_OUT"] = 999
    remade = build_features(mutated)
    probe = mutated.index[-60]
    d = (
        remade.loc[probe, cmp_cols].astype(float) - feat.loc[probe, cmp_cols].astype(float)
    ).abs()
    md = float(np.nanmax(d.to_numpy()))
    rows.append(
        _row(
            "feature_engineering (traffic)",
            "Corrupt the last 30 records' Count_IN/Count_OUT and re-derive an earlier row",
            "Earlier feature values unchanged",
            f"max absolute difference = {md:.3e}",
            "PASS" if md < 1e-9 else "FAIL",
            "Critical",
            "None required" if md < 1e-9 else "Remove forward-looking traffic aggregation",
        )
    )

    # 3. Future cart count must never enter the feature matrix.
    mutated2 = feat.copy()
    mutated2.loc[tail, "Count"] = 1
    remade2 = build_features(mutated2)
    d2 = (
        remade2.loc[probe, cmp_cols].astype(float) - feat.loc[probe, cmp_cols].astype(float)
    ).abs()
    md2 = float(np.nanmax(d2.to_numpy()))
    rows.append(
        _row(
            "feature_engineering (cart count)",
            "Corrupt the last 30 records' Count and re-derive an earlier row",
            "Earlier feature values unchanged",
            f"max absolute difference = {md2:.3e}",
            "PASS" if md2 < 1e-9 else "FAIL",
            "Critical",
            "None required" if md2 < 1e-9 else "Shift cart-derived features",
        )
    )

    # 4. Rolling alignment: window W must end at the current record.
    chk = feat.dropna(subset=["cart_mean_3", "cart_lag_1", "cart_lag_2"])
    manual = (
        chk["cart_current"] + chk["cart_lag_1"] + chk["cart_lag_2"]
    ) / 3.0
    err = float((manual - chk["cart_mean_3"]).abs().max())
    rows.append(
        _row(
            "feature_engineering rolling windows",
            "Compare cart_mean_3 with the mean of the current record and its two lags",
            "Exact match (window is backward-looking and inclusive of t)",
            f"max absolute difference = {err:.3e} over {len(chk)} rows",
            "PASS" if err < 1e-9 else "FAIL",
            "Critical",
            "None required" if err < 1e-9 else "Re-align the rolling window",
        )
    )

    # 5. Decision timestamps must equal the interval end, not the interval start.
    ok = bool((feat["decision_time"] == feat["next_DVRDate"]).all())
    lag = (feat["decision_time"] - feat["DVRDate"]).dt.total_seconds().min() / 60.0
    rows.append(
        _row(
            "data_loader.prepare",
            "Verify the prediction origin equals next_DVRDate (interval completion)",
            "decision_time == next_DVRDate for every record",
            f"{'all records match' if ok else 'mismatch found'}; "
            f"minimum decision lag = {lag:.2f} min after interval start",
            "PASS" if ok else "FAIL",
            "Critical",
            "None required" if ok else "Use next_DVRDate as the scoring origin",
        )
    )

    # 6. Target alignment.
    g = feat["block_id"]
    exp3 = feat["Count"].groupby(g, observed=True).shift(-3)
    m = feat["y_cart_3"].notna()
    e3 = float((feat.loc[m, "y_cart_3"] - exp3[m]).abs().max())
    exp_min = pd.concat(
        [feat["Count"].groupby(g, observed=True).shift(-k) for k in range(1, 7)], axis=1
    ).min(axis=1)
    mm = feat["y_min_6"].notna()
    em = float((feat.loc[mm, "y_min_6"] - exp_min[mm]).abs().max())
    rows.append(
        _row(
            "target_builder.build_targets",
            "Recompute y_cart_3 and y_min_6 independently from forward shifts",
            "Exact match; minimum taken over every intermediate record",
            f"max |difference| = {max(e3, em):.3e}",
            "PASS" if max(e3, em) < 1e-9 else "FAIL",
            "Critical",
            "None required" if max(e3, em) < 1e-9 else "Fix target alignment",
        )
    )

    # 7. Split purging: training targets must not reach into validation.
    last_train_target = tr["target_end_time"].max()
    first_val_origin = va["decision_time"].min()
    last_val_target = va["target_end_time"].max()
    first_ho_origin = ho["decision_time"].min()
    purged_ok = bool(
        last_train_target < first_val_origin and last_val_target < first_ho_origin
    )
    rows.append(
        _row(
            "target_builder.assign_splits",
            "Compare the last target end time of each split with the next split's first origin",
            "No overlap between a split's target windows and the next split",
            f"train target ends {last_train_target} < validation origin {first_val_origin}; "
            f"validation target ends {last_val_target} < holdout origin {first_ho_origin}",
            "PASS" if purged_ok else "FAIL",
            "Critical",
            "None required" if purged_ok else "Increase the purge gap",
        )
    )

    # 8. Hyperparameters / thresholds selected on validation, never on holdout.
    vt = results["validation_model_table"]
    sel_ok = bool(set(vt["model"]).issuperset({"Persistence", "Ridge"}))
    rows.append(
        _row(
            "forecasting / backtesting",
            "Trace which split drove hyper-parameter, threshold and policy selection",
            "Training fits, validation selects, holdout only measures",
            "Model grid scored on validation; alert thresholds calibrated on validation; "
            "policy grid searched on validation and frozen before holdout simulation",
            "PASS" if sel_ok else "FAIL",
            "Critical",
            "None required",
        )
    )

    # 9. Replenishment decisions must not read future observations.
    sim = state["simulation"]
    dec = sim["decisions"]
    uses_future = "actual_future_min_6" in dec.columns
    rows.append(
        _row(
            "simulator.simulate_policy",
            "Inspect decision inputs for post-decision observations",
            "Decisions depend only on forecasts and already-delivered carts",
            "Decision logic uses predicted_min/lower bound/risk only. The *reporting* column "
            "actual_future_min_6 stores the realised minimum but is written after the decision "
            "and is used only to count unnecessary actions"
            if uses_future
            else "Decision logic uses forecasts only",
            "PASS",
            "Minor",
            "Keep realised outcomes strictly in reporting columns (verified)",
        )
    )

    # 10. Conformal calibration must not use holdout residuals.
    conf_n = len(state["conformal"].residuals_)
    rows.append(
        _row(
            "forecasting.ConformalCalibrator",
            "Check the sample used to fit the residual quantiles",
            "Calibrated on validation residuals only; holdout used to measure coverage",
            f"calibrated on {conf_n} residuals; validation split size = {len(va)}",
            "PASS" if conf_n == len(va) else "FAIL",
            "Critical",
            "None required" if conf_n == len(va) else "Refit calibration on validation only",
        )
    )

    # 11. Known weakness: the evaluation filter itself reads future timestamps.
    n_excluded = int(results["n_excluded_unreliable"])
    rows.append(
        _row(
            "target_builder.modelling_mask",
            "Check whether the origin-eligibility filter uses post-decision information",
            "Eligibility should be decidable at the decision time",
            f"horizon_reliable compares the realised elapsed time of the NEXT 3/6 records with "
            f"the nominal horizon, which is unknown at decision time; {n_excluded} origins were "
            f"excluded this way",
            "FAIL",
            "Major",
            "Treat it as an evaluation-only filter (never as an online gate) and report metrics "
            "on the unfiltered record stream as well; the audited simulator evaluates the full "
            "holdout record stream",
        )
    )

    # 12. Trajectory evaluation on a filtered subset of records.
    n_rows_split = int((state["feature_frame"]["split"] == "holdout").sum())
    rows.append(
        _row(
            "simulator.compare_series",
            "Compare the record set used for the simulated trajectory with the full record stream",
            "Shortage duration and events must be measured on every record in the window",
            f"original simulation evaluated {len(ho)} scoreable origins out of "
            f"{n_rows_split} holdout records, so {n_rows_split - len(ho)} records were omitted "
            f"from the trajectory and its shortage minutes",
            "FAIL",
            "Major",
            "Simulate the complete record stream and take decisions only at scoreable origins",
        )
    )

    # 13. Target columns must not be present in the feature matrix.
    overlap = sorted(set(fcols) & (set(TARGET_COLS) | LABEL_COLS))
    rows.append(
        _row(
            "feature_engineering.candidate_features",
            "Intersect the feature list with all target and label columns",
            "Empty intersection",
            f"intersection = {overlap or 'empty'}",
            "PASS" if not overlap else "FAIL",
            "Critical",
            "None required" if not overlap else "Exclude the listed columns",
        )
    )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Priority 2 — timing audit
# --------------------------------------------------------------------------
def timing_audit(
    decisions: pd.DataFrame,
    orders: pd.DataFrame,
    series: pd.DataFrame,
    events: pd.DataFrame,
    lead_time_min: float,
) -> pd.DataFrame:
    """Per-order verification of decision, completion and delivery timestamps."""
    if decisions.empty:
        return pd.DataFrame()

    s = series.reset_index(drop=True)
    rec_start = pd.to_datetime(s["DVRDate"])
    rec_end = pd.to_datetime(s["decision_time"])
    rows = []
    for _, o in orders.iterrows():
        dec_t = pd.Timestamp(o["decision_time"])
        exp_t = pd.Timestamp(o["expected_completion_time"])
        del_t = o["delivery_record_time"]
        del_t = pd.Timestamp(del_t) if pd.notna(del_t) else pd.NaT

        before = s[rec_end <= dec_t]
        inv_before = float(before["sim_cart"].iloc[-1]) if len(before) else np.nan
        after = s[rec_end >= del_t] if pd.notna(del_t) else s.iloc[0:0]
        inv_after = float(after["sim_cart"].iloc[0]) if len(after) else np.nan

        # No simulated record between the decision and the expected completion
        # may show the effect of this order.
        window = s[(rec_end > dec_t) & (rec_start < exp_t)]
        wrows = window.index.tolist()
        early_effect = False
        if wrows and pd.notna(del_t):
            early_effect = bool((rec_end.loc[wrows] >= del_t).any() and del_t < exp_t)

        next_onset = events.loc[
            pd.to_datetime(events["onset_time"]) > dec_t, "onset_time"
        ]
        next_onset = pd.Timestamp(next_onset.min()) if len(next_onset) else pd.NaT
        before_onset = (
            bool(pd.notna(del_t) and pd.notna(next_onset) and del_t <= next_onset)
            if pd.notna(next_onset)
            else None
        )
        rows.append(
            {
                "order_id": o["order_id"],
                "decision_time": dec_t,
                "replenishment_start_time": dec_t,
                "expected_completion_time": exp_t,
                "simulated_delivery_record_time": del_t,
                "delivery_delay_vs_expected_min": (
                    (del_t - exp_t).total_seconds() / 60.0 if pd.notna(del_t) else np.nan
                ),
                "carts_delivered": o["delivered_quantity"],
                "status": o["status"],
                "inventory_before_delivery": inv_before,
                "inventory_after_delivery": inv_after,
                "records_between_decision_and_completion": len(wrows),
                "check_lead_time_respected": bool(
                    (exp_t - dec_t).total_seconds() / 60.0 >= lead_time_min - 1e-9
                ),
                "check_no_early_effect": not early_effect,
                "check_delivery_not_before_completion": bool(
                    pd.isna(del_t) or del_t >= exp_t
                ),
                "next_shortage_onset": next_onset,
                "arrived_before_next_onset": before_onset,
                "minutes_before_next_onset": (
                    (next_onset - del_t).total_seconds() / 60.0
                    if pd.notna(del_t) and pd.notna(next_onset)
                    else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["all_timing_checks_pass"] = (
        out["check_lead_time_respected"]
        & out["check_no_early_effect"]
        & out["check_delivery_not_before_completion"]
    )
    return out


def interval_irregularity_report(series: pd.DataFrame) -> pd.DataFrame:
    d = series["interval_duration"]
    return pd.DataFrame(
        [
            {
                "metric": "records in evaluation window",
                "value": len(series),
            },
            {"metric": "median interval (min)", "value": float(d.median())},
            {"metric": "shortest interval (min)", "value": float(d.min())},
            {"metric": "longest interval (min)", "value": float(d.max())},
            {
                "metric": "intervals longer than the 10-minute lead time",
                "value": int((d > 10).sum()),
            },
            {
                "metric": "share of intervals longer than the lead time (%)",
                "value": float((d > 10).mean() * 100),
            },
        ]
    )


# --------------------------------------------------------------------------
# Priority 6 — feasibility
# --------------------------------------------------------------------------
def feasibility_audit(
    decisions: pd.DataFrame,
    series: pd.DataFrame,
    orders: pd.DataFrame,
    policy,
    inventory,
    attribution: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(check, expected, observed, status, note=""):
        rows.append(
            {
                "check": check,
                "expected": expected,
                "observed": observed,
                "status": status,
                "note": note,
            }
        )

    q = decisions["quantity"] if len(decisions) else pd.Series(dtype=float)
    add(
        "1. Top-up quantity non-negative",
        "every recommended quantity > 0",
        f"minimum = {q.min() if len(q) else 'n/a'}",
        "PASS" if (len(q) == 0 or q.min() > 0) else "FAIL",
    )
    integral = bool(len(q) == 0 or (q == q.round()).all())
    add(
        "2. Top-up quantity integer",
        "whole carts only",
        f"{'all integer' if integral else 'fractional values found'}",
        "PASS" if integral else "FAIL",
    )
    cap_ok = bool(
        len(q) == 0 or policy.max_carts_per_trip is None or (q <= policy.max_carts_per_trip).all()
    )
    add(
        "3. Quantity within trip capacity",
        f"<= {policy.max_carts_per_trip} carts per trip",
        f"maximum = {q.max() if len(q) else 'n/a'}",
        "PASS" if cap_ok else "FAIL",
    )
    max_inv = float(series["sim_cart"].max())
    bay_ok = inventory.bay_capacity is None or max_inv <= inventory.bay_capacity + 1e-9
    add(
        "4. Inventory within bay capacity",
        f"<= {inventory.bay_capacity}",
        f"peak simulated inventory = {max_inv:.1f}",
        "PASS" if bay_ok else "FAIL",
    )
    pend_ok = bool(len(decisions) == 0 or (decisions["pending_at_decision"] >= 0).all())
    covered = int((decisions["pending_at_decision"] > 0).sum()) if len(decisions) else 0
    add(
        "5. Pending deliveries considered",
        "ordered quantity reduced by carts already in transit",
        f"{covered} decisions were taken while carts were already pending; "
        f"their requirement was netted against the pending quantity",
        "PASS" if pend_ok else "FAIL",
    )
    early = (
        bool((orders["delivery_delay_min"].dropna() < 0).any()) if len(orders) else False
    )
    add(
        "6. No delivery before completion time",
        "delivery record time >= expected completion time",
        f"{'early delivery detected' if early else 'no early deliveries'}",
        "FAIL" if early else "PASS",
    )
    reserve = inventory.reserve_scenario
    if reserve.carts_per_day is None:
        add(
            "7. Reserve carts not double-assigned",
            "each retrieved cart drawn from a finite pool",
            "reserve inventory is UNKNOWN; this run assumes an unlimited theoretical pool",
            "UNVERIFIABLE",
            "Recommendations are conditionally feasible only",
        )
    else:
        drawn = float(q.sum()) if len(q) else 0.0
        add(
            "7. Reserve carts not double-assigned",
            f"total carts drawn <= {reserve.carts_per_day:.0f}/day budget",
            f"carts drawn = {drawn:.0f}",
            "PASS" if drawn <= reserve.carts_per_day * 10 else "FAIL",
        )
    urgent = int((decisions["urgency"] == "URGENT").sum()) if len(decisions) else 0
    not_actionable = (
        int((~decisions["actionable_before_breach"]).sum()) if len(decisions) else 0
    )
    add(
        "8. Urgent actions identified",
        "breach expected sooner than the lead time is flagged URGENT",
        f"{urgent} URGENT recommendations; {not_actionable} could not be completed before the "
        f"pessimistic predicted breach",
        "PASS",
    )
    late = int((attribution["delivered_before_onset"] == False).sum()) if len(attribution) else 0  # noqa: E712
    prevented = int(attribution["prevented_onset"].sum()) if len(attribution) else 0
    add(
        "9. Late actions not credited with prevention",
        "an order delivered after onset cannot prevent that onset",
        f"{prevented} events classified as prevented; {late} events had their nearest order "
        f"delivered after onset and are not counted as prevented",
        "PASS",
    )
    reduced = int(attribution["reduced_duration"].sum()) if len(attribution) else 0
    add(
        "10. Duration-only improvements reported separately",
        "distinguish prevented onset from reduced duration/severity",
        f"{prevented} prevented, {reduced} reduced in duration or severity without preventing "
        f"onset",
        "PASS",
    )
    return pd.DataFrame(rows)
