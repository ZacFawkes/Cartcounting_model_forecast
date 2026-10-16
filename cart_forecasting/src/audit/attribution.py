"""Priority 7 / 8: event-level action attribution and event-metric validation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from audit.inventory_model import group_shortage_events

LOGGER = logging.getLogger(__name__)

CONTRIBUTION_LOOKBACK_MIN = 120.0


def attribute_events(
    series: pd.DataFrame,
    baseline_events: pd.DataFrame,
    policy_events: pd.DataFrame,
    orders: pd.DataFrame,
    alerts: pd.DataFrame | None,
    red: float,
    warning_window_min: float = 60.0,
) -> pd.DataFrame:
    """Explain, event by event, what the simulated policy did and why.

    An event is only classified as *prevented* when the simulated trajectory
    stays at or above RED across the **entire** historical event window and at
    least one order was delivered before the historical onset. A high final
    inventory is never sufficient on its own.
    """
    s = series.reset_index(drop=True)
    sim = s["sim_cart"].to_numpy(dtype=float)
    surplus = s["surplus"].to_numpy(dtype=float)
    dur = s["interval_duration"].to_numpy(dtype=float)
    t_end = pd.to_datetime(s["decision_time"])

    warn_times = pd.Series(dtype="datetime64[ns]")
    if alerts is not None and len(alerts):
        warn_times = pd.to_datetime(
            alerts.loc[alerts["new_predictive_alert"], "alert_time"]
        ).sort_values()

    delivered = orders[orders["status"] == "COMPLETED"].copy() if len(orders) else orders
    if len(delivered):
        delivered["delivery_record_time"] = pd.to_datetime(delivered["delivery_record_time"])

    rows = []
    for _, ev in baseline_events.iterrows():
        i0, i1 = int(ev["start_index"]), int(ev["end_index"])
        onset = pd.Timestamp(ev["onset_time"])
        seg = sim[i0 : i1 + 1]
        seg_dur = dur[i0 : i1 + 1]
        below = seg < red

        residual_minutes = float(seg_dur[below].sum())
        residual_severity = float(np.sum(np.clip(red - seg, 0, None)))
        inv_before = float(sim[max(i0 - 1, 0)])
        surplus_at_onset = float(surplus[i0])

        # Warning
        cand = warn_times[
            (warn_times < onset)
            & (warn_times >= onset - pd.Timedelta(minutes=warning_window_min))
        ]
        first_warn = cand.min() if len(cand) else pd.NaT
        lead = (onset - first_warn).total_seconds() / 60.0 if pd.notna(first_warn) else np.nan

        # Orders delivered before onset that could still be contributing
        contributing, primary = [], None
        if len(delivered):
            window = delivered[
                (delivered["delivery_record_time"] <= onset)
                & (
                    delivered["delivery_record_time"]
                    >= onset - pd.Timedelta(minutes=CONTRIBUTION_LOOKBACK_MIN)
                )
            ].sort_values("delivery_record_time")
            contributing = window["order_id"].tolist()
            if len(window):
                primary = window.iloc[-1]

        late = None
        if len(delivered):
            after = delivered[delivered["delivery_record_time"] > onset]
            if len(after):
                late = after.sort_values("delivery_record_time").iloc[0]

        delivered_before_onset = bool(contributing) and surplus_at_onset > 0
        prevented = bool((~below).all() and delivered_before_onset)
        # Delayed onset: the simulated trajectory breaches later than the
        # historical onset but still inside the historical window.
        first_below = int(np.argmax(below)) if below.any() else None
        delayed = bool(below.any() and first_below is not None and first_below > 0)
        reduced_duration = bool(
            not prevented and residual_minutes < float(ev["duration_min"]) - 1e-9
        )
        reduced_severity = bool(
            not prevented
            and residual_severity < float(ev["severity_cart_records"]) - 1e-9
        )

        if prevented:
            outcome = "PREVENTED_ONSET"
        elif delayed and reduced_duration:
            outcome = "DELAYED_ONSET"
        elif reduced_duration:
            outcome = "REDUCED_DURATION"
        elif reduced_severity:
            outcome = "REDUCED_SEVERITY"
        else:
            outcome = "UNCHANGED"

        if prevented:
            explanation = (
                f"Orders {','.join(contributing)} delivered before {onset:%m-%d %H:%M} left "
                f"{surplus_at_onset:.1f} surplus carts in the bay; the simulated count stayed at "
                f"or above RED for all {ev['n_records']} records of the historical event "
                f"(simulated minimum {seg.min():.1f})."
            )
        elif outcome == "UNCHANGED" and not contributing:
            explanation = (
                "No replenishment carts were in the bay before onset"
                + (
                    f"; the nearest order arrived at {late['delivery_record_time']:%m-%d %H:%M}, "
                    f"after onset, so it cannot be credited with prevention."
                    if late is not None
                    else "; no order was issued for this event."
                )
            )
        else:
            explanation = (
                f"Surplus of {surplus_at_onset:.1f} carts at onset reduced the shortage from "
                f"{ev['duration_min']:.1f} to {residual_minutes:.1f} minutes and severity from "
                f"{ev['severity_cart_records']:.1f} to {residual_severity:.1f} cart-records, "
                f"but the simulated count still fell below RED (minimum {seg.min():.1f})."
            )

        rows.append(
            {
                "event_id": int(ev["event_id"]),
                "actual_onset": onset,
                "actual_end": pd.Timestamp(ev["end_time"]),
                "actual_minimum_cart": float(ev["min_cart"]),
                "actual_duration_minutes": float(ev["duration_min"]),
                "actual_severity_cart_records": float(ev["severity_cart_records"]),
                "first_warning_time": first_warn,
                "warning_lead_minutes": lead,
                "warning_generated": bool(pd.notna(first_warn)),
                "replenishment_order_id": primary["order_id"] if primary is not None else None,
                "decision_time": primary["decision_time"] if primary is not None else pd.NaT,
                "delivery_time": (
                    primary["delivery_record_time"] if primary is not None else pd.NaT
                ),
                "recommended_quantity": (
                    float(primary["quantity"]) if primary is not None else np.nan
                ),
                "delivered_before_onset": delivered_before_onset,
                "effective_inventory_before_onset": inv_before,
                "surplus_at_onset": surplus_at_onset,
                "simulated_minimum_in_window": float(seg.min()),
                "prevented_onset": prevented,
                "delayed_onset": bool(outcome == "DELAYED_ONSET"),
                "reduced_duration": reduced_duration,
                "reduced_severity": reduced_severity,
                "residual_shortage_minutes": residual_minutes,
                "residual_severity_cart_records": residual_severity,
                "outcome": outcome,
                "contributing_previous_orders": ";".join(contributing),
                "n_contributing_orders": len(contributing),
                "attribution_explanation": explanation,
            }
        )

    att = pd.DataFrame(rows)

    # New shortage events created by the policy (should normally be none).
    new_rows = []
    for _, pe in policy_events.iterrows():
        overlap = baseline_events[
            (baseline_events["start_index"] <= pe["end_index"])
            & (baseline_events["end_index"] >= pe["start_index"])
        ]
        if len(overlap) == 0:
            new_rows.append(
                {
                    "event_id": f"NEW-{int(pe['event_id'])}",
                    "actual_onset": pd.Timestamp(pe["onset_time"]),
                    "actual_end": pd.Timestamp(pe["end_time"]),
                    "actual_minimum_cart": float(pe["min_cart"]),
                    "actual_duration_minutes": float(pe["duration_min"]),
                    "outcome": "NEW_EVENT_UNDER_POLICY",
                    "attribution_explanation": (
                        "Shortage present in the simulated trajectory with no overlapping "
                        "historical event."
                    ),
                }
            )
    if new_rows:
        att = pd.concat([att, pd.DataFrame(new_rows)], ignore_index=True)
    return att


def attribution_summary(att: pd.DataFrame) -> dict:
    hist = att[~att["event_id"].astype(str).str.startswith("NEW-")]
    n = len(hist)
    return {
        "historical_events": n,
        "prevented_onset": int(hist["prevented_onset"].sum()) if n else 0,
        "delayed_onset": int(hist["delayed_onset"].sum()) if n else 0,
        "reduced_duration_only": int(
            (hist["reduced_duration"] & ~hist["prevented_onset"]).sum()
        )
        if n
        else 0,
        "reduced_severity_only": int(
            (hist["reduced_severity"] & ~hist["reduced_duration"] & ~hist["prevented_onset"]).sum()
        )
        if n
        else 0,
        "unchanged": int((hist["outcome"] == "UNCHANGED").sum()) if n else 0,
        "warned": int(hist["warning_generated"].sum()) if n else 0,
        "delivered_before_onset": int(hist["delivered_before_onset"].sum()) if n else 0,
        "traceable_prevention": int(
            (hist["prevented_onset"] & (hist["n_contributing_orders"] > 0)).sum()
        )
        if n
        else 0,
        "new_events_under_policy": int(
            att["event_id"].astype(str).str.startswith("NEW-").sum()
        ),
        "residual_shortage_minutes": float(hist["residual_shortage_minutes"].sum()) if n else 0.0,
    }


# --------------------------------------------------------------------------
# Priority 8 — event grouping validation
# --------------------------------------------------------------------------
def event_grouping_validation(
    series: pd.DataFrame, red: float, recovery_variants=(1, 2, 3)
) -> pd.DataFrame:
    """Show both trajectories under identical grouping rules and variants."""
    s = series.reset_index(drop=True)
    start = pd.to_datetime(s["DVRDate"]).to_numpy()
    end = pd.to_datetime(s["decision_time"]).to_numpy()
    dur = s["interval_duration"].to_numpy(dtype=float)
    rows = []
    for rec in recovery_variants:
        for label, col in (("historical baseline", "baseline_cart"), ("simulated policy", "sim_cart")):
            vals = s[col].to_numpy(dtype=float)
            ev = group_shortage_events(vals, start, end, dur, red, rec)
            below = vals < red
            rows.append(
                {
                    "recovery_records": rec,
                    "trajectory": label,
                    "events": len(ev),
                    "records_below_red": int(below.sum()),
                    "shortage_minutes": float(dur[below].sum()),
                    "severity_cart_records": float(np.sum(np.clip(red - vals, 0, None))),
                    "min_cart": float(vals.min()),
                    "median_event_duration_min": float(ev["duration_min"].median())
                    if len(ev)
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    piv = out.pivot_table(
        index="recovery_records", columns="trajectory", values="events"
    ).reset_index()
    piv.columns.name = None
    piv = piv.rename(
        columns={
            "historical baseline": "baseline_events",
            "simulated policy": "policy_events",
        }
    )
    piv["event_reduction_pct"] = (
        (piv["baseline_events"] - piv["policy_events"]) / piv["baseline_events"] * 100
    )
    return out.merge(piv[["recovery_records", "event_reduction_pct"]], on="recovery_records")
