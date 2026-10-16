"""Chronological replenishment policy simulator.

The simulator extends the specified additive inventory model

    C_policy(t+k) = C_baseline(t+k) + sum_{j : tau_j + L_j <= t+k} Q_j

with two mandatory corrections: bay capacity absorbs carts that cannot
physically be stored, and the delivered boost decays with a configurable
retention half-life because extra carts are taken by customers and the bay
reverts towards its observed dynamics. Without the decay term the model would
grant a permanent, free inventory increase from a handful of trips.

It does **not** model customer behaviour feedback, cart return dynamics or
staff availability, and therefore produces *scenario estimates*, not measured
causal effects.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from config import RunConfig
from replenishment_optimizer import decide, policy_cost

LOGGER = logging.getLogger(__name__)


def simulate_policy(
    frame: pd.DataFrame,
    predicted_min: np.ndarray,
    lower_bound_min: np.ndarray,
    risk: np.ndarray,
    cfg: RunConfig,
    lambdas: dict,
    path_cols: list[str] | None = None,
) -> dict:
    """Replay telemetry and apply the replenishment policy event by event.

    Pending orders are tracked so the same top-up is never recommended twice
    while an earlier retrieval is still in transit.
    """
    df = frame.reset_index(drop=True)
    times = pd.to_datetime(df["decision_time"]).tolist()
    baseline = df["cart_current"].to_numpy(dtype=float)
    n = len(df)

    boost = 0.0
    pending: list[dict] = []
    decisions: list[dict] = []
    last_decision_time: pd.Timestamp | None = None
    sim_cart = np.empty(n, dtype=float)
    boost_track = np.zeros(n, dtype=float)
    cap = cfg.bay_capacity if cfg.bay_capacity is not None else np.inf
    halflife = max(float(cfg.retention_halflife_min), 1e-6)

    for i in range(n):
        ts = times[i]

        # 1. decay the surviving boost: extra carts are taken by customers and
        #    the bay reverts towards its observed dynamics.
        if i > 0 and boost > 0:
            dt = (ts - times[i - 1]).total_seconds() / 60.0
            boost *= 0.5 ** (max(dt, 0.0) / halflife)
            if boost < 0.5:
                boost = 0.0

        # 2. complete pending retrievals whose lead time has elapsed
        still_pending = []
        for order in pending:
            if order["effective_time"] <= ts:
                boost += order["quantity"]
                order["delivered_time"] = ts
            else:
                still_pending.append(order)
        pending = still_pending

        # 3. bay capacity absorbs carts that physically cannot be stored
        if np.isfinite(cap) and baseline[i] + boost > cap:
            boost = max(0.0, cap - baseline[i])

        current_sim = baseline[i] + boost
        sim_cart[i] = current_sim
        boost_track[i] = boost

        # 4. do not issue a new order while one is in transit or during cooldown
        if pending:
            continue
        if last_decision_time is not None and (
            (ts - last_decision_time).total_seconds() / 60.0 < cfg.cooldown_min
        ):
            continue
        if not np.isfinite(predicted_min[i]) or not np.isfinite(risk[i]):
            continue

        realised_boost = boost
        pred_min_sim = predicted_min[i] + realised_boost
        lb_sim = lower_bound_min[i] + realised_boost
        # Risk is shifted by the already-delivered carts: a boosted inventory
        # makes the forecast minimum less likely to breach RED.
        risk_sim = float(risk[i]) if realised_boost <= 0 else float(
            risk[i] * np.exp(-realised_boost / max(cfg.target_stock - cfg.red_threshold, 1))
        )

        mins_to_breach = np.inf
        if path_cols:
            path = df.loc[i, path_cols].to_numpy(dtype=float) + realised_boost
            step = (20.0 / 3.0)
            for k, v in enumerate(path, start=1):
                if v < cfg.red_threshold:
                    mins_to_breach = k * step
                    break

        headroom = cap - current_sim if np.isfinite(cap) else None
        dec = decide(
            ts=ts,
            current_cart=current_sim,
            predicted_min=pred_min_sim,
            lower_bound_min=lb_sim,
            risk=risk_sim,
            mins_to_breach=mins_to_breach,
            cfg=cfg,
            bay_headroom=headroom,
        )
        if dec is None:
            continue

        rec = asdict(dec)
        rec["baseline_cart"] = float(baseline[i])
        rec["origin_index"] = i
        rec["can_complete_before_breach"] = bool(mins_to_breach > cfg.lead_time_min)
        rec["actual_future_min_6"] = float(df.loc[i, "y_min_6"]) if "y_min_6" in df else np.nan
        decisions.append(rec)
        last_decision_time = ts
        pending.append(
            {
                "effective_time": dec.effective_time,
                "quantity": float(dec.quantity),
                "decision_time": ts,
            }
        )

    dec_df = pd.DataFrame(decisions)
    out = df.copy()
    out["sim_cart"] = sim_cart
    out["replenishment_boost"] = boost_track

    metrics = compare_series(
        out, baseline_col="cart_current", policy_col="sim_cart", cfg=cfg
    )
    n_trips = len(dec_df)
    carts_moved = int(dec_df["quantity"].sum()) if n_trips else 0
    metrics.update(
        {
            "n_replenishment_actions": n_trips,
            "total_carts_recommended": carts_moved,
            "avg_carts_per_action": float(dec_df["quantity"].mean()) if n_trips else np.nan,
            "trips_per_day": n_trips / max(metrics["span_days"], 1e-9),
            "carts_per_day": carts_moved / max(metrics["span_days"], 1e-9),
            "n_urgent": int((dec_df["urgency"] == "URGENT").sum()) if n_trips else 0,
            "n_capacity_constrained": int(dec_df["constrained"].sum()) if n_trips else 0,
            "pct_actionable_before_breach": float(
                dec_df["can_complete_before_breach"].mean() * 100
            )
            if n_trips
            else np.nan,
            "n_unnecessary_actions": int(
                (dec_df["actual_future_min_6"] >= cfg.target_stock).sum()
            )
            if n_trips and "actual_future_min_6" in dec_df
            else 0,
        }
    )
    metrics["policy_cost_J"] = policy_cost(
        shortage_severity=metrics["policy_severity_cart_records"],
        n_trips=n_trips,
        carts_moved=carts_moved,
        excess_inventory=metrics["policy_excess_cart_records"],
        lambdas=lambdas,
    )
    metrics["baseline_cost_J"] = policy_cost(
        shortage_severity=metrics["baseline_severity_cart_records"],
        n_trips=0,
        carts_moved=0,
        excess_inventory=metrics["baseline_excess_cart_records"],
        lambdas=lambdas,
    )
    return {"metrics": metrics, "decisions": dec_df, "series": out}


def _count_events(values: np.ndarray, red: float, recovery: int = 2) -> tuple[int, float]:
    """Number of independent sub-RED events and total records below RED."""
    below = values < red
    n_events = 0
    i = 0
    n = len(values)
    while i < n:
        if not below[i]:
            i += 1
            continue
        n_events += 1
        run = 0
        while i < n:
            if below[i]:
                run = 0
            else:
                run += 1
                if run >= recovery:
                    break
            i += 1
        i += 1
    return n_events, float(below.sum())


def compare_series(
    frame: pd.DataFrame, baseline_col: str, policy_col: str, cfg: RunConfig
) -> dict:
    b = frame[baseline_col].to_numpy(dtype=float)
    p = frame[policy_col].to_numpy(dtype=float)
    dur = frame["interval_duration"].to_numpy(dtype=float)
    red = cfg.red_threshold

    nb, rec_b = _count_events(b, red)
    npol, rec_p = _count_events(p, red)
    sev_b = float(np.sum(np.clip(red - b, 0, None)))
    sev_p = float(np.sum(np.clip(red - p, 0, None)))
    dur_b = float(np.sum(dur[b < red]))
    dur_p = float(np.sum(dur[p < red]))
    exc_b = float(np.sum(np.clip(b - cfg.target_stock, 0, None)))
    exc_p = float(np.sum(np.clip(p - cfg.target_stock, 0, None)))

    span_days = (
        pd.to_datetime(frame["decision_time"]).max()
        - pd.to_datetime(frame["decision_time"]).min()
    ).total_seconds() / 86400.0

    return {
        "span_days": max(span_days, 1e-9),
        "baseline_events": nb,
        "policy_events": npol,
        "baseline_records_below_red": rec_b,
        "policy_records_below_red": rec_p,
        "baseline_shortage_minutes": dur_b,
        "policy_shortage_minutes": dur_p,
        "baseline_severity_cart_records": sev_b,
        "policy_severity_cart_records": sev_p,
        "baseline_excess_cart_records": exc_b,
        "policy_excess_cart_records": exc_p,
        "baseline_min_cart": float(b.min()),
        "policy_min_cart": float(p.min()),
        "baseline_mean_cart": float(b.mean()),
        "policy_mean_cart": float(p.mean()),
        "event_reduction_pct": (nb - npol) / nb * 100 if nb else np.nan,
        "duration_reduction_pct": (dur_b - dur_p) / dur_b * 100 if dur_b > 0 else np.nan,
        "severity_reduction_pct": (sev_b - sev_p) / sev_b * 100 if sev_b > 0 else np.nan,
        "events_prevented": nb - npol,
    }
