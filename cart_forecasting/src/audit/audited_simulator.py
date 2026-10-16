"""Audited replenishment simulator: fixed-action replay and closed-loop modes.

MODE A — FIXED-ACTION REPLAY
    Forecasts and shortage risks are produced from the *observed* telemetry.
    Delivered carts are added to the observed trajectory with correct timing
    and capacity handling. Transparent, but the forecasts never see the state
    the policy itself created.

MODE B — CLOSED-LOOP SIMULATION
    The simulated cart series is fed back into feature construction at every
    step, so forecasts, risks and the next decision are all computed from the
    state the policy produced. Customer traffic is replayed as an exogenous
    input under the explicit assumption that replenishment does not change
    customer arrivals.

Neither mode measures a causal effect. Both are scenario estimates.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from audit.inventory_model import (
    InventoryConfig,
    ConservationLedger,
    apply_capacity,
    consume_surplus,
    observed_inventory,
    trajectory_metrics,
)
from audit.order_book import OrderBook
from feature_engineering import apply_value_clipper, build_features

LOGGER = logging.getLogger(__name__)

SLICE_LOOKBACK = 15  # records; max feature dependency is 9 lags


# --------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------
class ReplayPredictor:
    """Mode A: cached forecasts from the observed trajectory.

    The surplus is projected forward over the planning horizon using *recent*
    traffic only (never future traffic), then added to the baseline forecast.
    Shortage risk is recomputed from the conformal residual distribution at the
    shifted forecast rather than by an ad-hoc shrink factor.
    """

    mode = "fixed_action_replay"

    def __init__(self, pred_min_6, pred_min_3, pred_cart_3, pred_cart_6, conformal, red):
        self.pred_min_6 = np.asarray(pred_min_6, dtype=float)
        self.pred_min_3 = np.asarray(pred_min_3, dtype=float)
        self.pred_cart_3 = np.asarray(pred_cart_3, dtype=float)
        self.pred_cart_6 = np.asarray(pred_cart_6, dtype=float)
        self.conformal = conformal
        self.red = red

    def predict(self, origin_pos: int, surplus_projected: float, **_) -> dict:
        pm = self.pred_min_6[origin_pos] + surplus_projected
        p3 = self.pred_cart_3[origin_pos] + surplus_projected
        p6 = self.pred_cart_6[origin_pos] + surplus_projected
        lower = self.conformal.lower_bound(np.array([pm]))[0]
        risk = float(self.conformal.exceedance_probability(np.array([pm]), self.red)[0])
        return {
            "pred_min_6": float(pm),
            "pred_min_3": float(self.pred_min_3[origin_pos] + surplus_projected),
            "pred_cart_3": float(p3),
            "pred_cart_6": float(p6),
            "lower_min_6": float(lower),
            "risk": risk,
        }


class ClosedLoopPredictor:
    """Mode B: rebuild features from the simulated cart history at every step."""

    mode = "closed_loop"

    def __init__(
        self,
        full_frame: pd.DataFrame,
        feature_cols: list,
        clip_bounds: pd.DataFrame,
        estimators: dict,
        conformal,
        red: float,
        cart_bounds: tuple,
        red_model: float | None = None,
        amber_model: float | None = None,
        feature_windows: list | None = None,
    ):
        self.frame = full_frame.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.clip_bounds = clip_bounds
        self.estimators = estimators
        self.conformal = conformal
        self.red = red
        self.cart_floor, self.cart_ceiling = cart_bounds
        # Features MUST be rebuilt with the thresholds the model was trained on.
        # ``red`` is the operational threshold used for risk; it is deliberately
        # not reused for feature construction.
        self.red_model = red if red_model is None else red_model
        self.amber_model = amber_model
        self.feature_windows = feature_windows
        self.overrides: dict[int, float] = {}

    def set_simulated_count(self, global_index: int, value: float) -> None:
        self.overrides[int(global_index)] = float(value)

    def _slice(self, gi: int) -> pd.DataFrame:
        lo = max(0, gi - SLICE_LOOKBACK)
        sl = self.frame.iloc[lo : gi + 1].copy()
        if self.overrides:
            idx = sl.index.to_numpy()
            vals = sl["Count"].to_numpy(dtype=float)
            for k, pos in enumerate(idx):
                if int(pos) in self.overrides:
                    vals[k] = self.overrides[int(pos)]
            sl["Count"] = vals
        return sl

    def predict(self, origin_pos: int, surplus_projected: float, global_index: int = None, **_):
        sl = self._slice(int(global_index))
        feats = build_features(sl, windows=self.feature_windows,
                               red_threshold=self.red_model,
                               amber_threshold=self.amber_model)
        feats = apply_value_clipper(feats, self.clip_bounds)
        row = feats.iloc[[-1]][self.feature_cols]
        if row.isna().any(axis=None):
            return None
        out = {}
        for target, key in (
            ("y_min_6", "pred_min_6"),
            ("y_min_3", "pred_min_3"),
            ("y_cart_3", "pred_cart_3"),
            ("y_cart_6", "pred_cart_6"),
        ):
            raw = float(self.estimators[target].predict(row)[0])
            out[key] = float(np.clip(raw, self.cart_floor, self.cart_ceiling))
        out["lower_min_6"] = float(
            self.conformal.lower_bound(np.array([out["pred_min_6"]]))[0]
        )
        out["risk"] = float(
            self.conformal.exceedance_probability(np.array([out["pred_min_6"]]), self.red)[0]
        )
        return out


# --------------------------------------------------------------------------
# Policy configuration
# --------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    red_threshold: float = 15.0
    amber_threshold: float = 20.0
    target_stock: float = 28.0
    risk_threshold: float = 0.40
    lead_time_min: float = 10.0
    cooldown_min: float = 10.0
    min_carts_per_trip: int = 3
    max_carts_per_trip: int | None = 20
    quantity_rule: str = "point"          # "point" | "risk"
    planning_horizon: int = 6

    def replace(self, **kw) -> "PolicyConfig":
        return PolicyConfig(**{**self.__dict__, **kw})


COST_LAMBDAS = {"shortage": 10.0, "trip": 5.0, "cart": 0.25, "excess": 0.10}


def policy_cost(shortage_severity, n_trips, carts_moved, excess, lambdas=COST_LAMBDAS) -> float:
    return (
        lambdas["shortage"] * shortage_severity
        + lambdas["trip"] * n_trips
        + lambdas["cart"] * carts_moved
        + lambdas["excess"] * excess
    )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def plan_replenishment(
    pred: dict,
    current_sim: float,
    pending_qty: float,
    policy: PolicyConfig,
    inventory: InventoryConfig,
    reserve_left: float,
    effective_lead: float,
) -> dict | None:
    """Size and classify a replenishment action for one decision point.

    Shared by the backtest engine and the online scoring API so the two can
    never diverge. Returns ``None`` when no action is warranted, otherwise a
    plan describing the quantity, urgency and any constraint that bound it.

    The requirement is netted against carts already in transit, so the same
    shortfall is never ordered twice.
    """
    risk = float(pred["risk"])
    if risk < policy.risk_threshold:
        return None

    anchor = pred["lower_min_6"] if policy.quantity_rule == "risk" else pred["pred_min_6"]
    required = policy.target_stock - anchor
    need = required - pending_qty
    q = int(math.ceil(max(0.0, need)))
    if q <= 0:
        return {"skip_reason": "covered by pending order"} if pending_qty > 0 else None
    if q < policy.min_carts_per_trip:
        q = policy.min_carts_per_trip
    if policy.max_carts_per_trip is not None:
        q = min(q, int(policy.max_carts_per_trip))

    headroom = (
        np.inf
        if inventory.bay_capacity is None
        else max(0.0, inventory.bay_capacity - current_sim)
    )
    capacity_rejected = max(0.0, q - headroom) if np.isfinite(headroom) else 0.0
    if np.isfinite(headroom):
        q = int(min(q, math.floor(headroom)))

    reserve_rejected = 0.0
    if np.isfinite(reserve_left):
        reserve_rejected = max(0.0, q - reserve_left)
        q = int(min(q, math.floor(max(reserve_left, 0.0))))

    if q <= 0:
        return {"skip_reason": "no feasible quantity (capacity/reserve)"}

    step = 20.0 / 3.0
    lo_path = _interpolate_path(
        current_sim, pred["pred_cart_3"], pred["pred_cart_6"],
        policy.planning_horizon, pred["lower_min_6"] - pred["pred_min_6"],
    )
    mins_to_breach = np.inf
    for k, v in enumerate(lo_path, start=1):
        if v < policy.red_threshold:
            mins_to_breach = k * step
            break

    return {
        "quantity": int(q),
        "required": float(required),
        "netted_need": float(need),
        "urgency": _urgency(pred, mins_to_breach, effective_lead, policy),
        "minutes_to_predicted_breach": float(mins_to_breach),
        "actionable_before_breach": bool(mins_to_breach > effective_lead),
        "capacity_rejected": float(capacity_rejected),
        "reserve_rejected": float(reserve_rejected),
        "lower_path": [float(v) for v in lo_path],
    }


def run_simulation(
    rows: pd.DataFrame,
    origin_positions: dict,
    predictor,
    policy: PolicyConfig,
    inventory: InventoryConfig,
    lambdas: dict = COST_LAMBDAS,
    recovery_records: int = 2,
) -> dict:
    """Replay telemetry chronologically under the replenishment policy.

    ``rows`` must contain *every* record of the evaluation window (not only the
    scoreable origins) so that the simulated trajectory and its shortage events
    are continuous. ``origin_positions`` maps a row position to the position of
    that origin in the cached prediction arrays; rows absent from the mapping
    are simulated but never trigger a decision.
    """
    df = rows.reset_index(drop=True)
    n = len(df)
    baseline = df["Count"].to_numpy(dtype=float)
    traffic_in = df["Count_IN"].to_numpy(dtype=float)
    durations = df["interval_duration"].to_numpy(dtype=float)
    start_times = pd.to_datetime(df["DVRDate"]).to_numpy()
    end_times = pd.to_datetime(df["decision_time"]).to_numpy()
    global_index = (
        df["global_index"].to_numpy() if "global_index" in df else np.arange(n)
    )

    book = OrderBook()
    ledger = ConservationLedger()
    reserve = inventory.reserve_scenario
    window_days = max(
        (pd.Timestamp(end_times[-1]) - pd.Timestamp(end_times[0])).total_seconds() / 86400.0,
        1e-9,
    )
    reserve_left = (
        np.inf if reserve.carts_per_day is None else float(reserve.carts_per_day) * window_days
    )
    effective_lead = policy.lead_time_min + reserve.availability_delay_min

    surplus = 0.0
    sim_cart = np.empty(n, dtype=float)
    surplus_track = np.zeros(n, dtype=float)
    pending_track = np.zeros(n, dtype=float)
    consumed_track = np.zeros(n, dtype=float)
    decisions = []
    skipped = []
    last_decision_time = None

    for i in range(n):
        ts = pd.Timestamp(end_times[i])
        ts_start = pd.Timestamp(start_times[i])

        # 1. deliveries whose lead time elapsed before this interval started
        for o in book.deliver_due(record_time=ts, record_start_time=ts_start):
            surplus += o.delivered_quantity
            ledger.delivered += o.delivered_quantity
            ledger.record("delivery", ts, o.delivered_quantity, o.order_id)

        # 2. physical bay capacity
        surplus, absorbed = apply_capacity(surplus, baseline[i], inventory)
        if absorbed > 0:
            ledger.absorbed_by_capacity += absorbed
            ledger.record("capacity_absorbed", ts, absorbed, "bay full")

        # 3. customers remove carts from the shared pool
        surplus, consumed = consume_surplus(
            surplus, baseline[i], traffic_in[i], durations[i], inventory
        )
        if consumed > 0:
            ledger.consumed_by_customers += consumed
            ledger.record("consumed", ts, consumed, f"IN={traffic_in[i]:.0f}")
        consumed_track[i] = consumed

        sim_cart[i] = observed_inventory(baseline[i], surplus, inventory)
        surplus_track[i] = surplus
        pending_track[i] = book.pending_quantity()

        # Closed-loop feedback: the simulated state becomes the history that
        # the next feature vector is built from.
        if hasattr(predictor, "set_simulated_count"):
            predictor.set_simulated_count(int(global_index[i]), sim_cart[i])

        # ---------------- decision ----------------
        if i not in origin_positions:
            continue
        if last_decision_time is not None and (
            (ts - last_decision_time).total_seconds() / 60.0 < policy.cooldown_min
        ):
            skipped.append({"time": ts, "reason": "cooldown"})
            continue

        # Project the surviving surplus across the planning horizon using only
        # *recent* traffic (future traffic is never read).
        recent_in = float(np.mean(traffic_in[max(0, i - 2) : i + 1]))
        projected = surplus
        for _ in range(policy.planning_horizon):
            projected, _c = consume_surplus(
                projected, baseline[i], recent_in, durations[i], inventory
            )

        pred = predictor.predict(
            origin_positions[i], projected, global_index=int(global_index[i])
        )
        if pred is None:
            skipped.append({"time": ts, "reason": "features unavailable"})
            continue

        risk = pred["risk"]
        plan = plan_replenishment(
            pred=pred,
            current_sim=float(sim_cart[i]),
            pending_qty=book.pending_quantity(),
            policy=policy,
            inventory=inventory,
            reserve_left=reserve_left,
            effective_lead=effective_lead,
        )
        if plan is None:
            continue
        if "skip_reason" in plan:
            skipped.append({"time": ts, "reason": plan["skip_reason"]})
            continue

        pending_qty = book.pending_quantity()
        q = plan["quantity"]
        urgency = plan["urgency"]
        mins_to_breach = plan["minutes_to_predicted_breach"]
        capacity_rejected = plan["capacity_rejected"]
        reserve_rejected = plan["reserve_rejected"]

        order = book.place(
            decision_time=ts,
            quantity=q,
            lead_time_min=effective_lead,
            origin_index=i,
            urgency=urgency,
            risk_at_decision=risk,
            predicted_min_at_decision=pred["pred_min_6"],
            capacity_rejected=capacity_rejected,
        )
        if np.isfinite(reserve_left):
            reserve_left -= q
            ledger.reserve_drawn += q
        last_decision_time = ts
        decisions.append(
            {
                "order_id": order.order_id,
                "decision_time": ts,
                "record_start_time": ts_start,
                "baseline_cart": float(baseline[i]),
                "simulated_cart": float(sim_cart[i]),
                "surplus_at_decision": float(surplus),
                "pending_at_decision": float(pending_qty),
                "predicted_min_6": pred["pred_min_6"],
                "lower_min_6": pred["lower_min_6"],
                "risk": risk,
                "quantity": q,
                "expected_completion_time": order.expected_completion_time,
                "urgency": urgency,
                "minutes_to_predicted_breach": float(mins_to_breach),
                "actionable_before_breach": bool(mins_to_breach > effective_lead),
                "capacity_rejected": capacity_rejected,
                "reserve_rejected": reserve_rejected,
                "row_index": i,
            }
        )

    ledger.residual_surplus = surplus
    for o in list(book.pending):
        book.cancel(o, "evaluation window ended before delivery")

    base_m = trajectory_metrics(
        baseline, start_times, end_times, durations,
        policy.red_threshold, policy.target_stock, recovery_records,
    )
    pol_m = trajectory_metrics(
        sim_cart, start_times, end_times, durations,
        policy.red_threshold, policy.target_stock, recovery_records,
    )

    series = df.copy()
    series["baseline_cart"] = baseline
    series["sim_cart"] = sim_cart
    series["surplus"] = surplus_track
    series["pending_quantity"] = pending_track
    series["carts_consumed"] = consumed_track

    dec_df = pd.DataFrame(decisions)
    n_trips = len(dec_df)
    carts = int(dec_df["quantity"].sum()) if n_trips else 0
    delivered_carts = float(sum(o.delivered_quantity for o in book.completed))
    span_days = max(
        (pd.Timestamp(end_times[-1]) - pd.Timestamp(end_times[0])).total_seconds() / 86400.0,
        1e-9,
    )

    metrics = {
        "mode": predictor.mode,
        "usage_scenario": inventory.usage,
        "reserve_scenario": inventory.reserve,
        "consumption_model": inventory.consumption_model,
        "risk_threshold": policy.risk_threshold,
        "target_stock": policy.target_stock,
        "lead_time_min": policy.lead_time_min,
        "effective_lead_time_min": effective_lead,
        "max_carts_per_trip": policy.max_carts_per_trip,
        "n_records": n,
        "span_days": span_days,
        "baseline_events": base_m["events"],
        "policy_events": pol_m["events"],
        "baseline_records_below_red": base_m["records_below_red"],
        "policy_records_below_red": pol_m["records_below_red"],
        "baseline_shortage_minutes": base_m["shortage_minutes"],
        "policy_shortage_minutes": pol_m["shortage_minutes"],
        "baseline_severity_cart_records": base_m["severity_cart_records"],
        "policy_severity_cart_records": pol_m["severity_cart_records"],
        "baseline_excess_cart_records": base_m["excess_cart_records"],
        "policy_excess_cart_records": pol_m["excess_cart_records"],
        "baseline_min_cart": base_m["min_cart"],
        "policy_min_cart": pol_m["min_cart"],
        "baseline_mean_cart": base_m["mean_cart"],
        "policy_mean_cart": pol_m["mean_cart"],
        "n_replenishment_actions": n_trips,
        "n_orders_delivered": len(book.completed),
        "n_orders_cancelled": len([o for o in book.orders if o.status == "CANCELLED"]),
        "total_carts_recommended": carts,
        "total_carts_delivered": delivered_carts,
        "avg_carts_per_action": float(dec_df["quantity"].mean()) if n_trips else np.nan,
        "trips_per_day": n_trips / span_days,
        "carts_per_day": carts / span_days,
        "n_urgent": int((dec_df["urgency"] == "URGENT").sum()) if n_trips else 0,
        "n_capacity_constrained": int((dec_df["capacity_rejected"] > 0).sum()) if n_trips else 0,
        "n_reserve_constrained": int((dec_df["reserve_rejected"] > 0).sum()) if n_trips else 0,
        "n_skipped_cooldown": sum(1 for s in skipped if s["reason"] == "cooldown"),
        "n_skipped_pending": sum(1 for s in skipped if "pending" in s["reason"]),
    }
    for key, b, p in (
        ("event", base_m["events"], pol_m["events"]),
        ("duration", base_m["shortage_minutes"], pol_m["shortage_minutes"]),
        ("severity", base_m["severity_cart_records"], pol_m["severity_cart_records"]),
    ):
        metrics[f"{key}_reduction_pct"] = (b - p) / b * 100 if b > 0 else np.nan
    metrics["events_prevented"] = base_m["events"] - pol_m["events"]
    metrics["policy_cost_J"] = policy_cost(
        pol_m["severity_cart_records"], n_trips, carts, pol_m["excess_cart_records"], lambdas
    )
    metrics["baseline_cost_J"] = policy_cost(
        base_m["severity_cart_records"], 0, 0, base_m["excess_cart_records"], lambdas
    )
    metrics.update({f"ledger_{k}": v for k, v in ledger.summary().items()})

    return {
        "metrics": metrics,
        "series": series,
        "decisions": dec_df,
        "orders": book,
        "ledger": ledger,
        "baseline_events": base_m["events_frame"],
        "policy_events": pol_m["events_frame"],
        "skipped": pd.DataFrame(skipped),
    }


def _interpolate_path(c0, p3, p6, horizon, offset=0.0):
    path = []
    for k in range(1, horizon + 1):
        if k <= 3:
            v = c0 + (k / 3.0) * (p3 - c0)
        else:
            v = p3 + ((k - 3) / 3.0) * (p6 - p3)
        path.append(v + offset)
    return path


def _urgency(pred, mins_to_breach, lead, policy) -> str:
    if pred["lower_min_6"] < policy.red_threshold and mins_to_breach <= lead:
        return "URGENT"
    if pred["pred_min_6"] < policy.red_threshold:
        return "HIGH"
    if pred["pred_min_6"] < policy.amber_threshold:
        return "MEDIUM"
    return "LOW"
