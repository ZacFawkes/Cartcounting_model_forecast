"""Timing-aware replenishment quantity and schedule optimisation."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import RunConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class ReplenishmentDecision:
    decision_time: pd.Timestamp
    effective_time: pd.Timestamp
    current_cart: float
    predicted_min: float
    lower_bound_min: float
    risk: float
    quantity_point: int
    quantity_risk: int
    quantity: int
    urgency: str
    reason: str
    constrained: bool
    expected_min_after: float


def recommend_quantity(
    predicted_min: float,
    lower_bound_min: float,
    cfg: RunConfig,
) -> tuple[int, int, int]:
    """Point and risk-adjusted top-up quantities (rounded up to whole carts)."""
    q_point = max(0, math.ceil(cfg.target_stock - predicted_min))
    q_risk = max(0, math.ceil(cfg.target_stock - lower_bound_min))
    q = q_risk if cfg.quantity_rule == "risk" else q_point
    if q > 0 and q < cfg.min_carts_per_trip:
        q = cfg.min_carts_per_trip
    if cfg.max_carts_per_trip is not None:
        q = min(q, cfg.max_carts_per_trip)
    if cfg.reserve_cart_inventory is not None:
        q = min(q, cfg.reserve_cart_inventory)
    return int(q_point), int(q_risk), int(q)


def classify_urgency(
    predicted_min: float,
    current_cart: float,
    lower_bound_min: float,
    minutes_to_predicted_breach: float,
    cfg: RunConfig,
) -> str:
    """Urgent when the breach is expected before replenishment can complete."""
    predicted_urgent = (
            lower_bound_min <= cfg.red_threshold
            and pd.notna(minutes_to_predicted_breach)
            and minutes_to_predicted_breach <= cfg.lead_time_min

    )
    reactive_urgent = (
            current_cart <= cfg.red_threshold
    )
    if predicted_urgent or  reactive_urgent:
        return "URGENT"
    if predicted_min < cfg.red_threshold:
        return "HIGH"
    if predicted_min < cfg.amber_threshold:
        return "MEDIUM"
    return "LOW"


def minutes_to_breach(
    path_times: np.ndarray, path_values: np.ndarray, origin: pd.Timestamp, threshold: float
) -> float:
    """First predicted time (minutes from origin) at which the path breaches."""
    for t, v in zip(path_times, path_values):
        if v < threshold:
            return max(0.0, (pd.Timestamp(t) - origin).total_seconds() / 60.0)
    return np.inf


def decide(
    ts: pd.Timestamp,
    current_cart: float,
    predicted_min: float,
    lower_bound_min: float,
    risk: float,
    mins_to_breach: float,
    cfg: RunConfig,
    bay_headroom: float | None = None,
) -> ReplenishmentDecision | None:
    """Return a replenishment decision when the risk threshold is exceeded."""
    if risk < cfg.risk_threshold:
        return None

    q_point, q_risk, q = recommend_quantity(predicted_min, lower_bound_min, cfg)
    constrained = False
    if bay_headroom is not None:
        capped = int(max(0, min(q, math.floor(bay_headroom))))
        constrained = capped < q
        q = capped
    if q <= 0:
        return None

    urgency = classify_urgency(
        current_cart=current_cart,
        predicted_min=predicted_min,
        lower_bound_min=lower_bound_min,
        minutes_to_predicted_breach=mins_to_breach,
        cfg=cfg,
    )
    reason = (
        f"risk={risk:.2f} >= {cfg.risk_threshold:.2f}; predicted 40-min minimum "
        f"{predicted_min:.1f} (90% lower bound {lower_bound_min:.1f}) vs target "
        f"{cfg.target_stock}"
    )
    return ReplenishmentDecision(
        decision_time=ts,
        effective_time=ts + pd.Timedelta(minutes=cfg.lead_time_min),
        current_cart=current_cart,
        predicted_min=predicted_min,
        lower_bound_min=lower_bound_min,
        risk=float(risk),
        quantity_point=q_point,
        quantity_risk=q_risk,
        quantity=int(q),
        urgency=urgency,
        reason=reason,
        constrained=constrained,
        expected_min_after=float(predicted_min + q),
    )


def policy_cost(
    shortage_severity: float,
    n_trips: int,
    carts_moved: int,
    excess_inventory: float,
    lambdas: dict,
) -> float:
    """J = l1*Shortage + l2*Trips + l3*CartsMoved + l4*Excess."""
    return (
        lambdas["shortage"] * shortage_severity
        + lambdas["trip"] * n_trips
        + lambdas["cart"] * carts_moved
        + lambdas["excess"] * excess_inventory
    )
