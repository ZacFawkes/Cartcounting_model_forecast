"""Priority 9 / 10: sensitivity analysis and policy re-optimisation."""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from audit.audited_simulator import PolicyConfig, run_simulation
from audit.inventory_model import InventoryConfig

LOGGER = logging.getLogger(__name__)

LEAD_TIMES = [5.0, 10.0, 15.0, 20.0]
TARGET_STOCKS = [20, 25, 28, 30]
MAX_TRIP = [10, 15, 20]
USAGE = ["LOW", "MEDIUM", "HIGH"]
RESERVE = ["UNLIMITED", "LIMITED", "DELAYED"]
RETENTION = {"SHORT": 20.0, "MEDIUM": 45.0, "LONG": 180.0}
CONSUMPTION = ["persistent", "unmet_demand", "turnover"]

SCENARIOS = {
    "BEST_CASE": {
        "usage": "LOW",
        "reserve": "UNLIMITED",
        "lead_time_min": 5.0,
        "consumption_model": "persistent",
        "description": "5-min lead time, unlimited reserve, surplus removed only by bay capacity",
    },
    "BASE_CASE": {
        "usage": "MEDIUM",
        "reserve": "LIMITED",
        "lead_time_min": 10.0,
        "consumption_model": "unmet_demand",
        "description": "10-min lead time, 60 carts/day reserve, surplus serves unmet demand",
    },
    "CONSERVATIVE": {
        "usage": "HIGH",
        "reserve": "DELAYED",
        "lead_time_min": 20.0,
        "consumption_model": "turnover",
        "description": "20-min lead time, delayed reserve, surplus eroded at full turnover rate",
    },
}


def _run(rows, origins, predictor, policy, inventory):
    return run_simulation(rows, origins, predictor, policy, inventory)["metrics"]


def _record(metrics, **extra) -> dict:
    keys = [
        "baseline_events",
        "policy_events",
        "events_prevented",
        "event_reduction_pct",
        "duration_reduction_pct",
        "severity_reduction_pct",
        "policy_shortage_minutes",
        "policy_severity_cart_records",
        "n_replenishment_actions",
        "total_carts_recommended",
        "avg_carts_per_action",
        "trips_per_day",
        "carts_per_day",
        "policy_cost_J",
        "baseline_cost_J",
        "n_reserve_constrained",
        "n_capacity_constrained",
    ]
    return {**extra, **{k: metrics[k] for k in keys}}


def sensitivity_analysis(
    rows: pd.DataFrame,
    origins: dict,
    predictor,
    base_policy: PolicyConfig,
    base_inventory: InventoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full lead-time x target-stock grid plus one-factor-at-a-time sweeps."""
    grid_rows = []
    for lead, target in itertools.product(LEAD_TIMES, TARGET_STOCKS):
        pol = base_policy.replace(lead_time_min=lead, target_stock=target)
        m = _run(rows, origins, predictor, pol, base_inventory)
        grid_rows.append(_record(m, lead_time_min=lead, target_stock=target))
    grid = pd.DataFrame(grid_rows)

    ofat = []
    for lead in LEAD_TIMES:
        m = _run(rows, origins, predictor, base_policy.replace(lead_time_min=lead), base_inventory)
        ofat.append(_record(m, parameter="lead_time_min", value=lead))
    for target in TARGET_STOCKS:
        m = _run(rows, origins, predictor, base_policy.replace(target_stock=target), base_inventory)
        ofat.append(_record(m, parameter="target_stock", value=target))
    for mx in MAX_TRIP:
        m = _run(
            rows, origins, predictor, base_policy.replace(max_carts_per_trip=mx), base_inventory
        )
        ofat.append(_record(m, parameter="max_carts_per_trip", value=mx))
    for usage in USAGE:
        inv = InventoryConfig(**{**base_inventory.__dict__, "usage": usage})
        m = _run(rows, origins, predictor, base_policy, inv)
        ofat.append(_record(m, parameter="cart_usage", value=usage))
    for reserve in RESERVE:
        inv = InventoryConfig(**{**base_inventory.__dict__, "reserve": reserve})
        m = _run(rows, origins, predictor, base_policy, inv)
        ofat.append(_record(m, parameter="reserve_availability", value=reserve))
    for cm in CONSUMPTION:
        inv = InventoryConfig(**{**base_inventory.__dict__, "consumption_model": cm})
        m = _run(rows, origins, predictor, base_policy, inv)
        ofat.append(_record(m, parameter="consumption_model", value=cm))
    for name, hl in RETENTION.items():
        inv = InventoryConfig(
            **{
                **base_inventory.__dict__,
                "consumption_model": "halflife",
                "retention_halflife_min": hl,
            }
        )
        m = _run(rows, origins, predictor, base_policy, inv)
        ofat.append(_record(m, parameter="retention_halflife", value=f"{name} ({hl:.0f} min)"))
    for thr in [0.20, 0.30, 0.40, 0.50, 0.60]:
        m = _run(rows, origins, predictor, base_policy.replace(risk_threshold=thr), base_inventory)
        ofat.append(_record(m, parameter="risk_threshold", value=thr))

    return grid, pd.DataFrame(ofat)


def scenario_analysis(
    rows: pd.DataFrame,
    origins: dict,
    predictor,
    base_policy: PolicyConfig,
    base_inventory: InventoryConfig,
) -> pd.DataFrame:
    out = []
    for name, spec in SCENARIOS.items():
        inv = InventoryConfig(
            **{
                **base_inventory.__dict__,
                "usage": spec["usage"],
                "reserve": spec["reserve"],
                "consumption_model": spec["consumption_model"],
            }
        )
        pol = base_policy.replace(lead_time_min=spec["lead_time_min"])
        m = _run(rows, origins, predictor, pol, inv)
        out.append(
            _record(
                m,
                scenario=name,
                description=spec["description"],
                usage=spec["usage"],
                reserve=spec["reserve"],
                consumption_model=spec["consumption_model"],
                lead_time_min=spec["lead_time_min"],
            )
        )
    return pd.DataFrame(out)


def fragility_ranking(ofat: pd.DataFrame) -> pd.DataFrame:
    """Rank assumptions by how much they move the headline reduction."""
    rows = []
    for param, sub in ofat.groupby("parameter", observed=True):
        vals = sub["severity_reduction_pct"].astype(float)
        ev = sub["event_reduction_pct"].astype(float)
        rows.append(
            {
                "assumption": param,
                "severity_reduction_min_pct": float(vals.min()),
                "severity_reduction_max_pct": float(vals.max()),
                "severity_reduction_spread_pct": float(vals.max() - vals.min()),
                "event_reduction_min_pct": float(ev.min()),
                "event_reduction_max_pct": float(ev.max()),
                "event_reduction_spread_pct": float(ev.max() - ev.min()),
                "trips_min": float(sub["n_replenishment_actions"].min()),
                "trips_max": float(sub["n_replenishment_actions"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "severity_reduction_spread_pct", ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Priority 10 — policy re-optimisation (validation only)
# --------------------------------------------------------------------------
POLICY_GRID = {
    "risk_threshold": [0.25, 0.35, 0.45, 0.55],
    "target_stock": [22, 25, 28],
    "cooldown_min": [10.0, 20.0, 30.0],
    "quantity_rule": ["point", "risk"],
}


def search_policy(
    rows: pd.DataFrame,
    origins: dict,
    predictor,
    base_policy: PolicyConfig,
    inventory: InventoryConfig,
    grid: dict | None = None,
    max_trips_per_day: float = 8.0,
) -> tuple[PolicyConfig, pd.DataFrame]:
    """Select a policy on validation data under an explicit workload budget.

    Minimising J alone can reward a policy that eliminates shortages by moving
    an unrealistic number of carts, so candidates exceeding the workload budget
    are marked infeasible and only used as reference points.
    """
    grid = grid or POLICY_GRID
    keys = list(grid)
    rows_out = []
    best, best_cost = None, np.inf
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        pol = base_policy.replace(**params)
        m = _run(rows, origins, predictor, pol, inventory)
        feasible = m["trips_per_day"] <= max_trips_per_day
        rows_out.append(_record(m, **params, workload_feasible=feasible))
        if feasible and m["policy_cost_J"] < best_cost:
            best_cost, best = m["policy_cost_J"], pol
    table = pd.DataFrame(rows_out).sort_values("policy_cost_J").reset_index(drop=True)
    if best is None:  # every candidate exceeded the workload budget
        top = table.iloc[0]
        best = base_policy.replace(
            **{k: top[k] for k in keys if k in top.index}
        )
        LOGGER.warning("No candidate met the workload budget; falling back to lowest J")
    LOGGER.info(
        "Audited policy search: %d candidates, selected risk=%.2f target=%.0f cooldown=%.0f "
        "rule=%s (J=%.1f)",
        len(table), best.risk_threshold, best.target_stock, best.cooldown_min,
        best.quantity_rule, best_cost,
    )
    return best, table
