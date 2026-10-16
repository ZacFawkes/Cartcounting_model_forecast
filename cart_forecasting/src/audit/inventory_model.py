"""Inventory state-transition model, usage scenarios and conservation ledger.

Audit finding driving this module
---------------------------------
The original simulator eroded the delivered cart surplus with a fixed 45-minute
exponential half-life. That constant was an arbitrary wall-clock assumption: it
decayed surplus carts at 03:00 when no customer was in the store, and it
decayed them at the same speed during the midday rush.

The audited model ties consumption to observed customer throughput instead of
wall-clock time, while refusing to assume ``1 customer = 1 cart``. The cart
pickup rate per entering customer is treated as an *uncertain parameter* and
evaluated under LOW / MEDIUM / HIGH scenarios.

State transition
----------------
Let ``B(t)`` be the surplus carts still attributable to replenishment orders,
``C_base(t)`` the observed baseline cart count and ``IN(t)`` the customers
entering during interval ``t``.

    1. deliveries        B <- B + sum(Q_j) for orders completed before t
    2. capacity          B <- min(B, BayCapacity - C_base(t))        (absorbed)
    3. consumption       u(t) = clip(p * IN(t) / (C_base(t) + B), 0, 1)
                         B <- B * (1 - u(t))
    4. observation       C_sim(t) = clip(C_base(t) + B, 0, BayCapacity)

Rationale for step 3: surplus carts are physically indistinguishable from the
carts already in the bay, so they are taken at the same per-cart rate. Customer
*returns* are deliberately **not** credited to ``B`` because returns are already
embedded in the observed baseline trajectory; crediting them again would create
carts from nothing.

``p`` is the unknown carts-per-entering-customer ratio. It is not identifiable
from this dataset, so it is a scenario parameter, never a measured fact.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Scenario parameters (assumption-driven, documented, never measured)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UsageScenario:
    """Carts removed per entering customer."""

    name: str
    cart_pickup_per_entry: float
    description: str


USAGE_SCENARIOS = {
    "LOW": UsageScenario("LOW", 0.25, "One cart per four entering customers"),
    "MEDIUM": UsageScenario("MEDIUM", 0.50, "One cart per two entering customers"),
    "HIGH": UsageScenario("HIGH", 0.80, "Four carts per five entering customers"),
}

# Retention scenarios express the same uncertainty as a wall-clock half-life and
# are used to reproduce / bracket the original simulator's behaviour.
RETENTION_SCENARIOS = {"SHORT": 20.0, "MEDIUM": 45.0, "LONG": 180.0}

CONSUMPTION_MODELS = {
    "persistent": "Surplus removed only by bay capacity (optimistic bound)",
    "unmet_demand": "Surplus serves demand the observed trajectory could not meet (base case)",
    "turnover": "Surplus taken at the same per-cart rate as baseline stock (pessimistic bound)",
    "halflife": "Wall-clock exponential decay (reproduces the original implementation)",
}


@dataclass(frozen=True)
class ReserveScenario:
    """Availability of physical carts to retrieve.

    The true reserve is unknown. It is expressed as a daily retrieval budget
    because carts returned to the car park become retrievable again; a single
    fixed pool would understate availability over a multi-day window.
    """

    name: str
    carts_per_day: float | None       # None => unlimited (must be labelled)
    availability_delay_min: float     # extra minutes before a retrieval can start
    description: str


RESERVE_SCENARIOS = {
    "UNLIMITED": ReserveScenario(
        "UNLIMITED", None, 0.0,
        "Theoretical: carts are always available in the car park (unconstrained)",
    ),
    "LIMITED": ReserveScenario(
        "LIMITED", 60.0, 0.0,
        "At most 60 carts can be retrieved per day",
    ),
    "DELAYED": ReserveScenario(
        "DELAYED", 60.0, 10.0,
        "At most 60 carts per day, plus 10 extra minutes before a retrieval can start",
    ),
}


@dataclass
class InventoryConfig:
    bay_capacity: float | None = 60.0
    usage: str = "MEDIUM"
    reserve: str = "UNLIMITED"
    # "unmet_demand" | "persistent" | "turnover" | "halflife"
    consumption_model: str = "unmet_demand"
    retention_halflife_min: float = 45.0   # used when consumption_model="halflife"

    @property
    def pickup_rate(self) -> float:
        return USAGE_SCENARIOS[self.usage].cart_pickup_per_entry

    @property
    def reserve_scenario(self) -> ReserveScenario:
        return RESERVE_SCENARIOS[self.reserve]


# --------------------------------------------------------------------------
# Conservation ledger
# --------------------------------------------------------------------------
@dataclass
class ConservationLedger:
    """Tracks every cart entering or leaving the simulated surplus pool."""

    delivered: float = 0.0
    consumed_by_customers: float = 0.0
    absorbed_by_capacity: float = 0.0
    residual_surplus: float = 0.0
    reserve_drawn: float = 0.0
    events: list = field(default_factory=list)

    def record(self, kind: str, ts, amount: float, note: str = "") -> None:
        self.events.append({"time": ts, "kind": kind, "carts": float(amount), "note": note})

    def balance_error(self) -> float:
        return float(
            self.delivered
            - self.consumed_by_customers
            - self.absorbed_by_capacity
            - self.residual_surplus
        )

    def summary(self) -> dict:
        return {
            "carts_delivered": self.delivered,
            "carts_consumed_by_customers": self.consumed_by_customers,
            "carts_absorbed_by_capacity": self.absorbed_by_capacity,
            "residual_surplus_at_end": self.residual_surplus,
            "reserve_drawn": self.reserve_drawn,
            "balance_error": self.balance_error(),
            "conserved": bool(abs(self.balance_error()) < 1e-6),
        }


def consume_surplus(
    surplus: float,
    baseline_cart: float,
    traffic_in: float,
    minutes: float,
    cfg: InventoryConfig,
) -> tuple[float, float]:
    """Return ``(new_surplus, carts_consumed)`` for one interval.

    Four candidate transition models, none of which is identifiable from this
    dataset. They are evaluated as scenarios and must never be presented as
    measured behaviour.

    ``persistent``
        The surplus only disappears through bay capacity. Justification: if the
        bay is not supply-constrained, a customer takes a cart whether or not
        the extra carts are there, so the *difference* between the simulated
        and observed trajectories is unchanged. This is what the original
        implementation did between its decay steps, and it is the optimistic
        bound.

    ``unmet_demand`` (base case)
        The surplus is only drawn down by demand the observed trajectory could
        not serve: ``consumed = min(surplus, max(0, p*IN - C_base))``. Extra
        carts matter exactly when the store ran out, which is the regime the
        policy targets. Physically motivated and the default.

    ``turnover``
        Surplus carts are indistinguishable from baseline carts and are taken
        at the same per-cart rate, with returns credited only to the baseline.
        This is the pessimistic bound.

    ``halflife``
        Wall-clock exponential decay. Retained only to reproduce and bracket
        the original implementation's behaviour.
    """
    if surplus <= 0:
        return 0.0, 0.0

    model = cfg.consumption_model
    if model in ("none", "persistent"):
        return surplus, 0.0

    if model == "halflife":
        kept = surplus * 0.5 ** (max(minutes, 0.0) / max(cfg.retention_halflife_min, 1e-6))
        return kept, surplus - kept

    demand = cfg.pickup_rate * max(traffic_in, 0.0)

    if model == "unmet_demand":
        unmet = max(0.0, demand - max(baseline_cart, 0.0))
        consumed = float(min(surplus, unmet))
        return surplus - consumed, consumed

    if model == "turnover":
        total_stock = max(baseline_cart + surplus, 1e-6)
        u = float(np.clip(demand / total_stock, 0.0, 1.0))
        kept = surplus * (1.0 - u)
        return kept, surplus - kept

    raise ValueError(f"Unknown consumption model: {model}")


def apply_capacity(
    surplus: float, baseline_cart: float, cfg: InventoryConfig
) -> tuple[float, float]:
    """Clip the surplus to the physical bay capacity. Returns (surplus, absorbed)."""
    if cfg.bay_capacity is None:
        return surplus, 0.0
    headroom = max(0.0, float(cfg.bay_capacity) - float(baseline_cart))
    if surplus <= headroom:
        return surplus, 0.0
    return headroom, surplus - headroom


def observed_inventory(baseline_cart: float, surplus: float, cfg: InventoryConfig) -> float:
    """Simulated cart count, clipped to the bay capacity.

    The declared capacity is an assumption while the baseline is an
    observation, so the cap can never push the simulated trajectory *below* the
    observed one: if the store really held more carts than the assumed
    capacity, the observation wins.
    """
    hi = cfg.bay_capacity if cfg.bay_capacity is not None else np.inf
    hi = max(hi, float(baseline_cart))
    return float(np.clip(float(baseline_cart) + float(surplus), 0.0, hi))


# --------------------------------------------------------------------------
# Shared shortage-event grouping (identical rule for observed and simulated)
# --------------------------------------------------------------------------
def group_shortage_events(
    values: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    durations: np.ndarray,
    red: float,
    recovery_records: int = 2,
) -> pd.DataFrame:
    """Group consecutive sub-RED observations into independent events.

    The *same* function is used for the observed baseline and every simulated
    trajectory, so event-reduction percentages can never be produced by two
    different grouping rules.
    """
    values = np.asarray(values, dtype=float)
    below = values < red
    n = len(values)
    rows = []
    i = 0
    while i < n:
        if not below[i]:
            i += 1
            continue
        start = i
        j = i
        run = 0
        last_below = i
        while j < n:
            if below[j]:
                run = 0
                last_below = j
            else:
                run += 1
                if run >= recovery_records:
                    break
            j += 1
        end = last_below
        seg = values[start : end + 1]
        rows.append(
            {
                "event_id": len(rows) + 1,
                "onset_time": pd.Timestamp(start_times[start]),
                "onset_decision_time": pd.Timestamp(end_times[start]),
                "end_time": pd.Timestamp(end_times[end]),
                "start_index": int(start),
                "end_index": int(end),
                "n_records": int(end - start + 1),
                "min_cart": float(seg.min()),
                "mean_cart": float(seg.mean()),
                "severity_cart_records": float(np.sum(np.clip(red - seg, 0, None))),
                "duration_min": float(np.sum(durations[start : end + 1])),
            }
        )
        i = j + 1
    return pd.DataFrame(rows)


def trajectory_metrics(
    values: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    durations: np.ndarray,
    red: float,
    target_stock: float,
    recovery_records: int = 2,
) -> dict:
    ev = group_shortage_events(values, start_times, end_times, durations, red, recovery_records)
    values = np.asarray(values, dtype=float)
    below = values < red
    return {
        "events": int(len(ev)),
        "records_below_red": int(below.sum()),
        "shortage_minutes": float(np.sum(np.asarray(durations)[below])),
        "severity_cart_records": float(np.sum(np.clip(red - values, 0, None))),
        "excess_cart_records": float(np.sum(np.clip(values - target_stock, 0, None))),
        "min_cart": float(values.min()),
        "mean_cart": float(values.mean()),
        "events_frame": ev,
    }
