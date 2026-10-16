"""Request / response schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from server.settings import DEFAULT_INVENTORY, DEFAULT_POLICY


class TelemetryRecord(BaseModel):
    DVRDate: str = Field(..., description="Interval start timestamp")
    next_DVRDate: str = Field(..., description="Interval end = decision time")
    Count: float = Field(..., ge=0, description="Cart availability measurement")
    Count_IN: float = Field(0, ge=0, description="Customers entering during the interval")
    Count_OUT: float = Field(0, ge=0, description="Customers leaving during the interval")
    siteKey: int | None = None


class PolicyOverrides(BaseModel):
    red_threshold: float | None = None
    amber_threshold: float | None = None
    target_stock: float | None = None
    risk_threshold: float | None = Field(None, ge=0.0, le=1.0)
    lead_time_min: float | None = Field(None, gt=0)
    cooldown_min: float | None = Field(None, ge=0)
    min_carts_per_trip: int | None = Field(None, ge=0)
    max_carts_per_trip: int | None = Field(None, gt=0)
    quantity_rule: Literal["point", "risk"] | None = None

    def merged(self) -> dict:
        out = dict(DEFAULT_POLICY)
        out.update({k: v for k, v in self.model_dump().items() if v is not None})
        return out


class InventoryOverrides(BaseModel):
    bay_capacity: float | None = Field(None, gt=0)
    usage: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    reserve: Literal["UNLIMITED", "LIMITED", "DELAYED"] | None = None
    consumption_model: Literal["persistent", "unmet_demand", "turnover", "halflife"] | None = None
    retention_halflife_min: float | None = Field(None, gt=0)

    def merged(self) -> dict:
        out = dict(DEFAULT_INVENTORY)
        out.update({k: v for k, v in self.model_dump().items() if v is not None})
        return out


# --------------------------------------------------------------------------
# /score
# --------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    site_key: int = 163
    history: list[TelemetryRecord] = Field(
        ...,
        description="Chronological telemetry ending with the newest completed record. "
                    "At least 12 records are required to build the feature window.",
    )
    policy: PolicyOverrides = Field(default_factory=PolicyOverrides)
    inventory: InventoryOverrides = Field(default_factory=InventoryOverrides)
    session_id: str | None = Field(
        None,
        description="Groups consecutive calls so pending orders and alert state persist. "
                    "Defaults to the site key.",
    )
    client: str | None = None

    @field_validator("history")
    @classmethod
    def _min_history(cls, v):
        if len(v) < 2:
            raise ValueError("history must contain at least 2 records")
        return v


class ForecastBlock(BaseModel):
    cart_at_20min: float
    cart_at_40min: float
    min_within_20min: float
    min_within_40min: float
    lower_bound_min_within_40min: float
    horizon_note: str


class ActionBlock(BaseModel):
    should_replenish: bool
    quantity: int
    urgency: str
    order_id: str | None = None
    decision_time: str | None = None
    expected_completion_time: str | None = None
    actionable_before_breach: bool | None = None
    minutes_to_predicted_breach: float | None = None
    reason: str


class ScoreResponse(BaseModel):
    request_id: str
    site_key: int
    session_id: str
    decision_time: str
    current_cart: float
    simulated_cart: float
    forecast: ForecastBlock
    shortage_risk: float
    alert_state: str
    new_predictive_alert: bool
    reactive_critical: bool
    action: ActionBlock
    pending_carts: float
    pending_orders: list[dict]
    model_version: str
    assumptions: dict


# --------------------------------------------------------------------------
# /backtest
# --------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    records: list[TelemetryRecord] | None = Field(
        None, description="Telemetry window to backtest. Omit to use the bundled dataset."
    )
    use_bundled_data: bool = Field(
        False, description="Backtest the dataset shipped with the service."
    )
    start: str | None = Field(None, description="ISO timestamp filter (bundled data only)")
    end: str | None = Field(None, description="ISO timestamp filter (bundled data only)")
    mode: Literal["replay", "closed_loop"] = "replay"
    policy: PolicyOverrides = Field(default_factory=PolicyOverrides)
    inventory: InventoryOverrides = Field(default_factory=InventoryOverrides)
    include_decisions: bool = True
    include_events: bool = True
    include_series: bool = False
    label: str | None = Field(None, description="Free-text label stored with the run")
    client: str | None = None


class BacktestResponse(BaseModel):
    request_id: str
    run_id: str
    label: str | None
    mode: str
    window_start: str
    window_end: str
    n_records: int
    scored_origins: int
    metrics: dict
    forecast_accuracy: dict
    attribution_summary: dict
    decisions: list[dict] | None = None
    shortage_events: list[dict] | None = None
    series: list[dict] | None = None
    artifact_dir: str
    assumptions: dict
    caveat: str


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class SessionResetRequest(BaseModel):
    session_id: str | None = None
