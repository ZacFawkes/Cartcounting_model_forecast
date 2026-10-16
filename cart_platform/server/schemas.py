"""Request / response contracts for the platform API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TelemetryRecord(BaseModel):
    DVRDate: str
    next_DVRDate: str
    Count: float = Field(..., ge=0)
    Count_IN: float = Field(0, ge=0)
    Count_OUT: float = Field(0, ge=0)


class ParameterOverrides(BaseModel):
    """Level-1 overrides. Omitted values fall back to the entity config."""

    red_threshold: int | None = None
    amber_threshold: int | None = None
    target_stock: int | None = None
    risk_threshold: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Per-call override of the model's clustered WARNING band. "
                    "Diagnostics and audited exceptions only — it is never "
                    "persisted to entity_config.")
    critical_risk_threshold: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Per-call override of the clustered CRITICAL band.")
    lead_time_min: float | None = Field(None, gt=0)
    cooldown_min: float | None = Field(None, ge=0)
    min_carts_per_trip: int | None = Field(None, ge=0)
    max_carts_per_trip: int | None = Field(None, gt=0)
    quantity_rule: Literal["point", "risk"] | None = None
    bay_capacity: int | None = Field(None, gt=0)
    reserve_per_day: int | None = Field(None, ge=0)
    consumption_model: Literal["persistent", "unmet_demand", "turnover", "halflife"] | None = None
    usage: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    retention_halflife_min: float | None = Field(None, gt=0)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    entity_id: str | None = Field(None, description='"KDVR:ChanNo", e.g. "215:0"')
    kdvr: int | None = None
    chan_no: int | None = None
    site_key: int | None = None
    records: list[TelemetryRecord] = Field(
        ...,
        description="Consecutive, chronological; last = newest COMPLETED record. "
                    "Minimum 12, recommended 20.",
    )
    params: ParameterOverrides = Field(default_factory=ParameterOverrides)
    model_id: str | None = None
    write_to_db: bool = True
    client: str | None = None

    @field_validator("records")
    @classmethod
    def _non_empty(cls, v):
        if len(v) < 2:
            raise ValueError("records must contain at least 2 entries")
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
    request_id: str | None
    entity_id: str
    kdvr: int
    chan_no: int
    site_key: int
    entity_status: str
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
    model_id: str
    calibration_id: str
    calibration_scope: str
    resolved_config: dict
    warnings: list[str]
    caveat: str
    persisted: bool | None = None
    artifact_path: str | None = None
    artifact_request_id: str | None = None
    idempotent_replay: bool = False
    frozen_model_config: dict | None = None
    config_hash: str | None = None


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    entity_id: str | None = None
    entity_ids: list[str] | None = None
    site_key: int | None = Field(None, description="Backtest every entity in a store")
    records: list[TelemetryRecord] | None = None
    start: str | None = None
    end: str | None = None
    mode: Literal["replay", "closed_loop"] = "replay"
    params: ParameterOverrides = Field(default_factory=ParameterOverrides)
    model_id: str | None = None
    include_decisions: bool = True
    include_events: bool = True
    include_series: bool = False
    label: str | None = None
    client: str | None = None


class BacktestResponse(BaseModel):
    request_id: str | None
    runs: list[dict]
    n_entities: int
    caveat: str


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
class SiteUpsert(BaseModel):
    site_key: int
    display_name: str | None = None
    timezone: str | None = None
    lead_time_min: float | None = None
    reserve_per_day: int | None = None
    min_carts_per_trip: int | None = None
    max_carts_per_trip: int | None = None
    cooldown_min: float | None = None
    updated_by: str = "api"


class EntityUpsert(BaseModel):
    entity_id: str | None = None
    kdvr: int | None = None
    chan_no: int | None = None
    site_key: int | None = None
    area_id: int | None = None
    display_name: str | None = None
    status: Literal["active", "shadow", "disabled"] | None = None
    bay_capacity: int | None = None
    red_threshold: int | None = None
    amber_threshold: int | None = None
    target_stock: int | None = None
    threshold_source: Literal["business", "derived", "manual_override"] | None = None
    lead_time_min: float | None = None
    reserve_per_day: int | None = None
    min_carts_per_trip: int | None = None
    max_carts_per_trip: int | None = None
    cooldown_min: float | None = None
    quantity_rule: Literal["point", "risk"] | None = None
    consumption_model: str | None = None
    usage: str | None = None
    retention_halflife_min: float | None = None
    max_sequence_gap_min: float | None = None
    model_id: str | None = None
    calibration_id: str | None = None
    updated_by: str = "api"


class StatusUpdate(BaseModel):
    status: Literal["active", "shadow", "disabled"]
    updated_by: str = "api"


class ExecutionLog(BaseModel):
    executed_by: str
    actual_carts: int = Field(..., ge=0)
    notes: str | None = None


class ResetRequest(BaseModel):
    entity_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    champion_model_id: str | None
    n_entities: int
    n_active: int
    uptime_seconds: float
