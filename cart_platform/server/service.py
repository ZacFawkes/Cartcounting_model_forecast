"""Entity-aware scoring and backtesting services.

Both call the same ``plan_replenishment`` from the audited simulator, so an
online recommendation can never diverge from what a backtest would produce.
"""
from __future__ import annotations

import logging
import uuid

import numpy as np
import pandas as pd

from alert_engine import AlertEngineConfig
from audit.attribution import attribute_events, attribution_summary
from audit.audited_simulator import (
    ClosedLoopPredictor,
    PolicyConfig,
    ReplayPredictor,
    plan_replenishment,
    run_simulation,
)
from audit.inventory_model import (
    InventoryConfig,
    apply_capacity,
    consume_surplus,
    observed_inventory,
)
from settings import (
    ABSOLUTE_MIN_HISTORY,
    BUNDLED_TELEMETRY,
    CAVEAT,
    COST_LAMBDAS,
    INCLUDE_BAY_IN_ENTITY_ID,
    INCLUDE_SITE_IN_ENTITY_ID,
    MIN_HISTORY_RECORDS,
    RECOMMENDED_HISTORY,
)
from data_loader import BAY_COL, load_raw, prepare
from feature_engineering import apply_value_clipper, build_features
from registry import entities as reg
from registry import models as mreg
from registry.resolver import resolve
from server import store
from target_builder import build_targets

LOGGER = logging.getLogger("platform.service")

TARGETS = ["y_cart_3", "y_cart_6", "y_min_3", "y_min_6"]
HORIZON_NOTE = (
    "Horizons are record-based: +3 records is nominally 20 minutes, +6 records "
    "nominally 40 minutes. Elapsed time varies with telemetry cadence."
)


class InsufficientHistory(ValueError):
    pass


class EntityNotScoreable(ValueError):
    pass


class ModelNotReady(RuntimeError):
    """No promoted model is bound to this entity (HTTP 409)."""

    def __init__(self, message: str, readiness: dict | None = None):
        super().__init__(message)
        self.readiness = readiness or {}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def records_to_frame(records: list[dict], entity: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for c in ("DVRDate", "next_DVRDate"):
        df[c] = pd.to_datetime(df[c])
    for c in ("Count", "Count_IN", "Count_OUT"):
        df[c] = pd.to_numeric(df[c])
    n = len(df)
    df["CountID"] = range(n)
    df["CNN_Index"] = range(n)
    df["CatID"] = 4000
    if entity is not None:
        df["KDVR"] = entity["kdvr"]
        df["ChanNo"] = entity["chan_no"]
        df["siteKey"] = entity["site_key"]
        df["AreaID"] = entity.get("area_id") or 0
        if entity.get("bay_id"):
            df[BAY_COL] = entity["bay_id"]
    else:
        for c, v in (("KDVR", 0), ("ChanNo", 0), ("siteKey", 0), ("AreaID", 0)):
            if c not in df:
                df[c] = v
    include_bay = INCLUDE_BAY_IN_ENTITY_ID and BAY_COL in df.columns
    return prepare(df, include_site_in_entity=INCLUDE_SITE_IN_ENTITY_ID,
                   include_bay_in_entity=include_bay)


def _policy(cfg) -> PolicyConfig:
    """Operational policy. ``risk_threshold`` here is the clustered WARNING
    band: alerting and replenishment share one measured operating point rather
    than two independently hand-set constants."""
    return PolicyConfig(**cfg.policy_kwargs())


def _inventory(cfg, reserve_scenario: str = "LIMITED") -> InventoryConfig:
    kw = cfg.inventory_kwargs()
    reserve = "UNLIMITED" if cfg.values.get("reserve_per_day") is None else reserve_scenario
    return InventoryConfig(reserve=reserve, **kw)


def _alert_cfg(cfg) -> AlertEngineConfig:
    """Bands come from the model's K-Means clustering, not from configuration."""
    return AlertEngineConfig(
        warning_risk=float(cfg["risk_threshold"]),
        critical_risk=float(cfg["critical_risk_threshold"]),
        cooldown_min=float(cfg["cooldown_min"]),
        red_threshold=float(cfg["red_threshold"]),
    )


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else round(float(v), 4)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif isinstance(v, pd.DataFrame):
            continue
        else:
            out[k] = v
    return out


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or not len(df):
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out.replace({np.nan: None}).to_dict(orient="records")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def resolve_entity_model(entity_id: str, model_id: str | None = None,
                         allow_fallback: bool = False):
    """Resolve the model+calibrator bound to THIS entity.

    Precedence: explicit ``model_id`` -> promoted per-entity binding -> (only if
    the caller explicitly opts in) a compatible global fallback, clearly marked
    degraded. A route resolving is never treated as a model being ready.
    """
    from registry import bindings as bind
    from training import calibration as cal

    warnings: list[str] = []
    binding = None
    if model_id:
        binding = bind.get_binding(entity_id, model_id)
        if binding is None:
            raise ModelNotReady(
                f"model '{model_id}' is not bound to entity '{entity_id}'",
                bind.readiness(entity_id),
            )
    else:
        binding = bind.active_binding(entity_id)

    if binding is not None:
        model = mreg.load_model(binding["model_id"])
        record = cal.get_calibration(binding["calibration_id"]) if binding.get(
            "calibration_id") else None
        calib = cal.load_calibrator_from_record(record)
        scope = (record or {}).get("scope", "none")
        cal_id = (record or {}).get("calibration_id")
        if calib is None:
            # The binding passed promotion gates, so a missing artifact is an
            # integrity problem rather than a normal state.
            try:
                row = mreg.resolve_calibrator(model.model_id, entity_id)
                calib = mreg.load_calibrator(row["calibration_id"])
                cal_id, scope = row["calibration_id"], "global_fallback"
                warnings.append(
                    "entity calibrator artifact unavailable; using the global "
                    "fallback. Shortage probabilities are NOT calibrated to this "
                    "entity."
                )
            except Exception as exc:
                raise ModelNotReady(
                    f"calibrator for '{entity_id}' could not be loaded: {exc}",
                    bind.readiness(entity_id),
                )
        if binding["role"] != bind.ACTIVE:
            warnings.append(
                f"model '{binding['model_id']}' is a CANDIDATE, not promoted; "
                f"this result is degraded/shadow"
            )
        return model, calib, cal_id, scope, warnings, binding

    # No per-entity binding at all.
    if not allow_fallback:
        raise ModelNotReady(
            f"MODEL_NOT_READY: no promoted model is bound to entity '{entity_id}'. "
            f"Train it from an upload and promote a candidate before scoring.",
            bind.readiness(entity_id),
        )
    try:
        model = mreg.load_model(None)
        row = mreg.resolve_calibrator(model.model_id, entity_id)
        calib = mreg.load_calibrator(row["calibration_id"])
    except mreg.NoModelAvailable as exc:
        raise ModelNotReady(str(exc), bind.readiness(entity_id))
    warnings.append(
        "DEGRADED: no entity-specific model exists; scoring with the platform "
        "fallback model explicitly requested by the caller. Not validated for "
        "this entity."
    )
    return model, calib, row["calibration_id"], "global_fallback", warnings, None


def score(
    entity_id: str,
    records: list[dict],
    overrides: dict | None = None,
    model_id: str | None = None,
    write_to_db: bool = True,
    request_id: str | None = None,
    latency_ms: float = 0.0,
    allow_fallback_model: bool = True,
    write_artifact: bool = False,
    publish: bool = False,
) -> dict:
    entity = reg.get_entity(entity_id)
    if entity["status"] == reg.DISABLED:
        raise EntityNotScoreable(f"entity '{entity_id}' is disabled")

    model, calib, calibration_id, calibration_scope, model_warnings, _binding = (
        resolve_entity_model(entity_id, model_id, allow_fallback=allow_fallback_model)
    )
    cfg = resolve(entity_id, overrides, model_meta=model.meta)
    calib_row = {"calibration_id": calibration_id, "scope": calibration_scope}

    raw = records_to_frame(records, entity)
    if len(raw) < MIN_HISTORY_RECORDS:
        raise InsufficientHistory(
            f"{len(raw)} usable records supplied; at least {MIN_HISTORY_RECORDS} "
            f"consecutive records are required (hard floor {ABSOLUTE_MIN_HISTORY}, "
            f"recommended {RECOMMENDED_HISTORY})."
        )
    if raw["entity_id"].nunique() > 1:
        raise InsufficientHistory(
            f"records span {raw['entity_id'].nunique()} entities; send one stream per call"
        )

    # Features are rebuilt with the thresholds FROZEN in this model, never with
    # the current operational values — that is the trained contract.
    feats = apply_value_clipper(
        build_features(raw, windows=model.feature_windows,
                       red_threshold=model.red_model,
                       amber_threshold=model.amber_model),
        model.clip_bounds,
    )
    row = feats.iloc[[-1]]
    X = row[model.feature_cols]
    if X.isna().any(axis=None):
        missing = [c for c in model.feature_cols if X[c].isna().any()]
        raise InsufficientHistory(
            f"feature window incomplete ({len(missing)} unavailable, e.g. {missing[:3]}). "
            f"Records must be consecutive with no gap above "
            f"{cfg['max_sequence_gap_min']:.0f} minutes."
        )

    ts = pd.Timestamp(row["decision_time"].iloc[0])
    ts_start = pd.Timestamp(row["DVRDate"].iloc[0])
    baseline = float(row["Count"].iloc[0])
    traffic_in = float(row["Count_IN"].iloc[0])
    duration = float(row["interval_duration"].iloc[0])

    policy = _policy(cfg)
    inventory = _inventory(cfg)
    session = store.load_session(entity_id, _alert_cfg(cfg))
    reserve = inventory.reserve_scenario
    effective_lead = policy.lead_time_min + reserve.availability_delay_min

    warnings = list(cfg.warnings) + list(model_warnings)
    if entity["status"] == reg.SHADOW:
        warnings.append(
            f"entity is in SHADOW mode ({entity['n_days_history']} days of history); "
            f"the recommendation is logged but should not be surfaced to staff"
        )
    from training.model_config import FrozenModelConfig, compatibility_warnings

    frozen_cfg = getattr(model, "frozen", None) or FrozenModelConfig.from_mapping(
        model.model_config, model.red_model, model.amber_model)
    warnings += compatibility_warnings(
        frozen_cfg,
        {"red_threshold": cfg.values.get("red_threshold"),
         "amber_threshold": cfg.values.get("amber_threshold")},
    )
    if calib_row["scope"] in ("global", "global_fallback"):
        warnings.append(
            "using a global fallback calibrator; shortage probabilities are not "
            "calibrated to this entity"
        )

    advance = session["last_record_time"] is None or ts > session["last_record_time"]
    if advance:
        delivered = session["book"].deliver_due(record_time=ts, record_start_time=ts_start)
        for o in delivered:
            session["surplus"] += o.delivered_quantity
            store.persist_order(o, entity)
        session["surplus"], _ = apply_capacity(session["surplus"], baseline, inventory)
        session["surplus"], _ = consume_surplus(
            session["surplus"], baseline, traffic_in, duration, inventory
        )
        session["last_record_time"] = ts
        session["n_calls"] += 1

    current_sim = observed_inventory(baseline, session["surplus"], inventory)

    # Project the surviving surplus over the horizon using recent traffic only.
    recent_in = float(feats["Count_IN"].tail(3).mean())
    projected = session["surplus"]
    for _ in range(policy.planning_horizon):
        projected, _c = consume_surplus(projected, baseline, recent_in, duration, inventory)

    pred = model.predict_row(X)
    for k in ("pred_min_6", "pred_min_3", "pred_cart_3", "pred_cart_6"):
        pred[k] += projected
    pred["lower_min_6"] = float(calib.lower_bound(np.array([pred["pred_min_6"]]))[0])
    pred["risk"] = float(
        calib.exceedance_probability(
            np.array([pred["pred_min_6"]]), policy.red_threshold
        )[0]
    )

    if advance:
        alert = session["alert"].step(ts, pred["risk"], current_sim)
    else:
        alert = {
            "state": session["alert"].state,
            "new_predictive_alert": False,
            "reactive_critical": current_sim < policy.red_threshold,
        }

    cooldown_active = (
        session["last_decision_time"] is not None
        and (ts - session["last_decision_time"]).total_seconds() / 60.0 < policy.cooldown_min
    )
    reserve_left = _reserve_left(session, cfg, ts)

    plan, order = None, None
    if advance and not cooldown_active:
        plan = plan_replenishment(
            pred=pred,
            current_sim=current_sim,
            observed_cart=baseline,
            pending_qty=session["book"].pending_quantity(),
            policy=policy,
            inventory=inventory,
            reserve_left=reserve_left,
            effective_lead=effective_lead,
        )
    if plan and "skip_reason" not in plan:
        pending_before = session["book"].pending_quantity()
        order = session["book"].place(
            decision_time=ts,
            quantity=plan["quantity"],
            lead_time_min=effective_lead,
            urgency=plan["urgency"],
            risk_at_decision=pred["risk"],
            predicted_min_at_decision=pred["pred_min_6"],
            capacity_rejected=plan["capacity_rejected"],
        )
        order.order_id = store.new_order_id(entity_id)
        store.persist_order(order, entity)
        session["last_decision_time"] = ts
        session["reserve_used_today"] += plan["quantity"]
        session["reserve_day"] = ts.date().isoformat()
        reason = (
            f"risk {pred['risk']:.2f} >= {policy.risk_threshold:.2f}; predicted 40-minute "
            f"minimum {pred['pred_min_6']:.1f} carts against a target of "
            f"{policy.target_stock:.0f}; ordered {plan['quantity']} after netting "
            f"{pending_before:.0f} already in transit"
        )
    elif plan and "skip_reason" in plan:
        reason = f"No action: {plan['skip_reason']}"
    elif not advance:
        # Checked before the cooldown: a replayed record is the more specific
        # and more useful explanation for the caller.
        reason = "No action: record already processed for this entity"
    elif cooldown_active:
        reason = f"No action: within the {policy.cooldown_min:.0f}-minute cooldown"
    else:
        reason = (
            f"No action: risk {pred['risk']:.2f} below the "
            f"{policy.risk_threshold:.2f} threshold"
        )

    if advance:
        store.save_session(session)

    result = {
        "request_id": request_id,
        "entity_id": entity_id,
        "kdvr": entity["kdvr"],
        "chan_no": entity["chan_no"],
        "site_key": entity["site_key"],
        "bay_id": entity.get("bay_id"),
        "entity_status": entity["status"],
        "decision_time": ts.isoformat(),
        "current_cart": baseline,
        "simulated_cart": round(current_sim, 2),
        "forecast": {
            "cart_at_20min": round(pred["pred_cart_3"], 2),
            "cart_at_40min": round(pred["pred_cart_6"], 2),
            "min_within_20min": round(pred["pred_min_3"], 2),
            "min_within_40min": round(pred["pred_min_6"], 2),
            "lower_bound_min_within_40min": round(pred["lower_min_6"], 2),
            "horizon_note": HORIZON_NOTE,
        },
        "shortage_risk": round(pred["risk"], 4),
        "alert_state": alert["state"],
        "new_predictive_alert": bool(alert["new_predictive_alert"]),
        "reactive_critical": bool(alert["reactive_critical"]),
        "action": {
            "should_replenish": order is not None,
            "quantity": int(order.quantity) if order else 0,
            "urgency": plan["urgency"] if plan and "urgency" in plan else "NONE",
            "order_id": order.order_id if order else None,
            "decision_time": ts.isoformat() if order else None,
            "expected_completion_time": (
                order.expected_completion_time.isoformat() if order else None
            ),
            "actionable_before_breach": (
                plan["actionable_before_breach"] if plan and "urgency" in plan else None
            ),
            "minutes_to_predicted_breach": (
                None
                if not plan or "urgency" not in plan
                or not np.isfinite(plan["minutes_to_predicted_breach"])
                else round(plan["minutes_to_predicted_breach"], 1)
            ),
            "reason": reason,
        },
        "pending_carts": session["book"].pending_quantity(),
        "pending_orders": [
            {
                "order_id": o.order_id,
                "quantity": o.quantity,
                "expected_completion_time": o.expected_completion_time.isoformat(),
                "status": o.status,
            }
            for o in session["book"].pending
        ],
        "model_id": model.model_id,
        "calibration_id": calib_row["calibration_id"],
        "calibration_scope": calib_row["scope"],
        "resolved_config": cfg.as_payload(),
        "frozen_model_config": frozen_cfg.as_dict(),
        "config_hash": model.config_hash,
        "warnings": warnings,
        "caveat": CAVEAT,
        "request_id": request_id,
    }

    from server import artifacts

    existing = store.stored_score(entity_id, result["decision_time"])
    is_new_decision = existing is None
    result["idempotent_replay"] = not is_new_decision
    if not is_new_decision:
        prior = artifacts.existing_artifact(entity_id, result["decision_time"])
        result["artifact_path"] = prior.get("artifact_path") if prior else None
        result["artifact_request_id"] = prior.get("request_id") if prior else None
        result["persisted"] = False
        return result

    if write_artifact:
        art = artifacts.write_score_artifact(result, entity)
        result["artifact_path"] = art["artifact_path"]
        result["artifact_request_id"] = art["request_id"]
    if write_to_db:
        result["persisted"] = store.write_score(result, entity, latency_ms)
    if publish and result.get("persisted", False):
        from integrations import events as ev

        ev.publish_scoring(result)
    return result


def _reserve_left(session: dict, cfg, ts: pd.Timestamp) -> float:
    per_day = cfg.values.get("reserve_per_day")
    if per_day is None:
        return np.inf
    today = ts.date().isoformat()
    used = session["reserve_used_today"] if session["reserve_day"] == today else 0.0
    if session["reserve_day"] != today:
        session["reserve_used_today"] = 0.0
        session["reserve_day"] = today
    return max(0.0, float(per_day) - used)


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------
_BUNDLED: pd.DataFrame | None = None


def bundled_telemetry() -> pd.DataFrame:
    global _BUNDLED
    if _BUNDLED is None:
        _BUNDLED = prepare(
            load_raw(BUNDLED_TELEMETRY), include_site_in_entity=INCLUDE_SITE_IN_ENTITY_ID,
            include_bay_in_entity=False
        )
    return _BUNDLED


def backtest(
    entity_id: str,
    records: list[dict] | None = None,
    start: str | None = None,
    end: str | None = None,
    mode: str = "replay",
    overrides: dict | None = None,
    model_id: str | None = None,
    include_decisions: bool = True,
    include_events: bool = True,
    include_series: bool = False,
    label: str | None = None,
    request_id: str | None = None,
) -> tuple[dict, dict]:
    entity = reg.get_entity(entity_id)
    model = mreg.load_model(model_id or entity.get("model_id"))
    cfg = resolve(entity_id, overrides, model_meta=model.meta)
    calib_row = mreg.resolve_calibrator(model.model_id, entity_id)
    calib = mreg.load_calibrator(calib_row["calibration_id"])

    if records:
        raw = records_to_frame(records, entity)
    else:
        raw = bundled_telemetry()
        raw = raw[raw["entity_id"] == entity_id].copy()
        if start:
            raw = raw[raw["decision_time"] >= pd.Timestamp(start)]
        if end:
            raw = raw[raw["decision_time"] <= pd.Timestamp(end)]
        raw = raw.reset_index(drop=True)

    policy = _policy(cfg)
    if len(raw) < MIN_HISTORY_RECORDS + policy.planning_horizon:
        raise InsufficientHistory(
            f"{len(raw)} records available for '{entity_id}'; a backtest needs at least "
            f"{MIN_HISTORY_RECORDS + policy.planning_horizon}."
        )

    inventory = _inventory(cfg)
    frame = build_targets(
        build_features(raw, windows=model.feature_windows,
                       red_threshold=model.red_model,
                       amber_threshold=model.amber_model),
        red=model.red_model,
    )
    frame = apply_value_clipper(frame, model.clip_bounds).reset_index(drop=True)
    frame["global_index"] = frame.index.to_numpy()

    usable = frame[model.feature_cols].notna().all(axis=1)
    origins = frame[usable].copy()
    if not len(origins):
        raise InsufficientHistory("no record had a complete feature window")
    origins = model.predict_frame(origins)
    origins["lower_y_min_6"] = calib.lower_bound(origins["pred_y_min_6"].to_numpy())
    origins["risk"] = calib.exceedance_probability(
        origins["pred_y_min_6"].to_numpy(), policy.red_threshold
    )

    pos = {int(gi): k for k, gi in enumerate(origins["global_index"])}
    origin_positions = {
        i: pos[int(gi)] for i, gi in enumerate(frame["global_index"]) if int(gi) in pos
    }

    if mode == "closed_loop":
        predictor = ClosedLoopPredictor(
            full_frame=frame, feature_cols=model.feature_cols,
            clip_bounds=model.clip_bounds, estimators=model.estimators,
            conformal=calib, red=policy.red_threshold,
            cart_bounds=(0.0, model.cart_ceiling),
            red_model=model.red_model, amber_model=model.amber_model,
            feature_windows=model.feature_windows,
        )
    else:
        predictor = ReplayPredictor(
            pred_min_6=origins["pred_y_min_6"].to_numpy(),
            pred_min_3=origins["pred_y_min_3"].to_numpy(),
            pred_cart_3=origins["pred_y_cart_3"].to_numpy(),
            pred_cart_6=origins["pred_y_cart_6"].to_numpy(),
            conformal=calib, red=policy.red_threshold,
        )

    result = run_simulation(
        frame, origin_positions, predictor, policy, inventory, COST_LAMBDAS
    )
    orders = result["orders"].to_frame()
    att = attribute_events(
        result["series"], result["baseline_events"], result["policy_events"],
        orders, None, policy.red_threshold,
    )

    run_id = f"bt-{entity_id.replace(':', '_')}-{pd.Timestamp.utcnow():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:5]}"
    payload = {
        "request_id": request_id,
        "run_id": run_id,
        "entity_id": entity_id,
        "site_key": entity["site_key"],
        "label": label,
        "mode": mode,
        "model_id": model.model_id,
        "calibration_id": calib_row["calibration_id"],
        "window_start": str(frame["decision_time"].min()),
        "window_end": str(frame["decision_time"].max()),
        "n_records": int(len(frame)),
        "scored_origins": int(len(origins)),
        "metrics": _clean(result["metrics"]),
        "forecast_accuracy": _forecast_accuracy(origins),
        "attribution_summary": _clean(attribution_summary(att)) if len(att) else {},
        "resolved_config": cfg.as_payload(),
        "warnings": list(cfg.warnings),
        "caveat": CAVEAT,
    }
    if include_decisions:
        payload["decisions"] = _records(result["decisions"])
    if include_events:
        payload["shortage_events"] = _records(result["baseline_events"])
    if include_series:
        payload["series"] = _records(
            result["series"][
                ["decision_time", "Count", "sim_cart", "surplus", "pending_quantity"]
            ]
        )

    frames = {
        "decisions": result["decisions"],
        "order_book": orders,
        "baseline_events": result["baseline_events"],
        "policy_events": result["policy_events"],
        "event_attribution": att,
        "series": result["series"][
            ["DVRDate", "decision_time", "entity_id", "Count", "sim_cart", "surplus",
             "pending_quantity", "carts_consumed"]
        ],
        "forecasts": origins[
            ["decision_time", "Count"] + [f"pred_{t}" for t in TARGETS]
            + ["lower_y_min_6", "risk"]
        ],
    }
    return payload, frames


def _forecast_accuracy(origins: pd.DataFrame) -> dict:
    out = {}
    for target, label in (
        ("y_cart_3", "cart_at_20min"), ("y_cart_6", "cart_at_40min"),
        ("y_min_3", "min_within_20min"), ("y_min_6", "min_within_40min"),
    ):
        if target not in origins:
            continue
        m = origins[target].notna()
        if m.sum() == 0:
            continue
        err = origins.loc[m, f"pred_{target}"] - origins.loc[m, target]
        out[f"mae_{label}"] = round(float(err.abs().mean()), 4)
        out[f"rmse_{label}"] = round(float(np.sqrt((err ** 2).mean())), 4)
        out[f"bias_{label}"] = round(float(err.mean()), 4)
    if "y_min_6" in origins:
        m = origins["y_min_6"].notna()
        if m.sum():
            covered = origins.loc[m, "y_min_6"] >= origins.loc[m, "lower_y_min_6"]
            out["lower_bound_coverage"] = round(float(covered.mean()), 4)
            out["n_scored"] = int(m.sum())
    return out
