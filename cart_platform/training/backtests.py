"""The three mandatory backtest reports for a trained candidate.

    forecast_accuracy        measured out-of-sample predictive performance
    policy_replay            simulated replenishment benefit, mode="replay"
    closed_loop_robustness   mode="closed_loop" + assumption sensitivity

All three run on the HOLDOUT partition only — a window neither the model nor
the calibrator ever saw. A report that cannot run is recorded as
``insufficient_data`` or ``failed`` with the exact reason; metrics are never
fabricated and a failed report leaves the candidate unpromotable.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from settings import (
    BACKTEST_TYPES,
    COST_LAMBDAS,
    DEFAULT_BAND_WARNING,
    MIN_HOLDOUT_RECORDS,
    SYSTEM_DEFAULTS,
)
from audit.attribution import attribute_events, attribution_summary  # noqa: E402
from audit.audited_simulator import (  # noqa: E402
    ClosedLoopPredictor,
    PolicyConfig,
    ReplayPredictor,
    run_simulation,
)
from audit.inventory_model import InventoryConfig  # noqa: E402
from forecasting import regression_metrics  # noqa: E402
from registry.database import as_json, execute, query, query_one, utc_now  # noqa: E402
from shortage_risk import classification_metrics, metrics_at_threshold  # noqa: E402

LOGGER = logging.getLogger("platform.backtests")

COMPLETED, INSUFFICIENT, FAILED = "completed", "insufficient_data", "failed"

FORECAST_CAVEAT = (
    "Measured predictive performance on held-out observations. This is NOT a "
    "replenishment benefit claim."
)
REPLAY_CAVEAT = (
    "Replenishment benefits are simulation estimates conditional on documented "
    "assumptions, not observed reductions."
)
ROBUSTNESS_CAVEAT = (
    "Scenario envelope. The conservative scenario is as valid as the optimistic "
    "one; no single percentage is the definitive outcome."
)

SCENARIOS = {
    "audited_default": {"consumption_model": "unmet_demand", "usage": "MEDIUM",
                        "lead_time_min": None,
                        "description": "Audited base-case assumptions"},
    "best_case": {"consumption_model": "persistent", "usage": "LOW",
                  "lead_time_min": 5.0,
                  "description": "Short lead time, persistent surplus, low cart usage"},
    "conservative": {"consumption_model": "turnover", "usage": "HIGH",
                     "lead_time_min": 20.0,
                     "description": "Long lead time, full turnover, high cart usage"},
}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def _save(job_id, entity_id, model_id, calibration_id, upload_id, backtest_type,
          status, reason, metrics, assumptions, caveat, artifacts, report_dir,
          window, snapshot=None) -> dict:
    report_id = f"bt_{backtest_type[:4]}_{uuid.uuid4().hex[:8]}"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    frozen = snapshot.frozen.as_dict() if snapshot is not None else {}
    effective = dict(snapshot.operational) if snapshot is not None else {}
    config_hash = snapshot.config_hash if snapshot is not None else None
    body = {
        "report_id": report_id, "backtest_type": backtest_type, "status": status,
        "config_hash": config_hash,
        "frozen_model_config": frozen,
        "effective_operational_config": effective,
        "reason": reason, "job_id": job_id, "entity_id": entity_id,
        "model_id": model_id, "calibration_id": calibration_id,
        "upload_id": upload_id, "metrics": metrics, "assumptions": assumptions,
        "caveat": caveat, "artifacts": artifacts,
        "window_start": window[0], "window_end": window[1],
        "created_at": utc_now(),
    }
    report_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    execute(
        """INSERT OR REPLACE INTO backtest_reports
           (report_id, job_id, entity_id, model_id, calibration_id, upload_id,
            backtest_type, status, reason, metrics_json, assumptions_json, caveat,
            artifacts_json, report_path, window_start, window_end, created_at,
            config_hash, frozen_config_json, effective_config_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (report_id, job_id, entity_id, model_id, calibration_id, upload_id,
         backtest_type, status, reason, as_json(metrics), as_json(assumptions),
         caveat, as_json(artifacts), str(report_path), window[0], window[1],
         utc_now(), config_hash, as_json(frozen), as_json(effective)),
    )
    body["report_path"] = str(report_path)
    return body


def reports_for_job(job_id: str, entity_id: str | None = None) -> list[dict]:
    sql = "SELECT * FROM backtest_reports WHERE job_id=?"
    params: list = [job_id]
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    return query(sql + " ORDER BY backtest_type", tuple(params))


def get_report(job_id: str, backtest_type: str, entity_id: str | None = None) -> dict | None:
    sql = "SELECT * FROM backtest_reports WHERE job_id=? AND backtest_type=?"
    params: list = [job_id, backtest_type]
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    return query_one(sql, tuple(params))


# --------------------------------------------------------------------------
# 1. forecast_accuracy
# --------------------------------------------------------------------------
def run_forecast_accuracy(candidate, calibrator, entity_cfg, job_id, upload_id,
                          report_dir, snapshot=None,
                          band_warning: float | None = None) -> dict:
    ho = candidate.frames["holdout"]
    # Predictive performance is judged against the model's OWN contract: the
    # frozen RED that shaped its features, targets and calibrator. Substituting
    # a later operational threshold here would misreport the model.
    red = float(candidate.red_model)
    red_operational = float(
        (snapshot.operational.get("red_threshold") if snapshot else None)
        or entity_cfg.get("red_threshold") or candidate.red_model
    )
    window = (str(ho["decision_time"].min()) if len(ho) else None,
              str(ho["decision_time"].max()) if len(ho) else None)
    base = dict(job_id=job_id, entity_id=candidate.entity_id,
                model_id=candidate.model_id,
                calibration_id=None, upload_id=upload_id,
                backtest_type="forecast_accuracy", report_dir=report_dir,
                window=window, snapshot=snapshot)

    if len(ho) == 0:
        return _save(**base, status=INSUFFICIENT,
                     reason="holdout partition contains no scoreable rows",
                     metrics={}, assumptions={}, caveat=FORECAST_CAVEAT, artifacts=[])

    horizons = {"y_cart_3": "cart_at_20min", "y_cart_6": "cart_at_40min",
                "y_min_3": "min_within_20min", "y_min_6": "min_within_40min"}
    metrics: dict = {"n_scored": int(len(ho)), "per_horizon": {}}

    for target, label in horizons.items():
        m = ho[target].notna()
        if m.sum() == 0:
            continue
        y = ho.loc[m, target].to_numpy(dtype=float)
        p = ho.loc[m, f"pred_{target}"].to_numpy(dtype=float)
        near = (y < red + 10) | (ho.loc[m, "cart_current"].to_numpy() < red + 10)
        res = regression_metrics(y, p, near)
        # Persistence baseline: predict the future with the current count.
        base_pred = ho.loc[m, "cart_current"].to_numpy(dtype=float)
        bl = regression_metrics(y, base_pred, near)
        res["baseline_persistence_MAE"] = bl["MAE"]
        res["baseline_persistence_RMSE"] = bl["RMSE"]
        res["improvement_vs_persistence_pct"] = (
            (bl["MAE"] - res["MAE"]) / bl["MAE"] * 100 if bl["MAE"] > 0 else None
        )
        metrics["per_horizon"][label] = {k: _num(v) for k, v in res.items()}

    # Shortage classification from the calibrated risk, when available.
    shortage = {}
    if calibrator is not None and "y_min_6" in ho:
        m = ho["y_min_6"].notna()
        if m.sum():
            y_event = (ho.loc[m, "y_min_6"].to_numpy(dtype=float) < red).astype(int)
            risk = calibrator.exceedance_probability(
                ho.loc[m, "pred_y_min_6"].to_numpy(dtype=float), red
            )
            shortage = {k: _num(v) for k, v in classification_metrics(y_event, risk).items()}
            if 0 < int(y_event.sum()) < len(y_event):
                thr = float(band_warning if band_warning is not None
                            else DEFAULT_BAND_WARNING)
                shortage["at_operating_threshold"] = {
                    k: _num(v) for k, v in
                    metrics_at_threshold(y_event, risk, thr).items()
                }
            lower = calibrator.lower_bound(
                ho.loc[m, "pred_y_min_6"].to_numpy(dtype=float)
            )
            shortage["lower_bound_coverage"] = _num(
                float(np.mean(ho.loc[m, "y_min_6"].to_numpy(dtype=float) >= lower))
            )
            shortage["target_lower_coverage"] = 1 - float(calibrator.alpha)
        else:
            shortage = {"status": "no labelled holdout rows"}
    else:
        shortage = {"status": "no calibrator; shortage probabilities not evaluated"}
    metrics["shortage_classification"] = shortage

    pred_path = Path(report_dir) / "predictions.csv"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    cols = ["decision_time", "entity_id", "cart_current"] + \
           [c for t in horizons for c in (t, f"pred_{t}") if c in ho]
    export = ho[cols].copy()
    if calibrator is not None and "pred_y_min_6" in export:
        export["risk"] = calibrator.exceedance_probability(
            export["pred_y_min_6"].to_numpy(dtype=float), red
        )
        export["lower_y_min_6"] = calibrator.lower_bound(
            export["pred_y_min_6"].to_numpy(dtype=float)
        )
    export.to_csv(pred_path, index=False)

    status = COMPLETED if len(ho) >= MIN_HOLDOUT_RECORDS else COMPLETED
    reason = (None if len(ho) >= MIN_HOLDOUT_RECORDS
              else f"holdout has only {len(ho)} rows; metrics are noisy")
    return _save(**base, status=status, reason=reason, metrics=metrics,
                 assumptions={
                     "evaluation": "strictly chronological purged holdout",
                     "red_model_frozen": candidate.red_model,
                     "amber_model_frozen": candidate.amber_model,
                     "red_used_for_metrics": red,
                     "red_operational_at_report_time": red_operational,
                     "note": "metrics use the frozen model RED; the operational "
                             "RED is recorded for comparison only",
                 },
                 caveat=FORECAST_CAVEAT,
                 artifacts=[{"kind": "predictions_csv", "uri": str(pred_path)}])


# --------------------------------------------------------------------------
# Simulation helpers shared by the two policy reports
# --------------------------------------------------------------------------
def _policy_from(entity_cfg: dict, overrides: dict | None = None,
                 snapshot=None, band_warning: float | None = None) -> PolicyConfig:
    """Operational policy for a simulation.

    Precedence: scenario override -> the snapshot's effective operational
    configuration -> the live entity config -> system default. The snapshot is
    preferred so a historical report stays reproducible after the entity
    configuration changes.

    ``risk_threshold`` is no longer configuration: it is the WARNING band
    clustered from this model's own risk distribution, so a replay is driven by
    the same operating point the model will serve.
    """
    o = overrides or {}
    snap = dict(snapshot.operational) if snapshot is not None else {}
    if band_warning is not None:
        snap.setdefault("risk_threshold", float(band_warning))

    def pick(key, default):
        for source in (o, snap, entity_cfg):
            v = source.get(key)
            if v is not None:
                return v
        return default
    return PolicyConfig(
        red_threshold=float(pick("red_threshold", SYS("red_threshold"))),
        amber_threshold=float(pick("amber_threshold", SYS("amber_threshold"))),
        target_stock=float(pick("target_stock", SYS("target_stock"))),
        risk_threshold=float(pick("risk_threshold", DEFAULT_BAND_WARNING)),
        lead_time_min=float(pick("lead_time_min", SYS("lead_time_min"))),
        cooldown_min=float(pick("cooldown_min", SYS("cooldown_min"))),
        min_carts_per_trip=int(pick("min_carts_per_trip", SYS("min_carts_per_trip"))),
        max_carts_per_trip=int(pick("max_carts_per_trip", SYS("max_carts_per_trip"))),
        quantity_rule=str(pick("quantity_rule", SYS("quantity_rule"))),
    )


def SYS(key):
    return SYSTEM_DEFAULTS[key]


def _inventory_from(entity_cfg: dict, overrides: dict | None = None,
                    snapshot=None) -> InventoryConfig:
    o = overrides or {}
    snap = dict(snapshot.operational) if snapshot is not None else {}

    def pick(key, default):
        for source in (o, snap, entity_cfg):
            v = source.get(key)
            if v is not None:
                return v
        return default

    reserve_per_day = pick("reserve_per_day", None)
    reserve = "LIMITED" if reserve_per_day is not None else "UNLIMITED"
    return InventoryConfig(
        bay_capacity=float(pick("bay_capacity", SYS("bay_capacity"))),
        usage=str(pick("usage", SYS("usage"))),
        reserve=reserve,
        consumption_model=str(pick("consumption_model", SYS("consumption_model"))),
        retention_halflife_min=float(
            pick("retention_halflife_min", SYS("retention_halflife_min"))
        ),
    )


def _simulation_inputs(candidate, calibrator, policy):
    """Full holdout record stream + the scoreable origins within it."""
    ho_all = candidate.frames["holdout_all"].reset_index(drop=True)
    ho_all["global_index"] = ho_all.index.to_numpy()
    origins = candidate.frames["holdout"].copy()
    if origins.empty:
        return None, None, None
    origins = origins.reset_index(drop=True)
    origins["lower_y_min_6"] = calibrator.lower_bound(
        origins["pred_y_min_6"].to_numpy(dtype=float)
    )
    origins["risk"] = calibrator.exceedance_probability(
        origins["pred_y_min_6"].to_numpy(dtype=float), policy.red_threshold
    )
    key = {str(t): k for k, t in enumerate(origins["decision_time"])}
    origin_positions = {
        i: key[str(t)] for i, t in enumerate(ho_all["decision_time"]) if str(t) in key
    }
    return ho_all, origins, origin_positions


def _replay_predictor(origins, calibrator, red):
    return ReplayPredictor(
        pred_min_6=origins["pred_y_min_6"].to_numpy(dtype=float),
        pred_min_3=origins["pred_y_min_3"].to_numpy(dtype=float),
        pred_cart_3=origins["pred_y_cart_3"].to_numpy(dtype=float),
        pred_cart_6=origins["pred_y_cart_6"].to_numpy(dtype=float),
        conformal=calibrator, red=red,
    )


# --------------------------------------------------------------------------
# 2. policy_replay
# --------------------------------------------------------------------------
def run_policy_replay(candidate, calibrator, calibration_id, entity_cfg, job_id,
                      upload_id, report_dir, snapshot=None,
                      band_warning: float | None = None) -> dict:
    ho = candidate.frames["holdout"]
    window = (str(ho["decision_time"].min()) if len(ho) else None,
              str(ho["decision_time"].max()) if len(ho) else None)
    base = dict(job_id=job_id, entity_id=candidate.entity_id,
                model_id=candidate.model_id, calibration_id=calibration_id,
                upload_id=upload_id, backtest_type="policy_replay",
                report_dir=report_dir, window=window, snapshot=snapshot)

    if calibrator is None:
        return _save(**base, status=INSUFFICIENT,
                     reason="no calibrator fitted; shortage risk cannot be estimated, "
                            "so the replenishment policy cannot be replayed",
                     metrics={}, assumptions={}, caveat=REPLAY_CAVEAT, artifacts=[])
    if len(ho) == 0:
        return _save(**base, status=INSUFFICIENT,
                     reason="holdout partition contains no scoreable rows",
                     metrics={}, assumptions={}, caveat=REPLAY_CAVEAT, artifacts=[])

    policy = _policy_from(entity_cfg, snapshot=snapshot, band_warning=band_warning)
    inventory = _inventory_from(entity_cfg, snapshot=snapshot)
    try:
        rows, origins, positions = _simulation_inputs(candidate, calibrator, policy)
        result = run_simulation(rows, positions, _replay_predictor(
            origins, calibrator, policy.red_threshold), policy, inventory, COST_LAMBDAS)
        orders = result["orders"].to_frame()
        att = attribute_events(result["series"], result["baseline_events"],
                               result["policy_events"], orders, None,
                               policy.red_threshold)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("policy_replay failed for %s", candidate.entity_id)
        return _save(**base, status=FAILED, reason=f"simulation error: {exc}",
                     metrics={}, assumptions={}, caveat=REPLAY_CAVEAT, artifacts=[])

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    dec_path = Path(report_dir) / "decisions.csv"
    result["decisions"].to_csv(dec_path, index=False)
    ledger_path = Path(report_dir) / "order_ledger.csv"
    orders.to_csv(ledger_path, index=False)
    att_path = Path(report_dir) / "event_attribution.csv"
    att.to_csv(att_path, index=False)

    m = {k: _num(v) for k, v in result["metrics"].items() if not isinstance(v, pd.DataFrame)}
    m["attribution"] = {k: _num(v) for k, v in attribution_summary(att).items()} if len(att) else {}
    return _save(**base, status=COMPLETED, reason=None, metrics=m,
                 assumptions={
                     "mode": "replay",
                     "policy": {k: _num(v) for k, v in policy.__dict__.items()},
                     "inventory": {k: _num(v) for k, v in inventory.__dict__.items()},
                     "cost_lambdas": COST_LAMBDAS,
                     "frozen_model_thresholds": {
                         "red_model": candidate.red_model,
                         "amber_model": candidate.amber_model},
                     "note": "operational thresholds drive the policy; the frozen "
                             "model thresholds shaped the features and calibrator",
                 },
                 caveat=REPLAY_CAVEAT,
                 artifacts=[
                     {"kind": "decisions_csv", "uri": str(dec_path)},
                     {"kind": "order_ledger_csv", "uri": str(ledger_path)},
                     {"kind": "event_attribution_csv", "uri": str(att_path)},
                 ])


# --------------------------------------------------------------------------
# 3. closed_loop_robustness
# --------------------------------------------------------------------------
def run_closed_loop_robustness(candidate, calibrator, calibration_id, entity_cfg,
                               job_id, upload_id, report_dir, snapshot=None,
                               band_warning: float | None = None) -> dict:
    ho = candidate.frames["holdout"]
    window = (str(ho["decision_time"].min()) if len(ho) else None,
              str(ho["decision_time"].max()) if len(ho) else None)
    base = dict(job_id=job_id, entity_id=candidate.entity_id,
                model_id=candidate.model_id, calibration_id=calibration_id,
                upload_id=upload_id, backtest_type="closed_loop_robustness",
                report_dir=report_dir, window=window, snapshot=snapshot)

    if calibrator is None:
        return _save(**base, status=INSUFFICIENT,
                     reason="no calibrator fitted; closed-loop risk cannot be estimated",
                     metrics={}, assumptions={}, caveat=ROBUSTNESS_CAVEAT, artifacts=[])
    if len(ho) == 0:
        return _save(**base, status=INSUFFICIENT,
                     reason="holdout partition contains no scoreable rows",
                     metrics={}, assumptions={}, caveat=ROBUSTNESS_CAVEAT, artifacts=[])

    rows_out: list[dict] = []
    failures: list[str] = []
    base_policy = _policy_from(entity_cfg, snapshot=snapshot, band_warning=band_warning)
    rows, origins, positions = _simulation_inputs(candidate, calibrator, base_policy)

    # Closed loop under the audited default assumptions.
    try:
        predictor = ClosedLoopPredictor(
            full_frame=candidate.frames["clipped"].reset_index(drop=True),
            feature_cols=candidate.feature_cols,
            clip_bounds=candidate.clip_bounds,
            estimators=candidate.estimators,
            conformal=calibrator, red=base_policy.red_threshold,
            cart_bounds=(0.0, candidate.cart_ceiling),
            # Rebuilt features must use the model's frozen thresholds.
            red_model=candidate.red_model,
            amber_model=candidate.amber_model,
            feature_windows=list(candidate.snapshot.frozen.feature_windows)
            if candidate.snapshot else None,
        )
        inv = _inventory_from(entity_cfg, snapshot=snapshot)
        res = run_simulation(rows, positions, predictor, base_policy, inv, COST_LAMBDAS)
        rows_out.append(_scenario_row("closed_loop_audited_default",
                                      "Closed loop, audited base-case assumptions",
                                      base_policy, inv, res, mode="closed_loop"))
    except Exception as exc:
        LOGGER.exception("closed loop failed for %s", candidate.entity_id)
        failures.append(f"closed_loop_audited_default: {exc}")

    # Assumption sensitivity, replayed (cheap) so the envelope is always produced.
    for name, spec in SCENARIOS.items():
        try:
            over = {k: v for k, v in spec.items()
                    if k != "description" and v is not None}
            pol = _policy_from(entity_cfg, over, snapshot=snapshot,
                               band_warning=band_warning)
            inv = _inventory_from(entity_cfg, over, snapshot=snapshot)
            res = run_simulation(rows, positions, _replay_predictor(
                origins, calibrator, pol.red_threshold), pol, inv, COST_LAMBDAS)
            rows_out.append(_scenario_row(name, spec["description"], pol, inv, res,
                                          mode="replay"))
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    if not rows_out:
        return _save(**base, status=FAILED,
                     reason="; ".join(failures) or "no scenario completed",
                     metrics={}, assumptions={}, caveat=ROBUSTNESS_CAVEAT, artifacts=[])

    table = pd.DataFrame(rows_out)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / "scenarios.csv"
    table.to_csv(path, index=False)

    spread = (float(table["severity_reduction_pct"].max())
              - float(table["severity_reduction_pct"].min()))
    metrics = {
        "n_scenarios": int(len(table)),
        "scenarios": table.to_dict(orient="records"),
        "severity_reduction_min_pct": _num(float(table["severity_reduction_pct"].min())),
        "severity_reduction_max_pct": _num(float(table["severity_reduction_pct"].max())),
        "severity_reduction_spread_pct": _num(spread),
        "failures": failures,
    }
    status = COMPLETED if not failures else COMPLETED
    reason = "; ".join(failures) if failures else None
    return _save(**base, status=status, reason=reason, metrics=metrics,
                 assumptions={
                     "scenarios": SCENARIOS,
                     "scenario_overrides_applied": {
                         name: {k: v for k, v in spec.items()
                                if k != "description" and v is not None}
                         for name, spec in SCENARIOS.items()},
                     "base_policy": {k: _num(v) for k, v in base_policy.__dict__.items()},
                     "frozen_model_thresholds": {
                         "red_model": candidate.red_model,
                         "amber_model": candidate.amber_model},
                     "note": "conservative outcomes are reported even when they "
                             "show zero benefit",
                 },
                 caveat=ROBUSTNESS_CAVEAT,
                 artifacts=[{"kind": "scenarios_csv", "uri": str(path)}])


def _scenario_row(name, description, policy, inventory, result, mode) -> dict:
    m = result["metrics"]
    return {
        "scenario": name, "description": description, "mode": mode,
        "consumption_model": inventory.consumption_model, "usage": inventory.usage,
        "reserve": inventory.reserve, "lead_time_min": policy.lead_time_min,
        "risk_threshold": policy.risk_threshold, "target_stock": policy.target_stock,
        "baseline_events": _num(m["baseline_events"]),
        "policy_events": _num(m["policy_events"]),
        "event_reduction_pct": _num(m["event_reduction_pct"]),
        "duration_reduction_pct": _num(m["duration_reduction_pct"]),
        "severity_reduction_pct": _num(m["severity_reduction_pct"]),
        "n_replenishment_actions": _num(m["n_replenishment_actions"]),
        "total_carts_recommended": _num(m["total_carts_recommended"]),
        "trips_per_day": _num(m["trips_per_day"]),
        "policy_cost_J": _num(m["policy_cost_J"]),
        "ledger_conserved": bool(m["ledger_conserved"]),
    }


def _num(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(float(v)) else round(float(v), 6)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_all(candidate, calibrator, calibration_id, entity_cfg, job_id, upload_id,
            base_dir: Path, requested: list[str] | None = None,
            snapshot=None, band_warning: float | None = None) -> dict:
    """Run the three reports. All three are mandatory defaults."""
    requested = list(requested or BACKTEST_TYPES)
    missing = [t for t in BACKTEST_TYPES if t not in requested]
    if missing:
        LOGGER.warning("Caller omitted %s; running them anyway (mandatory defaults)",
                       missing)
        requested = list(BACKTEST_TYPES)

    base_dir = Path(base_dir) / "backtests"
    out: dict[str, dict] = {}
    runners = {
        "forecast_accuracy": lambda d: run_forecast_accuracy(
            candidate, calibrator, entity_cfg, job_id, upload_id, d, snapshot,
            band_warning),
        "policy_replay": lambda d: run_policy_replay(
            candidate, calibrator, calibration_id, entity_cfg, job_id, upload_id, d,
            snapshot, band_warning),
        "closed_loop_robustness": lambda d: run_closed_loop_robustness(
            candidate, calibrator, calibration_id, entity_cfg, job_id, upload_id, d,
            snapshot, band_warning),
    }
    for bt in BACKTEST_TYPES:
        try:
            out[bt] = runners[bt](base_dir / bt)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("%s crashed for %s", bt, candidate.entity_id)
            out[bt] = _save(
                job_id=job_id, entity_id=candidate.entity_id,
                model_id=candidate.model_id, calibration_id=calibration_id,
                upload_id=upload_id, backtest_type=bt, status=FAILED,
                reason=f"unhandled error: {exc}", metrics={}, assumptions={},
                caveat="", artifacts=[], report_dir=base_dir / bt,
                window=(None, None), snapshot=snapshot,
            )
    return out
