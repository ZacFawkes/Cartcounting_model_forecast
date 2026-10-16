"""Event construction and enqueueing for each result category.

Called from the training job runner and the scoring service. Every function is
best-effort: a publishing failure must never fail the local operation that
produced the result.
"""
from __future__ import annotations

import logging

from registry.database import utc_now
from integrations import outbox
from integrations.result_events import (
    BACKTEST_COMPLETED,
    CALIBRATION_COMPLETED,
    SCORING_RECORDED,
    TRAINING_COMPLETED,
    ResultEvent,
    backtest_event_id,
    calibration_event_id,
    scoring_event_id,
    training_event_id,
)

LOGGER = logging.getLogger("platform.events")

TRAINING_CAVEAT = (
    "Training metrics are measured on held-out observations for this entity only."
)
CALIBRATION_CAVEAT = (
    "Calibration diagnostics are computed on an out-of-training window; a "
    "regression versus the uncalibrated reference is reported, not hidden."
)
SCORING_CAVEAT = (
    "Advisory recommendation. No execution is implied unless a separate action "
    "log confirms it."
)


def _safe(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:  # pragma: no cover - publishing is never critical
            LOGGER.exception("Failed to enqueue result event")
            return False
    return wrapper


def _site_key(entity_id: str) -> str | None:
    from registry.entities import try_get_entity

    row = try_get_entity(entity_id)
    return None if row is None else str(row.get("site_key"))


@_safe
def publish_training(job_id, entity_id, model_id, result: dict, upload_id=None) -> bool:
    payload = {
        "status": result.get("status"),
        "error": result.get("error"),
        "detail": result.get("detail"),
        "n_features": result.get("n_features"),
        "selected_models": result.get("selected_models"),
        "frozen_thresholds": result.get("frozen_thresholds"),
        "split_windows": result.get("split_windows"),
        "metrics": result.get("metrics"),
        "config_snapshot": result.get("config_snapshot"),
        "config_hash": (result.get("config_snapshot") or {}).get("config_hash"),
        "effective_config": result.get("effective_config"),
        "promotable": result.get("promotable"),
        "blocking_reasons": result.get("blocking_reasons"),
        "promoted": result.get("promoted", False),
        "warnings": result.get("warnings", []),
        "artifacts": [
            {"kind": "model_metadata_json", "uri": result.get("metadata_path")},
            {"kind": "model_bundle", "uri": result.get("artifact_path"),
             "note": "server-local reference; binary artifacts are never published"},
        ],
        "caveat": TRAINING_CAVEAT,
    }
    event = ResultEvent(
        event_id=training_event_id(job_id, entity_id, result.get("status")),
        event_type=TRAINING_COMPLETED, payload=payload, entity_id=entity_id,
        site_key=_site_key(entity_id), upload_id=upload_id, job_id=job_id,
        model_id=model_id,
    )
    return outbox.enqueue(event)


@_safe
def publish_calibration(job_id, entity_id, model_id, calibration: dict,
                        upload_id=None) -> bool:
    payload = {
        "calibrator_id": calibration.get("calibration_id"),
        "status": calibration.get("status"),
        "scope": calibration.get("scope"),
        "method": calibration.get("method"),
        "metrics": calibration.get("metrics"),
        "window_start": calibration.get("window_start"),
        "window_end": calibration.get("window_end"),
        "calibration_red": calibration.get("calibration_red"),
        "config_hash": calibration.get("config_hash"),
        "warnings": calibration.get("warnings", []),
        "artifacts": [
            {"kind": "reliability_csv", "uri": calibration.get("reliability_path")},
            {"kind": "calibrator_npy", "uri": calibration.get("artifact_path")},
        ],
        "caveat": CALIBRATION_CAVEAT,
    }
    event = ResultEvent(
        event_id=calibration_event_id(job_id, entity_id,
                                      calibration.get("calibration_id")),
        event_type=CALIBRATION_COMPLETED, payload=payload, entity_id=entity_id,
        site_key=_site_key(entity_id), upload_id=upload_id, job_id=job_id,
        model_id=model_id,
    )
    return outbox.enqueue(event)


@_safe
def publish_backtest(job_id, entity_id, model_id, report: dict, upload_id=None) -> bool:
    payload = {
        "backtest_type": report["backtest_type"],
        "report_id": report["report_id"],
        "status": report["status"],
        "reason": report.get("reason"),
        "metrics": report.get("metrics"),
        "assumptions": report.get("assumptions"),
        "frozen_model_config": report.get("frozen_model_config"),
        "effective_operational_config": report.get("effective_operational_config"),
        "config_hash": report.get("config_hash"),
        "artifacts": report.get("artifacts", []),
        "window_start": report.get("window_start"),
        "window_end": report.get("window_end"),
        "caveat": report.get("caveat"),
    }
    event = ResultEvent(
        event_id=backtest_event_id(job_id, entity_id, report["backtest_type"],
                                   report["status"]),
        event_type=BACKTEST_COMPLETED, payload=payload, entity_id=entity_id,
        site_key=_site_key(entity_id), upload_id=upload_id, job_id=job_id,
        model_id=model_id,
    )
    return outbox.enqueue(event)


@_safe
def publish_scoring(result: dict) -> bool:
    payload = {
        "request_id": result.get("request_id"),
        "decision_time": result["decision_time"],
        "current_cart": result["current_cart"],
        "simulated_cart": result["simulated_cart"],
        "forecast": result["forecast"],
        "shortage_risk": result["shortage_risk"],
        "alert_state": result["alert_state"],
        "new_predictive_alert": result["new_predictive_alert"],
        "reactive_critical": result["reactive_critical"],
        "action": result["action"],
        "pending_carts": result["pending_carts"],
        "calibration_id": result.get("calibration_id"),
        "scored_at": utc_now(),
        "resolved_config": result["resolved_config"],
        "frozen_thresholds": {
            "red_model": result["resolved_config"].get("red_model", {}).get("value"),
            "amber_model": result["resolved_config"].get("amber_model", {}).get("value"),
        },
        "frozen_model_config": result.get("frozen_model_config"),
        "config_hash": result.get("config_hash"),
        "calibration_scope": result.get("calibration_scope"),
        "warnings": result.get("warnings", []),
        "artifact": {"kind": "scoring_csv", "uri": result.get("artifact_path")},
        "caveat": SCORING_CAVEAT,
    }
    event = ResultEvent(
        event_id=scoring_event_id(result["entity_id"], result["decision_time"],
                                  result["model_id"]),
        event_type=SCORING_RECORDED, payload=payload,
        entity_id=result["entity_id"], site_key=str(result.get("site_key")),
        upload_id=None, job_id=None, model_id=result["model_id"],
    )
    return outbox.enqueue(event)
