"""Immutable per-decision CSV artifacts for scoring calls.

One file per **new successful scoring decision**:

    logs/scores/<entity>/<YYYY-MM-DD>/<request_id>.csv

Written atomically (temp file + ``os.replace``) and made read-only. An
idempotent retry returns the original artifact instead of creating a duplicate.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from settings import SCORE_ARTIFACT_DIR

from registry.database import execute, query, query_one, utc_now  # noqa: E402

LOGGER = logging.getLogger("platform.artifacts")


def _entity_dir(entity_id: str, decision_time: str) -> Path:
    day = str(decision_time)[:10] or "unknown"
    return Path(SCORE_ARTIFACT_DIR) / entity_id.replace(":", "_") / day


def existing_artifact(entity_id: str, decision_time: str) -> dict | None:
    return query_one(
        "SELECT * FROM scoring_artifacts WHERE entity_id=? AND decision_time=?",
        (entity_id, decision_time),
    )


def get_artifact(request_id: str) -> dict | None:
    return query_one("SELECT * FROM scoring_artifacts WHERE request_id=?", (request_id,))


def list_artifacts(entity_id: str | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM scoring_artifacts WHERE 1=1"
    params: list = []
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return query(sql, tuple(params))


def _flatten(result: dict, entity: dict) -> dict:
    f = result["forecast"]
    a = result["action"]
    cfg = result.get("resolved_config", {}) or {}
    return {
        "request_id": result.get("request_id"),
        "entity_id": result["entity_id"],
        "kdvr": entity["kdvr"],
        "chan_no": entity["chan_no"],
        "site_key": entity["site_key"],
        "entity_status": result.get("entity_status"),
        "decision_time": result["decision_time"],
        "scored_at": utc_now(),
        "current_cart": result["current_cart"],
        "simulated_cart": result["simulated_cart"],
        "pred_cart_20min": f["cart_at_20min"],
        "pred_cart_40min": f["cart_at_40min"],
        "pred_min_20min": f["min_within_20min"],
        "pred_min_40min": f["min_within_40min"],
        "lower_bound_min_40min": f["lower_bound_min_within_40min"],
        "shortage_risk": result["shortage_risk"],
        "alert_state": result["alert_state"],
        "new_predictive_alert": result["new_predictive_alert"],
        "reactive_critical": result["reactive_critical"],
        "should_replenish": a["should_replenish"],
        "quantity": a["quantity"],
        "urgency": a["urgency"],
        "order_id": a["order_id"],
        "deliver_by": a["expected_completion_time"],
        "minutes_to_predicted_breach": a.get("minutes_to_predicted_breach"),
        "reason": a["reason"],
        "pending_carts": result["pending_carts"],
        "model_id": result["model_id"],
        "calibration_id": result["calibration_id"],
        "calibration_scope": result["calibration_scope"],
        "red_model": (cfg.get("red_model") or {}).get("value"),
        "amber_model": (cfg.get("amber_model") or {}).get("value"),
        "red_threshold_eval": (cfg.get("red_threshold") or {}).get("value"),
        "red_threshold_source": (cfg.get("red_threshold") or {}).get("source"),
        "target_stock": (cfg.get("target_stock") or {}).get("value"),
        "target_stock_source": (cfg.get("target_stock") or {}).get("source"),
        "risk_threshold": (cfg.get("risk_threshold") or {}).get("value"),
        "risk_threshold_source": (cfg.get("risk_threshold") or {}).get("source"),
        "critical_risk_threshold": (
            cfg.get("critical_risk_threshold") or {}).get("value"),
        "critical_risk_threshold_source": (
            cfg.get("critical_risk_threshold") or {}).get("source"),
        "alert_band_source": (cfg.get("risk_threshold") or {}).get("source"),
        "resolved_config_json": json.dumps(cfg, default=str),
        "warnings_json": json.dumps(result.get("warnings", []), default=str),
        "caveat": result.get("caveat"),
    }


def write_score_artifact(result: dict, entity: dict) -> dict:
    """Atomically write one immutable CSV row for this decision."""
    entity_id = result["entity_id"]
    decision_time = result["decision_time"]
    prior = existing_artifact(entity_id, decision_time)
    if prior is not None:
        return {"request_id": prior["request_id"],
                "artifact_path": prior["artifact_path"], "created": False}

    request_id = result.get("request_id") or f"score-{uuid.uuid4().hex[:12]}"
    out_dir = _entity_dir(entity_id, decision_time)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{request_id}.csv"

    row = _flatten({**result, "request_id": request_id}, entity)
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp")
    os.close(fd)
    try:
        pd.DataFrame([row]).to_csv(tmp, index=False)
        os.replace(tmp, dest)
        dest.chmod(0o444)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)

    execute(
        """INSERT OR IGNORE INTO scoring_artifacts
           (request_id, entity_id, decision_time, artifact_path, model_id,
            calibration_id, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (request_id, entity_id, decision_time, str(dest), result.get("model_id"),
         result.get("calibration_id"), utc_now()),
    )
    LOGGER.info("Scoring artifact written: %s", dest)
    return {"request_id": request_id, "artifact_path": str(dest), "created": True}


def cleanup(retention_days: int = 90, entity_id: str | None = None) -> int:
    """Delete artifacts older than the retention window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    sql = "SELECT * FROM scoring_artifacts WHERE created_at < ?"
    params: list = [cutoff]
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    removed = 0
    for row in query(sql, tuple(params)):
        p = Path(row["artifact_path"])
        if p.exists():
            p.chmod(0o644)
            p.unlink(missing_ok=True)
        execute("DELETE FROM scoring_artifacts WHERE request_id=?", (row["request_id"],))
        removed += 1
    if removed:
        LOGGER.info("Cleaned up %d scoring artifact(s) older than %d days",
                    removed, retention_days)
    return removed
