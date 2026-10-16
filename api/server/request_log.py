"""Structured per-call logging.

Every API call produces three artefacts:

1. a JSONL line in ``requests.jsonl`` (request metadata + latency + outcome);
2. a flat row in ``score_results.csv`` or ``backtest_results.csv`` so results
   can be opened directly in a spreadsheet;
3. for backtests, a full artefact folder under ``logs/runs/<run_id>/``.

Logging never raises: a failure to write a log must not fail an API call.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from server.settings import (
    BACKTEST_LOG,
    ERROR_LOG,
    REQUEST_LOG,
    RUN_DIR,
    SCORE_LOG,
    SERVER_LOG,
)

_LOCK = threading.Lock()
LOGGER = logging.getLogger("cart_api")


def configure_logging(level: int = logging.INFO) -> None:
    LOGGER.setLevel(level)
    if LOGGER.handlers:
        return
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(SERVER_LOG)
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


def new_request_id(prefix: str = "req") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def _append_jsonl(path: Path, payload: dict) -> None:
    try:
        with _LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=_json_default) + "\n")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to write %s: %s", path.name, exc)


def _append_csv(path: Path, row: dict) -> None:
    try:
        df = pd.DataFrame([row])
        with _LOCK:
            header = not path.exists()
            df.to_csv(path, mode="a", header=header, index=False)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to write %s: %s", path.name, exc)


def log_request(
    request_id: str,
    endpoint: str,
    method: str,
    status: str,
    latency_ms: float,
    client: str | None = None,
    request_summary: dict | None = None,
    response_summary: dict | None = None,
    error: str | None = None,
) -> None:
    _append_jsonl(
        REQUEST_LOG,
        {
            "timestamp": _utc_now(),
            "request_id": request_id,
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "latency_ms": round(float(latency_ms), 2),
            "client": client,
            "request": request_summary or {},
            "response": response_summary or {},
            "error": error,
        },
    )
    if error:
        _append_jsonl(
            ERROR_LOG,
            {
                "timestamp": _utc_now(),
                "request_id": request_id,
                "endpoint": endpoint,
                "error": error,
            },
        )


def log_score_result(request_id: str, payload: dict, latency_ms: float) -> None:
    """One flat row per scoring decision."""
    f = payload.get("forecast", {})
    a = payload.get("action", {})
    _append_csv(
        SCORE_LOG,
        {
            "logged_at": _utc_now(),
            "request_id": request_id,
            "site_key": payload.get("site_key"),
            "decision_time": payload.get("decision_time"),
            "current_cart": payload.get("current_cart"),
            "simulated_cart": payload.get("simulated_cart"),
            "pred_cart_20min": f.get("cart_at_20min"),
            "pred_cart_40min": f.get("cart_at_40min"),
            "pred_min_20min": f.get("min_within_20min"),
            "pred_min_40min": f.get("min_within_40min"),
            "lower_bound_min_40min": f.get("lower_bound_min_within_40min"),
            "shortage_risk": payload.get("shortage_risk"),
            "alert_state": payload.get("alert_state"),
            "new_alert": payload.get("new_predictive_alert"),
            "reactive_critical": payload.get("reactive_critical"),
            "should_replenish": a.get("should_replenish"),
            "quantity": a.get("quantity"),
            "urgency": a.get("urgency"),
            "order_id": a.get("order_id"),
            "deliver_by": a.get("expected_completion_time"),
            "pending_carts": payload.get("pending_carts"),
            "reason": a.get("reason"),
            "latency_ms": round(float(latency_ms), 2),
        },
    )


def log_backtest_result(request_id: str, run_id: str, payload: dict, latency_ms: float) -> None:
    m = payload.get("metrics", {})
    _append_csv(
        BACKTEST_LOG,
        {
            "logged_at": _utc_now(),
            "request_id": request_id,
            "run_id": run_id,
            "mode": payload.get("mode"),
            "records": m.get("n_records"),
            "scored_origins": payload.get("scored_origins"),
            "window_start": payload.get("window_start"),
            "window_end": payload.get("window_end"),
            "risk_threshold": m.get("risk_threshold"),
            "target_stock": m.get("target_stock"),
            "lead_time_min": m.get("lead_time_min"),
            "usage_scenario": m.get("usage_scenario"),
            "reserve_scenario": m.get("reserve_scenario"),
            "consumption_model": m.get("consumption_model"),
            "baseline_events": m.get("baseline_events"),
            "policy_events": m.get("policy_events"),
            "event_reduction_pct": m.get("event_reduction_pct"),
            "duration_reduction_pct": m.get("duration_reduction_pct"),
            "severity_reduction_pct": m.get("severity_reduction_pct"),
            "replenishment_actions": m.get("n_replenishment_actions"),
            "carts_moved": m.get("total_carts_recommended"),
            "trips_per_day": m.get("trips_per_day"),
            "policy_cost_J": m.get("policy_cost_J"),
            "baseline_cost_J": m.get("baseline_cost_J"),
            "forecast_mae_min40": payload.get("forecast_accuracy", {}).get("mae_min_within_40min"),
            "conserved": m.get("ledger_conserved"),
            "latency_ms": round(float(latency_ms), 2),
        },
    )


def run_artifact_dir(run_id: str) -> Path:
    d = RUN_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_artifacts(run_id: str, frames: dict, payload: dict) -> Path:
    """Persist the full backtest output so any call can be reproduced."""
    d = run_artifact_dir(run_id)
    try:
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame) and len(df):
                df.to_csv(d / f"{name}.csv", index=False)
        with open(d / "response.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to persist run artefacts for %s: %s", run_id, exc)
    return d
