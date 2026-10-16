"""Entity-keyed logging: every record carries entity_id and site_key."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pandas as pd

from settings import LOG_DIR, RUN_DIR, SERVER_LOG
from registry.database import as_json, execute, query, utc_now

LOGGER = logging.getLogger("platform")
_LOCK = threading.Lock()


def configure_logging(level: int = logging.INFO) -> None:
    for name in ("platform", "platform.db", "platform.registry", "platform.models",
                 "platform.resolver", "platform.service", "platform.store"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
    if LOGGER.handlers:
        return
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(SERVER_LOG)
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    LOGGER.propagate = False


def log_request(
    request_id: str,
    endpoint: str,
    method: str,
    status: str,
    latency_ms: float,
    entity_id: str | None = None,
    site_key: int | None = None,
    status_code: int | None = None,
    client: str | None = None,
    request_summary: dict | None = None,
    response_summary: dict | None = None,
    error: str | None = None,
) -> None:
    try:
        execute(
            """INSERT INTO api_requests
               (request_id, entity_id, site_key, endpoint, method, status, status_code,
                latency_ms, client, request_json, response_json, error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                request_id, entity_id, site_key, endpoint, method, status, status_code,
                round(float(latency_ms), 2), client, as_json(request_summary or {}),
                as_json(response_summary or {}), error, utc_now(),
            ),
        )
    except Exception as exc:  # pragma: no cover — logging must never fail a call
        LOGGER.warning("request log write failed: %s", exc)


def log_backtest(payload: dict, latency_ms: float, artifact_dir: str) -> None:
    m = payload.get("metrics", {})
    acc = payload.get("forecast_accuracy", {})
    try:
        execute(
            """INSERT OR REPLACE INTO backtest_runs
               (run_id, request_id, entity_id, site_key, label, mode, model_id,
                window_start, window_end, n_records, scored_origins, baseline_events,
                policy_events, event_reduction_pct, duration_reduction_pct,
                severity_reduction_pct, replenishment_actions, carts_moved,
                trips_per_day, policy_cost_j, baseline_cost_j, mae_min_40min,
                lower_bound_coverage, resolved_config, artifact_dir, latency_ms,
                created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload["run_id"], payload.get("request_id"), payload["entity_id"],
                payload.get("site_key"), payload.get("label"), payload.get("mode"),
                payload.get("model_id"), payload.get("window_start"),
                payload.get("window_end"), payload.get("n_records"),
                payload.get("scored_origins"), m.get("baseline_events"),
                m.get("policy_events"), m.get("event_reduction_pct"),
                m.get("duration_reduction_pct"), m.get("severity_reduction_pct"),
                m.get("n_replenishment_actions"), m.get("total_carts_recommended"),
                m.get("trips_per_day"), m.get("policy_cost_J"), m.get("baseline_cost_J"),
                acc.get("mae_min_within_40min"), acc.get("lower_bound_coverage"),
                as_json(payload.get("resolved_config", {})), artifact_dir,
                round(float(latency_ms), 2), utc_now(),
            ),
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("backtest log write failed: %s", exc)


def save_run_artifacts(run_id: str, frames: dict, payload: dict) -> Path:
    d = RUN_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    try:
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame) and len(df):
                df.to_csv(d / f"{name}.csv", index=False)
        with open(d / "response.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("artefact write failed for %s: %s", run_id, exc)
    return d


def recent_requests(entity_id: str | None = None, status: str | None = None,
                    limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM api_requests WHERE 1=1"
    params: list = []
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return query(sql, tuple(params))


def recent_backtests(entity_id: str | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM backtest_runs WHERE 1=1"
    params: list = []
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return query(sql, tuple(params))
