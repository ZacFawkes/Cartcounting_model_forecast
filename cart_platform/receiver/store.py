"""Idempotent inserts and filtered queries for the receiver."""
from __future__ import annotations

import hashlib
import json
import logging

from receiver.database import as_json, execute, from_json, query, query_one, utc_now

LOGGER = logging.getLogger("receiver.store")

CREATED, REPLAY, CONFLICT = "created", "replay", "conflict"

TABLE_FOR_TYPE = {
    "training.completed": "training_results",
    "calibration.completed": "calibration_results",
    "backtest.completed": "backtest_results",
    "scoring.recorded": "scoring_results",
}


def canonical_hash(payload) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _event_row(event: dict) -> dict:
    return {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "schema_version": int(event.get("schema_version", 1)),
        "entity_id": event.get("entity_id"),
        "site_key": None if event.get("site_key") is None else str(event["site_key"]),
        "upload_id": event.get("upload_id"),
        "job_id": event.get("job_id"),
        "model_id": event.get("model_id"),
        "occurred_at": event.get("occurred_at"),
    }


def store_event(event: dict) -> tuple[str, dict]:
    """Insert an event idempotently.

    Returns ``(outcome, record)`` where outcome is ``created`` for a new event,
    ``replay`` for a byte-identical resend, or ``conflict`` when the same
    ``event_id`` arrives with a different canonical payload — historical results
    are never silently overwritten.
    """
    eid = event["event_id"]
    digest = canonical_hash(event.get("payload", {}))
    existing = query_one("SELECT * FROM received_events WHERE event_id=?", (eid,))
    if existing is not None:
        if existing["payload_hash"] == digest:
            return REPLAY, existing
        return CONFLICT, existing

    base = _event_row(event)
    execute(
        """INSERT INTO received_events
           (event_id, event_type, schema_version, entity_id, site_key, upload_id,
            job_id, model_id, occurred_at, received_at, payload_hash, payload_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (base["event_id"], base["event_type"], base["schema_version"],
         base["entity_id"], base["site_key"], base["upload_id"], base["job_id"],
         base["model_id"], base["occurred_at"], utc_now(), digest,
         as_json(event.get("payload", {}))),
    )
    _store_projection(event, base)
    row = query_one("SELECT * FROM received_events WHERE event_id=?", (eid,))
    LOGGER.info("Stored %s event %s (entity=%s)", base["event_type"], eid,
                base["entity_id"])
    return CREATED, row


def _split_entity(entity_id: str | None) -> tuple[int | None, int | None]:
    """Entity id -> (kdvr, chan_no), whatever shape the platform publishes.

    Accepts "KDVR:ChanNo", "site:KDVR:ChanNo", "KDVR:ChanNo:bay" and
    "site:KDVR:ChanNo:bay". The numeric pair is located positionally rather
    than from the end, because a bay id is not numeric.
    """
    if not entity_id:
        return None, None
    parts = str(entity_id).split(":")
    if len(parts) >= 4:
        pair = parts[1:3]
    elif len(parts) == 3:
        pair = parts[:2] if not parts[2].lstrip("-").isdigit() else parts[1:3]
    elif len(parts) == 2:
        pair = parts
    else:
        return None, None
    try:
        return int(pair[0]), int(pair[1])
    except (ValueError, IndexError):
        return None, None


def _store_projection(event: dict, base: dict) -> None:
    p = event.get("payload", {}) or {}
    etype = base["event_type"]
    received = utc_now()
    cfg_hash = p.get("config_hash")
    frozen = p.get("frozen_model_config") or p.get("frozen_thresholds")
    effective = (p.get("effective_operational_config")
                 or p.get("effective_config_at_training")
                 or p.get("effective_config"))

    if etype == "training.completed":
        execute(
            """INSERT INTO training_results
               (event_id, job_id, entity_id, site_key, upload_id, model_id, status,
                occurred_at, received_at, metrics_json, metadata_json,
                config_hash, frozen_config_json, effective_config_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base["event_id"], base["job_id"], base["entity_id"], base["site_key"],
             base["upload_id"], base["model_id"], p.get("status"),
             base["occurred_at"], received, as_json(p.get("metrics")), as_json(p),
             cfg_hash, as_json(frozen) if frozen else None,
             as_json(effective) if effective else None),
        )
    elif etype == "calibration.completed":
        execute(
            """INSERT INTO calibration_results
               (event_id, job_id, entity_id, site_key, upload_id, model_id,
                calibrator_id, status, scope, method, occurred_at, received_at,
                metrics_json, metadata_json, config_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base["event_id"], base["job_id"], base["entity_id"], base["site_key"],
             base["upload_id"], base["model_id"], p.get("calibrator_id"),
             p.get("status"), p.get("scope"), p.get("method"), base["occurred_at"],
             received, as_json(p.get("metrics")), as_json(p), cfg_hash),
        )
    elif etype == "backtest.completed":
        execute(
            """INSERT INTO backtest_results
               (event_id, job_id, entity_id, site_key, upload_id, model_id, report_id,
                backtest_type, status, reason, occurred_at, received_at, metrics_json,
                assumptions_json, caveat, artifacts_json,
                config_hash, frozen_config_json, effective_config_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base["event_id"], base["job_id"], base["entity_id"], base["site_key"],
             base["upload_id"], base["model_id"], p.get("report_id"),
             p.get("backtest_type"), p.get("status"), p.get("reason"),
             base["occurred_at"], received, as_json(p.get("metrics")),
             as_json(p.get("assumptions")), p.get("caveat"),
             as_json(p.get("artifacts")),
             cfg_hash, as_json(frozen) if frozen else None,
             as_json(effective) if effective else None),
        )
    elif etype == "scoring.recorded":
        action = p.get("action", {}) or {}
        fc = p.get("forecast", {}) or {}
        kdvr, chan_no = _split_entity(base["entity_id"])
        execute(
            """INSERT INTO scoring_results
               (event_id, request_id, entity_id, kdvr, chan_no, site_key,
                decision_time, current_cart, simulated_cart,
                pred_cart_20min, pred_cart_40min, pred_min_20min, pred_min_40min,
                lower_bound_min_40min, shortage_risk, alert_state, quantity,
                model_id, calibration_id, scored_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (base["event_id"], p.get("request_id"), base["entity_id"],
             kdvr, chan_no, base["site_key"],
             p.get("decision_time"), p.get("current_cart"), p.get("simulated_cart"),
             fc.get("cart_at_20min"), fc.get("cart_at_40min"),
             fc.get("min_within_20min"), fc.get("min_within_40min"),
             fc.get("lower_bound_min_within_40min"),
             p.get("shortage_risk"), p.get("alert_state"),
             action.get("quantity"),
             base["model_id"], p.get("calibration_id"),
             p.get("scored_at") or received),
        )


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
def _filtered(table: str, filters: dict, limit: int, offset: int,
              order_col: str = "occurred_at") -> dict:
    where, params = ["1=1"], []
    for col, val in filters.items():
        if val is None:
            continue
        if col == "since":
            where.append(f"{order_col} >= ?")
        elif col == "until":
            where.append(f"{order_col} <= ?")
        else:
            where.append(f"{col} = ?")
        params.append(val)
    clause = " AND ".join(where)
    total = query_one(f"SELECT COUNT(*) AS n FROM {table} WHERE {clause}",
                      tuple(params))["n"]
    rows = query(
        f"SELECT * FROM {table} WHERE {clause} ORDER BY {order_col} DESC, id DESC "
        f"LIMIT ? OFFSET ?",
        tuple(params) + (limit, offset),
    )
    return {"total": int(total), "limit": limit, "offset": offset, "items": rows}


def list_training(**kw) -> dict:
    limit = kw.pop("limit", 50)
    offset = kw.pop("offset", 0)
    return _filtered("training_results", kw, limit, offset)


def list_calibration(**kw) -> dict:
    limit = kw.pop("limit", 50)
    offset = kw.pop("offset", 0)
    return _filtered("calibration_results", kw, limit, offset)


def list_backtests(**kw) -> dict:
    limit = kw.pop("limit", 50)
    offset = kw.pop("offset", 0)
    return _filtered("backtest_results", kw, limit, offset)


def list_scoring(**kw) -> dict:
    limit = kw.pop("limit", 50)
    offset = kw.pop("offset", 0)
    return _filtered("scoring_results", kw, limit, offset, order_col="decision_time")


def get_event(event_id: str) -> dict | None:
    row = query_one("SELECT * FROM received_events WHERE event_id=?", (event_id,))
    if row is None:
        return None
    row["payload"] = from_json(row.pop("payload_json"), {})
    return row


def entity_summary(entity_id: str) -> dict:
    def _count(table):
        return int(query_one(f"SELECT COUNT(*) AS n FROM {table} WHERE entity_id=?",
                             (entity_id,))["n"])

    backtests = query(
        "SELECT backtest_type, status, report_id, job_id, model_id, occurred_at "
        "FROM backtest_results WHERE entity_id=? ORDER BY occurred_at DESC",
        (entity_id,),
    )
    latest_training = query_one(
        "SELECT * FROM training_results WHERE entity_id=? ORDER BY occurred_at DESC",
        (entity_id,),
    )
    latest_calibration = query_one(
        "SELECT * FROM calibration_results WHERE entity_id=? ORDER BY occurred_at DESC",
        (entity_id,),
    )
    latest_score = query_one(
        "SELECT * FROM scoring_results WHERE entity_id=? ORDER BY decision_time DESC",
        (entity_id,),
    )
    return {
        "entity_id": entity_id,
        "counts": {
            "training": _count("training_results"),
            "calibration": _count("calibration_results"),
            "backtest": _count("backtest_results"),
            "scoring": _count("scoring_results"),
        },
        "backtests": backtests,
        "latest_training": latest_training,
        "latest_calibration": latest_calibration,
        "latest_scoring": latest_score,
    }


def health() -> dict:
    row = query_one("SELECT COUNT(*) AS n FROM received_events")
    return {"events_stored": int(row["n"]) if row else 0}
