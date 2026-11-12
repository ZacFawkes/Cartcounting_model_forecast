"""Durable operational state: sessions, order book and score write-back.

The previous API kept pending orders in memory, so a restart could place a
duplicate retrieval. Here the order book is the source of truth in SQLite and
session state is rebuilt from it on startup.
"""
from __future__ import annotations

import logging
import uuid

import pandas as pd

from alert_engine import AlertEngine, AlertEngineConfig
from audit.order_book import CANCELLED, COMPLETED, PENDING, OrderBook
from registry.database import (
    as_json,
    execute,
    insert_ignore,
    query,
    query_one,
    upsert,
    utc_now,
)

LOGGER = logging.getLogger("platform.store")


# --------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------
def new_order_id(entity_id: str) -> str:
    return f"ORD-{entity_id.replace(':', '_')}-{uuid.uuid4().hex[:8]}"


def persist_order(order, entity: dict, source: str = "score") -> None:
    execute(
        """INSERT OR REPLACE INTO replenishment_orders
           (order_id, entity_id, kdvr, chan_no, site_key, bay_id, decision_time,
            quantity, expected_completion_time, status, delivered_quantity,
            delivered_at, urgency, risk_at_decision, predicted_min_at_decision,
            source, executed_by, executed_at, actual_carts, execution_notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                   (SELECT executed_by FROM replenishment_orders WHERE order_id=?),
                   (SELECT executed_at FROM replenishment_orders WHERE order_id=?),
                   (SELECT actual_carts FROM replenishment_orders WHERE order_id=?),
                   (SELECT execution_notes FROM replenishment_orders WHERE order_id=?))""",
        (
            order.order_id, entity["entity_id"], entity["kdvr"], entity["chan_no"],
            entity["site_key"], entity.get("bay_id"), str(order.decision_time),
            int(order.quantity),
            str(order.expected_completion_time), order.status,
            float(order.delivered_quantity),
            str(order.delivered_time) if order.delivered_time is not None else None,
            order.urgency, order.risk_at_decision, order.predicted_min_at_decision,
            source,
            order.order_id, order.order_id, order.order_id, order.order_id,
        ),
    )


def load_order_book(entity_id: str) -> OrderBook:
    """Rebuild the in-memory order book from the durable ledger."""
    book = OrderBook()
    rows = query(
        "SELECT * FROM replenishment_orders WHERE entity_id=? AND status=? "
        "ORDER BY decision_time",
        (entity_id, PENDING),
    )
    from audit.order_book import ReplenishmentOrder

    for r in rows:
        book.orders.append(
            ReplenishmentOrder(
                order_id=r["order_id"],
                decision_time=pd.Timestamp(r["decision_time"]),
                quantity=int(r["quantity"]),
                expected_completion_time=pd.Timestamp(r["expected_completion_time"]),
                status=r["status"],
                delivered_quantity=float(r["delivered_quantity"] or 0),
                urgency=r["urgency"],
                risk_at_decision=r["risk_at_decision"],
                predicted_min_at_decision=r["predicted_min_at_decision"],
            )
        )
    return book


def record_execution(order_id: str, executed_by: str, actual_carts: int,
                     notes: str | None = None) -> dict:
    """Operations logs what was actually done — the audit's blocking gap."""
    execute(
        "UPDATE replenishment_orders SET executed_by=?, executed_at=?, actual_carts=?, "
        "execution_notes=? WHERE order_id=?",
        (executed_by, utc_now(), int(actual_carts), notes, order_id),
    )
    return query_one("SELECT * FROM replenishment_orders WHERE order_id=?", (order_id,))


def list_orders(entity_id: str | None = None, status: str | None = None,
                limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM replenishment_orders WHERE 1=1"
    params: list = []
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY decision_time DESC LIMIT ?"
    params.append(limit)
    return query(sql, tuple(params))


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def load_session(entity_id: str, alert_cfg: AlertEngineConfig) -> dict:
    row = query_one("SELECT * FROM session_state WHERE entity_id=?", (entity_id,))
    engine = AlertEngine(alert_cfg)
    if row:
        engine.state = row["alert_state"] or "NORMAL"
        engine._streak = int(row["alert_streak"] or 0)
        engine.last_alert_time = (
            pd.Timestamp(row["last_alert_time"]) if row["last_alert_time"] else None
        )
    return {
        "entity_id": entity_id,
        "surplus": float(row["surplus"]) if row else 0.0,
        "last_record_time": pd.Timestamp(row["last_record_time"])
        if row and row["last_record_time"] else None,
        "last_decision_time": pd.Timestamp(row["last_decision_time"])
        if row and row["last_decision_time"] else None,
        "n_calls": int(row["n_calls"]) if row else 0,
        "reserve_used_today": float(row["reserve_used_today"]) if row else 0.0,
        "reserve_day": row["reserve_day"] if row else None,
        "alert": engine,
        "book": load_order_book(entity_id),
    }


def save_session(session: dict) -> None:
    engine: AlertEngine = session["alert"]
    upsert(
        "session_state",
        {
            "entity_id": session["entity_id"],
            "surplus": float(session["surplus"]),
            "last_record_time": str(session["last_record_time"])
            if session["last_record_time"] is not None else None,
            "last_decision_time": str(session["last_decision_time"])
            if session["last_decision_time"] is not None else None,
            "alert_state": engine.state,
            "alert_streak": int(engine._streak),
            "last_alert_time": str(engine.last_alert_time)
            if engine.last_alert_time is not None else None,
            "n_calls": int(session["n_calls"]),
            "reserve_used_today": float(session["reserve_used_today"]),
            "reserve_day": session["reserve_day"],
            "updated_at": utc_now(),
        },
        ["entity_id"],
    )


def reset_session(entity_id: str | None = None) -> int:
    if entity_id:
        execute("DELETE FROM session_state WHERE entity_id=?", (entity_id,))
        cur = execute(
            "UPDATE replenishment_orders SET status=?, execution_notes=COALESCE("
            "execution_notes,'') || ' [session reset]' WHERE entity_id=? AND status=?",
            (CANCELLED, entity_id, PENDING),
        )
        return cur.rowcount
    execute("DELETE FROM session_state")
    cur = execute("UPDATE replenishment_orders SET status=? WHERE status=?",
                  (CANCELLED, PENDING))
    return cur.rowcount


def session_snapshot(entity_id: str | None = None) -> list[dict]:
    sql = """
        SELECT s.*, (SELECT COUNT(*) FROM replenishment_orders o
                     WHERE o.entity_id = s.entity_id AND o.status='PENDING')
                    AS pending_orders,
               (SELECT COALESCE(SUM(quantity),0) FROM replenishment_orders o
                WHERE o.entity_id = s.entity_id AND o.status='PENDING')
                    AS pending_carts
        FROM session_state s
    """
    params: tuple = ()
    if entity_id:
        sql += " WHERE s.entity_id = ?"
        params = (entity_id,)
    return query(sql + " ORDER BY s.entity_id", params)


# --------------------------------------------------------------------------
# Score write-back
# --------------------------------------------------------------------------
def write_score(result: dict, entity: dict, latency_ms: float = 0.0) -> bool:
    """Idempotent write; UNIQUE(entity_id, decision_time) is the DB-level guard."""
    f = result["forecast"]
    a = result["action"]
    row = {
        "request_id": result.get("request_id"),
        "entity_id": entity["entity_id"],
        "kdvr": entity["kdvr"],
        "chan_no": entity["chan_no"],
        "site_key": entity["site_key"],
        "bay_id": entity.get("bay_id"),
        "decision_time": result["decision_time"],
        "current_cart": result["current_cart"],
        "simulated_cart": result["simulated_cart"],
        "pred_cart_20min": f["cart_at_20min"],
        "pred_cart_40min": f["cart_at_40min"],
        "pred_min_20min": f["min_within_20min"],
        "pred_min_40min": f["min_within_40min"],
        "lower_bound_min_40min": f["lower_bound_min_within_40min"],
        "shortage_risk": result["shortage_risk"],
        "alert_state": result["alert_state"],
        "quantity": a["quantity"],
        "model_id": result["model_id"],
        "calibration_id": result["calibration_id"],
        "scored_at": utc_now(),
    }
    return insert_ignore("scoring_results", row)


def recent_scores(entity_id: str | None = None, site_key: int | None = None,
                  limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM scoring_results WHERE 1=1"
    params: list = []
    if entity_id:
        sql += " AND entity_id=?"
        params.append(entity_id)
    if site_key is not None:
        sql += " AND site_key=?"
        params.append(site_key)
    sql += " ORDER BY decision_time DESC LIMIT ?"
    params.append(limit)
    return query(sql, tuple(params))


def stored_score(entity_id: str, decision_time: str) -> dict | None:
    return query_one(
        "SELECT * FROM scoring_results WHERE entity_id=? AND decision_time=?",
        (entity_id, decision_time),
    )
