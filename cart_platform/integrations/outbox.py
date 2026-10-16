"""Transactional outbox for result publishing.

Events are written to ``result_outbox`` in ``platform.db`` next to the local
result they describe. A background worker delivers them over HTTP. Receiver
downtime therefore cannot fail training or scoring, and nothing is lost when the
process restarts.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone

from settings import (
    PUBLISH_BASE_BACKOFF_SEC,
    PUBLISH_MAX_ATTEMPTS,
    PUBLISH_MAX_BACKOFF_SEC,
    RESULT_RECEIVER_URL,
)

from integrations.result_events import ResultEvent, canonical_json  # noqa: E402
from registry.database import execute, query, query_one, utc_now  # noqa: E402

LOGGER = logging.getLogger("platform.outbox")

PENDING, SENDING, SENT, DEAD = "pending", "sending", "sent", "dead_letter"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue(event: ResultEvent, destination: str | None = None) -> bool:
    """Insert an event. Returns False when the event_id already exists."""
    dest = destination or RESULT_RECEIVER_URL
    cur = execute(
        """INSERT OR IGNORE INTO result_outbox
           (event_id, event_type, destination, entity_id, job_id, model_id,
            payload_json, status, attempts, next_attempt_at, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (event.event_id, event.event_type, dest, event.entity_id, event.job_id,
         event.model_id, canonical_json(event.to_dict()), PENDING, 0,
         _now().isoformat(), utc_now()),
    )
    created = cur.rowcount > 0
    if not created:
        LOGGER.debug("Outbox replay suppressed for %s", event.event_id)
    return created


def claim_batch(limit: int = 20) -> list[dict]:
    """Atomically move due pending rows to ``sending`` and return them."""
    now = _now().isoformat()
    rows = query(
        "SELECT * FROM result_outbox WHERE status=? AND next_attempt_at<=? "
        "ORDER BY id LIMIT ?",
        (PENDING, now, limit),
    )
    claimed = []
    for r in rows:
        cur = execute(
            "UPDATE result_outbox SET status=? WHERE id=? AND status=?",
            (SENDING, r["id"], PENDING),
        )
        if cur.rowcount:  # lost races are simply skipped
            claimed.append(r)
    return claimed


def mark_sent(row_id: int, response_code: int) -> None:
    execute(
        "UPDATE result_outbox SET status=?, sent_at=?, response_code=?, last_error=NULL "
        "WHERE id=?",
        (SENT, utc_now(), response_code, row_id),
    )


def mark_retry(row_id: int, attempts: int, error: str,
               response_code: int | None = None,
               retry_after: float | None = None) -> None:
    attempts += 1
    if attempts >= PUBLISH_MAX_ATTEMPTS:
        execute(
            "UPDATE result_outbox SET status=?, attempts=?, last_error=?, "
            "response_code=? WHERE id=?",
            (DEAD, attempts, f"max attempts reached: {error}", response_code, row_id),
        )
        LOGGER.error("Outbox row %s dead-lettered after %d attempts", row_id, attempts)
        return
    if retry_after is not None:
        delay = float(retry_after)
    else:
        delay = min(PUBLISH_BASE_BACKOFF_SEC * (2 ** (attempts - 1)),
                    PUBLISH_MAX_BACKOFF_SEC)
        delay *= 0.5 + random.random()  # jitter
    nxt = (_now() + timedelta(seconds=delay)).isoformat()
    execute(
        "UPDATE result_outbox SET status=?, attempts=?, next_attempt_at=?, "
        "last_error=?, response_code=? WHERE id=?",
        (PENDING, attempts, nxt, error, response_code, row_id),
    )


def mark_dead(row_id: int, attempts: int, error: str, response_code: int) -> None:
    execute(
        "UPDATE result_outbox SET status=?, attempts=?, last_error=?, response_code=? "
        "WHERE id=?",
        (DEAD, attempts + 1, error, response_code, row_id),
    )


def recover_stuck(max_age_seconds: float = 120.0) -> int:
    """Return rows abandoned mid-flight by a crashed worker to ``pending``."""
    cutoff = (_now() - timedelta(seconds=max_age_seconds)).isoformat()
    cur = execute(
        "UPDATE result_outbox SET status=? WHERE status=? AND "
        "(sent_at IS NULL AND created_at<=?)",
        (PENDING, SENDING, cutoff),
    )
    if cur.rowcount:
        LOGGER.warning("Recovered %d stuck outbox row(s)", cur.rowcount)
    return cur.rowcount


def reset_sending() -> int:
    """Called at startup: nothing can legitimately be in flight yet."""
    cur = execute("UPDATE result_outbox SET status=? WHERE status=?", (PENDING, SENDING))
    if cur.rowcount:
        LOGGER.info("Reset %d in-flight outbox row(s) after restart", cur.rowcount)
    return cur.rowcount


def retry_dead_letters(event_id: str | None = None) -> int:
    sql = "UPDATE result_outbox SET status=?, attempts=0, next_attempt_at=?, " \
          "last_error=NULL WHERE status=?"
    params: list = [PENDING, _now().isoformat(), DEAD]
    if event_id:
        sql += " AND event_id=?"
        params.append(event_id)
    cur = execute(sql, tuple(params))
    return cur.rowcount


def stats() -> dict:
    rows = query("SELECT status, COUNT(*) AS n FROM result_outbox GROUP BY status")
    out = {PENDING: 0, SENDING: 0, SENT: 0, DEAD: 0}
    for r in rows:
        out[r["status"]] = int(r["n"])
    out["total"] = sum(v for k, v in out.items() if k != "total")
    return out


def list_events(status: str | None = None, entity_id: str | None = None,
                event_type: str | None = None, limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM result_outbox WHERE 1=1"
    params: list = []
    for col, val in (("status", status), ("entity_id", entity_id),
                     ("event_type", event_type)):
        if val:
            sql += f" AND {col}=?"
            params.append(val)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return query(sql, tuple(params))


def get_event(event_id: str) -> dict | None:
    return query_one("SELECT * FROM result_outbox WHERE event_id=?", (event_id,))
