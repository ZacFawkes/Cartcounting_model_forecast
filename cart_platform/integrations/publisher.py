"""HTTP publisher and background retry worker for the result outbox."""
from __future__ import annotations

import json
import logging
import threading
import time

import requests

from settings import (
    PUBLISH_TIMEOUT_SEC,
    PUBLISH_WORKER_INTERVAL_SEC,
    RECEIVER_TOKEN,
    RESULT_RECEIVER_ENABLED,
    RESULT_RECEIVER_URL,
)

from integrations import outbox  # noqa: E402
from integrations.result_events import ROUTES  # noqa: E402

LOGGER = logging.getLogger("platform.publisher")

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class DeliveryOutcome:
    def __init__(self, ok: bool, status_code: int | None = None,
                 error: str | None = None, retryable: bool = False,
                 retry_after: float | None = None):
        self.ok = ok
        self.status_code = status_code
        self.error = error
        self.retryable = retryable
        self.retry_after = retry_after


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if RECEIVER_TOKEN:
        h["Authorization"] = f"Bearer {RECEIVER_TOKEN}"
    return h


def deliver(row: dict, session: requests.Session | None = None) -> DeliveryOutcome:
    """POST one outbox row to the receiver. Never raises."""
    payload = json.loads(row["payload_json"])
    route = ROUTES.get(row["event_type"])
    if route is None:
        return DeliveryOutcome(False, None, f"unknown event_type '{row['event_type']}'",
                               retryable=False)
    url = f"{row['destination'].rstrip('/')}{route}"
    sess = session or requests
    try:
        r = sess.post(url, json=payload, headers=_headers(), timeout=PUBLISH_TIMEOUT_SEC)
    except requests.RequestException as exc:
        return DeliveryOutcome(False, None, f"network error: {exc}", retryable=True)

    if r.status_code in (200, 201):
        return DeliveryOutcome(True, r.status_code)
    if r.status_code == 409:
        # Same event_id with a different payload: a real contract violation.
        return DeliveryOutcome(False, 409,
                               f"receiver reported a conflicting payload: {r.text[:300]}",
                               retryable=False)
    if r.status_code in RETRYABLE_STATUS:
        retry_after = None
        if "Retry-After" in r.headers:
            try:
                retry_after = float(r.headers["Retry-After"])
            except ValueError:
                retry_after = None
        return DeliveryOutcome(False, r.status_code,
                               f"receiver returned {r.status_code}: {r.text[:300]}",
                               retryable=True, retry_after=retry_after)
    return DeliveryOutcome(False, r.status_code,
                           f"receiver returned {r.status_code}: {r.text[:300]}",
                           retryable=False)


def flush_once(limit: int = 20, session: requests.Session | None = None) -> dict:
    """Deliver one batch. Returns counts; used by the worker and by tests."""
    if not RESULT_RECEIVER_ENABLED:
        return {"skipped": True, "sent": 0, "retried": 0, "dead": 0}
    sess = session or requests.Session()
    sent = retried = dead = 0
    for row in outbox.claim_batch(limit):
        outcome = deliver(row, sess)
        if outcome.ok:
            outbox.mark_sent(row["id"], outcome.status_code)
            sent += 1
        elif outcome.retryable:
            outbox.mark_retry(row["id"], int(row["attempts"] or 0), outcome.error,
                              outcome.status_code, outcome.retry_after)
            retried += 1
        else:
            outbox.mark_dead(row["id"], int(row["attempts"] or 0), outcome.error,
                             outcome.status_code or 0)
            dead += 1
    if sent or retried or dead:
        LOGGER.info("Outbox flush: %d sent, %d retried, %d dead-lettered",
                    sent, retried, dead)
    return {"skipped": False, "sent": sent, "retried": retried, "dead": dead}


class RetryWorker:
    """Background delivery loop, independent of API response latency."""

    def __init__(self, interval: float = PUBLISH_WORKER_INTERVAL_SEC):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not RESULT_RECEIVER_ENABLED:
            LOGGER.info("Result publishing disabled; retry worker not started")
            return
        if self._thread and self._thread.is_alive():
            return
        outbox.reset_sending()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="outbox-retry")
        self._thread.start()
        LOGGER.info("Outbox retry worker started (target %s, every %.0fs)",
                    RESULT_RECEIVER_URL, self.interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        session = requests.Session()
        while not self._stop.is_set():
            try:
                outbox.recover_stuck()
                flush_once(session=session)
            except Exception:  # pragma: no cover - the worker must never die
                LOGGER.exception("Outbox worker iteration failed")
            self._stop.wait(self.interval)


WORKER = RetryWorker()
