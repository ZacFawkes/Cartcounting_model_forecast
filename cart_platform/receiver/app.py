"""Standalone result receiver API (port 8200).

Represents a future external backend. It owns ``results_receiver.db`` and is
written to **only over HTTP** by the ML server. Keeping it separate means the
receiver URL can later be swapped for a real backend without touching any
training or scoring logic.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Literal

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C  # noqa: E402
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from receiver import database as db  # noqa: E402
from receiver import store  # noqa: E402

LOGGER = logging.getLogger("receiver")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="Cart Result Receiver",
    version="1.0.0",
    description=(
        "Standalone SQLite-backed result store. Receives training, calibration, "
        "backtest and scoring events from the ML platform over HTTP."
    ),
)
_STARTED = time.time()

EVENT_TYPES = {
    "/v1/results/training": "training.completed",
    "/v1/results/calibration": "calibration.completed",
    "/v1/results/backtest": "backtest.completed",
    "/v1/results/scoring": "scoring.recorded",
}


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------
class ResultEventIn(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=512)
    event_type: str
    payload: dict
    schema_version: int = 1
    occurred_at: str | None = None
    entity_id: str | None = None
    site_key: str | None = None
    upload_id: str | None = None
    job_id: str | None = None
    model_id: str | None = None


class StoreAck(BaseModel):
    event_id: str
    outcome: Literal["created", "replay"]
    stored_record: dict


class PageOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def require_token(authorization: str | None = Header(default=None)) -> None:
    """Bearer auth for writes when a token is configured."""
    if not C.RECEIVER_TOKEN:
        return
    expected = f"Bearer {C.RECEIVER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@app.on_event("startup")
def _startup() -> None:
    db.configure(C.RECEIVER_DB_PATH)
    db.init_db()
    LOGGER.info("Receiver listening; database %s", C.RECEIVER_DB_PATH)


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    try:
        info = store.health()
        ok = True
        error = None
    except Exception as exc:  # pragma: no cover
        info, ok, error = {}, False, str(exc)
    return {
        "status": "ok" if ok else "degraded",
        "service": "cart-result-receiver",
        "version": "1.0.0",
        "database": str(db.db_path()),
        "sqlite_ok": ok,
        "error": error,
        "uptime_seconds": round(time.time() - _STARTED, 1),
        **info,
    }


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def _ingest(event: ResultEventIn, expected_type: str, response: Response) -> StoreAck:
    if event.event_type != expected_type:
        raise HTTPException(
            status_code=422,
            detail=f"event_type '{event.event_type}' does not match this route "
                   f"(expected '{expected_type}')",
        )
    outcome, record = store.store_event(event.model_dump())
    if outcome == store.CONFLICT:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "event_id already stored with a different payload; "
                           "historical results are never overwritten",
                "event_id": event.event_id,
                "stored_payload_hash": record["payload_hash"],
            },
        )
    response.status_code = 201 if outcome == store.CREATED else 200
    return StoreAck(event_id=event.event_id, outcome=outcome, stored_record={
        k: record[k] for k in ("event_id", "event_type", "entity_id", "job_id",
                               "model_id", "received_at") if k in record
    })


@app.post("/v1/results/training", response_model=StoreAck,
          dependencies=[Depends(require_token)])
def post_training(event: ResultEventIn, response: Response) -> StoreAck:
    return _ingest(event, "training.completed", response)


@app.post("/v1/results/calibration", response_model=StoreAck,
          dependencies=[Depends(require_token)])
def post_calibration(event: ResultEventIn, response: Response) -> StoreAck:
    return _ingest(event, "calibration.completed", response)


@app.post("/v1/results/backtest", response_model=StoreAck,
          dependencies=[Depends(require_token)])
def post_backtest(event: ResultEventIn, response: Response) -> StoreAck:
    bt = (event.payload or {}).get("backtest_type")
    if not bt:
        raise HTTPException(status_code=422,
                            detail="payload.backtest_type is required")
    return _ingest(event, "backtest.completed", response)


@app.post("/v1/results/scoring", response_model=StoreAck,
          dependencies=[Depends(require_token)])
def post_scoring(event: ResultEventIn, response: Response) -> StoreAck:
    return _ingest(event, "scoring.recorded", response)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
@app.get("/v1/results/training", response_model=PageOut)
def get_training(entity_id: str | None = None, site_key: str | None = None,
                 upload_id: str | None = None, job_id: str | None = None,
                 model_id: str | None = None, status: str | None = None,
                 since: str | None = None, until: str | None = None,
                 limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return store.list_training(entity_id=entity_id, site_key=site_key,
                               upload_id=upload_id, job_id=job_id, model_id=model_id,
                               status=status, since=since, until=until,
                               limit=limit, offset=offset)


@app.get("/v1/results/calibration", response_model=PageOut)
def get_calibration(entity_id: str | None = None, site_key: str | None = None,
                    upload_id: str | None = None, job_id: str | None = None,
                    model_id: str | None = None, status: str | None = None,
                    scope: str | None = None, since: str | None = None,
                    until: str | None = None,
                    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return store.list_calibration(entity_id=entity_id, site_key=site_key,
                                  upload_id=upload_id, job_id=job_id,
                                  model_id=model_id, status=status, scope=scope,
                                  since=since, until=until, limit=limit, offset=offset)


@app.get("/v1/results/backtest", response_model=PageOut)
def get_backtest(entity_id: str | None = None, site_key: str | None = None,
                 upload_id: str | None = None, job_id: str | None = None,
                 model_id: str | None = None, backtest_type: str | None = None,
                 status: str | None = None, since: str | None = None,
                 until: str | None = None,
                 limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return store.list_backtests(entity_id=entity_id, site_key=site_key,
                                upload_id=upload_id, job_id=job_id, model_id=model_id,
                                backtest_type=backtest_type, status=status,
                                since=since, until=until, limit=limit, offset=offset)


@app.get("/v1/results/scoring", response_model=PageOut)
def get_scoring(entity_id: str | None = None, site_key: str | None = None,
                model_id: str | None = None, since: str | None = None,
                until: str | None = None,
                limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return store.list_scoring(entity_id=entity_id, site_key=site_key,
                              model_id=model_id, since=since, until=until,
                              limit=limit, offset=offset)


@app.get("/v1/results/entities/{entity_id:path}")
def get_entity_results(entity_id: str) -> dict:
    return store.entity_summary(entity_id)


@app.get("/v1/results/events/{event_id:path}")
def get_event(event_id: str) -> dict:
    row = store.get_event(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event '{event_id}' not found")
    return row
