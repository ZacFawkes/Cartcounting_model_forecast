"""Cart Forecasting Platform — FastAPI application (hosted on the server).

Routes
------
GET    /health                        liveness, model and registry status
GET    /v1/model/info                 champion model, defaults, provenance
GET    /v1/models                     model registry
GET    /v1/sites                      site registry
GET    /v1/entities                   entity registry (filter by site/status)
GET    /v1/entities/{entity_id}       one entity's resolved configuration
PUT    /v1/sites                      create / update a site
PUT    /v1/entities                   create / update an entity
POST   /v1/entities/{id}/status       activate / shadow / disable
POST   /v1/registry/bootstrap         discover entities + adopt the pipeline model
POST   /v1/score                      one live decision, written back to the DB
POST   /v1/backtest                   backtest one entity, many entities or a site
GET    /v1/sessions                   durable online state
POST   /v1/sessions/reset             clear state and cancel pending orders
GET    /v1/orders                     replenishment order ledger
POST   /v1/orders/{id}/execute        log what staff actually did
GET    /v1/logs/requests|scores|backtests
GET    /v1/runs/{run_id}              stored backtest artefacts
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from registry import entities as reg  # noqa: E402
from registry import models as mreg  # noqa: E402
from registry.database import init_db, query_one  # noqa: E402
from registry.resolver import resolve  # noqa: E402
from server import logs, store  # noqa: E402
from server.ondemand import router as ondemand_router  # noqa: E402
from server import service as svc  # noqa: E402
from server.schemas import (  # noqa: E402
    BacktestRequest,
    BacktestResponse,
    EntityUpsert,
    ExecutionLog,
    HealthResponse,
    ResetRequest,
    ScoreRequest,
    ScoreResponse,
    SiteUpsert,
    StatusUpdate,
)

logs.configure_logging()
LOGGER = logs.LOGGER

app = FastAPI(
    title=C.API_TITLE,
    version=C.API_VERSION,
    description=(
        "Multi-entity cart availability forecasting and replenishment. The modelling "
        "grain is one camera stream: entity_id = KDVR:ChanNo. Parameters resolve "
        "request override > entity/site config > system default, and every response "
        "records which level supplied each value."
    ),
)
app.include_router(ondemand_router)
_STARTED = time.time()


@app.on_event("startup")
def _startup() -> None:
    init_db()
    try:
        meta = mreg.get_model_meta()
        LOGGER.info("Champion model: %s (red_model=%s)", meta["model_id"], meta["red_model"])
    except mreg.NoModelAvailable:
        LOGGER.info("No platform-wide fallback model registered; entity models are "
                    "trained on demand via /v1/datasets/upload + /v1/training/jobs")
    # A job still marked running after a restart never finished.
    from training.jobs import recover_orphaned_jobs

    recover_orphaned_jobs()
    # Durable outbox delivery runs independently of API latency.
    from integrations.publisher import WORKER

    WORKER.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    from integrations.publisher import WORKER

    WORKER.stop()


@app.middleware("http")
async def _authenticate(request: Request, call_next):
    """Bearer auth for mutating and data routes when a token is configured."""
    if C.API_TOKEN and request.url.path not in C.AUTH_EXEMPT_PATHS:
        header = request.headers.get("authorization")
        if header != f"Bearer {C.API_TOKEN}":
            return JSONResponse(status_code=401,
                                content={"detail": "invalid or missing bearer token"})
    return await call_next(request)


@app.middleware("http")
async def _trace(request: Request, call_next):
    rid = f"req-{int(time.time() * 1000) % 10**10:010d}"
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover
        dt = (time.perf_counter() - t0) * 1000
        logs.log_request(rid, request.url.path, request.method, "error", dt,
                         client=request.client.host if request.client else None,
                         error=str(exc))
        LOGGER.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "internal error", "request_id": rid})
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{dt:.2f}"
    if not request.url.path.endswith(("/score", "/backtest")):
        logs.log_request(rid, request.url.path, request.method,
                         "ok" if response.status_code < 400 else "error", dt,
                         status_code=response.status_code,
                         client=request.client.host if request.client else None)
    return response


def _resolve_entity_id(entity_id, kdvr, chan_no, site_key) -> str:
    if entity_id:
        return entity_id
    if kdvr is None or chan_no is None:
        raise HTTPException(
            status_code=422,
            detail="supply entity_id, or both kdvr and chan_no",
        )
    return reg.entity_id_for(kdvr, chan_no, site_key)


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        meta = mreg.get_model_meta()
        model_id = meta["model_id"]
    except mreg.NoModelAvailable:
        model_id = None
    ents = reg.list_entities()
    return HealthResponse(
        status="ok" if model_id and ents else "degraded",
        version=C.API_VERSION,
        model_loaded=model_id is not None,
        champion_model_id=model_id,
        n_entities=len(ents),
        n_active=sum(1 for e in ents if e["status"] == reg.ACTIVE),
        uptime_seconds=round(time.time() - _STARTED, 1),
    )


@app.get("/v1/model/info")
def model_info() -> dict:
    try:
        meta = mreg.get_model_meta()
    except mreg.NoModelAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    model = mreg.load_model(meta["model_id"])
    return {
        "model_id": meta["model_id"],
        "version": meta["version"],
        "status": meta["status"],
        "trained_at": meta["trained_at"],
        "n_features": meta["feature_count"],
        "selected_models": model.selected,
        "feature_time_thresholds": {
            "red_model": meta["red_model"],
            "amber_model": meta["amber_model"],
            "note": "Frozen into the feature vector. A request may change the "
                    "evaluation threshold (red_threshold) but not these; the "
                    "difference is reported as a warning.",
        },
        "system_defaults": C.SYSTEM_DEFAULTS,
        "history_requirements": {
            "absolute_minimum": C.ABSOLUTE_MIN_HISTORY,
            "enforced_minimum": C.MIN_HISTORY_RECORDS,
            "recommended": C.RECOMMENDED_HISTORY,
            "note": "10 is the mathematical floor (features reach back to t-9). "
                    "Identical output from 12 upward; 20 gives margin against gaps.",
        },
        "entity_grain": "KDVR:ChanNo"
        + (" prefixed by siteKey" if C.INCLUDE_SITE_IN_ENTITY_ID else ""),
        "caveat": C.CAVEAT,
    }


@app.get("/v1/models")
def list_models() -> dict:
    return {"models": mreg.list_models()}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
@app.get("/v1/sites")
def list_sites() -> dict:
    return {"sites": reg.list_sites()}


@app.put("/v1/sites")
def put_site(req: SiteUpsert) -> dict:
    data = req.model_dump(exclude={"site_key", "updated_by"})
    return {"site": reg.upsert_site(req.site_key, req.updated_by, **data)}


@app.get("/v1/entities")
def list_entities(site_key: int | None = None, status: str | None = None) -> dict:
    return {"entities": reg.list_entities(site_key, status)}


@app.get("/v1/entities/{entity_id}")
def get_entity(entity_id: str) -> dict:
    try:
        entity = reg.get_entity(entity_id)
    except reg.UnknownEntity as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        meta = mreg.get_model_meta(entity.get("model_id"))
    except mreg.NoModelAvailable:
        meta = None
    cfg = resolve(entity_id, None, model_meta=meta)
    return {
        "entity": entity,
        "site": reg.get_site(entity["site_key"]),
        "resolved_config": cfg.as_payload(),
        "warnings": cfg.warnings,
        "unconfigured": cfg.unconfigured,
    }


@app.put("/v1/entities")
def put_entity(req: EntityUpsert) -> dict:
    data = req.model_dump(exclude={"entity_id", "updated_by"})
    entity_id = req.entity_id or reg.entity_id_for(req.kdvr, req.chan_no, req.site_key)
    try:
        return {"entity": reg.upsert_entity(entity_id, req.updated_by, **data)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/v1/entities/{entity_id}/status")
def set_status(entity_id: str, req: StatusUpdate) -> dict:
    try:
        return {"entity": reg.set_status(entity_id, req.status, req.updated_by)}
    except reg.UnknownEntity as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/v1/registry/bootstrap")
def bootstrap(adopt_model: bool = True) -> dict:
    """Discover entities from the bundled telemetry and adopt the pipeline model."""
    prepared = svc.bundled_telemetry()
    registered = reg.register_from_telemetry(prepared)
    out = {"entities_registered": len(registered),
           "entities": registered.to_dict(orient="records")}
    if adopt_model:
        try:
            mreg.get_model_meta()
            out["model"] = "champion already registered"
        except mreg.NoModelAvailable:
            model_id, cal_id = mreg.bootstrap_from_pipeline(
                entities=sorted(prepared["entity_id"].unique().tolist()),
                sites=sorted(int(s) for s in prepared["siteKey"].unique()),
            )
            for eid in prepared["entity_id"].unique():
                reg.bind_model(eid, model_id, None)
            out["model"] = {"model_id": model_id, "calibration_id": cal_id}
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    rid = getattr(request.state, "request_id", None)
    entity_id = _resolve_entity_id(req.entity_id, req.kdvr, req.chan_no, req.site_key)
    t0 = time.perf_counter()
    try:
        result = svc.score(
            entity_id=entity_id,
            records=[r.model_dump() for r in req.records],
            overrides=req.params.as_dict(),
            model_id=req.model_id,
            write_to_db=req.write_to_db,
            request_id=rid,
            latency_ms=0.0,
        )
    except reg.UnknownEntity as exc:
        _log_fail(rid, "/v1/score", t0, entity_id, req.client, str(exc))
        raise HTTPException(status_code=404, detail=str(exc))
    except svc.InsufficientHistory as exc:
        _log_fail(rid, "/v1/score", t0, entity_id, req.client, str(exc))
        raise HTTPException(status_code=422, detail=str(exc))
    except (svc.EntityNotScoreable, mreg.NoModelAvailable) as exc:
        _log_fail(rid, "/v1/score", t0, entity_id, req.client, str(exc))
        raise HTTPException(status_code=503, detail=str(exc))

    dt = (time.perf_counter() - t0) * 1000
    if req.write_to_db:
        store.write_score(result, reg.get_entity(entity_id), dt)
    logs.log_request(
        rid, "/v1/score", "POST", "ok", dt, entity_id, result["site_key"], 200, req.client,
        {"records": len(req.records), "overrides": req.params.as_dict()},
        {"risk": result["shortage_risk"], "alert_state": result["alert_state"],
         "should_replenish": result["action"]["should_replenish"],
         "quantity": result["action"]["quantity"]},
    )
    if result["action"]["should_replenish"]:
        LOGGER.info(
            "[%s] %s REPLENISH %s carts by %s (risk %.2f, %s)",
            entity_id, result["decision_time"][:16], result["action"]["quantity"],
            (result["action"]["expected_completion_time"] or "")[11:16],
            result["shortage_risk"], result["action"]["urgency"],
        )
    return ScoreResponse(**result)


def _log_fail(rid, endpoint, t0, entity_id, client, error) -> None:
    logs.log_request(rid, endpoint, "POST", "rejected",
                     (time.perf_counter() - t0) * 1000, entity_id,
                     client=client, error=error)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
@app.post("/v1/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest, request: Request) -> BacktestResponse:
    rid = getattr(request.state, "request_id", None)
    targets = req.entity_ids or ([req.entity_id] if req.entity_id else None)
    if targets is None:
        if req.site_key is None:
            raise HTTPException(
                status_code=422,
                detail="supply entity_id, entity_ids or site_key",
            )
        targets = [e["entity_id"] for e in reg.list_entities(site_key=req.site_key)]
        if not targets:
            raise HTTPException(status_code=404,
                                detail=f"no entities registered for site {req.site_key}")

    runs = []
    for entity_id in targets:
        t0 = time.perf_counter()
        try:
            payload, frames = svc.backtest(
                entity_id=entity_id,
                records=[r.model_dump() for r in req.records] if req.records else None,
                start=req.start, end=req.end, mode=req.mode,
                overrides=req.params.as_dict(), model_id=req.model_id,
                include_decisions=req.include_decisions,
                include_events=req.include_events,
                include_series=req.include_series,
                label=req.label, request_id=rid,
            )
        except reg.UnknownEntity as exc:
            _log_fail(rid, "/v1/backtest", t0, entity_id, req.client, str(exc))
            raise HTTPException(status_code=404, detail=str(exc))
        except svc.InsufficientHistory as exc:
            _log_fail(rid, "/v1/backtest", t0, entity_id, req.client, str(exc))
            raise HTTPException(status_code=422, detail=f"{entity_id}: {exc}")
        except mreg.NoModelAvailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        dt = (time.perf_counter() - t0) * 1000
        artifact_dir = logs.save_run_artifacts(payload["run_id"], frames, payload)
        payload["artifact_dir"] = str(artifact_dir)
        logs.log_backtest(payload, dt, str(artifact_dir))
        m = payload["metrics"]
        logs.log_request(
            rid, "/v1/backtest", "POST", "ok", dt, entity_id, payload["site_key"],
            200, req.client,
            {"mode": req.mode, "label": req.label},
            {"run_id": payload["run_id"], "baseline_events": m.get("baseline_events"),
             "policy_events": m.get("policy_events"),
             "severity_reduction_pct": m.get("severity_reduction_pct")},
        )
        LOGGER.info(
            "[%s] backtest %s: events %s->%s, severity -%.1f%%, %s trips (%.0f ms)",
            entity_id, payload["run_id"], m.get("baseline_events"),
            m.get("policy_events"), m.get("severity_reduction_pct") or 0.0,
            m.get("n_replenishment_actions"), dt,
        )
        runs.append(payload)

    return BacktestResponse(request_id=rid, runs=runs, n_entities=len(runs),
                            caveat=C.CAVEAT)


# --------------------------------------------------------------------------
# Operational state
# --------------------------------------------------------------------------
@app.get("/v1/sessions")
def sessions(entity_id: str | None = None) -> dict:
    return {"sessions": store.session_snapshot(entity_id)}


@app.post("/v1/sessions/reset")
def reset(req: ResetRequest) -> dict:
    n = store.reset_session(req.entity_id)
    LOGGER.info("Session reset (%s): %d pending order(s) cancelled",
                req.entity_id or "all entities", n)
    return {"entity_id": req.entity_id, "orders_cancelled": n}


@app.get("/v1/orders")
def orders(entity_id: str | None = None, status: str | None = None,
           limit: int = 100) -> dict:
    return {"orders": store.list_orders(entity_id, status, limit)}


@app.post("/v1/orders/{order_id}/execute")
def execute_order(order_id: str, req: ExecutionLog) -> dict:
    row = query_one("SELECT * FROM replenishment_orders WHERE order_id=?", (order_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"order '{order_id}' not found")
    updated = store.record_execution(order_id, req.executed_by, req.actual_carts, req.notes)
    LOGGER.info("Order %s executed by %s (%d carts)", order_id, req.executed_by,
                req.actual_carts)
    return {
        "order": updated,
        "note": "Execution logging is the gap the simulation audit identified as "
                "blocking; these records are what eventually turn a simulated "
                "benefit into a measured one.",
    }


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------
@app.get("/v1/logs/requests")
def log_requests(entity_id: str | None = None, status: str | None = None,
                 limit: int = 50) -> dict:
    return {"entries": logs.recent_requests(entity_id, status, limit)}


@app.get("/v1/logs/scores")
def log_scores(entity_id: str | None = None, site_key: int | None = None,
               limit: int = 50) -> dict:
    return {"entries": store.recent_scores(entity_id, site_key, limit)}


@app.get("/v1/logs/backtests")
def log_backtests(entity_id: str | None = None, limit: int = 50) -> dict:
    return {"entries": logs.recent_backtests(entity_id, limit)}


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict:
    d = C.RUN_DIR / run_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    row = query_one("SELECT * FROM backtest_runs WHERE run_id=?", (run_id,))
    return {
        "run_id": run_id,
        "artifact_dir": str(d),
        "files": sorted(p.name for p in d.iterdir()),
        "summary": row,
    }
