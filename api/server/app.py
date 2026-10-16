"""FastAPI application — the service hosted on the server.

Endpoints
---------
GET  /health              liveness + model status
GET  /model/info          which models, policy and assumptions are served
POST /score               one live replenishment decision
POST /backtest            run a backtest over a telemetry window
GET  /sessions            online state (pending orders, alert state)
POST /sessions/reset      clear online state
GET  /logs/requests       recent call log
GET  /logs/scores         recent scoring decisions
GET  /logs/backtests      recent backtest runs
GET  /runs/{run_id}       artefacts of a stored backtest run
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from server import request_log as rlog  # noqa: E402
from server.engine import Engine, InsufficientHistory, ModelNotLoaded  # noqa: E402
from server.schemas import (  # noqa: E402
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    SessionResetRequest,
)
from server.settings import (  # noqa: E402
    API_TITLE,
    API_VERSION,
    BACKTEST_LOG,
    DEFAULT_INVENTORY,
    DEFAULT_POLICY,
    REQUEST_LOG,
    RUN_DIR,
    SCORE_LOG,
)

rlog.configure_logging()
LOGGER = rlog.LOGGER

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Serves the validated cart-availability forecasting models and the audited "
        "replenishment decision engine. Forecast accuracy is measured; shortage-reduction "
        "figures are simulation estimates, not observed causal effects."
    ),
)
engine = Engine()


@app.on_event("startup")
def _startup() -> None:
    try:
        engine.load()
        LOGGER.info("Model bundle loaded: %s", engine.bundle.version)
    except ModelNotLoaded as exc:
        LOGGER.error("Startup without a model: %s", exc)


@app.middleware("http")
async def _log_calls(request: Request, call_next):
    rid = rlog.new_request_id()
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover
        dt = (time.perf_counter() - t0) * 1000
        rlog.log_request(rid, request.url.path, request.method, "error", dt,
                         client=request.client.host if request.client else None,
                         error=str(exc))
        LOGGER.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "internal error", "request_id": rid})
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{dt:.2f}"
    if request.url.path not in ("/score", "/backtest"):
        rlog.log_request(rid, request.url.path, request.method,
                         "ok" if response.status_code < 400 else "error", dt,
                         client=request.client.host if request.client else None)
    return response


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if engine.ready else "degraded",
        version=API_VERSION,
        model_loaded=engine.ready,
        model_version=engine.bundle.version if engine.ready else "none",
        uptime_seconds=round(time.time() - engine.started_at, 1),
    )


@app.get("/model/info")
def model_info() -> dict:
    if not engine.ready:
        raise HTTPException(status_code=503, detail="model bundle not loaded")
    b = engine.bundle
    return {
        "model_version": b.version,
        "selected_models": b.selected,
        "n_features": len(b.feature_cols),
        "targets": {
            "y_cart_3": "cart count at +3 records (~20 min)",
            "y_cart_6": "cart count at +6 records (~40 min)",
            "y_min_3": "minimum cart count within 3 records",
            "y_min_6": "minimum cart count within 6 records",
        },
        "uncertainty": {
            "method": "split conformal on validation residuals",
            "alpha": b.conformal.alpha,
            "n_calibration_residuals": int(len(b.conformal.residuals_)),
            "lower_offset_carts": round(b.conformal.lower_offset, 3),
        },
        "default_policy": DEFAULT_POLICY,
        "default_inventory": DEFAULT_INVENTORY,
        "policy_provenance": (
            "Audited re-optimised policy frozen on validation data. The original "
            "pre-audit policy (risk 0.40, cooldown 10 min) is NOT served."
        ),
        "caveat": (
            "Forecasting and alerting are validated on held-out telemetry. Replenishment "
            "benefit figures are simulator outputs under documented assumptions."
        ),
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    rid = getattr(request.state, "request_id", rlog.new_request_id())
    t0 = time.perf_counter()
    payload = {
        "site_key": req.site_key,
        "history": [r.model_dump() for r in req.history],
        "policy": req.policy.merged(),
        "inventory": req.inventory.merged(),
        "session_id": req.session_id,
    }
    try:
        result = engine.score(payload)
    except InsufficientHistory as exc:
        dt = (time.perf_counter() - t0) * 1000
        rlog.log_request(rid, "/score", "POST", "rejected", dt, req.client,
                         {"records": len(req.history), "site_key": req.site_key},
                         error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc))
    except ModelNotLoaded as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    dt = (time.perf_counter() - t0) * 1000
    result["request_id"] = rid
    rlog.log_score_result(rid, result, dt)
    rlog.log_request(
        rid, "/score", "POST", "ok", dt, req.client,
        {"site_key": req.site_key, "records": len(req.history),
         "session_id": result["session_id"]},
        {"risk": result["shortage_risk"], "alert_state": result["alert_state"],
         "should_replenish": result["action"]["should_replenish"],
         "quantity": result["action"]["quantity"]},
    )
    if result["action"]["should_replenish"]:
        LOGGER.info(
            "[%s] REPLENISH %s carts at %s (risk %.2f, urgency %s)",
            rid, result["action"]["quantity"], result["decision_time"],
            result["shortage_risk"], result["action"]["urgency"],
        )
    return ScoreResponse(**result)


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------
@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest, request: Request) -> BacktestResponse:
    rid = getattr(request.state, "request_id", rlog.new_request_id())
    t0 = time.perf_counter()
    payload = {
        "records": [r.model_dump() for r in req.records] if req.records else None,
        "use_bundled_data": req.use_bundled_data,
        "start": req.start,
        "end": req.end,
        "mode": req.mode,
        "policy": req.policy.merged(),
        "inventory": req.inventory.merged(),
        "include_decisions": req.include_decisions,
        "include_events": req.include_events,
        "include_series": req.include_series,
        "label": req.label,
    }
    try:
        result, frames = engine.backtest(payload)
    except InsufficientHistory as exc:
        dt = (time.perf_counter() - t0) * 1000
        rlog.log_request(rid, "/backtest", "POST", "rejected", dt, req.client,
                         {"mode": req.mode, "label": req.label}, error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc))
    except ModelNotLoaded as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    dt = (time.perf_counter() - t0) * 1000
    result["request_id"] = rid
    artifact_dir = rlog.save_run_artifacts(result["run_id"], frames, result)
    result["artifact_dir"] = str(artifact_dir)
    rlog.log_backtest_result(rid, result["run_id"], result, dt)
    m = result["metrics"]
    rlog.log_request(
        rid, "/backtest", "POST", "ok", dt, req.client,
        {"mode": req.mode, "label": req.label, "records": result["n_records"]},
        {"run_id": result["run_id"],
         "baseline_events": m.get("baseline_events"),
         "policy_events": m.get("policy_events"),
         "severity_reduction_pct": m.get("severity_reduction_pct"),
         "actions": m.get("n_replenishment_actions")},
    )
    LOGGER.info(
        "[%s] backtest %s: %s events -> %s, severity -%.1f%%, %s trips (%.0f ms)",
        rid, result["run_id"], m.get("baseline_events"), m.get("policy_events"),
        m.get("severity_reduction_pct") or 0.0, m.get("n_replenishment_actions"), dt,
    )
    return BacktestResponse(**result)


# --------------------------------------------------------------------------
# Online state
# --------------------------------------------------------------------------
@app.get("/sessions")
def sessions() -> dict:
    return {"sessions": engine.sessions.snapshot()}


@app.post("/sessions/reset")
def reset_sessions(req: SessionResetRequest) -> dict:
    n = engine.sessions.reset(req.session_id)
    LOGGER.info("Reset %d session(s)", n)
    return {"reset": n, "session_id": req.session_id}


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------
def _tail_jsonl(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _tail_csv(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path).tail(limit)
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.get("/logs/requests")
def logs_requests(limit: int = 50) -> dict:
    return {"count": limit, "entries": _tail_jsonl(REQUEST_LOG, limit)}


@app.get("/logs/scores")
def logs_scores(limit: int = 50) -> dict:
    return {"entries": _tail_csv(SCORE_LOG, limit)}


@app.get("/logs/backtests")
def logs_backtests(limit: int = 50) -> dict:
    return {"entries": _tail_csv(BACKTEST_LOG, limit)}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    d = RUN_DIR / run_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    response_file = d / "response.json"
    body = json.loads(response_file.read_text()) if response_file.exists() else {}
    return {
        "run_id": run_id,
        "artifact_dir": str(d),
        "files": sorted(p.name for p in d.iterdir()),
        "response": body,
    }


@app.get("/runs")
def list_runs(limit: int = 50) -> dict:
    runs = sorted((p.name for p in RUN_DIR.iterdir() if p.is_dir()), reverse=True)[:limit]
    return {"runs": runs, "count": len(runs)}
