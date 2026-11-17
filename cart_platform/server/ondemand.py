"""On-demand dataset, training, promotion, entity scoring and integration routes.

Mounted on the existing ML app. Every existing route keeps working unchanged.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

import settings as C
from integrations import outbox
from integrations.publisher import WORKER, flush_once
from registry import bindings as bind
from registry import entities as reg
from registry import uploads as up
from registry.database import from_json
from server import artifacts as art
from server import logs
from server import service as svc
from server.schemas import ParameterOverrides, ScoreResponse, TelemetryRecord
from training import backtests as bt
from training import calibration as cal
from training import jobs as jobs_mod

LOGGER = logging.getLogger("platform.ondemand")
router = APIRouter()


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------
class ColumnMapping(BaseModel):
    KDVR: str = "KDVR"
    ChanNo: str = "ChanNo"
    siteKey: str | None = "siteKey"
    BayID: str | None = "BayID"
    DVRDate: str = "DVRDate"
    next_DVRDate: str = "next_DVRDate"
    Count: str = "Count"
    Count_IN: str = "Count_IN"
    Count_OUT: str = "Count_OUT"


class UploadConfig(BaseModel):
    name: str | None = None
    timezone: str | None = None
    column_mapping: ColumnMapping = Field(default_factory=ColumnMapping)
    entity_defaults: dict = Field(default_factory=dict)
    entity_overrides: dict = Field(default_factory=dict)
    auto_start_training: bool = False
    allow_duplicate: bool = False

    @field_validator("entity_overrides")
    @classmethod
    def _keys(cls, v):
        for k in v:
            if ":" not in str(k):
                raise ValueError(
                    f"entity_overrides key '{k}' must be an entity id like "
                    f"'163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8'"
                )
        return v


class TrainingJobRequest(BaseModel):
    upload_id: str
    entity_ids: list[str] | None = None
    strategy: Literal["per_entity", "pooled"] = "per_entity"
    validation: dict = Field(default_factory=dict)
    auto_calibrate: bool = True
    run_backtests: list[str] = Field(default_factory=lambda: list(C.BACKTEST_TYPES))
    promote: bool = False
    requested_by: str = "api"


class PromoteRequest(BaseModel):
    promoted_by: str = "api"
    audited_exception_reason: str | None = Field(
        None,
        description="Recorded justification for promoting a candidate that did not "
                    "pass every gate. Leave unset for a normal promotion.",
    )


class EntityScoreRequest(BaseModel):
    records: list[TelemetryRecord]
    params: ParameterOverrides = Field(default_factory=ParameterOverrides)
    model_id: str | None = None
    write_to_db: bool = True
    allow_fallback_model: bool = False
    kdvr: int | None = None
    chan_no: int | None = None
    site_key: int | None = None
    bay_id: str | None = None
    client: str | None = None


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
@router.post("/v1/datasets/upload")
async def upload_dataset(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(..., description="CSV/TSV telemetry export"),
    config: str = Form("{}", description="JSON upload configuration"),
    uploaded_by: str = Form("api"),
) -> dict:
    rid = getattr(request.state, "request_id", None)
    t0 = time.perf_counter()
    try:
        cfg = UploadConfig.model_validate_json(config)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"invalid config: {exc.errors()}")

    suffix = Path(file.filename or "upload.csv").suffix.lower() or ".csv"
    if suffix not in C.ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type '{suffix}'; allowed "
                   f"{sorted(C.ALLOWED_UPLOAD_SUFFIXES)}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="cart_upload_"))
    tmp_path = tmp_dir / (Path(file.filename or "upload.csv").name)
    size = 0
    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > C.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds the {C.MAX_UPLOAD_BYTES} byte limit",
                    )
                out.write(chunk)

        result = up.register_upload(
            tmp_path, file.filename or "upload.csv",
            cfg.model_dump(), uploaded_by, allow_duplicate=cfg.allow_duplicate,
        )
    except up.UploadRejected as exc:
        logs.log_request(rid, "/v1/datasets/upload", "POST", "rejected",
                         (time.perf_counter() - t0) * 1000, error=str(exc))
        raise HTTPException(status_code=422,
                            detail={"message": str(exc), "errors": exc.errors})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    body = {
        "upload_id": result.upload_id,
        "status": result.status,
        "duplicate_of": result.duplicate_of,
        "entities": result.entities,
        "n_entities": len(result.entities),
        "quality": result.quality,
        "warnings": result.warnings,
        "note": "Entities listed here were discovered in THIS upload only. No "
                "registry-wide or database-wide scan was performed.",
    }

    if cfg.auto_start_training and result.duplicate_of is None:
        trainable = [e["entity_id"] for e in result.entities if e.get("trainable")]
        if trainable:
            job_id = jobs_mod.create_job(
                result.upload_id, trainable, "per_entity",
                {"auto_calibrate": True, "run_backtests": list(C.BACKTEST_TYPES),
                 "source": "auto_start_training"},
                uploaded_by,
            )
            # Heavy work never runs inside the upload handler.
            background.add_task(jobs_mod.run_job, job_id)
            body["training_job_id"] = job_id
            body["auto_started_entities"] = trainable

    logs.log_request(rid, "/v1/datasets/upload", "POST", "ok",
                     (time.perf_counter() - t0) * 1000,
                     request_summary={"filename": file.filename, "bytes": size},
                     response_summary={"upload_id": result.upload_id,
                                       "n_entities": len(result.entities)})
    return body


@router.get("/v1/datasets")
def list_datasets(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"uploads": up.list_uploads(limit)}


@router.get("/v1/datasets/{upload_id}")
def get_dataset(upload_id: str) -> dict:
    row = up.get_upload(upload_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"upload '{upload_id}' not found")
    out = dict(row)
    out["config"] = from_json(row["config_json"], {})
    out["quality"] = from_json(row["quality_json"], {})
    out["warnings"] = from_json(row["warnings_json"], [])
    for k in ("config_json", "quality_json", "warnings_json", "errors_json"):
        out.pop(k, None)
    out["entities"] = _decorate_upload_entities(up.get_upload_entities(upload_id))
    return out


@router.get("/v1/datasets/{upload_id}/entities")
def get_dataset_entities(upload_id: str) -> dict:
    if up.get_upload(upload_id) is None:
        raise HTTPException(status_code=404, detail=f"upload '{upload_id}' not found")
    rows = _decorate_upload_entities(up.get_upload_entities(upload_id))
    return {"upload_id": upload_id, "n_entities": len(rows), "entities": rows}


def _decorate_upload_entities(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        d["blocking_reasons"] = from_json(r.get("blocking_reasons"), []) or []
        d["quality"] = from_json(r.get("quality_json"), {}) or {}
        d.pop("quality_json", None)
        d["trainable"] = bool(r.get("trainable"))
        d["model_status"] = bind.readiness(r["entity_id"])["model_status"]
        out.append(d)
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@router.post("/v1/training/jobs")
def create_training_job(req: TrainingJobRequest, background: BackgroundTasks) -> dict:
    upload = up.get_upload(req.upload_id)
    if upload is None:
        raise HTTPException(status_code=404,
                            detail=f"upload '{req.upload_id}' not found")

    discovered = {r["entity_id"]: r for r in up.get_upload_entities(req.upload_id)}
    if req.entity_ids:
        unknown = [e for e in req.entity_ids if e not in discovered]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "entity_ids must come from this upload; training is "
                               "never extended to other registered entities",
                    "unknown_for_upload": unknown,
                    "available": sorted(discovered),
                },
            )
        selected = list(req.entity_ids)
    else:
        selected = sorted(discovered)
        if len(selected) > 1:
            LOGGER.info("Job covers all %d entities discovered in %s",
                        len(selected), req.upload_id)

    if not selected:
        raise HTTPException(status_code=422,
                            detail=f"upload '{req.upload_id}' contains no entities")

    if req.strategy == "pooled":
        raise HTTPException(
            status_code=501,
            detail="strategy='pooled' must be requested explicitly and is not "
                   "implemented; unrelated bays are never silently pooled",
        )

    missing = [t for t in C.BACKTEST_TYPES if t not in req.run_backtests]
    config = {
        "auto_calibrate": True,          # mandatory default; cannot be disabled
        "run_backtests": list(C.BACKTEST_TYPES),
        "validation": req.validation,
        "requested_backtests": req.run_backtests,
        "promote_requested": req.promote,
    }
    job_id = jobs_mod.create_job(req.upload_id, selected, req.strategy, config,
                                 req.requested_by)
    background.add_task(jobs_mod.run_job, job_id)

    notes = [
        "Training is scoped to this upload and these entities only.",
        "Promotion is always manual; a finished job yields a candidate.",
    ]
    if missing:
        notes.append(
            f"auto_calibrate and all three backtests are mandatory defaults; "
            f"{missing} will run even though they were omitted"
        )
    if req.promote:
        notes.append("promote=true is ignored: candidates are never auto-promoted")
    return {
        "job_id": job_id, "upload_id": req.upload_id, "entity_ids": selected,
        "strategy": req.strategy, "status": jobs_mod.QUEUED,
        "auto_calibrate": True, "run_backtests": list(C.BACKTEST_TYPES),
        "promoted": False, "notes": notes,
    }


@router.get("/v1/training/jobs")
def list_training_jobs(upload_id: str | None = None,
                       limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"jobs": [_decorate_job(j) for j in jobs_mod.list_jobs(upload_id, limit)]}


@router.get("/v1/training/jobs/{job_id}")
def get_training_job(job_id: str) -> dict:
    job = jobs_mod.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return _decorate_job(job)


def _decorate_job(job: dict) -> dict:
    out = dict(job)
    out["entity_ids"] = from_json(job["entity_ids"], []) or []
    out["config"] = from_json(job.get("config_json"), {}) or {}
    out["results"] = from_json(job.get("results_json"), {}) or {}
    for k in ("entity_ids_json", "config_json", "results_json"):
        out.pop(k, None)
    reports = bt.reports_for_job(job["job_id"])
    out["backtests"] = {
        r["entity_id"]: {} for r in reports
    }
    for r in reports:
        out["backtests"].setdefault(r["entity_id"], {})[r["backtest_type"]] = {
            "report_id": r["report_id"], "status": r["status"],
            "reason": r.get("reason"),
        }
    return out


@router.get("/v1/training/jobs/{job_id}/backtests")
def job_backtests(job_id: str, entity_id: str | None = None) -> dict:
    if jobs_mod.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    rows = bt.reports_for_job(job_id, entity_id)
    by_entity: dict[str, dict] = {}
    for r in rows:
        by_entity.setdefault(r["entity_id"], {})[r["backtest_type"]] = {
            "report_id": r["report_id"], "backtest_type": r["backtest_type"],
            "status": r["status"], "reason": r.get("reason"),
            "model_id": r["model_id"], "calibration_id": r.get("calibration_id"),
            "window_start": r.get("window_start"), "window_end": r.get("window_end"),
            "artifacts": from_json(r.get("artifacts_json"), []) or [],
            "report_path": r.get("report_path"),
            "metrics_summary": _summary(r),
        }
    for entity, found in by_entity.items():
        for t in C.BACKTEST_TYPES:
            found.setdefault(t, {"backtest_type": t, "status": "missing",
                                 "reason": "report was not produced"})
    return {"job_id": job_id, "backtest_types": list(C.BACKTEST_TYPES),
            "entities": by_entity}


def _summary(report: dict) -> dict:
    m = from_json(report.get("metrics_json"), {}) or {}
    t = report["backtest_type"]
    if t == "forecast_accuracy":
        per = m.get("per_horizon", {})
        return {
            "n_scored": m.get("n_scored"),
            "mae_min_within_40min": (per.get("min_within_40min") or {}).get("MAE"),
            "mae_cart_at_20min": (per.get("cart_at_20min") or {}).get("MAE"),
            "shortage_pr_auc": (m.get("shortage_classification") or {}).get("PR_AUC"),
        }
    if t == "policy_replay":
        return {k: m.get(k) for k in
                ("baseline_events", "policy_events", "severity_reduction_pct",
                 "n_replenishment_actions", "total_carts_recommended")}
    return {k: m.get(k) for k in
            ("n_scenarios", "severity_reduction_min_pct",
             "severity_reduction_max_pct", "severity_reduction_spread_pct")}


@router.get("/v1/training/jobs/{job_id}/backtests/{backtest_type}")
def job_backtest(job_id: str, backtest_type: str, entity_id: str | None = None) -> dict:
    if backtest_type not in C.BACKTEST_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown backtest_type '{backtest_type}'; expected one of "
                   f"{list(C.BACKTEST_TYPES)}",
        )
    row = bt.get_report(job_id, backtest_type, entity_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"no '{backtest_type}' report for job '{job_id}'"
                   + (f" / entity '{entity_id}'" if entity_id else ""),
        )
    out = dict(row)
    out["metrics"] = from_json(row.get("metrics_json"), {})
    out["assumptions"] = from_json(row.get("assumptions_json"), {})
    out["artifacts"] = from_json(row.get("artifacts_json"), [])
    for k in ("metrics_json", "assumptions_json", "artifacts_json"):
        out.pop(k, None)
    return out


@router.get("/v1/training/jobs/{job_id}/calibration")
def job_calibration(job_id: str, entity_id: str | None = None) -> dict:
    if jobs_mod.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    record = cal.calibration_for_job(job_id, entity_id)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"no calibration record for job '{job_id}'")
    out = dict(record)
    out["metrics"] = from_json(record.get("metrics_json"), {})
    out["warnings"] = from_json(record.get("warnings_json"), [])
    for k in ("metrics_json", "warnings_json"):
        out.pop(k, None)
    return out


# --------------------------------------------------------------------------
# Model bindings and promotion
# --------------------------------------------------------------------------
def _entity_from_path(kdvr: int, chan_no: int, site_key: int | None = None,
                      bay_id: str | None = None) -> str:
    try:
        entity_id = reg.entity_id_for(kdvr, chan_no, site_key, bay_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        reg.get_entity(entity_id)
    except reg.UnknownEntity as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return entity_id


def _gone(kdvr: int, chan_no: int, suffix: str, bay_id: str | None = None) -> None:
    """Entity routes are site- (and bay-) scoped. Point the caller at the new URL."""
    bay = f"/bays/{bay_id or '{bay_id}'}" if C.INCLUDE_BAY_IN_ENTITY_ID else ""
    raise HTTPException(
        status_code=410,
        detail={
            "code": "ROUTE_MOVED",
            "message": "entity routes are site- and bay-scoped; the site and the "
                       "bay are part of the entity identity and cannot be inferred",
            "use": f"/v1/sites/{{siteKey}}/entities/{kdvr}/{chan_no}{bay}{suffix}",
        },
    )


def _model_readiness(site_key: int, kdvr: int, chan_no: int,
                     bay_id: str | None = None) -> dict:
    entity_id = _entity_from_path(kdvr, chan_no, site_key, bay_id)
    out = bind.readiness(entity_id)
    out["score_url"] = up.score_url_for(kdvr, chan_no, site_key, bay_id)
    return out


def _promote(site_key: int, kdvr: int, chan_no: int, model_id: str,
             req: PromoteRequest, bay_id: str | None = None) -> dict:
    entity_id = _entity_from_path(kdvr, chan_no, site_key, bay_id)
    try:
        return bind.promote(entity_id, model_id, req.promoted_by,
                            req.audited_exception_reason)
    except bind.BindingNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except bind.PromotionBlocked as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "blocking_reasons": exc.reasons,
                    "hint": "fix the gate or record an audited exception via "
                            "audited_exception_reason"},
        )


def _rollback(site_key: int, kdvr: int, chan_no: int, req: PromoteRequest,
              bay_id: str | None = None) -> dict:
    entity_id = _entity_from_path(kdvr, chan_no, site_key, bay_id)
    try:
        return bind.rollback(entity_id, req.promoted_by)
    except bind.BindingNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/bays/{bay_id}/model")
def entity_model_bay_scoped(site_key: int, kdvr: int, chan_no: int,
                            bay_id: str) -> dict:
    return _model_readiness(site_key, kdvr, chan_no, bay_id)


@router.post("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/bays/{bay_id}"
             "/models/{model_id}/promote")
def promote_model_bay_scoped(site_key: int, kdvr: int, chan_no: int, bay_id: str,
                             model_id: str, req: PromoteRequest) -> dict:
    return _promote(site_key, kdvr, chan_no, model_id, req, bay_id)


@router.post("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/bays/{bay_id}"
             "/models/rollback")
def rollback_model_bay_scoped(site_key: int, kdvr: int, chan_no: int, bay_id: str,
                              req: PromoteRequest) -> dict:
    return _rollback(site_key, kdvr, chan_no, req, bay_id)


@router.get("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/model")
def entity_model_site_scoped(site_key: int, kdvr: int, chan_no: int,
                             bay_id: str | None = None) -> dict:
    if C.INCLUDE_BAY_IN_ENTITY_ID and bay_id is None:
        _gone(kdvr, chan_no, "/model")
    return _model_readiness(site_key, kdvr, chan_no, bay_id)


@router.post("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}"
             "/models/{model_id}/promote")
def promote_model_site_scoped(site_key: int, kdvr: int, chan_no: int,
                              model_id: str, req: PromoteRequest,
                              bay_id: str | None = None) -> dict:
    if C.INCLUDE_BAY_IN_ENTITY_ID and bay_id is None:
        _gone(kdvr, chan_no, f"/models/{model_id}/promote")
    return _promote(site_key, kdvr, chan_no, model_id, req, bay_id)


@router.post("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/models/rollback")
def rollback_model_site_scoped(site_key: int, kdvr: int, chan_no: int,
                               req: PromoteRequest,
                               bay_id: str | None = None) -> dict:
    if C.INCLUDE_BAY_IN_ENTITY_ID and bay_id is None:
        _gone(kdvr, chan_no, "/models/rollback")
    return _rollback(site_key, kdvr, chan_no, req, bay_id)


# --- legacy, site-less forms -------------------------------------------
@router.get("/v1/entities/{kdvr}/{chan_no}/model")
def entity_model(kdvr: int, chan_no: int, site_key: int | None = None,
                 bay_id: str | None = None) -> dict:
    if (C.INCLUDE_SITE_IN_ENTITY_ID and site_key is None) or (
            C.INCLUDE_BAY_IN_ENTITY_ID and bay_id is None):
        _gone(kdvr, chan_no, "/model")
    return _model_readiness(site_key, kdvr, chan_no, bay_id)


@router.post("/v1/entities/{kdvr}/{chan_no}/models/{model_id}/promote")
def promote_model(kdvr: int, chan_no: int, model_id: str, req: PromoteRequest,
                  site_key: int | None = None, bay_id: str | None = None) -> dict:
    if (C.INCLUDE_SITE_IN_ENTITY_ID and site_key is None) or (
            C.INCLUDE_BAY_IN_ENTITY_ID and bay_id is None):
        _gone(kdvr, chan_no, f"/models/{model_id}/promote")
    return _promote(site_key, kdvr, chan_no, model_id, req, bay_id)


@router.post("/v1/entities/{kdvr}/{chan_no}/models/rollback")
def rollback_model(kdvr: int, chan_no: int, req: PromoteRequest,
                   site_key: int | None = None, bay_id: str | None = None) -> dict:
    if (C.INCLUDE_SITE_IN_ENTITY_ID and site_key is None) or (
            C.INCLUDE_BAY_IN_ENTITY_ID and bay_id is None):
        _gone(kdvr, chan_no, "/models/rollback")
    return _rollback(site_key, kdvr, chan_no, req, bay_id)


# --------------------------------------------------------------------------
# Entity scoring
# --------------------------------------------------------------------------
def _score_entity(entity_id: str, req: EntityScoreRequest, request: Request,
                  path_kdvr: int, path_chan: int) -> ScoreResponse:
    rid = getattr(request.state, "request_id", None)
    # Log the URL that was actually called, never a reconstruction of it: the
    # same handler serves the bay-scoped, site-scoped and legacy routes.
    endpoint = request.url.path
    if req.kdvr is not None and int(req.kdvr) != int(path_kdvr):
        raise HTTPException(status_code=422,
                            detail=f"body kdvr {req.kdvr} does not match path "
                                   f"{path_kdvr}")
    if req.chan_no is not None and int(req.chan_no) != int(path_chan):
        raise HTTPException(status_code=422,
                            detail=f"body chan_no {req.chan_no} does not match path "
                                   f"{path_chan}")
    t0 = time.perf_counter()
    try:
        result = svc.score(
            entity_id=entity_id,
            records=[r.model_dump() for r in req.records],
            overrides=req.params.as_dict(),
            model_id=req.model_id,
            write_to_db=req.write_to_db,
            request_id=rid,
            allow_fallback_model=req.allow_fallback_model,
            write_artifact=True,
            publish=True,
        )
    except svc.ModelNotReady as exc:
        logs.log_request(rid, endpoint, "POST",
                         "rejected", (time.perf_counter() - t0) * 1000, entity_id,
                         client=req.client, error=str(exc))
        raise HTTPException(status_code=409,
                            detail={"code": "MODEL_NOT_READY", "message": str(exc),
                                    "readiness": exc.readiness})
    except svc.InsufficientHistory as exc:
        logs.log_request(rid, endpoint, "POST",
                         "rejected", (time.perf_counter() - t0) * 1000, entity_id,
                         client=req.client, error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc))
    except svc.EntityNotScoreable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    dt = (time.perf_counter() - t0) * 1000
    logs.log_request(
        rid, endpoint, "POST", "ok", dt,
        entity_id, result["site_key"], 200, req.client,
        {"records": len(req.records)},
        {"risk": result["shortage_risk"], "alert_state": result["alert_state"],
         "should_replenish": result["action"]["should_replenish"],
         "artifact": result.get("artifact_path")},
    )
    if result["action"]["should_replenish"]:
        LOGGER.info("[%s] %s REPLENISH %s carts (risk %.2f)", entity_id,
                    result["decision_time"][:16], result["action"]["quantity"],
                    result["shortage_risk"])
    return ScoreResponse(**result)


@router.post("/v1/entities/{kdvr}/{chan_no}/score", response_model=ScoreResponse)
def score_entity(kdvr: int, chan_no: int, req: EntityScoreRequest,
                 request: Request) -> ScoreResponse:
    if (C.INCLUDE_SITE_IN_ENTITY_ID and req.site_key is None) or (
            C.INCLUDE_BAY_IN_ENTITY_ID and req.bay_id is None):
        _gone(kdvr, chan_no, "/score")
    entity_id = _entity_from_path(kdvr, chan_no, req.site_key, req.bay_id)
    return _score_entity(entity_id, req, request, kdvr, chan_no)


@router.post("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/score",
             response_model=ScoreResponse)
def score_entity_site_scoped(site_key: int, kdvr: int, chan_no: int,
                             req: EntityScoreRequest, request: Request) -> ScoreResponse:
    _check_site(req, site_key)
    if C.INCLUDE_BAY_IN_ENTITY_ID and req.bay_id is None:
        _gone(kdvr, chan_no, "/score")
    entity_id = _entity_from_path(kdvr, chan_no, site_key, req.bay_id)
    return _score_entity(entity_id, req, request, kdvr, chan_no)


@router.post("/v1/sites/{site_key}/entities/{kdvr}/{chan_no}/bays/{bay_id}/score",
             response_model=ScoreResponse)
def score_entity_bay_scoped(site_key: int, kdvr: int, chan_no: int, bay_id: str,
                            req: EntityScoreRequest,
                            request: Request) -> ScoreResponse:
    _check_site(req, site_key)
    if req.bay_id is not None and str(req.bay_id) != str(bay_id):
        raise HTTPException(status_code=422,
                            detail=f"body bay_id {req.bay_id} does not match path "
                                   f"{bay_id}")
    entity_id = _entity_from_path(kdvr, chan_no, site_key, bay_id)
    return _score_entity(entity_id, req, request, kdvr, chan_no)


def _check_site(req: EntityScoreRequest, site_key: int) -> None:
    if req.site_key is not None and int(req.site_key) != int(site_key):
        raise HTTPException(status_code=422,
                            detail=f"body site_key {req.site_key} does not match path "
                                   f"{site_key}")


# --------------------------------------------------------------------------
# Scoring artifacts
# --------------------------------------------------------------------------
@router.get("/v1/scoring/artifacts")
def list_scoring_artifacts(entity_id: str | None = None,
                           limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"artifacts": art.list_artifacts(entity_id, limit)}


@router.get("/v1/scoring/artifacts/{request_id}")
def get_scoring_artifact(request_id: str, download: bool = False):
    row = art.get_artifact(request_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"artifact '{request_id}' not found")
    path = Path(row["artifact_path"])
    if not path.exists():
        raise HTTPException(status_code=410,
                            detail=f"artifact file no longer available: {path.name}")
    if download:
        return FileResponse(path, media_type="text/csv", filename=path.name)
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {**row, "rows": rows}


# --------------------------------------------------------------------------
# Integration / outbox observability
# --------------------------------------------------------------------------
@router.get("/v1/integrations/outbox")
def outbox_status(status: str | None = None, entity_id: str | None = None,
                  event_type: str | None = None,
                  limit: int = Query(100, ge=1, le=1000)) -> dict:
    return {
        "receiver_url": C.RESULT_RECEIVER_URL,
        "publishing_enabled": C.RESULT_RECEIVER_ENABLED,
        "worker_running": WORKER.running,
        "counts": outbox.stats(),
        "events": [
            {k: v for k, v in e.items() if k != "payload_json"}
            for e in outbox.list_events(status, entity_id, event_type, limit)
        ],
    }


@router.get("/v1/integrations/outbox/{event_id:path}")
def outbox_event(event_id: str) -> dict:
    row = outbox.get_event(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event '{event_id}' not found")
    out = dict(row)
    out["payload"] = from_json(row["payload_json"], {})
    out.pop("payload_json", None)
    return out


class OutboxRetryRequest(BaseModel):
    event_id: str | None = None
    flush_now: bool = True


@router.post("/v1/integrations/outbox/retry")
def outbox_retry(req: OutboxRetryRequest) -> dict:
    requeued = outbox.retry_dead_letters(req.event_id)
    flushed = flush_once() if req.flush_now else {"skipped": True}
    return {"requeued": requeued, "flush": flushed, "counts": outbox.stats()}
