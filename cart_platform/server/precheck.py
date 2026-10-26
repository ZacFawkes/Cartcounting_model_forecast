"""API for RED calibration eligibility precheck."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
import logging

LOGGER = logging.getLogger(__name__)
from settings import INCLUDE_SITE_IN_ENTITY_ID

from data_loader import prepare

from registry import uploads as up
from registry import entities as reg

from training.model_config import resolve_for_training
from training.precheck import find_valid_red_threshold


router = APIRouter(
    prefix="/v1/training",
    tags=["Training Precheck"],
)


@router.post("/precheck")
async def training_precheck(
    file: UploadFile = File(...),
    entity_id: str = Form(...),
    red_threshold: int = Form(...),
):
    """
    Check RED calibration data eligibility.

    No upload registration.
    No training.
    No model promotion.
    No configuration update.
    """

    if red_threshold < 0:
        raise HTTPException(
            status_code=422,
            detail="red_threshold must be >= 0",
        )

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in up.ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file extension: {suffix}",
        )

    # Use the same configuration resolver as trainer.py.
    # This reads the existing entity configuration.
    try:
        entity_cfg = reg.get_entity(entity_id)

        if not entity_cfg:
            raise HTTPException(
                status_code=404,
                detail=f"Entity not found: {entity_id}",
            )

        capacity = entity_cfg.get("bay_capacity")

        if capacity is None:
            raise HTTPException(
                status_code=422,
                detail="Entity has no configured bay_capacity",
            )

        max_red = int(float(capacity))

        # A precheck identifier, not a training job.
        snapshot = resolve_for_training(
            entity_id,
            job_id="precheck",
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Configuration resolution failed: {exc}",
        )

    # Read the uploaded file using the same upload parser.
    # No registration or database mutation.
    try:
        with tempfile.TemporaryDirectory(
            prefix="cart_precheck_"
        ) as temp_dir:

            source = Path(temp_dir) / (
                "dataset" + suffix
            )

            with source.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    out.write(chunk)

                    if out.tell() > up.MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Uploaded file exceeds size limit",
                        )

            raw = up.read_dataset(source)

            errors = up.validate_schema(
                raw,
                require_site=INCLUDE_SITE_IN_ENTITY_ID,
            )

            if errors:
                raise HTTPException(
                    status_code=422,
                    detail=errors,
                )

            typed, _ = up.coerce_types(raw)

            clean, quality, warnings = up.quality_report(
                typed
            )

            prepared = prepare(
                clean,
                include_site_in_entity=INCLUDE_SITE_IN_ENTITY_ID,
            )

            sub = prepared[
                prepared["entity_id"] == entity_id
            ].reset_index(drop=True)

            if sub.empty:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Entity {entity_id} is not present "
                        "in the uploaded CSV"
                    ),
                )

            result = find_valid_red_threshold(
                prepared=sub,
                entity_cfg=entity_cfg,
                frozen=snapshot.frozen,
                initial_red=red_threshold,
                max_red=max_red,
            )

            return {
                "entity_id": entity_id,
                "filename": file.filename,
                "bay_capacity": capacity,
                "n_entity_records": len(sub),
                "quality": quality,
                "upload_warnings": warnings,
                **result,
            }

    except HTTPException:
        raise

    except up.UploadRejected as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "errors": exc.errors,
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Precheck failed: {exc}",
        )
# ============================================================
# ASYNC PRECHECK JOB API
# ============================================================

import asyncio
import threading
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import BackgroundTasks
from fastapi.concurrency import run_in_threadpool


PRECHECK_JOBS = {}
PRECHECK_LOCK = threading.Lock()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def update_precheck_job(job_id: str, **kwargs):
    with PRECHECK_LOCK:
        PRECHECK_JOBS[job_id].update(kwargs)


def get_precheck_job(job_id: str):
    with PRECHECK_LOCK:
        job = PRECHECK_JOBS.get(job_id)
        return dict(job) if job else None


def execute_precheck(
    job_id: str,
    source: Path,
    entity_id: str,
    red_threshold: int,
    original_filename: str,
):
    update_precheck_job(
        job_id,
        status="RUNNING",
        started_at=now_iso(),
    )

    try:
        entity_cfg = reg.get_entity(entity_id)

        if not entity_cfg:
            raise ValueError(
                f"Entity not found: {entity_id}"
            )

        capacity = entity_cfg.get("bay_capacity")

        if capacity is None:
            raise ValueError(
                "Entity has no configured bay_capacity"
            )

        max_red = int(float(capacity))

        snapshot = resolve_for_training(
            entity_id,
            job_id="precheck",
        )

        raw = up.read_dataset(source)

        errors = up.validate_schema(
            raw,
            require_site=INCLUDE_SITE_IN_ENTITY_ID,
        )

        if errors:
            raise ValueError(str(errors))

        typed, _ = up.coerce_types(raw)

        clean, quality, warnings = up.quality_report(
            typed
        )

        prepared = prepare(
            clean,
            include_site_in_entity=INCLUDE_SITE_IN_ENTITY_ID,
        )

        sub = prepared[
            prepared["entity_id"] == entity_id
        ].reset_index(drop=True)

        if sub.empty:
            raise ValueError(
                f"Entity {entity_id} not found in dataset"
            )

        result = find_valid_red_threshold(
            prepared=sub,
            entity_cfg=entity_cfg,
            frozen=snapshot.frozen,
            initial_red=red_threshold,
            max_red=max_red,
        )

        update_precheck_job(
            job_id,
            status="SUCCEEDED",
            finished_at=now_iso(),
            result={
                "entity_id": entity_id,
                "filename": original_filename,
                "bay_capacity": capacity,
                "n_entity_records": len(sub),
                "quality": quality,
                "upload_warnings": warnings,
                **result,
            },
        )

    except Exception as exc:
        LOGGER.exception(
            "Precheck job %s failed",
            job_id,
        )

        update_precheck_job(
            job_id,
            status="FAILED",
            finished_at=now_iso(),
            error=str(exc),
        )

    finally:
        source.unlink(missing_ok=True)


@router.post("/precheck/jobs", status_code=202)
async def create_precheck_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    entity_id: str = Form(...),
    red_threshold: int = Form(...),
):
    if red_threshold < 0:
        raise HTTPException(
            status_code=422,
            detail="red_threshold must be >= 0",
        )

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in up.ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported extension: {suffix}",
        )

    job_id = f"precheck_{uuid4().hex[:12]}"

    # Persist uploaded file until background job completes.
    temp_dir = Path(tempfile.gettempdir()) / "cart_precheck_jobs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    source = temp_dir / f"{job_id}{suffix}"

    try:
        with source.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)

                if out.tell() > up.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds size limit",
                    )
    except Exception:
        source.unlink(missing_ok=True)
        raise

    with PRECHECK_LOCK:
        PRECHECK_JOBS[job_id] = {
            "job_id": job_id,
            "entity_id": entity_id,
            "status": "PENDING",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    background_tasks.add_task(
        execute_precheck,
        job_id,
        source,
        entity_id,
        red_threshold,
        file.filename,
    )

    return {
        "job_id": job_id,
        "status": "PENDING",
        "status_url": f"/v1/training/precheck/jobs/{job_id}",
    }


@router.get("/precheck/jobs/{job_id}")
def check_precheck_job(job_id: str):
    job = get_precheck_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Precheck job not found",
        )

    return job