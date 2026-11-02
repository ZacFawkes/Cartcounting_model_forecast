"""API for RED calibration eligibility precheck."""

from __future__ import annotations

import logging
import tempfile
import threading
import time

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from settings import INCLUDE_SITE_IN_ENTITY_ID

from data_loader import prepare

from registry import uploads as up

from config import FEATURE_WINDOWS
from training.model_config import FrozenModelConfig
from training.precheck import find_valid_red_threshold


# ============================================================
# CONFIGURATION
# ============================================================

LOGGER = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/training",
    tags=["Training Precheck"],
)

# Keep completed jobs for 10 minutes.
PRECHECK_JOB_TTL_SECONDS = 600

# Run cleanup every 60 seconds.
PRECHECK_CLEANUP_INTERVAL_SECONDS = 60

# Amber threshold offset.
AMBER_THRESHOLD_OFFSET = 15

# Temporary job storage.
PRECHECK_JOBS: dict[str, dict] = {}

PRECHECK_LOCK = threading.Lock()


# ============================================================
# TIME HELPERS
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_monotonic() -> float:
    return time.monotonic()


# ============================================================
# JOB STORAGE
# ============================================================

def create_job(
    job_id: str,
    entity_id: str,
):
    """Create temporary precheck job."""

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
            "_completed_monotonic": None,
        }


def update_precheck_job(
    job_id: str,
    **kwargs,
):
    """Thread-safe job update."""

    with PRECHECK_LOCK:

        job = PRECHECK_JOBS.get(job_id)

        if job is None:
            LOGGER.warning(
                "Cannot update missing precheck job: %s",
                job_id,
            )
            return

        job.update(kwargs)

        if kwargs.get("status") in (
            "SUCCEEDED",
            "FAILED",
        ):
            job["_completed_monotonic"] = now_monotonic()


def get_precheck_job(
    job_id: str,
) -> dict | None:
    """Get job without exposing internal metadata."""

    with PRECHECK_LOCK:

        job = PRECHECK_JOBS.get(job_id)

        if job is None:
            return None

        # Check whether completed job has expired.
        completed_at = job.get(
            "_completed_monotonic"
        )

        if completed_at is not None:

            elapsed = (
                now_monotonic() - completed_at
            )

            if elapsed >= PRECHECK_JOB_TTL_SECONDS:

                del PRECHECK_JOBS[job_id]

                return None

        return {
            key: value
            for key, value in job.items()
            if not key.startswith("_")
        }


# ============================================================
# CLEANUP EXPIRED JOBS
# ============================================================

def cleanup_expired_jobs() -> int:
    """
    Remove completed jobs older than TTL.

    PENDING and RUNNING jobs are preserved.
    """

    current_time = now_monotonic()

    with PRECHECK_LOCK:

        expired_ids = []

        for job_id, job in PRECHECK_JOBS.items():

            completed_at = job.get(
                "_completed_monotonic"
            )

            if completed_at is None:
                continue

            elapsed = current_time - completed_at

            if elapsed >= PRECHECK_JOB_TTL_SECONDS:
                expired_ids.append(job_id)

        for job_id in expired_ids:
            del PRECHECK_JOBS[job_id]

    if expired_ids:
        LOGGER.info(
            "Removed %d expired precheck jobs",
            len(expired_ids),
        )

    return len(expired_ids)


def precheck_cleanup_worker():
    """Periodically clean expired jobs."""

    while True:

        time.sleep(
            PRECHECK_CLEANUP_INTERVAL_SECONDS
        )

        try:
            cleanup_expired_jobs()

        except Exception:
            LOGGER.exception(
                "Precheck cleanup failed"
            )


def start_precheck_cleanup_worker():
    """
    Start cleanup daemon.

    Should be called once per Python process.
    """

    global _CLEANUP_STARTED

    with _CLEANUP_START_LOCK:

        if _CLEANUP_STARTED:
            return

        worker = threading.Thread(
            target=precheck_cleanup_worker,
            name="precheck-cleanup",
            daemon=True,
        )

        worker.start()

        _CLEANUP_STARTED = True

        LOGGER.info(
            "Precheck cleanup worker started"
        )


_CLEANUP_STARTED = False
_CLEANUP_START_LOCK = threading.Lock()


# ============================================================
# VALIDATE REQUEST
# ============================================================

def validate_precheck_request(
    filename: str | None,
    red_threshold: int,
) -> str:
    """Validate file extension. Legacy RED input is ignored; search starts at 0."""

    suffix = Path(
        filename or ""
    ).suffix.lower()

    if suffix not in up.ALLOWED_UPLOAD_SUFFIXES:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file extension: {suffix}"
            ),
        )

    return suffix


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

async def save_upload_file(
    file: UploadFile,
    destination: Path,
):
    """Save uploaded file with size validation."""

    try:

        with destination.open("wb") as out:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                out.write(chunk)

                if out.tell() > up.MAX_UPLOAD_BYTES:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file exceeds "
                            "maximum allowed size"
                        ),
                    )

    except Exception:

        destination.unlink(
            missing_ok=True
        )

        raise


# ============================================================
# RESOLVE ENTITY CONFIGURATION
# ============================================================

# ============================================================
# LOAD DATASET
# ============================================================

def load_precheck_dataset(
    source: Path,
    entity_id: str,
):
    """Read, validate, clean and prepare dataset."""

    raw = up.read_dataset(
        source
    )

    errors = up.validate_schema(
        raw,
        require_site=INCLUDE_SITE_IN_ENTITY_ID,
    )

    if errors:

        raise ValueError(
            f"Dataset schema validation failed: {errors}"
        )

    typed, _ = up.coerce_types(
        raw
    )

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
            f"Entity {entity_id} is not present "
            "in uploaded dataset"
        )

    return (
        sub,
        quality,
        warnings,
    )


# ============================================================
# CALCULATE BAY CAPACITY
# ============================================================

def calculate_bay_capacity(
    sub: pd.DataFrame,
) -> int:
    """
    bay_capacity = MAX(Count).

    Supports:
        Count
        cart_current
    """

    if "Count" in sub.columns:
        count_col = "Count"

    elif "cart_current" in sub.columns:
        count_col = "cart_current"

    else:

        raise ValueError(
            "Cannot calculate bay_capacity: "
            "Count/cart_current column missing"
        )

    values = pd.to_numeric(
        sub[count_col],
        errors="coerce",
    ).dropna()

    if values.empty:

        raise ValueError(
            "No valid Count values found"
        )

    bay_capacity = int(
        values.max()
    )

    if bay_capacity <= 0:

        raise ValueError(
            "Calculated bay_capacity must be > 0"
        )

    return bay_capacity


# ============================================================
# BUILD PRECHECK RESPONSE
# ============================================================

def build_precheck_result(

    *,

    entity_id: str,

    filename: str | None,

    sub: pd.DataFrame,

    quality,

    warnings,

    result: dict,


):

    """

    Build enriched response.

    bay_capacity = MAX(Count)

    amber_threshold = recommended_red + 15

    """

    bay_capacity = calculate_bay_capacity(sub)

    recommended_red = result.get("recommended_red")

    amber_threshold = (

        int(recommended_red) + AMBER_THRESHOLD_OFFSET

        if recommended_red is not None

        else None

    )

    return {

        **result,

        "entity_id": entity_id,

        "filename": filename,

        "bay_capacity": bay_capacity,


        "amber_threshold": amber_threshold,

        "n_entity_records": len(sub),

        "quality": quality,

        "upload_warnings": warnings,

    }


# ============================================================
# RUN PRECHECK
# ============================================================

def run_precheck(
    *,
    source: Path,
    entity_id: str,
    red_threshold: int,
    original_filename: str | None,
):
    """
    Execute RED calibration eligibility precheck.

    No upload registration.
    No training.
    No model promotion.
    No configuration update.
    """

    sub, quality, warnings = load_precheck_dataset(source, entity_id)
    bay_capacity = calculate_bay_capacity(sub)

    # Build only feature/target settings; never resolve entity operations.
    frozen = FrozenModelConfig(
        red_model=0,
        amber_model=AMBER_THRESHOLD_OFFSET,
        feature_windows=tuple(FEATURE_WINDOWS),
        primary_horizon=3,
        planning_horizon=6,
        calibration_red=0,
    )
    result = find_valid_red_threshold(
        prepared=sub,
        frozen=frozen,
        max_red=bay_capacity,
    )

    return build_precheck_result(
        entity_id=entity_id,
        filename=original_filename,
        sub=sub,
        quality=quality,
        warnings=warnings,
        result=result,
    )


# ============================================================
# 1. SYNCHRONOUS PRECHECK API
# ============================================================

@router.post("/precheck")
async def training_precheck(
    file: UploadFile = File(...),
    entity_id: str = Form(...),
    red_threshold: int = Form(0),
):
    """
    Run precheck synchronously.

    Result is returned directly.
    No job is stored in RAM.
    """

    suffix = validate_precheck_request(
        file.filename,
        red_threshold,
    )

    try:

        with tempfile.TemporaryDirectory(
            prefix="cart_precheck_"
        ) as temp_dir:

            source = (
                Path(temp_dir)
                / f"dataset{suffix}"
            )

            await save_upload_file(
                file,
                source,
            )

            result = run_precheck(
                source=source,
                entity_id=entity_id,
                red_threshold=red_threshold,
                original_filename=file.filename,
            )

            return result

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

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:

        LOGGER.exception(
            "Synchronous precheck failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Precheck failed: {exc}",
        )


# ============================================================
# 2. BACKGROUND PRECHECK WORKER
# ============================================================

def execute_precheck(
    job_id: str,
    source: Path,
    entity_id: str,
    red_threshold: int,
    original_filename: str | None,
):
    """Execute asynchronous precheck."""

    update_precheck_job(
        job_id,
        status="RUNNING",
        started_at=now_iso(),
    )

    try:

        result = run_precheck(
            source=source,
            entity_id=entity_id,
            red_threshold=red_threshold,
            original_filename=original_filename,
        )

        update_precheck_job(
            job_id,
            status="SUCCEEDED",
            finished_at=now_iso(),
            result=result,
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

        source.unlink(
            missing_ok=True
        )


# ============================================================
# 3. CREATE ASYNC PRECHECK JOB
# ============================================================

@router.post(
    "/precheck/jobs",
    status_code=202,
)
async def create_precheck_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    entity_id: str = Form(...),
    red_threshold: int = Form(0),
):
    """
    Create asynchronous precheck job.

    Returns immediately with job_id.
    """

    suffix = validate_precheck_request(
        file.filename,
        red_threshold,
    )

    job_id = (
        f"precheck_{uuid4().hex[:12]}"
    )

    temp_dir = (
        Path(tempfile.gettempdir())
        / "cart_precheck_jobs"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = (
        temp_dir
        / f"{job_id}{suffix}"
    )

    # Save uploaded file.
    await save_upload_file(
        file,
        source,
    )

    # Register job in RAM.
    create_job(
        job_id,
        entity_id,
    )

    # Schedule background execution.
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
        "status_url": (
            f"/v1/training/precheck/jobs/{job_id}"
        ),
    }


# ============================================================
# 4. GET ASYNC PRECHECK RESULT
# ============================================================

@router.get(
    "/precheck/jobs/{job_id}"
)
def check_precheck_job(
    job_id: str,
):
    """
    Get job status and result.

    Completed jobs remain available for 10 minutes,
    allowing CMS to retry requests.
    """

    job = get_precheck_job(
        job_id
    )

    if job is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Precheck job not found or expired"
            ),
        )

    return job


# ============================================================
# 5. START CLEANUP WORKER
# ============================================================

# Start one cleanup daemon per Python process.
start_precheck_cleanup_worker()