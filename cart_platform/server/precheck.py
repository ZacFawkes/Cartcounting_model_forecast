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