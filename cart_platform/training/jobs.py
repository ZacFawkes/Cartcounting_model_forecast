"""Durable training job runner.

A job trains **only** the entities selected from **one** upload. Nothing else in
the registry is read, trained or modified. Calibration and all three backtests
run automatically; promotion never does.

Jobs run on a background thread and persist their state at every transition, so
a restart leaves a crashed job marked ``failed`` rather than a false SUCCESS.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from settings import BACKTEST_TYPES, TRAINING_ARTIFACT_DIR

from integrations import events as ev  # noqa: E402
from registry import entities as reg  # noqa: E402
from registry import uploads as up  # noqa: E402
from registry.database import as_json, execute, from_json, query, query_one, utc_now  # noqa: E402
import settings as C  # noqa: E402
from training import alert_bands as alb  # noqa: E402
from training import backtests as bt  # noqa: E402
from training import calibration as cal  # noqa: E402
from training.model_config import (  # noqa: E402
    InvalidEntityConfig,
    compatibility_warnings,
    persist as persist_snapshot,
    resolve_for_training,
)
from training.trainer import InsufficientTrainingData, train_entity  # noqa: E402

LOGGER = logging.getLogger("platform.jobs")

QUEUED, RUNNING, SUCCEEDED, PARTIAL, FAILED = (
    "queued", "running", "succeeded", "partial", "failed"
)
CANDIDATE_READY = "candidate_ready"

_THREADS: dict[str, threading.Thread] = {}


# --------------------------------------------------------------------------
# Job records
# --------------------------------------------------------------------------
def new_job_id() -> str:
    return f"train_{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}"


def create_job(upload_id: str, entity_ids: list[str], strategy: str, config: dict,
               requested_by: str = "api") -> str:
    job_id = new_job_id()
    execute(
        """INSERT INTO training_jobs
           (job_id, upload_id, entity_ids, strategy, config_json, status, progress,
            message, requested_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (job_id, upload_id, as_json(entity_ids), strategy, as_json(config), QUEUED,
         0.0, "queued", requested_by, utc_now()),
    )
    return job_id


def get_job(job_id: str) -> dict | None:
    return query_one("SELECT * FROM training_jobs WHERE job_id=?", (job_id,))


def list_jobs(upload_id: str | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM training_jobs WHERE 1=1"
    params: list = []
    if upload_id:
        sql += " AND upload_id=?"
        params.append(upload_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return query(sql, tuple(params))


def _update(job_id: str, **fields) -> None:
    fields["heartbeat_at"] = utc_now()
    sets = ", ".join(f"{k}=?" for k in fields)
    execute(f"UPDATE training_jobs SET {sets} WHERE job_id=?",
            tuple(fields.values()) + (job_id,))


def recover_orphaned_jobs() -> int:
    """A job still marked running after a restart did not finish."""
    rows = query("SELECT job_id FROM training_jobs WHERE status IN (?,?)",
                 (RUNNING, QUEUED))
    alive = {j for j, t in _THREADS.items() if t.is_alive()}
    n = 0
    for r in rows:
        if r["job_id"] in alive:
            continue
        _update(r["job_id"], status=FAILED,
                error="process restarted before the job completed",
                message="orphaned", finished_at=utc_now())
        n += 1
    if n:
        LOGGER.warning("Marked %d orphaned training job(s) as failed", n)
    return n


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def start_job(job_id: str) -> None:
    t = threading.Thread(target=run_job, args=(job_id,), daemon=True,
                         name=f"train-{job_id}")
    _THREADS[job_id] = t
    t.start()


def run_job(job_id: str) -> dict:
    """Synchronously execute a job. Used by the worker thread and by tests."""
    job = get_job(job_id)
    if job is None:
        raise KeyError(f"job '{job_id}' not found")
    upload_id = job["upload_id"]
    entity_ids = from_json(job["entity_ids"], []) or []
    strategy = job["strategy"]
    config = from_json(job["config_json"], {}) or {}

    _update(job_id, status=RUNNING, started_at=utc_now(), message="loading upload")
    results: dict[str, dict] = {}
    try:
        upload = up.get_upload(upload_id)
        if upload is None:
            raise KeyError(f"upload '{upload_id}' not found")
        prepared = up.load_prepared_upload(upload_id)
        checksum = upload["file_checksum"]
    except Exception as exc:
        LOGGER.exception("Job %s could not load upload", job_id)
        _update(job_id, status=FAILED, error=str(exc), message="upload load failed",
                finished_at=utc_now(), progress=1.0)
        return {"job_id": job_id, "status": FAILED, "error": str(exc)}

    total = max(len(entity_ids), 1)
    for i, entity_id in enumerate(entity_ids):
        _update(job_id, progress=i / total, message=f"training {entity_id}")
        results[entity_id] = _train_one(
            job_id, upload_id, entity_id, prepared, checksum, strategy, config
        )

    statuses = {r["status"] for r in results.values()}
    if statuses == {CANDIDATE_READY}:
        status = SUCCEEDED
    elif CANDIDATE_READY in statuses:
        status = PARTIAL
    else:
        status = FAILED
    _update(job_id, status=status, progress=1.0, message="finished",
            results_json=as_json(results), finished_at=utc_now())
    LOGGER.info("Job %s finished: %s", job_id, status)
    return {"job_id": job_id, "status": status, "results": results}


def _train_one(job_id, upload_id, entity_id, prepared, checksum, strategy,
               config) -> dict:
    out: dict = {"entity_id": entity_id, "status": FAILED, "warnings": []}
    artifact_dir = Path(TRAINING_ARTIFACT_DIR) / job_id / entity_id.replace(":", "_")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        entity_cfg = reg.get_entity(entity_id)
    except reg.UnknownEntity as exc:
        out["error"] = str(exc)
        return out

    sub = prepared[prepared["entity_id"] == entity_id].reset_index(drop=True)
    if sub.empty:
        out["status"] = "insufficient_training_data"
        out["error"] = f"entity '{entity_id}' has no rows in upload '{upload_id}'"
        ev.publish_training(job_id, entity_id, None, out, upload_id)
        return out

    # ---------------- resolve this entity's own configuration ----------------
    # Each entity resolves independently; no shared mutable state is involved,
    # so concurrent jobs cannot contaminate one another.
    try:
        snapshot = resolve_for_training(entity_id, job_id)
    except InvalidEntityConfig as exc:
        LOGGER.error("%s", exc)
        out.update({"status": "invalid_config", "error": str(exc),
                    "detail": {"problems": exc.problems}})
        ev.publish_training(job_id, entity_id, None, out, upload_id)
        return out

    persist_snapshot(snapshot)
    out["config_snapshot"] = snapshot.as_payload()
    (artifact_dir / "config_snapshot.json").write_text(
        as_json(snapshot.as_payload()), encoding="utf-8")

    # ---------------- train ----------------
    try:
        candidate = train_entity(entity_id, sub, entity_cfg, job_id, upload_id,
                                 checksum, strategy, snapshot=snapshot)
    except InsufficientTrainingData as exc:
        LOGGER.warning("%s: %s", entity_id, exc)
        out.update({"status": "insufficient_training_data", "error": str(exc),
                    "detail": exc.detail})
        reg.set_status(entity_id, reg.SHADOW, updated_by=f"job:{job_id}")
        ev.publish_training(job_id, entity_id, None, out, upload_id)
        return out
    except Exception as exc:
        LOGGER.exception("Training crashed for %s", entity_id)
        out.update({"status": FAILED, "error": str(exc),
                    "traceback": traceback.format_exc(limit=4)})
        ev.publish_training(job_id, entity_id, None, out, upload_id)
        return out

    persist_snapshot(snapshot, model_id=candidate.model_id)
    out.update({
        "model_id": candidate.model_id,
        "artifact_path": candidate.artifact_path,
        "metadata_path": candidate.metadata_path,
        "n_features": len(candidate.feature_cols),
        "selected_models": candidate.selected,
        "split_windows": candidate.split_windows,
        "frozen_thresholds": {"red_model": candidate.red_model,
                              "amber_model": candidate.amber_model},
        "config_hash": snapshot.config_hash,
        "effective_config": dict(snapshot.operational),
        "metrics": candidate.metrics,
        "warnings": list(candidate.warnings),
    })
    (artifact_dir / "training_metrics.json").write_text(
        as_json({"model_id": candidate.model_id, "entity_id": entity_id,
                 "metrics": candidate.metrics,
                 "split_windows": candidate.split_windows,
                 "warnings": candidate.warnings}),
        encoding="utf-8",
    )

    # ---------------- automatic calibration ----------------
    # Risk is P(min < RED). The calibrator is fitted against the RED frozen with
    # this model so the probability definition stays reproducible afterwards.
    red_cal = float(snapshot.frozen.calibration_red or candidate.red_model)
    calibration = cal.auto_calibrate(
        entity_id=entity_id, model_id=candidate.model_id, job_id=job_id,
        calibration_frame=candidate.frames["calibration"],
        pred_col="pred_y_min_6", target_col="y_min_6", red_threshold=red_cal,
        artifact_dir=artifact_dir / "calibration",
        config_hash=snapshot.config_hash,
    )
    out["calibration"] = {
        "calibration_id": calibration["calibration_id"],
        "status": calibration["status"], "scope": calibration["scope"],
        "metrics": calibration["metrics"], "warnings": calibration["warnings"],
    }
    out["warnings"] += calibration["warnings"]
    (artifact_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "calibration" / "metadata.json").write_text(
        as_json(calibration), encoding="utf-8")
    ev.publish_calibration(job_id, entity_id, candidate.model_id, calibration,
                           upload_id)

    calibrator = cal.load_calibrator_from_record(
        cal.get_calibration(calibration["calibration_id"])
    )
    if calibrator is None and calibration["scope"] == cal.SCOPE_GLOBAL:
        from registry import models as mreg
        try:
            calibrator = mreg.load_calibrator(calibration.get("fallback_calibration_id"))
        except Exception:  # pragma: no cover
            calibrator = None

    # ---------------- alert bands from the risk distribution ----------------
    # The operator should not have to invent risk_threshold. Cluster this
    # entity's own calibrated risk into three groups and take the band edges
    # from where the clusters separate. band_warning also gates replenishment,
    # so alerting and ordering share one measured operating point.
    from registry import models as mreg

    bands = {
        "band_id": None, "status": alb.INSUFFICIENT,
        "reason": "no calibrator available; risk is undefined without one",
        "bands": {"warning": float(C.DEFAULT_BAND_WARNING),
                  "critical": float(C.DEFAULT_BAND_CRITICAL)},
        "source": "system_default", "clusters": [], "diagnostics": {},
        "warnings": [],
    }
    if calibrator is not None:
        ca_frame = candidate.frames["calibration"]
        risk = calibrator.exceedance_probability(
            ca_frame["pred_y_min_6"].to_numpy(dtype=float), red_cal
        )
        bands = alb.auto_band(
            entity_id=entity_id, model_id=candidate.model_id, job_id=job_id,
            risk=risk, artifact_dir=artifact_dir / "alert_bands",
            config_hash=snapshot.config_hash,
        )
    mreg.save_alert_bands(candidate.model_id, bands)
    try:
        meta_path = Path(candidate.metadata_path)
        meta_doc = from_json(meta_path.read_text(encoding="utf-8")) or {}
        meta_doc["alert_bands"] = bands
        meta_path.write_text(as_json(meta_doc), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - metadata is a mirror
        LOGGER.warning("could not update metadata.json for %s: %s",
                       candidate.model_id, exc)
    out["alert_bands"] = {
        "band_id": bands.get("band_id"), "status": bands["status"],
        "reason": bands.get("reason"), "bands": bands["bands"],
        "source": bands["source"], "clusters": bands["clusters"],
        "diagnostics": bands.get("diagnostics", {}),
    }
    out["warnings"] += bands.get("warnings", [])
    if bands["status"] != alb.FITTED:
        out["warnings"].append(
            f"alert bands could not be clustered ({bands['status']}); serving the "
            f"system default {bands['bands']['warning']}/{bands['bands']['critical']}"
        )

    # ---------------- three mandatory backtests ----------------
    reports = bt.run_all(candidate, calibrator, calibration["calibration_id"],
                         entity_cfg, job_id, upload_id, artifact_dir,
                         config.get("run_backtests"), snapshot=snapshot,
                         band_warning=bands["bands"]["warning"])
    out["backtests"] = {
        t: {"report_id": r["report_id"], "status": r["status"],
            "reason": r.get("reason")}
        for t, r in reports.items()
    }
    for t, r in reports.items():
        ev.publish_backtest(job_id, entity_id, candidate.model_id, r, upload_id)

    # ---------------- candidate binding (never auto-promoted) ----------------
    promotable, blocking = _promotion_gates(calibration, reports)
    execute(
        """INSERT OR REPLACE INTO entity_model_bindings
           (entity_id, model_id, calibration_id, role, job_id, upload_id,
            promotable, blocking_reasons, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (entity_id, candidate.model_id, calibration["calibration_id"], "candidate",
         job_id, upload_id, int(promotable), as_json(blocking), utc_now()),
    )
    execute("UPDATE model_registry SET calibration_status=? WHERE model_id=?",
            (calibration["status"], candidate.model_id))

    out["status"] = CANDIDATE_READY
    out["promotable"] = promotable
    out["blocking_reasons"] = blocking
    out["promoted"] = False
    ev.publish_training(job_id, entity_id, candidate.model_id, out, upload_id)
    return out


def _promotion_gates(calibration: dict, reports: dict) -> tuple[bool, list[str]]:
    """A candidate is promotable only when calibration and all three reports pass."""
    blocking: list[str] = []
    if calibration["status"] != cal.FITTED:
        blocking.append(
            f"calibration status is '{calibration['status']}'; calibrated risk "
            f"scoring cannot be promoted"
        )
    for t in BACKTEST_TYPES:
        r = reports.get(t)
        if r is None:
            blocking.append(f"backtest '{t}' did not run")
        elif r["status"] != bt.COMPLETED:
            blocking.append(
                f"backtest '{t}' status is '{r['status']}'"
                + (f": {r['reason']}" if r.get("reason") else "")
            )
    return (not blocking), blocking
