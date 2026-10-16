"""Versioned model registry and per-entity conformal calibrators.

Replaces the single overwritten pickle, which had no version and no provenance.
A model records the thresholds frozen into its feature vector (``red_model`` /
``amber_model``) so the resolver can detect a request that would invalidate it.
"""
from __future__ import annotations

import logging
import pickle
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from settings import CONFORMAL_ALPHA, MODEL_DIR, SOURCE_MODEL_BUNDLE, SYSTEM_DEFAULTS
from forecasting import ConformalCalibrator
from registry.database import as_json, execute, from_json, query, query_one, utc_now

LOGGER = logging.getLogger("platform.models")

CHAMPION, CANDIDATE, RETIRED = "champion", "candidate", "retired"
TARGETS = ["y_cart_3", "y_cart_6", "y_min_3", "y_min_6"]


class NoModelAvailable(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def register_model(
    artifact_path: Path,
    red_model: int,
    amber_model: int,
    entities: list[str] | None = None,
    sites: list[int] | None = None,
    metrics: dict | None = None,
    loso_metrics: dict | None = None,
    train_window: str | None = None,
    status: str = CANDIDATE,
    notes: str | None = None,
) -> str:
    with open(artifact_path, "rb") as fh:
        payload = pickle.load(fh)
    version = (query_one("SELECT MAX(version) AS v FROM model_registry") or {}).get("v") or 0
    version += 1
    model_id = f"mdl-{datetime.now(timezone.utc):%Y%m%d}-v{version}-{uuid.uuid4().hex[:6]}"
    execute(
        """INSERT INTO model_registry
           (model_id, version, status, trained_at, artifact_path, feature_count,
            selected_models, entities_trained_on, sites_trained_on, train_window,
            red_model, amber_model, metrics, loso_metrics, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            model_id, version, status, utc_now(), str(artifact_path),
            len(payload["feature_cols"]), as_json(payload.get("selected", {})),
            as_json(entities or []), as_json(sites or []), train_window,
            int(red_model), int(amber_model), as_json(metrics or {}),
            as_json(loso_metrics or {}), notes,
        ),
    )
    LOGGER.info("Registered model %s (version %d, status %s)", model_id, version, status)
    return model_id


def promote(model_id: str) -> None:
    execute("UPDATE model_registry SET status=? WHERE status=?", (RETIRED, CHAMPION))
    execute("UPDATE model_registry SET status=? WHERE model_id=?", (CHAMPION, model_id))
    LOGGER.info("Model %s promoted to champion", model_id)


def get_model_meta(model_id: str | None = None) -> dict:
    if model_id:
        row = query_one("SELECT * FROM model_registry WHERE model_id=?", (model_id,))
        if row is None:
            raise NoModelAvailable(f"model '{model_id}' is not registered")
        return row
    row = query_one("SELECT * FROM model_registry WHERE status=? ORDER BY version DESC",
                    (CHAMPION,))
    if row is None:
        raise NoModelAvailable("no champion model registered; run bootstrap first")
    return row


def list_models() -> list[dict]:
    return query("SELECT * FROM model_registry ORDER BY version DESC")


# --------------------------------------------------------------------------
# Alert bands
# --------------------------------------------------------------------------
def save_alert_bands(model_id: str, record: dict) -> None:
    """Freeze the clustered alert bands onto the model row."""
    execute(
        "UPDATE model_registry SET alert_band_id=?, alert_band_status=?, "
        "alert_band_json=? WHERE model_id=?",
        (record.get("band_id"), record.get("status"), as_json(record), model_id),
    )
    LOGGER.info("Model %s alert bands %s: %s", model_id, record.get("status"),
                record.get("bands") or "{}")



# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------
def register_calibrator(
    model_id: str,
    residuals: np.ndarray,
    entity_id: str | None = None,
    alpha: float = CONFORMAL_ALPHA,
    window: tuple | None = None,
    coverage: float | None = None,
    brier: float | None = None,
    pr_auc: float | None = None,
) -> str:
    scope = "entity" if entity_id else "global"
    cal_id = f"cal-{scope}-{entity_id or 'all'}-{uuid.uuid4().hex[:6]}".replace(":", "_")
    path = MODEL_DIR / f"{cal_id}.npy"
    np.save(path, np.asarray(residuals, dtype=float))
    calib = ConformalCalibrator(alpha=alpha)
    calib.residuals_ = np.asarray(residuals, dtype=float)
    execute(
        """INSERT INTO calibrator_registry
           (calibration_id, model_id, entity_id, scope, fitted_at, window_start,
            window_end, n_residuals, alpha, lower_offset, empirical_coverage,
            brier, pr_auc, status, artifact_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            cal_id, model_id, entity_id, scope, utc_now(),
            str(window[0]) if window else None, str(window[1]) if window else None,
            len(residuals), alpha, calib.lower_offset, coverage, brier, pr_auc,
            "active", str(path),
        ),
    )
    LOGGER.info("Registered %s calibrator %s (%d residuals)", scope, cal_id, len(residuals))
    return cal_id


def resolve_calibrator(model_id: str, entity_id: str) -> dict:
    """Entity-specific calibrator if one exists, otherwise the global fallback."""
    row = query_one(
        "SELECT * FROM calibrator_registry WHERE model_id=? AND entity_id=? "
        "AND status='active' ORDER BY fitted_at DESC",
        (model_id, entity_id),
    )
    if row:
        return row
    row = query_one(
        "SELECT * FROM calibrator_registry WHERE model_id=? AND scope='global' "
        "AND status='active' ORDER BY fitted_at DESC",
        (model_id,),
    )
    if row is None:
        raise NoModelAvailable(f"no calibrator available for model '{model_id}'")
    return row


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class LoadedModel:
    model_id: str
    meta: dict
    estimators: dict
    selected: dict
    feature_cols: list
    clip_bounds: pd.DataFrame
    cart_ceiling: float
    red_model: float
    amber_model: float
    model_config: dict = None
    config_hash: str | None = None
    feature_windows: list | None = None

    def predict_row(self, X: pd.DataFrame) -> dict:
        out = {}
        for target, key in (
            ("y_min_6", "pred_min_6"), ("y_min_3", "pred_min_3"),
            ("y_cart_3", "pred_cart_3"), ("y_cart_6", "pred_cart_6"),
        ):
            out[key] = float(
                np.clip(self.estimators[target].predict(X)[0], 0.0, self.cart_ceiling)
            )
        return out

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.feature_cols]
        out = frame.copy()
        for t in TARGETS:
            out[f"pred_{t}"] = np.clip(self.estimators[t].predict(X), 0.0, self.cart_ceiling)
        return out


_MODEL_CACHE: dict[str, LoadedModel] = {}
_CALIB_CACHE: dict[str, ConformalCalibrator] = {}


def load_model(model_id: str | None = None) -> LoadedModel:
    meta = get_model_meta(model_id)
    mid = meta["model_id"]
    if mid in _MODEL_CACHE:
        return _MODEL_CACHE[mid]
    with open(meta["artifact_path"], "rb") as fh:
        payload = pickle.load(fh)
    # Frozen model-dependent config: prefer the bundle (written by the
    # per-entity trainer), fall back to the registry row, then to the legacy
    # defaults so pre-existing artifacts stay loadable.
    from training.model_config import FrozenModelConfig

    frozen = FrozenModelConfig.from_mapping(
        payload.get("model_config"),
        fallback_red=float(meta["red_model"]),
        fallback_amber=float(meta["amber_model"]),
    )
    lm = LoadedModel(
        model_id=mid,
        meta=meta,
        estimators=payload["estimators"],
        selected=payload.get("selected", {}),
        feature_cols=payload["feature_cols"],
        clip_bounds=payload["clip_bounds"],
        cart_ceiling=float(payload.get("cart_ceiling", 64.0)),
        red_model=frozen.red_model,
        amber_model=frozen.amber_model,
        model_config=frozen.as_dict(),
        config_hash=payload.get("config_hash") or meta.get("feature_schema_hash"),
        feature_windows=list(frozen.feature_windows),
    )
    lm.frozen = frozen
    _MODEL_CACHE[mid] = lm
    return lm


def load_calibrator(calibration_id: str) -> ConformalCalibrator:
    if calibration_id in _CALIB_CACHE:
        return _CALIB_CACHE[calibration_id]
    row = query_one("SELECT * FROM calibrator_registry WHERE calibration_id=?",
                    (calibration_id,))
    if row is None:
        raise NoModelAvailable(f"calibrator '{calibration_id}' not found")
    calib = ConformalCalibrator(alpha=row["alpha"])
    calib.residuals_ = np.load(row["artifact_path"])
    _CALIB_CACHE[calibration_id] = calib
    return calib


def clear_cache() -> None:
    _MODEL_CACHE.clear()
    _CALIB_CACHE.clear()


# --------------------------------------------------------------------------
# Bootstrap: adopt the validated single-site bundle as model v1
# --------------------------------------------------------------------------
def bootstrap_from_pipeline(source: Path = SOURCE_MODEL_BUNDLE,
                            entities: list[str] | None = None,
                            sites: list[int] | None = None) -> tuple[str, str]:
    """Import ``cart_forecasting/outputs/models`` as the first champion."""
    if not Path(source).exists():
        raise NoModelAvailable(
            f"source bundle not found at {source}; run cart_forecasting/main.py first"
        )
    with open(source, "rb") as fh:
        payload = pickle.load(fh)

    dest = MODEL_DIR / f"bundle-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.pkl"
    shutil.copy2(source, dest)

    model_id = register_model(
        artifact_path=dest,
        red_model=SYSTEM_DEFAULTS["red_threshold"],
        amber_model=SYSTEM_DEFAULTS["amber_threshold"],
        entities=entities or [],
        sites=sites or [],
        status=CANDIDATE,
        notes="Imported from the validated single-site pipeline (audit-approved).",
    )
    promote(model_id)
    cal_id = register_calibrator(
        model_id=model_id,
        residuals=np.asarray(payload["conformal_residuals"], dtype=float),
        entity_id=None,
        alpha=float(payload.get("conformal_alpha", CONFORMAL_ALPHA)),
    )
    return model_id, cal_id
