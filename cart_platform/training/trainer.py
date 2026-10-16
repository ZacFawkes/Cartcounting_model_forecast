"""Per-entity training scoped to one upload.

Reuses the audited pipeline end to end: ``build_features`` / ``build_targets``
for the feature and target contract, ``select_features`` + ``fit_value_clipper``
for screening, and ``train_target_models`` + ``select_model`` for fitting and
operational model selection. Nothing about the trained feature definition is
re-implemented here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from settings import (
    MIN_HOLDOUT_RECORDS,
    MIN_TRAIN_DAYS,
    MIN_TRAIN_RECORDS,
    PER_ENTITY_MODEL_DIR,
    SYSTEM_DEFAULTS,
)

from feature_engineering import (  # noqa: E402
    apply_value_clipper,
    build_features,
    candidate_features,
    fit_value_clipper,
    select_features,
)
from forecasting import select_model, train_target_models  # noqa: E402
from registry.database import as_json, execute, utc_now  # noqa: E402
from target_builder import build_targets  # noqa: E402
from training.model_config import (  # noqa: E402
    ConfigSnapshot,
    InvalidEntityConfig,
    resolve_for_training,
)
from training.splits import CALIBRATION, HOLDOUT, TRAIN, assign_chronological_splits  # noqa: E402

LOGGER = logging.getLogger("platform.trainer")

TARGETS = ["y_cart_3", "y_cart_6", "y_min_3", "y_min_6"]
PLANNING_HORIZON = 6
# Hard feature floor: cart_lag_9 / cart_momentum_9 / cart_diff_volatility_9
# all reach back to t-9, so a usable row needs 10 consecutive records.
MIN_HISTORY_RECORDS = 10


class InsufficientTrainingData(ValueError):
    """Raised instead of manufacturing metrics for a thin stream."""

    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}


@dataclass
class TrainedCandidate:
    entity_id: str
    model_id: str
    artifact_path: str
    metadata_path: str
    feature_cols: list
    selected: dict
    metrics: dict
    split_windows: dict
    red_model: int
    amber_model: int
    cart_ceiling: float
    feature_schema_hash: str
    snapshot: "ConfigSnapshot | None" = None
    clip_bounds: pd.DataFrame | None = None
    estimators: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def feature_schema_hash(feature_cols: list, red_model: int, amber_model: int) -> str:
    payload = json.dumps(
        {"features": sorted(feature_cols), "red_model": int(red_model),
         "amber_model": int(amber_model)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def new_model_id(entity_id: str) -> str:
    return (
        f"mdl_{entity_id.replace(':', '_')}_"
        f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}"
    )


def feature_schema_hash_for(snapshot: ConfigSnapshot, feature_cols: list) -> str:
    """Schema identity includes the frozen thresholds, not only the columns."""
    return feature_schema_hash(feature_cols, snapshot.frozen.red_model,
                               snapshot.frozen.amber_model)


def preflight(frame: pd.DataFrame, entity_cfg: dict) -> list[str]:
    """Reasons this entity cannot be trained. Empty list == trainable."""
    reasons = []
    if len(frame) < MIN_TRAIN_RECORDS:
        reasons.append(
            f"{len(frame)} usable records; {MIN_TRAIN_RECORDS} required for a "
            f"trustworthy per-entity model"
        )
    n_days = frame["date"].nunique() if "date" in frame else 0
    if n_days < MIN_TRAIN_DAYS:
        reasons.append(f"{n_days} day(s) of history; {MIN_TRAIN_DAYS} required")
    if entity_cfg.get("bay_capacity") is None:
        reasons.append("bay_capacity is not configured (physical value, never invented)")
    return reasons


def train_entity(
    entity_id: str,
    prepared: pd.DataFrame,
    entity_cfg: dict,
    job_id: str,
    upload_id: str,
    dataset_checksum: str,
    strategy: str = "per_entity",
    snapshot: ConfigSnapshot | None = None,
) -> TrainedCandidate:
    """Train one entity from one upload. Never touches other entities.

    ``snapshot`` carries this entity's resolved configuration. It is passed
    explicitly rather than read from any global, so two concurrent jobs can
    never contaminate each other.
    """
    reasons = preflight(prepared, entity_cfg)
    if reasons:
        raise InsufficientTrainingData(
            "INSUFFICIENT_TRAINING_DATA: " + "; ".join(reasons),
            {"reasons": reasons, "n_records": int(len(prepared))},
        )

    if snapshot is None:
        snapshot = resolve_for_training(entity_id, job_id)
    if snapshot.entity_id != entity_id:
        raise InvalidEntityConfig(
            entity_id,
            [f"configuration snapshot belongs to '{snapshot.entity_id}'"],
        )

    red_model = snapshot.frozen.red_model
    amber_model = snapshot.frozen.amber_model
    warnings: list[str] = list(snapshot.warnings)

    # The entity's own thresholds shape the 17 threshold-derived features and
    # the shortage targets. Nothing here reads a module-level constant.
    featured = build_targets(
        build_features(prepared, windows=list(snapshot.frozen.feature_windows),
                       red_threshold=red_model, amber_threshold=amber_model),
        primary=snapshot.frozen.primary_horizon,
        planning=snapshot.frozen.planning_horizon,
        red=red_model,
    )
    plan = assign_chronological_splits(
        featured, purge_records=snapshot.frozen.planning_horizon
    )

    candidates = candidate_features(plan.frame)
    train_mask = (
        (plan.frame["split"] == TRAIN)
        & (~plan.frame["purged"])
        & plan.frame[TARGETS].notna().all(axis=1)
    )
    if int(train_mask.sum()) < MIN_HISTORY_RECORDS * 3:
        raise InsufficientTrainingData(
            f"INSUFFICIENT_TRAINING_DATA: only {int(train_mask.sum())} complete "
            f"training rows after feature warm-up and target alignment",
            {"n_training_rows": int(train_mask.sum())},
        )

    feature_cols, selection_report = select_features(plan.frame, candidates, train_mask)
    clip_bounds = fit_value_clipper(plan.frame, feature_cols, train_mask)
    clipped = apply_value_clipper(plan.frame, clip_bounds)

    usable = clipped[feature_cols].notna().all(axis=1) & clipped[TARGETS].notna().all(axis=1)
    tr = clipped[(clipped["split"] == TRAIN) & (~clipped["purged"]) & usable]
    ca = clipped[(clipped["split"] == CALIBRATION) & (~clipped["purged"]) & usable]
    ho_all = clipped[clipped["split"] == HOLDOUT]
    ho = ho_all[usable.reindex(ho_all.index, fill_value=False)]

    if len(tr) < MIN_HISTORY_RECORDS * 3:
        raise InsufficientTrainingData(
            f"INSUFFICIENT_TRAINING_DATA: {len(tr)} usable training rows",
            {"n_training_rows": int(len(tr))},
        )
    if len(ca) == 0:
        raise InsufficientTrainingData(
            "INSUFFICIENT_TRAINING_DATA: calibration partition is empty after "
            "purging; the stream is too short to calibrate without leakage",
            {"split_counts": plan.counts()},
        )
    if len(ho) < MIN_HOLDOUT_RECORDS:
        warnings.append(
            f"holdout has only {len(ho)} scoreable rows "
            f"({MIN_HOLDOUT_RECORDS} preferred); backtest metrics are noisy"
        )

    cart_ceiling = float(tr["cart_current"].max()) * 1.1

    # --- fit every target on TRAIN, select on CALIBRATION -------------------
    models: dict[str, dict] = {}
    selected: dict[str, str] = {}
    metrics: dict[str, dict] = {}
    Xtr, Xca = tr[feature_cols], ca[feature_cols]
    for target in TARGETS:
        ytr = tr[target].to_numpy(dtype=float)
        yca = ca[target].to_numpy(dtype=float)
        near = (yca < red_model + 10) | (ca["cart_current"].to_numpy() < red_model + 10)
        fitted, table = train_target_models(Xtr, ytr, Xca, yca, target, near)
        models[target] = fitted
        selected[target] = select_model(fitted, target)
        metrics[target] = {
            "selected": selected[target],
            "params": fitted[selected[target]].params,
            "validation": {k: _num(v) for k, v in
                           fitted[selected[target]].val_metrics.items()},
            "all_models": {name: {k: _num(v) for k, v in fm.val_metrics.items()}
                           for name, fm in fitted.items()},
        }

    estimators = {t: models[t][selected[t]].estimator for t in TARGETS}

    # --- out-of-training predictions for calibration and backtests ----------
    def _predict(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        out = frame.copy()
        X = out[feature_cols]
        for t in TARGETS:
            out[f"pred_{t}"] = np.clip(estimators[t].predict(X), 0.0, cart_ceiling)
        return out

    ca_pred = _predict(ca)
    ho_pred = _predict(ho)

    schema_hash = feature_schema_hash_for(snapshot, feature_cols)
    model_id = new_model_id(entity_id)
    model_dir = PER_ENTITY_MODEL_DIR / entity_id.replace(":", "_") / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "estimators": estimators,
        "selected": selected,
        "params": {t: models[t][selected[t]].params for t in TARGETS},
        "feature_cols": feature_cols,
        "clip_bounds": clip_bounds,
        "cart_ceiling": cart_ceiling,
        "red_model": red_model,
        "amber_model": amber_model,
        "entity_id": entity_id,
        "upload_id": upload_id,
        "job_id": job_id,
        "strategy": strategy,
        "dataset_checksum": dataset_checksum,
        "feature_schema_hash": schema_hash,
        "min_history_records": MIN_HISTORY_RECORDS,
        # Frozen model-dependent configuration travels with the artifact.
        "model_config": snapshot.frozen.as_dict(),
        "config_hash": snapshot.config_hash,
        "effective_config_at_training": dict(snapshot.operational),
        "trained_at": utc_now(),
    }
    artifact = model_dir / "bundle.pkl"
    with open(artifact, "wb") as fh:
        pickle.dump(bundle, fh)
    artifact.chmod(0o444)

    split_windows = plan.windows()
    metadata = {
        "model_id": model_id, "entity_id": entity_id, "upload_id": upload_id,
        "job_id": job_id, "strategy": strategy,
        "dataset_checksum": dataset_checksum,
        "feature_schema_hash": schema_hash, "n_features": len(feature_cols),
        "feature_cols": feature_cols, "selected_models": selected,
        "frozen_thresholds": {"red_model": red_model, "amber_model": amber_model},
        "model_dependent": snapshot.frozen.as_dict(),
        "effective_config": dict(snapshot.operational),
        "config_sources": dict(snapshot.sources),
        "config_hash": snapshot.config_hash,
        "cart_ceiling": cart_ceiling,
        "min_history_records": MIN_HISTORY_RECORDS,
        "split_counts": plan.counts(), "split_windows": split_windows,
        "metrics": metrics, "warnings": warnings,
        "trained_at": utc_now(),
    }
    meta_path = model_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    execute(
        """INSERT INTO model_registry
           (model_id, version, status, trained_at, artifact_path, feature_count,
            selected_models, entities_trained_on, sites_trained_on, train_window,
            red_model, amber_model, metrics, loso_metrics, notes, entity_id,
            upload_id, job_id, strategy, dataset_checksum, feature_schema_hash,
            calibration_status, cart_ceiling, min_history_records)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            model_id, _next_version(), "candidate", utc_now(), str(artifact),
            len(feature_cols), as_json(selected), as_json([entity_id]),
            as_json([entity_cfg.get("site_key")]),
            as_json(split_windows), float(red_model), float(amber_model),
            as_json(metrics),
            as_json({}), f"per-entity candidate from upload {upload_id}",
            entity_id, upload_id, job_id, strategy, dataset_checksum, schema_hash,
            "pending", cart_ceiling, MIN_HISTORY_RECORDS,
        ),
    )

    LOGGER.info("Trained %s -> %s (%d features, holdout %d rows)",
                entity_id, model_id, len(feature_cols), len(ho))
    return TrainedCandidate(
        entity_id=entity_id, model_id=model_id, artifact_path=str(artifact),
        metadata_path=str(meta_path), feature_cols=feature_cols, selected=selected,
        metrics=metrics, split_windows=split_windows, red_model=red_model,
        amber_model=amber_model, cart_ceiling=cart_ceiling,
        feature_schema_hash=schema_hash,
        snapshot=snapshot,
        clip_bounds=clip_bounds,
        estimators=estimators,
        frames={
            "clipped": clipped, "train": tr, "calibration": ca_pred,
            "holdout": ho_pred, "holdout_all": ho_all,
            "selection_report": selection_report,
        },
        warnings=warnings,
    )


def _num(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(float(v)) else round(float(v), 6)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def _next_version() -> int:
    from registry.database import query_one

    row = query_one("SELECT MAX(version) AS v FROM model_registry")
    return int((row or {}).get("v") or 0) + 1
