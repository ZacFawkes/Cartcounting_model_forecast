"""Precheck RED calibration eligibility without training a model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from settings import (
    MIN_CALIBRATION_POSITIVES,
    MIN_CALIBRATION_SAMPLES,
)

from feature_engineering import (
    apply_value_clipper,
    build_features,
    candidate_features,
    fit_value_clipper,
    select_features,
)
from target_builder import build_targets
from training.splits import (
    TRAIN,
    CALIBRATION,
    assign_chronological_splits,
)
from training.trainer import (
    TARGETS,
    MIN_HISTORY_RECORDS,
    preflight,
)


def _calibration_for_red(
    prepared: pd.DataFrame,
    frozen,
    red: int,
) -> tuple[pd.DataFrame, dict]:
    """Reproduce trainer.py preparation, splitting and usable-row filtering."""

    featured = build_targets(
        build_features(
            prepared,
            windows=list(frozen.feature_windows),
            red_threshold=red,
            amber_threshold=frozen.amber_model,
        ),
        primary=frozen.primary_horizon,
        planning=frozen.planning_horizon,
        red=red,
    )

    plan = assign_chronological_splits(
        featured,
        purge_records=frozen.planning_horizon,
    )

    candidates = candidate_features(plan.frame)

    train_mask = (
        (plan.frame["split"] == TRAIN)
        & (~plan.frame["purged"])
        & plan.frame[TARGETS].notna().all(axis=1)
    )

    n_training_rows = int(train_mask.sum())

    if n_training_rows < MIN_HISTORY_RECORDS * 3:
        raise ValueError(
            "Insufficient complete training rows after target alignment: "
            f"{n_training_rows}"
        )

    feature_cols, _ = select_features(
        plan.frame,
        candidates,
        train_mask,
    )

    clip_bounds = fit_value_clipper(
        plan.frame,
        feature_cols,
        train_mask,
    )

    clipped = apply_value_clipper(
        plan.frame,
        clip_bounds,
    )

    usable = (
        clipped[feature_cols].notna().all(axis=1)
        & clipped[TARGETS].notna().all(axis=1)
    )

    calibration = clipped[
        (clipped["split"] == CALIBRATION)
        & (~clipped["purged"])
        & usable
    ].copy()

    return calibration, {
        "split_counts": plan.counts(),
        "split_windows": plan.windows(),
        "n_training_rows": n_training_rows,
        "n_selected_features": len(feature_cols),
    }


def find_valid_red_threshold(
    prepared: pd.DataFrame,
    entity_cfg: dict,
    frozen,
    initial_red: int,
    max_red: int,
) -> dict:
    """Find the first RED meeting auto_calibrate's data eligibility gates."""

    initial_red = int(initial_red)
    max_red = int(max_red)

    if initial_red < 0:
        raise ValueError("red_threshold must be >= 0")

    if max_red < initial_red:
        raise ValueError(
            "Initial RED exceeds the configured search ceiling"
        )

    reasons = preflight(prepared, entity_cfg)

    if reasons:
        return {
            "status": "FAIL",
            "reason": "INSUFFICIENT_TRAINING_DATA",
            "blocking_reasons": reasons,
            "initial_red": initial_red,
            "recommended_red": None,
        }

    attempts = []

    for red in range(initial_red, max_red + 1):
        try:
            calibration, split_info = _calibration_for_red(
                prepared=prepared,
                frozen=frozen,
                red=red,
            )

        except (ValueError, KeyError) as exc:
            # Cannot verify this candidate reliably.
            # Do not silently convert a processing failure into PASS.
            return {
                "status": "FAIL",
                "reason": "PRECHECK_PROCESSING_FAILED",
                "initial_red": initial_red,
                "recommended_red": None,
                "failed_red": red,
                "detail": str(exc),
                "attempts": attempts,
            }

        y = pd.to_numeric(
            calibration["y_min_6"],
            errors="coerce",
        ).to_numpy(dtype=float)

        y = y[np.isfinite(y)]

        n = len(y)
        n_pos = int((y < red).sum())
        n_neg = n - n_pos

        passed = (
            n >= MIN_CALIBRATION_SAMPLES
            and n_pos >= MIN_CALIBRATION_POSITIVES
            and n_neg > 0
        )

        attempts.append({
            "red": red,
            "n_samples": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "eligible": passed,
        })

        if passed:
            return {
                "status": "PASS",
                "check_type": "calibration_data_eligibility",
                "initial_red": initial_red,
                "recommended_red": red,
                "adjusted": red != initial_red,
                "required_samples": MIN_CALIBRATION_SAMPLES,
                "required_positives": MIN_CALIBRATION_POSITIVES,
                "calibration": attempts[-1],
                "split_info": split_info,
                "attempts": attempts,
                "warning": (
                    "Data eligibility passed. Model fitting, prediction "
                    "validity and promotion gates have not been tested."
                ),
            }

    return {
        "status": "FAIL",
        "reason": "NO_ELIGIBLE_RED_FOUND",
        "initial_red": initial_red,
        "recommended_red": None,
        "required_samples": MIN_CALIBRATION_SAMPLES,
        "required_positives": MIN_CALIBRATION_POSITIVES,
        "attempts": attempts,
    }