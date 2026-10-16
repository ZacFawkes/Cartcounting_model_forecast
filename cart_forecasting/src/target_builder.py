"""Forecast target construction and chronological split assignment."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    CART_COL,
    HORIZON_DEVIATION_TOLERANCE,
    MIN_RECORDS_FOR_COMPLETE_DAY,
    NOMINAL_RECORD_MINUTES,
    PLANNING_HORIZON,
    PRIMARY_HORIZON,
    RED_THRESHOLD,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)

LOGGER = logging.getLogger(__name__)

TARGET_COLS = ["y_cart_3", "y_cart_6", "y_min_3", "y_min_6"]
SHORTAGE_COLS = ["shortage_next_3", "shortage_next_6"]


def build_targets(
    df: pd.DataFrame,
    primary: int = PRIMARY_HORIZON,
    planning: int = PLANNING_HORIZON,
    red: int = RED_THRESHOLD,
) -> pd.DataFrame:
    """Add forecast targets, shortage labels and horizon reliability flags.

    Targets are aligned to the *decision time* of record ``t``:
    ``y_cart_h = Count(t+h)`` and ``y_min_h = min(Count(t+1) .. Count(t+h))``.
    Shifts never cross a sequence break.
    """
    out = df.copy()
    g = out["block_id"]
    cart = out[CART_COL].astype(float)

    fwd = {}
    for k in range(1, planning + 1):
        fwd[k] = cart.groupby(g, observed=True).shift(-k)

    out["y_cart_3"] = fwd[primary]
    out["y_cart_6"] = fwd[planning]
    out["y_min_3"] = pd.concat([fwd[k] for k in range(1, primary + 1)], axis=1).min(axis=1)
    out["y_min_6"] = pd.concat([fwd[k] for k in range(1, planning + 1)], axis=1).min(axis=1)
    out["y_max_6"] = pd.concat([fwd[k] for k in range(1, planning + 1)], axis=1).max(axis=1)

    out["shortage_next_3"] = (out["y_min_3"] < red).astype("float")
    out["shortage_next_6"] = (out["y_min_6"] < red).astype("float")
    out.loc[out["y_min_3"].isna(), "shortage_next_3"] = np.nan
    out.loc[out["y_min_6"].isna(), "shortage_next_6"] = np.nan

    # Shortage state decomposition at the current record.
    below = (cart < red).astype(int)
    prev_below = below.groupby(g, observed=True).shift(1).fillna(0).astype(int)
    out["shortage_active"] = below
    out["shortage_onset"] = ((below == 1) & (prev_below == 0)).astype(int)
    out["shortage_recovery"] = ((below == 0) & (prev_below == 1)).astype(int)

    # New-onset label: a shortage that begins within the horizon while the
    # current record is *not* already in shortage.
    for h, name in ((primary, "onset_next_3"), (planning, "onset_next_6")):
        fut_min = out[f"y_min_{h}"]
        out[name] = ((fut_min < red) & (below == 0)).astype("float")
        out.loc[fut_min.isna(), name] = np.nan

    # Realised horizon duration and reliability.
    dec = out["decision_time"]
    for h in (primary, planning):
        end = dec.groupby(g, observed=True).shift(-h)
        elapsed = (end - dec).dt.total_seconds() / 60.0
        nominal = h * NOMINAL_RECORD_MINUTES
        out[f"horizon_minutes_{h}"] = elapsed
        out[f"horizon_deviation_{h}"] = (elapsed - nominal) / nominal
        out[f"horizon_reliable_{h}"] = (
            out[f"horizon_deviation_{h}"].abs() <= HORIZON_DEVIATION_TOLERANCE
        ).fillna(False)
        out[f"target_end_time_{h}"] = end

    out["target_end_time"] = out[f"target_end_time_{planning}"]
    out["horizon_reliable"] = out[f"horizon_reliable_{primary}"] & out[
        f"horizon_reliable_{planning}"
    ]
    return out


def assign_splits(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    purge_records: int = PLANNING_HORIZON,
) -> tuple[pd.DataFrame, dict]:
    """Chronological day-based split with purging at the boundaries.

    The last ``purge_records`` origins of each split are removed so that no
    training/validation target window overlaps the next split.
    """
    out = df.copy()
    day_counts = out.groupby("date", observed=True).size()
    complete_days = day_counts[day_counts >= MIN_RECORDS_FOR_COMPLETE_DAY].index
    partial_days = day_counts[day_counts < MIN_RECORDS_FOR_COMPLETE_DAY].index
    days = sorted(complete_days)
    n = len(days)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * validation_fraction)))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    train_days = days[:n_train]
    val_days = days[n_train : n_train + n_val]
    hold_days = days[n_train + n_val :] + sorted(partial_days)

    def _label(d):
        if d in set(train_days):
            return "train"
        if d in set(val_days):
            return "validation"
        return "holdout"

    out["split"] = out["date"].map(_label)

    # Purge the tail of each split so target windows do not cross boundaries.
    out["purged"] = False
    for name in ("train", "validation"):
        idx = out.index[out["split"] == name]
        if len(idx) > purge_records:
            out.loc[idx[-purge_records:], "purged"] = True

    meta = {
        "train_days": [str(pd.Timestamp(d).date()) for d in train_days],
        "validation_days": [str(pd.Timestamp(d).date()) for d in val_days],
        "holdout_days": [str(pd.Timestamp(d).date()) for d in hold_days],
        "partial_days": [str(pd.Timestamp(d).date()) for d in partial_days],
        "purge_records": purge_records,
        "n_train": int((out["split"] == "train").sum()),
        "n_validation": int((out["split"] == "validation").sum()),
        "n_holdout": int((out["split"] == "holdout").sum()),
    }
    LOGGER.info("Split sizes: %s", {k: meta[k] for k in ("n_train", "n_validation", "n_holdout")})
    return out, meta


def modelling_mask(
    frame: pd.DataFrame, feature_cols: list[str], target_cols: list[str] | None = None
) -> pd.Series:
    """Rows usable for supervised learning/evaluation."""
    target_cols = target_cols or TARGET_COLS
    mask = frame[target_cols].notna().all(axis=1)
    mask &= frame[feature_cols].notna().all(axis=1)
    mask &= ~frame["purged"].fillna(False)
    mask &= frame["horizon_reliable"].fillna(False)
    return mask


def shortage_events(
    frame: pd.DataFrame, red: int = RED_THRESHOLD, recovery_records: int = 2
) -> pd.DataFrame:
    """Group consecutive sub-RED observations into independent events.

    An event ends only after ``recovery_records`` consecutive observations at or
    above RED (explicit recovery rule), which avoids splitting a single physical
    shortage into many events because of measurement flicker.
    """
    sub = frame.reset_index(drop=True)
    below = (sub[CART_COL] < red).to_numpy()
    times = sub["decision_time"].to_numpy()
    start_times = sub["DVRDate"].to_numpy()
    carts = sub[CART_COL].to_numpy()

    events = []
    i = 0
    n = len(sub)
    while i < n:
        if not below[i]:
            i += 1
            continue
        start = i
        j = i
        recovered_run = 0
        last_below = i
        while j < n:
            if below[j]:
                recovered_run = 0
                last_below = j
            else:
                recovered_run += 1
                if recovered_run >= recovery_records:
                    break
            j += 1
        end = last_below
        rec_idx = min(end + 1, n - 1)
        events.append(
            {
                "event_id": len(events) + 1,
                "onset_time": pd.Timestamp(start_times[start]),
                "onset_decision_time": pd.Timestamp(times[start]),
                "recovery_time": pd.Timestamp(times[rec_idx]),
                "n_records": int(end - start + 1),
                "min_cart": float(np.min(carts[start : end + 1])),
                "mean_cart": float(np.mean(carts[start : end + 1])),
                "severity_cart_records": float(
                    np.sum(np.clip(red - carts[start : end + 1], 0, None))
                ),
                "duration_min": float(
                    (pd.Timestamp(times[end]) - pd.Timestamp(start_times[start])).total_seconds()
                    / 60.0
                ),
                "start_index": int(start),
                "end_index": int(end),
            }
        )
        i = j + 1
    return pd.DataFrame(events)
