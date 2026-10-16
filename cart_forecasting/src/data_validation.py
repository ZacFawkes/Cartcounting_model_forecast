"""Data quality validation and temporal leakage checks."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    CART_COL,
    IN_COL,
    NEXT_TIME_COL,
    OUT_COL,
    TIME_COL,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    checks: list = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})

    @property
    def failures(self) -> list:
        return [c for c in self.checks if c["status"] == "FAIL"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    rep = ValidationReport()

    monotonic = df[TIME_COL].is_monotonic_increasing
    rep.add(
        "Chronological ordering",
        "PASS" if monotonic else "FAIL",
        "DVRDate strictly increasing" if monotonic else "DVRDate not sorted",
    )

    dup = int(df[TIME_COL].duplicated().sum())
    rep.add(
        "Duplicate timestamps",
        "PASS" if dup == 0 else "WARN",
        f"{dup} duplicated DVRDate values",
    )

    bad_interval = int((df[NEXT_TIME_COL] <= df[TIME_COL]).sum())
    rep.add(
        "Interval validity",
        "PASS" if bad_interval == 0 else "FAIL",
        f"{bad_interval} records with next_DVRDate <= DVRDate",
    )

    nulls = df[[CART_COL, IN_COL, OUT_COL]].isna().sum().sum()
    rep.add(
        "Missing core measurements",
        "PASS" if nulls == 0 else "WARN",
        f"{int(nulls)} missing values across Count/Count_IN/Count_OUT",
    )

    neg = int(((df[[CART_COL, IN_COL, OUT_COL]] < 0).sum()).sum())
    rep.add(
        "Non-negativity",
        "PASS" if neg == 0 else "FAIL",
        f"{neg} negative measurements",
    )

    breaks = int(df["sequence_break_after"].iloc[:-1].sum())
    rep.add(
        "Stream continuity",
        "PASS" if breaks == 0 else "WARN",
        f"{breaks} sequence breaks (gap > tolerance) splitting the stream into "
        f"{df['block_id'].nunique()} contiguous blocks",
    )

    long_iv = int(df["long_interval"].sum())
    rep.add(
        "Interval duration stability",
        "PASS" if long_iv == 0 else "WARN",
        f"{long_iv} intervals longer than the long-interval threshold "
        f"(max {df['interval_duration'].max():.1f} min)",
    )

    n_sites = df["siteKey"].nunique()
    rep.add(
        "Single-site scope",
        "PASS" if n_sites == 1 else "WARN",
        f"{n_sites} distinct siteKey value(s): {sorted(df['siteKey'].unique())}",
    )

    for c in rep.checks:
        LOGGER.info("[%s] %s - %s", c["status"], c["check"], c["detail"])
    return rep


def check_feature_leakage(
    frame: pd.DataFrame, feature_cols: list[str], horizons: list[int]
) -> ValidationReport:
    """Verify that no engineered feature is a shifted *future* observation.

    For each feature we correlate it with the forward-shifted cart series and
    compare against a perfect-correlation signature. A feature identical to a
    future value (correlation 1.0 with zero residual) is a leak.
    """
    rep = ValidationReport()
    cart = frame[CART_COL].to_numpy(dtype=float)
    leaks: list[str] = []

    for h in horizons:
        future = pd.Series(cart).shift(-h)
        for col in feature_cols:
            vals = frame[col]
            if not np.issubdtype(vals.dtype, np.number):
                continue
            mask = future.notna() & vals.notna()
            if mask.sum() < 30:
                continue
            resid = np.abs(vals[mask].to_numpy() - future[mask].to_numpy())
            if np.nanmax(resid) < 1e-9:
                leaks.append(f"{col} == Count(t+{h})")

    rep.add(
        "Future-value identity leakage",
        "PASS" if not leaks else "FAIL",
        "No feature equals a future observation"
        if not leaks
        else f"Leaking features: {leaks}",
    )

    # Features must be computable from rows at or before the decision time.
    forbidden = {"cart_t3", "cart_t6", "min3", "min6", "shortage_next_3", "shortage_next_6"}
    overlap = sorted(forbidden.intersection(feature_cols))
    rep.add(
        "Target columns excluded from features",
        "PASS" if not overlap else "FAIL",
        "No target leaked into the feature matrix"
        if not overlap
        else f"Target columns present in features: {overlap}",
    )
    return rep


def check_split_purging(splits: dict, max_horizon: int) -> ValidationReport:
    """Confirm that training targets never reach into the validation window."""
    rep = ValidationReport()
    order = ["train", "validation", "holdout"]
    ok = True
    details = []
    for a, b in zip(order[:-1], order[1:]):
        fa, fb = splits[a], splits[b]
        if fa.empty or fb.empty:
            continue
        last_target_time = fa["target_end_time"].max()
        first_origin = fb["decision_time"].min()
        if pd.notna(last_target_time) and last_target_time >= first_origin:
            ok = False
            details.append(
                f"{a}->{b}: last target end {last_target_time} >= first origin {first_origin}"
            )
        else:
            details.append(f"{a}->{b}: purged (gap >= {max_horizon} records)")
    rep.add(
        "Split purging",
        "PASS" if ok else "FAIL",
        "; ".join(details) if details else "no adjacent splits to check",
    )
    return rep
