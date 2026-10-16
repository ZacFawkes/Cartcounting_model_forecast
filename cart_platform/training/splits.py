"""Chronological, purged splits for upload-scoped per-entity training.

Three disjoint, time-ordered partitions:

    TRAIN         model fitting and feature screening
    CALIBRATION   conformal residuals and risk calibration (out-of-training)
    HOLDOUT       the three backtests only — never touched before that

The tail of TRAIN and CALIBRATION is purged by the planning horizon so a target
window can never reach across a boundary. This mirrors the audited pipeline's
``assign_splits`` but uses fractions of records rather than whole days, because
an uploaded stream may be shorter than the multi-week dataset that function
assumes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from settings import CALIBRATION_FRACTION, TRAIN_FRACTION

LOGGER = logging.getLogger("platform.training")

TRAIN, CALIBRATION, HOLDOUT = "train", "calibration", "holdout"


@dataclass
class SplitPlan:
    frame: pd.DataFrame
    purge_records: int
    train_fraction: float
    calibration_fraction: float

    def counts(self) -> dict:
        c = self.frame["split"].value_counts().to_dict()
        return {
            "train": int(c.get(TRAIN, 0)),
            "calibration": int(c.get(CALIBRATION, 0)),
            "holdout": int(c.get(HOLDOUT, 0)),
            "purged": int(self.frame["purged"].sum()),
        }

    def windows(self) -> dict:
        out = {}
        for name in (TRAIN, CALIBRATION, HOLDOUT):
            sub = self.frame[(self.frame["split"] == name) & (~self.frame["purged"])]
            out[name] = {
                "start": str(sub["decision_time"].min()) if len(sub) else None,
                "end": str(sub["decision_time"].max()) if len(sub) else None,
                "n": int(len(sub)),
            }
        return out

    def part(self, name: str, usable_mask: pd.Series | None = None) -> pd.DataFrame:
        sel = (self.frame["split"] == name) & (~self.frame["purged"])
        if usable_mask is not None:
            sel &= usable_mask
        return self.frame.loc[sel].copy()


def assign_chronological_splits(
    frame: pd.DataFrame,
    purge_records: int,
    train_fraction: float = TRAIN_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> SplitPlan:
    """Split one entity's prepared+featured frame by time, then purge boundaries."""
    out = frame.sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    n = len(out)
    n_train = int(n * train_fraction)
    n_calib = int(n * calibration_fraction)
    # Guarantee a non-empty holdout even on short streams.
    n_train = min(n_train, max(1, n - 2))
    n_calib = min(n_calib, max(0, n - n_train - 1))

    split = pd.Series([HOLDOUT] * n, index=out.index, dtype=object)
    split.iloc[:n_train] = TRAIN
    split.iloc[n_train:n_train + n_calib] = CALIBRATION
    out["split"] = split

    out["purged"] = False
    for name in (TRAIN, CALIBRATION):
        idx = out.index[out["split"] == name]
        if len(idx) > purge_records:
            out.loc[idx[-purge_records:], "purged"] = True
        elif len(idx):
            # Too short to purge safely: drop the whole partition tail.
            out.loc[idx, "purged"] = True

    plan = SplitPlan(out, purge_records, train_fraction, calibration_fraction)
    LOGGER.info("Split plan: %s", plan.counts())
    return plan
