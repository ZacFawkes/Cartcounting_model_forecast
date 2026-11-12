"""Dataset loading, parsing and canonical ordering."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CART_COL,
    CSV_ENCODING,
    CSV_SEPARATOR,
    IN_COL,
    LONG_INTERVAL_MIN,
    MAX_SEQUENCE_GAP_MIN,
    NEXT_TIME_COL,
    OUT_COL,
    RAW_DATA_FILE,
    TIME_COL,
)

LOGGER = logging.getLogger(__name__)

# Physical bay watched by a camera stream. Optional in the pipeline's own
# datasets; required when entity identity includes the bay (see entity_id).
BAY_COL = "BayID"

REQUIRED_COLUMNS = [
    "CountID",
    TIME_COL,
    NEXT_TIME_COL,
    "KDVR",
    "ChanNo",
    "CatID",
    "AreaID",
    CART_COL,
    IN_COL,
    OUT_COL,
    "CNN_Index",
    "siteKey",
]


def _detect_separator(path: Path) -> str:
    with open(path, "r", encoding=CSV_ENCODING) as fh:
        header = fh.readline()
    for candidate in ("\t", ";", ",", "|"):
        if header.count(candidate) >= 5:
            return candidate
    return CSV_SEPARATOR


def load_raw(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Read the telemetry CSV with delimiter sniffing and type coercion."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    sep = _detect_separator(path)
    LOGGER.info("Reading %s (separator=%r)", path.name, sep)
    df = pd.read_csv(path, sep=sep, encoding=CSV_ENCODING)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    for col in (TIME_COL, NEXT_TIME_COL):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in (CART_COL, IN_COL, OUT_COL):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_bay_id(value) -> str:
    """Canonical form of a bay identifier.

    ``:`` separates the parts of an entity id and the id travels through URL
    paths, so every character outside ``[A-Za-z0-9_]`` collapses to ``_``.
    A GUID ``06053B44-0A47-42FC-BE3E-F88CB85F99D8`` therefore becomes
    ``06053B44_0A47_42FC_BE3E_F88CB85F99D8``.
    """
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        raise ValueError("bay id is empty")
    return re.sub(r"[^0-9A-Za-z_]", "_", text)


def normalize_bay_series(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.strip()
    # object dtype, never a pandas extension string dtype: the column travels
    # into feature engineering, which selects columns by numeric dtype.
    return cleaned.str.replace(r"[^0-9A-Za-z_]", "_", regex=True).astype(object)


def entity_id(
    df: pd.DataFrame,
    include_site: bool = False,
    include_bay: bool = False,
) -> pd.Series:
    """Modelling grain: one camera stream watching one cart bay.

    ``KDVR`` identifies the DVR device and ``ChanNo`` the channel on it, so the
    pair is what physically produces a telemetry stream. ``siteKey`` is kept as
    a grouping attribute (store-level pooling, roll-ups, leave-one-site-out),
    never as the stream identity.

    Set ``include_site`` when DVR ids are only unique within a store, producing
    a three-part key ``site:kdvr:chan``. Set ``include_bay`` when one channel
    can watch more than one physical bay, producing the four-part key
    ``site:kdvr:chan:bay``.
    """
    kdvr = df["KDVR"].astype("int64").astype(str)
    chan = df["ChanNo"].astype("int64").astype(str)
    parts = [kdvr, chan]
    if include_site:
        parts.insert(0, df["siteKey"].astype("int64").astype(str))
    if include_bay:
        if BAY_COL not in df.columns:
            raise ValueError(
                f"column '{BAY_COL}' is required when the entity id includes the bay"
            )
        bay = normalize_bay_series(df[BAY_COL])
        if (bay == "").any() or bay.isna().any():
            raise ValueError(f"column '{BAY_COL}' contains empty bay identifiers")
        parts.append(bay)
    out = parts[0]
    for part in parts[1:]:
        out = out + ":" + part
    return out


def prepare(
    df: pd.DataFrame,
    include_site_in_entity: bool = False,
    include_bay_in_entity: bool = False,
) -> pd.DataFrame:
    """Sort per entity, drop unusable rows and derive interval metadata.

    ``next_DVRDate`` is the earliest timestamp at which the traffic accumulated
    over ``[DVRDate, next_DVRDate)`` is known, so it is treated as the
    *availability* (decision) time of the record. ``Count`` is the cart
    availability measurement associated with the interval and is therefore also
    only usable from ``next_DVRDate`` onwards.

    Rolling windows, lags and targets must never cross either a time gap **or**
    an entity boundary, so ``block_id`` is keyed by entity and only then split
    on gaps. With a single entity this reduces exactly to the previous
    behaviour.
    """
    out = df.copy()
    n0 = len(out)

    out = out.dropna(subset=[TIME_COL, NEXT_TIME_COL, CART_COL])
    if BAY_COL in out.columns:
        out[BAY_COL] = normalize_bay_series(out[BAY_COL])
    out["entity_id"] = entity_id(out, include_site_in_entity, include_bay_in_entity)
    out = out.drop_duplicates(subset=["entity_id", TIME_COL], keep="first")
    out = out.sort_values(["entity_id", TIME_COL], kind="mergesort").reset_index(drop=True)
    n_entities = out["entity_id"].nunique()
    LOGGER.info(
        "Prepared %d rows across %d entity stream(s) (dropped %d)",
        len(out), n_entities, n0 - len(out),
    )

    out["interval_duration"] = (
        out[NEXT_TIME_COL] - out[TIME_COL]
    ).dt.total_seconds() / 60.0
    # Availability timestamp of the record = end of its interval.
    out["decision_time"] = out[NEXT_TIME_COL]

    g = out.groupby("entity_id", observed=True)
    gap = (g[TIME_COL].shift(-1) - out[NEXT_TIME_COL]).dt.total_seconds() / 60.0
    out["gap_to_next_min"] = gap
    out["sequence_break_after"] = (gap.isna()) | (gap > MAX_SEQUENCE_GAP_MIN)

    # Contiguous block id: never crosses a sequence break OR an entity boundary.
    break_before = (
        out.groupby("entity_id", observed=True)["sequence_break_after"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )
    seq = break_before.groupby(out["entity_id"], observed=True).cumsum().astype(int)
    out["block_seq"] = seq
    out["block_id"] = out["entity_id"] + "#" + seq.astype(str)

    out["long_interval"] = out["interval_duration"] > LONG_INTERVAL_MIN
    out["date"] = out[TIME_COL].dt.normalize()
    out["traffic_net_current"] = out[IN_COL] - out[OUT_COL]

    out["elapsed_from_start_min"] = (
        out[TIME_COL] - g[TIME_COL].transform("min")
    ).dt.total_seconds() / 60.0
    return out


def load_prepared(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    return prepare(load_raw(path))


def dataset_overview(df: pd.DataFrame) -> dict:
    """Headline descriptive statistics used in the report."""
    dur = df["interval_duration"]
    return {
        "n_records": int(len(df)),
        "start": df[TIME_COL].min(),
        "end": df[NEXT_TIME_COL].max(),
        "n_days": int(df["date"].nunique()),
        "cart_mean": float(df[CART_COL].mean()),
        "cart_median": float(df[CART_COL].median()),
        "cart_std": float(df[CART_COL].std()),
        "cart_min": int(df[CART_COL].min()),
        "cart_max": int(df[CART_COL].max()),
        "traffic_in_mean": float(df[IN_COL].mean()),
        "traffic_out_mean": float(df[OUT_COL].mean()),
        "traffic_in_total": int(df[IN_COL].sum()),
        "traffic_out_total": int(df[OUT_COL].sum()),
        "traffic_in_max": int(df[IN_COL].max()),
        "traffic_out_max": int(df[OUT_COL].max()),
        "interval_min": float(dur.min()),
        "interval_median": float(dur.median()),
        "interval_mean": float(dur.mean()),
        "interval_max": float(dur.max()),
        "n_long_intervals": int(df["long_interval"].sum()),
        "n_sequence_breaks": int(df["sequence_break_after"].iloc[:-1].sum()),
        "records_below_red": int((df[CART_COL] < 15).sum()),
        "records_below_amber": int((df[CART_COL] < 20).sum()),
        "site_keys": sorted(df["siteKey"].unique().tolist()),
        "entity_ids": sorted(df["entity_id"].unique().tolist()),
        "n_entities": int(df["entity_id"].nunique()),
        "records_per_day_mean": float(len(df) / max(df["date"].nunique(), 1)),
    }


def daily_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day record counts, coverage and shortage exposure."""
    grp = df.groupby("date", observed=True)
    out = pd.DataFrame(
        {
            "n_records": grp.size(),
            "first_record": grp[TIME_COL].min(),
            "last_record": grp[NEXT_TIME_COL].max(),
            "hours_covered": grp[TIME_COL].apply(lambda s: s.dt.hour.nunique()),
            "cart_min": grp[CART_COL].min(),
            "cart_mean": grp[CART_COL].mean(),
            "records_below_red": grp[CART_COL].apply(lambda s: int((s < 15).sum())),
            "mean_interval_min": grp["interval_duration"].mean(),
        }
    ).reset_index()
    out["coverage_ratio"] = out["n_records"] / out["n_records"].max()
    return out


def summarise_horizon_timing(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Realised elapsed minutes for each nominal record horizon."""
    rows = []
    for h in horizons:
        nominal = h * (20.0 / 3.0)
        elapsed = (
            df.groupby("block_id", observed=True)["decision_time"].shift(-h)
            - df["decision_time"]
        ).dt.total_seconds() / 60.0
        valid = elapsed.dropna()
        rows.append(
            {
                "horizon_records": h,
                "nominal_minutes": nominal,
                "median_minutes": float(np.median(valid)) if len(valid) else np.nan,
                "mean_minutes": float(valid.mean()) if len(valid) else np.nan,
                "p05_minutes": float(np.percentile(valid, 5)) if len(valid) else np.nan,
                "p95_minutes": float(np.percentile(valid, 95)) if len(valid) else np.nan,
                "max_minutes": float(valid.max()) if len(valid) else np.nan,
                "pct_within_50pct_of_nominal": float(
                    (np.abs(valid - nominal) / nominal <= 0.5).mean() * 100
                )
                if len(valid)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
