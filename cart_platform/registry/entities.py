"""Entity and site registry: CRUD, bootstrap from telemetry, threshold derivation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from settings import (
    BAY_CAPACITY_MARGIN,
    INCLUDE_SITE_IN_ENTITY_ID,
    MIN_DAYS_FOR_ACTIVE,
    THRESHOLD_PERCENTILES,
)
from registry.database import execute, query, query_df, query_one, upsert, utc_now

LOGGER = logging.getLogger("platform.registry")

ACTIVE, SHADOW, DISABLED = "active", "shadow", "disabled"


class UnknownEntity(KeyError):
    """Raised instead of silently scoring with default thresholds."""


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------
def get_entity(entity_id: str) -> dict:
    row = query_one("SELECT * FROM entity_config WHERE entity_id = ?", (entity_id,))
    if row is None:
        raise UnknownEntity(
            f"entity '{entity_id}' is not registered. Register it before scoring; "
            f"scoring with default thresholds would be meaningless."
        )
    return row


def try_get_entity(entity_id: str) -> dict | None:
    return query_one("SELECT * FROM entity_config WHERE entity_id = ?", (entity_id,))


def get_site(site_key: int) -> dict | None:
    return query_one("SELECT * FROM site_config WHERE site_key = ?", (site_key,))


def list_entities(site_key: int | None = None, status: str | None = None) -> list[dict]:
    sql = "SELECT * FROM entity_config WHERE 1=1"
    params: list = []
    if site_key is not None:
        sql += " AND site_key = ?"
        params.append(site_key)
    if status:
        sql += " AND status = ?"
        params.append(status)
    return query(sql + " ORDER BY site_key, kdvr, chan_no", tuple(params))


def list_sites() -> list[dict]:
    return query("SELECT * FROM site_config ORDER BY site_key")


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------
def upsert_site(site_key: int, updated_by: str = "api", **fields) -> dict:
    row = {"site_key": int(site_key), "updated_at": utc_now(), "updated_by": updated_by}
    row.update({k: v for k, v in fields.items() if v is not None})
    upsert("site_config", row, ["site_key"])
    return get_site(site_key)


def upsert_entity(entity_id: str, updated_by: str = "api", **fields) -> dict:
    existing = try_get_entity(entity_id)
    row = {"updated_at": utc_now(), "updated_by": updated_by}
    row.update({k: v for k, v in fields.items() if v is not None})

    if existing is None:
        for required in ("kdvr", "chan_no", "site_key"):
            if required not in row:
                raise ValueError(f"'{required}' is required when creating an entity")
        row["entity_id"] = entity_id
        row.setdefault("status", SHADOW)
        upsert("entity_config", row, ["entity_id"])
        return get_entity(entity_id)

    # Partial update: a PUT that names only a few columns must not have to
    # restate NOT NULL identity columns.
    sets = ", ".join(f"{c}=?" for c in row)
    execute(
        f"UPDATE entity_config SET {sets} WHERE entity_id=?",
        tuple(row.values()) + (entity_id,),
    )
    return get_entity(entity_id)


def set_status(entity_id: str, status: str, updated_by: str = "api") -> dict:
    if status not in (ACTIVE, SHADOW, DISABLED):
        raise ValueError(f"invalid status '{status}'")
    execute(
        "UPDATE entity_config SET status=?, updated_at=?, updated_by=? WHERE entity_id=?",
        (status, utc_now(), updated_by, entity_id),
    )
    return get_entity(entity_id)


def bind_model(entity_id: str, model_id: str, calibration_id: str | None = None) -> None:
    execute(
        "UPDATE entity_config SET model_id=?, calibration_id=?, updated_at=? "
        "WHERE entity_id=?",
        (model_id, calibration_id, utc_now(), entity_id),
    )


# --------------------------------------------------------------------------
# Threshold derivation (decision D3)
# --------------------------------------------------------------------------
def derive_thresholds(counts: pd.Series) -> dict:
    """Percentile-derived thresholds used only when no business value exists.

    These encode *current* service levels as the target, which may simply
    entrench today's behaviour. ``threshold_source='derived'`` records that, so
    no report presents them as a business rule.
    """
    c = pd.Series(counts).dropna().astype(float)
    if len(c) < 30:
        raise ValueError(f"need at least 30 observations to derive thresholds, got {len(c)}")
    red = max(5, int(round(np.percentile(c, THRESHOLD_PERCENTILES["red"]))))
    amber = int(round(np.percentile(c, THRESHOLD_PERCENTILES["amber"])))
    target = int(round(np.percentile(c, THRESHOLD_PERCENTILES["target"])))
    amber = max(amber, red + 1)
    target = max(target, amber + 1)
    capacity = int(np.ceil(c.max() * BAY_CAPACITY_MARGIN))
    return {
        "red_threshold": red,
        "amber_threshold": amber,
        "target_stock": target,
        "bay_capacity": capacity,
        "threshold_source": "derived",
    }


# --------------------------------------------------------------------------
# Bootstrap from telemetry
# --------------------------------------------------------------------------
def discover_entities(prepared: pd.DataFrame) -> pd.DataFrame:
    """Summarise every (KDVR, ChanNo) stream present in a prepared frame."""
    g = prepared.groupby("entity_id", observed=True)
    out = pd.DataFrame(
        {
            "kdvr": g["KDVR"].first().astype(int),
            "chan_no": g["ChanNo"].first().astype(int),
            "site_key": g["siteKey"].first().astype(int),
            "area_id": g["AreaID"].first().astype(int),
            "n_records": g.size(),
            "n_days": g["date"].nunique(),
            "first_record": g["DVRDate"].min(),
            "last_record": g["next_DVRDate"].max(),
            "cart_min": g["Count"].min(),
            "cart_max": g["Count"].max(),
            "cart_median": g["Count"].median(),
        }
    ).reset_index()
    return out


def register_from_telemetry(
    prepared: pd.DataFrame,
    updated_by: str = "bootstrap",
    overwrite_thresholds: bool = False,
) -> pd.DataFrame:
    """Create/refresh registry rows for every stream found in the telemetry.

    Thresholds are derived only when the entity has none, or when
    ``overwrite_thresholds`` is set. An entity with fewer than
    ``MIN_DAYS_FOR_ACTIVE`` days of history stays in shadow (decision D7).
    """
    summary = discover_entities(prepared)
    rows = []
    for _, r in summary.iterrows():
        site_key = int(r["site_key"])
        if get_site(site_key) is None:
            upsert_site(site_key, updated_by=updated_by,
                        display_name=f"Site {site_key}")

        existing = try_get_entity(r["entity_id"])
        fields = {
            "kdvr": int(r["kdvr"]),
            "chan_no": int(r["chan_no"]),
            "site_key": site_key,
            "area_id": int(r["area_id"]),
            "display_name": f"Site {site_key} / DVR {int(r['kdvr'])} ch{int(r['chan_no'])}",
            "n_days_history": int(r["n_days"]),
        }
        needs_thresholds = (
            existing is None
            or existing.get("red_threshold") is None
            or overwrite_thresholds
        )
        if needs_thresholds:
            counts = prepared.loc[prepared["entity_id"] == r["entity_id"], "Count"]
            try:
                fields.update(derive_thresholds(counts))
            except ValueError as exc:
                LOGGER.warning("%s: %s", r["entity_id"], exc)

        if existing is None:
            fields["status"] = (
                ACTIVE if int(r["n_days"]) >= MIN_DAYS_FOR_ACTIVE else SHADOW
            )
        upsert_entity(r["entity_id"], updated_by=updated_by, **fields)
        rows.append({**fields, "entity_id": r["entity_id"]})

    LOGGER.info(
        "Registered %d entity stream(s) across %d site(s)",
        len(rows), summary["site_key"].nunique(),
    )
    return pd.DataFrame(rows)


def entity_summary() -> pd.DataFrame:
    return query_df(
        """
        SELECT e.entity_id, e.site_key, e.kdvr, e.chan_no, e.status,
               e.red_threshold, e.amber_threshold, e.target_stock, e.bay_capacity,
               e.threshold_source, e.n_days_history, e.model_id, e.calibration_id,
               s.lead_time_min AS site_lead_time_min, s.reserve_per_day AS site_reserve
        FROM entity_config e
        LEFT JOIN site_config s ON s.site_key = e.site_key
        ORDER BY e.site_key, e.kdvr, e.chan_no
        """
    )


def entity_id_for(kdvr: int, chan_no: int, site_key: int | None = None) -> str:
    if INCLUDE_SITE_IN_ENTITY_ID:
        if site_key is None:
            raise ValueError("site_key is required when entity ids include the site")
        return f"{int(site_key)}:{int(kdvr)}:{int(chan_no)}"
    return f"{int(kdvr)}:{int(chan_no)}"
