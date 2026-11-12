"""Entity and site registry: CRUD, bootstrap from telemetry, threshold derivation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from settings import (
    BAY_CAPACITY_MARGIN,
    INCLUDE_BAY_IN_ENTITY_ID,
    INCLUDE_SITE_IN_ENTITY_ID,
    MIN_DAYS_FOR_ACTIVE,
    THRESHOLD_PERCENTILES,
)
from data_loader import BAY_COL, normalize_bay_id
from registry.database import execute, query, query_df, query_one, upsert, utc_now

LOGGER = logging.getLogger("platform.registry")

ACTIVE, SHADOW, DISABLED = "active", "shadow", "disabled"


class UnknownEntity(KeyError):
    """Raised instead of silently scoring with default thresholds."""


def entity_display_name(site_key: int, kdvr: int, chan_no: int, bay_id=None) -> str:
    name = f"Site {site_key} / DVR {kdvr} ch{chan_no}"
    return f"{name} / bay {bay_id}" if bay_id else name


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
    """Summarise every (KDVR, ChanNo[, bay]) stream present in a prepared frame."""
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
    if BAY_COL in prepared.columns:
        out["bay_id"] = out["entity_id"].map(
            g[BAY_COL].first().map(normalize_bay_id).to_dict()
        )
    else:
        out["bay_id"] = None
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
            "bay_id": r.get("bay_id"),
            "area_id": int(r["area_id"]),
            "display_name": entity_display_name(site_key, int(r["kdvr"]), int(r["chan_no"]),
                                          r.get("bay_id")),
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
        SELECT e.entity_id, e.site_key, e.kdvr, e.chan_no, e.bay_id, e.status,
               e.red_threshold, e.amber_threshold, e.target_stock, e.bay_capacity,
               e.threshold_source, e.n_days_history, e.model_id, e.calibration_id,
               s.lead_time_min AS site_lead_time_min, s.reserve_per_day AS site_reserve
        FROM entity_config e
        LEFT JOIN site_config s ON s.site_key = e.site_key
        ORDER BY e.site_key, e.kdvr, e.chan_no
        """
    )


def entity_id_for(
    kdvr: int,
    chan_no: int,
    site_key: int | None = None,
    bay_id: str | None = None,
) -> str:
    parts = [str(int(kdvr)), str(int(chan_no))]
    if INCLUDE_SITE_IN_ENTITY_ID:
        if site_key is None:
            raise ValueError("site_key is required when entity ids include the site")
        parts.insert(0, str(int(site_key)))
    if INCLUDE_BAY_IN_ENTITY_ID and bay_id is not None and str(bay_id).strip():
        # A dataset that predates bay identity produces bay-less ids; they stay
        # addressable instead of becoming unreachable.
        parts.append(normalize_bay_id(bay_id))
    return ":".join(parts)


def entity_ids_for_channel(site_key, kdvr: int, chan_no: int) -> list[str]:
    """Registered entities watching one channel — one per bay."""
    sql = "SELECT entity_id FROM entity_config WHERE kdvr=? AND chan_no=?"
    params: list = [int(kdvr), int(chan_no)]
    if site_key is not None:
        sql += " AND site_key=?"
        params.append(int(site_key))
    return [r["entity_id"] for r in query(sql + " ORDER BY entity_id", tuple(params))]


def split_entity_id(entity_id: str) -> dict:
    """Decompose an entity id into its parts, whatever the configured shape.

    Returns ``{site_key, kdvr, chan_no, bay_id}`` with ``None`` for parts the
    id does not carry. ``kdvr``/``chan_no`` are always the numeric pair, so a
    consumer never has to know how the platform is configured.
    """
    parts = str(entity_id).split(":")
    site_key = kdvr = chan_no = bay_id = None
    numeric = [i for i, p in enumerate(parts) if p.lstrip("-").isdigit()]
    if len(parts) >= 4:
        site_key, kdvr, chan_no, bay_id = parts[0], parts[1], parts[2], ":".join(parts[3:])
    elif len(parts) == 3:
        # "site:kdvr:chan" or "kdvr:chan:bay"
        if len(numeric) == 3:
            site_key, kdvr, chan_no = parts
        else:
            kdvr, chan_no, bay_id = parts
    elif len(parts) == 2:
        kdvr, chan_no = parts
    else:
        return {"site_key": None, "kdvr": None, "chan_no": None, "bay_id": None}
    try:
        kdvr, chan_no = int(kdvr), int(chan_no)
    except (TypeError, ValueError):
        return {"site_key": None, "kdvr": None, "chan_no": None, "bay_id": bay_id}
    return {
        "site_key": int(site_key) if site_key is not None else None,
        "kdvr": kdvr,
        "chan_no": chan_no,
        "bay_id": bay_id,
    }
