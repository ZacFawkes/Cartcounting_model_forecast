"""Dataset upload: immutable snapshot, validation, upload-scoped discovery.

Discovery is deliberately scoped to a single ``upload_id``. The platform never
scans the registry or an external database to decide what exists or what to
train — only the rows in the file that was just uploaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from settings import (
    ALLOWED_UPLOAD_SUFFIXES,
    INCLUDE_SITE_IN_ENTITY_ID,
    MAX_UPLOAD_BYTES,
    MIN_TRAIN_DAYS,
    MIN_TRAIN_RECORDS,
    REQUIRED_DATASET_COLUMNS,
    UPLOAD_DIR,
)

# settings extends sys.path to the modelling pipeline; import after it.
from data_loader import prepare  # noqa: E402
from registry.database import (  # noqa: E402
    as_json,
    execute,
    from_json,
    query,
    query_one,
    utc_now,
)

LOGGER = logging.getLogger("platform.uploads")

REGISTERED, REJECTED = "registered", "rejected"


class UploadRejected(ValueError):
    """Raised when a dataset cannot be registered at all."""

    def __init__(self, message: str, errors: list | None = None):
        super().__init__(message)
        self.errors = errors or [message]


@dataclass
class UploadResult:
    upload_id: str
    status: str
    entities: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    duplicate_of: str | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_fingerprint(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def score_url_for(kdvr: int, chan_no: int, site_key: int | None = None) -> str:
    if INCLUDE_SITE_IN_ENTITY_ID:
        return f"/v1/sites/{int(site_key)}/entities/{int(kdvr)}/{int(chan_no)}/score"
    return f"/v1/entities/{int(kdvr)}/{int(chan_no)}/score"


def _sniff_separator(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    for candidate in ("\t", ";", "|", ","):
        if header.count(candidate) >= 5:
            return candidate
    return ","


def read_dataset(path: Path, column_mapping: dict | None = None) -> pd.DataFrame:
    """Load an uploaded CSV/TSV and apply the caller's column mapping."""
    if path.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise UploadRejected(
            f"unsupported file type '{path.suffix}'; allowed: "
            f"{sorted(ALLOWED_UPLOAD_SUFFIXES)}"
        )
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"file is {path.stat().st_size} bytes, limit is {MAX_UPLOAD_BYTES}"
        )
    df = pd.read_csv(path, sep=_sniff_separator(path), encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    if column_mapping:
        # mapping is {canonical_name: source_column}
        rename = {src: canon for canon, src in column_mapping.items()
                  if src and src in df.columns and src != canon}
        if rename:
            df = df.rename(columns=rename)
    return df


# --------------------------------------------------------------------------
# Validation and quality
# --------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, require_site: bool) -> list[str]:
    errors = []
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing required column(s): {missing}")
    if require_site and "siteKey" not in df.columns:
        errors.append(
            "siteKey is required because entity identity is site-scoped "
            "(CART_ENTITY_INCLUDE_SITE=1)"
        )
    if not errors and len(df) == 0:
        errors.append("dataset contains no rows")
    return errors


def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    warnings: list[str] = []
    for c in ("DVRDate", "next_DVRDate"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in ("Count", "Count_IN", "Count_OUT", "KDVR", "ChanNo"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "siteKey" in out.columns:
        out["siteKey"] = pd.to_numeric(out["siteKey"], errors="coerce")
    else:
        out["siteKey"] = 0
        warnings.append("siteKey absent; defaulted to 0 (metadata only)")
    for c, v in (("AreaID", 0), ("CatID", 0)):
        if c not in out.columns:
            out[c] = v
    if "CountID" not in out.columns:
        out["CountID"] = range(len(out))
    if "CNN_Index" not in out.columns:
        out["CNN_Index"] = range(len(out))
    return out, warnings


def quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """Drop unusable rows and describe exactly what was removed and why."""
    warnings: list[str] = []
    n0 = len(df)
    issues: dict = {"n_rows_raw": n0}

    bad_ts = df["DVRDate"].isna() | df["next_DVRDate"].isna()
    bad_id = df["KDVR"].isna() | df["ChanNo"].isna()
    bad_count = df["Count"].isna()
    negative = (df[["Count", "Count_IN", "Count_OUT"]] < 0).any(axis=1)
    # An interval whose end is not after its start is incomplete / corrupt.
    incomplete = df["next_DVRDate"] <= df["DVRDate"]

    issues.update({
        "n_bad_timestamps": int(bad_ts.sum()),
        "n_missing_identity": int(bad_id.sum()),
        "n_missing_count": int(bad_count.sum()),
        "n_negative_values": int(negative.sum()),
        "n_incomplete_intervals": int(incomplete.sum()),
    })

    drop = bad_ts | bad_id | bad_count | negative | incomplete
    clean = df.loc[~drop].copy()

    dup_mask = clean.duplicated(subset=["KDVR", "ChanNo", "DVRDate"], keep="first")
    issues["n_duplicate_records"] = int(dup_mask.sum())
    clean = clean.loc[~dup_mask].copy()

    site_conflicts = (
        clean.groupby(["KDVR", "ChanNo"], observed=True)["siteKey"].nunique()
    )
    conflicting = site_conflicts[site_conflicts > 1]
    issues["n_site_mapping_conflicts"] = int(len(conflicting))
    if len(conflicting):
        warnings.append(
            f"{len(conflicting)} (KDVR,ChanNo) pair(s) map to more than one siteKey: "
            f"{[f'{k}:{c}' for k, c in conflicting.index.tolist()]}"
        )

    issues["n_rows_usable"] = int(len(clean))
    for key, label in (
        ("n_bad_timestamps", "unparseable timestamps"),
        ("n_missing_identity", "missing KDVR/ChanNo"),
        ("n_missing_count", "missing Count"),
        ("n_negative_values", "negative measurements"),
        ("n_incomplete_intervals", "incomplete intervals (end <= start)"),
        ("n_duplicate_records", "duplicate (KDVR,ChanNo,DVRDate) rows"),
    ):
        if issues[key]:
            warnings.append(f"dropped {issues[key]} row(s): {label}")

    if clean.empty:
        raise UploadRejected("no usable rows remain after validation", warnings)
    return clean, issues, warnings


# --------------------------------------------------------------------------
# Discovery — scoped to this upload only
# --------------------------------------------------------------------------
def discover(prepared: pd.DataFrame) -> pd.DataFrame:
    """Distinct (KDVR, ChanNo) streams present in THIS upload."""
    g = prepared.groupby("entity_id", observed=True)
    out = pd.DataFrame({
        "kdvr": g["KDVR"].first().astype(int),
        "chan_no": g["ChanNo"].first().astype(int),
        "site_key": g["siteKey"].first().astype(int),
        "n_records": g.size(),
        "n_days": g["date"].nunique(),
        "first_record": g["DVRDate"].min(),
        "last_record": g["next_DVRDate"].max(),
        "cart_min": g["Count"].min(),
        "cart_max": g["Count"].max(),
        "cart_median": g["Count"].median(),
        "median_interval_min": g["interval_duration"].median(),
        "n_blocks": g["block_id"].nunique(),
    }).reset_index()
    return out


def _shortage_events(counts: pd.Series, red: float) -> int:
    below = (counts.to_numpy(dtype=float) < red)
    return int(np.sum(below & ~np.r_[False, below[:-1]]))


def assess_trainability(row: pd.Series, entity_cfg: dict | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(row["n_records"]) < MIN_TRAIN_RECORDS:
        reasons.append(
            f"only {int(row['n_records'])} usable records; "
            f"{MIN_TRAIN_RECORDS} required"
        )
    if int(row["n_days"]) < MIN_TRAIN_DAYS:
        reasons.append(f"only {int(row['n_days'])} day(s) of history; "
                       f"{MIN_TRAIN_DAYS} required")
    if entity_cfg is not None and entity_cfg.get("bay_capacity") is None:
        reasons.append("bay_capacity is not configured (physical value, never invented)")
    return (not reasons), reasons


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def find_duplicate(checksum: str, fingerprint: str) -> dict | None:
    return query_one(
        "SELECT * FROM dataset_uploads WHERE file_checksum=? AND config_fingerprint=?",
        (checksum, fingerprint),
    )


def get_upload(upload_id: str) -> dict | None:
    return query_one("SELECT * FROM dataset_uploads WHERE upload_id=?", (upload_id,))


def get_upload_entities(upload_id: str) -> list[dict]:
    return query(
        "SELECT * FROM upload_entities WHERE upload_id=? ORDER BY kdvr, chan_no",
        (upload_id,),
    )


def list_uploads(limit: int = 50) -> list[dict]:
    return query("SELECT * FROM dataset_uploads ORDER BY created_at DESC LIMIT ?", (limit,))


def store_snapshot(src: Path, upload_id: str, original_name: str) -> Path:
    """Copy the uploaded file into the configured data directory, immutably."""
    dest_dir = UPLOAD_DIR / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(original_name).name or "dataset.csv"
    dest = dest_dir / safe_name
    shutil.copy2(src, dest)
    dest.chmod(0o444)
    return dest


def new_upload_id() -> str:
    return f"upl_{datetime.now(timezone.utc):%Y%m%d}_{uuid.uuid4().hex[:8]}"


def load_prepared_upload(upload_id: str) -> pd.DataFrame:
    """Re-read an upload snapshot and return the prepared, cleaned frame."""
    row = get_upload(upload_id)
    if row is None:
        raise KeyError(f"upload '{upload_id}' not found")
    cfg = from_json(row["config_json"], {}) or {}
    df = read_dataset(Path(row["stored_path"]), cfg.get("column_mapping"))
    df, _ = coerce_types(df)
    clean, _, _ = quality_report(df)
    return prepare(clean, include_site_in_entity=INCLUDE_SITE_IN_ENTITY_ID)


def prepared_entity_frame(upload_id: str, entity_id: str) -> pd.DataFrame:
    prepared = load_prepared_upload(upload_id)
    sub = prepared[prepared["entity_id"] == entity_id].copy()
    if sub.empty:
        raise KeyError(f"entity '{entity_id}' is not present in upload '{upload_id}'")
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------------
# Registration — the single entry point used by POST /v1/datasets/upload
# --------------------------------------------------------------------------
def api_entity_rows(upload_id: str) -> list[dict]:
    """Upload entities in the exact shape the upload endpoint returns.

    Used for both a fresh registration and an idempotent replay so the contract
    never changes shape between the two.
    """
    out = []
    for r in get_upload_entities(upload_id):
        out.append({
            "entity_id": r["entity_id"], "kdvr": r["kdvr"], "chan_no": r["chan_no"],
            "site_key": r["site_key"], "records": r["n_records"], "days": r["n_days"],
            "shortage_events": r["shortage_events"], "score_url": r["score_url"],
            "trainable": bool(r["trainable"]),
            "blocking_reasons": from_json(r.get("blocking_reasons"), []) or [],
            "model_status": _model_status(r["entity_id"]),
            "entity_status": (_entity_status(r["entity_id"])),
        })
    return out


def _entity_status(entity_id: str) -> str | None:
    row = query_one("SELECT status FROM entity_config WHERE entity_id=?", (entity_id,))
    return None if row is None else row["status"]


def register_upload(
    source_path: Path,
    original_filename: str,
    config: dict,
    uploaded_by: str = "api",
    allow_duplicate: bool = False,
) -> UploadResult:
    """Validate a dataset, snapshot it and register the entities it contains.

    Only the ``(KDVR, ChanNo)`` pairs present in *this file* are discovered and
    registered. The registry is never scanned to decide what exists.
    """
    from registry import entities as reg  # local import avoids a cycle

    checksum = file_checksum(source_path)
    fingerprint = config_fingerprint(config)
    existing = find_duplicate(checksum, fingerprint)
    if existing and not allow_duplicate:
        LOGGER.info("Upload %s is a replay of %s", original_filename,
                    existing["upload_id"])
        return UploadResult(
            upload_id=existing["upload_id"],
            status=existing["status"],
            entities=api_entity_rows(existing["upload_id"]),
            warnings=from_json(existing["warnings_json"], []) or [],
            quality=from_json(existing["quality_json"], {}) or {},
            duplicate_of=existing["upload_id"],
        )

    raw = read_dataset(source_path, config.get("column_mapping"))
    errors = validate_schema(raw, require_site=INCLUDE_SITE_IN_ENTITY_ID)
    if errors:
        raise UploadRejected("dataset schema validation failed", errors)

    typed, warnings = coerce_types(raw)
    clean, quality, qwarn = quality_report(typed)
    warnings += qwarn

    prepared = prepare(clean, include_site_in_entity=INCLUDE_SITE_IN_ENTITY_ID)
    found = discover(prepared)

    upload_id = new_upload_id()
    stored = store_snapshot(source_path, upload_id, original_filename)

    defaults = config.get("entity_defaults") or {}
    overrides = config.get("entity_overrides") or {}

    execute(
        """INSERT INTO dataset_uploads
           (upload_id, name, original_filename, stored_path, file_checksum,
            config_fingerprint, config_json, status, n_rows, n_rows_usable,
            n_entities, timezone, quality_json, warnings_json, errors_json,
            uploaded_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            upload_id, config.get("name"), original_filename, str(stored), checksum,
            fingerprint, as_json(config), REGISTERED, int(quality["n_rows_raw"]),
            int(quality["n_rows_usable"]), int(len(found)), config.get("timezone"),
            as_json(quality), as_json(warnings), as_json([]), uploaded_by, utc_now(),
        ),
    )

    entities_out = []
    for _, r in found.iterrows():
        eid = r["entity_id"]
        site_key = int(r["site_key"])
        if reg.get_site(site_key) is None:
            # The entity FK requires the site row to exist first.
            reg.upsert_site(site_key, updated_by=f"upload:{upload_id}",
                            display_name=f"Site {site_key}")
        merged_cfg = {**defaults, **(overrides.get(eid) or {})}

        existing_entity = reg.try_get_entity(eid)
        fields = {
            "kdvr": int(r["kdvr"]), "chan_no": int(r["chan_no"]), "site_key": site_key,
            "display_name": f"Site {site_key} / DVR {int(r['kdvr'])} ch{int(r['chan_no'])}",
            "n_days_history": int(r["n_days"]),
            "last_upload_id": upload_id,
        }
        for key in ("bay_capacity", "red_threshold", "amber_threshold", "target_stock",
                    "cooldown_min",
                    "lead_time_min", "min_carts_per_trip", "max_carts_per_trip",
                    "quantity_rule", "reserve_per_day", "max_sequence_gap_min",
                    "consumption_model", "usage", "retention_halflife_min"):
            if merged_cfg.get(key) is not None:
                fields[key] = merged_cfg[key]
        if any(k in fields for k in ("red_threshold", "amber_threshold", "target_stock")):
            fields["threshold_source"] = "business"
        if existing_entity is None:
            # New stream: registered but never auto-activated.
            fields["status"] = reg.SHADOW
        reg.upsert_entity(eid, updated_by=f"upload:{upload_id}", **fields)
        entity_cfg = reg.get_entity(eid)

        red = entity_cfg.get("red_threshold")
        counts = prepared.loc[prepared["entity_id"] == eid, "Count"]
        n_events = _shortage_events(counts, red) if red is not None else None

        trainable, reasons = assess_trainability(r, entity_cfg)
        equality = {
            "median_interval_min": float(r["median_interval_min"]),
            "n_blocks": int(r["n_blocks"]),
            "cart_min": float(r["cart_min"]),
            "cart_max": float(r["cart_max"]),
            "has_physical_config": entity_cfg.get("bay_capacity") is not None,
        }
        url = score_url_for(int(r["kdvr"]), int(r["chan_no"]), site_key)
        execute(
            """INSERT INTO upload_entities
               (upload_id, entity_id, kdvr, chan_no, site_key, n_records, n_usable,
                n_days, first_record, last_record, cart_min, cart_max, cart_median,
                shortage_events, trainable, blocking_reasons, quality_json,
                score_url, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                upload_id, eid, int(r["kdvr"]), int(r["chan_no"]), site_key,
                int(r["n_records"]), int(r["n_records"]), int(r["n_days"]),
                str(r["first_record"]), str(r["last_record"]), float(r["cart_min"]),
                float(r["cart_max"]), float(r["cart_median"]),
                n_events, int(trainable), as_json(reasons), as_json(equality),
                url, utc_now(),
            ),
        )
        entities_out.append({
            "entity_id": eid, "kdvr": int(r["kdvr"]), "chan_no": int(r["chan_no"]),
            "site_key": site_key, "records": int(r["n_records"]),
            "days": int(r["n_days"]), "shortage_events": n_events,
            "score_url": url, "trainable": bool(trainable),
            "blocking_reasons": reasons,
            "model_status": _model_status(eid),
            "entity_status": entity_cfg["status"],
        })
        if not trainable:
            warnings.append(f"{eid}: not trainable — {'; '.join(reasons)}")

    LOGGER.info("Upload %s registered %d entity stream(s) from %s",
                upload_id, len(entities_out), original_filename)
    return UploadResult(upload_id=upload_id, status=REGISTERED, entities=entities_out,
                        warnings=warnings, quality=quality)


def _model_status(entity_id: str) -> str:
    row = query_one(
        "SELECT role FROM entity_model_bindings WHERE entity_id=? AND role='active'",
        (entity_id,),
    )
    if row:
        return "ready"
    cand = query_one(
        "SELECT role FROM entity_model_bindings WHERE entity_id=? AND role='candidate'",
        (entity_id,),
    )
    return "candidate_not_promoted" if cand else "not_ready"
