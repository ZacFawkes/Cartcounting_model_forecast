"""
One parameterised experiment:
Upload -> Train -> Validate Config -> Promote -> Score -> CSV.

Each experiment:
    1. Uploads telemetry with CLI operational configuration.
    2. Rejects duplicate uploads to avoid silently reusing old configs.
    3. Trains a candidate model.
    4. Validates effective configuration against requested parameters.
    5. Promotes only when validation succeeds.
    6. Optionally clears previous scoring decisions.
    7. Scores all observations.
    8. Exports scoring_results + six API response fields.
    9. Writes an experiment configuration JSON sidecar.

IMPORTANT:
    This script modifies the platform SQLite database directly.
    Use only in a local development/test environment.

    Training responses must expose effective_config for validation.
    If the server only stores this in model metadata, adapt the
    metadata retrieval in validate_model_config().
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

import pandas as pd
import requests

import settings as C
from client.platform_client import load_telemetry, to_records


DEFAULT_DATA = PLATFORM_ROOT / "data" / "real_data.csv"
HISTORY_WINDOW = 20

EXTRA_RESPONSE_FIELDS = (
    "urgency",
    "new_predictive_alert",
    "reactive_critical",
    "expected_completion_time",
    "actionable_before_breach",
    "minutes_to_predicted_breach",
)

BOOLEAN_RESPONSE_FIELDS = (
    "new_predictive_alert",
    "reactive_critical",
    "actionable_before_breach",
)


# ============================================================
# UTILITIES
# ============================================================

def _log(message: str) -> None:
    print(
        f"[{datetime.now():%H:%M:%S}] {message}",
        flush=True,
    )


def _parts(entity_id: str) -> tuple[str, str, str]:

    parts = entity_id.split(":")

    if len(parts) != 3:
        raise SystemExit(
            'Entity must use "siteKey:KDVR:ChanNo"'
        )

    return tuple(parts)


def _entity_config(a) -> dict:

    cfg = {
        "bay_capacity": a.bay_capacity,
        "red_threshold": a.red_threshold,
        "amber_threshold": a.amber_threshold,
        "target_stock": a.target_stock,
        "quantity_rule": a.quantity_rule,
        "min_carts_per_trip": a.min_carts_per_trip,
        "max_carts_per_trip": a.max_carts_per_trip,
        "reserve_per_day": a.reserve_per_day,
    }

    return {
        k: v
        for k, v in cfg.items()
        if v is not None
    }


def _entity_defaults(a) -> dict:

    cfg = {
        "max_sequence_gap_min": a.max_sequence_gap_min,
        "cooldown_min": a.cooldown_min,
        "lead_time_min": a.lead_time_min,
    }

    return {
        k: v
        for k, v in cfg.items()
        if v is not None
    }


def _request_params(a) -> dict:

    cfg = {
        "risk_threshold": a.override_risk_threshold,
        "critical_risk_threshold":
            a.override_critical_risk_threshold,
    }

    return {
        k: v
        for k, v in cfg.items()
        if v is not None
    }


def _experiment_name(a) -> str:

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    unique_id = uuid4().hex[:8]

    return (
        f"{a.out.stem}"
        f"_red{a.red_threshold}"
        f"_amber{a.amber_threshold}"
        f"_target{a.target_stock}"
        f"_{timestamp}_{unique_id}"
    )


def _extract_response_fields(result: dict) -> dict:

    action = result.get("action") or {}

    return {
        "urgency": action.get("urgency"),

        "new_predictive_alert": result.get(
            "new_predictive_alert"
        ),

        "reactive_critical": result.get(
            "reactive_critical"
        ),

        "expected_completion_time": action.get(
            "expected_completion_time"
        ),

        "actionable_before_breach": action.get(
            "actionable_before_breach"
        ),

        "minutes_to_predicted_breach": action.get(
            "minutes_to_predicted_breach"
        ),
    }


def _response_summary(frame: pd.DataFrame) -> dict:

    summary = {}

    for field in EXTRA_RESPONSE_FIELDS:

        if field not in frame.columns:

            summary[field] = {
                "matched": 0,
                "missing": len(frame),
            }

            continue

        series = frame[field]

        field_summary = {
            "matched": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
        }

        if field == "urgency":

            field_summary["distribution"] = (
                series
                .fillna("MISSING")
                .value_counts()
                .to_dict()
            )

        elif field in BOOLEAN_RESPONSE_FIELDS:

            field_summary["true"] = int(
                series.eq(True).sum()
            )

            field_summary["false"] = int(
                series.eq(False).sum()
            )

        elif field == "minutes_to_predicted_breach":

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            field_summary["min"] = (
                float(numeric.min())
                if numeric.notna().any()
                else None
            )

            field_summary["median"] = (
                float(numeric.median())
                if numeric.notna().any()
                else None
            )

            field_summary["max"] = (
                float(numeric.max())
                if numeric.notna().any()
                else None
            )

        summary[field] = field_summary

    return summary


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

def upload(
    base: str,
    data: Path,
    entity: str,
    a,
) -> tuple[str, dict]:

    cfg = {
        "name": a.name,
        "timezone": a.timezone,

        "entity_defaults": _entity_defaults(a),

        "entity_overrides": {
            entity: _entity_config(a)
        },

        "auto_start_training": False,
    }

    _log("=" * 65)
    _log("UPLOAD CONFIGURATION")
    _log("=" * 65)

    print(
        json.dumps(
            cfg,
            indent=2,
        )
    )

    with open(data, "rb") as fh:

        response = requests.post(
            f"{base}/v1/datasets/upload",

            files={
                "file": (
                    data.name,
                    fh,
                    "text/csv",
                )
            },

            data={
                "config": json.dumps(cfg)
            },

            timeout=600,
        )

    response.raise_for_status()

    body = response.json()

    _log("UPLOAD API RESPONSE")

    print(
        json.dumps(
            body,
            indent=2,
        )
    )

    upload_id = body["upload_id"]

    duplicate_of = body.get("duplicate_of")

    if duplicate_of:

        raise RuntimeError(
            "DUPLICATE UPLOAD DETECTED!\n"
            f"upload_id={upload_id}\n"
            f"duplicate_of={duplicate_of}\n"
            "Training stopped to avoid reusing "
            "potentially stale configuration."
        )

    entities = {
        item["entity_id"]: item
        for item in body.get("entities", [])
    }

    if entity not in entities:

        raise RuntimeError(
            f"Entity {entity} not found in upload."
        )

    _log(
        f"Upload registered successfully: {upload_id}"
    )

    return upload_id, cfg


# ============================================================
# STEP 2 — TRAIN
# ============================================================

def train(
    base: str,
    upload_id: str,
    entity: str,
    timeout_s: int,
) -> tuple[dict, str]:

    payload = {
        "upload_id": upload_id,
        "entity_ids": [entity],
        "strategy": "per_entity",
    }

    _log("TRAINING REQUEST")

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    response = requests.post(
        f"{base}/v1/training/jobs",
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    body = response.json()

    job_id = body["job_id"]

    _log(f"Training job: {job_id}")

    deadline = time.time() + timeout_s

    while time.time() < deadline:

        time.sleep(5)

        response = requests.get(
            f"{base}/v1/training/jobs/{job_id}",
            timeout=60,
        )

        response.raise_for_status()

        body = response.json()

        status = body.get("status")

        if status in (
            "succeeded",
            "completed",
            "failed",
        ):
            break

    else:

        raise TimeoutError(
            f"Training timed out: {job_id}"
        )

    if status not in (
        "succeeded",
        "completed",
    ):

        raise RuntimeError(
            f"Training failed: {status}\n"
            f"{json.dumps(body, indent=2)[:2000]}"
        )

    results = body.get("results") or {}

    result = results.get(entity)

    if not result:

        raise RuntimeError(
            f"No training result for {entity}"
        )

    if not result.get("model_id"):

        raise RuntimeError(
            "Training produced no model:\n"
            + json.dumps(
                result,
                indent=2,
                default=str,
            )
        )

    _log(
        f"Candidate model: {result['model_id']}"
    )

    return result, job_id


# ============================================================
# STEP 3 — VALIDATE CONFIGURATION
# ============================================================

def validate_model_config(
    result: dict,
    a,
) -> dict:

    effective = result.get("effective_config")

    if not isinstance(effective, dict):

        raise RuntimeError(
            "Training response has no effective_config.\n"
            "Cannot verify configuration before promotion.\n"
            "Inspect the candidate model metadata."
        )

    expected = {
        **_entity_defaults(a),
        **_entity_config(a),
    }

    mismatches = {}

    for key, expected_value in expected.items():

        actual_value = effective.get(key)

        if actual_value != expected_value:

            mismatches[key] = {
                "expected": expected_value,
                "actual": actual_value,
            }

    _log("CONFIGURATION VALIDATION")

    print(
        json.dumps(
            {
                "expected": expected,
                "effective": effective,
                "mismatches": mismatches,
            },
            indent=2,
            default=str,
        )
    )

    if mismatches:

        raise RuntimeError(
            "MODEL CONFIGURATION MISMATCH!\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )

    _log(
        "Configuration validated successfully."
    )

    return effective


# ============================================================
# STEP 4 — PROMOTE
# ============================================================

def promote(
    base: str,
    entity: str,
    model_id: str,
) -> None:

    site, kdvr, chan = _parts(entity)

    url = (
        f"{base}/v1/sites/{site}/entities/"
        f"{kdvr}/{chan}/models/{model_id}/promote"
    )

    response = requests.post(
        url,

        json={
            "promoted_by": "run_experiment"
        },

        timeout=120,
    )

    if response.status_code >= 400:

        raise RuntimeError(
            "Model promotion blocked:\n"
            + response.text[:1000]
        )

    _log(
        f"Model promoted: {model_id}"
    )


# ============================================================
# STEP 5 — CLEAR PREVIOUS DECISIONS
# ============================================================

def clear_previous(
    db: Path,
    entity: str,
    base: str,
) -> None:

    response = requests.post(
        f"{base}/v1/sessions/reset",

        json={
            "entity_id": entity
        },

        timeout=60,
    )

    response.raise_for_status()

    con = sqlite3.connect(db)

    try:

        with con:

            for sql in (
                "DELETE FROM scoring_results WHERE entity_id=?",
                "DELETE FROM scoring_artifacts WHERE entity_id=?",
                "DELETE FROM replenishment_orders WHERE entity_id=?",
                "DELETE FROM session_state WHERE entity_id=?",
            ):

                con.execute(
                    sql,
                    (entity,),
                )

    finally:
        con.close()

    safe = entity.replace(":", "_")

    artifact_dir = (
        Path(C.SCORE_ARTIFACT_DIR) / safe
    )

    if artifact_dir.exists():

        for path in sorted(
            artifact_dir.rglob("*"),
            key=lambda p: len(p.parts),
            reverse=True,
        ):

            path.chmod(0o700)

            if path.is_file():
                path.unlink()
            else:
                path.rmdir()

        artifact_dir.rmdir()

    _log(
        "Previous scoring decisions cleared."
    )


# ============================================================
# STEP 6 — SCORE
# ============================================================

def score_all(
    base: str,
    entity: str,
    data: Path,
    window: int,
    params: dict,
) -> dict:

    df = load_telemetry(data)

    df = df[
        df.entity_id == entity
    ].reset_index(drop=True)

    if len(df) < window:

        raise RuntimeError(
            f"Need {window} records, got {len(df)}"
        )

    site, kdvr, chan = _parts(entity)

    url = (
        f"{base}/v1/sites/{site}/entities/"
        f"{kdvr}/{chan}/score"
    )

    session = requests.Session()

    response_map = {}

    missing_request_ids = 0

    missing_fields = {
        field: 0
        for field in EXTRA_RESPONSE_FIELDS
    }

    n = 0

    start = time.perf_counter()

    try:

        for i in range(window - 1, len(df)):

            records = to_records(
                df.iloc[
                    i - window + 1:i + 1
                ]
            )

            payload = {
                "records": records,
                "write_to_db": True,
                "client": "run_experiment",
            }

            if params:
                payload["params"] = params

            response = session.post(
                url,
                json=payload,
                timeout=180,
            )

            response.raise_for_status()

            result = response.json()

            request_id = result.get(
                "request_id"
            )

            extra_fields = (
                _extract_response_fields(result)
            )

            if request_id is None:

                missing_request_ids += 1

            else:

                response_map[
                    str(request_id)
                ] = extra_fields

            for field, value in extra_fields.items():

                if value is None:
                    missing_fields[field] += 1

            n += 1

            if n % 500 == 0:

                _log(
                    f"Scored {n} decisions; "
                    f"captured {len(response_map)} responses."
                )

    finally:

        session.close()

    elapsed = time.perf_counter() - start

    _log(
        f"Scored {n} decisions in {elapsed:.1f}s"
    )

    if missing_request_ids:

        _log(
            f"WARNING: {missing_request_ids} "
            "responses have no request_id."
        )

    for field, count in missing_fields.items():

        if count:

            _log(
                f"WARNING: {field} returned "
                f"{count} null values."
            )

    return response_map


# ============================================================
# STEP 7 — EXPORT CSV
# ============================================================

def export(
    db: Path,
    entity: str,
    out: Path,
    response_map: dict,
) -> pd.DataFrame:

    con = sqlite3.connect(db)

    try:

        frame = pd.read_sql(
            """
            SELECT *
            FROM scoring_results
            WHERE entity_id=?
            ORDER BY decision_time
            """,

            con,

            params=(entity,),
        )

    finally:

        con.close()

    if "request_id" not in frame.columns:

        raise RuntimeError(
            "scoring_results has no request_id."
        )

    if response_map:

        extra = pd.DataFrame.from_dict(
            response_map,
            orient="index",
        )

        extra.index.name = "request_id"

        extra = extra.reset_index()

    else:

        extra = pd.DataFrame(
            columns=[
                "request_id",
                *EXTRA_RESPONSE_FIELDS,
            ]
        )

    for field in EXTRA_RESPONSE_FIELDS:

        if field not in extra.columns:
            extra[field] = None

    extra = extra[
        ["request_id", *EXTRA_RESPONSE_FIELDS]
    ]

    frame["request_id"] = (
        frame["request_id"].astype("string")
    )

    extra["request_id"] = (
        extra["request_id"].astype("string")
    )

    extra = extra.rename(
        columns={
            field: f"__api_{field}"
            for field in EXTRA_RESPONSE_FIELDS
        }
    )

    frame = frame.merge(
        extra,
        on="request_id",
        how="left",
        validate="many_to_one",
    )

    matched_ids = int(
        frame["request_id"].isin(
            list(response_map.keys())
        ).sum()
    )

    for field in EXTRA_RESPONSE_FIELDS:

        api_column = f"__api_{field}"

        if field in frame.columns:

            frame[field] = (
                frame[api_column]
                .combine_first(frame[field])
            )

        else:

            frame[field] = frame[api_column]

        frame = frame.drop(
            columns=[api_column]
        )

    for field in BOOLEAN_RESPONSE_FIELDS:

        frame[field] = (
            frame[field].astype("boolean")
        )

    frame["minutes_to_predicted_breach"] = (
        pd.to_numeric(
            frame["minutes_to_predicted_breach"],
            errors="coerce",
        )
    )

    _log(
        f"Matched API responses: "
        f"{matched_ids}/{len(frame)}"
    )

    for field in EXTRA_RESPONSE_FIELDS:

        matched = int(
            frame[field].notna().sum()
        )

        missing = len(frame) - matched

        _log(
            f"{field}: matched={matched}, "
            f"missing={missing}"
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        out,
        index=False,
    )

    _log(
        f"Exported {len(frame)} rows x "
        f"{len(frame.columns)} columns -> {out}"
    )

    return frame


# ============================================================
# STEP 8 — SUMMARY
# ============================================================

def summarise(frame: pd.DataFrame) -> None:

    if frame.empty:

        _log("No scoring results.")
        return

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        f"Decisions: {len(frame)}"
    )

    print(
        f"Window: "
        f"{frame.decision_time.min()} -> "
        f"{frame.decision_time.max()}"
    )

    print(
        f"Model ID: {frame.model_id.iloc[0]}"
    )

    print(
        "Alert states:",
        frame.alert_state.value_counts().to_dict(),
    )

    print(
        "Urgency:",
        frame["urgency"]
        .fillna("MISSING")
        .value_counts()
        .to_dict(),
    )

    for field in (
        "new_predictive_alert",
        "reactive_critical",
        "actionable_before_breach",
    ):

        count = int(
            frame[field].eq(True).sum()
        )

        print(
            f"{field}: {count}"
        )

    breach_minutes = pd.to_numeric(
        frame["minutes_to_predicted_breach"],
        errors="coerce",
    )

    if breach_minutes.notna().any():

        print(
            "Minutes to breach:",
            f"median={breach_minutes.median():.1f}",
            f"min={breach_minutes.min():.1f}",
            f"max={breach_minutes.max():.1f}",
        )

    orders = frame[
        frame.quantity > 0
    ]

    print(
        f"Orders: {len(orders)}"
    )

    print(
        f"Carts ordered: {int(orders.quantity.sum())}"
    )

    surplus = (
        frame.simulated_cart
        - frame.current_cart
    )

    print(
        "Simulated surplus:",
        f"median={surplus.median():.1f}",
        f"mean={surplus.mean():.2f}",
    )

    print(
        "Shortage risk:",
        f"median={frame.shortage_risk.median():.4f}",
        f"q90={frame.shortage_risk.quantile(0.9):.4f}",
        f"max={frame.shortage_risk.max():.4f}",
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    p = argparse.ArgumentParser(
        description=(
            "Parameterised upload, train, validate, "
            "promote, score and export experiment."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8100",
    )

    p.add_argument(
        "--entity",
        default="163:215:0",
    )

    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/experiment.csv"),
    )

    p.add_argument(
        "--name",
        default=None,
    )

    p.add_argument(
        "--timezone",
        default="Asia/Ho_Chi_Minh",
    )

    p.add_argument(
        "--window",
        type=int,
        default=HISTORY_WINDOW,
    )

    p.add_argument(
        "--train-timeout",
        type=int,
        default=900,
    )

    p.add_argument(
        "--skip-training",
        action="store_true",
    )

    p.add_argument(
        "--keep-previous",
        action="store_true",
    )

    # --------------------------------------------------------
    # Operational parameters
    # --------------------------------------------------------

    g = p.add_argument_group(
        "Operational parameters"
    )

    g.add_argument(
        "--bay-capacity",
        type=int,
        default=60,
    )

    g.add_argument(
        "--red-threshold",
        type=int,
        default=15,
    )

    g.add_argument(
        "--amber-threshold",
        type=int,
        default=20,
    )

    g.add_argument(
        "--target-stock",
        type=int,
        default=28,
    )

    g.add_argument(
        "--cooldown-min",
        type=float,
        default=30.0,
    )

    g.add_argument(
        "--lead-time-min",
        type=float,
        default=10.0,
    )

    g.add_argument(
        "--max-sequence-gap-min",
        type=float,
        default=15.0,
    )

    g.add_argument(
        "--quantity-rule",
        choices=["point", "risk"],
        default=None,
    )

    g.add_argument(
        "--min-carts-per-trip",
        type=int,
        default=None,
    )

    g.add_argument(
        "--max-carts-per-trip",
        type=int,
        default=None,
    )

    g.add_argument(
        "--reserve-per-day",
        type=int,
        default=None,
    )

    # --------------------------------------------------------
    # Risk overrides
    # --------------------------------------------------------

    r = p.add_argument_group(
        "Risk-band overrides"
    )

    r.add_argument(
        "--override-risk-threshold",
        type=float,
        default=None,
    )

    r.add_argument(
        "--override-critical-risk-threshold",
        type=float,
        default=None,
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    a = p.parse_args()

    _parts(a.entity)

    if not a.data.exists():

        raise SystemExit(
            f"Dataset not found: {a.data}"
        )

    if not (
        0 <= a.red_threshold
        < a.amber_threshold
        <= a.bay_capacity
    ):

        raise SystemExit(
            "Invalid thresholds: require "
            "0 <= RED < AMBER <= bay_capacity."
        )

    if not a.skip_training:

        a.name = a.name or _experiment_name(a)

    else:

        a.name = a.name or a.out.stem

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    response = requests.get(
        f"{a.base_url}/health",
        timeout=30,
    )

    response.raise_for_status()

    health = response.json()

    _log(
        f"Server status: {health.get('status')}"
    )

    # --------------------------------------------------------
    # Upload / Train / Validate / Promote
    # --------------------------------------------------------

    result = {}
    upload_id = None
    job_id = None
    effective_config = None
    upload_config = None

    if a.skip_training:

        _log(
            "Skipping training; using promoted model."
        )

    else:

        upload_id, upload_config = upload(
            a.base_url,
            a.data,
            a.entity,
            a,
        )

        result, job_id = train(
            a.base_url,
            upload_id,
            a.entity,
            a.train_timeout,
        )

        effective_config = validate_model_config(
            result,
            a,
        )

        promote(
            a.base_url,
            a.entity,
            result["model_id"],
        )

    # --------------------------------------------------------
    # Clear
    # --------------------------------------------------------

    if not a.keep_previous:

        clear_previous(
            Path(C.DB_PATH),
            a.entity,
            a.base_url,
        )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    params = _request_params(a)

    response_map = score_all(
        a.base_url,
        a.entity,
        a.data,
        a.window,
        params,
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    frame = export(
        Path(C.DB_PATH),
        a.entity,
        a.out,
        response_map,
    )

    # --------------------------------------------------------
    # Sidecar
    # --------------------------------------------------------

    sidecar = a.out.with_suffix(
        ".config.json"
    )

    config_data = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),

        "experiment_name": a.name,

        "entity_id": a.entity,

        "dataset": str(a.data),

        "upload_id": upload_id,

        "training_job_id": job_id,

        "history_window": a.window,

        "upload_config": upload_config,

        "entity_config_sent": _entity_config(a),

        "entity_defaults_sent": _entity_defaults(a),

        "effective_config": effective_config,

        "request_overrides": params,

        "model_id": (
            result.get("model_id")
            or (
                frame.model_id.iloc[0]
                if not frame.empty
                else None
            )
        ),

        "config_hash": result.get(
            "config_hash"
        ),

        "alert_bands": result.get(
            "alert_bands"
        ),

        "n_decisions": int(len(frame)),

        "urgency_distribution": (
            frame["urgency"]
            .fillna("MISSING")
            .value_counts()
            .to_dict()
        ),

        "api_response_fields": _response_summary(
            frame
        ),

        "api_response_request_ids": len(
            response_map
        ),
    }

    sidecar.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sidecar.write_text(
        json.dumps(
            config_data,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    _log(
        f"Configuration sidecar: {sidecar}"
    )

    summarise(frame)


if __name__ == "__main__":
    main()