"""
Two independent CSV/entity experiments in parallel:
Upload -> Train -> Validate Config -> Promote -> Score -> CSV.

Includes strict configuration debugging for:
- max_carts_per_trip
- reserve_per_day
- quantity_rule
- bay_capacity
- thresholds

Additional API extraction:
- pending_carts
- pending_orders
- pending_order_count
- pending_order_ids
- pending_order_quantities
- pending_order_completion_times
- pending_before_order
- current_cart_api
- simulated_cart_api
- forecast_min_40_api

Run this script from the platform scripts/ directory in a test environment.
Each CSV must already contain its assigned entity, using the platform CSV schema.
Upload is expected to onboard unknown entities; no registration endpoint is assumed.
Training, promotion and scoring write through the API. SQLite is read-only here.
C.DB_PATH must refer to the database used by the API server.
No previous state is reset. Use fresh entity IDs for a fresh-state experiment.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# ============================================================
# API RESPONSE FIELDS
# ============================================================

EXTRA_RESPONSE_FIELDS = (
    # Existing
    "urgency",
    "reason",
    "new_predictive_alert",
    "reactive_critical",
    "expected_completion_time",
    "actionable_before_breach",
    "minutes_to_predicted_breach",

    # Pending orders
    "pending_carts",
    "pending_orders",
    "pending_order_count",
    "pending_order_ids",
    "pending_order_quantities",
    "pending_order_completion_times",
    "pending_before_order",

    # Inventory
    "current_cart_api",
    "simulated_cart_api",

    # Forecast
    "forecast_min_40_api",
)

BOOLEAN_RESPONSE_FIELDS = (
    "new_predictive_alert",
    "reactive_critical",
    "actionable_before_breach",
)

NUMERIC_RESPONSE_FIELDS = (
    "pending_carts",
    "pending_order_count",
    "pending_before_order",
    "current_cart_api",
    "simulated_cart_api",
    "forecast_min_40_api",
    "minutes_to_predicted_breach",
)

DEBUG_FIELDS = (
    "red_threshold",
    "amber_threshold",
    "target_stock",
    "bay_capacity",
    "quantity_rule",
    "min_carts_per_trip",
    "max_carts_per_trip",
    "reserve_per_day",
    "lead_time_min",
    "cooldown_min",
    "max_sequence_gap_min",
)


# ============================================================
# UTILITIES
# ============================================================

def log(message: str) -> None:
    print(
        f"[{datetime.now():%H:%M:%S}] {message}",
        flush=True,
    )


def pretty(obj) -> None:
    print(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def parts(entity_id: str) -> tuple[str, str, str, str]:
    values = entity_id.split(":")

    if len(values) == 3:
        # Pre-bay deployment: the entity has no bay segment.
        values = values + [""]

    if len(values) != 4:
        raise SystemExit(
            'Entity must use "siteKey:KDVR:ChanNo:BayID"'
        )

    return tuple(values)


def entity_path(base: str, entity: str) -> str:
    """URL prefix addressing one bay (or one channel on a pre-bay platform)."""
    site, kdvr, chan, bay = parts(entity)
    url = f"{base}/v1/sites/{site}/entities/{kdvr}/{chan}"
    return f"{url}/bays/{bay}" if bay else url


def unwrap(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]

    return value


def json_string(value) -> str:
    """
    Convert nested API objects to a valid JSON string for CSV.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def entity_config(a) -> dict:
    values = {
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
        key: value
        for key, value in values.items()
        if value is not None
    }


def entity_defaults(a) -> dict:
    values = {
        "max_sequence_gap_min": a.max_sequence_gap_min,
        "cooldown_min": a.cooldown_min,
        "lead_time_min": a.lead_time_min,
    }

    return {
        key: value
        for key, value in values.items()
        if value is not None
    }


def request_params(a) -> dict:
    values = {
        "risk_threshold": a.override_risk_threshold,
        "critical_risk_threshold":
            a.override_critical_risk_threshold,
    }

    return {
        key: value
        for key, value in values.items()
        if value is not None
    }


def experiment_name(a) -> str:
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    return (
        f"{a.out.stem}"
        f"_red{a.red_threshold}"
        f"_amber{a.amber_threshold}"
        f"_target{a.target_stock}"
        f"_{timestamp}_{uuid4().hex[:8]}"
    )


# ============================================================
# DEBUG REPORT
# ============================================================

class ConfigDebugger:

    def __init__(
        self,
        output_path: Path,
        entity_id: str,
        expected: dict,
    ):
        self.output_path = output_path
        self.expected = expected

        self.report = {
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "entity_id": entity_id,
            "expected_config": expected,
            "stages": {},
        }

    def save(self) -> None:
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_path.write_text(
            json.dumps(
                self.report,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    def inspect(
        self,
        stage: str,
        config: dict | None,
        strict: bool = False,
    ) -> dict:

        print("\n" + "=" * 75)
        print(f"CONFIG DEBUG: {stage}")
        print("=" * 75)

        if not isinstance(config, dict):
            result = {
                "status": "UNAVAILABLE",
                "config": config,
                "mismatches": {},
                "missing": list(self.expected),
            }

            self.report["stages"][stage] = result
            self.save()

            print("Config unavailable")

            if strict:
                raise RuntimeError(
                    f"{stage}: configuration unavailable"
                )

            return result

        mismatches = {}
        missing = []
        values = {}

        for key in DEBUG_FIELDS:
            raw = config.get(key)

            actual = unwrap(raw)

            source = (
                raw.get("source")
                if isinstance(raw, dict)
                else None
            )

            values[key] = {
                "value": actual,
                "source": source,
                "present": key in config,
            }

            expected = self.expected.get(key)

            print(
                f"{key:<28} "
                f"expected={str(expected):<12} "
                f"actual={str(actual):<12} "
                f"source={source}"
            )

            if key not in self.expected:
                continue

            if key not in config:
                missing.append(key)
                continue

            if actual != expected:
                mismatches[key] = {
                    "expected": expected,
                    "actual": actual,
                    "source": source,
                }

        if mismatches:
            status = "MISMATCH"
        elif missing:
            status = "INCOMPLETE"
        else:
            status = "MATCH"

        result = {
            "status": status,
            "values": values,
            "mismatches": mismatches,
            "missing": missing,
            "raw_config": config,
        }

        self.report["stages"][stage] = result
        self.save()

        print("\nSTATUS:", status)

        if mismatches:
            pretty(mismatches)

        if missing:
            print("Missing fields:", missing)

        if strict and status != "MATCH":
            raise RuntimeError(
                f"{stage}: CONFIG VALIDATION FAILED. "
                f"See {self.output_path}"
            )

        return result

    def attach(
        self,
        name: str,
        value,
    ) -> None:
        self.report[name] = value
        self.save()


# ============================================================
# UPLOAD
# ============================================================

def upload(
    base: str,
    data: Path,
    entity: str,
    a,
    debugger: ConfigDebugger,
) -> tuple[str, dict]:

    cfg = {
        "name": a.name,
        "timezone": a.timezone,
        "entity_defaults": entity_defaults(a),
        "entity_overrides": {
            entity: entity_config(a)
        },
        "auto_start_training": False,
    }

    log("UPLOAD CONFIGURATION")
    pretty(cfg)

    debugger.inspect(
        "upload_request_entity_overrides",
        cfg["entity_overrides"][entity],
        strict=False,
    )

    debugger.attach(
        "upload_request",
        cfg,
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

    log("UPLOAD RESPONSE")
    pretty(body)

    debugger.attach(
        "upload_response",
        body,
    )

    upload_id = body["upload_id"]

    if body.get("duplicate_of"):
        raise RuntimeError(
            "DUPLICATE UPLOAD DETECTED!\n"
            f"upload_id={upload_id}\n"
            f"duplicate_of={body['duplicate_of']}"
        )

    entities = {
        item["entity_id"]: item
        for item in body.get("entities", [])
    }

    if entity not in entities:
        raise RuntimeError(
            f"Entity {entity} not found in upload"
        )

    return upload_id, cfg


# ============================================================
# TRAIN
# ============================================================

def train(
    base: str,
    upload_id: str,
    entity: str,
    timeout_s: int,
    debugger: ConfigDebugger,
) -> tuple[dict, str]:

    payload = {
        "upload_id": upload_id,
        "entity_ids": [entity],
        "strategy": "per_entity",
    }

    log("TRAINING REQUEST")
    pretty(payload)

    response = requests.post(
        f"{base}/v1/training/jobs",
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    job_id = response.json()["job_id"]

    log(f"Training job: {job_id}")

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

    result = (
        body.get("results") or {}
    ).get(entity)

    if not result or not result.get("model_id"):
        raise RuntimeError(
            "Training produced no model:\n"
            + json.dumps(
                result,
                indent=2,
                default=str,
            )
        )

    debugger.attach(
        "training_result",
        result,
    )

    log(
        f"Candidate model: {result['model_id']}"
    )

    return result, job_id


# ============================================================
# PROMOTE
# ============================================================

def promote(
    base: str,
    entity: str,
    model_id: str,
) -> None:

    url = f"{entity_path(base, entity)}/models/{model_id}/promote"

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

    log(
        f"Model promoted: {model_id}"
    )


# ============================================================
# API RESPONSE EXTRACTION
# ============================================================

def extract_response_fields(
    result: dict,
) -> dict:
    """
    Extract additional fields directly from the scoring API.

    pending_carts is the pending quantity AFTER the scoring decision.

    pending_before_order is inferred as:
        pending_carts - action.quantity

    This matches the current score() implementation, where the new
    order is placed before the response is assembled.
    """

    action = result.get("action") or {}
    forecast = result.get("forecast") or {}

    pending_orders = result.get("pending_orders") or []

    if not isinstance(pending_orders, list):
        pending_orders = []

    pending_carts = result.get("pending_carts")

    quantity = action.get("quantity") or 0

    if pending_carts is not None:
        pending_before_order = (
            float(pending_carts) - float(quantity)
        )
    else:
        pending_before_order = None

    pending_order_ids = []
    pending_order_quantities = []
    pending_order_completion_times = []

    for order in pending_orders:
        if not isinstance(order, dict):
            continue

        pending_order_ids.append(
            order.get("order_id")
        )

        pending_order_quantities.append(
            order.get("quantity")
        )

        pending_order_completion_times.append(
            order.get("expected_completion_time")
        )

    return {
        # Existing fields
        "urgency": action.get("urgency"),

        "reason": action.get("reason"),

        "new_predictive_alert":
            result.get("new_predictive_alert"),

        "reactive_critical":
            result.get("reactive_critical"),

        "expected_completion_time":
            action.get("expected_completion_time"),

        "actionable_before_breach":
            action.get("actionable_before_breach"),

        "minutes_to_predicted_breach":
            action.get("minutes_to_predicted_breach"),

        # Pending
        "pending_carts": pending_carts,

        "pending_orders": json_string(
            pending_orders
        ),

        "pending_order_count": len(
            pending_orders
        ),

        "pending_order_ids": json_string(
            pending_order_ids
        ),

        "pending_order_quantities": json_string(
            pending_order_quantities
        ),

        "pending_order_completion_times": json_string(
            pending_order_completion_times
        ),

        "pending_before_order":
            pending_before_order,

        # Inventory
        "current_cart_api":
            result.get("current_cart"),

        "simulated_cart_api":
            result.get("simulated_cart"),

        # Forecast
        "forecast_min_40_api":
            forecast.get("min_within_40min"),
    }


# ============================================================
# SCORE
# ============================================================

def score_all(
    base: str,
    entity: str,
    data: Path,
    window: int,
    params: dict,
    debugger: ConfigDebugger,
    expected_model_id: str | None = None,
) -> dict:

    df = load_telemetry(data)

    df = df[
        df.entity_id == entity
    ].reset_index(drop=True)

    if len(df) < window:
        raise RuntimeError(
            f"Need {window} records, got {len(df)}"
        )

    url = f"{entity_path(base, entity)}/score"

    session = requests.Session()

    response_map = {}

    n = 0

    start = time.perf_counter()

    try:
        for i in range(
            window - 1,
            len(df),
        ):

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

            if n == 0:
                log(
                    "FIRST SCORING RESPONSE"
                )

                pretty(
                    {
                        "model_id":
                            result.get("model_id"),

                        "config_hash":
                            result.get("config_hash"),

                        "resolved_config":
                            result.get("resolved_config"),

                        "frozen_model_config":
                            result.get("frozen_model_config"),

                        "action":
                            result.get("action"),

                        "pending_carts":
                            result.get("pending_carts"),

                        "pending_orders":
                            result.get("pending_orders"),
                    }
                )

                debugger.attach(
                    "first_scoring_response",
                    result,
                )

                resolved = result.get(
                    "resolved_config"
                )

                debugger.inspect(
                    "scoring_resolved_config",
                    resolved,
                    strict=(
                        expected_model_id is not None
                    ),
                )

                if expected_model_id is not None:
                    actual_model_id = result.get(
                        "model_id"
                    )

                    if actual_model_id != expected_model_id:
                        raise RuntimeError(
                            "SCORING MODEL MISMATCH!\n"
                            f"Expected: {expected_model_id}\n"
                            f"Actual: {actual_model_id}"
                        )

            request_id = result.get(
                "request_id"
            )

            if request_id is None or not str(request_id):
                raise RuntimeError("Scoring response missing request_id; cannot join API and DB results")
            if str(request_id) in response_map:
                raise RuntimeError(f"Duplicate scoring request_id: {request_id}")
            if expected_model_id is not None and result.get("model_id") != expected_model_id:
                raise RuntimeError(f"Model changed during scoring: {result.get('model_id')}")

            if request_id is not None:
                response_map[
                    str(request_id)
                ] = extract_response_fields(
                    result
                )

            n += 1

            if n % 500 == 0:
                log(
                    f"Scored {n} decisions"
                )

    finally:
        session.close()

    elapsed = time.perf_counter() - start

    log(
        f"Scored {n} decisions "
        f"in {elapsed:.1f}s"
    )

    return response_map


# ============================================================
# EXPORT
# ============================================================

def export(
    db: Path,
    entity: str,
    out: Path,
    response_map: dict,
) -> pd.DataFrame:

    con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)

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

    # Export only requests from this experiment, even if the entity has history.
    frame = frame[frame["request_id"].astype("string").isin(response_map)].copy()
    found_ids = set(frame["request_id"].astype(str))
    missing_ids = set(response_map) - found_ids
    if missing_ids or frame["request_id"].duplicated().any():
        raise RuntimeError(
            f"DB/API mismatch: {len(missing_ids)} requests missing or duplicate DB rows. "
            "Check that settings.DB_PATH is the API server database and writes have completed."
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
        [
            "request_id",
            *EXTRA_RESPONSE_FIELDS,
        ]
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

    for field in NUMERIC_RESPONSE_FIELDS:
        frame[field] = pd.to_numeric(
            frame[field],
            errors="coerce",
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        out,
        index=False,
    )

    log(
        f"Exported {len(frame)} rows "
        f"-> {out}"
    )

    return frame


# ============================================================
# SUMMARY
# ============================================================

def summarise(frame: pd.DataFrame) -> None:

    if frame.empty:
        log("No scoring results.")
        return

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        f"Decisions: {len(frame)}"
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

    orders = frame[
        frame.quantity > 0
    ]

    print(
        f"Orders: {len(orders)}"
    )

    print(
        f"Carts ordered: "
        f"{int(orders.quantity.sum())}"
    )

    print(
        "Maximum pending carts:",
        frame["pending_carts"].max(),
    )

    print(
        "Maximum pending orders:",
        frame["pending_order_count"].max(),
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8100",
    )

    p.add_argument(
        "--case", nargs=2, action="append", required=True,
        metavar=("ENTITY", "CSV"),
        help="Repeat exactly twice, with a different entity and CSV each time.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/parallel"))

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
        default=0,
    )

    g.add_argument(
        "--lead-time-min",
        type=float,
        default=20.0,
    )

    g.add_argument(
        "--max-sequence-gap-min",
        type=float,
        default=15.0,
    )

    g.add_argument(
        "--quantity-rule",
        choices=[
            "point",
            "risk",
        ],
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

    return p


def run_experiment(a) -> dict:
    parts(a.entity)

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

    if a.max_carts_per_trip is not None:
        if a.max_carts_per_trip <= 0:
            raise SystemExit(
                "--max-carts-per-trip must be > 0"
            )

    if a.reserve_per_day is not None:
        if a.reserve_per_day < 0:
            raise SystemExit(
                "--reserve-per-day must be >= 0"
            )

    if not a.skip_training:
        a.name = (
            a.name
            or experiment_name(a)
        )
    else:
        a.name = (
            a.name
            or a.out.stem
        )

    expected_config = {
        **entity_defaults(a),
        **entity_config(a),
    }

    debug_path = a.out.with_suffix(
        ".debug.json"
    )

    debugger = ConfigDebugger(
        output_path=debug_path,
        entity_id=a.entity,
        expected=expected_config,
    )

    debugger.save()

    response = requests.get(
        f"{a.base_url}/health",
        timeout=30,
    )

    response.raise_for_status()

    log(
        f"Server status: "
        f"{response.json().get('status')}"
    )

    result = {}
    upload_id = None
    job_id = None
    effective_config = None
    upload_config = None

    if a.skip_training:
        log(
            "Skipping training; using promoted model."
        )

    else:
        upload_id, upload_config = upload(
            a.base_url,
            a.data,
            a.entity,
            a,
            debugger,
        )

        result, job_id = train(
            a.base_url,
            upload_id,
            a.entity,
            a.train_timeout,
            debugger,
        )

        effective_config = result.get(
            "effective_config"
        )

        debugger.inspect(
            "training_effective_config",
            effective_config,
            strict=True,
        )

        promote(
            a.base_url,
            a.entity,
            result["model_id"],
        )

    params = request_params(a)

    response_map = score_all(
        a.base_url,
        a.entity,
        a.data,
        a.window,
        params,
        debugger,
        expected_model_id=(
            result.get("model_id")
            if not a.skip_training
            else None
        ),
    )

    frame = export(
        Path(C.DB_PATH),
        a.entity,
        a.out,
        response_map,
    )

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
        "entity_config_sent": entity_config(a),
        "entity_defaults_sent": entity_defaults(a),
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
        "config_hash": result.get("config_hash"),
        "alert_bands": result.get("alert_bands"),
        "n_decisions": int(len(frame)),
        "debug_report": str(debug_path),
    }

    sidecar.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sidecar.write_text(
        json.dumps(
            config_data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    log(
        f"Configuration sidecar: {sidecar}"
    )

    log(
        f"Debug report: {debug_path}"
    )

    summarise(frame)

    return {
        "entity_id": a.entity,
        "dataset": str(a.data),
        "status": "succeeded",
        "csv": str(a.out),
        "config": str(sidecar),
        "debug": str(debug_path),
        "upload_id": upload_id,
        "training_job_id": job_id,
        "model_id": result.get("model_id"),
        "n_decisions": len(frame),
    }


def run_worker(a) -> dict:
    log_path = a.out.with_suffix(".log")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                result = run_experiment(a)
            except Exception as exc:
                traceback.print_exc()
                result = {
                    "entity_id": a.entity,
                    "dataset": str(a.data),
                    "status": "failed",
                    "error": str(exc),
                }
    result["log"] = str(log_path)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def preflight(a, parser) -> None:
    # Validate both inputs before either process can mutate the backend.
    if len(a.case) != 2:
        parser.error("Provide exactly two --case ENTITY CSV arguments")
    entities = [entity for entity, _ in a.case]
    paths = [Path(path).expanduser().resolve() for _, path in a.case]
    if len(set(entities)) != 2:
        parser.error("The two entity IDs must be different")
    if paths[0] == paths[1] or (all(p.is_file() for p in paths) and paths[0].samefile(paths[1])):
        parser.error("Use two different CSV files")
    if a.window < 1 or a.train_timeout <= 0:
        parser.error("--window and --train-timeout must be positive")
    if not (0 <= a.red_threshold < a.amber_threshold <= a.bay_capacity):
        parser.error("Require 0 <= RED < AMBER <= bay_capacity")
    if a.max_carts_per_trip is not None and a.max_carts_per_trip <= 0:
        parser.error("--max-carts-per-trip must be > 0")
    if a.reserve_per_day is not None and a.reserve_per_day < 0:
        parser.error("--reserve-per-day must be >= 0")
    db = Path(C.DB_PATH).resolve()
    if not db.is_file():
        parser.error(f"API server SQLite DB not found locally: {db}")
    # Fail before uploads if the local export schema is unavailable.
    with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as con:
        con.execute("SELECT entity_id, request_id FROM scoring_results LIMIT 0")
    for entity, path in zip(entities, paths):
        if len(entity.split(":")) != 4 or any(not part for part in entity.split(":")):
            parser.error('Entity must use "siteKey:KDVR:ChanNo:BayID" with '
                         'non-empty parts')
        if any(character in entity for character in "/?#\\"):
            parser.error("Entity IDs cannot contain URL/path separators")
        if not path.is_file():
            parser.error(f"CSV not found: {path}")
        df = load_telemetry(path)
        actual = set(df.entity_id.dropna().astype(str))
        if df.entity_id.isna().any() or actual != {entity}:
            parser.error(f"{path}: must contain only entity {entity}; found {sorted(actual)}")
        if len(df) < a.window:
            parser.error(f"{path}: need at least {a.window} records; found {len(df)}")
    a.case = list(zip(entities, paths))


def main() -> None:
    parser = build_parser()
    a = parser.parse_args()
    preflight(a, parser)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = a.out_dir.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "results.json"
    results = [None, None]
    log(f"Starting two independent experiment workers; results: {run_dir}")
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for index, (entity, data) in enumerate(a.case):
            worker = copy.deepcopy(a)
            worker.entity = entity
            worker.data = data
            worker.out = run_dir / f"entity_{index + 1}.csv"
            worker.skip_training = False
            worker.name = f"{a.name or 'parallel'}_{index + 1}_{run_id}"
            futures[pool.submit(run_worker, worker)] = (index, worker)
        for future in as_completed(futures):
            index, worker = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "entity_id": worker.entity,
                    "dataset": str(worker.data),
                    "status": "failed",
                    "error": str(exc),
                    "log": str(worker.out.with_suffix(".log")),
                }
            results[index] = result
            # Each result is returned immediately when its worker finishes.
            pretty(result)
            temporary = summary_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"run_id": run_id, "results": results},
                                            indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(summary_path)
    log(f"Combined result: {summary_path}")
    if any(result["status"] != "succeeded" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
