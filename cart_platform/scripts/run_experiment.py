"""
One parameterised experiment:
Upload -> Train -> Validate Config -> Promote -> Score -> CSV.

Includes strict configuration debugging for:
- max_carts_per_trip
- reserve_per_day
- quantity_rule
- bay_capacity
- thresholds

IMPORTANT:
This script modifies the platform SQLite database directly.
Use only in a local development/test environment.
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
    "reason",
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


def parts(entity_id: str) -> tuple[str, str, str]:
    values = entity_id.split(":")

    if len(values) != 3:
        raise SystemExit(
            'Entity must use "siteKey:KDVR:ChanNo"'
        )

    return tuple(values)


def unwrap(value):
    """
    Support both:
        100
    and:
        {"value": 100, "source": "entity_config"}
    """
    if isinstance(value, dict) and "value" in value:
        return value["value"]

    return value


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

    site, kdvr, chan = parts(entity)

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

    log(
        f"Model promoted: {model_id}"
    )


# ============================================================
# CLEAR PREVIOUS
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

    log(
        "Previous scoring decisions cleared."
    )


# ============================================================
# API RESPONSE EXTRACTION
# ============================================================

def extract_response_fields(
    result: dict,
) -> dict:

    action = result.get("action") or {}

    return {
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

    site, kdvr, chan = parts(entity)

    url = (
        f"{base}/v1/sites/{site}/entities/"
        f"{kdvr}/{chan}/score"
    )

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

    frame[
        "minutes_to_predicted_breach"
    ] = pd.to_numeric(
        frame[
            "minutes_to_predicted_breach"
        ],
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

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

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
        default=Path(
            "outputs/experiment.csv"
        ),
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
        default=10.0,
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

    a = p.parse_args()

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

    if not a.keep_previous:
        clear_previous(
            Path(C.DB_PATH),
            a.entity,
            a.base_url,
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


if __name__ == "__main__":
    main()