"""
Cart Counting — CSV Backtest via Scoring API

Workflow:
    CSV -> Sliding Window -> Scoring API -> JSON Response -> CSV

Features:
    - No training
    - No upload
    - No model promotion
    - Uses currently promoted model
    - Exports scoring results directly from API response
    - should_replenish converted to 1/0
    - No SQLite dependency
    - Progress logging
    - Partial CSV saved on failure

Example:
    python scripts/run_experiment.py \
        --entity "163:215:0" \
        --data "/path/to/Thinh(2).csv" \
        --out outputs/backtest.csv
"""

from __future__ import annotations

import argparse
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PLATFORM_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PLATFORM_ROOT))

from client.platform_client import load_telemetry, to_records


DEFAULT_DATA = PLATFORM_ROOT / "data" / "real_data.csv"

DEFAULT_OUTPUT = PLATFORM_ROOT / "outputs" / "backtest.csv"

HISTORY_WINDOW = 20


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [
    "id",
    "event_id",
    "request_id",
    "entity_id",
    "kdvr",
    "chan_no",
    "site_key",
    "decision_time",
    "current_cart",
    "simulated_cart",
    "pred_cart_20min",
    "pred_cart_40min",
    "pred_min_20min",
    "pred_min_40min",
    "lower_bound_min_40min",
    "shortage_risk",
    "alert_state",
    "should_replenish",
    "quantity",
    "model_id",
    "calibration_id",
    "scored_at",
]


# ============================================================
# LOGGING
# ============================================================

def log(message: str) -> None:

    timestamp = datetime.now().strftime("%H:%M:%S")

    print(
        f"[{timestamp}] {message}",
        flush=True
    )


# ============================================================
# ENTITY PARSER
# ============================================================

def parse_entity(entity_id: str):

    parts = entity_id.split(":")

    if len(parts) != 3:

        raise ValueError(
            "Entity format must be siteKey:KDVR:ChanNo"
        )

    return parts


# ============================================================
# EXTRACT API RESPONSE
# ============================================================

def extract_result(response: dict) -> dict:

    forecast = response.get("forecast") or {}

    action = response.get("action") or {}

    should_replenish = action.get("should_replenish")

    if should_replenish is None:
        should_replenish_value = None
    elif isinstance(should_replenish, bool):
        should_replenish_value = int(should_replenish)
    else:
        raise ValueError(
            f"Invalid should_replenish value: {should_replenish!r}"
        )

    result = {

        # ----------------------------------------------------
        # IDENTIFIERS
        # ----------------------------------------------------

        "id": None,

        "event_id": (
            response.get("artifact_request_id")
            or response.get("request_id")
        ),

        "request_id": response.get("request_id"),

        "entity_id": response.get("entity_id"),

        "kdvr": response.get("kdvr"),

        "chan_no": response.get("chan_no"),

        "site_key": response.get("site_key"),

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        "decision_time": response.get("decision_time"),

        # ----------------------------------------------------
        # CURRENT CART STATE
        # ----------------------------------------------------

        "current_cart": response.get("current_cart"),

        "simulated_cart": response.get("simulated_cart"),

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        "pred_cart_20min": forecast.get(
            "cart_at_20min"
        ),

        "pred_cart_40min": forecast.get(
            "cart_at_40min"
        ),

        "pred_min_20min": forecast.get(
            "min_within_20min"
        ),

        "pred_min_40min": forecast.get(
            "min_within_40min"
        ),

        "lower_bound_min_40min": forecast.get(
            "lower_bound_min_within_40min"
        ),

        # ----------------------------------------------------
        # RISK & ALERT
        # ----------------------------------------------------

        "shortage_risk": response.get(
            "shortage_risk"
        ),

        "alert_state": response.get(
            "alert_state"
        ),

        # ----------------------------------------------------
        # REPLENISHMENT ACTION
        # ----------------------------------------------------

        "should_replenish": should_replenish_value,

        "quantity": action.get(
            "quantity",
            0
        ),

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        "model_id": response.get(
            "model_id"
        ),

        "calibration_id": response.get(
            "calibration_id"
        ),

        # ----------------------------------------------------
        # SCORING TIMESTAMP
        # ----------------------------------------------------

        "scored_at": datetime.now(
            timezone.utc
        ).isoformat(),

    }

    return result


# ============================================================
# EXPORT RESULTS
# ============================================================

def export_results(
    results: list,
    output_path: Path
) -> pd.DataFrame:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    frame = pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS
    )

    frame.to_csv(
        output_path,
        index=False
    )

    return frame


# ============================================================
# SUMMARY
# ============================================================

def summarise(frame: pd.DataFrame) -> None:

    if frame.empty:

        log("No scoring results")

        return

    print("\n" + "=" * 65)

    print("CART COUNTING — BACKTEST SUMMARY")

    print("=" * 65)

    print(
        f"Decisions: {len(frame)}"
    )

    print(
        f"Model ID: {frame['model_id'].iloc[0]}"
    )

    print(
        f"Decision window: "
        f"{frame['decision_time'].min()} "
        f"-> "
        f"{frame['decision_time'].max()}"
    )

    print("\nAlert distribution:")

    print(
        frame["alert_state"].value_counts(
            dropna=False
        )
    )

    print("\nReplenishment:")

    replenish_count = int(
        frame["should_replenish"].eq(1).sum()
    )

    no_replenish_count = int(
        frame["should_replenish"].eq(0).sum()
    )

    print(
        f"Should replenish = 1: {replenish_count}"
    )

    print(
        f"Should replenish = 0: {no_replenish_count}"
    )

    print(
        f"Total carts ordered: "
        f"{pd.to_numeric(frame['quantity'], errors='coerce').sum():.0f}"
    )

    risk = pd.to_numeric(
        frame["shortage_risk"],
        errors="coerce"
    )

    print("\nShortage risk:")

    print(
        f"Median: {risk.median():.4f}"
    )

    print(
        f"P90: {risk.quantile(0.9):.4f}"
    )

    print(
        f"Max: {risk.max():.4f}"
    )

    print("=" * 65)


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    base_url: str,
    entity: str,
    data_path: Path,
    output_path: Path,
    window: int,
    write_to_db: bool,
    risk_threshold: float | None = None,
    critical_risk_threshold: float | None = None,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    log(f"Loading dataset: {data_path}")

    df = load_telemetry(data_path)

    df = df[
        df["entity_id"] == entity
    ].reset_index(drop=True)

    log(
        f"Entity: {entity}"
    )

    log(
        f"Records: {len(df)}"
    )

    if len(df) < window:

        raise SystemExit(
            f"Not enough records. "
            f"Required: {window}, "
            f"Available: {len(df)}"
        )

    # --------------------------------------------------------
    # API ENDPOINT
    # --------------------------------------------------------

    site, kdvr, chan = parse_entity(entity)

    url = (
        f"{base_url}/v1/sites/{site}"
        f"/entities/{kdvr}/{chan}/score"
    )

    log(
        f"Scoring API: {url}"
    )

    # --------------------------------------------------------
    # OPTIONAL RISK THRESHOLD OVERRIDES
    # --------------------------------------------------------

    params = {}

    if risk_threshold is not None:

        params["risk_threshold"] = (
            risk_threshold
        )

    if critical_risk_threshold is not None:

        params["critical_risk_threshold"] = (
            critical_risk_threshold
        )

    if params:

        log(
            f"Request overrides: {params}"
        )

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    results = []

    session = requests.Session()

    total = len(df) - window + 1

    start_time = time.perf_counter()

    log(
        f"Starting backtest: {total} decisions"
    )

    log(
        f"Write to DB: {write_to_db}"
    )

    # --------------------------------------------------------
    # SLIDING WINDOW
    # --------------------------------------------------------

    for i in range(
        window - 1,
        len(df)
    ):

        history = df.iloc[
            i - window + 1: i + 1
        ]

        payload = {

            "records": to_records(history),

            "write_to_db": write_to_db,

            "client": "csv_backtest",

        }

        if params:

            payload["params"] = params

        # ----------------------------------------------------
        # CALL SCORING API
        # ----------------------------------------------------

        try:

            response = session.post(
                url,
                json=payload,
                timeout=180
            )

            response.raise_for_status()

            body = response.json()

            # ------------------------------------------------
            # EXTRACT RESULT
            # ------------------------------------------------

            result = extract_result(body)

            results.append(result)

        except (
            requests.RequestException,
            ValueError,
            KeyError
        ) as exc:

            log(
                f"ERROR at record index {i}: {exc}"
            )

            # Save completed results before exiting
            partial_output = output_path.with_name(
                output_path.stem
                + "_partial.csv"
            )

            export_results(
                results,
                partial_output
            )

            log(
                f"Partial results saved: "
                f"{partial_output}"
            )

            raise

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        completed = len(results)

        if (
            completed % 500 == 0
            or completed == total
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            rate = (
                completed / elapsed
                if elapsed > 0
                else 0
            )

            log(
                f"{completed}/{total} decisions | "
                f"{elapsed:.1f}s | "
                f"{rate:.2f} decisions/sec"
            )

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    frame = export_results(
        results,
        output_path
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    log(
        f"Exported {len(frame)} rows "
        f"x {len(frame.columns)} columns"
    )

    log(
        f"Output: {output_path}"
    )

    log(
        f"Total runtime: {elapsed:.1f}s"
    )

    summarise(frame)

    return frame


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Backtest CSV using the currently "
            "promoted Cart Counting model"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8100"
    )

    # --------------------------------------------------------
    # ENTITY
    # --------------------------------------------------------

    parser.add_argument(
        "--entity",
        default="163:215:0"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    parser.add_argument(
        "--window",
        type=int,
        default=HISTORY_WINDOW
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    parser.add_argument(
        "--write-to-db",
        action="store_true",
        help="Persist scoring results to DB"
    )

    # --------------------------------------------------------
    # OPTIONAL RISK OVERRIDES
    # --------------------------------------------------------

    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=None
    )

    parser.add_argument(
        "--critical-risk-threshold",
        type=float,
        default=None
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # VALIDATE DATASET
    # --------------------------------------------------------

    if not args.data.exists():

        raise SystemExit(
            f"Dataset not found: {args.data}"
        )

    if args.window < 1:

        raise SystemExit(
            "--window must be >= 1"
        )

    # --------------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------------

    log(
        f"Checking API server: {args.base_url}"
    )

    response = requests.get(
        f"{args.base_url}/health",
        timeout=30
    )

    response.raise_for_status()

    health = response.json()

    log(
        f"Server status: {health.get('status')}"
    )

    log(
        f"Champion model: "
        f"{health.get('champion_model_id')}"
    )

    # --------------------------------------------------------
    # RUN BACKTEST
    # --------------------------------------------------------

    run_backtest(

        base_url=args.base_url,

        entity=args.entity,

        data_path=args.data,

        output_path=args.out,

        window=args.window,

        write_to_db=args.write_to_db,

        risk_threshold=args.risk_threshold,

        critical_risk_threshold=(
            args.critical_risk_threshold
        ),

    )


if __name__ == "__main__":
    main()