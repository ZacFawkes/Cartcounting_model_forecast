"""
One parameterised experiment: upload -> train -> promote -> score -> CSV.

Runs the whole on-demand pipeline for a single entity with the operational
parameters supplied on the command line, then exports every decision as CSV.

The exported CSV contains:
    - All columns from the scoring_results table
    - urgency, extracted from the scoring API response's action.urgency

Each run is self-contained: the parameters are frozen into the model, recorded
in the CSV where supported by the database schema, and written to a sidecar
JSON next to it.

Examples
--------
    # Baseline
    python scripts/run_experiment.py --out outputs/base.csv

    # Change RED threshold
    python scripts/run_experiment.py \
        --red-threshold 12 \
        --out outputs/red12.csv

    # Score with an already promoted model, no retraining
    python scripts/run_experiment.py \
        --skip-training \
        --out outputs/rescore.csv

    # Override risk bands for diagnostics
    python scripts/run_experiment.py \
        --override-risk-threshold 0.15 \
        --override-critical-risk-threshold 0.90 \
        --out outputs/risk015_crit090.csv
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

import settings as C  # noqa: E402
from client.platform_client import load_telemetry, to_records  # noqa: E402


DEFAULT_DATA = PLATFORM_ROOT / "data" / "real_data.csv"
HISTORY_WINDOW = 20


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _parts(entity_id: str) -> tuple[str, str, str]:
    """Convert siteKey:KDVR:ChanNo into three path segments."""

    bits = entity_id.split(":")

    if len(bits) != 3:
        raise SystemExit(
            f'entity must be "siteKey:KDVR:ChanNo" '
            f"(got {entity_id!r})"
        )

    return bits[0], bits[1], bits[2]


def _log(msg: str) -> None:
    print(
        f"[{datetime.now():%H:%M:%S}] {msg}",
        flush=True
    )


def _entity_config(a) -> dict:
    """
    Only parameters actually supplied are sent.

    Omitting a parameter allows the platform's own configuration
    resolution to apply.
    """

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


# --------------------------------------------------------------------------
# Step 1: Upload
# --------------------------------------------------------------------------

def upload(
    base: str,
    data: Path,
    entity: str,
    a
) -> str:

    cfg = {
        "name": a.name,
        "timezone": a.timezone,

        "entity_defaults": {
            k: v
            for k, v in (
                ("max_sequence_gap_min", a.max_sequence_gap_min),
                ("cooldown_min", a.cooldown_min),
                ("lead_time_min", a.lead_time_min),
            )
            if v is not None
        },

        "entity_overrides": {
            entity: _entity_config(a)
        },

        "auto_start_training": False,
    }

    _log(
        f"upload {data.name} — "
        f"overrides: {cfg['entity_overrides'][entity]}"
    )

    with open(data, "rb") as fh:

        r = requests.post(
            f"{base}/v1/datasets/upload",

            files={
                "file": (
                    data.name,
                    fh,
                    "text/csv"
                )
            },

            data={
                "config": json.dumps(cfg)
            },

            timeout=600
        )

    r.raise_for_status()

    body = r.json()

    upload_id = body["upload_id"]

    if body.get("duplicate_of"):
        _log(
            "  identical upload already registered: "
            f"{body['duplicate_of']}"
        )

    for w in body.get("warnings", [])[:3]:
        _log(f"  warning: {w}")

    _log(f"  upload_id={upload_id}")

    return upload_id


# --------------------------------------------------------------------------
# Step 2: Train
# --------------------------------------------------------------------------

def train(
    base: str,
    upload_id: str,
    entity: str,
    timeout_s: int
) -> dict:

    r = requests.post(
        f"{base}/v1/training/jobs",

        timeout=60,

        json={
            "upload_id": upload_id,
            "entity_ids": [entity],
            "strategy": "per_entity"
        }
    )

    r.raise_for_status()

    job_id = r.json()["job_id"]

    _log(f"training job {job_id}")

    deadline = time.time() + timeout_s

    status = "running"
    body = {}

    while time.time() < deadline:

        time.sleep(5)

        response = requests.get(
            f"{base}/v1/training/jobs/{job_id}",
            timeout=60
        )

        response.raise_for_status()

        body = response.json()
        status = body["status"]

        if status in (
            "succeeded",
            "completed",
            "failed"
        ):
            break

    if status not in ("succeeded", "completed"):

        raise SystemExit(
            f"training did not succeed: {status}\n"
            f"{json.dumps(body)[:800]}"
        )

    res = body["results"][entity]

    if not res.get("model_id"):

        raise SystemExit(
            f"no model produced: "
            f"{res.get('status')} — "
            f"{res.get('error')}"
        )

    _log(
        f"  model_id={res['model_id']} "
        f"promotable={res.get('promotable')}"
    )

    calibration = res.get("calibration") or {}
    backtests = res.get("backtests") or {}

    backtest_statuses = {
        k: v.get("status")
        for k, v in backtests.items()
    }

    _log(
        f"  calibration={calibration.get('status')} "
        f"backtests={backtest_statuses}"
    )

    for w in res.get("warnings", [])[:3]:
        _log(f"  warning: {w}")

    return res


# --------------------------------------------------------------------------
# Step 3: Promote
# --------------------------------------------------------------------------

def promote(
    base: str,
    entity: str,
    model_id: str
) -> None:

    site, kdvr, chan = _parts(entity)

    url = (
        f"{base}/v1/sites/{site}/entities/"
        f"{kdvr}/{chan}/models/{model_id}/promote"
    )

    r = requests.post(
        url,

        json={
            "promoted_by": "run_experiment"
        },

        timeout=120
    )

    if r.status_code >= 400:

        raise SystemExit(
            f"promotion blocked: {r.text[:500]}"
        )

    _log(f"promoted {model_id}")


# --------------------------------------------------------------------------
# Step 4: Clear previous decisions
# --------------------------------------------------------------------------

def clear_previous(
    db: Path,
    entity: str,
    base: str
) -> None:

    """
    Remove earlier decisions for this entity.

    Scoring is idempotent on (entity_id, decision_time).

    Without clearing previous results, requests may return decisions
    produced by an earlier experiment.
    """

    response = requests.post(
        f"{base}/v1/sessions/reset",

        json={
            "entity_id": entity
        },

        timeout=60
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
                    (entity,)
                )

    finally:
        con.close()

    safe = entity.replace(":", "_")

    art = Path(C.SCORE_ARTIFACT_DIR) / safe

    if art.exists():

        for p in sorted(
            art.rglob("*"),
            reverse=True
        ):

            p.chmod(0o700)

            if p.is_file():
                p.unlink()
            else:
                p.rmdir()

        art.rmdir()

    _log("cleared previous decisions for this entity")


# --------------------------------------------------------------------------
# Step 5: Score all observations
#
# NEW:
# Capture action.urgency from each API response.
# --------------------------------------------------------------------------

def score_all(
    base: str,
    entity: str,
    data: Path,
    window: int,
    params: dict
) -> dict:

    df = load_telemetry(data)

    df = df[
        df.entity_id == entity
    ].reset_index(drop=True)

    if len(df) < window:

        raise SystemExit(
            f"need at least {window} records, "
            f"got {len(df)}"
        )

    site, kdvr, chan = _parts(entity)

    url = (
        f"{base}/v1/sites/{site}/entities/"
        f"{kdvr}/{chan}/score"
    )

    session = requests.Session()

    n = 0
    t0 = time.perf_counter()

    # NEW: request_id -> urgency
    urgency_map = {}

    missing_request_ids = 0
    missing_urgency = 0

    try:

        # Window ends at i.
        # First scoreable decision: i = window - 1.

        for i in range(window - 1, len(df)):

            records = to_records(
                df.iloc[
                    i - window + 1: i + 1
                ]
            )

            payload = {
                "records": records,
                "write_to_db": True,
                "client": "run_experiment"
            }

            if params:
                payload["params"] = params

            resp = session.post(
                url,
                json=payload,
                timeout=180
            )

            resp.raise_for_status()

            # ----------------------------------------------------------
            # NEW: Extract urgency from API response
            # ----------------------------------------------------------

            result = resp.json()

            request_id = result.get("request_id")

            action = result.get("action") or {}

            urgency = action.get("urgency")

            if request_id is None:
                missing_request_ids += 1

            else:
                urgency_map[str(request_id)] = urgency

            if urgency is None:
                missing_urgency += 1

            n += 1

            if n % 500 == 0:

                _log(
                    f"  {n} decisions… "
                    f"urgency captured={len(urgency_map)}"
                )

    finally:
        session.close()

    elapsed = time.perf_counter() - t0

    _log(
        f"scored {n} decisions in {elapsed:.0f}s"
    )

    _log(
        f"urgency mapping: "
        f"{len(urgency_map)} request IDs"
    )

    if missing_request_ids:

        _log(
            f"WARNING: {missing_request_ids} responses "
            f"have no request_id"
        )

    if missing_urgency:

        _log(
            f"WARNING: {missing_urgency} responses "
            f"have no action.urgency"
        )

    return urgency_map


# --------------------------------------------------------------------------
# Step 6: Export CSV
#
# NEW:
# Add urgency column using request_id mapping.
# --------------------------------------------------------------------------

def export(
    db: Path,
    entity: str,
    out: Path,
    urgency_map: dict
) -> pd.DataFrame:

    """
    Export scoring_results with an additional urgency column.

    Existing database columns are preserved.

    urgency is extracted from the scoring API response.

    Matching key:
        scoring_results.request_id
        =
        scoring API response.request_id
    """

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

            params=(entity,)
        )

    finally:
        con.close()

    # ----------------------------------------------------------
    # NEW: Attach urgency
    # ----------------------------------------------------------

    if "request_id" not in frame.columns:

        raise SystemExit(
            "Cannot attach urgency: scoring_results "
            "does not contain request_id. "
            "A different matching key is required."
        )

    frame["urgency"] = (
        frame["request_id"]
        .astype("string")
        .map(urgency_map)
    )

    # ----------------------------------------------------------
    # Validate mapping
    # ----------------------------------------------------------

    matched = int(
        frame["urgency"].notna().sum()
    )

    total = len(frame)

    missing = total - matched

    _log(
        f"urgency matched: {matched}/{total}"
    )

    if missing > 0:

        _log(
            f"WARNING: {missing} rows have "
            f"missing urgency"
        )

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    frame.to_csv(
        out,
        index=False
    )

    _log(
        f"exported {len(frame)} rows x "
        f"{len(frame.columns)} cols -> {out}"
    )

    return frame


# --------------------------------------------------------------------------
# Step 7: Summary
#
# NEW:
# Display urgency distribution.
# --------------------------------------------------------------------------

def summarise(frame: pd.DataFrame) -> None:

    if frame.empty:
        return

    print("\n" + "=" * 62)

    print(f"{'KẾT QUẢ':<30}")

    print("=" * 62)

    print(
        f"  decisions            {len(frame)}"
    )

    print(
        f"  window               "
        f"{frame.decision_time.min()} "
        f"-> {frame.decision_time.max()}"
    )

    print(
        f"  model_id             "
        f"{frame.model_id.iloc[0]}"
    )

    # ----------------------------------------------------------
    # Alert state
    # ----------------------------------------------------------

    counts = (
        frame.alert_state
        .value_counts()
        .to_dict()
    )

    print(
        f"  alert_state          {counts}"
    )

    # ----------------------------------------------------------
    # NEW: Urgency distribution
    # ----------------------------------------------------------

    urgency_counts = (
        frame["urgency"]
        .fillna("MISSING")
        .value_counts()
        .to_dict()
    )

    print(
        f"  urgency              {urgency_counts}"
    )

    # ----------------------------------------------------------
    # Orders
    # ----------------------------------------------------------

    orders = frame[
        frame.quantity > 0
    ]

    print(
        f"  orders / carts       "
        f"{len(orders)} / "
        f"{int(orders.quantity.sum())}"
    )

    # ----------------------------------------------------------
    # Simulated surplus
    # ----------------------------------------------------------

    surplus = (
        frame.simulated_cart
        - frame.current_cart
    )

    print(
        f"  simulated surplus    "
        f"median={surplus.median():.1f} "
        f"mean={surplus.mean():.2f} "
        f"(present in {(surplus > 0).mean():.0%})"
    )

    # ----------------------------------------------------------
    # Risk distribution
    # ----------------------------------------------------------

    print(
        f"  risk  "
        f"median={frame.shortage_risk.median():.4f} "
        f"q90={frame.shortage_risk.quantile(0.9):.4f} "
        f"max={frame.shortage_risk.max():.4f}"
    )

    print("=" * 62)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:

    p = argparse.ArgumentParser(
        description=(
            "Parameterised train + score run, "
            "exported as CSV with urgency."
        ),

        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    # ------------------------------------------------------------------
    # General arguments
    # ------------------------------------------------------------------

    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8100"
    )

    p.add_argument(
        "--entity",
        default="163:215:0",
        help='"siteKey:KDVR:ChanNo"'
    )

    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/experiment.csv")
    )

    p.add_argument(
        "--name",
        default=None,
        help="upload name; defaults to CSV stem"
    )

    p.add_argument(
        "--timezone",
        default="Asia/Ho_Chi_Minh"
    )

    p.add_argument(
        "--window",
        type=int,
        default=HISTORY_WINDOW
    )

    p.add_argument(
        "--train-timeout",
        type=int,
        default=900
    )

    p.add_argument(
        "--skip-training",
        action="store_true",
        help="score with currently promoted model"
    )

    p.add_argument(
        "--keep-previous",
        action="store_true",
        help="do not delete earlier decisions"
    )

    # ------------------------------------------------------------------
    # Operational parameters
    # ------------------------------------------------------------------

    g = p.add_argument_group(
        "operational parameters (frozen into the model)"
    )

    g.add_argument(
        "--bay-capacity",
        type=int,
        default=60
    )

    g.add_argument(
        "--red-threshold",
        type=int,
        default=15
    )

    g.add_argument(
        "--amber-threshold",
        type=int,
        default=20
    )

    g.add_argument(
        "--target-stock",
        type=int,
        default=28
    )

    g.add_argument(
        "--cooldown-min",
        type=float,
        default=30.0
    )

    g.add_argument(
        "--lead-time-min",
        type=float,
        default=10.0
    )

    g.add_argument(
        "--max-sequence-gap-min",
        type=float,
        default=15.0
    )

    g.add_argument(
        "--quantity-rule",
        choices=["point", "risk"],
        default=None
    )

    g.add_argument(
        "--min-carts-per-trip",
        type=int,
        default=None
    )

    g.add_argument(
        "--max-carts-per-trip",
        type=int,
        default=None
    )

    g.add_argument(
        "--reserve-per-day",
        type=int,
        default=None
    )

    # ------------------------------------------------------------------
    # Risk band overrides
    # ------------------------------------------------------------------

    r = p.add_argument_group(
        "per-call band overrides — diagnostics only. "
        "Bands are normally clustered from the entity's "
        "own risk at training time (K-Means, K=3)."
    )

    r.add_argument(
        "--override-risk-threshold",
        type=float,
        default=None
    )

    r.add_argument(
        "--override-critical-risk-threshold",
        type=float,
        default=None
    )

    # ------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------

    a = p.parse_args()

    a.name = a.name or a.out.stem

    if not a.data.exists():

        raise SystemExit(
            f"dataset not found: {a.data}"
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    health_response = requests.get(
        f"{a.base_url}/health",
        timeout=30
    )

    health_response.raise_for_status()

    health = health_response.json()

    _log(
        f"server {health['status']} "
        f"champion={health.get('champion_model_id')}"
    )

    # ------------------------------------------------------------------
    # Upload -> Train -> Promote
    # ------------------------------------------------------------------

    result = {}

    if a.skip_training:

        _log(
            "skipping training; "
            "using the promoted model"
        )

    else:

        upload_id = upload(
            a.base_url,
            a.data,
            a.entity,
            a
        )

        result = train(
            a.base_url,
            upload_id,
            a.entity,
            a.train_timeout
        )

        promote(
            a.base_url,
            a.entity,
            result["model_id"]
        )

    # ------------------------------------------------------------------
    # Clear previous decisions
    # ------------------------------------------------------------------

    if not a.keep_previous:

        clear_previous(
            Path(C.DB_PATH),
            a.entity,
            a.base_url
        )

    # ------------------------------------------------------------------
    # Request-time overrides
    # ------------------------------------------------------------------

    params = {
        k: v
        for k, v in (
            (
                "risk_threshold",
                a.override_risk_threshold
            ),
            (
                "critical_risk_threshold",
                a.override_critical_risk_threshold
            ),
        )
        if v is not None
    }

    if params:

        _log(
            f"request-time overrides: {params}"
        )

    # ------------------------------------------------------------------
    # Score and capture urgency
    # ------------------------------------------------------------------

    urgency_map = score_all(
        a.base_url,
        a.entity,
        a.data,
        a.window,
        params
    )

    # ------------------------------------------------------------------
    # Export CSV with urgency
    # ------------------------------------------------------------------

    frame = export(
        Path(C.DB_PATH),
        a.entity,
        a.out,
        urgency_map
    )

    # ------------------------------------------------------------------
    # Write configuration sidecar
    # ------------------------------------------------------------------

    sidecar = a.out.with_suffix(
        ".config.json"
    )

    config_data = {

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        ),

        "entity_id": a.entity,

        "dataset": str(a.data),

        "history_window": a.window,

        "entity_config_sent": _entity_config(a),

        "entity_defaults_sent": {
            "cooldown_min": a.cooldown_min,
            "lead_time_min": a.lead_time_min,
            "max_sequence_gap_min": a.max_sequence_gap_min
        },

        "request_overrides": params,

        "alert_bands": result.get(
            "alert_bands"
        ),

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

        "effective_config": result.get(
            "effective_config"
        ),

        "n_decisions": int(
            len(frame)
        ),

        # NEW: urgency summary
        "urgency_distribution": (
            frame["urgency"]
            .fillna("MISSING")
            .value_counts()
            .to_dict()
        ),

        "urgency_matched": int(
            frame["urgency"].notna().sum()
        ),

        "urgency_missing": int(
            frame["urgency"].isna().sum()
        )
    }

    sidecar.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sidecar.write_text(
        json.dumps(
            config_data,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    _log(
        f"config sidecar -> {sidecar}"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summarise(frame)


if __name__ == "__main__":
    main()