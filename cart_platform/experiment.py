"""One parameterised experiment: upload -> train -> promote -> score -> CSV.

Runs the whole on-demand pipeline for a single entity with the operational
parameters you pass on the command line, then exports every decision as a CSV
with exactly the columns of the ``scoring_results`` table so the run can be
backtested offline.

Each run is self-contained: the parameters are frozen into the model, recorded
in the CSV, and written to a sidecar JSON next to it, so two experiments can
always be told apart afterwards.

Examples
--------
    # baseline
    python scripts/run_experiment.py --out outputs/base.csv

    # warn early, reserve CRITICAL for a genuinely high probability
    python scripts/run_experiment.py \\
        --risk-threshold 0.15 --critical-risk-threshold 0.90 \\
        --out outputs/risk015_crit090.csv

    # score with an already promoted model, no retraining
    python scripts/run_experiment.py --skip-training --out outputs/rescore.csv
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
def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _entity_config(a) -> dict:
    """Only the parameters the caller actually supplied are sent.

    Omitting one lets the platform's own resolution apply instead of this
    script inventing a value.
    """
    cfg = {
        "bay_capacity": a.bay_capacity,
        "red_threshold": a.red_threshold,
        "amber_threshold": a.amber_threshold,
        "target_stock": a.target_stock,
        "risk_threshold": a.risk_threshold,
        "critical_risk_threshold": a.critical_risk_threshold,
        "quantity_rule": a.quantity_rule,
        "min_carts_per_trip": a.min_carts_per_trip,
        "max_carts_per_trip": a.max_carts_per_trip,
        "reserve_per_day": a.reserve_per_day,
    }
    return {k: v for k, v in cfg.items() if v is not None}


def upload(base: str, data: Path, entity: str, a) -> str:
    cfg = {
        "name": a.name,
        "timezone": a.timezone,
        "entity_defaults": {k: v for k, v in (
            ("max_sequence_gap_min", a.max_sequence_gap_min),
            ("cooldown_min", a.cooldown_min),
            ("lead_time_min", a.lead_time_min),
        ) if v is not None},
        "entity_overrides": {entity: _entity_config(a)},
        "auto_start_training": False,
    }
    _log(f"upload {data.name} — overrides: {cfg['entity_overrides'][entity]}")
    with open(data, "rb") as fh:
        r = requests.post(f"{base}/v1/datasets/upload",
                          files={"file": (data.name, fh, "text/csv")},
                          data={"config": json.dumps(cfg)}, timeout=600)
    r.raise_for_status()
    body = r.json()
    upload_id = body["upload_id"]
    if body.get("duplicate_of"):
        _log(f"  (identical upload already registered: {body['duplicate_of']})")
    for w in body.get("warnings", [])[:3]:
        _log(f"  warning: {w}")
    _log(f"  upload_id={upload_id}")
    return upload_id


def train(base: str, upload_id: str, entity: str, timeout_s: int) -> dict:
    r = requests.post(f"{base}/v1/training/jobs", timeout=60,
                      json={"upload_id": upload_id, "entity_ids": [entity],
                            "strategy": "per_entity"})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    _log(f"training job {job_id}")

    deadline = time.time() + timeout_s
    status, body = "running", {}
    while time.time() < deadline:
        time.sleep(5)
        body = requests.get(f"{base}/v1/training/jobs/{job_id}", timeout=60).json()
        status = body["status"]
        if status in ("succeeded", "completed", "failed"):
            break
    if status not in ("succeeded", "completed"):
        raise SystemExit(f"training did not succeed: {status}\n{json.dumps(body)[:800]}")

    res = body["results"][entity]
    if not res.get("model_id"):
        raise SystemExit(f"no model produced: {res.get('status')} — {res.get('error')}")
    _log(f"  model_id={res['model_id']}  promotable={res.get('promotable')}")
    _log(f"  calibration={res['calibration']['status']}  "
         f"backtests={ {k: v['status'] for k, v in res['backtests'].items()} }")
    for w in res.get("warnings", [])[:3]:
        _log(f"  warning: {w}")
    return res


def promote(base: str, entity: str, model_id: str) -> None:
    kdvr, chan = entity.split(":")[-2:]
    r = requests.post(
        f"{base}/v1/entities/{kdvr}/{chan}/models/{model_id}/promote",
        json={"promoted_by": "run_experiment"}, timeout=120)
    if r.status_code >= 400:
        raise SystemExit(f"promotion blocked: {r.text[:500]}")
    _log(f"promoted {model_id}")


def clear_previous(db: Path, entity: str, base: str) -> None:
    """Remove this entity's earlier decisions so the run starts clean.

    Scoring is idempotent on (entity_id, decision_time): without this every
    call would be answered from the previous run's artifact.
    """
    requests.post(f"{base}/v1/sessions/reset", json={"entity_id": entity}, timeout=60)
    con = sqlite3.connect(db)
    with con:
        for sql in (
            "DELETE FROM scoring_results WHERE entity_id=?",
            "DELETE FROM scoring_artifacts WHERE entity_id=?",
            "DELETE FROM replenishment_orders WHERE entity_id=?",
            "DELETE FROM session_state WHERE entity_id=?",
        ):
            con.execute(sql, (entity,))
    con.close()
    safe = entity.replace(":", "_")
    art = Path(C.SCORE_ARTIFACT_DIR) / safe
    if art.exists():
        for p in sorted(art.rglob("*"), reverse=True):
            p.chmod(0o700)
            p.unlink() if p.is_file() else p.rmdir()
        art.rmdir()
    _log("cleared previous decisions for this entity")


def score_all(base: str, entity: str, data: Path, window: int, params: dict) -> int:
    df = load_telemetry(data)
    df = df[df.entity_id == entity].reset_index(drop=True)
    if len(df) <= window:
        raise SystemExit(f"need more than {window} records, got {len(df)}")
    kdvr, chan = entity.split(":")[-2:]
    url = f"{base}/v1/entities/{kdvr}/{chan}/score"

    # The window ENDS at i, so the first scoreable decision is at i = window-1.
    s, n, t0 = requests.Session(), 0, time.perf_counter()
    for i in range(window - 1, len(df)):
        payload = {"records": to_records(df.iloc[i - window + 1: i + 1]),
                   "write_to_db": True, "client": "run_experiment"}
        if params:
            payload["params"] = params
        resp = s.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        n += 1
        if n % 500 == 0:
            _log(f"  {n} decisions…")
    _log(f"scored {n} decisions in {time.perf_counter() - t0:.0f}s")
    return n


def export(db: Path, entity: str, out: Path) -> pd.DataFrame:
    """Dump scoring_results verbatim — same columns, same order."""
    con = sqlite3.connect(db)
    frame = pd.read_sql(
        "SELECT * FROM scoring_results WHERE entity_id=? ORDER BY decision_time",
        con, params=(entity,))
    con.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    _log(f"exported {len(frame)} rows x {len(frame.columns)} cols -> {out}")
    return frame


def summarise(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    print("\n" + "=" * 62)
    print(f"{'KẾT QUẢ':<30}")
    print("=" * 62)
    print(f"  decisions            {len(frame)}")
    print(f"  window               {frame.decision_time.min()} -> {frame.decision_time.max()}")
    print(f"  model_id             {frame.model_id.iloc[0]}")
    counts = frame.alert_state.value_counts().to_dict()
    print(f"  alert_state          {counts}")
    print(f"  new_predictive_alert {int(frame.new_predictive_alert.sum())}")
    print(f"  reactive_critical    {int(frame.reactive_critical.sum())}")
    orders = frame[frame.should_replenish == 1]
    print(f"  orders / carts       {len(orders)} / {int(orders.quantity.sum())}")
    if not orders.empty:
        print(f"  urgency              {orders.urgency.value_counts().to_dict()}")
    surplus = frame.simulated_cart - frame.current_cart
    print(f"  simulated surplus    median={surplus.median():.1f} "
          f"mean={surplus.mean():.2f} (present in {(surplus > 0).mean():.0%})")
    print("=" * 62)


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Parameterised train + score run, exported as CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8100")
    p.add_argument("--entity", default="215:0")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=Path("outputs/experiment.csv"))
    p.add_argument("--name", default=None, help="upload name; defaults to the CSV stem")
    p.add_argument("--timezone", default="Asia/Ho_Chi_Minh")
    p.add_argument("--window", type=int, default=HISTORY_WINDOW)
    p.add_argument("--train-timeout", type=int, default=900)
    p.add_argument("--skip-training", action="store_true",
                   help="score with the currently promoted model")
    p.add_argument("--keep-previous", action="store_true",
                   help="do not delete this entity's earlier decisions")

    g = p.add_argument_group("operational parameters (frozen into the model)")
    g.add_argument("--bay-capacity", type=int, default=60)
    g.add_argument("--red-threshold", type=int, default=15)
    g.add_argument("--amber-threshold", type=int, default=20)
    g.add_argument("--target-stock", type=int, default=28)
    g.add_argument("--risk-threshold", type=float, default=0.55,
                   help="opens the WARNING band AND gates replenishment")
    g.add_argument("--critical-risk-threshold", type=float, default=None,
                   help="opens CRITICAL; omit to derive risk_threshold + 0.25")
    g.add_argument("--cooldown-min", type=float, default=30.0)
    g.add_argument("--lead-time-min", type=float, default=10.0)
    g.add_argument("--max-sequence-gap-min", type=float, default=15.0)
    g.add_argument("--quantity-rule", choices=["point", "risk"], default=None)
    g.add_argument("--min-carts-per-trip", type=int, default=None)
    g.add_argument("--max-carts-per-trip", type=int, default=None)
    g.add_argument("--reserve-per-day", type=int, default=None)

    r = p.add_argument_group("request-time overrides (NOT frozen; applied per call)")
    r.add_argument("--override-risk-threshold", type=float, default=None)
    r.add_argument("--override-critical-risk-threshold", type=float, default=None)

    a = p.parse_args()
    a.name = a.name or a.out.stem
    if not a.data.exists():
        raise SystemExit(f"dataset not found: {a.data}")

    health = requests.get(f"{a.base_url}/health", timeout=30).json()
    _log(f"server {health['status']}  champion={health.get('champion_model_id')}")

    result = {}
    if a.skip_training:
        _log("skipping training; using the promoted model")
    else:
        upload_id = upload(a.base_url, a.data, a.entity, a)
        result = train(a.base_url, upload_id, a.entity, a.train_timeout)
        promote(a.base_url, a.entity, result["model_id"])

    if not a.keep_previous:
        clear_previous(Path(C.DB_PATH), a.entity, a.base_url)

    params = {k: v for k, v in (
        ("risk_threshold", a.override_risk_threshold),
        ("critical_risk_threshold", a.override_critical_risk_threshold),
    ) if v is not None}
    if params:
        _log(f"request-time overrides: {params}")

    score_all(a.base_url, a.entity, a.data, a.window, params)
    frame = export(Path(C.DB_PATH), a.entity, a.out)

    sidecar = a.out.with_suffix(".config.json")
    sidecar.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entity_id": a.entity,
        "dataset": str(a.data),
        "history_window": a.window,
        "entity_config_sent": _entity_config(a),
        "entity_defaults_sent": {"cooldown_min": a.cooldown_min,
                                 "lead_time_min": a.lead_time_min,
                                 "max_sequence_gap_min": a.max_sequence_gap_min},
        "request_overrides": params,
        "model_id": result.get("model_id") or (
            frame.model_id.iloc[0] if not frame.empty else None),
        "config_hash": result.get("config_hash"),
        "effective_config": result.get("effective_config"),
        "n_decisions": int(len(frame)),
    }, indent=2), encoding="utf-8")
    _log(f"config sidecar -> {sidecar}")
    summarise(frame)


if __name__ == "__main__":
    main()
