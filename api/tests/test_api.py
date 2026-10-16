"""API tests using FastAPI's TestClient (no running server required).

Run with:  python -m pytest api/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from server.app import app  # noqa: E402
from server.settings import BUNDLED_DATA, REQUEST_LOG, SCORE_LOG  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def telemetry():
    df = pd.read_csv(BUNDLED_DATA, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df["DVRDate"] = pd.to_datetime(df["DVRDate"])
    df["next_DVRDate"] = pd.to_datetime(df["next_DVRDate"])
    return df.sort_values("DVRDate").reset_index(drop=True)


def _records(df):
    return [
        {
            "DVRDate": r["DVRDate"].isoformat(),
            "next_DVRDate": r["next_DVRDate"].isoformat(),
            "Count": float(r["Count"]),
            "Count_IN": float(r["Count_IN"]),
            "Count_OUT": float(r["Count_OUT"]),
            "siteKey": int(r["siteKey"]),
        }
        for _, r in df.iterrows()
    ]


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
def test_health_reports_a_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("cart-forecast-")


def test_model_info_serves_the_audited_policy(client):
    body = client.get("/model/info").json()
    # The audited re-optimised policy must be served, not the pre-audit one.
    assert body["default_policy"]["risk_threshold"] == 0.55
    assert body["default_policy"]["cooldown_min"] == 30.0
    assert body["default_inventory"]["consumption_model"] == "unmet_demand"
    assert "simulator outputs" in body["caveat"]


def test_every_response_carries_a_request_id(client):
    r = client.get("/health")
    assert r.headers.get("X-Request-ID", "").startswith("req-")
    assert float(r.headers["X-Response-Time-ms"]) >= 0


# --------------------------------------------------------------------------
# /score
# --------------------------------------------------------------------------
def test_score_returns_forecast_risk_and_action(client, telemetry):
    payload = {
        "site_key": 163,
        "history": _records(telemetry.iloc[200:225]),
        "session_id": "test-score",
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["decision_time"] == telemetry.iloc[224]["next_DVRDate"].isoformat()
    assert b["current_cart"] == float(telemetry.iloc[224]["Count"])
    f = b["forecast"]
    for k in ("cart_at_20min", "cart_at_40min", "min_within_20min", "min_within_40min"):
        assert 0 <= f[k] <= 70
    # The lower bound must sit below the point forecast.
    assert f["lower_bound_min_within_40min"] <= f["min_within_40min"]
    assert 0.0 <= b["shortage_risk"] <= 1.0
    assert b["alert_state"] in ("NORMAL", "WARNING", "CRITICAL")
    assert isinstance(b["action"]["should_replenish"], bool)
    assert b["action"]["reason"]


def test_score_rejects_short_history(client, telemetry):
    r = client.post("/score", json={"history": _records(telemetry.iloc[0:4])})
    assert r.status_code == 422
    assert "records" in r.json()["detail"].lower()


def test_score_rejects_malformed_records(client):
    r = client.post("/score", json={"history": [{"DVRDate": "x", "Count": 1}]})
    assert r.status_code == 422


def test_score_never_orders_twice_while_carts_are_in_transit(client, telemetry):
    client.post("/sessions/reset", json={"session_id": "test-pending"})
    ordered = []
    for i in range(240, 270):
        body = client.post(
            "/score",
            json={
                "history": _records(telemetry.iloc[i - 20 : i]),
                "session_id": "test-pending",
                "policy": {"risk_threshold": 0.05, "cooldown_min": 0.0},
            },
        ).json()
        if body["action"]["should_replenish"]:
            ordered.append(body)

    if ordered:
        ids = [o["action"]["order_id"] for o in ordered]
        assert len(ids) == len(set(ids))  # no duplicate order ids
        for o in ordered:
            dec = pd.Timestamp(o["action"]["decision_time"])
            eff = pd.Timestamp(o["action"]["expected_completion_time"])
            assert (eff - dec).total_seconds() / 60.0 >= 10.0  # lead time respected
            assert o["action"]["quantity"] <= 20  # trip capacity respected


def test_score_session_state_is_isolated_and_resettable(client, telemetry):
    payload = {
        "history": _records(telemetry.iloc[300:325]),
        "session_id": "test-isolated",
        "policy": {"risk_threshold": 0.01},
    }
    client.post("/sessions/reset", json={"session_id": "test-isolated"})
    client.post("/score", json=payload)
    snap = {s["session_id"]: s for s in client.get("/sessions").json()["sessions"]}
    assert "test-isolated" in snap
    assert snap["test-isolated"]["calls"] >= 1
    reset = client.post("/sessions/reset", json={"session_id": "test-isolated"}).json()
    assert reset["reset"] == 1
    snap2 = {s["session_id"] for s in client.get("/sessions").json()["sessions"]}
    assert "test-isolated" not in snap2


def test_score_is_idempotent_for_an_already_processed_record(client, telemetry):
    payload = {
        "history": _records(telemetry.iloc[400:425]),
        "session_id": "test-idem",
        "policy": {"risk_threshold": 0.01},
    }
    client.post("/sessions/reset", json={"session_id": "test-idem"})
    first = client.post("/score", json=payload).json()
    second = client.post("/score", json=payload).json()
    # Re-sending the same record must not place a second order.
    assert second["action"]["should_replenish"] is False
    assert "already processed" in second["action"]["reason"]
    assert second["pending_carts"] == first["pending_carts"]


# --------------------------------------------------------------------------
# /backtest
# --------------------------------------------------------------------------
def test_backtest_on_bundled_data(client):
    r = client.post(
        "/backtest",
        json={"use_bundled_data": True, "start": "2026-10-10 00:00",
              "end": "2026-10-12 23:59", "label": "test-run"},
    )
    assert r.status_code == 200, r.text
    b = r.json()
    m = b["metrics"]
    assert b["run_id"].startswith("bt-")
    assert m["baseline_events"] >= m["policy_events"]
    assert m["ledger_conserved"] is True
    assert 0 <= m["severity_reduction_pct"] <= 100
    assert b["forecast_accuracy"]["mae_min_within_40min"] > 0
    assert "not observed causal effects" in b["caveat"]
    assert Path(b["artifact_dir"]).exists()


def test_backtest_accepts_posted_records(client, telemetry):
    window = telemetry[
        (telemetry["DVRDate"] >= "2026-10-11 06:00")
        & (telemetry["DVRDate"] <= "2026-10-11 23:00")
    ]
    r = client.post("/backtest", json={"records": _records(window), "label": "posted"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["n_records"] == len(window)
    assert b["scored_origins"] <= b["n_records"]


def test_backtest_policy_overrides_change_the_outcome(client):
    base = {"use_bundled_data": True, "start": "2026-10-10 00:00", "end": "2026-10-12 23:59"}
    aggressive = client.post(
        "/backtest", json={**base, "policy": {"risk_threshold": 0.25}}
    ).json()
    cautious = client.post(
        "/backtest", json={**base, "policy": {"risk_threshold": 0.90}}
    ).json()
    assert (
        aggressive["metrics"]["n_replenishment_actions"]
        >= cautious["metrics"]["n_replenishment_actions"]
    )


def test_backtest_conservative_assumptions_reduce_the_benefit(client):
    base = {"use_bundled_data": True, "start": "2026-10-10 00:00", "end": "2026-10-12 23:59"}
    optimistic = client.post(
        "/backtest", json={**base, "inventory": {"consumption_model": "persistent"}}
    ).json()
    pessimistic = client.post(
        "/backtest",
        json={**base, "inventory": {"consumption_model": "turnover", "usage": "HIGH"}},
    ).json()
    assert (
        optimistic["metrics"]["severity_reduction_pct"]
        > pessimistic["metrics"]["severity_reduction_pct"]
    )


def test_backtest_rejects_a_window_that_is_too_short(client, telemetry):
    r = client.post("/backtest", json={"records": _records(telemetry.iloc[0:5])})
    assert r.status_code == 422


def test_backtest_run_artifacts_are_retrievable(client):
    run_id = client.post(
        "/backtest",
        json={"use_bundled_data": True, "start": "2026-10-11 00:00",
              "end": "2026-10-12 00:00", "label": "artifact-test"},
    ).json()["run_id"]
    r = client.get(f"/runs/{run_id}")
    assert r.status_code == 200
    files = r.json()["files"]
    for expected in ("decisions.csv", "order_book.csv", "series.csv", "response.json"):
        assert expected in files
    assert client.get("/runs/does-not-exist").status_code == 404


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def test_calls_are_logged_server_side(client, telemetry):
    client.post("/score", json={"history": _records(telemetry.iloc[500:525]),
                                "session_id": "test-logging", "client": "pytest"})
    assert REQUEST_LOG.exists() and SCORE_LOG.exists()

    entries = client.get("/logs/requests", params={"limit": 20}).json()["entries"]
    assert entries, "request log is empty"
    last = entries[-1]
    for key in ("timestamp", "request_id", "endpoint", "status", "latency_ms"):
        assert key in last
    assert last["latency_ms"] >= 0

    scores = client.get("/logs/scores", params={"limit": 10}).json()["entries"]
    assert scores
    for key in ("request_id", "decision_time", "shortage_risk", "should_replenish"):
        assert key in scores[-1]

    bts = client.get("/logs/backtests", params={"limit": 10}).json()["entries"]
    assert bts
    assert "run_id" in bts[-1] and "severity_reduction_pct" in bts[-1]


def test_request_log_is_valid_jsonl():
    lines = REQUEST_LOG.read_text(encoding="utf-8").strip().splitlines()[-25:]
    for ln in lines:
        json.loads(ln)  # must not raise


def test_rejected_calls_are_logged_with_the_error(client, telemetry):
    client.post("/score", json={"history": _records(telemetry.iloc[0:3]),
                                "client": "pytest-reject"})
    entries = client.get("/logs/requests", params={"limit": 30}).json()["entries"]
    rejected = [e for e in entries if e["status"] == "rejected"]
    assert rejected
    assert rejected[-1]["error"]
