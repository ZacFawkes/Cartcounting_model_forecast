"""Platform tests: entity isolation, config resolution, D10 split, durability.

Run with:  python -m pytest cart_platform/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))
import settings  # noqa: E402,F401  — extends sys.path to the modelling pipeline


@pytest.fixture(scope="module")
def tmp_db(tmp_path_factory):
    """Isolated registry per test module."""
    db = tmp_path_factory.mktemp("registry") / "test.db"
    import settings

    settings.DB_PATH = db
    import registry.database as database

    database.DB_PATH = db
    from registry.database import init_db

    init_db(db)
    return db


@pytest.fixture(scope="module")
def client(tmp_db):
    from fastapi.testclient import TestClient
    from server.app import app

    with TestClient(app) as c:
        c.post("/v1/registry/bootstrap")
        yield c


@pytest.fixture(scope="module")
def telemetry():
    from settings import BUNDLED_TELEMETRY

    df = pd.read_csv(BUNDLED_TELEMETRY, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df["DVRDate"] = pd.to_datetime(df["DVRDate"])
    df["next_DVRDate"] = pd.to_datetime(df["next_DVRDate"])
    return df.sort_values("DVRDate").reset_index(drop=True)


def _rec(df):
    return [
        {
            "DVRDate": r.DVRDate.isoformat(),
            "next_DVRDate": r.next_DVRDate.isoformat(),
            "Count": float(r.Count),
            "Count_IN": float(r.Count_IN),
            "Count_OUT": float(r.Count_OUT),
        }
        for r in df.itertuples()
    ]


AUDITED = {
    "red_threshold": 15, "amber_threshold": 20, "target_stock": 28,
    "cooldown_min": 30.0, "lead_time_min": 10.0,
}
ENTITY = "163:215:0"   # bundled telemetry carries no bay column


# --------------------------------------------------------------------------
# 1. Entity grain and isolation
# --------------------------------------------------------------------------
def test_entity_id_is_site_kdvr_channel_and_bay():
    from data_loader import entity_id

    df = pd.DataFrame({"KDVR": [215, 215, 216], "ChanNo": [0, 1, 0],
                       "siteKey": [163, 163, 163],
                       "BayID": ["06053B44-0A47-42FC-BE3E-F88CB85F99D8",
                                 "bay-b", "bay-c"]})
    assert list(entity_id(df)) == ["215:0", "215:1", "216:0"]
    assert list(entity_id(df, include_site=True)) == [
        "163:215:0", "163:215:1", "163:216:0"]
    # A bay key is normalised so it can travel inside an id and a URL path.
    assert list(entity_id(df, include_site=True, include_bay=True)) == [
        "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8",
        "163:215:1:bay_b", "163:216:0:bay_c"]


def test_entity_id_requires_the_bay_column_when_bay_scoped():
    from data_loader import entity_id

    df = pd.DataFrame({"KDVR": [215], "ChanNo": [0], "siteKey": [163]})
    with pytest.raises(ValueError, match="BayID"):
        entity_id(df, include_site=True, include_bay=True)


def test_blocks_never_span_two_entities():
    """The P0 guarantee: a rolling window cannot mix two camera streams."""
    from data_loader import prepare
    from feature_engineering import build_features

    base = pd.Timestamp("2026-01-01 08:00")
    rows = []
    for kdvr, chan, level in ((215, 0, 40), (216, 0, 5)):
        for i in range(14):
            s = base + pd.Timedelta(minutes=20 / 3 * i)
            rows.append({
                "CountID": len(rows), "DVRDate": s,
                "next_DVRDate": s + pd.Timedelta(minutes=20 / 3),
                "KDVR": kdvr, "ChanNo": chan, "CatID": 4000, "AreaID": 69,
                "Count": level, "CNN_Index": len(rows), "siteKey": 163,
                "Count_IN": 3, "Count_OUT": 2,
            })
    prepared = prepare(pd.DataFrame(rows), include_site_in_entity=True)
    assert prepared["entity_id"].nunique() == 2
    # Identical timestamps in both streams must land in different blocks.
    assert prepared.groupby("block_id")["entity_id"].nunique().max() == 1

    feats = build_features(prepared)
    for eid, level in (("163:215:0", 40.0), ("163:216:0", 5.0)):
        sub = feats[feats["entity_id"] == eid].dropna(subset=["cart_mean_9"])
        assert (sub["cart_mean_9"] == level).all(), f"{eid} leaked across entities"
        # First rows of each stream have no window — never borrowed from the other.
        head = feats[feats["entity_id"] == eid].iloc[0]
        assert pd.isna(head["cart_lag_1"])


def test_entity_summary_counts_streams():
    from data_loader import load_prepared
    from registry.entities import discover_entities
    from settings import BUNDLED_TELEMETRY

    prepared = load_prepared(BUNDLED_TELEMETRY)
    summary = discover_entities(prepared)
    assert len(summary) == prepared["entity_id"].nunique()
    assert set(summary.columns) >= {"entity_id", "kdvr", "chan_no", "site_key", "n_days"}


# --------------------------------------------------------------------------
# 2. Registry
# --------------------------------------------------------------------------
def test_bootstrap_registers_entities_and_model(client):
    h = client.get("/health").json()
    assert h["model_loaded"] is True
    assert h["n_entities"] >= 1
    ents = client.get("/v1/entities").json()["entities"]
    e = next(x for x in ents if x["entity_id"] == ENTITY)
    assert (e["kdvr"], e["chan_no"], e["site_key"]) == (215, 0, 163)
    assert e["threshold_source"] == "derived"
    assert e["red_threshold"] < e["amber_threshold"] < e["target_stock"] <= e["bay_capacity"]


def test_unknown_entity_is_rejected_not_defaulted(client, telemetry):
    r = client.post("/v1/score", json={"entity_id": "163:999:9",
                                       "records": _rec(telemetry.iloc[100:120])})
    assert r.status_code == 404
    assert "not registered" in r.json()["detail"]


def test_entity_can_be_updated_and_disabled(client, telemetry):
    client.put("/v1/entities", json={"entity_id": ENTITY, "target_stock": 30,
                                     "threshold_source": "manual_override"})
    e = client.get(f"/v1/entities/{ENTITY}").json()["entity"]
    assert e["target_stock"] == 30 and e["threshold_source"] == "manual_override"

    client.post(f"/v1/entities/{ENTITY}/status", json={"status": "disabled"})
    r = client.post("/v1/score", json={"entity_id": ENTITY,
                                       "records": _rec(telemetry.iloc[100:120])})
    assert r.status_code == 503
    client.post(f"/v1/entities/{ENTITY}/status", json={"status": "active"})


# --------------------------------------------------------------------------
# 3. Config resolution (request > entity/site > system)
# --------------------------------------------------------------------------
def test_three_level_resolution_and_sources(client):
    client.put("/v1/sites", json={"site_key": 163, "lead_time_min": 12.0})
    client.put("/v1/entities", json={"entity_id": ENTITY, "target_stock": 30})
    cfg = client.get(f"/v1/entities/{ENTITY}").json()["resolved_config"]

    assert cfg["target_stock"]["source"] == "entity_config"
    assert cfg["lead_time_min"]["value"] == 12.0
    assert cfg["lead_time_min"]["source"] == "site_config"
    assert cfg["usage"]["source"] == "system_default"


def test_request_override_wins_and_is_recorded(client, telemetry):
    body = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1800:1820]),
        "params": {**AUDITED, "target_stock": 99},
    }).json()
    cfg = body["resolved_config"]
    assert cfg["target_stock"]["value"] == 99
    assert cfg["target_stock"]["source"] == "request_override"
    assert cfg["usage"]["source"] == "system_default"


def test_omitted_parameter_falls_back_to_entity_default(client, telemetry):
    client.put("/v1/entities", json={"entity_id": ENTITY, "target_stock": 33})
    body = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1800:1820]),
    }).json()
    assert body["resolved_config"]["target_stock"]["value"] == 33
    assert body["resolved_config"]["target_stock"]["source"] == "entity_config"
    client.put("/v1/entities", json={"entity_id": ENTITY, "target_stock": 28})


def test_alert_bands_are_measured_not_configured(client, telemetry):
    """The bands come off the model, never out of entity_config."""
    body = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1800:1820]),
    }).json()
    cfg = body["resolved_config"]
    assert cfg["risk_threshold"]["source"] in ("model_kmeans", "system_default")
    assert cfg["critical_risk_threshold"]["source"] in (
        "model_kmeans", "system_default")

    # entity_config cannot hold them any more: the field is simply not part of
    # the contract, so writing it must not change what is served
    before = cfg["risk_threshold"]["value"]
    client.put("/v1/entities", json={"entity_id": ENTITY, "risk_threshold": 0.77})
    after = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1800:1820]),
    }).json()["resolved_config"]["risk_threshold"]
    assert after["value"] == before
    assert after["source"] != "entity_config"


def test_unconfigured_parameters_are_flagged(client):
    detail = client.get(f"/v1/entities/{ENTITY}").json()
    assert detail["unconfigured"], "system-default fallbacks must be listed"
    assert any("system defaults" in w for w in detail["warnings"])


# --------------------------------------------------------------------------
# 4. The RED/AMBER trap (decision D10)
# --------------------------------------------------------------------------
def test_feature_time_thresholds_are_frozen_and_reported(client):
    info = client.get("/v1/model/info").json()
    ft = info["feature_time_thresholds"]
    assert ft["red_model"] == 15 and ft["amber_model"] == 20
    assert "frozen" in ft["note"].lower()


def test_eval_threshold_override_warns_but_does_not_change_features(client, telemetry):
    recs = _rec(telemetry.iloc[1800:1820])
    base = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": recs, "params": {**AUDITED, "red_threshold": 15},
    }).json()
    shifted = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": recs, "params": {**AUDITED, "red_threshold": 25},
    }).json()

    assert shifted["resolved_config"]["red_model"]["value"] == 15
    assert shifted["resolved_config"]["red_model"]["source"] == "model_registry"
    assert any("red_threshold=25 differs from red_model=15" in w
               for w in shifted["warnings"]), shifted["warnings"]
    # Point forecasts are threshold-free, so they must be unchanged...
    assert base["forecast"]["min_within_40min"] == shifted["forecast"]["min_within_40min"]
    # ...while the evaluation-side risk does change.
    assert shifted["shortage_risk"] > base["shortage_risk"]


# --------------------------------------------------------------------------
# 5. Scoring contract
# --------------------------------------------------------------------------
def test_score_by_kdvr_and_channel(client, telemetry):
    r = client.post("/v1/score", json={
        "kdvr": 215, "chan_no": 0, "site_key": 163,
        "records": _rec(telemetry.iloc[1800:1820]), "params": AUDITED,
    })
    assert r.status_code == 200
    b = r.json()
    assert b["entity_id"] == ENTITY and b["kdvr"] == 215 and b["chan_no"] == 0
    assert 0.0 <= b["shortage_risk"] <= 1.0
    assert b["forecast"]["lower_bound_min_within_40min"] <= b["forecast"]["min_within_40min"]
    assert b["alert_state"] in ("NORMAL", "WARNING", "CRITICAL")


def test_score_requires_entity_identification(client, telemetry):
    r = client.post("/v1/score", json={"records": _rec(telemetry.iloc[100:120])})
    assert r.status_code == 422
    assert "entity_id" in r.json()["detail"]


def test_score_rejects_short_history(client, telemetry):
    r = client.post("/v1/score", json={"entity_id": ENTITY,
                                       "records": _rec(telemetry.iloc[0:5])})
    assert r.status_code == 422
    assert "at least" in r.json()["detail"]


def test_history_size_above_the_floor_does_not_change_the_answer(client, telemetry):
    out = {}
    for n in (12, 20, 40):
        client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
        b = client.post("/v1/score", json={
            "entity_id": ENTITY, "records": _rec(telemetry.iloc[1820 - n:1820]),
            "params": AUDITED,
        }).json()
        out[n] = (b["forecast"]["min_within_40min"], b["shortage_risk"],
                  b["action"]["quantity"])
    assert out[12] == out[20] == out[40]


# --------------------------------------------------------------------------
# 6. Durability: write-back, idempotency, restart recovery
# --------------------------------------------------------------------------
def test_score_is_written_back_to_the_database(client, telemetry):
    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    b = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1900:1920]),
        "params": AUDITED,
    }).json()
    rows = client.get("/v1/logs/scores", params={"entity_id": ENTITY, "limit": 10}).json()
    match = [r for r in rows["entries"] if r["decision_time"] == b["decision_time"]]
    assert match, "scoring result was not persisted"
    row = match[0]
    assert row["entity_id"] == ENTITY and row["kdvr"] == 215 and row["site_key"] == 163
    assert row["shortage_risk"] == b["shortage_risk"]
    assert row["model_id"]
    # The configuration behind a decision is traced through model_id rather
    # than copied onto every row.
    assert row["model_id"] == b["model_id"]


def test_replaying_a_processed_record_places_no_second_order(client, telemetry):
    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    payload = {"entity_id": ENTITY, "records": _rec(telemetry.iloc[1820:1840]),
               "params": {**AUDITED, "risk_threshold": 0.0, "target_stock": 55}}
    first = client.post("/v1/score", json=payload).json()
    second = client.post("/v1/score", json=payload).json()
    assert second["action"]["should_replenish"] is False
    assert "already processed" in second["action"]["reason"]
    assert second["pending_carts"] == first["pending_carts"]


def test_pending_orders_survive_a_server_restart(client, telemetry, tmp_db):
    """The order book lives in SQLite, so a restart cannot duplicate a retrieval."""
    from fastapi.testclient import TestClient

    from server.app import app

    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    before = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1840:1860]),
        "params": {**AUDITED, "risk_threshold": 0.0, "target_stock": 55},
    }).json()
    assert before["action"]["should_replenish"], "fixture expected an order"
    pending_before = before["pending_carts"]
    order_id = before["action"]["order_id"]

    # A brand-new app instance = a restarted process.
    with TestClient(app) as fresh:
        snap = fresh.get("/v1/sessions", params={"entity_id": ENTITY}).json()["sessions"]
        assert snap and snap[0]["pending_carts"] == pending_before
        orders = fresh.get("/v1/orders", params={"entity_id": ENTITY,
                                                 "status": "PENDING"}).json()["orders"]
        assert any(o["order_id"] == order_id for o in orders)
        # Scoring the very next record must net against the surviving order.
        nxt = fresh.post("/v1/score", json={
            "entity_id": ENTITY, "records": _rec(telemetry.iloc[1841:1861]),
            "params": {**AUDITED, "risk_threshold": 0.0, "target_stock": 55,
                       "cooldown_min": 0.0},
        }).json()
        assert nxt["pending_carts"] >= pending_before


def test_session_reset_cancels_pending_orders(client, telemetry):
    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1860:1880]),
        "params": {**AUDITED, "risk_threshold": 0.0, "target_stock": 55},
    })
    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    pending = client.get("/v1/orders", params={"entity_id": ENTITY,
                                               "status": "PENDING"}).json()["orders"]
    assert pending == []


def test_execution_log_records_what_staff_actually_did(client, telemetry):
    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    b = client.post("/v1/score", json={
        "entity_id": ENTITY, "records": _rec(telemetry.iloc[1880:1900]),
        "params": {**AUDITED, "risk_threshold": 0.0, "target_stock": 55},
    }).json()
    order_id = b["action"]["order_id"]
    assert order_id, b["action"]["reason"]
    done = client.post(f"/v1/orders/{order_id}/execute",
                       json={"executed_by": "tester", "actual_carts": 7,
                             "notes": "unit test"}).json()
    assert done["order"]["executed_by"] == "tester"
    assert done["order"]["actual_carts"] == 7
    assert client.post("/v1/orders/NOPE/execute",
                       json={"executed_by": "x", "actual_carts": 1}).status_code == 404


# --------------------------------------------------------------------------
# 7. Backtest
# --------------------------------------------------------------------------
def test_backtest_reproduces_the_audited_result(client):
    r = client.post("/v1/backtest", json={
        "entity_id": ENTITY, "start": "2026-10-10 00:00", "end": "2026-10-12 23:59",
        "mode": "replay", "params": AUDITED, "label": "audit-repro",
    })
    assert r.status_code == 200
    run = r.json()["runs"][0]
    m = run["metrics"]
    assert run["entity_id"] == ENTITY
    assert m["baseline_events"] == 13
    assert m["policy_events"] <= 3
    assert 80 <= m["event_reduction_pct"] <= 90
    assert 90 <= m["severity_reduction_pct"] <= 99
    assert m["ledger_conserved"] is True
    assert run["attribution_summary"]["traceable_prevention"] >= 1
    assert run["forecast_accuracy"]["mae_min_within_40min"] < 5


def test_backtest_by_site_covers_every_entity(client):
    r = client.post("/v1/backtest", json={
        "site_key": 163, "start": "2026-10-11 00:00", "end": "2026-10-12 00:00",
        "params": AUDITED,
    })
    assert r.status_code == 200
    body = r.json()
    registered = client.get("/v1/entities", params={"site_key": 163}).json()["entities"]
    assert body["n_entities"] == len(registered)


def test_backtest_assumptions_change_the_benefit(client):
    base = {"entity_id": ENTITY, "start": "2026-10-10 00:00", "end": "2026-10-12 23:59"}
    optimistic = client.post("/v1/backtest", json={
        **base, "params": {**AUDITED, "consumption_model": "persistent"}}).json()["runs"][0]
    pessimistic = client.post("/v1/backtest", json={
        **base, "params": {**AUDITED, "consumption_model": "turnover",
                           "usage": "HIGH"}}).json()["runs"][0]
    assert (optimistic["metrics"]["severity_reduction_pct"]
            > pessimistic["metrics"]["severity_reduction_pct"])
    assert "not observed causal effects" in optimistic["caveat"]


def test_backtest_artifacts_are_retrievable(client):
    run_id = client.post("/v1/backtest", json={
        "entity_id": ENTITY, "start": "2026-10-11 00:00", "end": "2026-10-12 00:00",
        "params": AUDITED, "label": "artifacts",
    }).json()["runs"][0]["run_id"]
    r = client.get(f"/v1/runs/{run_id}")
    assert r.status_code == 200
    files = r.json()["files"]
    for expected in ("decisions.csv", "order_book.csv", "series.csv", "response.json"):
        assert expected in files
    assert r.json()["summary"]["entity_id"] == ENTITY
    assert client.get("/v1/runs/missing").status_code == 404


def test_backtest_requires_a_target(client):
    assert client.post("/v1/backtest", json={"start": "2026-10-10"}).status_code == 422


# --------------------------------------------------------------------------
# 8. Logging keyed by entity
# --------------------------------------------------------------------------
def test_logs_are_filtered_by_entity(client, telemetry):
    client.post("/v1/score", json={"entity_id": ENTITY,
                                   "records": _rec(telemetry.iloc[1500:1520]),
                                   "params": AUDITED, "client": "pytest"})
    reqs = client.get("/v1/logs/requests",
                      params={"entity_id": ENTITY, "limit": 20}).json()["entries"]
    assert reqs and all(e["entity_id"] == ENTITY for e in reqs)
    last = reqs[0]
    for key in ("request_id", "endpoint", "status", "latency_ms", "site_key"):
        assert key in last

    bts = client.get("/v1/logs/backtests",
                     params={"entity_id": ENTITY, "limit": 5}).json()["entries"]
    assert bts and all(e["entity_id"] == ENTITY for e in bts)
    assert bts[0]["resolved_config"]


def test_rejected_calls_are_logged_with_the_error(client, telemetry):
    client.post("/v1/score", json={"entity_id": ENTITY,
                                   "records": _rec(telemetry.iloc[0:3]),
                                   "client": "pytest-reject"})
    entries = client.get("/v1/logs/requests",
                         params={"status": "rejected", "limit": 10}).json()["entries"]
    assert entries and entries[0]["error"]


def test_every_response_carries_trace_headers(client):
    r = client.get("/health")
    assert r.headers["X-Request-ID"].startswith("req-")
    assert float(r.headers["X-Response-Time-ms"]) >= 0
