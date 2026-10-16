"""Receiver API contract tests (spec §11.2, §11.3, §11.5)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

import settings  # noqa: E402,F401

from integrations.result_events import (  # noqa: E402
    BACKTEST_COMPLETED,
    CALIBRATION_COMPLETED,
    SCORING_RECORDED,
    TRAINING_COMPLETED,
)


def _event(event_id: str, event_type: str, payload: dict, **kw) -> dict:
    base = {
        "event_id": event_id,
        "event_type": event_type,
        "schema_version": 1,
        "occurred_at": "2026-10-12T14:00:00.000Z",
        "entity_id": "215:0",
        "site_key": "163",
        "upload_id": "upl_001",
        "job_id": "train_001",
        "model_id": "model_001",
        "payload": payload,
    }
    base.update(kw)
    return base


def _backtest(bt_type: str, event_id: str | None = None, status: str = "completed",
              **kw) -> dict:
    return _event(
        event_id or f"train_001:215:0:{bt_type}:{status}:v1",
        BACKTEST_COMPLETED,
        {"backtest_type": bt_type, "status": status,
         "metrics": {"mae": 2.9, "rmse": 4.9},
         "artifacts": [{"kind": "predictions_csv", "uri": "logs/x.csv"}],
         "caveat": "measured, not a benefit claim"},
        **kw,
    )


# --------------------------------------------------------------------------
# Health and schema
# --------------------------------------------------------------------------
def test_health_reports_sqlite_connectivity(receiver_client):
    body = receiver_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "cart-result-receiver"
    assert body["sqlite_ok"] is True
    assert "results_receiver.db" in body["database"]


def test_receiver_database_is_separate_from_platform_db(receiver_client):
    import settings as C

    assert Path(C.RECEIVER_DB_PATH) != Path(C.DB_PATH)
    assert receiver_client.get("/health").json()["database"] == str(C.RECEIVER_DB_PATH)


def test_openapi_documents_every_result_route(receiver_client):
    paths = receiver_client.get("/openapi.json").json()["paths"]
    for route in ("/v1/results/training", "/v1/results/calibration",
                  "/v1/results/backtest", "/v1/results/scoring",
                  "/v1/results/entities/{entity_id}", "/v1/results/events/{event_id}"):
        assert route in paths, route


# --------------------------------------------------------------------------
# Ingestion and idempotency (spec §11.2)
# --------------------------------------------------------------------------
def test_new_event_returns_201(receiver_client):
    r = receiver_client.post("/v1/results/training", json=_event(
        "t:new:v1", TRAINING_COMPLETED, {"status": "candidate_ready"}))
    assert r.status_code == 201
    assert r.json()["outcome"] == "created"
    assert r.json()["event_id"] == "t:new:v1"


def test_identical_replay_returns_200_and_one_row(receiver_client):
    ev = _event("t:replay:v1", TRAINING_COMPLETED, {"status": "candidate_ready"})
    first = receiver_client.post("/v1/results/training", json=ev)
    second = receiver_client.post("/v1/results/training", json=ev)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["outcome"] == "replay"
    rows = receiver_client.get("/v1/results/training",
                               params={"job_id": "train_001"}).json()["items"]
    assert len([r for r in rows if r["event_id"] == "t:replay:v1"]) == 1


def test_same_id_different_payload_returns_409(receiver_client):
    ev = _event("t:conflict:v1", TRAINING_COMPLETED, {"status": "candidate_ready"})
    assert receiver_client.post("/v1/results/training", json=ev).status_code == 201
    mutated = {**ev, "payload": {"status": "failed"}}
    r = receiver_client.post("/v1/results/training", json=mutated)
    assert r.status_code == 409
    assert "never overwritten" in r.json()["detail"]["message"]
    stored = receiver_client.get("/v1/results/events/t:conflict:v1").json()
    assert stored["payload"]["status"] == "candidate_ready"


def test_event_type_must_match_the_route(receiver_client):
    r = receiver_client.post("/v1/results/training",
                             json=_event("t:wrongroute:v1", BACKTEST_COMPLETED,
                                         {"backtest_type": "forecast_accuracy"}))
    assert r.status_code == 422
    assert "does not match this route" in r.json()["detail"]


def test_backtest_requires_a_type(receiver_client):
    r = receiver_client.post("/v1/results/backtest",
                             json=_event("bt:notype:v1", BACKTEST_COMPLETED,
                                         {"status": "completed"}))
    assert r.status_code == 422
    assert "backtest_type is required" in r.json()["detail"]


def test_validation_error_on_malformed_event(receiver_client):
    assert receiver_client.post("/v1/results/training",
                                json={"event_type": "training.completed"}).status_code == 422


# --------------------------------------------------------------------------
# The three backtest types are independently queryable
# --------------------------------------------------------------------------
def test_three_backtest_types_stored_separately(receiver_client):
    for t in ("forecast_accuracy", "policy_replay", "closed_loop_robustness"):
        assert receiver_client.post("/v1/results/backtest",
                                    json=_backtest(t)).status_code == 201
    all_rows = receiver_client.get("/v1/results/backtest",
                                   params={"job_id": "train_001"}).json()
    assert all_rows["total"] >= 3
    for t in ("forecast_accuracy", "policy_replay", "closed_loop_robustness"):
        filtered = receiver_client.get("/v1/results/backtest",
                                       params={"backtest_type": t,
                                               "job_id": "train_001"}).json()
        assert filtered["total"] >= 1
        assert all(i["backtest_type"] == t for i in filtered["items"])


def test_failed_backtest_is_stored_with_its_real_status(receiver_client):
    ev = _backtest("policy_replay", "bt:insufficient:v1", status="insufficient_data")
    ev["payload"]["reason"] = "holdout has no scoreable rows"
    ev["payload"]["metrics"] = {}
    assert receiver_client.post("/v1/results/backtest", json=ev).status_code == 201
    row = receiver_client.get("/v1/results/backtest",
                              params={"status": "insufficient_data"}).json()["items"][0]
    assert row["status"] == "insufficient_data"
    assert "no scoreable rows" in row["reason"]


# --------------------------------------------------------------------------
# Filtering, pagination and aggregation
# --------------------------------------------------------------------------
def test_entity_filtering_excludes_other_entities(receiver_client):
    other = _event("t:other-entity:v1", TRAINING_COMPLETED, {"status": "candidate_ready"},
                   entity_id="300:2", site_key="177")
    receiver_client.post("/v1/results/training", json=other)
    mine = receiver_client.get("/v1/results/training",
                               params={"entity_id": "215:0"}).json()
    assert mine["total"] >= 1
    assert all(i["entity_id"] == "215:0" for i in mine["items"])
    theirs = receiver_client.get("/v1/results/training",
                                 params={"entity_id": "300:2"}).json()
    assert all(i["entity_id"] == "300:2" for i in theirs["items"])


def test_pagination(receiver_client):
    page = receiver_client.get("/v1/results/backtest",
                               params={"limit": 2, "offset": 0}).json()
    assert page["limit"] == 2 and page["offset"] == 0
    assert len(page["items"]) <= 2
    assert page["total"] >= len(page["items"])


def test_entity_aggregate_endpoint(receiver_client):
    receiver_client.post("/v1/results/calibration", json=_event(
        "c:agg:v1", CALIBRATION_COMPLETED,
        {"calibrator_id": "cal_1", "status": "fitted", "scope": "entity",
         "method": "split_conformal", "metrics": {"brier": 0.07}}))
    receiver_client.post("/v1/results/scoring", json=_event(
        "s:agg:v1", SCORING_RECORDED,
        {"decision_time": "2026-10-12T10:00:00", "shortage_risk": 0.61,
         "alert_state": "WARNING", "action": {"should_replenish": True, "quantity": 12}}))
    body = receiver_client.get("/v1/results/entities/215:0").json()
    assert body["entity_id"] == "215:0"
    assert body["counts"]["training"] >= 1
    assert body["counts"]["calibration"] >= 1
    assert body["counts"]["backtest"] >= 3
    assert body["counts"]["scoring"] >= 1
    assert {b["backtest_type"] for b in body["backtests"]} >= {
        "forecast_accuracy", "policy_replay", "closed_loop_robustness"}


def test_unrelated_entity_aggregate_is_isolated(receiver_client):
    body = receiver_client.get("/v1/results/entities/300:2").json()
    assert body["counts"]["backtest"] == 0
    assert body["counts"]["scoring"] == 0


def test_event_retrieval_and_404(receiver_client):
    body = receiver_client.get("/v1/results/events/t:new:v1").json()
    assert body["event_id"] == "t:new:v1"
    assert body["payload"]["status"] == "candidate_ready"
    assert "payload_json" not in body
    assert receiver_client.get("/v1/results/events/missing").status_code == 404


def test_scoring_projection_extracts_key_columns(receiver_client):
    rows = receiver_client.get("/v1/results/scoring",
                               params={"entity_id": "215:0"}).json()["items"]
    assert rows
    row = rows[0]
    assert row["shortage_risk"] == 0.61
    assert row["alert_state"] == "WARNING"
    assert row["quantity"] == 12
    # the projection is flattened, not a second copy of the payload
    assert "result_json" not in row
    assert row["entity_id"] == "215:0" and row["kdvr"] == 215 and row["chan_no"] == 0


def test_no_binary_artifacts_are_stored(receiver_client):
    """Only metadata and safe references — never a pickled model."""
    import json

    rows = receiver_client.get("/v1/results/backtest").json()["items"]
    for r in rows:
        blob = json.dumps(r)
        assert ".pkl" not in blob or "uri" in blob
        assert "base64" not in blob.lower()


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------
def test_bearer_token_is_enforced_when_configured(receiver_client, monkeypatch):
    import settings as C

    monkeypatch.setattr(C, "RECEIVER_TOKEN", "s3cret")
    ev = _event("t:auth:v1", TRAINING_COMPLETED, {"status": "candidate_ready"})
    assert receiver_client.post("/v1/results/training", json=ev).status_code == 401
    ok = receiver_client.post("/v1/results/training", json=ev,
                              headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 201
    assert receiver_client.post(
        "/v1/results/training", json=ev,
        headers={"Authorization": "Bearer wrong"}).status_code == 401


# --------------------------------------------------------------------------
# Configuration provenance (TASK 8)
# --------------------------------------------------------------------------
def _rows(table: str) -> list[dict]:
    from receiver.database import query
    return query(f"SELECT * FROM {table}")


def test_backtest_projection_records_config_provenance(receiver_client):
    """A backtest event must stay queryable by the config it was run under."""
    import json

    ev = _backtest("policy_replay", event_id="train_cfg:300:2:policy_replay:v1")
    ev["entity_id"] = "300:2"
    ev["payload"].update({
        "config_hash": "549777393e8a3846",
        "frozen_model_config": {"red_model": 25, "amber_model": 35},
        "effective_operational_config": {"red_threshold": 25, "amber_threshold": 35,
                                         "bay_capacity": 120, "target_stock": 50},
    })
    assert receiver_client.post("/v1/results/backtest", json=ev).status_code == 201

    row = [r for r in _rows("backtest_results")
           if r["event_id"] == "train_cfg:300:2:policy_replay:v1"][0]
    assert row["config_hash"] == "549777393e8a3846"
    assert json.loads(row["frozen_config_json"])["red_model"] == 25
    assert json.loads(row["effective_config_json"])["target_stock"] == 50


def test_training_projection_accepts_existing_event_contract(receiver_client):
    """The published contract uses frozen_thresholds/effective_config; honour it."""
    import json

    ev = _event("t:cfg:300:2:v1", TRAINING_COMPLETED, {
        "status": "candidate_ready",
        "config_hash": "549777393e8a3846",
        "frozen_thresholds": {"red_model": 25, "amber_model": 35},
        "effective_config": {"red_threshold": 25, "bay_capacity": 120},
    }, entity_id="300:2")
    assert receiver_client.post("/v1/results/training", json=ev).status_code == 201

    row = [r for r in _rows("training_results") if r["event_id"] == "t:cfg:300:2:v1"][0]
    assert row["config_hash"] == "549777393e8a3846"
    assert json.loads(row["frozen_config_json"])["amber_model"] == 35
    assert json.loads(row["effective_config_json"])["bay_capacity"] == 120


def test_events_without_config_metadata_still_store(receiver_client):
    """Backward compatibility: older publishers send no config fields."""
    ev = _backtest("forecast_accuracy", event_id="train_legacy:215:0:fa:v1")
    assert receiver_client.post("/v1/results/backtest", json=ev).status_code == 201
    row = [r for r in _rows("backtest_results")
           if r["event_id"] == "train_legacy:215:0:fa:v1"][0]
    assert row["config_hash"] is None
    assert row["frozen_config_json"] is None


def test_receiver_migration_adds_config_columns_to_old_database(tmp_path):
    """A receiver DB created before this change must migrate, not crash."""
    import sqlite3

    from receiver import database as rdb

    old = tmp_path / "legacy_receiver.db"
    schema = (PLATFORM_ROOT / "receiver" / "schema.sql").read_text(encoding="utf-8")
    for col in ("config_hash", "frozen_config_json", "effective_config_json"):
        schema = "\n".join(l for l in schema.splitlines()
                           if not l.strip().startswith(col))
    conn = sqlite3.connect(old)
    conn.executescript(schema)
    conn.execute(
        "INSERT INTO received_events (event_id, event_type, schema_version, "
        "received_at, payload_hash, payload_json) "
        "VALUES ('legacy:1', 'backtest.completed', 1, '2026-01-01T00:00:00Z', 'h', '{}')")
    conn.execute("INSERT INTO backtest_results (event_id, backtest_type) "
                 "VALUES ('legacy:1', 'policy_replay')")
    conn.commit()
    conn.close()

    original = rdb.db_path()
    try:
        rdb.configure(old)
        rdb.reset_connection()
        rdb.init_db()
        cols = {r["name"] for r in rdb.query("PRAGMA table_info(backtest_results)")}
        assert {"config_hash", "frozen_config_json", "effective_config_json"} <= cols
        assert rdb.query_one(
            "SELECT * FROM backtest_results WHERE event_id='legacy:1'") is not None
    finally:
        rdb.configure(original)
        rdb.reset_connection()
