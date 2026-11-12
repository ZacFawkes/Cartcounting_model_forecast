"""ML-side publishing: outbox durability, retry, dead-letter and recovery.

Covers spec §11.4 and the end-to-end acceptance cases in §11.6.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

import settings as C  # noqa: E402

UPLOAD_CONFIG = {
    "name": "publish_batch",
    "entity_defaults": {"cooldown_min": 30,
                        "lead_time_min": 10},
    "entity_overrides": {
        "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8": {"bay_capacity": 60, "red_threshold": 15, "amber_threshold": 20,
                  "target_stock": 28},
    },
}
ENTITY = "163:215:0:06053B44_0A47_42FC_BE3E_F88CB85F99D8"


@pytest.fixture(scope="module")
def client(isolated_db):
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def telemetry(sample_csv) -> pd.DataFrame:
    return pd.read_csv(sample_csv, parse_dates=["DVRDate", "next_DVRDate"])


def _window(df, kdvr=215, chan=0, n=20, offset=60):
    s = df[(df.KDVR == kdvr) & (df.ChanNo == chan)].sort_values("DVRDate")
    sel = s.tail(offset).head(n)
    return [{"DVRDate": r.DVRDate.isoformat(),
             "next_DVRDate": r.next_DVRDate.isoformat(),
             "Count": float(r.Count), "Count_IN": float(r.Count_IN),
             "Count_OUT": float(r.Count_OUT)} for r in sel.itertuples()]


@pytest.fixture(scope="module")
def trained(client, sample_csv) -> dict:
    """Upload, train 215:0 and promote it — the publishing source of truth."""
    with open(sample_csv, "rb") as fh:
        up = client.post("/v1/datasets/upload",
                         files={"file": ("sample.csv", fh, "text/csv")},
                         data={"config": json.dumps(UPLOAD_CONFIG)}).json()
    job_id = client.post("/v1/training/jobs",
                         json={"upload_id": up["upload_id"],
                               "entity_ids": [ENTITY]}).json()["job_id"]
    deadline = time.time() + 900
    while time.time() < deadline:
        job = client.get(f"/v1/training/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "partial", "failed"):
            break
        time.sleep(1.0)
    assert job["status"] == "succeeded", job.get("error")
    model_id = job["results"][ENTITY]["model_id"]
    promoted = client.post(f"/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/models/{model_id}/promote",
                           json={"promoted_by": "pytest"})
    assert promoted.status_code == 200, promoted.text
    return {"upload_id": up["upload_id"], "job_id": job_id, "model_id": model_id,
            "job": job}


# ==========================================================================
# Outbox is populated by the local results
# ==========================================================================
def test_training_calibration_and_three_backtests_are_enqueued(trained):
    from integrations import outbox

    events = outbox.list_events(limit=200)
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)

    assert len(by_type.get("training.completed", [])) >= 1
    assert len(by_type.get("calibration.completed", [])) >= 1
    backtests = by_type.get("backtest.completed", [])
    assert len(backtests) == 3, [e["event_id"] for e in backtests]

    kinds = set()
    for e in backtests:
        payload = json.loads(e["payload_json"])
        kinds.add(payload["payload"]["backtest_type"])
        assert payload["job_id"] == trained["job_id"]
        assert payload["entity_id"] == ENTITY
        assert payload["model_id"] == trained["model_id"]
    assert kinds == set(C.BACKTEST_TYPES)


def test_no_binary_model_artifact_is_placed_in_an_event(trained):
    from integrations import outbox

    for e in outbox.list_events(limit=200):
        blob = e["payload_json"]
        assert "base64" not in blob.lower()
        payload = json.loads(blob)["payload"]
        for art in payload.get("artifacts", []) or []:
            assert isinstance(art.get("uri", ""), str)
            assert len(art.get("uri", "")) < 500


def test_scoring_decision_is_enqueued(client, telemetry, trained):
    from integrations import outbox

    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    before = len(outbox.list_events(event_type="scoring.recorded", limit=500))
    r = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score", json={"records": _window(telemetry)})
    assert r.status_code == 200
    after = outbox.list_events(event_type="scoring.recorded", limit=500)
    assert len(after) == before + 1
    payload = json.loads(after[0]["payload_json"])["payload"]
    assert payload["decision_time"] == r.json()["decision_time"]
    assert "resolved_config" in payload
    assert payload["caveat"]


def test_replayed_decision_does_not_enqueue_twice(client, telemetry, trained):
    from integrations import outbox

    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})
    recs = _window(telemetry, offset=90)
    client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score", json={"records": recs})
    n1 = len(outbox.list_events(event_type="scoring.recorded", limit=500))
    client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score", json={"records": recs})
    n2 = len(outbox.list_events(event_type="scoring.recorded", limit=500))
    assert n2 == n1


# ==========================================================================
# Delivery to the receiver (spec §11.6.2, §11.6.3)
# ==========================================================================
def test_flush_delivers_every_result_category(trained, receiver_client,
                                              publish_session, monkeypatch):
    import integrations.publisher as pub

    monkeypatch.setattr(C, "RESULT_RECEIVER_ENABLED", True)
    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", True)
    for _ in range(6):
        if pub.flush_once(session=publish_session)["sent"] == 0:
            break

    assert receiver_client.get("/v1/results/training",
                               params={"entity_id": ENTITY}).json()["total"] >= 1
    assert receiver_client.get("/v1/results/calibration",
                               params={"entity_id": ENTITY}).json()["total"] >= 1
    bt = receiver_client.get("/v1/results/backtest",
                             params={"job_id": trained["job_id"]}).json()
    assert bt["total"] == 3
    assert {i["backtest_type"] for i in bt["items"]} == set(C.BACKTEST_TYPES)
    assert receiver_client.get("/v1/results/scoring",
                               params={"entity_id": ENTITY}).json()["total"] >= 1


def test_delivered_records_carry_the_correct_identifiers(trained, receiver_client):
    rows = receiver_client.get("/v1/results/backtest",
                               params={"job_id": trained["job_id"]}).json()["items"]
    for r in rows:
        assert r["entity_id"] == ENTITY
        assert r["job_id"] == trained["job_id"]
        assert r["model_id"] == trained["model_id"]
        assert r["upload_id"] == trained["upload_id"]
        assert r["status"] in ("completed", "insufficient_data", "failed")


def test_receiver_filtering_isolates_this_entity(receiver_client, trained):
    body = receiver_client.get(f"/v1/results/entities/{ENTITY}").json()
    assert body["counts"]["backtest"] == 3
    other = receiver_client.get("/v1/results/entities/215:1").json()
    assert other["counts"]["training"] == 0
    assert other["counts"]["backtest"] == 0


def test_outbox_is_drained(trained):
    from integrations import outbox

    stats = outbox.stats()
    assert stats["pending"] == 0
    assert stats["dead_letter"] == 0
    assert stats["sent"] >= 5


# ==========================================================================
# Resilience (spec §11.6.5-7)
# ==========================================================================
def test_receiver_outage_does_not_fail_scoring(client, telemetry, trained,
                                               receiver_client, monkeypatch):
    """Score while the receiver is down, then deliver exactly one record."""
    import integrations.publisher as pub
    from integrations import outbox
    from tests.conftest import _InProcessSession

    monkeypatch.setattr(C, "RESULT_RECEIVER_ENABLED", True)
    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", True)
    client.post("/v1/sessions/reset", json={"entity_id": ENTITY})

    r = client.post("/v1/sites/163/entities/215/0/bays/06053B44_0A47_42FC_BE3E_F88CB85F99D8/score",
                    json={"records": _window(telemetry, offset=140)})
    assert r.status_code == 200, "scoring must not depend on the receiver"
    decision_time = r.json()["decision_time"]
    assert r.json()["persisted"] is True

    down = _InProcessSession(receiver_client, fail_times=5, status=503)
    out = pub.flush_once(session=down)
    assert out["sent"] == 0 and out["retried"] >= 1

    stored = client.get("/v1/logs/scores",
                        params={"entity_id": ENTITY, "limit": 20}).json()["entries"]
    assert any(s["decision_time"] == decision_time for s in stored)

    # Receiver comes back: the pending event must be retried and land once.
    for row in outbox.list_events(status=outbox.PENDING, limit=50):
        from registry.database import execute

        execute("UPDATE result_outbox SET next_attempt_at=? WHERE id=?",
                ("2000-01-01T00:00:00+00:00", row["id"]))
    up = pub.flush_once(session=_InProcessSession(receiver_client))
    assert up["sent"] >= 1

    rows = receiver_client.get("/v1/results/scoring",
                               params={"entity_id": ENTITY}).json()["items"]
    matching = [x for x in rows if x["decision_time"] == decision_time]
    assert len(matching) == 1, "exactly one receiver record per decision"


def test_transient_5xx_is_retried_with_backoff(receiver_client, monkeypatch):
    import integrations.publisher as pub
    from integrations import outbox
    from integrations.result_events import TRAINING_COMPLETED, ResultEvent
    from tests.conftest import _InProcessSession

    monkeypatch.setattr(C, "RESULT_RECEIVER_ENABLED", True)
    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", True)
    ev = ResultEvent(event_id="retry:case:v1", event_type=TRAINING_COMPLETED,
                     payload={"status": "candidate_ready"}, entity_id=ENTITY,
                     job_id="j", model_id="m")
    assert outbox.enqueue(ev) is True

    res = pub.flush_once(session=_InProcessSession(receiver_client, fail_times=1,
                                                   status=500))
    assert res["retried"] == 1
    row = outbox.get_event("retry:case:v1")
    assert row["status"] == outbox.PENDING
    assert row["attempts"] == 1
    assert row["next_attempt_at"] > row["created_at"]


def test_non_retryable_4xx_is_dead_lettered_and_can_be_retried(receiver_client,
                                                               monkeypatch):
    import integrations.publisher as pub
    from integrations import outbox
    from integrations.result_events import TRAINING_COMPLETED, ResultEvent
    from registry.database import execute
    from tests.conftest import _InProcessSession

    monkeypatch.setattr(C, "RESULT_RECEIVER_ENABLED", True)
    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", True)
    ev = ResultEvent(event_id="dead:case:v1", event_type=TRAINING_COMPLETED,
                     payload={"status": "candidate_ready"}, entity_id=ENTITY,
                     job_id="j", model_id="m")
    outbox.enqueue(ev)

    res = pub.flush_once(session=_InProcessSession(receiver_client, fail_times=1,
                                                   status=400))
    assert res["dead"] == 1
    assert outbox.get_event("dead:case:v1")["status"] == outbox.DEAD

    assert outbox.retry_dead_letters("dead:case:v1") == 1
    execute("UPDATE result_outbox SET next_attempt_at=? WHERE event_id=?",
            ("2000-01-01T00:00:00+00:00", "dead:case:v1"))
    out = pub.flush_once(session=_InProcessSession(receiver_client))
    assert out["sent"] >= 1
    assert outbox.get_event("dead:case:v1")["status"] == outbox.SENT


def test_invalid_token_is_not_retried_forever(receiver_client, monkeypatch):
    import integrations.publisher as pub
    from integrations import outbox
    from integrations.result_events import TRAINING_COMPLETED, ResultEvent
    from tests.conftest import _InProcessSession

    monkeypatch.setattr(C, "RESULT_RECEIVER_ENABLED", True)
    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", True)
    ev = ResultEvent(event_id="auth:case:v1", event_type=TRAINING_COMPLETED,
                     payload={"status": "candidate_ready"}, entity_id=ENTITY)
    outbox.enqueue(ev)
    res = pub.flush_once(session=_InProcessSession(receiver_client, fail_times=1,
                                                   status=401))
    assert res["dead"] == 1
    assert outbox.get_event("auth:case:v1")["status"] == outbox.DEAD


def test_in_flight_rows_are_recovered_after_restart(receiver_client, monkeypatch):
    import integrations.publisher as pub
    from integrations import outbox
    from integrations.result_events import TRAINING_COMPLETED, ResultEvent
    from registry.database import execute
    from tests.conftest import _InProcessSession

    monkeypatch.setattr(C, "RESULT_RECEIVER_ENABLED", True)
    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", True)
    ev = ResultEvent(event_id="restart:case:v1", event_type=TRAINING_COMPLETED,
                     payload={"status": "candidate_ready"}, entity_id=ENTITY)
    outbox.enqueue(ev)
    execute("UPDATE result_outbox SET status=? WHERE event_id=?",
            (outbox.SENDING, "restart:case:v1"))

    assert outbox.reset_sending() >= 1
    assert outbox.get_event("restart:case:v1")["status"] == outbox.PENDING
    execute("UPDATE result_outbox SET next_attempt_at=? WHERE event_id=?",
            ("2000-01-01T00:00:00+00:00", "restart:case:v1"))
    assert pub.flush_once(session=_InProcessSession(receiver_client))["sent"] >= 1


def test_publishing_can_be_disabled(monkeypatch):
    import integrations.publisher as pub

    monkeypatch.setattr(pub, "RESULT_RECEIVER_ENABLED", False)
    assert pub.flush_once()["skipped"] is True


def test_outbox_observability_endpoint(client, trained):
    body = client.get("/v1/integrations/outbox").json()
    assert "counts" in body and "events" in body
    assert body["receiver_url"]
    assert all("payload_json" not in e for e in body["events"])
    one = body["events"][0]["event_id"]
    detail = client.get(f"/v1/integrations/outbox/{one}").json()
    assert detail["event_id"] == one
    assert "payload" in detail and "payload_json" not in detail


def test_manual_dead_letter_retry_endpoint(client):
    r = client.post("/v1/integrations/outbox/retry", json={"flush_now": False})
    assert r.status_code == 200
    assert "counts" in r.json()


def test_api_keys_are_never_logged(client, caplog, monkeypatch):
    monkeypatch.setattr(C, "RECEIVER_TOKEN", "super-secret-token")
    client.get("/v1/integrations/outbox")
    assert "super-secret-token" not in caplog.text
